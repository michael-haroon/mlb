"""
tests/test_fetch_weather.py
----------------------------
Tests for data_curation/scripts/fetch_weather.py.

Structure:
  Unit tests   — pure logic, no network, no S3
  Empirical tests — live Open-Meteo API calls; use a single venue (Yankee Stadium)
                    and a narrow 7-day window to minimise cost and rate-limit exposure.

Run:
  conda run -n pred python -m pytest tests/test_fetch_weather.py -v
  conda run -n pred python -m pytest tests/test_fetch_weather.py -v -m empirical   # API only
  conda run -n pred python -m pytest tests/test_fetch_weather.py -v -m "not empirical"  # unit only
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.scripts.fetch_weather import (
    _make_pressure_level_vars,
    _parse_hourly,
    _parse_daily_to_hourly,
    _parse_ensemble,
    _fetch_archive,
    _fetch_historical_forecast,
    _fetch_marine,
    _fetch_flood,
    _archive_key,
    _forecast_key,
    _ensemble_key,
    ERA5_VARS,
    ERA5_LAND_VARS,
    ERA5_PRESSURE_VARS,
    ERA5_PRESSURE_LEVELS,
    ERA5_PRESSURE_BASE_VARS,
    HRRR_PRESSURE_LEVELS,
    HRRR_PRESSURE_BASE_VARS,
    HRRR_PRESSURE_VARS,
    HISTORICAL_FORECAST_SURFACE_VARS,
    AIR_QUALITY_VARS,
    ENSEMBLE_VARS,
    MARINE_VARS,
    FLOOD_VARS,
    FORECAST_VARS,
    TORONTO_VENUE_ID,
    HRRR_START_YEAR,
    ECMWF_HRES_START_YEAR,
    GFS_START_YEAR,
    BACKFILL_SOURCES,
    S3_PREFIX,
)

# ── Test venue: Yankee Stadium ────────────────────────────────────────────────
YANKEE_VENUE_ID = 3313
YANKEE_LAT      = 40.8296
YANKEE_LON      = -73.9262

# Short empirical window: one week in a completed past year (no lag issues)
EMPIRICAL_START = date(2023, 7, 1)
EMPIRICAL_END   = date(2023, 7, 7)
EMPIRICAL_YEAR  = 2023


# =============================================================================
# Unit tests — no network
# =============================================================================

class TestPressureLevelVarGeneration:
    def test_count_matches_levels_times_vars(self):
        levels = [850, 700, 500]
        base   = ["temperature", "wind_speed"]
        result = _make_pressure_level_vars(levels, base)
        names  = result.split(",")
        assert len(names) == len(levels) * len(base)

    def test_format_is_var_level_hpa(self):
        result = _make_pressure_level_vars([850], ["temperature"])
        assert result == "temperature_850hPa"

    def test_era5_pressure_vars_count(self):
        expected = len(ERA5_PRESSURE_LEVELS) * len(ERA5_PRESSURE_BASE_VARS)
        assert len(ERA5_PRESSURE_VARS.split(",")) == expected

    def test_hrrr_pressure_vars_count(self):
        expected = len(HRRR_PRESSURE_LEVELS) * len(HRRR_PRESSURE_BASE_VARS)
        assert len(HRRR_PRESSURE_VARS.split(",")) == expected

    def test_era5_has_19_levels(self):
        assert len(ERA5_PRESSURE_LEVELS) == 19

    def test_hrrr_has_44_levels(self):
        assert len(HRRR_PRESSURE_LEVELS) == 44

    def test_no_duplicates_in_era5_pressure_vars(self):
        names = ERA5_PRESSURE_VARS.split(",")
        assert len(names) == len(set(names))

    def test_no_duplicates_in_hrrr_pressure_vars(self):
        names = HRRR_PRESSURE_VARS.split(",")
        assert len(names) == len(set(names))


class TestParseHourly:
    def _mock_response(self, extra_vars: dict | None = None) -> dict:
        n = 24
        base = {
            "hourly": {
                "time": [f"2023-07-01T{h:02d}:00" for h in range(n)],
                "temperature_2m":   [70.0 + h * 0.1 for h in range(n)],
                "wind_speed_10m":   [5.0 + h * 0.05 for h in range(n)],
                "wind_direction_10m": [180.0] * n,
            }
        }
        if extra_vars:
            base["hourly"].update(extra_vars)
        return base

    def test_returns_dataframe(self):
        df = _parse_hourly(self._mock_response(), venue_id=1)
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_time_entries(self):
        df = _parse_hourly(self._mock_response(), venue_id=1)
        assert len(df) == 24

    def test_venue_id_column_set(self):
        df = _parse_hourly(self._mock_response(), venue_id=42)
        assert (df["venue_id"] == 42).all()

    def test_timestamp_is_utc(self):
        df = _parse_hourly(self._mock_response(), venue_id=1)
        assert str(df["timestamp"].dt.tz) == "UTC"

    def test_wind_decomposition_10m(self):
        # 180° wind at 10 mph — should blow in -y (south) direction
        df = _parse_hourly(self._mock_response(), venue_id=1)
        # wind_u_10m = speed * sin(180°) ≈ 0, wind_v_10m = speed * cos(180°) ≈ -speed
        assert "wind_u_10m" in df.columns
        assert "wind_v_10m" in df.columns
        np.testing.assert_allclose(df["wind_u_10m"].astype(float), 0.0, atol=1e-4)
        np.testing.assert_allclose(
            df["wind_v_10m"].astype(float),
            -df["wind_speed_10m"].astype(float),
            atol=1e-4,
        )

    def test_wind_decomposition_multi_level(self):
        extra = {
            "wind_speed_80m": [8.0] * 24,
            "wind_direction_80m": [90.0] * 24,  # due east
        }
        df = _parse_hourly(self._mock_response(extra), venue_id=1)
        assert "wind_u_80m" in df.columns
        assert "wind_v_80m" in df.columns
        # 90° → u = speed * sin(90°) = speed, v = speed * cos(90°) = 0
        np.testing.assert_allclose(
            df["wind_u_80m"].astype(float),
            df["wind_speed_80m"].astype(float),
            atol=1e-4,
        )
        np.testing.assert_allclose(df["wind_v_80m"].astype(float), 0.0, atol=1e-4)

    def test_no_decomposition_without_direction(self):
        # If direction col is absent, no u/v should be added
        resp = {
            "hourly": {
                "time": ["2023-07-01T00:00"],
                "wind_speed_80m": [8.0],
            }
        }
        df = _parse_hourly(resp, venue_id=1)
        assert "wind_u_80m" not in df.columns

    def test_float32_dtype(self):
        df = _parse_hourly(self._mock_response(), venue_id=1)
        assert df["temperature_2m"].dtype == pd.Float32Dtype()

    def test_null_values_preserved_as_na(self):
        resp = self._mock_response({"cape": [None] * 24})
        df = _parse_hourly(resp, venue_id=1)
        assert df["cape"].isna().all()


class TestParseDailyToHourly:
    def _mock_flood_response(self) -> dict:
        return {
            "daily": {
                "time": ["2023-07-01", "2023-07-02"],
                "river_discharge": [150.0, 160.0],
            }
        }

    def test_expands_2_days_to_48_rows(self):
        df = _parse_daily_to_hourly(self._mock_flood_response(), venue_id=1)
        assert len(df) == 48

    def test_timestamp_column_is_utc(self):
        df = _parse_daily_to_hourly(self._mock_flood_response(), venue_id=1)
        assert str(df["timestamp"].dt.tz) == "UTC"

    def test_discharge_value_repeated_24_times_per_day(self):
        df = _parse_daily_to_hourly(self._mock_flood_response(), venue_id=1)
        day1 = df[df["timestamp"].dt.date == date(2023, 7, 1)]
        assert len(day1) == 24
        assert (day1["river_discharge"].astype(float) == 150.0).all()

    def test_venue_id_correct(self):
        df = _parse_daily_to_hourly(self._mock_flood_response(), venue_id=99)
        assert (df["venue_id"] == 99).all()

    def test_all_24_hours_present_for_each_day(self):
        df = _parse_daily_to_hourly(self._mock_flood_response(), venue_id=1)
        hours = df[df["timestamp"].dt.date == date(2023, 7, 1)]["timestamp"].dt.hour.tolist()
        assert sorted(hours) == list(range(24))


class TestParseEnsemble:
    def _mock_ensemble_response(self) -> dict:
        n = 6
        times = [f"2023-07-01T{h:02d}:00" for h in range(n)]
        hourly: dict = {"time": times}
        for m in range(1, 4):
            hourly[f"temperature_2m_member{m:02d}"] = [70.0 + m] * n
        return {"hourly": hourly}

    def test_returns_mean_and_std(self):
        df = _parse_ensemble(self._mock_ensemble_response(), venue_id=1)
        assert "temperature_2m_ens_mean" in df.columns
        assert "temperature_2m_ens_std" in df.columns

    def test_mean_correct(self):
        df = _parse_ensemble(self._mock_ensemble_response(), venue_id=1)
        # members are 71, 72, 73 → mean 72
        np.testing.assert_allclose(df["temperature_2m_ens_mean"], 72.0)

    def test_std_correct(self):
        df = _parse_ensemble(self._mock_ensemble_response(), venue_id=1)
        expected_std = np.std([71.0, 72.0, 73.0])
        np.testing.assert_allclose(df["temperature_2m_ens_std"], expected_std, atol=1e-5)


class TestS3Keys:
    def test_archive_key_format(self):
        key = _archive_key("era5", 123, 2023)
        assert key == f"{S3_PREFIX}/weather/source=era5/venue_id=123/year=2023.parquet"

    def test_forecast_key_format(self):
        key = _forecast_key(123, date(2023, 7, 1))
        assert key == f"{S3_PREFIX}/weather/source=forecast/venue_id=123/date=2023-07-01.parquet"

    def test_ensemble_key_format(self):
        key = _ensemble_key(123, date(2023, 7, 1))
        assert key == f"{S3_PREFIX}/weather/source=ensemble/venue_id=123/date=2023-07-01.parquet"

    def test_archive_key_source_embedded(self):
        for source in ("era5", "era5_pressure", "hrrr_forecast", "marine", "flood"):
            key = _archive_key(source, 1, 2023)
            assert f"source={source}" in key


class TestYearGuards:
    """Verify sources return None for years before their data availability."""

    def test_ecmwf_ifs_archive_before_2017_returns_none(self):
        result = _fetch_archive(YANKEE_VENUE_ID, YANKEE_LAT, YANKEE_LON, "ecmwf_ifs", 2016)
        assert result is None

    def test_air_quality_before_2013_returns_none(self):
        result = _fetch_archive(YANKEE_VENUE_ID, YANKEE_LAT, YANKEE_LON, "air_quality", 2012)
        assert result is None

    def test_hrrr_forecast_before_start_year_returns_none(self):
        result = _fetch_historical_forecast(
            YANKEE_VENUE_ID, YANKEE_LAT, YANKEE_LON,
            "gfs_hrrr", HRRR_START_YEAR - 1,
        )
        assert result is None

    def test_ecmwf_hres_forecast_before_start_year_returns_none(self):
        result = _fetch_historical_forecast(
            YANKEE_VENUE_ID, YANKEE_LAT, YANKEE_LON,
            "ecmwf_ifs", ECMWF_HRES_START_YEAR - 1,
        )
        assert result is None

    def test_gfs_forecast_before_2021_returns_none(self):
        result = _fetch_historical_forecast(
            YANKEE_VENUE_ID, YANKEE_LAT, YANKEE_LON,
            "gfs_global", GFS_START_YEAR - 1,
        )
        assert result is None


class TestHRRRConus:
    """HRRR is CONUS-only, so the excluded venue must not be fetched from it.

    NOTE: this class does NOT prove anything about Toronto, despite its history. The guard
    keys on venue_id alone, and TORONTO_VENUE_ID is 2523 = Steinbrenner Field (Tampa), not
    Rogers Centre (venue 14). Passing Rogers Centre's coordinates alongside the wrong id made
    the test read as a Toronto test while exercising a Tampa id — which is how the mislabel
    survived. Kept green deliberately: it pins the CURRENT routing, which the already-built
    weather artifacts depend on.
    """

    def test_excluded_venue_not_fetched_from_hrrr(self):
        # Coords are inert here — the CONUS guard compares venue_id only. Left at Rogers
        # Centre's to document the original (mistaken) intent rather than hide it.
        result = _fetch_historical_forecast(
            TORONTO_VENUE_ID, 43.6414, -79.3894,
            "gfs_hrrr", HRRR_START_YEAR,
        )
        assert result is None

    def test_yankee_stadium_not_excluded_from_hrrr(self):
        # Should not return None due to CONUS guard (year guard doesn't apply)
        # Just verifies the guard doesn't incorrectly exclude US venues.
        # (Will be None only if year < HRRR_START_YEAR)
        result_future = _fetch_historical_forecast(
            YANKEE_VENUE_ID, YANKEE_LAT, YANKEE_LON,
            "gfs_hrrr", HRRR_START_YEAR - 1,  # triggers year guard
        )
        assert result_future is None  # year guard, not CONUS guard


class TestExclusionIsGeographicallyJustified:
    """Defines the behaviour task #15 will implement: DELETE the id-based exclusion.

    These tests FAIL today by design (strict xfail, so they turn red the moment the fix lands
    and the marker goes stale). They exist because `TestHRRRConus` above pins the *current*
    routing and therefore cannot express what the routing should be.

    The premise measured on 2026-08-31 against game_meta's 2015+ population (31,830 games,
    100 venues): the excluded id 2523 is Steinbrenner Field, Tampa (27.980, -82.507) — deep
    inside the HRRR CONUS grid and 252 games — while venue 14 Rogers Centre (43.642, -79.389)
    is also inside it and carries 860 games. So there is no venue here that needs the ECMWF
    route, and repointing the constant to 14 would break 860 games to rescue 252. The only
    populated venues genuinely off the grid are Tokyo Dome, Gocheok Sky Dome, London Stadium
    and Hiram Bithorn (26 games, 0.082%) and none of them is excluded.

    Bind the deletion to a weather_features/weather_asof rebuild: every stored artifact was
    built with 2523-based routing, so changing the code alone converts a consistent feature
    degradation into genuine train/serve skew.
    """

    # Conservative interior box for the HRRR CONUS domain. The true grid is Lambert conformal,
    # so this box UNDER-claims coverage at the corners — anything it calls "outside" deserves a
    # real look, and anything it calls "inside" certainly is.
    CONUS_BOX = (21.5, 52.5, -134.0, -60.5)  # lat_lo, lat_hi, lon_lo, lon_hi

    STEINBRENNER = (2523, 27.97997, -82.50702)   # what the constant actually names
    ROGERS_CENTRE = (14, 43.64155, -79.38915)    # what it was meant to name

    @staticmethod
    def _inside(lat: float, lon: float) -> bool:
        lat_lo, lat_hi, lon_lo, lon_hi = TestExclusionIsGeographicallyJustified.CONUS_BOX
        return lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi

    @pytest.mark.xfail(strict=True, reason="task #15: exclusion still names a CONUS venue")
    def test_excluded_venue_lies_outside_the_hrrr_domain(self):
        """An HRRR exclusion is only defensible for a venue HRRR cannot cover."""
        for vid, lat, lon in (self.STEINBRENNER, self.ROGERS_CENTRE):
            if vid == TORONTO_VENUE_ID:
                assert not self._inside(lat, lon), (
                    f"venue {vid} at ({lat}, {lon}) is inside the HRRR CONUS grid, so excluding "
                    "it from HRRR discards real physics for no reason"
                )

    @pytest.mark.xfail(strict=True, reason="task #15: id guard still fires inside _fetch_pressure_forecast")
    def test_pressure_forecast_attempts_hrrr_for_a_conus_venue(self, monkeypatch):
        """Offline: prove the guard *lets the request through* rather than short-circuiting.

        Reaching `_get_json` is the observable that distinguishes "guard removed" from "guard
        fired"; a None return cannot, because a failed fetch also returns None. Raising from the
        stub keeps this a unit test with no network call.
        """
        import data_curation.scripts.fetch_weather as fw

        class _Reached(Exception):
            pass

        monkeypatch.setattr(fw, "_get_json", lambda *a, **k: (_ for _ in ()).throw(_Reached()))
        vid, lat, lon = self.STEINBRENNER
        with pytest.raises(_Reached):
            fw._fetch_pressure_forecast(vid, lat, lon)

    @pytest.mark.xfail(strict=True, reason="task #15: id guard still fires inside _fetch_historical_forecast")
    def test_historical_forecast_attempts_hrrr_for_a_conus_venue(self, monkeypatch):
        import data_curation.scripts.fetch_weather as fw

        class _Reached(Exception):
            pass

        monkeypatch.setattr(fw, "_get_json", lambda *a, **k: (_ for _ in ()).throw(_Reached()))
        vid, lat, lon = self.STEINBRENNER
        with pytest.raises(_Reached):
            fw._fetch_historical_forecast(
                vid, lat, lon, "gfs_hrrr", HRRR_START_YEAR,
                start_date=date(HRRR_START_YEAR, 7, 1), end_date=date(HRRR_START_YEAR, 7, 2),
            )


class TestVariableSetCompleteness:
    """Verify variable sets contain expected critical variables."""

    def test_era5_has_pressure_msl(self):
        assert "pressure_msl" in ERA5_VARS

    def test_forecast_has_cape(self):
        assert "cape" in FORECAST_VARS

    def test_forecast_has_lifted_index(self):
        assert "lifted_index" in FORECAST_VARS

    def test_forecast_has_wind_80m(self):
        assert "wind_speed_80m" in FORECAST_VARS

    def test_historical_forecast_has_cape(self):
        assert "cape" in HISTORICAL_FORECAST_SURFACE_VARS

    def test_historical_forecast_has_precipitable_water(self):
        assert "total_column_integrated_water_vapour" in HISTORICAL_FORECAST_SURFACE_VARS

    def test_air_quality_has_ozone(self):
        assert "ozone" in AIR_QUALITY_VARS

    def test_air_quality_has_carbon_monoxide(self):
        assert "carbon_monoxide" in AIR_QUALITY_VARS

    def test_air_quality_has_nitrogen_dioxide(self):
        assert "nitrogen_dioxide" in AIR_QUALITY_VARS

    def test_marine_has_sea_surface_temperature(self):
        assert "sea_surface_temperature" in MARINE_VARS

    def test_marine_has_wave_height(self):
        assert "wave_height" in MARINE_VARS

    def test_marine_has_ocean_current(self):
        assert "ocean_current_velocity" in MARINE_VARS

    def test_ensemble_has_cape(self):
        assert "cape" in ENSEMBLE_VARS

    def test_no_duplicate_vars_in_forecast(self):
        names = FORECAST_VARS.split(",")
        assert len(names) == len(set(names))

    def test_no_duplicate_vars_in_historical_forecast_surface(self):
        names = HISTORICAL_FORECAST_SURFACE_VARS.split(",")
        assert len(names) == len(set(names))


class TestBackfillSourceRegistry:
    def test_all_expected_sources_present(self):
        keys = {s for s, _, _ in BACKFILL_SOURCES}
        for expected in (
            "era5", "era5_land", "era5_pressure",
            "ecmwf_ifs", "air_quality",
            "hrrr_forecast", "hrrr_forecast_pressure",
            "ecmwf_ifs_hres_forecast", "ecmwf_ifs_hres_forecast_pressure",
            "gfs_forecast", "gfs_forecast_pressure",
            "marine", "flood",
        ):
            assert expected in keys, f"Missing source: {expected}"

    def test_start_years_are_integers(self):
        for source, start_year, _ in BACKFILL_SOURCES:
            assert isinstance(start_year, int), f"{source} start_year not int"

    def test_era5_starts_at_or_before_2015(self):
        era5_entry = next(s for s in BACKFILL_SOURCES if s[0] == "era5")
        assert era5_entry[1] <= 2015

    def test_era5_land_starts_at_or_before_1950(self):
        entry = next(s for s in BACKFILL_SOURCES if s[0] == "era5_land")
        assert entry[1] <= 1950

    def test_flood_start_year_is_1984(self):
        flood_entry = next(s for s in BACKFILL_SOURCES if s[0] == "flood")
        assert flood_entry[1] == 1984

    def test_marine_start_year_is_1940(self):
        marine_entry = next(s for s in BACKFILL_SOURCES if s[0] == "marine")
        assert marine_entry[1] == 1940


class TestERA5LandVarCompleteness:
    def test_has_all_4_soil_temperature_depths(self):
        for depth in ("0_to_7cm", "7_to_28cm", "28_to_100cm", "100_to_255cm"):
            assert f"soil_temperature_{depth}" in ERA5_LAND_VARS

    def test_has_all_4_soil_moisture_depths(self):
        for depth in ("0_to_7cm", "7_to_28cm", "28_to_100cm", "100_to_255cm"):
            assert f"soil_moisture_{depth}" in ERA5_LAND_VARS

    def test_has_snowmelt(self):
        assert "snowmelt" in ERA5_LAND_VARS

    def test_has_surface_runoff(self):
        assert "surface_runoff" in ERA5_LAND_VARS

    def test_has_leaf_area_index(self):
        assert "leaf_area_index_low_vegetation" in ERA5_LAND_VARS
        assert "leaf_area_index_high_vegetation" in ERA5_LAND_VARS

    def test_no_duplicate_vars(self):
        names = ERA5_LAND_VARS.split(",")
        assert len(names) == len(set(names))


# =============================================================================
# Empirical tests — live Open-Meteo API calls
# =============================================================================

# =============================================================================
# Helpers shared across empirical classes
# =============================================================================

def _schema_summary(df: pd.DataFrame) -> dict:
    """Return schema info useful for assertions: dtypes, null counts, col list."""
    return {
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
        "shape": df.shape,
    }


def _assert_base_schema(df: pd.DataFrame, venue_id: int, expected_rows: int = 168):
    """Every weather source must satisfy these invariants."""
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
    assert "venue_id"  in df.columns
    assert "timestamp" in df.columns
    assert (df["venue_id"] == venue_id).all(), "venue_id mismatch"
    assert str(df["timestamp"].dt.tz) == "UTC", "timestamp not UTC"
    assert df["timestamp"].is_monotonic_increasing, "timestamps not sorted"
    assert df["timestamp"].nunique() == expected_rows, "duplicate timestamps"


def _assert_no_all_null_columns(df: pd.DataFrame, exclude=("venue_id", "timestamp")):
    """No declared variable column should be entirely null — indicates API changed."""
    bad = [c for c in df.columns if c not in exclude and df[c].isna().all()]
    assert not bad, f"Columns returned entirely null: {bad}"


def _assert_float_dtype(df: pd.DataFrame, exclude=("venue_id", "timestamp")):
    """All data columns must be Float32 or Float64 (nullable float)."""
    bad = {c: str(df[c].dtype) for c in df.columns
           if c not in exclude and not str(df[c].dtype).startswith("Float")}
    assert not bad, f"Non-float dtypes: {bad}"


# =============================================================================
# Empirical tests — live Open-Meteo API calls
# Schema, dtypes, null rates, column presence.
# =============================================================================

@pytest.mark.empirical
class TestERA5ArchiveEmpirical:
    """ERA5 surface archive: schema, dtypes, null rates, declared vars present."""

    @pytest.fixture(scope="class")
    def df(self):
        result = _fetch_archive(
            YANKEE_VENUE_ID, YANKEE_LAT, YANKEE_LON, "era5", EMPIRICAL_YEAR,
            start_date=EMPIRICAL_START, end_date=EMPIRICAL_END,
        )
        assert result is not None, "ERA5 fetch returned None"
        return result.reset_index(drop=True)

    def test_base_schema(self, df):
        _assert_base_schema(df, YANKEE_VENUE_ID)

    def test_all_declared_vars_present(self, df):
        missing = set(ERA5_VARS.split(",")) - set(df.columns)
        assert not missing, f"API did not return declared ERA5_VARS: {missing}"

    def test_wind_uv_components_computed(self, df):
        assert "wind_u_10m" in df.columns and "wind_v_10m" in df.columns

    def test_no_all_null_columns(self, df):
        _assert_no_all_null_columns(df)

    def test_float_dtypes_on_data_columns(self, df):
        _assert_float_dtype(df)

    def test_null_rate_summary(self, df):
        # Document null rates — not a hard assertion, but fails if >50% null
        # for core columns (indicates data gap or changed variable name)
        core = ["temperature_2m", "wind_speed_10m", "surface_pressure", "precipitation"]
        for col in core:
            null_rate = df[col].isna().mean()
            assert null_rate < 0.5, f"{col} null rate too high: {null_rate:.1%}"


@pytest.mark.empirical
class TestERA5PressureLevelsEmpirical:
    """ERA5 pressure levels: all 19 levels present, dtypes correct, nulls documented."""

    @pytest.fixture(scope="class")
    def df(self):
        result = _fetch_archive(
            YANKEE_VENUE_ID, YANKEE_LAT, YANKEE_LON, "era5_pressure", EMPIRICAL_YEAR,
            start_date=EMPIRICAL_START, end_date=EMPIRICAL_END,
        )
        assert result is not None, "ERA5 pressure fetch returned None"
        return result.reset_index(drop=True)

    def test_base_schema(self, df):
        _assert_base_schema(df, YANKEE_VENUE_ID)

    def test_column_count(self, df):
        # 19 levels × 6 vars + venue_id + timestamp = 116
        expected = len(ERA5_PRESSURE_LEVELS) * len(ERA5_PRESSURE_BASE_VARS) + 2
        assert len(df.columns) == expected, (
            f"Expected {expected} cols, got {len(df.columns)}: {df.columns.tolist()}"
        )

    def test_all_declared_pressure_level_columns_present(self, df):
        missing = set(ERA5_PRESSURE_VARS.split(",")) - set(df.columns)
        assert not missing, f"Missing pressure level columns: {sorted(missing)[:5]}..."

    def test_float_dtypes_on_data_columns(self, df):
        _assert_float_dtype(df)

    def test_null_rate_per_level_documented(self, df):
        # For each level, record the null rate of temperature.
        # Assert < 100% — a fully null level means the API no longer provides it.
        for level in ERA5_PRESSURE_LEVELS:
            col = f"temperature_{level}hPa"
            null_rate = df[col].isna().mean()
            assert null_rate < 1.0, f"temperature_{level}hPa is 100% null"


@pytest.mark.empirical
class TestHRRRForecastEmpirical:
    """HRRR historical forecast: new vars (CAPE, LI, 80m wind, precipitable water)
    are present; schema matches production inference pipeline expectations."""

    @pytest.fixture(scope="class")
    def df(self):
        result = _fetch_historical_forecast(
            YANKEE_VENUE_ID, YANKEE_LAT, YANKEE_LON,
            "gfs_hrrr", EMPIRICAL_YEAR,
            start_date=EMPIRICAL_START, end_date=EMPIRICAL_END,
        )
        assert result is not None, "HRRR historical forecast returned None"
        return result.reset_index(drop=True)

    def test_base_schema(self, df):
        _assert_base_schema(df, YANKEE_VENUE_ID)

    def test_convective_vars_present(self, df):
        for col in ("cape", "lifted_index", "convective_inhibition"):
            assert col in df.columns, f"Missing convective var: {col}"

    def test_multi_level_wind_present(self, df):
        for col in ("wind_speed_10m", "wind_speed_80m", "wind_speed_100m",
                    "wind_u_10m", "wind_v_10m", "wind_u_80m", "wind_v_80m"):
            assert col in df.columns, f"Missing wind col: {col}"

    def test_precipitable_water_present(self, df):
        assert "total_column_integrated_water_vapour" in df.columns

    def test_no_all_null_columns(self, df):
        _assert_no_all_null_columns(df)

    def test_float_dtypes_on_data_columns(self, df):
        _assert_float_dtype(df)

    def test_null_rates_for_new_vars(self, df):
        new_vars = [
            "cape", "lifted_index", "convective_inhibition",
            "total_column_integrated_water_vapour", "wind_speed_80m",
        ]
        for col in new_vars:
            null_rate = df[col].isna().mean()
            assert null_rate < 1.0, f"{col} entirely null — not returned by API"


@pytest.mark.empirical
class TestHRRRPressureLevelsEmpirical:
    """HRRR pressure levels via POST (414 fallback): all 44 levels, 8 vars each."""

    @pytest.fixture(scope="class")
    def df(self):
        result = _fetch_historical_forecast(
            YANKEE_VENUE_ID, YANKEE_LAT, YANKEE_LON,
            "gfs_hrrr", EMPIRICAL_YEAR, pressure_levels=True,
            start_date=EMPIRICAL_START, end_date=EMPIRICAL_END,
        )
        assert result is not None, "HRRR pressure level fetch returned None"
        return result.reset_index(drop=True)

    def test_base_schema(self, df):
        _assert_base_schema(df, YANKEE_VENUE_ID)

    def test_column_count(self, df):
        expected = len(HRRR_PRESSURE_LEVELS) * len(HRRR_PRESSURE_BASE_VARS) + 2
        assert len(df.columns) == expected, (
            f"Expected {expected} cols, got {len(df.columns)}"
        )

    def test_all_declared_pressure_columns_present(self, df):
        missing = set(HRRR_PRESSURE_VARS.split(",")) - set(df.columns)
        assert not missing, f"Missing pressure columns: {sorted(missing)[:5]}..."

    def test_float_dtypes(self, df):
        _assert_float_dtype(df)

    def test_vertical_velocity_column_present(self, df):
        assert "vertical_velocity_500hPa" in df.columns

    def test_dew_point_at_pressure_levels(self, df):
        assert "dew_point_850hPa" in df.columns

    def test_null_rate_per_level(self, df):
        for level in HRRR_PRESSURE_LEVELS:
            col = f"temperature_{level}hPa"
            null_rate = df[col].isna().mean()
            assert null_rate < 1.0, f"temperature_{level}hPa entirely null"


@pytest.mark.empirical
class TestECMWFHRESForecastEmpirical:
    """ECMWF IFS HRES historical forecast: global coverage incl. Toronto."""

    @pytest.fixture(scope="class")
    def df(self):
        result = _fetch_historical_forecast(
            YANKEE_VENUE_ID, YANKEE_LAT, YANKEE_LON,
            "ecmwf_ifs", EMPIRICAL_YEAR,
            start_date=EMPIRICAL_START, end_date=EMPIRICAL_END,
        )
        assert result is not None, "ECMWF IFS HRES historical forecast returned None"
        return result.reset_index(drop=True)

    def test_base_schema(self, df):
        _assert_base_schema(df, YANKEE_VENUE_ID)

    def test_convective_vars_present(self, df):
        for col in ("cape", "lifted_index"):
            assert col in df.columns, f"Missing: {col}"

    def test_no_all_null_columns(self, df):
        _assert_no_all_null_columns(df)

    def test_float_dtypes(self, df):
        _assert_float_dtype(df)

    def test_toronto_covered(self):
        # ECMWF is global — the only non-CONUS MLB park must work
        result = _fetch_historical_forecast(
            TORONTO_VENUE_ID, 43.6414, -79.3894,
            "ecmwf_ifs", EMPIRICAL_YEAR,
            start_date=EMPIRICAL_START, end_date=EMPIRICAL_END,
        )
        assert result is not None
        _assert_base_schema(result, TORONTO_VENUE_ID)


@pytest.mark.empirical
class TestMarineEmpirical:
    """ERA5-Ocean: wave vars, SST, currents, sea level — all hourly."""

    @pytest.fixture(scope="class")
    def df(self):
        result = _fetch_marine(
            YANKEE_VENUE_ID, YANKEE_LAT, YANKEE_LON, EMPIRICAL_YEAR,
            start_date=EMPIRICAL_START, end_date=EMPIRICAL_END,
        )
        assert result is not None, "Marine fetch returned None"
        return result.reset_index(drop=True)

    def test_base_schema(self, df):
        _assert_base_schema(df, YANKEE_VENUE_ID)

    def test_all_declared_marine_vars_present(self, df):
        missing = set(MARINE_VARS.split(",")) - set(df.columns)
        assert not missing, f"Missing marine vars: {missing}"

    def test_float_dtypes(self, df):
        _assert_float_dtype(df)

    def test_null_rates_documented(self, df):
        # Some marine vars (secondary/tertiary swell) may be sparse — allow high null.
        # Core vars must be non-null.
        core = ["sea_surface_temperature", "wave_height", "wind_wave_height"]
        for col in core:
            null_rate = df[col].isna().mean()
            assert null_rate < 1.0, f"{col} entirely null"


@pytest.mark.empirical
class TestFloodEmpirical:
    """GloFAS river discharge: daily → hourly expansion, constant per day."""

    @pytest.fixture(scope="class")
    def df(self):
        result = _fetch_flood(
            YANKEE_VENUE_ID, YANKEE_LAT, YANKEE_LON, EMPIRICAL_YEAR,
            start_date=EMPIRICAL_START, end_date=EMPIRICAL_END,
        )
        assert result is not None, "Flood fetch returned None"
        return result.reset_index(drop=True)

    def test_base_schema(self, df):
        _assert_base_schema(df, YANKEE_VENUE_ID)

    def test_river_discharge_column_present(self, df):
        assert "river_discharge" in df.columns

    def test_discharge_dtype_is_float(self, df):
        assert str(df["river_discharge"].dtype).startswith("Float")

    def test_discharge_constant_within_each_day(self, df):
        for day in df["timestamp"].dt.date.unique():
            day_vals = df[df["timestamp"].dt.date == day]["river_discharge"].dropna().astype(float)
            if len(day_vals) > 0:
                assert day_vals.nunique() == 1, f"Day {day}: discharge not constant"

    def test_no_discharge_nulls(self, df):
        assert df["river_discharge"].isna().sum() == 0, "river_discharge has nulls"


@pytest.mark.empirical
class TestAirQualityEmpirical:
    """CAMS global air quality: all declared gas and AQI columns present."""

    @pytest.fixture(scope="class")
    def df(self):
        result = _fetch_archive(
            YANKEE_VENUE_ID, YANKEE_LAT, YANKEE_LON, "air_quality", EMPIRICAL_YEAR,
            start_date=EMPIRICAL_START, end_date=EMPIRICAL_END,
        )
        assert result is not None, "Air quality fetch returned None"
        return result.reset_index(drop=True)

    def test_base_schema(self, df):
        _assert_base_schema(df, YANKEE_VENUE_ID)

    def test_all_declared_aq_vars_present(self, df):
        missing = set(AIR_QUALITY_VARS.split(",")) - set(df.columns)
        assert not missing, f"Missing air quality vars: {missing}"

    def test_float_dtypes(self, df):
        _assert_float_dtype(df)

    def test_null_rate_per_variable(self, df):
        for col in AIR_QUALITY_VARS.split(","):
            null_rate = df[col].isna().mean()
            assert null_rate < 1.0, f"{col} entirely null — not in API response"


@pytest.mark.empirical
class TestERA5LandArchiveEmpirical:
    """ERA5-Land 9km reanalysis: extra soil depths and vegetation vars present."""

    @pytest.fixture(scope="class")
    def df(self):
        result = _fetch_archive(
            YANKEE_VENUE_ID, YANKEE_LAT, YANKEE_LON, "era5_land", EMPIRICAL_YEAR,
            start_date=EMPIRICAL_START, end_date=EMPIRICAL_END,
        )
        assert result is not None, "ERA5-Land fetch returned None"
        return result.reset_index(drop=True)

    def test_base_schema(self, df):
        _assert_base_schema(df, YANKEE_VENUE_ID)

    def test_all_declared_vars_present(self, df):
        missing = set(ERA5_LAND_VARS.split(",")) - set(df.columns)
        assert not missing, f"API did not return declared ERA5_LAND_VARS: {missing}"

    def test_all_4_soil_temperature_depths_present(self, df):
        for depth in ("0_to_7cm", "7_to_28cm", "28_to_100cm", "100_to_255cm"):
            assert f"soil_temperature_{depth}" in df.columns

    def test_all_4_soil_moisture_depths_present(self, df):
        for depth in ("0_to_7cm", "7_to_28cm", "28_to_100cm", "100_to_255cm"):
            assert f"soil_moisture_{depth}" in df.columns

    def test_float_dtypes(self, df):
        _assert_float_dtype(df)

    def test_no_all_null_columns(self, df):
        _assert_no_all_null_columns(df)


@pytest.mark.empirical
class TestECMWFHRESPressureLevelsEmpirical:
    """ECMWF IFS HRES historical forecast pressure levels: global, covers Toronto."""

    @pytest.fixture(scope="class")
    def df(self):
        result = _fetch_historical_forecast(
            YANKEE_VENUE_ID, YANKEE_LAT, YANKEE_LON,
            "ecmwf_ifs", EMPIRICAL_YEAR, pressure_levels=True,
            start_date=EMPIRICAL_START, end_date=EMPIRICAL_END,
        )
        assert result is not None, "ECMWF IFS HRES pressure level fetch returned None"
        return result.reset_index(drop=True)

    def test_base_schema(self, df):
        _assert_base_schema(df, YANKEE_VENUE_ID)

    def test_column_count(self, df):
        # Uses ERA5 19 levels × 6 vars + 2 meta cols
        expected = len(ERA5_PRESSURE_LEVELS) * len(ERA5_PRESSURE_BASE_VARS) + 2
        assert len(df.columns) == expected, (
            f"Expected {expected} cols, got {len(df.columns)}"
        )

    def test_all_declared_pressure_columns_present(self, df):
        missing = set(ERA5_PRESSURE_VARS.split(",")) - set(df.columns)
        assert not missing, f"Missing: {sorted(missing)[:5]}..."

    def test_float_dtypes(self, df):
        _assert_float_dtype(df)

    def test_null_rate_per_level(self, df):
        for level in ERA5_PRESSURE_LEVELS:
            col = f"temperature_{level}hPa"
            assert df[col].isna().mean() < 1.0, f"{col} entirely null"

    def test_toronto_covered(self):
        result = _fetch_historical_forecast(
            TORONTO_VENUE_ID, 43.6414, -79.3894,
            "ecmwf_ifs", EMPIRICAL_YEAR, pressure_levels=True,
            start_date=EMPIRICAL_START, end_date=EMPIRICAL_END,
        )
        assert result is not None
        _assert_base_schema(result, TORONTO_VENUE_ID)


@pytest.mark.empirical
class TestGFSForecastPressureLevelsEmpirical:
    """GFS historical forecast pressure levels: global, 44 levels."""

    @pytest.fixture(scope="class")
    def df(self):
        result = _fetch_historical_forecast(
            YANKEE_VENUE_ID, YANKEE_LAT, YANKEE_LON,
            "gfs_global", EMPIRICAL_YEAR, pressure_levels=True,
            start_date=EMPIRICAL_START, end_date=EMPIRICAL_END,
        )
        assert result is not None, "GFS pressure level fetch returned None"
        return result.reset_index(drop=True)

    def test_base_schema(self, df):
        _assert_base_schema(df, YANKEE_VENUE_ID)

    def test_column_count(self, df):
        expected = len(HRRR_PRESSURE_LEVELS) * len(HRRR_PRESSURE_BASE_VARS) + 2
        assert len(df.columns) == expected, (
            f"Expected {expected} cols, got {len(df.columns)}"
        )

    def test_all_declared_pressure_columns_present(self, df):
        missing = set(HRRR_PRESSURE_VARS.split(",")) - set(df.columns)
        assert not missing, f"Missing: {sorted(missing)[:5]}..."

    def test_float_dtypes(self, df):
        _assert_float_dtype(df)

    def test_null_rate_per_level(self, df):
        for level in HRRR_PRESSURE_LEVELS:
            col = f"temperature_{level}hPa"
            assert df[col].isna().mean() < 1.0, f"{col} entirely null"
