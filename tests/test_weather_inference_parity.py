"""
tests/test_weather_inference_parity.py
---------------------------------------
Train/inference parity for the GameTransformer's [4, 22] weather tensor.

The DL weather tensor is built at training time from `ecmwf_ifs_hres_forecast`
(Open-Meteo Historical Forecast API, ~0-3h effective lead time). Inference must
read the *same NWP model*, or the model sees a different atmosphere than it was
fit on. Measured divergence between ECMWF HRES and `best_match` at zero lead is
0.17 SD on air_density and 1.0 SD on wind_speed — i.e. wind carries no signal
across the source boundary. These tests lock the parity contract.

Run:
  conda run -n pred python -m pytest tests/test_weather_inference_parity.py -v
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.scripts.fetch_weather import (
    ECMWF_FORECAST_VARS,
    FORECAST_PRODUCTS,
    _aq_forecast_key,
    _forecast_ecmwf_key,
    _pressure_forecast_key,
)
from deep_learning.mlb_dl.weather_context import (
    WEATHER_TEMPORAL_COLUMNS,
    WEATHER_TEMPORAL_HOURS,
    WEATHER_TOKEN_DIM,
    fetch_live_weather,
)

# Every raw column compute_hour_features() reads off the surface row.
# wind_u/wind_v are derived in _parse_hourly from speed+direction, so the
# fetch only needs the speed/direction pair.
_SURFACE_INPUTS_REQUIRED = [
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "vapour_pressure_deficit",
    "wet_bulb_temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "surface_pressure",
    "cloud_cover",
    "visibility",
    "precipitation",
    "boundary_layer_height",
    "shortwave_radiation",
    "soil_moisture_0_to_7cm",
]


class TestForecastVarCoverage:
    """The live ECMWF fetch must request every column the tensor consumes."""

    def test_ecmwf_forecast_vars_cover_all_tensor_inputs(self):
        requested = set(ECMWF_FORECAST_VARS.split(","))
        missing = [c for c in _SURFACE_INPUTS_REQUIRED if c not in requested]
        assert not missing, (
            f"ECMWF_FORECAST_VARS omits columns the weather tensor reads: {missing}. "
            f"Any omission hard-zeros its tensor dim at inference while training "
            f"had it populated — a silent distribution shift with no missingness mask."
        )

    def test_ecmwf_forecast_vars_exclude_unsupported(self):
        """models=ecmwf_ifs returns all-null for these; requesting them is dead weight."""
        unsupported = {
            "lifted_index", "freezing_level_height",
            "uv_index", "uv_index_clear_sky", "thunderstorm_probability",
        }
        requested = set(ECMWF_FORECAST_VARS.split(","))
        assert not (requested & unsupported), (
            f"ECMWF_FORECAST_VARS requests variables ecmwf_ifs does not produce: "
            f"{sorted(requested & unsupported)}"
        )


class TestForecastEcmwfKey:
    def test_key_layout(self):
        key = _forecast_ecmwf_key(3313, date(2026, 8, 28))
        assert key == (
            "data/weather/source=forecast_ecmwf/venue_id=3313/date=2026-08-28.parquet"
        )

    def test_distinct_from_best_match_forecast(self):
        """The classical pregame path still consumes source=forecast (best_match).
        The ECMWF source must not overwrite it."""
        from data_curation.scripts.fetch_weather import _forecast_key

        assert _forecast_ecmwf_key(1, date(2026, 8, 28)) != _forecast_key(
            1, date(2026, 8, 28)
        )


# ---------------------------------------------------------------------------
# fetch_live_weather — S3 read path, stubbed
# ---------------------------------------------------------------------------


def _synthetic_forecast(venue_id: int, start: pd.Timestamp, hours: int = 48) -> pd.DataFrame:
    """A forecast parquet shaped like _parse_hourly output, all columns populated."""
    ts = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"venue_id": venue_id, "timestamp": ts})
    values = {
        "temperature_2m": 78.0, "dew_point_2m": 60.0, "relative_humidity_2m": 55.0,
        "vapour_pressure_deficit": 1.4, "wet_bulb_temperature_2m": 66.0,
        "wind_speed_10m": 8.0, "wind_u_10m": 5.0, "wind_v_10m": 6.0,
        "wind_gusts_10m": 15.0, "surface_pressure": 1005.0,
        "cloud_cover": 40.0, "visibility": 24000.0, "precipitation": 0.0,
        "boundary_layer_height": 800.0, "shortwave_radiation": 300.0,
        "soil_moisture_0_to_7cm": 0.23,
    }
    for col, val in values.items():
        df[col] = np.float32(val) + rng.normal(0, 1e-3, hours).astype(np.float32)
    return df


@pytest.fixture
def stub_s3(monkeypatch):
    """Route _read_s3_parquet to an in-memory key→DataFrame map."""
    import deep_learning.mlb_dl.weather_context as wc

    store: dict[str, pd.DataFrame] = {}
    requested: list[str] = []

    def _fake_read(_client, _bucket, key):
        requested.append(key)
        return store.get(key)

    monkeypatch.setattr(wc, "_read_s3_parquet", _fake_read)
    monkeypatch.setattr("boto3.client", lambda *a, **k: object())
    return store, requested


class TestFetchLiveWeatherSource:
    def test_reads_ecmwf_forecast_source(self, stub_s3):
        """Inference must read the ECMWF source, matching the training NWP model."""
        store, requested = stub_s3
        game_hour = pd.Timestamp("2026-08-28 23:00", tz="UTC")
        store[_forecast_ecmwf_key(3313, date(2026, 8, 28))] = _synthetic_forecast(
            3313, pd.Timestamp("2026-08-28 00:00", tz="UTC")
        )

        out = fetch_live_weather(3313, game_hour, park_azimuths={3313: 0.0})

        assert out.shape == (WEATHER_TEMPORAL_HOURS, WEATHER_TOKEN_DIM)
        assert np.any(out != 0.0), (
            f"tensor is all zeros; keys requested were {requested}"
        )
        assert any("source=forecast_ecmwf" in k for k in requested), (
            f"fetch_live_weather did not read source=forecast_ecmwf; got {requested}"
        )

    def test_soil_moisture_dim_populated(self, stub_s3):
        """dim 16 was structurally dead at inference: soil_moisture_0_to_7cm was
        absent from the live fetch while training had it 86% non-zero."""
        store, _ = stub_s3
        idx = WEATHER_TEMPORAL_COLUMNS.index("wxt_soil_moisture")
        store[_forecast_ecmwf_key(3313, date(2026, 8, 28))] = _synthetic_forecast(
            3313, pd.Timestamp("2026-08-28 00:00", tz="UTC")
        )

        out = fetch_live_weather(
            3313, pd.Timestamp("2026-08-28 23:00", tz="UTC"), park_azimuths={3313: 0.0}
        )

        assert out[0, idx] == pytest.approx(0.23, abs=1e-2), (
            f"dim {idx} (soil_moisture) = {out[0, idx]}, expected ~0.23"
        )

    def test_soil_moisture_falls_back_to_era5_persistence(self, stub_s3):
        """No live Open-Meteo model serves soil_moisture_0_to_7cm with real values —
        operational ecmwf_ifs returns the column non-null but literally 0.0 for every
        hour. Zero is not a neutral fill here: training encodes *artificial turf* as
        0.0 (13.9% of rows), so a hard zero at a grass park tells the model it is a
        dome. ERA5 persistence at the 7-day archive lag retains r=0.622 / R^2=0.387
        (RMSE/SD=0.825), measured over 14 venue-years — degraded but in-distribution.
        """
        store, requested = stub_s3
        idx = WEATHER_TEMPORAL_COLUMNS.index("wxt_soil_moisture")
        vid = 3313
        game_hour = pd.Timestamp("2026-08-28 23:00", tz="UTC")

        # Production shape: the live forecast carries the column, all-zero.
        fc = _synthetic_forecast(vid, pd.Timestamp("2026-08-28 00:00", tz="UTC"))
        fc["soil_moisture_0_to_7cm"] = 0.0
        store[_forecast_ecmwf_key(vid, game_hour.date())] = fc

        # ERA5 archive stops 7 days short of the game (ARCHIVE_LAG_DAYS=7).
        arch_ts = pd.date_range(
            pd.Timestamp("2026-08-14 00:00", tz="UTC"), periods=192, freq="h"
        )
        store[f"data/weather/source=era5/venue_id={vid}/year=2026.parquet"] = (
            pd.DataFrame({
                "venue_id": vid,
                "timestamp": arch_ts,
                "soil_moisture_0_to_7cm": np.linspace(0.30, 0.19, len(arch_ts)),
            })
        )

        out = fetch_live_weather(vid, game_hour, park_azimuths={vid: 0.0})

        assert out[0, idx] == pytest.approx(0.19, abs=1e-3), (
            f"dim {idx} (soil_moisture) = {out[0, idx]}; expected the last available "
            f"ERA5 value 0.19, not the live forecast's 0.0. "
            f"keys requested: {requested}"
        )

    def test_soil_moisture_prefers_live_value_when_real(self, stub_s3):
        """Persistence is a fallback, not an override. If a live source ever starts
        serving the 0-7cm band, the fresher value must win."""
        store, _ = stub_s3
        idx = WEATHER_TEMPORAL_COLUMNS.index("wxt_soil_moisture")
        vid = 3313
        game_hour = pd.Timestamp("2026-08-28 23:00", tz="UTC")

        store[_forecast_ecmwf_key(vid, game_hour.date())] = _synthetic_forecast(
            vid, pd.Timestamp("2026-08-28 00:00", tz="UTC")
        )  # carries a real 0.23
        store[f"data/weather/source=era5/venue_id={vid}/year=2026.parquet"] = (
            pd.DataFrame({
                "venue_id": vid,
                "timestamp": pd.date_range(
                    pd.Timestamp("2026-08-14 00:00", tz="UTC"), periods=24, freq="h"
                ),
                "soil_moisture_0_to_7cm": 0.40,
            })
        )

        out = fetch_live_weather(vid, game_hour, park_azimuths={vid: 0.0})

        assert out[0, idx] == pytest.approx(0.23, abs=1e-2), (
            f"stale ERA5 (0.40) overrode a real live value (0.23): got {out[0, idx]}"
        )

    def test_soil_moisture_ignores_all_zero_archive(self, stub_s3):
        """The 4 turf venues are all-zero in ERA5 too. Walking back for a non-zero
        value must not run off the end of the file and crash or return a stale
        out-of-season reading for them."""
        store, _ = stub_s3
        idx = WEATHER_TEMPORAL_COLUMNS.index("wxt_soil_moisture")
        vid = 2523  # Rogers Centre, dome/turf
        game_hour = pd.Timestamp("2026-08-28 23:00", tz="UTC")

        fc = _synthetic_forecast(vid, pd.Timestamp("2026-08-28 00:00", tz="UTC"))
        fc["soil_moisture_0_to_7cm"] = 0.0
        store[_forecast_ecmwf_key(vid, game_hour.date())] = fc
        store[f"data/weather/source=era5/venue_id={vid}/year=2026.parquet"] = (
            pd.DataFrame({
                "venue_id": vid,
                "timestamp": pd.date_range(
                    pd.Timestamp("2026-08-14 00:00", tz="UTC"), periods=24, freq="h"
                ),
                "soil_moisture_0_to_7cm": 0.0,
            })
        )

        out = fetch_live_weather(vid, game_hour, park_azimuths={vid: 0.0})
        assert out[0, idx] == 0.0

    def test_late_west_coast_game_finds_prior_utc_date(self, stub_s3):
        """A 22:00 PT first pitch is 05:00 UTC the *next* day. Keying the forecast
        file on the game's UTC date looks for a file that won't be written until
        the following daily run — silently zeroing weather for late games."""
        store, requested = stub_s3
        game_hour = pd.Timestamp("2026-08-29 05:00", tz="UTC")  # 22:00 PT Aug 28
        # Only the Aug-28 file exists, as it would in production.
        store[_forecast_ecmwf_key(5325, date(2026, 8, 28))] = _synthetic_forecast(
            5325, pd.Timestamp("2026-08-28 00:00", tz="UTC"), hours=72
        )

        out = fetch_live_weather(5325, game_hour, park_azimuths={5325: 0.0})

        assert np.any(out != 0.0), (
            f"late West Coast game got an all-zero weather tensor; "
            f"keys requested: {requested}"
        )

    def test_missing_forecast_returns_zeros(self, stub_s3):
        """Graceful degradation is still required when nothing is available."""
        store, _ = stub_s3
        out = fetch_live_weather(
            9999, pd.Timestamp("2026-08-28 23:00", tz="UTC"), park_azimuths={}
        )
        assert out.shape == (WEATHER_TEMPORAL_HOURS, WEATHER_TOKEN_DIM)
        assert np.all(out == 0.0)


class TestAllDimsRecoverable:
    """Dims 17-21 previously read year-partitioned archives gated by
    ARCHIVE_LAG_DAYS=7, so they were hard-zero for any live game. Both have real
    forecast endpoints, so no dim needs to be lost at inference."""

    def test_no_dim_reads_a_lagged_archive(self):
        """fetch_live_weather must not read a year=-partitioned archive source."""
        import inspect

        from deep_learning.mlb_dl import weather_context as wc

        src = inspect.getsource(wc.fetch_live_weather)
        assert "source=air_quality/" not in src, (
            "dims 17-19 still read the 7-day-lagged air_quality archive"
        )
        assert "source=hrrr_forecast_pressure/" not in src, (
            "dims 20-21 still read the 7-day-lagged hrrr_forecast_pressure archive"
        )

    def test_pressure_forecast_uses_training_model(self):
        """Training dims 20-21 come from hrrr_forecast_pressure (models=gfs_hrrr).
        ecmwf_ifs serves no pressure levels, so gfs_hrrr is also the parity choice."""
        import inspect

        from data_curation.scripts import fetch_weather

        src = inspect.getsource(fetch_weather._fetch_pressure_forecast)
        assert '"gfs_hrrr"' in src, "pressure forecast must use models=gfs_hrrr"

    def test_all_22_dims_populated_end_to_end(self, stub_s3):
        """With all three forecast products present, no dim is silently zero."""
        store, requested = stub_s3
        vid, issue = 3313, date(2026, 8, 28)
        base = pd.Timestamp("2026-08-28 00:00", tz="UTC")
        game_hour = pd.Timestamp("2026-08-28 23:00", tz="UTC")

        store[_forecast_ecmwf_key(vid, issue)] = _synthetic_forecast(vid, base)

        ts = pd.date_range(base, periods=48, freq="h", tz="UTC")
        store[_aq_forecast_key(vid, issue)] = pd.DataFrame({
            "venue_id": vid, "timestamp": ts,
            "us_aqi": 38.0, "pm2_5": 5.5, "ozone": 71.0,
        })
        store[_pressure_forecast_key(vid, issue)] = pd.DataFrame({
            "venue_id": vid, "timestamp": ts,
            "temperature_1000hPa": 80.0, "temperature_850hPa": 62.0,
            "geopotential_height_1000hPa": 110.0,
            "geopotential_height_850hPa": 1500.0,
            "wind_speed_850hPa": 20.0, "wind_direction_850hPa": 240.0,
        })

        out = fetch_live_weather(vid, game_hour, park_azimuths={vid: 0.0})

        zero_dims = [
            WEATHER_TEMPORAL_COLUMNS[i]
            for i in range(WEATHER_TOKEN_DIM)
            if not out[:, i].any()
        ]
        assert not zero_dims, (
            f"dims still zero with all products present: {zero_dims}. "
            f"keys requested: {requested}"
        )

    def test_visibility_comes_from_hrrr_not_ecmwf(self, stub_s3):
        """build_multihour_weather_frame overwrites ECMWF's visibility with HRRR's
        (feature_store.py: "HRRR visibility (only source with it populated)"), so
        dim 11 is an HRRR feature. Measured live at Cleveland over 48h: ECMWF
        sd=4981 vs HRRR sd=10812, RMSE=8047 (0.744 SD) — ECMWF's diagnostic is far
        smoother, so serving it would compress the dim the model was fit on."""
        store, _ = stub_s3
        idx = WEATHER_TEMPORAL_COLUMNS.index("wxt_visibility")
        vid = 3313
        game_hour = pd.Timestamp("2026-08-28 23:00", tz="UTC")
        base = pd.Timestamp("2026-08-28 00:00", tz="UTC")

        fc = _synthetic_forecast(vid, base)
        fc["visibility"] = 44960.0  # ECMWF's smoother value
        store[_forecast_ecmwf_key(vid, game_hour.date())] = fc

        ts = pd.date_range(base, periods=48, freq="h", tz="UTC")
        store[_pressure_forecast_key(vid, game_hour.date())] = pd.DataFrame({
            "venue_id": vid, "timestamp": ts,
            "visibility": 18000.0,  # HRRR sees haze the coarse model misses
            "temperature_1000hPa": 80.0, "temperature_850hPa": 62.0,
            "geopotential_height_1000hPa": 110.0,
            "geopotential_height_850hPa": 1500.0,
            "wind_speed_850hPa": 20.0, "wind_direction_850hPa": 240.0,
        })

        out = fetch_live_weather(vid, game_hour, park_azimuths={vid: 0.0})
        assert out[0, idx] == pytest.approx(18000.0, rel=1e-3), (
            f"dim {idx} (visibility) = {out[0, idx]}; expected HRRR's 18000, "
            f"not ECMWF's 44960"
        )

    def test_hrrr_product_requests_visibility(self):
        from data_curation.scripts.fetch_weather import PRESSURE_FORECAST_VARS

        assert "visibility" in PRESSURE_FORECAST_VARS.split(","), (
            "the gfs_hrrr product must request visibility — it is the only source "
            "that populates dim 11 in training"
        )

    def test_missing_pressure_product_zeroes_dims_20_21_and_visibility(self, stub_s3):
        """A missing HRRR pressure read must zero dims 20, 21 AND 11 — not fall back to ECMWF.

        REWRITTEN 2026-08-31 with the venue-2523 exclusion. This used to assert the opposite —
        that venue 2523 must NOT trigger a pressure fetch — which pinned an exclusion that named
        Steinbrenner Field (Tampa), not Rogers Centre, and that HRRR's CONUS grid made
        unnecessary for either park. The invariant worth keeping was never about a venue: it is
        that whenever HRRR pressure is absent, inference must reproduce training's NaN→0 rather
        than leak ECMWF's ~45 km visibility into dim 11, a value the model never saw. Driving
        that with an absent product instead of a hardcoded id tests the real code path and also
        covers the case the id check never did — a transient S3 miss at any venue.
        """
        store, requested = stub_s3
        store[_forecast_ecmwf_key(2523, date(2026, 8, 28))] = _synthetic_forecast(
            2523, pd.Timestamp("2026-08-28 00:00", tz="UTC")
        )
        # No hrrr_pressure_forecast key is seeded, so the read comes back empty.
        out = fetch_live_weather(
            2523, pd.Timestamp("2026-08-28 23:00", tz="UTC"), park_azimuths={2523: 0.0}
        )
        assert any("hrrr_pressure_forecast" in k for k in requested), (
            "every venue must attempt the HRRR pressure read; no venue id may skip it"
        )
        assert not out[:, 20].any() and not out[:, 21].any()

        vis = WEATHER_TEMPORAL_COLUMNS.index("wxt_visibility")
        assert not out[:, vis].any(), (
            f"dim {vis} (visibility) = {out[0, vis]}; with no HRRR pressure row it must be 0 to "
            f"match training's NaN→0, not ECMWF's ~44960"
        )


class TestForecastProductRegistry:
    """Daily mode and the 6-hourly refresh must fetch the same product set — a
    product present in one path and absent from the other is the original bug."""

    def test_registry_covers_every_source_inference_reads(self):
        labels = {label for label, _, _ in FORECAST_PRODUCTS}
        for required in ("forecast_ecmwf", "air_quality_forecast", "hrrr_pressure_forecast"):
            assert required in labels, f"{required} missing from FORECAST_PRODUCTS"

    def test_registry_retains_classical_products(self):
        labels = {label for label, _, _ in FORECAST_PRODUCTS}
        assert "forecast" in labels, "classical pregame path reads source=forecast"
        assert "ensemble" in labels

    def test_both_runners_use_the_registry(self):
        import inspect

        from data_curation.scripts import fetch_weather

        for fn in (fetch_weather._run_daily, fetch_weather.run_forecast_refresh):
            assert "FORECAST_PRODUCTS" in inspect.getsource(fn), (
                f"{fn.__name__} does not iterate FORECAST_PRODUCTS and will drift"
            )


# ---------------------------------------------------------------------------
# InferenceEngine wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def engine(tmp_path):
    """A LiveInferenceEngine on a throwaway checkpoint (structure only)."""
    import torch

    from deep_learning.mlb_dl.inference_engine import LiveInferenceEngine
    from deep_learning.mlb_dl.models import LiveGameModel

    ckpt = tmp_path / "m.pt"
    torch.save(
        {
            "model_state_dict": LiveGameModel(
                feature_dim=40, hidden_dim=128, dropout=0.0
            ).state_dict(),
            "config": {"feature_dim": 40, "hidden_dim": 128},
            "feature_mean": {},
            "feature_std": {},
        },
        ckpt,
    )
    return LiveInferenceEngine(model_path=str(ckpt), device="cpu")


class TestInferenceEngineWiring:
    """weather_context defaulted to None and nothing ever populated it, so
    game_transformer.py's `if "weather_temporal" in context_batch` guard silently
    skipped the weather branch for every live game."""

    def test_register_game_fetches_weather(self, engine, monkeypatch):
        import deep_learning.mlb_dl.inference_engine as ie
        from deep_learning.mlb_dl.inference_engine import PregamePrior

        sentinel = np.arange(
            WEATHER_TEMPORAL_HOURS * WEATHER_TOKEN_DIM, dtype=np.float32
        ).reshape(WEATHER_TEMPORAL_HOURS, WEATHER_TOKEN_DIM)
        calls = []

        def _fake(venue_id, game_hour_utc, park_azimuths, **kw):
            calls.append((venue_id, game_hour_utc, park_azimuths))
            return sentinel

        monkeypatch.setattr(ie, "fetch_live_weather", _fake)

        engine.register_game(
            12345,
            PregamePrior(game_pk=12345),
            venue_id=3313,
            game_hour_utc=pd.Timestamp("2026-08-28 23:00", tz="UTC"),
        )

        assert calls, "register_game did not call fetch_live_weather"
        state = engine._games[12345]
        assert state.weather_context is not None, "weather_context still None"
        assert tuple(state.weather_context.shape) == (
            WEATHER_TEMPORAL_HOURS,
            WEATHER_TOKEN_DIM,
        )
        assert np.allclose(state.weather_context.numpy(), sentinel)

    def test_explicit_weather_context_wins(self, engine, monkeypatch):
        """A caller that already has the tensor must not trigger a second S3 read."""
        import torch

        import deep_learning.mlb_dl.inference_engine as ie
        from deep_learning.mlb_dl.inference_engine import PregamePrior

        def _boom(*a, **k):
            raise AssertionError("fetch_live_weather called despite explicit tensor")

        monkeypatch.setattr(ie, "fetch_live_weather", _boom)
        supplied = torch.ones(WEATHER_TEMPORAL_HOURS, WEATHER_TOKEN_DIM)

        engine.register_game(
            1,
            PregamePrior(game_pk=1),
            weather_context=supplied,
            venue_id=3313,
            game_hour_utc=pd.Timestamp("2026-08-28 23:00", tz="UTC"),
        )
        assert torch.equal(engine._games[1].weather_context, supplied)

    def test_fetch_failure_does_not_block_registration(self, engine, monkeypatch):
        """Weather is one of several context blocks. An S3 outage must degrade to
        a weather-less forward pass, not drop the game from live pricing."""
        import deep_learning.mlb_dl.inference_engine as ie
        from deep_learning.mlb_dl.inference_engine import PregamePrior

        monkeypatch.setattr(
            ie, "fetch_live_weather",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("s3 down")),
        )
        engine.register_game(
            7,
            PregamePrior(game_pk=7),
            venue_id=3313,
            game_hour_utc=pd.Timestamp("2026-08-28 23:00", tz="UTC"),
        )
        assert 7 in engine._games
        assert engine._games[7].weather_context is None

    def test_engine_uses_calibrated_park_azimuths(self, engine):
        """Dims 4-5 are wind rotated into park coordinates by CF azimuth. Training
        used the calibrated park_azimuths.json; defaulting to 0° would rotate every
        park to due north and invert wind_out_cf at roughly half of them."""
        assert engine.park_azimuths, (
            "engine loaded no park azimuths — dims 4-5 would use the 0° default"
        )
        assert len(engine.park_azimuths) >= 25, (
            f"only {len(engine.park_azimuths)} azimuths loaded; expected all 30 parks"
        )


# ---------------------------------------------------------------------------
# Daemon wiring
# ---------------------------------------------------------------------------


class TestDaemonWiring:
    def test_daily_enrichment_refreshes_weather(self, monkeypatch):
        """run_daily_weather() documented itself as called from
        _daily_enrichment, but nothing called it — the forecast parquet went
        stale and fetch_live_weather silently returned zeros."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_curation" / "scripts"))
        import live_daemon

        calls = []
        monkeypatch.setattr(
            "fetch_weather.run_daily_weather", lambda **kw: calls.append(kw)
        )
        monkeypatch.setattr(
            live_daemon.LiveDaemon, "__init__", lambda self: None
        )
        daemon = live_daemon.LiveDaemon()
        daemon._daily_enrichment()

        assert calls, "_daily_enrichment did not call fetch_weather.run_daily_weather"

    def test_forecast_refresh_is_lighter_than_full_daily(self):
        """The 6-hourly refresh must not re-pull all 13 archive sources for every
        venue — that is hundreds of multi-MB calls and will hit rate limits."""
        import inspect

        from data_curation.scripts import fetch_weather

        assert hasattr(fetch_weather, "run_forecast_refresh"), (
            "no run_forecast_refresh(); the 6-hourly loop would have to call "
            "run_daily_weather() and re-pull every archive source"
        )
        # Strip the docstring: it names _dispatch_backfill to explain the
        # exclusion, which would otherwise false-positive this check.
        fn = fetch_weather.run_forecast_refresh
        body = inspect.getsource(fn).replace(fn.__doc__ or "", "")
        assert "_dispatch_backfill" not in body, (
            "run_forecast_refresh must not touch archive sources"
        )
