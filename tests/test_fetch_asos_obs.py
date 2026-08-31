"""Contract tests for the ASOS observation ingestion (the obs channel's ground truth).

The as-of weather tensor's leakage guarantee rests on `available_time_utc`, and its
value integrity rests on IEM's 'M'/'T' markers being handled — a 'T' (trace precip)
parsed as NaN would wrongly mask a populated report, and an 'M' parsed as a string
would poison the numeric column.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_curation" / "scripts"))

from fetch_asos_obs import (  # noqa: E402
    ASOS_AVAILABILITY_LAG_MIN,
    IEM_FIELDS,
    _c_to_f,
    _hpa_to_inhg,
    _iem_params,
    _normalize,
    parse_ice_accretion,
    parse_peak_wind,
    parse_snowdepth,
)


def test_request_excludes_5min_madis_rows():
    """Without report_type, IEM interleaves 5-minute MADIS rows (91% NaN tmpf
    measured at OWD 2024) that would poison latest-report-in-window selection.
    Only routine METAR (3) + specials (4) are complete reports."""
    p = _iem_params("BOS", 2024)
    assert p["report_type"] == [3, 4]


def _raw_frame(**overrides):
    """One IEM CSV row as pandas parses it (numeric cols arrive as object dtype
    whenever any row contains 'M' or 'T')."""
    base = {
        "station": "BOS",
        "valid": "2015-06-01 00:15",
        "tmpf": "48.20", "dwpf": "46.40", "relh": "93.45",
        "drct": "50.00", "sknt": "12.00", "gust": "M",
        "alti": "30.26", "mslp": "M", "vsby": "10.00",
        "skyc1": "OVC", "skyl1": "3000.00",
        "skyc2": "M", "skyl2": "M", "skyc3": "M", "skyl3": "M",
        "skyc4": "M", "skyl4": "M",
        "p01i": "T",
        "wxcodes": "-RA BR", "peak_wind_gust": "35.00",
        "peak_wind_drct": "280.00", "peak_wind_time": "2015-06-01 00:02",
        "ice_accretion_1hr": "T", "ice_accretion_3hr": "M",
        "ice_accretion_6hr": "M", "snowdepth": "M",
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_missing_marker_becomes_nan_not_string():
    df = _normalize(_raw_frame())
    assert pd.isna(df["gust"].iloc[0])
    assert pd.isna(df["mslp"].iloc[0])
    # Column must be numeric so downstream arithmetic works
    assert pd.api.types.is_numeric_dtype(df["gust"])


def test_trace_precip_is_zero_not_missing():
    """A trace IS a measurement (~0.00\"); NaN would flip obs_mask to 0 for a
    populated report."""
    df = _normalize(_raw_frame())
    assert df["p01i"].iloc[0] == 0.0


def test_real_values_survive():
    df = _normalize(_raw_frame())
    assert df["tmpf"].iloc[0] == pytest.approx(48.20)
    assert df["sknt"].iloc[0] == pytest.approx(12.00)
    assert df["skyc1"].iloc[0] == "OVC"


def test_missing_sky_cover_is_missing_not_M():
    """Contract: the sentinel string must not survive; any NA representation is fine."""
    df = _normalize(_raw_frame())
    assert pd.isna(df["skyc2"].iloc[0])
    assert df["skyc2"].iloc[0] != "M"


def test_available_time_is_valid_plus_lag():
    """The leakage filter selects on available_time_utc; it must be report time
    plus the dissemination lag, never the report time itself."""
    df = _normalize(_raw_frame())
    expected = pd.Timestamp("2015-06-01 00:15", tz="UTC") + pd.Timedelta(
        minutes=ASOS_AVAILABILITY_LAG_MIN
    )
    assert df["available_time_utc"].iloc[0] == expected
    assert ASOS_AVAILABILITY_LAG_MIN > 0


def test_valid_utc_is_tz_aware_utc():
    df = _normalize(_raw_frame())
    assert str(df["valid_utc"].dt.tz) == "UTC"


def test_unit_conversions_match_awc_to_iem_semantics():
    """AWC serves °C and hPa; IEM serves °F and inHg. The converters make the
    live rows land in the archive's units."""
    assert _c_to_f(0.0) == pytest.approx(32.0)
    assert _c_to_f(100.0) == pytest.approx(212.0)
    assert _c_to_f(None) is None
    # 1013.25 hPa = 29.92 inHg (standard atmosphere)
    assert _hpa_to_inhg(1013.25) == pytest.approx(29.92, abs=0.01)
    assert _hpa_to_inhg(None) is None


def test_field_list_covers_observable_dims():
    """The 22-dim layout's observable dims (0-13) need temp, moisture, wind,
    pressure, visibility, sky, precip — all must be requested from IEM."""
    for needed in ("tmpf", "dwpf", "relh", "drct", "sknt", "gust",
                   "alti", "vsby", "skyc1", "p01i"):
        assert needed in IEM_FIELDS


def test_field_list_covers_game_relevant_extras():
    """Present weather, peak wind, icing, snow — everything IEM offers that is
    about the game and reconstructible live from the same METAR reports."""
    for needed in ("wxcodes", "peak_wind_gust", "peak_wind_drct",
                   "peak_wind_time", "ice_accretion_1hr", "skyc4", "snowdepth"):
        assert needed in IEM_FIELDS


def test_wxcodes_survive_and_missing_is_na():
    df = _normalize(_raw_frame())
    assert df["wxcodes"].iloc[0] == "-RA BR"
    df2 = _normalize(_raw_frame(wxcodes="M"))
    assert pd.isna(df2["wxcodes"].iloc[0])


def test_ice_trace_is_zero_and_peak_wind_numeric():
    df = _normalize(_raw_frame())
    assert df["ice_accretion_1hr"].iloc[0] == 0.0        # trace = measurement
    assert pd.isna(df["ice_accretion_3hr"].iloc[0])
    assert df["peak_wind_gust"].iloc[0] == pytest.approx(35.0)
    assert str(df["peak_wind_time"].dt.tz) == "UTC"


def test_parse_peak_wind_remark():
    rt = pd.Timestamp("2023-07-14 23:53", tz="UTC")
    g, d, t = parse_peak_wind("KBOS 142353Z ... RMK AO2 PK WND 28045/2317 SLP123", rt)
    assert (g, d) == (45.0, 280.0)
    assert t == pd.Timestamp("2023-07-14 23:17", tz="UTC")
    # minutes-only form, wrapping to previous hour when ahead of report time
    g, d, t = parse_peak_wind("RMK PK WND 31038/55", rt)
    assert t == pd.Timestamp("2023-07-14 22:55", tz="UTC")
    assert parse_peak_wind("RMK AO2 SLP123", rt) == (None, None, None)


def test_parse_ice_and_snowdepth_remarks():
    ice = parse_ice_accretion("RMK AO2 I1002 I3010 SLP123")
    assert ice["ice_accretion_1hr"] == pytest.approx(0.02)
    assert ice["ice_accretion_3hr"] == pytest.approx(0.10)
    assert ice["ice_accretion_6hr"] is None
    assert parse_snowdepth("RMK 4/012") == 12.0
    assert parse_snowdepth("RMK AO2") is None
