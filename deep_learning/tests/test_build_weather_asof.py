"""Builder logic tests: soil-lag leakage and pitch-offset alignment.

The soil merge is the D8 persistence rule; using same-hour reanalysis there is
exactly the hindsight the redesign removes, so a poison future value proves
absence. Offset alignment replicates build_pitch_sequence_frame's ordering —
a mismatch silently pairs pitches with wrong decision rows.
"""

import numpy as np
import pandas as pd
import pytest

from mlb_dl.build_weather_asof import (
    SOIL_LAG,
    compute_wx_hour_offsets,
    merge_lagged_soil,
)

GH = pd.Timestamp("2023-07-14 23:00", tz="UTC")
POISON = 9999.0


def test_soil_comes_from_seven_days_back_not_same_hour():
    fcst = pd.DataFrame({"valid_time_utc": [GH, GH + pd.Timedelta(hours=1)]})
    soil = pd.DataFrame({
        "timestamp": [GH - SOIL_LAG, GH + pd.Timedelta(hours=1) - SOIL_LAG,
                      GH],                       # same-hour reanalysis = the old leak
        "soil_moisture_0_to_7cm": [0.31, 0.33, POISON],
    })
    out = merge_lagged_soil(fcst, soil)
    assert out["soil_moisture_0_to_7cm"].tolist() == [0.31, 0.33]
    assert POISON not in out["soil_moisture_0_to_7cm"].values


def test_soil_missing_archive_yields_nan_not_zero():
    fcst = pd.DataFrame({"valid_time_utc": [GH]})
    out = merge_lagged_soil(fcst, pd.DataFrame())
    assert out["soil_moisture_0_to_7cm"].isna().all()  # NaN -> fcst_mask[16]=0


def _pitch_frame():
    return pd.DataFrame({
        "game_pk": [1, 1, 1, 1, 2, 2],
        "play_index": [0, 0, 1, 2, 0, 1],
        "pitch_sequence_index": [0, 1, 0, 0, 0, 0],
        "pitch_start_time": [
            "2023-07-14T23:08:00Z", None,                    # untimed -> ffill 0
            "2023-07-15T00:30:00Z", "2023-07-15T05:59:00Z",
            None, None,                                       # untimed game -> 0
        ],
        "game_hour_utc": [GH] * 4 + [GH] * 2,
    })


def test_offsets_and_fallbacks():
    out = compute_wx_hour_offsets(_pitch_frame())
    g1 = out[out.game_pk == 1]["wx_hour_offset"].tolist()
    assert g1 == [0, 0, 1, 6]                     # 6:59 elapsed -> clipped d=6
    assert out[out.game_pk == 2]["wx_hour_offset"].tolist() == [0, 0]
    assert out["wx_hour_offset"].dtype == np.int8


def test_sequence_index_matches_pitch_frame_ordering():
    """Same sort + cumcount as build_pitch_sequence_frame — rows arrive
    shuffled, alignment must be restored."""
    shuffled = _pitch_frame().sample(frac=1.0, random_state=7)
    out = compute_wx_hour_offsets(shuffled)
    g1 = out[out.game_pk == 1].sort_values("sequence_index")
    assert g1["sequence_index"].tolist() == [0, 1, 2, 3]
    assert g1["wx_hour_offset"].tolist() == [0, 0, 1, 6]


def test_no_future_decision_hour_from_fallback():
    """Fallback must never assign a later decision hour than any timed pitch
    implies — the pregame row (0) is the only safe default."""
    df = _pitch_frame()
    df["pitch_start_time"] = None
    out = compute_wx_hour_offsets(df)
    assert (out["wx_hour_offset"] == 0).all()
