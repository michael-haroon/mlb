"""Engine as-of weather mechanics: decision-hour clock, refresh gating,
stale-on-failure, and the batch slice — all without S3 or a checkpoint."""

import numpy as np
import pandas as pd
import pytest
import torch

from mlb_dl.inference_engine import GameInferenceState, LiveInferenceEngine, PregamePrior
from mlb_dl.weather_asof import ASOF_CHANNELS, N_DECISIONS, N_TARGET_HOURS


def _bare_engine(asof_mode=True):
    eng = LiveInferenceEngine.__new__(LiveInferenceEngine)
    eng._asof_mode = asof_mode
    eng._asof_norm_stats = {}
    eng._asof_station_map = {}
    eng.device = torch.device("cpu")
    import threading
    eng._lock = threading.Lock()
    return eng


def _state(hours_ago: float) -> GameInferenceState:
    T = np.zeros((N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS), dtype=np.float32)
    for d in range(N_DECISIONS):
        T[d, :, 0] = d  # value encodes its decision row
    return GameInferenceState(
        game_pk=1, pregame_prior=PregamePrior(game_pk=1),
        weather_asof=torch.from_numpy(T),
        venue_id=3313,
        game_hour_utc=pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours_ago),
        wx_last_decision_hour=0,
    )


def test_decision_hour_clock():
    eng = _bare_engine()
    assert eng._current_decision_hour(_state(0.5)) == 0
    assert eng._current_decision_hour(_state(2.5)) == 2
    assert eng._current_decision_hour(_state(11.0)) == 6   # clipped
    s = _state(1.0)
    s.game_hour_utc = None
    assert eng._current_decision_hour(s) == 0


def test_refresh_failure_keeps_stale_tensor(monkeypatch):
    """S3/assembly failure mid-game must not clear the tensor or advance d."""
    eng = _bare_engine()
    s = _state(3.5)
    import mlb_dl.weather_asof as wa
    monkeypatch.setattr(wa, "fetch_live_asof",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("s3 down")))
    before = s.weather_asof.clone()
    eng._maybe_refresh_weather(s)
    assert torch.equal(s.weather_asof, before)
    assert s.wx_last_decision_hour == 0
    assert s.context_version == 0


def test_refresh_advances_and_bumps_version(monkeypatch):
    eng = _bare_engine()
    s = _state(3.5)
    fresh = np.full((N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS), 7.0, dtype=np.float32)
    import mlb_dl.weather_asof as wa
    monkeypatch.setattr(wa, "fetch_live_asof", lambda *a, **k: fresh)
    eng._maybe_refresh_weather(s)
    assert s.wx_last_decision_hour == 3
    assert s.context_version == 1
    assert float(s.weather_asof[0, 0, 0]) == 7.0
    # same hour again -> no second fetch
    monkeypatch.setattr(wa, "fetch_live_asof",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not refetch")))
    eng._maybe_refresh_weather(s)
    assert s.context_version == 1


def test_no_refresh_outside_asof_mode(monkeypatch):
    eng = _bare_engine(asof_mode=False)
    s = _state(3.5)
    import mlb_dl.weather_asof as wa
    monkeypatch.setattr(wa, "fetch_live_asof",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("legacy mode")))
    eng._maybe_refresh_weather(s)
    assert s.context_version == 0


def test_batch_slice_uses_current_decision_row():
    """The batch must carry wx_asof[d] for the CURRENT hour — serving row 0 at
    hour 4 would price off pregame information."""
    eng = _bare_engine()
    s = _state(4.2)
    d = eng._current_decision_hour(s)
    row = s.weather_asof[d]
    assert row.shape == (N_TARGET_HOURS, ASOF_CHANNELS)
    assert float(row[0, 0]) == 4.0  # encoded decision index
