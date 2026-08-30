"""Append-to-prepared correctness: ordering, decision-hour mapping, gating."""

import json

import numpy as np
import pytest

from mlb_dl import append_weather_asof_to_prepared as awa
from mlb_dl.append_weather_asof_to_prepared import append_split
from mlb_dl.weather_asof import ASOF_CHANNELS, N_DECISIONS, N_TARGET_HOURS


def _fake_split(tmp_path, game_pks, samples):
    d = tmp_path / "train"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"n_games": len(game_pks),
                                                 "n_samples": len(samples)}))
    np.save(d / "game_pks.npy", np.array(game_pks, dtype=np.int64))
    np.save(d / "sample_to_game.npy", np.array([s[0] for s in samples], dtype=np.int32))
    np.save(d / "prefix_length.npy", np.array([s[1] for s in samples], dtype=np.int16))
    return d


def test_append_orders_by_game_pks_and_maps_decision_hours(tmp_path):
    pks = [111, 222]
    # samples: (game_index, prefix_len)
    samples = [(0, 0), (0, 50), (1, 120), (1, 5000)]
    d = _fake_split(tmp_path, pks, samples)

    asof = {pk: np.full((N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS), float(pk), np.float32)
            for pk in pks}
    offsets = {111: np.array([0] * 40 + [1] * 40, np.int8),   # 80 pitches
               222: np.array([0] * 100 + [2] * 100, np.int8)}  # 200 pitches
    append_split(d, asof, offsets)

    T = np.load(d / "weather_asof.npy")
    assert T.shape == (2, N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS)
    assert T[0, 0, 0, 0] == 111.0 and T[1, 0, 0, 0] == 222.0

    wx_d = np.load(d / "wx_decision_hour.npy")
    # pregame -> 0; cut 50 in game 111 -> offsets[49]=1; cut 120 in 222 -> offsets[119]=2;
    # cut past the array end clamps to the last offset (2)
    assert wx_d.tolist() == [0, 1, 2, 2]

    m = json.loads((d / "manifest.json").read_text())
    assert m["has_weather_asof"] is True and m["asof_channels"] == ASOF_CHANNELS


def test_append_refuses_sparse_coverage(tmp_path):
    d = _fake_split(tmp_path, [111, 222, 333], [(0, 0)])
    asof = {111: np.zeros((N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS), np.float32)}
    with pytest.raises(SystemExit):
        append_split(d, asof, {})


# --- the norm sidecar must exist BEFORE the values are baked in --------------
def test_append_refuses_to_run_without_the_norm_sidecar(tmp_path):
    """_load_weather_asof_artifacts z-scores only if weather_asof_norm.json exists; with
    no sidecar it logs a warning and returns RAW units (temperature ~300, pressure
    ~101325). Appending then bakes raw units permanently into the prepared tensors, and
    training reads the .npy without ever re-checking, so the treatment arm silently
    trains on unnormalized weather and the A/B reads as 'weather did not help'.

    norm-stats can only run after every season is built, so this ordering hazard is real
    rather than theoretical, and a warning is not enough.
    """
    fs = tmp_path / "fs"
    (fs / "weather_asof").mkdir(parents=True)
    with pytest.raises(SystemExit) as e:
        awa.assert_norm_sidecar(fs)
    assert "weather_asof_norm.json" in str(e.value)


def test_append_accepts_a_present_norm_sidecar(tmp_path):
    fs = tmp_path / "fs"
    (fs / "weather_asof").mkdir(parents=True)
    (fs / "weather_asof_norm.json").write_text(json.dumps(
        {"fcst_mean": [0.0], "fcst_std": [1.0], "obs_mean": [0.0], "obs_std": [1.0]}))
    awa.assert_norm_sidecar(fs)  # must not raise


def test_main_checks_the_sidecar_before_writing_anything():
    """A guard that runs after the write is no guard at all.

    The bound is deliberately append_split and not _load_weather_asof_artifacts: the
    loader is read-only (verified 2026-08-30), so a check sitting between the load and
    the first append is still correct, just slower to fail. Mutation testing flagged that
    reordering as uncaught; it is an equivalent mutant, not a gap.
    """
    import inspect
    src = inspect.getsource(awa.main)
    assert "assert_norm_sidecar" in src
    assert src.index("assert_norm_sidecar") < src.index("append_split")


# --- offset truncation must be counted, not silently clamped -----------------
def test_widespread_offset_truncation_fails(tmp_path):
    """prefix_length and wx_hour_offset both derive from the same pitches parquet, so a
    prefix that runs past the end of the offsets array means the two artifacts came from
    different snapshots -- which is exactly what refreshing the feature store without
    rebuilding the offsets would produce. The clamp then hands every affected sample the
    game's FINAL decision hour, the most leaky row available.
    """
    n = 2000
    samples = [(0, 5000) for _ in range(n)]        # every prefix past the array end
    d = _fake_split(tmp_path, [111], samples)
    asof = {111: np.zeros((N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS), np.float32)}
    offsets = {111: np.array([0] * 10, np.int8)}   # only 10 pitches
    with pytest.raises(SystemExit) as e:
        append_split(d, asof, offsets)
    assert "truncat" in str(e.value).lower()


def test_isolated_offset_truncation_is_still_tolerated(tmp_path):
    """The documented single-sample clamp stays legal: a rate over a handful of samples
    carries no information, so the guard needs volume before it judges."""
    samples = [(0, 5)] * 1500 + [(0, 5000)]
    d = _fake_split(tmp_path, [111], samples)
    asof = {111: np.zeros((N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS), np.float32)}
    offsets = {111: np.array([0] * 10 + [3] * 10, np.int8)}
    append_split(d, asof, offsets)   # 1/1501 truncated -> under the threshold
    wx = np.load(d / "wx_decision_hour.npy")
    assert wx[-1] == 3, "the clamp itself must still work for the rare legitimate case"


def test_truncation_threshold_has_headroom_over_zero_and_under_a_snapshot_mismatch():
    assert 0.0 < awa.MAX_OFFSET_TRUNCATION_RATE < 0.10
    assert awa.MIN_SAMPLES_FOR_TRUNCATION_RATE >= 100
