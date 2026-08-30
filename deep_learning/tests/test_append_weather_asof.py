"""Append-to-prepared correctness: ordering, decision-hour mapping, gating."""

import json

import numpy as np
import pytest

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
