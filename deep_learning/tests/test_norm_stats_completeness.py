"""The standardizer sidecar: season completeness, train-only scope, masked math.

`weather_asof_norm.json` is the single artifact that both training (via
_load_weather_asof_artifacts) and live serving (via weather_asof.load_norm_stats) use to
z-score every weather channel. It has no test coverage, and it is built by listing
whatever season parquets happen to exist in S3.

That listing is the defect this file pins down. The chain scripts build seasons
independently across six boxes, and a season that fails to build leaves no trace in the
sidecar — build_norm_stats accumulates the eleven that landed, divides by their counts,
and writes the result as authoritative. The written JSON records mean/std/count but not
WHICH seasons produced them, so nothing downstream can detect the omission either. A
standardizer fit on a subset of seasons shifts every z-score for every game in training
and in production, in the same direction, with no error raised anywhere.

The population is the definition of complete, so the guard derives expected seasons from
game_meta rather than a hardcoded range: it then stays correct when 2027 arrives.
"""

import json

import numpy as np
import pandas as pd
import pytest

import mlb_dl.build_weather_asof as bwa
from mlb_dl.build_weather_asof import (
    CHANNEL_COLS,
    N_DIMS,
    N_OBS_DIMS,
    OFF_FCST,
    OFF_FCST_MASK,
    OFF_OBS,
    OFF_OBS_MASK,
    TRAIN_END_DATE,
)

TRAIN_YEARS = (2015, 2016)  # both < TRAIN_END_DATE, so both are in-scope for stats


class _FakeS3:
    """Records put_object bodies and serves a fixed key listing."""

    def __init__(self, season_keys):
        self._season_keys = list(season_keys)
        self.puts: dict[str, bytes] = {}

    def list_objects_v2(self, Bucket, Prefix):
        return {"Contents": [{"Key": k} for k in self._season_keys]}

    def put_object(self, Bucket, Key, Body):
        self.puts[Key] = Body


def _game_meta(seasons=TRAIN_YEARS, per_season=2):
    """One R game per (season, slot); game_pk encodes the season for readability."""
    rows = []
    for s in seasons:
        for i in range(per_season):
            rows.append({"game_pk": s * 100 + i,
                         "game_date": pd.Timestamp(f"{s}-06-0{i + 1}"),
                         "game_type_code": "R"})
    return pd.DataFrame(rows)


def _season_frame(season, fcst_val, obs_val, mask=1.0, n_rows=4):
    """A season parquet whose fcst dims all hold fcst_val and obs dims obs_val."""
    arr = np.zeros((n_rows, len(CHANNEL_COLS)), dtype=np.float64)
    arr[:, OFF_FCST:OFF_FCST + N_DIMS] = fcst_val
    arr[:, OFF_FCST_MASK:OFF_FCST_MASK + N_DIMS] = mask
    arr[:, OFF_OBS:OFF_OBS + N_OBS_DIMS] = obs_val
    arr[:, OFF_OBS_MASK:OFF_OBS_MASK + N_OBS_DIMS] = mask
    df = pd.DataFrame(arr, columns=CHANNEL_COLS)
    df["game_pk"] = season * 100  # all rows belong to that season's first game
    return df


def _install(monkeypatch, present_seasons, frames_by_season, meta=None):
    fake = _FakeS3([f"{bwa.FS_PREFIX}/weather_asof/season={s}.parquet"
                    for s in present_seasons])
    monkeypatch.setattr(bwa, "s3", lambda: fake)
    gm = _game_meta() if meta is None else meta

    def fake_read(key, columns=None):
        if key.endswith("game_meta.parquet"):
            return gm[columns].copy() if columns else gm.copy()
        season = int(key.split("season=")[1].split(".")[0])
        return frames_by_season[season]

    monkeypatch.setattr(bwa, "_read_parquet", fake_read)
    return fake


def test_missing_season_raises_instead_of_writing_a_biased_standardizer(monkeypatch):
    """The core guard: 2016 exists in the population but never built. Silently fitting on
    2015 alone is what must not happen, because every downstream z-score inherits it."""
    fake = _install(monkeypatch, present_seasons=[2015],
                    frames_by_season={2015: _season_frame(2015, 10.0, 20.0)})
    with pytest.raises(RuntimeError, match="2016"):
        bwa.build_norm_stats()
    assert not fake.puts, "a sidecar was written despite an incomplete population"


def test_missing_val_or_test_season_does_not_block(monkeypatch):
    """The guard's scope must be the seasons that CONTRIBUTE. Stats are train-only, so an
    absent 2024 cannot shift any mean — and 2026's HRRR backfill is deliberately deferred,
    so blocking on non-train seasons would stall the standardizer indefinitely for no
    statistical gain. Whole-artifact coverage is verify_weather_asof_artifact.py's job."""
    meta = pd.concat([_game_meta(),
                      pd.DataFrame([{"game_pk": 202400,
                                     "game_date": pd.Timestamp("2024-06-01"),
                                     "game_type_code": "R"},
                                    {"game_pk": 202600,
                                     "game_date": pd.Timestamp("2026-06-01"),
                                     "game_type_code": "R"}])], ignore_index=True)
    fake = _install(monkeypatch, present_seasons=list(TRAIN_YEARS),  # 2024/2026 absent
                    frames_by_season={2015: _season_frame(2015, 10.0, 20.0),
                                      2016: _season_frame(2016, 10.0, 20.0)},
                    meta=meta)
    bwa.build_norm_stats()
    assert f"{bwa.FS_PREFIX}/weather_asof_norm.json" in fake.puts


def test_complete_population_writes_the_sidecar(monkeypatch):
    fake = _install(monkeypatch, present_seasons=list(TRAIN_YEARS),
                    frames_by_season={2015: _season_frame(2015, 10.0, 20.0),
                                      2016: _season_frame(2016, 10.0, 20.0)})
    bwa.build_norm_stats()
    key = f"{bwa.FS_PREFIX}/weather_asof_norm.json"
    assert key in fake.puts
    stats = json.loads(fake.puts[key])
    assert len(stats["fcst_mean"]) == N_DIMS
    assert len(stats["obs_mean"]) == N_OBS_DIMS


def test_sidecar_records_which_seasons_it_was_fit_on(monkeypatch):
    """Self-describing artifact. Without this, a sidecar built before a repair-build is
    indistinguishable from one built after, and the only symptom is slightly-off z-scores
    in both training and production."""
    fake = _install(monkeypatch, present_seasons=list(TRAIN_YEARS),
                    frames_by_season={2015: _season_frame(2015, 10.0, 20.0),
                                      2016: _season_frame(2016, 10.0, 20.0)})
    bwa.build_norm_stats()
    stats = json.loads(fake.puts[f"{bwa.FS_PREFIX}/weather_asof_norm.json"])
    assert stats["seasons"] == list(TRAIN_YEARS)


def test_val_and_test_games_are_excluded_from_the_standardizer(monkeypatch):
    """The no-leakage invariant on this artifact. A 2024 game (>= TRAIN_END_DATE) carries
    a poison value; if it reached the accumulator the mean would move off 10.0."""
    poison = 1e6
    meta = pd.concat([_game_meta(),
                      pd.DataFrame([{"game_pk": 202400,
                                     "game_date": pd.Timestamp("2024-06-01"),
                                     "game_type_code": "R"}])], ignore_index=True)
    frames = {2015: _season_frame(2015, 10.0, 20.0),
              2016: _season_frame(2016, 10.0, 20.0),
              2024: _season_frame(2024, poison, poison)}
    fake = _install(monkeypatch, present_seasons=[2015, 2016, 2024],
                    frames_by_season=frames, meta=meta)
    bwa.build_norm_stats()
    stats = json.loads(fake.puts[f"{bwa.FS_PREFIX}/weather_asof_norm.json"])
    assert stats["fcst_mean"][0] == pytest.approx(10.0), (
        f"got {stats['fcst_mean'][0]}: a val/test game leaked into the standardizer")
    assert stats["obs_mean"][0] == pytest.approx(20.0)


def test_masked_entries_do_not_drag_the_mean_toward_zero(monkeypatch):
    """Stats are over POPULATED entries only. A masked row stores raw 0, so counting it
    would halve the mean here — and _standardize_masked maps a masked-in raw 0 to
    -mean/std sigma, so a wrong mean is not a harmless offset."""
    populated = _season_frame(2015, 10.0, 20.0, mask=1.0, n_rows=2)
    masked = _season_frame(2015, 0.0, 0.0, mask=0.0, n_rows=2)
    frames = {2015: pd.concat([populated, masked], ignore_index=True),
              2016: _season_frame(2016, 10.0, 20.0)}
    fake = _install(monkeypatch, present_seasons=list(TRAIN_YEARS),
                    frames_by_season=frames)
    bwa.build_norm_stats()
    stats = json.loads(fake.puts[f"{bwa.FS_PREFIX}/weather_asof_norm.json"])
    assert stats["fcst_mean"][0] == pytest.approx(10.0), (
        f"got {stats['fcst_mean'][0]}; 5.0 means masked zeros were counted")
    # 2 populated of 2015's 4 rows, plus all 4 of 2016's. Counting the masked pair would
    # report 8.
    assert stats["fcst_count"][0] == 6.0, "count must be populated entries only"


def test_std_is_zero_only_when_the_dim_never_varies(monkeypatch):
    """A constant dim yields std 0, which both consumers guard with std > 1e-8 -> 1.0.
    This pins the value the guards are written against; if it were NaN instead, the
    `z * mask` product would be NaN rather than 0 even for masked entries."""
    fake = _install(monkeypatch, present_seasons=list(TRAIN_YEARS),
                    frames_by_season={2015: _season_frame(2015, 10.0, 20.0),
                                      2016: _season_frame(2016, 10.0, 20.0)})
    bwa.build_norm_stats()
    stats = json.loads(fake.puts[f"{bwa.FS_PREFIX}/weather_asof_norm.json"])
    assert stats["fcst_std"][0] == pytest.approx(0.0)
    assert not np.isnan(stats["fcst_std"]).any()
    assert not np.isnan(stats["obs_std"]).any()
