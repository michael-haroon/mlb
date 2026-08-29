"""Tests proving the pitches_raw memory fix works end-to-end.

Three layers:
  1. Structural: pitches_raw absent from _TABLE_CONFIG, load_pitches_raw() exists
  2. Behavioural: load_all() never returns pitches_raw key; load_pitches_raw()
     calls catalog.read_table("pitches", PITCH_LEVEL_COLUMNS, ...)
  3. Integration: build.py's branch — raw is del'd before pitches_raw is loaded,
     so they cannot coexist; compute_pitch_level_features receives the result

Run:
  conda run -n pred python -m pytest tests/test_pitches_raw_memory.py -v
"""
from __future__ import annotations

import gc
import sys
import types
import weakref
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.engineering.constants import PITCH_LEVEL_COLUMNS
from classical_learning.engineering.data_loader import (
    _TABLE_CONFIG,
    load_all,
    load_pitches_raw,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_catalog_mock(pitches_df: pd.DataFrame, other_df: pd.DataFrame) -> MagicMock:
    """Return a ParquetCatalog mock whose read_table distinguishes by columns."""
    mock = MagicMock()

    def _read_table(table, columns=None, seasons=None):
        if table == "pitches" and columns == PITCH_LEVEL_COLUMNS:
            return pitches_df[columns] if columns else pitches_df
        cols = columns or []
        return pd.DataFrame({c: [] for c in cols})

    mock.read_table.side_effect = _read_table
    return mock


@pytest.fixture()
def minimal_pitches_df():
    """Minimal pitches_raw DataFrame with all PITCH_LEVEL_COLUMNS."""
    row = {
        "game_pk": 1, "season": 2023, "game_date": "2023-04-01",
        "game_type_code": "R", "home_team_id": 147, "away_team_id": 111,
        "pitcher_id": 1001, "batter_id": 2001, "is_pitch": True,
        "release_speed": 93.0, "coord_x0": -1.5, "coord_z0": 5.8,
        "pitch_type": "FF", "bat_side_code": "R", "pitch_hand_code": "R",
        "event_type": "strikeout",
        "at_bat_index": 0, "pitch_number": 1, "inning": 1,
        "half_inning": "top", "cum_outs": 0,
        "pre_on_first_id": np.nan, "pre_on_second_id": np.nan, "pre_on_third_id": np.nan,
    }
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# 1. Structural tests
# ---------------------------------------------------------------------------

def test_pitches_raw_absent_from_table_config():
    """pitches_raw must not be in _TABLE_CONFIG — it was the OOM trigger."""
    assert "pitches_raw" not in _TABLE_CONFIG, (
        "pitches_raw is still in _TABLE_CONFIG. load_all() will load it "
        "alongside all other tables, exhausting 16 GB RAM."
    )


def test_load_pitches_raw_importable():
    """load_pitches_raw must be a public callable in data_loader."""
    from classical_learning.engineering import data_loader
    assert callable(getattr(data_loader, "load_pitches_raw", None))


def test_load_pitches_raw_signature():
    """load_pitches_raw must accept source, season_start, season_end."""
    import inspect
    sig = inspect.signature(load_pitches_raw)
    params = set(sig.parameters)
    assert {"source", "season_start", "season_end"} <= params


# ---------------------------------------------------------------------------
# 2. Behavioural tests — mock catalog, call real function code
# ---------------------------------------------------------------------------

def test_load_all_does_not_return_pitches_raw_key(minimal_pitches_df):
    """load_all() must never yield a 'pitches_raw' key."""
    catalog_mock = _make_catalog_mock(minimal_pitches_df, pd.DataFrame())

    with patch("pregame.engineering.data_loader.ParquetCatalog", return_value=catalog_mock), \
         patch("pregame.engineering.data_loader.season_range", return_value=[2023]):
        result = load_all("s3://fake/", season_start=2023)

    assert "pitches_raw" not in result
    assert "pitches" in result
    assert "boxscore_batting" in result


def test_load_pitches_raw_calls_catalog_with_correct_args(minimal_pitches_df):
    """load_pitches_raw() must call catalog.read_table('pitches', PITCH_LEVEL_COLUMNS, ...)."""
    catalog_mock = _make_catalog_mock(minimal_pitches_df, pd.DataFrame())

    with patch("pregame.engineering.data_loader.ParquetCatalog", return_value=catalog_mock), \
         patch("pregame.engineering.data_loader.season_range", return_value=[2023]):
        load_pitches_raw("s3://fake/", season_start=2023)

    calls = catalog_mock.read_table.call_args_list
    pitches_raw_calls = [
        c for c in calls
        if c.args[0] == "pitches" and c.kwargs.get("columns") == PITCH_LEVEL_COLUMNS
    ]
    assert len(pitches_raw_calls) == 1, (
        f"Expected exactly one read_table('pitches', PITCH_LEVEL_COLUMNS, ...) call, "
        f"got: {calls}"
    )


def test_load_pitches_raw_returns_dataframe(minimal_pitches_df):
    """load_pitches_raw() must return a DataFrame."""
    catalog_mock = _make_catalog_mock(minimal_pitches_df, pd.DataFrame())

    with patch("pregame.engineering.data_loader.ParquetCatalog", return_value=catalog_mock), \
         patch("pregame.engineering.data_loader.season_range", return_value=[2023]):
        result = load_pitches_raw("s3://fake/", season_start=2023)

    assert isinstance(result, pd.DataFrame)


def test_load_all_and_load_pitches_raw_use_separate_catalog_instances(minimal_pitches_df):
    """Each call must construct its own ParquetCatalog — they must not share a handle."""
    catalog_mock = _make_catalog_mock(minimal_pitches_df, pd.DataFrame())
    instantiation_count = []

    def counting_constructor(source):
        instantiation_count.append(source)
        return catalog_mock

    with patch("pregame.engineering.data_loader.ParquetCatalog", side_effect=counting_constructor), \
         patch("pregame.engineering.data_loader.season_range", return_value=[2023]):
        load_all("s3://fake/", season_start=2023)
        load_pitches_raw("s3://fake/", season_start=2023)

    assert len(instantiation_count) == 2, (
        "Expected 2 ParquetCatalog instantiations (one per call), "
        f"got {len(instantiation_count)}."
    )


# ---------------------------------------------------------------------------
# 3. build.py integration: raw freed before pitches_raw loaded
# ---------------------------------------------------------------------------

def test_build_py_frees_raw_before_pitches_raw(tmp_path, monkeypatch, minimal_pitches_df):
    """build.py must del raw before calling load_pitches_raw().

    We track object lifetime with weakref: by the time load_pitches_raw is
    called inside build_features(), the raw dict must have been freed
    (its weakref must be dead). If raw is still alive, both raw tables and
    pitches_raw coexist in memory — the OOM we're preventing.

    optuna is unavailable locally (EC2 only). We shim it in sys.modules
    so build.py can be imported; the shim is torn down after import.
    """
    # Shim optuna so build.py (which imports ratings_tuning → optuna) can load
    _optuna_missing = "optuna" not in sys.modules
    if _optuna_missing:
        optuna_stub = types.ModuleType("optuna")
        optuna_stub.logging = types.ModuleType("optuna.logging")
        optuna_stub.logging.set_verbosity = lambda *a, **kw: None
        optuna_stub.logging.WARNING = 30
        sys.modules["optuna"] = optuna_stub
        sys.modules["optuna.logging"] = optuna_stub.logging

    # Re-import build fresh (may have been cached before shim existed)
    for mod_name in list(sys.modules):
        if mod_name in ("pregame.engineering.build", "pregame.engineering.ratings_tuning"):
            del sys.modules[mod_name]

    import classical_learning.engineering.build as build_mod
    import classical_learning.engineering.game_builder as gb_mod
    import classical_learning.engineering.pitch_level_features as plf_mod
    import classical_learning.engineering.feature_engineering as fe_mod

    if _optuna_missing:
        sys.modules.pop("optuna", None)
        sys.modules.pop("optuna.logging", None)

    # ---- Tracking fixtures ----
    # weakref.ref cannot track plain dicts (no __weakref__ slot in CPython).
    # Wrap in a subclass that has one.
    class TrackableDict(dict):
        pass

    raw_sentinel = {}

    def fake_load_all(source, season_start, season_end=None):
        d = TrackableDict({
            "boxscore_batting": pd.DataFrame(),
            "boxscore_pitching": pd.DataFrame(),
            "linescore": pd.DataFrame(),
            "pitches": pd.DataFrame(),
            "players": pd.DataFrame(),
        })
        raw_sentinel["ref"] = weakref.ref(d)
        return d

    raw_alive_at_load_pitches_raw = []

    def fake_load_pitches_raw(source, season_start, season_end=None):
        gc.collect()
        raw_alive_at_load_pitches_raw.append(raw_sentinel["ref"]() is not None)
        return minimal_pitches_df

    def fake_build_game_frame(raw):
        return pd.DataFrame({
            "game_pk": [1], "game_date": ["2023-04-01"], "season": [2023],
            "game_type_code": ["R"], "home_team_id": [147], "away_team_id": [111],
            "probable_pitcher_home_id": [1001], "probable_pitcher_away_id": [2001],
        })

    # Patch all IO and heavy compute so the test runs without S3 or training.
    # Use monkeypatch so all module attributes are restored after the test —
    # direct assignment leaves stale lambdas that contaminate later tests.
    monkeypatch.setattr(build_mod, "load_all", fake_load_all)
    monkeypatch.setattr(build_mod, "load_pitches_raw", fake_load_pitches_raw)
    monkeypatch.setattr(build_mod, "attach_all_ratings", lambda games, params: games)
    monkeypatch.setattr(build_mod, "tune_all_ratings", lambda games, n_trials: {})
    monkeypatch.setattr(build_mod, "engineer_features", lambda games: games)
    monkeypatch.setattr(gb_mod, "build_game_frame", fake_build_game_frame)
    monkeypatch.setattr(plf_mod, "compute_pitch_level_features", lambda pitches_raw_df, game_frame: game_frame)
    monkeypatch.setattr(fe_mod, "_compute_pregame_pitcher_era", lambda games: games)

    build_mod.build_features(
        source="s3://fake/",
        output=tmp_path / "out",
        season_start=2023,
        tune_ratings=False,
        ratings_params={},
    )

    assert len(raw_alive_at_load_pitches_raw) == 1, (
        "load_pitches_raw was never called — build.py branch for pitch-level "
        "features was not reached."
    )
    assert not raw_alive_at_load_pitches_raw[0], (
        "raw dict was still alive when load_pitches_raw() was called. "
        "build.py must `del raw` before calling load_pitches_raw() to prevent "
        "both raw tables (~8–10 GB) and pitches_raw (~6 GB) coexisting in RAM."
    )
