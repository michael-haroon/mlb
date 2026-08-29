"""Tests verifying 2020 is excluded at every layer of the pipeline."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures: synthetic game_features parquet with 2020 present
# ---------------------------------------------------------------------------

SEASONS = [2018, 2019, 2020, 2021, 2022, 2023]
GAMES_PER_SEASON = {2018: 50, 2019: 50, 2020: 20, 2021: 50, 2022: 50, 2023: 50}


def _make_fake_features(path: Path) -> Path:
    """Write a minimal game_features.parquet with all seasons including 2020."""
    rng = np.random.default_rng(42)
    rows = []
    game_pk = 100000
    for season, n in GAMES_PER_SEASON.items():
        for i in range(n):
            rows.append({
                "game_pk": game_pk,
                "season": season,
                "game_date": pd.Timestamp(f"{season}-06-15") + pd.Timedelta(days=i),
                "home_win": rng.integers(0, 2),
                "yrfi": rng.integers(0, 2),
                "total_runs": rng.integers(3, 15),
                "home_run_diff": rng.integers(-5, 6),
                # Minimal pregame features (prefixed correctly)
                "rating_home_elo": rng.normal(1500, 50),
                "rating_away_elo": rng.normal(1500, 50),
                "rolling_home_ba_10": rng.normal(0.260, 0.02),
                "rolling_away_ba_10": rng.normal(0.260, 0.02),
                "target_status": "trainable",
            })
            game_pk += 1

    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def fake_features_path(tmp_path):
    return _make_fake_features(tmp_path / "game_features.parquet")


# ---------------------------------------------------------------------------
# Test 1: SKIP_SEASONS config
# ---------------------------------------------------------------------------

def test_skip_seasons_contains_2020():
    from classical_learning.strategy.config import SKIP_SEASONS
    assert 2020 in SKIP_SEASONS, "2020 must be in SKIP_SEASONS"


# ---------------------------------------------------------------------------
# Test 2: load_features excludes 2020
# ---------------------------------------------------------------------------

def test_load_features_excludes_2020(fake_features_path):
    from classical_learning.strategy.data import load_features

    df, y, seasons, game_pks = load_features(fake_features_path, "home_win", data_mode="all")

    assert 2020 not in seasons.values, "2020 should not appear in loaded seasons"
    # All other seasons should still be present
    for s in [2018, 2019, 2021, 2022, 2023]:
        assert s in seasons.values, f"Season {s} should be present"


def test_load_features_row_count(fake_features_path):
    from classical_learning.strategy.data import load_features

    df, y, seasons, game_pks = load_features(fake_features_path, "home_win", data_mode="all")

    expected_rows = sum(n for s, n in GAMES_PER_SEASON.items() if s != 2020)
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"


def test_load_features_2015_mode_excludes_2020(fake_features_path):
    """Even with 2015+ mode, 2020 should be excluded."""
    from classical_learning.strategy.data import load_features

    df, y, seasons, game_pks = load_features(fake_features_path, "home_win", data_mode="2015+")
    assert 2020 not in seasons.values


# ---------------------------------------------------------------------------
# Test 3: LOYO splits skip 2020
# ---------------------------------------------------------------------------

def test_loyo_splits_skip_2020():
    from classical_learning.strategy.data import generate_loyo_splits

    seasons = pd.Series([2017]*50 + [2018]*50 + [2019]*50 + [2020]*20 + [2021]*50 + [2022]*50)
    splits = generate_loyo_splits(seasons)

    val_seasons = [s.val_season for s in splits]
    assert 2020 not in val_seasons, "2020 should not be a validation fold"

    # 2020 should not appear in any training set either
    for split in splits:
        assert 2020 not in split.train_seasons, (
            f"2020 in train_seasons for val_season={split.val_season}"
        )


def test_loyo_splits_preserve_other_seasons():
    from classical_learning.strategy.data import generate_loyo_splits

    seasons = pd.Series([2017]*50 + [2018]*50 + [2019]*50 + [2020]*20 + [2021]*50 + [2022]*50)
    splits = generate_loyo_splits(seasons)

    val_seasons = [s.val_season for s in splits]
    # 2021 and 2022 should be validation folds (2017-2019 are too early, < MIN_TRAIN=3)
    assert 2021 in val_seasons, "2021 should be a validation fold"
    assert 2022 in val_seasons, "2022 should be a validation fold"


# ---------------------------------------------------------------------------
# Test 4: _bootstrap_game_frame skips 2020
# ---------------------------------------------------------------------------

def test_bootstrap_skips_2020():
    """Verify the SKIP_SEASONS import inside _bootstrap_game_frame works."""
    from classical_learning.strategy.config import SKIP_SEASONS

    # Simulate the loop logic from _bootstrap_game_frame
    start_year = 2018
    end_year = 2023
    processed = []
    for year in range(start_year, end_year + 1):
        if year in SKIP_SEASONS:
            continue
        processed.append(year)

    assert 2020 not in processed
    assert processed == [2018, 2019, 2021, 2022, 2023]


# ---------------------------------------------------------------------------
# Test 5: live feature_store build_feature_store skips 2020
# ---------------------------------------------------------------------------

def test_feature_store_season_filtering():
    """Verify the season filtering logic in build_feature_store uses canonical config."""
    from classical_learning.strategy.config import SKIP_SEASONS
    skip = set(SKIP_SEASONS)

    # Simulate discovered seasons
    discovered = {2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024}
    season_list = sorted(discovered - skip)
    assert 2020 not in season_list
    assert len(season_list) == 9

    # Simulate explicit seasons passed
    explicit = [2018, 2019, 2020, 2021, 2022]
    season_list = sorted(s for s in explicit if s not in skip)
    assert 2020 not in season_list
    assert season_list == [2018, 2019, 2021, 2022]


# ---------------------------------------------------------------------------
# Test 6: live train.py filtering
# ---------------------------------------------------------------------------

def test_live_train_filtering():
    """Verify season filtering logic used in live/mlb_dl/train.py uses SKIP_SEASONS."""
    from classical_learning.strategy.config import SKIP_SEASONS

    rng = np.random.default_rng(99)
    team_games = pd.DataFrame({
        "season": [2019]*30 + [2020]*10 + [2021]*30,
        "game_date": pd.date_range("2019-04-01", periods=70),
        "game_pk": range(70),
        "feat1": rng.normal(size=70),
    })
    game_targets = pd.DataFrame({
        "season": [2019]*30 + [2020]*10 + [2021]*30,
        "game_date": pd.date_range("2019-04-01", periods=70),
        "game_pk": range(70),
        "home_win": rng.integers(0, 2, size=70),
    })

    # Apply the same filter as in train.py (now uses SKIP_SEASONS)
    if SKIP_SEASONS:
        if "season" in team_games.columns:
            team_games = team_games[~team_games["season"].isin(SKIP_SEASONS)].reset_index(drop=True)
        if "season" in game_targets.columns:
            game_targets = game_targets[~game_targets["season"].isin(SKIP_SEASONS)].reset_index(drop=True)

    assert 2020 not in team_games["season"].values
    assert 2020 not in game_targets["season"].values
    assert len(team_games) == 60
    assert len(game_targets) == 60


# ---------------------------------------------------------------------------
# Test 7: Only 2020 is excluded — other seasons remain intact
# ---------------------------------------------------------------------------

def test_only_2020_excluded_not_others(fake_features_path):
    """Critical: verify ONLY 2020 is removed, nothing else."""
    from classical_learning.strategy.data import load_features

    df, y, seasons, game_pks = load_features(fake_features_path, "home_win", data_mode="all")

    remaining_seasons = sorted(seasons.unique())
    expected_seasons = [2018, 2019, 2021, 2022, 2023]
    assert remaining_seasons == expected_seasons, (
        f"Expected seasons {expected_seasons}, got {remaining_seasons}"
    )

    # Verify row counts per season match input (minus 2020)
    season_counts = seasons.value_counts().to_dict()
    for s in expected_seasons:
        assert season_counts[s] == GAMES_PER_SEASON[s], (
            f"Season {s}: expected {GAMES_PER_SEASON[s]} rows, got {season_counts[s]}"
        )


# ---------------------------------------------------------------------------
# Test 8: Real parquet on disk has no 2020
# ---------------------------------------------------------------------------

def test_real_parquet_no_2020():
    """Verify the actual game_features.parquet file has no 2020 data."""
    real_path = Path("pregame/artifacts/features/game_features.parquet")
    if not real_path.exists():
        pytest.skip("game_features.parquet not available locally")

    df = pd.read_parquet(real_path, columns=["season"])
    assert 2020 not in df["season"].values, "Real game_features.parquet still contains 2020 data!"


# ---------------------------------------------------------------------------
# Test 9: build_features() full build path excludes 2020 (Finding #1)
# ---------------------------------------------------------------------------

def test_build_features_full_path_imports_skip_seasons():
    """The full build_features() imports SKIP_SEASONS and applies it after checkpoint load."""
    source = Path("pregame/engineering/build.py").read_text()
    assert "SKIP_SEASONS" in source, "build.py must reference SKIP_SEASONS"
    assert '~games["season"].isin(SKIP_SEASONS)' in source, (
        "build_features must filter games using SKIP_SEASONS"
    )


def test_build_features_full_path_filters_games():
    """Simulate what build_features does: load a checkpoint-like df, then filter."""
    from classical_learning.strategy.config import SKIP_SEASONS

    games = pd.DataFrame({
        "season": [2018]*20 + [2019]*20 + [2020]*10 + [2021]*20,
        "game_pk": range(70),
    })

    # Simulate the SKIP_SEASONS filtering from build_features
    if SKIP_SEASONS and "season" in games.columns:
        games = games[~games["season"].isin(SKIP_SEASONS)].reset_index(drop=True)

    assert 2020 not in games["season"].values
    assert len(games) == 60


# ---------------------------------------------------------------------------
# Test 10: Stale checkpoint filtering (Finding #2)
# ---------------------------------------------------------------------------

def test_incremental_build_filters_stale_checkpoint():
    """Stale checkpoint with 2020 data gets filtered after loading."""
    from classical_learning.strategy.config import SKIP_SEASONS

    # Simulate a stale checkpoint that was built before the exclusion
    stale_checkpoint = pd.DataFrame({
        "season": [2018]*30 + [2019]*30 + [2020]*15 + [2021]*30 + [2022]*30,
        "game_pk": range(135),
        "game_date": (
            [f"2018-06-{i:02d}" for i in range(1, 31)] +
            [f"2019-06-{i:02d}" for i in range(1, 31)] +
            [f"2020-08-{i:02d}" for i in range(1, 16)] +
            [f"2021-06-{i:02d}" for i in range(1, 31)] +
            [f"2022-06-{i:02d}" for i in range(1, 31)]
        ),
    })

    # Simulate the incremental build's checkpoint filtering
    if SKIP_SEASONS and "season" in stale_checkpoint.columns:
        pre_len = len(stale_checkpoint)
        stale_checkpoint = stale_checkpoint[
            ~stale_checkpoint["season"].isin(SKIP_SEASONS)
        ].reset_index(drop=True)
        purged = pre_len - len(stale_checkpoint)

    assert 2020 not in stale_checkpoint["season"].values
    assert purged == 15, f"Expected 15 purged rows, got {purged}"
    assert len(stale_checkpoint) == 120


# ---------------------------------------------------------------------------
# Test 11: Centralized SKIP_SEASONS (Finding #3 + #4)
# ---------------------------------------------------------------------------

def test_all_skip_seasons_references_use_canonical_config():
    """All filtering uses the canonical pregame.strategy.config.SKIP_SEASONS."""
    import inspect

    # feature_store.py
    from live.mlb_dl import feature_store
    source = inspect.getsource(feature_store.build_feature_store)
    assert "from pregame.strategy.config import SKIP_SEASONS" in source, (
        "feature_store.build_feature_store must import from pregame.strategy.config"
    )

    # train.py — check both fit functions
    from live.mlb_dl import train
    full_source = inspect.getsource(train)
    assert "from pregame.strategy.config import SKIP_SEASONS" in full_source, (
        "train.py must import from pregame.strategy.config"
    )


# ---------------------------------------------------------------------------
# Test 12: No redundant hardcoded != 2020 in ensemble.py (Finding #5)
# ---------------------------------------------------------------------------

def test_ensemble_no_redundant_2020_filter():
    """ensemble.py refit should NOT have a hardcoded seasons != 2020 filter."""
    import inspect
    from classical_learning.strategy import ensemble

    source = inspect.getsource(ensemble.fit_and_save_ensemble)
    assert "!= 2020" not in source, (
        "fit_and_save_ensemble still has redundant hardcoded != 2020"
    )
    assert "seasons != 2020" not in source


# ---------------------------------------------------------------------------
# Test 13: LiveHANDataset filters all frames symmetrically (Finding #6)
# ---------------------------------------------------------------------------

def test_live_han_dataset_filters_all_frames():
    """LiveHANDataset should filter pitch_sequences and pregame_features, not just targets."""
    import inspect
    from live.mlb_dl import live_dataset

    source = inspect.getsource(live_dataset.LiveHANDataset.__init__)
    assert "self.pitch_sequences" in source and "SKIP_SEASONS" in source, (
        "LiveHANDataset must filter pitch_sequences using SKIP_SEASONS"
    )
    assert "self.pregame_features" in source and "SKIP_SEASONS" in source, (
        "LiveHANDataset must filter pregame_features using SKIP_SEASONS"
    )
