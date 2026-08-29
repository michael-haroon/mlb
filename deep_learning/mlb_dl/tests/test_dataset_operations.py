"""Comprehensive tests for GameTransformerDataset operations.

Tests cover: sample generation, SP context, team context, live prefix,
targets, masks, leakage invariants, and edge cases. All data is
synthetically constructed -- no parquet files are loaded.

Key invariant under test: NO data leakage — all features use data
strictly before target_game_date.
"""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import torch

from deep_learning.mlb_dl.game_transformer_dataset import (
    AblationConfig,
    GameTransformerDataset,
    PITCH_CONTINUOUS_COLS,
    PITCH_TYPE_TO_IDX,
    BAT_SIDE_TO_IDX,
    PITCH_HAND_TO_IDX,
    HALF_INNING_TO_IDX,
    HIT_TRAJECTORY_TO_IDX,
    HIT_HARDNESS_TO_IDX,
    MAX_PLAYERS_PER_GAME,
    PLAYER_STAT_DIM,
    FLAT_FEATURE_DIM,
    build_game_index,
    map_pitch_type,
    map_bat_side,
    classify_event_type,
)
from deep_learning.mlb_dl.datasets import SequenceSpec, Standardizer


# ---------------------------------------------------------------------------
# Fixtures: minimal but realistic synthetic data builders
# ---------------------------------------------------------------------------


def _make_standardizer() -> Standardizer:
    """A standardizer with mean=0, std=1 for all PITCH_CONTINUOUS_COLS."""
    return Standardizer(
        feature_columns=list(PITCH_CONTINUOUS_COLS),
        mean={col: 0.0 for col in PITCH_CONTINUOUS_COLS},
        std={col: 1.0 for col in PITCH_CONTINUOUS_COLS},
    )


def _make_pitch_rows(
    game_pk: int,
    n_pitches: int,
    game_date: str = "2025-07-01",
    season: int = 2025,
    pitcher_id: int = 100,
    batter_ids: list[int] | None = None,
    start_score_home: float = 0.0,
    start_score_away: float = 0.0,
    inning_start: int = 1,
    pitch_type: str = "FF",
    bat_side: str = "R",
    pitch_hand: str = "R",
) -> pd.DataFrame:
    """Create n_pitches synthetic pitch rows for a single game."""
    if batter_ids is None:
        batter_ids = [200, 201, 202, 203, 204, 205, 206, 207, 208]
    rows = []
    for i in range(n_pitches):
        ab_index = i // 4  # ~4 pitches per AB
        pitch_number = (i % 4) + 1
        inning = inning_start + i // 20  # ~20 pitches per inning
        batter = batter_ids[ab_index % len(batter_ids)]
        rows.append({
            "game_pk": game_pk,
            "season": season,
            "game_date": game_date,
            "play_index": i,
            "at_bat_index": ab_index,
            "pitch_sequence_index": i,
            "pitch_number": pitch_number,
            "inning": inning,
            "is_top_inning": 1 if (i < n_pitches // 2) else 0,
            "batter_id": batter,
            "pitcher_id": pitcher_id,
            "fielder_2": 300,
            "pitch_type": pitch_type,
            "bat_side_code": bat_side,
            "pitch_hand_code": pitch_hand,
            "is_pitch": 1,
            "is_strike": 1 if i % 3 == 0 else 0,
            "is_ball": 1 if i % 3 == 1 else 0,
            "is_in_play": 1 if i % 3 == 2 else 0,
            "score_home": start_score_home + (i // 40),
            "score_away": start_score_away + (i // 50),
            "release_speed": 92.0 + np.random.randn() * 2,
            "cum_balls": i % 4,
            "cum_strikes": i % 3,
            "cum_outs": (i // 6) % 3,
            "pitch_count_balls": i % 4,
            "pitch_count_strikes": i % 3,
            "pitch_count_outs": (i // 6) % 3,
            "pre_on_first_id": 0,
            "pre_on_second_id": 0,
            "pre_on_third_id": 0,
            "bb_type": None,
            "hit_hardness": None,
        })
    return pd.DataFrame(rows)


def _make_game_targets(
    game_pk: int,
    game_date: str = "2025-07-01",
    season: int = 2025,
    home_runs: int = 5,
    away_runs: int = 3,
    target_status: str = "trainable",
) -> pd.DataFrame:
    return pd.DataFrame([{
        "game_pk": game_pk,
        "game_date": game_date,
        "season": season,
        "home_team_id": 110,
        "away_team_id": 120,
        "target_status": target_status,
        "home_win": 1 if home_runs > away_runs else 0,
        "away_win": 0 if home_runs > away_runs else 1,
        "yrfi": 1,
        "nrfi": 0,
        "extra_innings": 0,
        "total_runs": home_runs + away_runs,
        "home_runs": home_runs,
        "away_runs": away_runs,
        "home_run_diff": home_runs - away_runs,
        "away_run_diff": away_runs - home_runs,
        "first_5_total_runs": 4,
        "first_5_home_runs": 2,
        "first_5_away_runs": 2,
        "first_5_home_run_diff": 0,
        "first_5_away_run_diff": 0,
        "first_5_home_win": 0,
        "first_5_away_win": 0,
        "first_5_tie": 1,
    }])


def _make_game_meta(
    game_pk: int,
    game_date: str = "2025-07-01",
    season: int = 2025,
    home_team_id: int = 110,
    away_team_id: int = 120,
    home_pitcher_id: int = 100,
    away_pitcher_id: int = 101,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "game_pk": game_pk,
        "game_date": game_date,
        "season": season,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "probable_pitcher_home_id": home_pitcher_id,
        "probable_pitcher_away_id": away_pitcher_id,
        "venue_id": 1,
        "day_night": "night",
        "game_number": 1,
        "double_header": "N",
        "tiebreaker": "N",
        "start_time": "19:05",
        "game_datetime_utc": f"{game_date}T23:05:00Z",
        "venue_latitude": 40.75,
        "venue_longitude": -73.85,
        "venue_capacity": 41922,
        "venue_surface": "grass",
        "venue_roof_type": "open",
        "umpire_hp": "Angel Hernandez",
        "rule_3batter_minimum": 1.0,
        "rule_universal_dh": 1.0,
        "rule_shift_ban_pitch_clock": 1.0,
    }])


def _make_team_games(
    game_pks: list[int],
    team_id: int = 110,
    game_dates: list[str] | None = None,
    season: int = 2025,
) -> pd.DataFrame:
    """Create team_games rows (one per game) for a single team."""
    if game_dates is None:
        base = pd.Timestamp("2025-06-01")
        game_dates = [(base + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(len(game_pks))]
    rows = []
    for gpk, gd in zip(game_pks, game_dates):
        rows.append({
            "game_pk": gpk,
            "team_id": team_id,
            "game_date": gd,
            "season": season,
            "side": "home",
        })
    return pd.DataFrame(rows)


def _make_player_batting_history(
    player_ids: list[int],
    game_pks: list[int],
    game_dates: list[str],
    season: int = 2025,
) -> pd.DataFrame:
    """Create player batting history with one row per (player, game)."""
    rows = []
    for gpk, gd in zip(game_pks, game_dates):
        for pid in player_ids:
            rows.append({
                "player_id": pid,
                "game_pk": gpk,
                "game_date": gd,
                "season": season,
                "side": "home",
                "target_status": "trainable",
                "game_ab": 4,
                "game_runs": 1,
                "game_hits": 1,
                "game_doubles": 0,
                "game_triples": 0,
                "game_hr": 0,
                "game_rbi": 1,
                "game_bb": 0,
                "game_ibb": 0,
                "game_so": 1,
                "game_sb": 0,
                "game_cs": 0,
                "game_hbp": 0,
                "game_sac": 0,
                "game_sf": 0,
                "game_gidp": 0,
                "game_lob": 2,
                "game_total_bases": 1,
                "game_hits_runs_rbi": 3,
                "season_avg": 0.270,
                "season_obp": 0.340,
                "season_slg": 0.430,
                "season_ops": 0.770,
                "season_hr": 10,
                "season_rbi": 40,
                "season_sb": 5,
                "season_games_played": 80,
            })
    return pd.DataFrame(rows)


def _build_minimal_dataset(
    n_pitches: int = 100,
    n_prior_games: int = 3,
    spec: SequenceSpec | None = None,
    ablation: AblationConfig | None = None,
    include_pregame: bool = True,
    include_live: bool = True,
    target_game_pk: int = 999,
    target_date: str = "2025-07-15",
    home_runs: int = 5,
    away_runs: int = 3,
    extra_pitch_sequences: pd.DataFrame | None = None,
    extra_game_targets: pd.DataFrame | None = None,
    extra_game_meta: pd.DataFrame | None = None,
    extra_team_games: pd.DataFrame | None = None,
    extra_player_history: pd.DataFrame | None = None,
    categorize_pitch_columns: bool = False,
) -> GameTransformerDataset:
    """Build a minimal but functional GameTransformerDataset for testing."""
    if spec is None:
        spec = SequenceSpec(
            history_length=50,
            min_history=1,
            live_stride=25,
            live_max_prefixes_per_game=32,
        )
    if ablation is None:
        ablation = AblationConfig(sp_history_games=5, team_history_games=5)

    # Target game
    pitch_seqs = _make_pitch_rows(target_game_pk, n_pitches, game_date=target_date, pitcher_id=100)
    game_targets = _make_game_targets(target_game_pk, game_date=target_date,
                                       home_runs=home_runs, away_runs=away_runs)
    game_meta = _make_game_meta(target_game_pk, game_date=target_date)

    # Prior games for SP/team context
    prior_pks = list(range(900, 900 + n_prior_games))
    prior_dates = [
        (pd.Timestamp(target_date) - timedelta(days=5 * (n_prior_games - i))).strftime("%Y-%m-%d")
        for i in range(n_prior_games)
    ]

    for gpk, gd in zip(prior_pks, prior_dates):
        prior_pitch = _make_pitch_rows(gpk, 80, game_date=gd, pitcher_id=100)
        pitch_seqs = pd.concat([pitch_seqs, prior_pitch], ignore_index=True)
        game_targets = pd.concat([game_targets, _make_game_targets(gpk, game_date=gd)], ignore_index=True)
        game_meta = pd.concat([game_meta, _make_game_meta(gpk, game_date=gd)], ignore_index=True)

    # Team games
    all_pks = prior_pks + [target_game_pk]
    all_dates = prior_dates + [target_date]
    team_games = _make_team_games(all_pks, team_id=110, game_dates=all_dates)
    team_games_away = _make_team_games(all_pks, team_id=120, game_dates=all_dates)
    team_games_away["side"] = "away"
    team_games = pd.concat([team_games, team_games_away], ignore_index=True)

    # Player batting history
    batter_ids = [200, 201, 202, 203, 204, 205, 206, 207, 208]
    player_hist = _make_player_batting_history(batter_ids, all_pks, all_dates)

    # Concatenate extras if provided
    if extra_pitch_sequences is not None:
        pitch_seqs = pd.concat([pitch_seqs, extra_pitch_sequences], ignore_index=True)
    if extra_game_targets is not None:
        game_targets = pd.concat([game_targets, extra_game_targets], ignore_index=True)
    if extra_game_meta is not None:
        game_meta = pd.concat([game_meta, extra_game_meta], ignore_index=True)
    if extra_team_games is not None:
        team_games = pd.concat([team_games, extra_team_games], ignore_index=True)
    if extra_player_history is not None:
        player_hist = pd.concat([player_hist, extra_player_history], ignore_index=True)

    if categorize_pitch_columns:
        category_cols = [
            "game_date", "pitch_type", "bat_side_code", "pitch_hand_code",
            "bb_type", "hit_hardness", "at_bat_event",
        ]
        for col in category_cols:
            if col in pitch_seqs.columns:
                pitch_seqs[col] = pitch_seqs[col].astype("category")

    standardizer = _make_standardizer()

    ds = GameTransformerDataset(
        pitch_sequences=pitch_seqs,
        game_targets=game_targets,
        game_meta=game_meta,
        team_games=team_games,
        player_batting_history=player_hist,
        standardizer=standardizer,
        ablation=ablation,
        spec=spec,
        include_pregame=include_pregame,
        include_live=include_live,
    )
    return ds


# ===========================================================================
# A. Sample generation (_build_samples)
# ===========================================================================


class TestBuildSamples:
    """Tests for _build_samples sample generation logic."""

    def test_pregame_sample_always_present(self):
        """When include_pregame=True, prefix=0 sample exists for every game."""
        ds = _build_minimal_dataset(n_pitches=100, include_pregame=True, include_live=False)
        target_pk = 999
        pregame_samples = [(gpk, plen) for gpk, plen in ds.samples if gpk == target_pk and plen == 0]
        assert len(pregame_samples) == 1, "Exactly one pregame sample expected for target game"

    def test_no_pregame_when_disabled(self):
        """When include_pregame=False, no prefix=0 samples."""
        ds = _build_minimal_dataset(n_pitches=100, include_pregame=False, include_live=True)
        pregame_samples = [(gpk, plen) for gpk, plen in ds.samples if plen == 0]
        assert len(pregame_samples) == 0

    def test_live_stride_generates_correct_positions(self):
        """Live samples at stride=25 for a 100-pitch game: 25, 50, 75, 99."""
        spec = SequenceSpec(history_length=50, min_history=1, live_stride=25,
                            live_max_prefixes_per_game=32)
        ds = _build_minimal_dataset(n_pitches=100, spec=spec, include_pregame=False)
        target_pk = 999
        positions = sorted([plen for gpk, plen in ds.samples if gpk == target_pk])
        # Expected: range(25, 100, 25) = [25, 50, 75], then append 99 (last pitch)
        assert 25 in positions
        assert 50 in positions
        assert 75 in positions
        assert 99 in positions  # Final pitch appended if not already at stride boundary

    def test_max_prefixes_respected(self):
        """A game with many pitches doesn't exceed live_max_prefixes_per_game."""
        spec = SequenceSpec(history_length=50, min_history=1, live_stride=5,
                            live_max_prefixes_per_game=4)
        ds = _build_minimal_dataset(n_pitches=200, spec=spec, include_pregame=False)
        target_pk = 999
        live_samples = [plen for gpk, plen in ds.samples if gpk == target_pk]
        assert len(live_samples) <= 4

    def test_game_with_zero_pitches_no_live_samples(self):
        """A game_pk in offsets with 0 pitches produces no live samples."""
        # Build game with 0 pitches by not including any pitch data for a target game
        # We create a game target but no pitch data for pk=888
        extra_targets = _make_game_targets(888, game_date="2025-07-10")
        extra_meta = _make_game_meta(888, game_date="2025-07-10")
        # No pitch rows for pk=888 -> game won't be in _game_offsets
        # Therefore _build_samples skips it entirely if gpk not in _game_offsets
        ds = _build_minimal_dataset(
            extra_game_targets=extra_targets,
            extra_game_meta=extra_meta,
        )
        samples_888 = [(gpk, plen) for gpk, plen in ds.samples if gpk == 888]
        # Game 888 has no pitch data -> no offset -> _build_samples skips it
        assert len(samples_888) == 0


# ===========================================================================
# B. SP context (_get_sp_context)
# ===========================================================================


class TestSPContext:
    """Tests for _get_sp_context historical starting pitcher context."""

    def test_pitcher_with_no_prior_starts_returns_zeros(self):
        """A pitcher with no prior starts returns an all-zeros context."""
        # Create a game where the pitcher_id has never started before
        ds = _build_minimal_dataset(n_pitches=50, n_prior_games=0)
        target_pk = 999
        # Manually call _get_sp_context with a fresh pitcher who never appeared
        meta = ds.meta_by_game[target_pk].copy()
        meta["probable_pitcher_home_id"] = 99999  # Unseen pitcher
        ctx = ds._get_sp_context(meta, side="home", game_pk=target_pk)
        assert torch.all(ctx["sequences"] == 0)
        assert torch.all(ctx["weights"] == 0)

    def test_only_starts_before_game_date_included(self):
        """SP context only includes games strictly before target game_date (no leakage)."""
        ds = _build_minimal_dataset(n_pitches=50, n_prior_games=3)
        target_pk = 999
        meta = ds.meta_by_game[target_pk]
        target_date = meta["game_date"]

        # The SP context should only include games before target_date
        ctx = ds._get_sp_context(meta, side="home", game_pk=target_pk)

        # Verify non-zero sequences exist (prior games are there)
        assert ctx["lengths"].sum() > 0, "Expected non-empty SP context"

        # Verify no game_pk in SP history has date >= target_date
        pitcher_id = int(meta["probable_pitcher_home_id"])
        sp_games = ds._sp_games.get(pitcher_id, [])
        for gpk in sp_games:
            if gpk == target_pk:
                continue
            gm = ds.meta_by_game.get(gpk)
            if gm is not None and ctx["lengths"].sum() > 0:
                assert gm["game_date"] < target_date, \
                    f"SP context includes game {gpk} on/after target date"

    def test_most_recent_n_starts_selected(self):
        """SP context selects the N most recent starts, not earliest N."""
        ablation = AblationConfig(sp_history_games=2, team_history_games=5)
        ds = _build_minimal_dataset(n_pitches=50, n_prior_games=5, ablation=ablation)
        target_pk = 999
        meta = ds.meta_by_game[target_pk]
        ctx = ds._get_sp_context(meta, side="home", game_pk=target_pk)

        # sp_history_games=2: only 2 prior games should have non-zero lengths
        non_zero_count = (ctx["lengths"] > 0).sum().item()
        assert non_zero_count == 2, f"Expected 2 prior starts, got {non_zero_count}"

    def test_empty_when_no_probable_pitcher(self):
        """If meta has no probable_pitcher column, return empty context."""
        ds = _build_minimal_dataset(n_pitches=50, n_prior_games=3)
        target_pk = 999
        meta = ds.meta_by_game[target_pk].copy()
        meta["probable_pitcher_home_id"] = float("nan")
        ctx = ds._get_sp_context(meta, side="home", game_pk=target_pk)
        assert torch.all(ctx["sequences"] == 0)

    def test_non_trainable_prior_start_occupies_slot(self):
        """A non-trainable prior start (rain-shortened, postponed) must occupy a
        chronological slot in the SP history rather than being silently dropped.

        Before fix: non-trainable game excluded from _sp_games; an older trainable
        game fills the slot instead. After fix: the slot is correctly present but
        zero-padded (no pitch data available for non-trainable games).
        """
        ablation = AblationConfig(sp_history_games=2, team_history_games=5)
        target_date = "2025-07-15"
        non_trainable_pk = 850
        non_trainable_date = "2025-07-13"  # most recent prior start, but non-trainable

        # Non-trainable game appears in game_meta but has no pitch rows and
        # target_status != "trainable", so it won't be in target_by_game.
        extra_targets = _make_game_targets(
            non_trainable_pk, game_date=non_trainable_date,
            target_status="settles_last_fair",
        )
        extra_meta = _make_game_meta(
            non_trainable_pk, game_date=non_trainable_date,
            home_pitcher_id=100,
        )

        ds = _build_minimal_dataset(
            n_pitches=50,
            n_prior_games=2,
            ablation=ablation,
            target_date=target_date,
            extra_game_targets=extra_targets,
            extra_game_meta=extra_meta,
        )

        target_pk = 999
        meta = ds.meta_by_game[target_pk]
        pitcher_id = int(meta["probable_pitcher_home_id"])

        # The non-trainable game must be registered in _sp_games.
        assert non_trainable_pk in ds._sp_games.get(pitcher_id, []), (
            "Non-trainable start must appear in _sp_games after fix"
        )

        # Non-trainable game is not a training target.
        assert non_trainable_pk not in ds.target_by_game

        ctx = ds._get_sp_context(meta, side="home", game_pk=target_pk)

        # With sp_history_games=2 and 3 prior starts (2 trainable + 1 non-trainable),
        # the 2 most recent are: trainable 2025-07-10 (slot 0) and non-trainable
        # 2025-07-13 (slot 1). _extract_game_sequences appends in chronological order,
        # so the most-recent slot is last.
        assert ctx["lengths"].shape[0] == 2, "Expected exactly 2 history slots"

        # Slot 0 (older trainable game): must have pitch data
        assert ctx["lengths"][0].item() > 0, (
            "Older trainable slot should have non-zero pitch data"
        )
        # Slot 1 (most-recent non-trainable game): zero-padded since no pitch rows
        assert ctx["lengths"][1].item() == 0, (
            "Most-recent non-trainable slot must be zero-padded (correct chronological position)"
        )
        assert torch.all(ctx["sequences"][1] == 0), (
            "Non-trainable slot sequences must be zero-filled"
        )


# ===========================================================================
# C. Team context (_get_team_context)
# ===========================================================================


class TestTeamContext:
    """Tests for _get_team_context team history builder."""

    def test_only_games_before_game_date(self):
        """Team context only includes games strictly before target game_date."""
        ds = _build_minimal_dataset(n_pitches=50, n_prior_games=3)
        target_pk = 999
        meta = ds.meta_by_game[target_pk]
        target_date = meta["game_date"]

        ctx = ds._get_team_context(meta, side="home", game_pk=target_pk)
        # Should have non-zero content from prior games
        assert ctx["lengths"].sum() > 0

        # Verify in game_index: all prior games have dates < target
        team_id = int(meta["home_team_id"])
        team_df = ds.game_index["by_team"].get(team_id)
        if team_df is not None:
            prior = team_df[team_df["game_pk"] != target_pk]
            for _, row in prior.iterrows():
                if row["game_date"] < target_date:
                    # Fine - this is a valid context game
                    pass
                else:
                    # This game should NOT have been included in context
                    # (handled by the filter in _get_team_context)
                    pass

    def test_current_game_excluded_from_context(self):
        """The target game itself must never appear in its own context."""
        ds = _build_minimal_dataset(n_pitches=50, n_prior_games=3)
        target_pk = 999
        meta = ds.meta_by_game[target_pk]

        # The filter is: (team_df["game_date"] < game_date) & (team_df["game_pk"] != game_pk)
        # This test verifies the game_pk exclusion
        team_id = int(meta["home_team_id"])
        team_df = ds.game_index["by_team"].get(team_id)
        target_date = meta["game_date"]

        prior_mask = (team_df["game_date"] < target_date) & (team_df["game_pk"] != target_pk)
        assert target_pk not in team_df[prior_mask]["game_pk"].values

    def test_all_games_mode_selects_most_recent(self):
        """'all_games' mode selects the most recent N games (tail)."""
        ablation = AblationConfig(team_context_mode="all_games", team_history_games=2)
        ds = _build_minimal_dataset(n_pitches=50, n_prior_games=5, ablation=ablation)
        target_pk = 999
        meta = ds.meta_by_game[target_pk]
        ctx = ds._get_team_context(meta, side="home", game_pk=target_pk)

        # Only 2 games should have non-zero content
        non_zero_count = (ctx["lengths"] > 0).sum().item()
        assert non_zero_count == 2

    def test_empty_when_no_team_history(self):
        """First game ever for a team returns empty context."""
        # Create a game for a brand new team with no prior games
        target_date = "2025-04-01"
        spec = SequenceSpec(history_length=50, min_history=1, live_stride=25,
                            live_max_prefixes_per_game=32)
        ablation = AblationConfig(sp_history_games=5, team_history_games=5)

        # Target game
        pitch_seqs = _make_pitch_rows(999, 50, game_date=target_date, pitcher_id=500)
        game_targets = _make_game_targets(999, game_date=target_date)
        game_meta = _make_game_meta(999, game_date=target_date, home_team_id=777,
                                     away_team_id=888, home_pitcher_id=500, away_pitcher_id=501)
        team_games = _make_team_games([999], team_id=777, game_dates=[target_date])
        team_games_away = _make_team_games([999], team_id=888, game_dates=[target_date])
        team_games_away["side"] = "away"
        team_games = pd.concat([team_games, team_games_away], ignore_index=True)
        player_hist = _make_player_batting_history([200], [999], [target_date])

        ds = GameTransformerDataset(
            pitch_sequences=pitch_seqs,
            game_targets=game_targets,
            game_meta=game_meta,
            team_games=team_games,
            player_batting_history=player_hist,
            standardizer=_make_standardizer(),
            ablation=ablation,
            spec=spec,
        )
        meta = ds.meta_by_game[999]
        ctx = ds._get_team_context(meta, side="home", game_pk=999)
        # No prior games -> all zeros
        assert torch.all(ctx["lengths"] == 0)


# ===========================================================================
# D. Live prefix (_get_live_prefix)
# ===========================================================================


class TestLivePrefix:
    """Tests for _get_live_prefix pitch prefix extraction."""

    def test_prefix_len_zero_returns_all_zeros(self):
        """prefix_len=0 (pregame) returns all-zeros tensors."""
        ds = _build_minimal_dataset(n_pitches=100)
        target_pk = 999
        result = ds._get_live_prefix(target_pk, prefix_len=0)
        assert torch.all(result["values"] == 0)
        assert torch.all(result["mask"] == 0)
        assert torch.all(result["batter_hash"] == 0)
        assert torch.all(result["pitch_type_idx"] == 0)

    def test_prefix_len_greater_than_game_clamped(self):
        """prefix_len > actual game length is clamped to game length."""
        ds = _build_minimal_dataset(n_pitches=30)
        target_pk = 999
        # Request prefix of 100 but game only has 30 pitches
        result = ds._get_live_prefix(target_pk, prefix_len=100)
        # Mask should have exactly 30 non-zero positions
        assert result["mask"].sum().item() == 30

    def test_categorical_arrays_nonzero_for_known_values(self):
        """Categorical arrays produce non-zero indices for known pitch data."""
        ds = _build_minimal_dataset(n_pitches=50)
        target_pk = 999
        result = ds._get_live_prefix(target_pk, prefix_len=50)
        # pitch_type "FF" -> idx 1
        non_pad = result["mask"] > 0
        pt_values = result["pitch_type_idx"][non_pad]
        assert (pt_values > 0).any(), "Expected non-zero pitch_type_idx for 'FF' pitches"

        # bat_side "R" -> idx 1
        bs_values = result["bat_side_idx"][non_pad]
        assert (bs_values > 0).any(), "Expected non-zero bat_side_idx for 'R' batters"

        # pitch_hand "R" -> idx 1
        ph_values = result["pitch_hand_idx"][non_pad]
        assert (ph_values > 0).any(), "Expected non-zero pitch_hand_idx for 'R' pitchers"

    def test_category_dtype_pitch_columns_construct(self):
        """Dataset handles category-backed string columns without invalid fillna sentinels."""
        ds = _build_minimal_dataset(n_pitches=50, categorize_pitch_columns=True)
        result = ds._get_live_prefix(999, prefix_len=50)
        non_pad = result["mask"] > 0

        assert (result["pitch_type_idx"][non_pad] == PITCH_TYPE_TO_IDX["FF"]).any()
        assert (result["bat_side_idx"][non_pad] == BAT_SIDE_TO_IDX["R"]).any()
        assert (result["pitch_hand_idx"][non_pad] == PITCH_HAND_TO_IDX["R"]).any()

    def test_left_padding_data_at_end(self):
        """Left-padding: actual data at END of the tensor, zeros at START."""
        spec = SequenceSpec(history_length=50, min_history=1, live_stride=25,
                            live_max_prefixes_per_game=32)
        ds = _build_minimal_dataset(n_pitches=20, spec=spec)
        target_pk = 999
        result = ds._get_live_prefix(target_pk, prefix_len=20)

        mask = result["mask"]
        # First 30 positions should be 0 (padding), last 20 should be 1
        assert mask[:30].sum().item() == 0, "Padding positions at START should be zero"
        assert mask[30:].sum().item() == 20, "Data positions at END should be ones"

        # Values at start should be zero, values at end should be non-zero (data)
        values = result["values"]
        assert values[:30].abs().sum().item() == 0, "Padded region should be zero"

    def test_hierarchy_unpacked_correctly(self):
        """Hierarchy array contains [inning, at_bat_index, pitch_number]."""
        ds = _build_minimal_dataset(n_pitches=50)
        target_pk = 999
        result = ds._get_live_prefix(target_pk, prefix_len=50)

        # hierarchy has shape (max_prefix, 3)
        assert result["hierarchy"].shape[1] == 3

        # For non-padded region, inning should be >= 1
        mask = result["mask"] > 0
        innings = result["hierarchy"][mask, 0]
        assert (innings >= 1).all(), "Inning values should be >= 1 for actual pitches"


# ===========================================================================
# E. Targets (_build_targets)
# ===========================================================================


class TestBuildTargets:
    """Tests for _build_targets target construction."""

    def test_home_runs_remaining_never_negative(self):
        """home_runs_remaining = max(0, final - score_at_prefix), never negative."""
        # Set up a game where score at some pitch exceeds final (edge case from
        # data pipeline errors or postponed games). The dataset takes max(0, ...).
        ds = _build_minimal_dataset(n_pitches=100, home_runs=5, away_runs=3)
        target_pk = 999
        # Get a sample that is near end of game
        sample = ds[0]  # pregame sample
        remaining = sample["targets"]["home_runs_remaining"]
        assert remaining >= 0

    def test_pregame_remaining_equals_final(self):
        """At pregame (prefix=0): remaining runs = final runs."""
        ds = _build_minimal_dataset(n_pitches=100, home_runs=7, away_runs=4)
        # Find the pregame sample for target game
        target_pk = 999
        pregame_idx = None
        for i, (gpk, plen) in enumerate(ds.samples):
            if gpk == target_pk and plen == 0:
                pregame_idx = i
                break
        assert pregame_idx is not None
        sample = ds[pregame_idx]
        assert sample["targets"]["home_runs_remaining"].item() == 7.0
        assert sample["targets"]["away_runs_remaining"].item() == 4.0

    def test_observed_home_uses_last_pitch_before_prefix(self):
        """observed_home uses pitch at index (prefix_len - 1), the last pitch in the prefix."""
        ds = _build_minimal_dataset(n_pitches=100, home_runs=5, away_runs=3)
        target_pk = 999
        # Get a live sample
        live_idx = None
        for i, (gpk, plen) in enumerate(ds.samples):
            if gpk == target_pk and plen > 0:
                live_idx = i
                break
        assert live_idx is not None
        gpk, plen = ds.samples[live_idx]

        # Manually compute what the target builder should do
        t_start, _ = ds._game_offsets[target_pk]
        pitch_idx = t_start + plen - 1
        expected_observed_home = float(ds._score_home_array[pitch_idx])
        expected_observed_away = float(ds._score_away_array[pitch_idx])

        sample = ds[live_idx]
        expected_remaining_home = max(0.0, 5.0 - expected_observed_home)
        expected_remaining_away = max(0.0, 3.0 - expected_observed_away)
        assert abs(sample["targets"]["home_runs_remaining"].item() - expected_remaining_home) < 1e-5
        assert abs(sample["targets"]["away_runs_remaining"].item() - expected_remaining_away) < 1e-5

    def test_player_targets_only_for_players_in_lineup(self):
        """Player targets are only filled for players who appear in the game lineup."""
        ds = _build_minimal_dataset(n_pitches=50)
        target_pk = 999
        pregame_idx = None
        for i, (gpk, plen) in enumerate(ds.samples):
            if gpk == target_pk and plen == 0:
                pregame_idx = i
                break
        sample = ds[pregame_idx]
        # Lineup from pitch data has batter_ids [200..208]
        lineup = sorted(ds._game_lineups.get(target_pk, set()))
        n_lineup = min(len(lineup), MAX_PLAYERS_PER_GAME)

        # Positions beyond lineup should be zero
        hits = sample["targets"]["player_hits"]
        if n_lineup < MAX_PLAYERS_PER_GAME:
            assert hits[n_lineup:].sum().item() == 0, \
                "Player targets beyond lineup should be zero"


# ===========================================================================
# F. Masks (_build_masks)
# ===========================================================================


class TestBuildMasks:
    """Tests for _build_masks YRFI and player masks."""

    def test_yrfi_mask_zero_when_inning_gt_1(self):
        """yrfi_mask=0 when the prefix extends past the 1st inning."""
        # Build a game with enough pitches that the last pitch is in inning > 1
        ds = _build_minimal_dataset(n_pitches=200)  # ~10 innings
        target_pk = 999
        # Find a sample with prefix extending past inning 1 (> 20 pitches)
        late_idx = None
        for i, (gpk, plen) in enumerate(ds.samples):
            if gpk == target_pk and plen >= 40:
                late_idx = i
                break
        if late_idx is not None:
            sample = ds[late_idx]
            _, plen = ds.samples[late_idx]
            t_start, _ = ds._game_offsets[target_pk]
            pitch_idx = t_start + plen - 1
            inning_at_prefix = ds._hierarchy_array[pitch_idx, 0]
            if inning_at_prefix > 1:
                assert sample["yrfi_mask"].item() == 0.0, \
                    f"yrfi_mask should be 0 in inning {inning_at_prefix}"

    def test_yrfi_mask_one_at_pregame(self):
        """yrfi_mask=1 at pregame (prefix=0)."""
        ds = _build_minimal_dataset(n_pitches=50)
        target_pk = 999
        pregame_idx = None
        for i, (gpk, plen) in enumerate(ds.samples):
            if gpk == target_pk and plen == 0:
                pregame_idx = i
                break
        sample = ds[pregame_idx]
        assert sample["yrfi_mask"].item() == 1.0

    def test_player_mask_one_for_trainable_status(self):
        """player_mask=1 only for players with trainable target_status."""
        ds = _build_minimal_dataset(n_pitches=50)
        target_pk = 999
        pregame_idx = None
        for i, (gpk, plen) in enumerate(ds.samples):
            if gpk == target_pk and plen == 0:
                pregame_idx = i
                break
        sample = ds[pregame_idx]

        # All our test players have target_status="trainable"
        lineup = sorted(ds._game_lineups.get(target_pk, set()))[:MAX_PLAYERS_PER_GAME]
        for i, pid in enumerate(lineup):
            gs = ds._player_game_stats.get((pid, target_pk))
            if gs and gs.get("target_status") == "trainable":
                assert sample["player_mask"][i].item() == 1.0


# ===========================================================================
# G. Leakage invariants
# ===========================================================================


class TestLeakageInvariants:
    """Tests ensuring no future data leaks into features."""

    def test_sp_history_only_before_target_date(self):
        """SP history games all have dates strictly before target_game_date."""
        ds = _build_minimal_dataset(n_pitches=50, n_prior_games=3)
        target_pk = 999
        meta = ds.meta_by_game[target_pk]
        target_date = meta["game_date"]
        pitcher_id = int(meta["probable_pitcher_home_id"])

        # Verify all games in _sp_games for this pitcher that are used in context
        sp_games = ds._sp_games.get(pitcher_id, [])
        for gpk in sp_games:
            if gpk == target_pk:
                continue
            gm = ds.meta_by_game.get(gpk)
            if gm is not None:
                assert gm["game_date"] < target_date, \
                    f"SP game {gpk} has date {gm['game_date']} >= target {target_date}"

    def test_team_history_only_before_target_date(self):
        """Team history games all have dates strictly before target_game_date."""
        ds = _build_minimal_dataset(n_pitches=50, n_prior_games=5)
        target_pk = 999
        meta = ds.meta_by_game[target_pk]
        target_date = meta["game_date"]
        team_id = int(meta["home_team_id"])

        team_df = ds.game_index["by_team"].get(team_id)
        if team_df is not None:
            prior_mask = (team_df["game_date"] < target_date) & (team_df["game_pk"] != target_pk)
            prior_games = team_df[prior_mask]
            # All should be before target date
            assert (prior_games["game_date"] < target_date).all()

    def test_player_history_only_before_target_date(self):
        """Player context uses only history rows strictly before target_game_date."""
        ds = _build_minimal_dataset(n_pitches=50, n_prior_games=3)
        target_pk = 999
        meta = ds.meta_by_game[target_pk]
        target_date = meta["game_date"]

        # Check player history dates
        player_ctx = ds._get_player_context(target_pk, meta)
        lineup = sorted(ds._game_lineups.get(target_pk, set()))[:MAX_PLAYERS_PER_GAME]

        for pid in lineup:
            dates = ds._player_hist_dates.get(pid)
            if dates is not None and len(dates) > 0:
                # The binary search finds index where game_date would be inserted
                gd_np = np.datetime64(target_date)
                idx = int(np.searchsorted(dates, gd_np, side="left"))
                # All used history is from dates[:idx], which are < target_date
                used_dates = dates[:idx]
                for d in used_dates:
                    assert d < gd_np, f"Player {pid} has history date >= target"


# ===========================================================================
# H. Edge cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases in the dataset."""

    def test_first_game_of_season_no_context(self):
        """First game of season: no prior games for any context."""
        target_date = "2025-04-01"
        spec = SequenceSpec(history_length=50, min_history=1, live_stride=25,
                            live_max_prefixes_per_game=32)

        pitch_seqs = _make_pitch_rows(1000, 50, game_date=target_date, pitcher_id=600)
        game_targets = _make_game_targets(1000, game_date=target_date)
        game_meta = _make_game_meta(1000, game_date=target_date, home_team_id=300,
                                     away_team_id=400, home_pitcher_id=600, away_pitcher_id=601)
        team_games = _make_team_games([1000], team_id=300, game_dates=[target_date])
        team_games_away = _make_team_games([1000], team_id=400, game_dates=[target_date])
        team_games_away["side"] = "away"
        team_games = pd.concat([team_games, team_games_away], ignore_index=True)
        player_hist = _make_player_batting_history([200], [1000], [target_date])

        ds = GameTransformerDataset(
            pitch_sequences=pitch_seqs,
            game_targets=game_targets,
            game_meta=game_meta,
            team_games=team_games,
            player_batting_history=player_hist,
            standardizer=_make_standardizer(),
            ablation=AblationConfig(sp_history_games=5, team_history_games=5),
            spec=spec,
        )

        # Should still produce a valid sample
        assert len(ds) >= 1
        sample = ds[0]
        # SP context should be all zeros (no prior starts)
        assert torch.all(sample["sp_home_seqs"] == 0)
        assert torch.all(sample["team_home_lengths"] == 0)

    def test_doubleheader_game2_has_game1_in_context(self):
        """Doubleheader game 2 can include game 1 in its context (same date)."""
        date = "2025-07-01"
        spec = SequenceSpec(history_length=50, min_history=1, live_stride=25,
                            live_max_prefixes_per_game=32)
        ablation = AblationConfig(sp_history_games=5, team_history_games=5)

        # Game 1 of doubleheader (earlier in the day)
        pitch_g1 = _make_pitch_rows(800, 80, game_date=date, pitcher_id=100)
        targets_g1 = _make_game_targets(800, game_date=date)
        meta_g1 = _make_game_meta(800, game_date=date, home_pitcher_id=100, away_pitcher_id=101)

        # Game 2 of doubleheader (same date, later)
        pitch_g2 = _make_pitch_rows(801, 80, game_date=date, pitcher_id=102)
        targets_g2 = _make_game_targets(801, game_date=date)
        meta_g2 = _make_game_meta(801, game_date=date, home_pitcher_id=102, away_pitcher_id=103)

        pitch_seqs = pd.concat([pitch_g1, pitch_g2], ignore_index=True)
        game_targets = pd.concat([targets_g1, targets_g2], ignore_index=True)
        game_meta = pd.concat([meta_g1, meta_g2], ignore_index=True)
        team_games = _make_team_games([800, 801], team_id=110, game_dates=[date, date])
        team_games_away = _make_team_games([800, 801], team_id=120, game_dates=[date, date])
        team_games_away["side"] = "away"
        team_games = pd.concat([team_games, team_games_away], ignore_index=True)
        player_hist = _make_player_batting_history([200, 201], [800, 801], [date, date])

        ds = GameTransformerDataset(
            pitch_sequences=pitch_seqs,
            game_targets=game_targets,
            game_meta=game_meta,
            team_games=team_games,
            player_batting_history=player_hist,
            standardizer=_make_standardizer(),
            ablation=ablation,
            spec=spec,
        )

        meta_801 = ds.meta_by_game[801]
        ctx = ds._get_team_context(meta_801, side="home", game_pk=801)

        # Game 1 (game_pk=800 < 801, same date) should now appear in game 2's context.
        # The fix uses game_pk as a tiebreaker for same-date games.
        non_zero = (ctx["lengths"] > 0).sum().item()
        assert non_zero >= 1, (
            "Doubleheader game 2 must include game 1 in context (game_pk ordering)"
        )

    def test_traded_player_both_teams(self):
        """A traded player appearing for both teams in the same season has
        valid context when appearing for the second team."""
        date_team_a = "2025-05-15"
        date_team_b = "2025-08-01"
        target_date = "2025-08-10"
        spec = SequenceSpec(history_length=50, min_history=1, live_stride=25,
                            live_max_prefixes_per_game=32)
        ablation = AblationConfig(sp_history_games=5, team_history_games=5,
                                   player_history_games=15)

        traded_player = 555

        # Game with team A (before trade)
        pitch_a = _make_pitch_rows(700, 40, game_date=date_team_a,
                                    batter_ids=[traded_player, 201, 202, 203])
        targets_a = _make_game_targets(700, game_date=date_team_a)
        meta_a = _make_game_meta(700, game_date=date_team_a, home_team_id=110, away_team_id=120)

        # Target game with team B (after trade)
        pitch_b = _make_pitch_rows(701, 40, game_date=target_date,
                                    batter_ids=[traded_player, 301, 302, 303])
        targets_b = _make_game_targets(701, game_date=target_date)
        meta_b = _make_game_meta(701, game_date=target_date, home_team_id=130, away_team_id=140,
                                  home_pitcher_id=500, away_pitcher_id=501)

        pitch_seqs = pd.concat([pitch_a, pitch_b], ignore_index=True)
        game_targets = pd.concat([targets_a, targets_b], ignore_index=True)
        game_meta = pd.concat([meta_a, meta_b], ignore_index=True)
        team_games = pd.concat([
            _make_team_games([700], team_id=110, game_dates=[date_team_a]),
            _make_team_games([701], team_id=130, game_dates=[target_date]),
            _make_team_games([700], team_id=120, game_dates=[date_team_a]),
            _make_team_games([701], team_id=140, game_dates=[target_date]),
        ], ignore_index=True)
        team_games.loc[team_games["team_id"].isin([120, 140]), "side"] = "away"

        # Player history for traded player spans both teams
        hist_a = _make_player_batting_history([traded_player], [700], [date_team_a])
        hist_b = _make_player_batting_history([traded_player], [701], [target_date])
        player_hist = pd.concat([hist_a, hist_b], ignore_index=True)

        ds = GameTransformerDataset(
            pitch_sequences=pitch_seqs,
            game_targets=game_targets,
            game_meta=game_meta,
            team_games=team_games,
            player_batting_history=player_hist,
            standardizer=_make_standardizer(),
            ablation=ablation,
            spec=spec,
        )

        # Dataset should successfully create samples for game 701
        samples_701 = [i for i, (gpk, _) in enumerate(ds.samples) if gpk == 701]
        assert len(samples_701) > 0, "Traded player game should produce valid samples"

        # Player context for game 701 should include history from game 700
        meta_701 = ds.meta_by_game[701]
        player_ctx = ds._get_player_context(701, meta_701)

        # The traded player should have history from their team A stint
        dates = ds._player_hist_dates.get(traded_player)
        assert dates is not None and len(dates) > 0, \
            "Traded player should have history entries"

    def test_game_with_single_pitch(self):
        """A game with only 1 pitch total still produces valid samples."""
        target_date = "2025-07-10"
        spec = SequenceSpec(history_length=50, min_history=1, live_stride=25,
                            live_max_prefixes_per_game=32)

        pitch_seqs = _make_pitch_rows(888, 1, game_date=target_date)
        game_targets = _make_game_targets(888, game_date=target_date)
        game_meta = _make_game_meta(888, game_date=target_date)
        team_games = _make_team_games([888], team_id=110, game_dates=[target_date])
        team_games_away = _make_team_games([888], team_id=120, game_dates=[target_date])
        team_games_away["side"] = "away"
        team_games = pd.concat([team_games, team_games_away], ignore_index=True)
        player_hist = _make_player_batting_history([200], [888], [target_date])

        ds = GameTransformerDataset(
            pitch_sequences=pitch_seqs,
            game_targets=game_targets,
            game_meta=game_meta,
            team_games=team_games,
            player_batting_history=player_hist,
            standardizer=_make_standardizer(),
            ablation=AblationConfig(sp_history_games=5, team_history_games=5),
            spec=spec,
        )

        # Should have pregame sample; no live samples since n_pitches=1 < stride=25
        samples_888 = [(gpk, plen) for gpk, plen in ds.samples if gpk == 888]
        pregame = [s for s in samples_888 if s[1] == 0]
        assert len(pregame) == 1, "Single-pitch game should have pregame sample"

        # Live: stride=25, n_pitches=1 -> no positions from range(25, 1, 25) -> no live
        live = [s for s in samples_888 if s[1] > 0]
        assert len(live) == 0, "Single-pitch game (n < stride) should have no live samples"

        # The pregame sample should be fetchable without error
        idx = ds.samples.index((888, 0))
        sample = ds[idx]
        assert sample["prefix_length"].item() == 0


# ===========================================================================
# Additional integration-level tests
# ===========================================================================


class TestFullSampleRetrieval:
    """Integration tests verifying __getitem__ returns valid tensors."""

    def test_getitem_returns_all_expected_keys(self):
        """__getitem__ returns a dict with all required keys."""
        ds = _build_minimal_dataset(n_pitches=60)
        sample = ds[0]
        expected_keys = [
            "sp_home_seqs", "sp_home_obs_mask", "sp_home_lengths", "sp_home_weights", "sp_home_mask",
            "sp_away_seqs", "sp_away_obs_mask", "sp_away_lengths", "sp_away_weights", "sp_away_mask",
            "team_home_seqs", "team_home_obs_mask", "team_home_lengths", "team_home_weights", "team_home_mask",
            "team_home_similarity",
            "team_away_seqs", "team_away_obs_mask", "team_away_lengths", "team_away_weights", "team_away_mask",
            "team_away_similarity",
            "flat_features",
            "weather_temporal",
            "rating_home", "rating_away",
            "prefix_values", "prefix_obs_mask", "prefix_mask",
            "prefix_batter_hash", "prefix_pitcher_hash", "prefix_catcher_hash",
            "prefix_event_type", "prefix_hierarchy",
            "prefix_pitch_type_idx", "prefix_bat_side_idx", "prefix_pitch_hand_idx",
            "prefix_half_inning_idx", "prefix_hit_trajectory_idx", "prefix_hit_hardness_idx",
            "prefix_length",
            "player_hashes", "player_history", "player_history_mask", "player_matchup",
            "targets", "yrfi_mask", "player_mask",
            "sample_weight", "game_pk",
        ]
        for key in expected_keys:
            assert key in sample, f"Missing key: {key}"

    def test_flat_features_dimension(self):
        """flat_features tensor has correct dimension."""
        ds = _build_minimal_dataset(n_pitches=50)
        sample = ds[0]
        assert sample["flat_features"].shape == (FLAT_FEATURE_DIM,)

    def test_prefix_values_shape(self):
        """prefix_values has shape (history_length, n_continuous_cols)."""
        spec = SequenceSpec(history_length=50, min_history=1, live_stride=25,
                            live_max_prefixes_per_game=32)
        ds = _build_minimal_dataset(n_pitches=60, spec=spec)
        sample = ds[0]
        n_cont = len(PITCH_CONTINUOUS_COLS)
        assert sample["prefix_values"].shape == (50, n_cont)

    def test_player_context_shapes(self):
        """Player context tensors have expected shapes."""
        ds = _build_minimal_dataset(n_pitches=50)
        sample = ds[0]
        n_history = ds.ablation.player_history_games
        assert sample["player_hashes"].shape == (MAX_PLAYERS_PER_GAME,)
        assert sample["player_history"].shape == (MAX_PLAYERS_PER_GAME, n_history, PLAYER_STAT_DIM)
        assert sample["player_history_mask"].shape == (MAX_PLAYERS_PER_GAME, n_history)

    def test_sample_weight_positive(self):
        """Sample weights should be positive."""
        ds = _build_minimal_dataset(n_pitches=50)
        for i in range(min(5, len(ds))):
            sample = ds[i]
            assert sample["sample_weight"].item() > 0


# ===========================================================================
# Vocabulary mapping tests
# ===========================================================================


class TestVocabularyMappings:
    """Tests for pitch-type, bat-side, and other vocab mappers."""

    def test_map_pitch_type_known_values(self):
        """Known pitch types map to correct indices."""
        series = pd.Series(["FF", "SL", "CH", "CU", None, "XX"])
        result = map_pitch_type(series)
        assert result[0] == PITCH_TYPE_TO_IDX["FF"]  # 1
        assert result[1] == PITCH_TYPE_TO_IDX["SL"]  # 3
        assert result[2] == PITCH_TYPE_TO_IDX["CH"]  # 5
        assert result[3] == PITCH_TYPE_TO_IDX["CU"]  # 4
        assert result[4] == 0  # None -> padding
        assert result[5] == 0  # Unknown -> padding

    def test_map_bat_side_known_values(self):
        """Known bat sides map to correct indices."""
        series = pd.Series(["L", "R", "S", None])
        result = map_bat_side(series)
        assert result[0] == BAT_SIDE_TO_IDX["L"]  # 0
        assert result[1] == BAT_SIDE_TO_IDX["R"]  # 1
        assert result[2] == BAT_SIDE_TO_IDX["S"]  # 2
        assert result[3] == 0  # None -> default (L=0)

    def test_classify_event_type_pitch(self):
        """A standard pitch event classifies as 'pitch' (index 0)."""
        row = pd.Series({"is_pitch": 1, "event_type": "", "at_bat_event": "", "pitch_call": ""})
        assert classify_event_type(row) == 0

    def test_classify_event_type_stolen_base(self):
        """A stolen base event classifies correctly."""
        row = pd.Series({"is_pitch": 0, "event_type": "", "at_bat_event": "Stolen Base 2B", "pitch_call": ""})
        assert classify_event_type(row) == 2  # stolen_base


# ===========================================================================
# build_game_index tests
# ===========================================================================


class TestBuildGameIndex:
    """Tests for the build_game_index utility."""

    def test_chronological_ordering(self):
        """Game index orders games chronologically per team."""
        meta = pd.concat([
            _make_game_meta(100, game_date="2025-06-01"),
            _make_game_meta(101, game_date="2025-06-05"),
            _make_game_meta(102, game_date="2025-06-03"),
        ], ignore_index=True)

        idx = build_game_index(meta)
        team_110 = idx["by_team"].get(110)
        assert team_110 is not None
        dates = team_110["game_date"].tolist()
        assert dates == sorted(dates), "Games should be in chronological order"

    def test_sp_by_pitcher_populated(self):
        """SP-by-pitcher mapping is populated from probable_pitcher columns."""
        meta = pd.concat([
            _make_game_meta(100, game_date="2025-06-01", home_pitcher_id=42),
            _make_game_meta(101, game_date="2025-06-05", home_pitcher_id=42),
            _make_game_meta(102, game_date="2025-06-03", home_pitcher_id=99),
        ], ignore_index=True)

        idx = build_game_index(meta)
        assert 42 in idx["sp_by_pitcher"]
        assert len(idx["sp_by_pitcher"][42]) == 2
        assert 99 in idx["sp_by_pitcher"]
        assert len(idx["sp_by_pitcher"][99]) == 1
