"""Round-trip audit: save_dataset → CachedGameTransformerDataset → __getitem__.

Checks four risk areas without loading any parquet files:
  1. Game offset correctness — _game_offsets[pk] slices the right pitch rows
  2. Standardizer fidelity — mean/std survive JSON serialization
  3. Array value fidelity — pitch feature values are numerically identical post-load
  4. Player history dates — binary-search dates are preserved and still enforce no-leakage

Run:
    conda run -n pred python -m pytest deep_learning/mlb_dl/tests/test_cache_roundtrip.py -v
"""

from __future__ import annotations

import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[3]))

from deep_learning.mlb_dl.dataset_cache import save_dataset, load_dataset
from deep_learning.mlb_dl.game_transformer_dataset import (
    GameTransformerDataset,
    AblationConfig,
    PITCH_CONTINUOUS_COLS,
)
from deep_learning.mlb_dl.datasets import SequenceSpec, Standardizer


# ---------------------------------------------------------------------------
# Minimal synthetic data builders (copied from test_dataset_operations.py)
# ---------------------------------------------------------------------------

def _make_standardizer() -> Standardizer:
    return Standardizer(
        feature_columns=list(PITCH_CONTINUOUS_COLS),
        mean={col: float(i) for i, col in enumerate(PITCH_CONTINUOUS_COLS)},
        std={col: float(i + 1) for i, col in enumerate(PITCH_CONTINUOUS_COLS)},
    )


def _make_pitch_rows(game_pk, n_pitches, game_date="2025-07-01", pitcher_id=100,
                     batter_ids=None, season=2025):
    if batter_ids is None:
        batter_ids = [200, 201, 202, 203, 204, 205, 206, 207, 208]
    rows = []
    for i in range(n_pitches):
        rows.append({
            "game_pk": game_pk,
            "season": season,
            "game_date": game_date,
            "play_index": i,
            "at_bat_index": i // 4,
            "pitch_sequence_index": i,
            "pitch_number": (i % 4) + 1,
            "inning": 1 + i // 20,
            "is_top_inning": 1 if i < n_pitches // 2 else 0,
            "batter_id": batter_ids[(i // 4) % len(batter_ids)],
            "pitcher_id": pitcher_id,
            "fielder_2": 300,
            "pitch_type": "FF",
            "bat_side_code": "R",
            "pitch_hand_code": "R",
            "is_pitch": 1,
            "is_strike": 1 if i % 3 == 0 else 0,
            "is_ball": 1 if i % 3 == 1 else 0,
            "is_in_play": 1 if i % 3 == 2 else 0,
            "score_home": float(i // 40),
            "score_away": float(i // 50),
            "release_speed": 90.0 + i * 0.1,   # deterministic, not random
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


def _make_game_targets(game_pk, game_date="2025-07-01", season=2025,
                       home_runs=5, away_runs=3, target_status="trainable"):
    return pd.DataFrame([{
        "game_pk": game_pk, "game_date": game_date, "season": season,
        "home_team_id": 110, "away_team_id": 120,
        "target_status": target_status,
        "home_win": 1 if home_runs > away_runs else 0,
        "away_win": 0 if home_runs > away_runs else 1,
        "yrfi": 1, "nrfi": 0, "extra_innings": 0,
        "total_runs": home_runs + away_runs,
        "home_runs": home_runs, "away_runs": away_runs,
        "home_run_diff": home_runs - away_runs,
        "away_run_diff": away_runs - home_runs,
        "first_5_total_runs": 4, "first_5_home_runs": 2, "first_5_away_runs": 2,
        "first_5_home_run_diff": 0, "first_5_away_run_diff": 0,
        "first_5_home_win": 0, "first_5_away_win": 0, "first_5_tie": 1,
    }])


def _make_game_meta(game_pk, game_date="2025-07-01", season=2025,
                    home_team_id=110, away_team_id=120,
                    home_pitcher_id=100, away_pitcher_id=101):
    return pd.DataFrame([{
        "game_pk": game_pk, "game_date": game_date, "season": season,
        "home_team_id": home_team_id, "away_team_id": away_team_id,
        "probable_pitcher_home_id": home_pitcher_id,
        "probable_pitcher_away_id": away_pitcher_id,
        "venue_id": 1, "day_night": "night", "game_number": 1,
        "double_header": "N", "tiebreaker": "N",
        "start_time": "19:05",
        "game_datetime_utc": f"{game_date}T23:05:00Z",
        "venue_latitude": 40.75, "venue_longitude": -73.85,
        "venue_capacity": 41922, "venue_surface": "grass", "venue_roof_type": "open",
        "umpire_hp": "Joe West",
        "rule_3batter_minimum": 1.0, "rule_universal_dh": 1.0,
        "rule_shift_ban_pitch_clock": 1.0,
    }])


def _make_team_games(game_pks, team_id=110, game_dates=None, season=2025):
    if game_dates is None:
        base = pd.Timestamp("2025-06-01")
        game_dates = [(base + timedelta(days=i)).strftime("%Y-%m-%d")
                      for i in range(len(game_pks))]
    rows = [{"game_pk": gpk, "team_id": team_id, "game_date": gd,
             "season": season, "side": "home"}
            for gpk, gd in zip(game_pks, game_dates)]
    return pd.DataFrame(rows)


def _make_player_batting_history(player_ids, game_pks, game_dates, season=2025):
    rows = []
    for gpk, gd in zip(game_pks, game_dates):
        for pid in player_ids:
            rows.append({
                "player_id": pid, "game_pk": gpk, "game_date": gd,
                "season": season, "side": "home", "target_status": "trainable",
                "game_ab": 4, "game_runs": 1, "game_hits": 1,
                "game_doubles": 0, "game_triples": 0, "game_hr": 0,
                "game_rbi": 1, "game_bb": 0, "game_ibb": 0, "game_so": 1,
                "game_sb": 0, "game_cs": 0, "game_hbp": 0, "game_sac": 0,
                "game_sf": 0, "game_gidp": 0, "game_lob": 2,
                "game_total_bases": 1, "game_hits_runs_rbi": 3,
                "season_avg": 0.270, "season_obp": 0.340, "season_slg": 0.430,
                "season_ops": 0.770, "season_hr": 10, "season_rbi": 40,
                "season_sb": 5, "season_games_played": 80,
            })
    return pd.DataFrame(rows)


def _build_dataset(n_prior=3, target_date="2025-07-15"):
    """Build a small GameTransformerDataset fully in memory."""
    spec = SequenceSpec(history_length=50, min_history=1, live_stride=25,
                        live_max_prefixes_per_game=8)
    ablation = AblationConfig(sp_history_games=3, team_history_games=3)

    target_pk = 999
    pitch_seqs = _make_pitch_rows(target_pk, 80, game_date=target_date)
    game_targets = _make_game_targets(target_pk, game_date=target_date)
    game_meta = _make_game_meta(target_pk, game_date=target_date)

    prior_pks = list(range(900, 900 + n_prior))
    prior_dates = [
        (pd.Timestamp(target_date) - timedelta(days=5 * (n_prior - i))).strftime("%Y-%m-%d")
        for i in range(n_prior)
    ]
    for gpk, gd in zip(prior_pks, prior_dates):
        pitch_seqs = pd.concat([pitch_seqs,
                                 _make_pitch_rows(gpk, 60, game_date=gd)],
                                ignore_index=True)
        game_targets = pd.concat([game_targets,
                                   _make_game_targets(gpk, game_date=gd)],
                                  ignore_index=True)
        game_meta = pd.concat([game_meta,
                                _make_game_meta(gpk, game_date=gd)],
                               ignore_index=True)

    all_pks = prior_pks + [target_pk]
    all_dates = prior_dates + [target_date]
    team_games = pd.concat([
        _make_team_games(all_pks, team_id=110, game_dates=all_dates),
        pd.DataFrame([{"game_pk": gpk, "team_id": 120, "game_date": gd,
                       "season": 2025, "side": "away"}
                      for gpk, gd in zip(all_pks, all_dates)]),
    ], ignore_index=True)

    batter_ids = [200, 201, 202, 203, 204, 205, 206, 207, 208]
    player_hist = _make_player_batting_history(batter_ids, all_pks, all_dates)

    return GameTransformerDataset(
        pitch_sequences=pitch_seqs,
        game_targets=game_targets,
        game_meta=game_meta,
        team_games=team_games,
        player_batting_history=player_hist,
        standardizer=_make_standardizer(),
        ablation=ablation,
        spec=spec,
        include_pregame=True,
        include_live=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def roundtrip(tmp_path_factory):
    """Build original dataset, save to disk, reload as CachedGameTransformerDataset."""
    tmp = tmp_path_factory.mktemp("cache_audit")
    original = _build_dataset()
    save_dataset(original, tmp, "train")
    cached = load_dataset(tmp, "train")
    return original, cached


# ---------------------------------------------------------------------------
# 1. Game offset correctness
# ---------------------------------------------------------------------------

class TestGameOffsets:
    def test_same_keys(self, roundtrip):
        orig, cached = roundtrip
        assert set(orig._game_offsets.keys()) == set(cached._game_offsets.keys()), \
            "Different game_pk keys after round-trip"

    def test_offsets_identical(self, roundtrip):
        orig, cached = roundtrip
        for gpk in orig._game_offsets:
            assert orig._game_offsets[gpk] == cached._game_offsets[gpk], \
                f"game_pk {gpk}: offset {orig._game_offsets[gpk]} != {cached._game_offsets[gpk]}"

    def test_offset_slices_correct_game(self, roundtrip):
        """The (start, end) slice for each game_pk should contain only rows for that game."""
        orig, cached = roundtrip
        for gpk, (start, end) in cached._game_offsets.items():
            # batter_id_array should be consistent with game's known batters
            batter_ids_in_slice = set(cached._batter_id_array[start:end].tolist())
            # The slice must be non-empty
            assert end > start, f"game_pk {gpk} has empty slice [{start},{end})"
            # batter IDs should be our known test batter range (200-208)
            for bid in batter_ids_in_slice:
                assert 200 <= bid <= 208, \
                    f"game_pk {gpk}: unexpected batter_id {bid} in slice [{start},{end})"

    def test_slices_non_overlapping_and_contiguous(self, roundtrip):
        """All game slices together should cover exactly 0..total_pitches with no gaps or overlaps."""
        _, cached = roundtrip
        total = len(cached._pitch_cont_array)
        covered = np.zeros(total, dtype=np.int32)
        for gpk, (start, end) in cached._game_offsets.items():
            covered[start:end] += 1
        assert (covered == 1).all(), \
            f"Pitch rows not covered exactly once: {np.where(covered != 1)[0][:10]}"


# ---------------------------------------------------------------------------
# 2. Standardizer fidelity
# ---------------------------------------------------------------------------

class TestStandardizerFidelity:
    def test_feature_columns_preserved(self, roundtrip):
        orig, cached = roundtrip
        assert orig.standardizer.feature_columns == cached.standardizer.feature_columns

    def test_mean_values_preserved(self, roundtrip):
        orig, cached = roundtrip
        for col in orig.standardizer.feature_columns:
            assert abs(orig.standardizer.mean[col] - cached.standardizer.mean[col]) < 1e-7, \
                f"mean[{col}] drifted: {orig.standardizer.mean[col]} -> {cached.standardizer.mean[col]}"

    def test_std_values_preserved(self, roundtrip):
        orig, cached = roundtrip
        for col in orig.standardizer.feature_columns:
            assert abs(orig.standardizer.std[col] - cached.standardizer.std[col]) < 1e-7, \
                f"std[{col}] drifted: {orig.standardizer.std[col]} -> {cached.standardizer.std[col]}"


# ---------------------------------------------------------------------------
# 3. Array value fidelity
# ---------------------------------------------------------------------------

class TestArrayValueFidelity:
    def test_pitch_cont_array_identical(self, roundtrip):
        orig, cached = roundtrip
        np.testing.assert_array_equal(
            orig._pitch_cont_array, cached._pitch_cont_array,
            err_msg="pitch_cont_array differs after round-trip"
        )

    def test_score_home_array_identical(self, roundtrip):
        orig, cached = roundtrip
        np.testing.assert_array_equal(
            orig._score_home_array, cached._score_home_array,
            err_msg="score_home_array differs after round-trip"
        )

    def test_hierarchy_array_identical(self, roundtrip):
        orig, cached = roundtrip
        np.testing.assert_array_equal(
            orig._hierarchy_array, cached._hierarchy_array,
            err_msg="hierarchy_array (inning/ab/pitch) differs after round-trip"
        )

    def test_batter_hash_array_identical(self, roundtrip):
        orig, cached = roundtrip
        np.testing.assert_array_equal(
            orig._batter_hash_array, cached._batter_hash_array,
            err_msg="batter_hash_array differs after round-trip"
        )

    def test_getitem_identical_for_all_samples(self, roundtrip):
        """Every sample produced by __getitem__ is numerically identical between
        original and cached datasets."""
        orig, cached = roundtrip
        assert len(orig) == len(cached), \
            f"Sample count differs: {len(orig)} vs {len(cached)}"
        for i in range(len(orig)):
            s_orig = orig[i]
            s_cached = cached[i]
            for key in s_orig:
                v_orig = s_orig[key]
                v_cached = s_cached[key]
                if isinstance(v_orig, torch.Tensor):
                    if not torch.equal(v_orig, v_cached):
                        diff = (v_orig - v_cached).abs().max().item()
                        pytest.fail(f"Sample {i}, key '{key}': max diff = {diff:.6e}")
                elif isinstance(v_orig, dict):
                    for subkey in v_orig:
                        sv = v_orig[subkey]
                        sc = v_cached[subkey]
                        if isinstance(sv, torch.Tensor) and not torch.equal(sv, sc):
                            diff = (sv - sc).abs().max().item()
                            pytest.fail(
                                f"Sample {i}, targets['{subkey}']: max diff = {diff:.6e}"
                            )


# ---------------------------------------------------------------------------
# 4. Player history date fidelity (no-leakage invariant preserved post-load)
# ---------------------------------------------------------------------------

class TestPlayerHistoryDates:
    def test_player_ids_preserved(self, roundtrip):
        orig, cached = roundtrip
        assert set(orig._player_hist_dates.keys()) == set(cached._player_hist_dates.keys())

    def test_dates_identical_per_player(self, roundtrip):
        orig, cached = roundtrip
        for pid in orig._player_hist_dates:
            orig_dates = orig._player_hist_dates[pid]
            cached_dates = cached._player_hist_dates[pid]
            assert len(orig_dates) == len(cached_dates), \
                f"Player {pid}: date array length {len(orig_dates)} vs {len(cached_dates)}"
            np.testing.assert_array_equal(
                orig_dates, cached_dates,
                err_msg=f"Player {pid}: dates differ after round-trip"
            )

    def test_stats_identical_per_player(self, roundtrip):
        orig, cached = roundtrip
        for pid in orig._player_hist_stat_arrays:
            np.testing.assert_array_almost_equal(
                orig._player_hist_stat_arrays[pid],
                cached._player_hist_stat_arrays[pid],
                decimal=5,
                err_msg=f"Player {pid}: stat arrays differ after round-trip"
            )

    def test_binary_search_returns_same_index(self, roundtrip):
        """np.searchsorted on cached dates returns same result as on original,
        ensuring no-leakage cutoff is identical after deserialization."""
        orig, cached = roundtrip
        target_date = np.datetime64("2025-07-15")
        for pid in orig._player_hist_dates:
            orig_dates = orig._player_hist_dates[pid]
            cached_dates = cached._player_hist_dates[pid]
            if len(orig_dates) == 0:
                continue
            idx_orig = int(np.searchsorted(orig_dates, target_date, side="left"))
            idx_cached = int(np.searchsorted(cached_dates, target_date, side="left"))
            assert idx_orig == idx_cached, \
                f"Player {pid}: searchsorted cutoff {idx_orig} vs {idx_cached}"
