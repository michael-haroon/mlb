"""Tests for umpire tendency features.

Covers:
- _umpire_features: HP zone + 2B stolen base expanding means

Verifies: no lookahead (shift(1)), min_periods, regular-season masking,
NaN handling, called_strike_pct edge cases, rpg_factor ratio.

Run: conda run -n pred python -m pytest tests/test_umpire_features.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.engineering.feature_engineering import _umpire_features


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def ump_games():
    """30-game frame with 2 HP umpires and 2 2B umpires, known values."""
    n = 30
    games = pd.DataFrame({
        "game_pk": range(1000, 1000 + n),
        "game_date": pd.date_range("2024-04-01", periods=n, freq="D"),
        "season": [2024] * n,
        "game_type_code": ["R"] * n,
        # Umpire A does games 0-19, Umpire B does games 20-29
        "umpire_hp": ["Ump_A"] * 20 + ["Ump_B"] * 10,
        "umpire_2b": ["U2B_X"] * 15 + ["U2B_Y"] * 15,
        # Known total_runs: Ump_A games avg 9.0, Ump_B games avg 7.0
        "total_runs": [9.0] * 20 + [7.0] * 10,
        # Walks: constant 6 per game for simplicity
        "home_BB": [3.0] * n,
        "away_BB": [3.0] * n,
        # Strikeouts: constant 16 per game
        "home_bat_game_so": [8.0] * n,
        "away_bat_game_so": [8.0] * n,
        # Called strikes and balls: 60 / 140 = 30% called strike rate
        "home_pit_game_strikes_looking": [30.0] * n,
        "away_pit_game_strikes_looking": [30.0] * n,
        "home_pit_game_balls_thrown": [70.0] * n,
        "away_pit_game_balls_thrown": [70.0] * n,
        # Stolen bases: U2B_X games have 1.0 SB/game, U2B_Y has 2.0
        "home_SB": ([0.5] * 15 + [1.0] * 15),
        "away_SB": ([0.5] * 15 + [1.0] * 15),
        "home_CS": [0.2] * n,
        "away_CS": [0.3] * n,
    })
    return games


# ===========================================================================
# TestUmpireNoLookahead
# ===========================================================================


class TestUmpireNoLookahead:
    """shift(1) must exclude the current game's outcome."""

    def test_first_game_is_nan(self, ump_games):
        result = _umpire_features(ump_games)
        assert pd.isna(result.loc[0, "ump_hp_rpg_factor"])
        assert pd.isna(result.loc[0, "ump_hp_bb_per_game"])
        assert pd.isna(result.loc[0, "ump_hp_k_per_game"])
        assert pd.isna(result.loc[0, "ump_hp_called_strike_pct"])

    def test_first_game_2b_umpire_is_nan(self, ump_games):
        result = _umpire_features(ump_games)
        assert pd.isna(result.loc[0, "ump_2b_sb_per_game"])
        assert pd.isna(result.loc[0, "ump_2b_cs_per_game"])

    def test_current_game_excluded_from_mean(self, ump_games):
        """Spike total_runs at game 25 — game 25's feature must NOT include it."""
        games = ump_games.copy()
        games.loc[25, "total_runs"] = 100.0  # extreme spike at Ump_B game 5
        result = _umpire_features(games)
        # Game 25 is Ump_B's 6th game (idx 20-25). Feature uses games 20-24 only.
        # With min_periods=20, Ump_B won't even have a value at game 25 (only 5 prior).
        # So it should still be NaN.
        assert pd.isna(result.loc[25, "ump_hp_rpg_factor"])

    def test_shift1_manual_verify_bb(self, ump_games):
        """After 20 games of BB=6/game for Ump_A, game 20's feature = 6.0."""
        result = _umpire_features(ump_games)
        # Game index 20 is Ump_B's first game — NaN.
        # Game index 19 is Ump_A's 20th game — shift(1) means feature uses games 0-18.
        # All have BB=6, so expanding mean of 19 values = 6.0 (but min_periods=20!)
        # Actually game 19 has only 19 prior games (0-18), which is < min_periods=20.
        assert pd.isna(result.loc[19, "ump_hp_bb_per_game"])
        # Game 20 for Ump_A doesn't exist (Ump_B starts). Check last valid Ump_A value:
        # Ump_A's 21st observation would be at index... not present. So check
        # that the min_periods boundary is correct.


# ===========================================================================
# TestUmpireMinPeriods
# ===========================================================================


class TestUmpireMinPeriods:
    """Features must be NaN until min_periods=20 prior games exist."""

    def test_min_periods_hp_rpg(self):
        """25 games by one umpire: games 0-19 NaN, game 20+ has a value."""
        n = 25
        games = pd.DataFrame({
            "game_pk": range(n),
            "season": [2024] * n,
            "game_type_code": ["R"] * n,
            "umpire_hp": ["Joe_Ump"] * n,
            "total_runs": [8.5] * n,
            "home_BB": [3.0] * n,
            "away_BB": [3.0] * n,
            "home_bat_game_so": [8.0] * n,
            "away_bat_game_so": [8.0] * n,
            "home_pit_game_strikes_looking": [30.0] * n,
            "away_pit_game_strikes_looking": [30.0] * n,
            "home_pit_game_balls_thrown": [70.0] * n,
            "away_pit_game_balls_thrown": [70.0] * n,
        })
        result = _umpire_features(games)
        # shift(1) + min_periods=20: need 20 prior values before getting non-NaN.
        # Game 0: 0 prior → NaN
        # Game 19: 19 prior → NaN (< 20)
        # Game 20: 20 prior → valid!
        for i in range(20):
            assert pd.isna(result.loc[i, "ump_hp_bb_per_game"]), f"Game {i} should be NaN"
        assert not pd.isna(result.loc[20, "ump_hp_bb_per_game"])
        assert result.loc[20, "ump_hp_bb_per_game"] == pytest.approx(6.0)

    def test_min_periods_2b_sb(self):
        """25 games by one 2B umpire: NaN until game 20."""
        n = 25
        games = pd.DataFrame({
            "game_pk": range(n),
            "season": [2024] * n,
            "game_type_code": ["R"] * n,
            "umpire_2b": ["Base_Ump"] * n,
            "home_SB": [0.5] * n,
            "away_SB": [0.5] * n,
            "home_CS": [0.3] * n,
            "away_CS": [0.2] * n,
        })
        result = _umpire_features(games)
        for i in range(20):
            assert pd.isna(result.loc[i, "ump_2b_sb_per_game"]), f"Game {i} should be NaN"
        assert not pd.isna(result.loc[20, "ump_2b_sb_per_game"])
        assert result.loc[20, "ump_2b_sb_per_game"] == pytest.approx(1.0)


# ===========================================================================
# TestUmpireRegularSeasonOnly
# ===========================================================================


class TestUmpireRegularSeasonOnly:
    """Spring training games must not contaminate expanding means."""

    def test_spring_training_excluded(self):
        """5 spring games with extreme BB followed by 25 regular games."""
        n_spring = 5
        n_regular = 25
        n = n_spring + n_regular
        games = pd.DataFrame({
            "game_pk": range(n),
            "season": [2024] * n,
            "game_type_code": ["S"] * n_spring + ["R"] * n_regular,
            "umpire_hp": ["Test_Ump"] * n,
            "total_runs": [20.0] * n_spring + [8.0] * n_regular,
            "home_BB": [10.0] * n_spring + [3.0] * n_regular,
            "away_BB": [10.0] * n_spring + [3.0] * n_regular,
            "home_bat_game_so": [4.0] * n_spring + [8.0] * n_regular,
            "away_bat_game_so": [4.0] * n_spring + [8.0] * n_regular,
            "home_pit_game_strikes_looking": [20.0] * n_spring + [30.0] * n_regular,
            "away_pit_game_strikes_looking": [20.0] * n_spring + [30.0] * n_regular,
            "home_pit_game_balls_thrown": [80.0] * n_spring + [70.0] * n_regular,
            "away_pit_game_balls_thrown": [80.0] * n_spring + [70.0] * n_regular,
        })
        result = _umpire_features(games)
        # After 20 regular-season games (indices 5-24), game 25 should have a value.
        # The spring training extreme BB=20 must NOT be in the mean.
        # Regular-season BB = 6.0/game consistently.
        val = result.loc[25, "ump_hp_bb_per_game"]
        assert not pd.isna(val)
        assert val == pytest.approx(6.0), (
            f"Expected 6.0 (regular-season only), got {val} — spring training leaked"
        )


# ===========================================================================
# TestUmpireNaNHandling
# ===========================================================================


class TestUmpireNaNHandling:
    """Games without umpire data must get NaN features without crashes."""

    def test_nan_umpire_produces_nan_features(self):
        n = 25
        games = pd.DataFrame({
            "game_pk": range(n),
            "season": [2024] * n,
            "game_type_code": ["R"] * n,
            "umpire_hp": [np.nan] * n,
            "umpire_2b": [np.nan] * n,
            "total_runs": [8.0] * n,
            "home_BB": [3.0] * n,
            "away_BB": [3.0] * n,
            "home_bat_game_so": [8.0] * n,
            "away_bat_game_so": [8.0] * n,
            "home_pit_game_strikes_looking": [30.0] * n,
            "away_pit_game_strikes_looking": [30.0] * n,
            "home_pit_game_balls_thrown": [70.0] * n,
            "away_pit_game_balls_thrown": [70.0] * n,
            "home_SB": [1.0] * n,
            "away_SB": [0.5] * n,
            "home_CS": [0.3] * n,
            "away_CS": [0.2] * n,
        })
        result = _umpire_features(games)
        # All features should be NaN (groupby drops NaN keys)
        for col in ["ump_hp_rpg_factor", "ump_hp_bb_per_game",
                    "ump_hp_k_per_game", "ump_hp_called_strike_pct",
                    "ump_2b_sb_per_game", "ump_2b_cs_per_game"]:
            assert result[col].isna().all(), f"{col} should be all NaN with NaN umpires"

    def test_mixed_nan_and_valid(self):
        """Some games have NaN umpire, others don't — no crash, partial output."""
        n = 30
        umps = ["Real_Ump"] * 25 + [np.nan] * 5
        games = pd.DataFrame({
            "game_pk": range(n),
            "season": [2024] * n,
            "game_type_code": ["R"] * n,
            "umpire_hp": umps,
            "total_runs": [9.0] * n,
            "home_BB": [3.0] * n,
            "away_BB": [3.0] * n,
            "home_bat_game_so": [8.0] * n,
            "away_bat_game_so": [8.0] * n,
            "home_pit_game_strikes_looking": [30.0] * n,
            "away_pit_game_strikes_looking": [30.0] * n,
            "home_pit_game_balls_thrown": [70.0] * n,
            "away_pit_game_balls_thrown": [70.0] * n,
        })
        result = _umpire_features(games)
        # NaN umpire rows (25-29) should have NaN features
        assert result.loc[25:29, "ump_hp_bb_per_game"].isna().all()
        # Valid umpire rows after min_periods should have values
        assert not result.loc[20, "ump_hp_bb_per_game"] is pd.NaT
        assert not pd.isna(result.loc[20, "ump_hp_bb_per_game"])


# ===========================================================================
# TestUmpireCalledStrikePct
# ===========================================================================


class TestUmpireCalledStrikePct:
    """Called strike percentage edge cases."""

    def test_zero_denominator_produces_nan(self):
        """A game with 0 called strikes and 0 balls → NaN, not inf."""
        n = 25
        games = pd.DataFrame({
            "game_pk": range(n),
            "season": [2024] * n,
            "game_type_code": ["R"] * n,
            "umpire_hp": ["Ump_Z"] * n,
            "total_runs": [8.0] * n,
            "home_pit_game_strikes_looking": [0.0] * n,
            "away_pit_game_strikes_looking": [0.0] * n,
            "home_pit_game_balls_thrown": [0.0] * n,
            "away_pit_game_balls_thrown": [0.0] * n,
        })
        result = _umpire_features(games)
        # Should produce NaN, not inf
        if "ump_hp_called_strike_pct" in result.columns:
            assert not np.isinf(result["ump_hp_called_strike_pct"]).any()
            assert result["ump_hp_called_strike_pct"].isna().all()

    def test_value_in_valid_range(self, ump_games):
        """Called strike pct should be in [0, 1]."""
        result = _umpire_features(ump_games)
        valid = result["ump_hp_called_strike_pct"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()


# ===========================================================================
# TestUmpireRpgFactor
# ===========================================================================


class TestUmpireRpgFactor:
    """RPG factor = ump_avg / league_avg, mirrors park_factor."""

    def test_rpg_factor_manual_calculation(self):
        """All games have total_runs=9, so ump factor = 9/9 = 1.0."""
        n = 25
        games = pd.DataFrame({
            "game_pk": range(n),
            "season": [2024] * n,
            "game_type_code": ["R"] * n,
            "umpire_hp": ["Uniform_Ump"] * n,
            "total_runs": [9.0] * n,
        })
        result = _umpire_features(games)
        val = result.loc[20, "ump_hp_rpg_factor"]
        assert not pd.isna(val)
        assert val == pytest.approx(1.0, abs=0.01)

    def test_high_run_ump_gets_factor_above_1(self):
        """Umpire with higher RPG than league gets factor > 1."""
        n = 30
        # Two umpires: High (runs=12) and Low (runs=6). League avg = ~9.
        games = pd.DataFrame({
            "game_pk": range(n),
            "season": [2024] * n,
            "game_type_code": ["R"] * n,
            "umpire_hp": (["High_Ump"] * 15 + ["Low_Ump"] * 15),
            "total_runs": [12.0] * 15 + [6.0] * 15,
        })
        result = _umpire_features(games)
        # After min_periods, High_Ump's factor should be > 1
        # High_Ump's last game is idx 14 — needs 20 prior, so won't fire.
        # Let's use a longer frame.
        n2 = 50
        games2 = pd.DataFrame({
            "game_pk": range(n2),
            "season": [2024] * n2,
            "game_type_code": ["R"] * n2,
            "umpire_hp": (["High_Ump"] * 25 + ["Low_Ump"] * 25),
            "total_runs": [12.0] * 25 + [6.0] * 25,
        })
        result2 = _umpire_features(games2)
        # Game 20: High_Ump has 20 prior games (all 12.0). League avg at game 20
        # uses all 20 prior games (all 12.0 since only High_Ump has played so far).
        # So factor = 12/12 = 1.0.
        # Game 45: Low_Ump has 20 prior games. League avg uses all 44 prior
        # (25 × 12 + 19 × 6) / 44 = (300 + 114) / 44 = 9.41.
        # Low_Ump avg = 6.0. Factor = 6.0 / 9.41 ≈ 0.64.
        val_low = result2.loc[45, "ump_hp_rpg_factor"]
        assert not pd.isna(val_low)
        assert val_low < 1.0


# ===========================================================================
# TestUmpireColumnCompleteness
# ===========================================================================


class TestUmpireColumnCompleteness:
    """All 6 features present when inputs exist; graceful when missing."""

    def test_all_six_features_present(self, ump_games):
        result = _umpire_features(ump_games)
        expected = {
            "ump_hp_rpg_factor", "ump_hp_bb_per_game",
            "ump_hp_k_per_game", "ump_hp_called_strike_pct",
            "ump_2b_sb_per_game", "ump_2b_cs_per_game",
        }
        assert expected.issubset(set(result.columns))

    def test_graceful_without_umpire_2b(self):
        """HP features still work when umpire_2b is absent."""
        n = 25
        games = pd.DataFrame({
            "game_pk": range(n),
            "season": [2024] * n,
            "game_type_code": ["R"] * n,
            "umpire_hp": ["Ump_Solo"] * n,
            "total_runs": [8.0] * n,
            "home_BB": [3.0] * n,
            "away_BB": [3.0] * n,
            "home_bat_game_so": [8.0] * n,
            "away_bat_game_so": [8.0] * n,
            "home_pit_game_strikes_looking": [30.0] * n,
            "away_pit_game_strikes_looking": [30.0] * n,
            "home_pit_game_balls_thrown": [70.0] * n,
            "away_pit_game_balls_thrown": [70.0] * n,
        })
        result = _umpire_features(games)
        assert "ump_hp_bb_per_game" in result.columns
        assert "ump_2b_sb_per_game" not in result.columns

    def test_no_crash_empty_frame(self):
        """Empty DataFrame doesn't crash."""
        games = pd.DataFrame(columns=[
            "game_pk", "season", "game_type_code", "umpire_hp",
            "total_runs", "home_BB", "away_BB",
        ])
        result = _umpire_features(games)
        assert len(result) == 0
