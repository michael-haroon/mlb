"""Tests for schedule context features and H2H removal.

Defines expected behavior BEFORE implementation:
1. H2H features (`h2h_home_winrate_10`, `h2h_rd_mean_10`) must NOT be produced
2. Schedule context flags (`is_same_division`, `is_same_league`) must be produced
3. Interaction features (elo_prob * flag) must be produced for linear models
4. Adversarial edge cases: missing IDs, same division number across leagues,
   all-interleague stretches, Houston 2013 division change

Run: conda run -n pred python -m pytest tests/test_schedule_context.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Real MLB IDs: league 103=AL, 104=NL
# AL East division_id=201, AL Central=202, AL West=203
# NL East=204, NL Central=205, NL West=206
# NYY=147 (AL East), BOS=111 (AL East), LAD=119 (NL West), CHC=112 (NL Central)

@pytest.fixture
def intra_division_game():
    """NYY vs BOS — same division (AL East)."""
    return pd.DataFrame({
        "game_pk": [1001],
        "game_date": pd.to_datetime(["2024-06-15"]),
        "home_team_id": [147],
        "away_team_id": [111],
        "home_league_id": [103],
        "away_league_id": [103],
        "home_division_id": [201],
        "away_division_id": [201],
        "elo_prob": [0.58],
        "consensus_home_win_prob": [0.56],
    })


@pytest.fixture
def intra_league_cross_division_game():
    """NYY vs HOU — same league (AL), different divisions (East vs West)."""
    return pd.DataFrame({
        "game_pk": [1002],
        "game_date": pd.to_datetime(["2024-06-15"]),
        "home_team_id": [147],
        "away_team_id": [117],
        "home_league_id": [103],
        "away_league_id": [103],
        "home_division_id": [201],
        "away_division_id": [203],
        "elo_prob": [0.52],
        "consensus_home_win_prob": [0.51],
    })


@pytest.fixture
def interleague_game():
    """NYY vs LAD — different leagues (AL vs NL)."""
    return pd.DataFrame({
        "game_pk": [1003],
        "game_date": pd.to_datetime(["2024-06-15"]),
        "home_team_id": [147],
        "away_team_id": [119],
        "home_league_id": [103],
        "away_league_id": [104],
        "home_division_id": [201],
        "away_division_id": [206],
        "elo_prob": [0.61],
        "consensus_home_win_prob": [0.59],
    })


@pytest.fixture
def mixed_context_games():
    """Multiple games spanning all three matchup contexts."""
    return pd.DataFrame({
        "game_pk": [2001, 2002, 2003, 2004, 2005],
        "game_date": pd.date_range("2024-04-01", periods=5, freq="D"),
        "home_team_id": [147, 147, 147, 112, 119],
        "away_team_id": [111, 117, 119, 158, 137],
        # NYY vs BOS (same div), NYY vs HOU (same lg), NYY vs LAD (interleague),
        # CHC vs MIL (same div NL Central), LAD vs SF (same div NL West)
        "home_league_id": [103, 103, 103, 104, 104],
        "away_league_id": [103, 103, 104, 104, 104],
        "home_division_id": [201, 201, 201, 205, 206],
        "away_division_id": [201, 203, 206, 205, 206],
        "elo_prob": [0.55, 0.52, 0.61, 0.48, 0.53],
        "consensus_home_win_prob": [0.54, 0.51, 0.59, 0.47, 0.52],
    })


# ---------------------------------------------------------------------------
# Test 1: H2H removal — engineer_features must NOT produce H2H columns
# ---------------------------------------------------------------------------

class TestH2HRemoval:
    """H2H features (h2h_home_winrate_10, h2h_rd_mean_10) are pure noise
    (r=0.01 with outcome, N=2178 games needed to detect 3% effect at 80% power).
    They must not be produced by the engineering pipeline."""

    def test_engineer_features_excludes_h2h_columns(self, mixed_context_games):
        """The top-level engineer_features must not produce H2H columns."""
        from classical_learning.engineering.feature_engineering import engineer_features

        # Add minimum required columns for engineer_features to run
        df = mixed_context_games.copy()
        df["season"] = 2024
        df["game_type_code"] = "R"
        df["venue_id"] = 1
        df["total_runs"] = 8
        df["home_win"] = [1, 0, 1, 0, 1]
        df["home_run_diff"] = [2, -1, 3, -2, 1]

        result = engineer_features(df)
        h2h_cols = [c for c in result.columns if "h2h" in c.lower()]
        assert h2h_cols == [], (
            f"H2H columns should not be produced but found: {h2h_cols}"
        )

    def test_h2h_function_still_exists_but_unused(self):
        """The _head_to_head function may still exist (for reference) but
        must not be called from engineer_features."""
        from classical_learning.engineering import feature_engineering
        import inspect

        source = inspect.getsource(feature_engineering.engineer_features)
        assert "_head_to_head" not in source, (
            "engineer_features still calls _head_to_head — H2H should be removed"
        )


# ---------------------------------------------------------------------------
# Test 2: Schedule context flags — correct classification of matchup type
# ---------------------------------------------------------------------------

class TestScheduleContextFlags:
    """Binary flags indicating matchup context for model interaction learning."""

    def test_same_division_game(self, intra_division_game):
        """NYY vs BOS: same division=1, same league=1."""
        from classical_learning.engineering.feature_engineering import _schedule_context

        result = _schedule_context(intra_division_game)
        assert result["is_same_division"].iloc[0] == 1.0
        assert result["is_same_league"].iloc[0] == 1.0

    def test_same_league_different_division(self, intra_league_cross_division_game):
        """NYY vs HOU: same division=0, same league=1."""
        from classical_learning.engineering.feature_engineering import _schedule_context

        result = _schedule_context(intra_league_cross_division_game)
        assert result["is_same_division"].iloc[0] == 0.0
        assert result["is_same_league"].iloc[0] == 1.0

    def test_interleague_game(self, interleague_game):
        """NYY vs LAD: same division=0, same league=0."""
        from classical_learning.engineering.feature_engineering import _schedule_context

        result = _schedule_context(interleague_game)
        assert result["is_same_division"].iloc[0] == 0.0
        assert result["is_same_league"].iloc[0] == 0.0

    def test_all_contexts_in_batch(self, mixed_context_games):
        """Multiple games correctly classified in one pass."""
        from classical_learning.engineering.feature_engineering import _schedule_context

        result = _schedule_context(mixed_context_games)
        # game 0: NYY vs BOS (same div)
        assert result["is_same_division"].iloc[0] == 1.0
        assert result["is_same_league"].iloc[0] == 1.0
        # game 1: NYY vs HOU (same lg, diff div)
        assert result["is_same_division"].iloc[1] == 0.0
        assert result["is_same_league"].iloc[1] == 1.0
        # game 2: NYY vs LAD (interleague)
        assert result["is_same_division"].iloc[2] == 0.0
        assert result["is_same_league"].iloc[2] == 0.0
        # game 3: CHC vs MIL (same div NL Central)
        assert result["is_same_division"].iloc[3] == 1.0
        assert result["is_same_league"].iloc[3] == 1.0
        # game 4: LAD vs SF (same div NL West)
        assert result["is_same_division"].iloc[4] == 1.0
        assert result["is_same_league"].iloc[4] == 1.0

    def test_output_dtype_is_float32(self, intra_division_game):
        """Flags must be float32 for consistency with other features."""
        from classical_learning.engineering.feature_engineering import _schedule_context

        result = _schedule_context(intra_division_game)
        assert result["is_same_division"].dtype == np.float32
        assert result["is_same_league"].dtype == np.float32

    def test_same_division_implies_same_league(self, mixed_context_games):
        """Invariant: is_same_division=1 must always have is_same_league=1."""
        from classical_learning.engineering.feature_engineering import _schedule_context

        result = _schedule_context(mixed_context_games)
        same_div_mask = result["is_same_division"] == 1.0
        assert (result.loc[same_div_mask, "is_same_league"] == 1.0).all(), (
            "Found game marked same_division=1 but same_league=0 — impossible"
        )


# ---------------------------------------------------------------------------
# Test 3: Interaction features for linear models
# ---------------------------------------------------------------------------

class TestScheduleInteractions:
    """Pre-computed interactions: elo_prob * flag, consensus * flag.
    Linear models can't discover splits — they need explicit product terms."""

    def test_elo_interaction_same_league(self, intra_division_game):
        """elo_prob_x_same_league = elo_prob when same league."""
        from classical_learning.engineering.feature_engineering import _schedule_context

        result = _schedule_context(intra_division_game)
        expected = 0.58 * 1.0  # elo_prob * is_same_league
        assert abs(result["elo_prob_x_same_league"].iloc[0] - expected) < 1e-5

    def test_elo_interaction_interleague(self, interleague_game):
        """elo_prob_x_same_league = 0.0 when interleague."""
        from classical_learning.engineering.feature_engineering import _schedule_context

        result = _schedule_context(interleague_game)
        assert result["elo_prob_x_same_league"].iloc[0] == 0.0

    def test_elo_interaction_same_division(self, intra_division_game):
        """elo_prob_x_same_division = elo_prob when same division."""
        from classical_learning.engineering.feature_engineering import _schedule_context

        result = _schedule_context(intra_division_game)
        expected = 0.58 * 1.0
        assert abs(result["elo_prob_x_same_division"].iloc[0] - expected) < 1e-5

    def test_consensus_interaction(self, interleague_game):
        """consensus_home_win_prob_x_same_league = 0.0 when interleague."""
        from classical_learning.engineering.feature_engineering import _schedule_context

        result = _schedule_context(interleague_game)
        assert result["consensus_prob_x_same_league"].iloc[0] == 0.0

    def test_interactions_are_float32(self, mixed_context_games):
        """All interaction columns must be float32."""
        from classical_learning.engineering.feature_engineering import _schedule_context

        result = _schedule_context(mixed_context_games)
        interaction_cols = [c for c in result.columns if "_x_" in c]
        assert len(interaction_cols) > 0, "No interaction columns found"
        for col in interaction_cols:
            assert result[col].dtype == np.float32, f"{col} is {result[col].dtype}"


# ---------------------------------------------------------------------------
# Test 4: Adversarial edge cases
# ---------------------------------------------------------------------------

class TestScheduleContextAdversarial:
    """Edge cases that occur in real MLB data."""

    def test_missing_league_id_both_sides(self):
        """Games with NaN league_id (spring training, exhibition) get NaN flags."""
        from classical_learning.engineering.feature_engineering import _schedule_context

        df = pd.DataFrame({
            "game_pk": [9001],
            "game_date": pd.to_datetime(["2024-03-01"]),
            "home_team_id": [147],
            "away_team_id": [111],
            "home_league_id": [np.nan],
            "away_league_id": [np.nan],
            "home_division_id": [np.nan],
            "away_division_id": [np.nan],
            "elo_prob": [0.55],
            "consensus_home_win_prob": [0.53],
        })
        result = _schedule_context(df)
        # NaN comparison should produce NaN or 0.0 — never 1.0
        assert result["is_same_division"].iloc[0] != 1.0
        assert result["is_same_league"].iloc[0] != 1.0

    def test_missing_league_id_one_side(self):
        """One team has NaN league_id — should not produce 1.0 for same_league."""
        from classical_learning.engineering.feature_engineering import _schedule_context

        df = pd.DataFrame({
            "game_pk": [9002],
            "game_date": pd.to_datetime(["2024-03-01"]),
            "home_team_id": [147],
            "away_team_id": [111],
            "home_league_id": [103],
            "away_league_id": [np.nan],
            "home_division_id": [201],
            "away_division_id": [np.nan],
            "elo_prob": [0.55],
            "consensus_home_win_prob": [0.53],
        })
        result = _schedule_context(df)
        assert result["is_same_league"].iloc[0] != 1.0

    def test_same_division_number_different_leagues(self):
        """Hypothetical: if division IDs were reused across leagues (they're not
        in current MLB, but tests should be robust to schema changes).
        Same division_id alone is NOT sufficient — must also be same league."""
        from classical_learning.engineering.feature_engineering import _schedule_context

        df = pd.DataFrame({
            "game_pk": [9003],
            "game_date": pd.to_datetime(["2024-06-15"]),
            "home_team_id": [147],
            "away_team_id": [119],
            "home_league_id": [103],  # AL
            "away_league_id": [104],  # NL
            # Force same division_id to test this edge case
            "home_division_id": [201],
            "away_division_id": [201],
            "elo_prob": [0.55],
            "consensus_home_win_prob": [0.53],
        })
        result = _schedule_context(df)
        # Different leagues = NOT same division regardless of division_id match
        assert result["is_same_division"].iloc[0] == 0.0
        assert result["is_same_league"].iloc[0] == 0.0

    def test_missing_elo_prob_interactions_are_nan(self):
        """If elo_prob is NaN, interactions must be NaN (not 0.0).
        NaN * 1.0 = NaN. This prevents imputation from silently turning
        missing ratings into 'zero home advantage' signals."""
        from classical_learning.engineering.feature_engineering import _schedule_context

        df = pd.DataFrame({
            "game_pk": [9004],
            "game_date": pd.to_datetime(["2024-06-15"]),
            "home_team_id": [147],
            "away_team_id": [111],
            "home_league_id": [103],
            "away_league_id": [103],
            "home_division_id": [201],
            "away_division_id": [201],
            "elo_prob": [np.nan],
            "consensus_home_win_prob": [np.nan],
        })
        result = _schedule_context(df)
        # NaN * 1.0 should remain NaN, not become 0.0
        assert pd.isna(result["elo_prob_x_same_league"].iloc[0])
        assert pd.isna(result["elo_prob_x_same_division"].iloc[0])

    def test_missing_elo_with_zero_flag(self):
        """If elo_prob is NaN and flag is 0.0, interaction should still be NaN.
        NaN * 0.0 = NaN in pandas (not 0.0)."""
        from classical_learning.engineering.feature_engineering import _schedule_context

        df = pd.DataFrame({
            "game_pk": [9005],
            "game_date": pd.to_datetime(["2024-06-15"]),
            "home_team_id": [147],
            "away_team_id": [119],
            "home_league_id": [103],
            "away_league_id": [104],
            "home_division_id": [201],
            "away_division_id": [206],
            "elo_prob": [np.nan],
            "consensus_home_win_prob": [0.53],
        })
        result = _schedule_context(df)
        # NaN * 0.0 = NaN (not 0.0), preserving missingness signal
        assert pd.isna(result["elo_prob_x_same_league"].iloc[0])

    def test_all_interleague_stretch(self):
        """10 consecutive interleague games — all flags should be 0."""
        from classical_learning.engineering.feature_engineering import _schedule_context

        n = 10
        df = pd.DataFrame({
            "game_pk": range(9010, 9010 + n),
            "game_date": pd.date_range("2024-06-01", periods=n, freq="D"),
            "home_team_id": [147] * n,
            "away_team_id": [119] * n,
            "home_league_id": [103] * n,
            "away_league_id": [104] * n,
            "home_division_id": [201] * n,
            "away_division_id": [206] * n,
            "elo_prob": np.linspace(0.50, 0.60, n),
            "consensus_home_win_prob": np.linspace(0.49, 0.58, n),
        })
        result = _schedule_context(df)
        assert (result["is_same_division"] == 0.0).all()
        assert (result["is_same_league"] == 0.0).all()
        assert (result["elo_prob_x_same_league"] == 0.0).all()
        assert (result["elo_prob_x_same_division"] == 0.0).all()

    def test_no_league_division_columns_graceful_fallback(self):
        """If league_id/division_id columns are absent entirely (legacy data),
        function should still produce columns filled with NaN (not crash)."""
        from classical_learning.engineering.feature_engineering import _schedule_context

        df = pd.DataFrame({
            "game_pk": [9020],
            "game_date": pd.to_datetime(["2024-06-15"]),
            "home_team_id": [147],
            "away_team_id": [111],
            "elo_prob": [0.55],
            "consensus_home_win_prob": [0.53],
        })
        result = _schedule_context(df)
        assert "is_same_division" in result.columns
        assert "is_same_league" in result.columns
        # Should be NaN when source columns are missing
        assert pd.isna(result["is_same_division"].iloc[0])
        assert pd.isna(result["is_same_league"].iloc[0])

    def test_missing_consensus_prob_column(self):
        """If consensus_home_win_prob doesn't exist yet (called before
        _consensus_probability), interaction should gracefully handle absence."""
        from classical_learning.engineering.feature_engineering import _schedule_context

        df = pd.DataFrame({
            "game_pk": [9021],
            "game_date": pd.to_datetime(["2024-06-15"]),
            "home_team_id": [147],
            "away_team_id": [111],
            "home_league_id": [103],
            "away_league_id": [103],
            "home_division_id": [201],
            "away_division_id": [201],
            "elo_prob": [0.55],
            # No consensus_home_win_prob column
        })
        result = _schedule_context(df)
        # Should either not produce consensus interaction, or produce NaN
        if "consensus_prob_x_same_league" in result.columns:
            assert pd.isna(result["consensus_prob_x_same_league"].iloc[0])


# ---------------------------------------------------------------------------
# Rest and schedule density — non-monotonic date crash (pandas 3.0.5 regression)
# ---------------------------------------------------------------------------

class TestRestAndScheduleMonotonic:
    """_rest_and_schedule must handle duplicate dates (doubleheaders) without crashing.

    pandas 3.0.5 enforces strict monotonicity for time-based rolling windows.
    Two games on the same date for the same team produce a non-strictly-monotonic
    datetime index in _games_last_7d, which raises ValueError on pandas >=3.0.5.
    """

    _T1, _T2 = 147, 111

    def _make_games(self, rows):
        from classical_learning.engineering.feature_engineering import _rest_and_schedule
        import pandas as pd
        return pd.DataFrame(rows).reset_index(drop=True), _rest_and_schedule

    def test_doubleheader_same_date_does_not_crash(self):
        """Two games same date for same team (doubleheader) must not raise ValueError."""
        import pandas as pd
        from classical_learning.engineering.feature_engineering import _rest_and_schedule

        games = pd.DataFrame([
            {"game_pk": 1, "game_date": "2015-03-07", "season": 2015,
             "home_team_id": self._T1, "away_team_id": self._T2, "game_type_code": "S",
             "game_number": 1},
            {"game_pk": 2, "game_date": "2015-03-07", "season": 2015,
             "home_team_id": self._T1, "away_team_id": self._T2, "game_type_code": "S",
             "game_number": 2},
            {"game_pk": 3, "game_date": "2015-03-08", "season": 2015,
             "home_team_id": self._T1, "away_team_id": self._T2, "game_type_code": "S",
             "game_number": 1},
        ]).reset_index(drop=True)

        result = _rest_and_schedule(games)
        assert "home_days_rest" in result.columns
        assert "home_games_last_7d" in result.columns

    def test_games_last_7d_counts_doubleheader_correctly(self):
        """After the fix, games_last_7d for game 3 must count both DH games from Mar 7."""
        import pandas as pd
        from classical_learning.engineering.feature_engineering import _rest_and_schedule

        games = pd.DataFrame([
            {"game_pk": 1, "game_date": "2015-03-07", "season": 2015,
             "home_team_id": self._T1, "away_team_id": self._T2, "game_type_code": "S",
             "game_number": 1},
            {"game_pk": 2, "game_date": "2015-03-07", "season": 2015,
             "home_team_id": self._T1, "away_team_id": self._T2, "game_type_code": "S",
             "game_number": 2},
            {"game_pk": 3, "game_date": "2015-03-10", "season": 2015,
             "home_team_id": self._T1, "away_team_id": self._T2, "game_type_code": "S",
             "game_number": 1},
        ]).reset_index(drop=True)

        result = _rest_and_schedule(games)
        # game 3 (Mar 10): 2 prior games in last 7d (both DH games on Mar 7)
        assert result.loc[2, "home_games_last_7d"] == 2.0
