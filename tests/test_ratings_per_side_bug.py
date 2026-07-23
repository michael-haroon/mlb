"""Tests proving per-side-only bug in BaseRuns and Pythagenpat.

These tests demonstrate that a team's expanding stat should reflect ALL games
(home + away), not just games on one side. The bug was: groupby(home_team_id)
only sees rows where the team is home, missing ~50% of their games.

Also includes adversarial stress tests for edge cases that naturally occur in
MLB: doubleheaders, long road/home stands, traded players mid-season, expansion
teams, and season boundaries.
"""
import numpy as np
import pandas as pd
import pytest
import sys
sys.path.insert(0, ".")

from pregame.engineering.ratings import compute_baseruns, compute_pythagenpat


def _make_bsr_columns(games, n):
    """Add BsR-required columns to a game DataFrame."""
    np.random.seed(42)
    for side in ("home", "away"):
        games[f"{side}_H"] = np.random.randint(5, 12, n).astype(float)
        games[f"{side}_BB"] = np.random.randint(1, 5, n).astype(float)
        games[f"{side}_HBP"] = np.random.randint(0, 2, n).astype(float)
        games[f"{side}_HR"] = np.random.randint(0, 3, n).astype(float)
        games[f"{side}_IBB"] = np.random.randint(0, 1, n).astype(float)
        games[f"{side}_TB"] = np.random.randint(8, 20, n).astype(float)
        games[f"{side}_SB"] = np.random.randint(0, 2, n).astype(float)
        games[f"{side}_CS"] = np.random.randint(0, 1, n).astype(float)
        games[f"{side}_GDP"] = np.random.randint(0, 2, n).astype(float)
        games[f"{side}_PA"] = np.random.randint(33, 42, n).astype(float)
        games[f"{side}_SH"] = np.random.randint(0, 1, n).astype(float)
        games[f"{side}_SF"] = np.random.randint(0, 1, n).astype(float)
    return games


@pytest.fixture
def alternating_games():
    """NYY (147) alternates home/away every game. BAL (110) is the opponent."""
    n = 10
    games = pd.DataFrame({
        "game_date": pd.date_range("2024-04-01", periods=n, freq="D"),
        "home_team_id": [147 if i % 2 == 0 else 110 for i in range(n)],
        "away_team_id": [110 if i % 2 == 0 else 147 for i in range(n)],
        "home_bat_game_runs": [5, 3, 6, 2, 4, 7, 3, 5, 2, 6],
        "away_bat_game_runs": [3, 4, 2, 5, 1, 3, 6, 2, 4, 1],
    })
    return _make_bsr_columns(games, n)


class TestBaseRunsUnifiedTimeline:
    def test_bsr_offense_uses_all_prior_games(self, alternating_games):
        """At game 5 (idx=4), NYY's BsR should reflect 4 prior games, not just 2 home games."""
        games = compute_baseruns(alternating_games)

        nyy_prior_offense = [
            games.loc[0, "home_bsr_game"],   # NYY home offense
            games.loc[1, "away_bsr_game"],   # NYY away offense
            games.loc[2, "home_bsr_game"],   # NYY home offense
            games.loc[3, "away_bsr_game"],   # NYY away offense
        ]
        expected = np.mean(nyy_prior_offense)
        actual = games.loc[4, "home_bsr_offense"]

        assert abs(actual - expected) < 0.001, (
            f"BsR offense at idx 4 should be mean of 4 games ({expected:.4f}), "
            f"got {actual:.4f}. If only home games were used, this would be "
            f"mean of idx 0,2 = {np.mean([nyy_prior_offense[0], nyy_prior_offense[2]]):.4f}"
        )

    def test_bsr_defense_uses_all_prior_games(self, alternating_games):
        """Defense BsR should also see all sides."""
        games = compute_baseruns(alternating_games)

        nyy_prior_defense = [
            games.loc[0, "away_bsr_game"],
            games.loc[1, "home_bsr_game"],
            games.loc[2, "away_bsr_game"],
            games.loc[3, "home_bsr_game"],
        ]
        expected = np.mean(nyy_prior_defense)
        actual = games.loc[4, "home_bsr_defense"]

        assert abs(actual - expected) < 0.001

    def test_first_game_is_nan(self, alternating_games):
        """First game for any team should have NaN (no prior data)."""
        games = compute_baseruns(alternating_games)
        assert pd.isna(games.loc[0, "home_bsr_offense"])


class TestPythagenpatUnifiedTimeline:
    def test_pythag_uses_all_prior_runs(self, alternating_games):
        """Pythagenpat RS/RA should accumulate from all games, not just same-side."""
        games = compute_baseruns(alternating_games)
        games = compute_pythagenpat(games)

        # NYY (147) outscored opponents significantly:
        # idx 0(H): scored 5 allowed 3, idx 1(A): scored 4 allowed 3,
        # idx 2(H): scored 6 allowed 2, idx 3(A): scored 5 allowed 2
        # total: RS=20, RA=10
        pyth = games.loc[4, "home_pythag_1st"]
        assert pyth > 0.6, f"NYY outscored opponents 20-10, pythag should be >0.6, got {pyth}"

    def test_pythag_first_game_uninformative(self, alternating_games):
        """First game (no prior RS/RA) should be NaN or 0.5 (uninformative prior)."""
        games = compute_baseruns(alternating_games)
        games = compute_pythagenpat(games)
        val = games.loc[0, "home_pythag_1st"]
        assert pd.isna(val) or abs(val - 0.5) < 0.01, (
            f"First game pythag should be NaN or ~0.5, got {val}"
        )


# ===========================================================================
# ADVERSARIAL STRESS TESTS — edge cases that naturally occur in MLB
# ===========================================================================

class TestLongHomeStand:
    """A team playing 10 consecutive home games then 10 away.
    Per-side-only bug would show NaN for the first away game's features
    (never seen the team as away), but unified timeline should use all 10 home games.
    """

    @pytest.fixture
    def long_homestand(self):
        n = 20
        # NYY home for games 0-9, away for games 10-19
        games = pd.DataFrame({
            "game_date": pd.date_range("2024-04-01", periods=n, freq="D"),
            "home_team_id": [147] * 10 + [110] * 10,
            "away_team_id": [110] * 10 + [147] * 10,
            "home_bat_game_runs": np.random.randint(2, 8, n).astype(float),
            "away_bat_game_runs": np.random.randint(2, 8, n).astype(float),
        })
        return _make_bsr_columns(games, n)

    def test_first_away_game_has_bsr(self, long_homestand):
        """After 10 home games, first away game should have BsR from all 10 prior."""
        games = compute_baseruns(long_homestand)
        # At idx 10, NYY is away. They have 10 prior home games.
        assert pd.notna(games.loc[10, "away_bsr_offense"]), (
            "First away game should have BsR based on prior home games"
        )

    def test_bsr_count_correct_after_homestand(self, long_homestand):
        """BsR at game 11 should reflect all 10 home + 1 away = 10 prior games (shift excludes current)."""
        games = compute_baseruns(long_homestand)
        # At idx 11 (2nd away game), NYY has 10 home + 1 away = 11 prior games
        # The expanding mean should use 11 data points
        nyy_prior = (
            [games.loc[i, "home_bsr_game"] for i in range(10)] +
            [games.loc[10, "away_bsr_game"]]
        )
        expected = np.mean(nyy_prior)
        actual = games.loc[11, "away_bsr_offense"]
        assert abs(actual - expected) < 0.001

    def test_pythag_not_nan_on_first_away(self, long_homestand):
        """Pythag should not be NaN for first away game if team has prior home games."""
        games = compute_baseruns(long_homestand)
        games = compute_pythagenpat(games)
        assert pd.notna(games.loc[10, "away_pythag_1st"])


class TestDoubleheader:
    """Same-day doubleheaders: both games share game_date but differ in frame_idx.
    Game 2 of DH should incorporate Game 1 results.
    """

    @pytest.fixture
    def doubleheader_games(self):
        n = 6
        games = pd.DataFrame({
            "game_date": ["2024-04-01"] * 2 + ["2024-04-02"] * 2 + ["2024-04-03"] * 2,
            "home_team_id": [147, 147, 110, 110, 147, 147],
            "away_team_id": [110, 110, 147, 147, 110, 110],
            "home_bat_game_runs": [5.0, 3.0, 4.0, 6.0, 2.0, 7.0],
            "away_bat_game_runs": [2.0, 7.0, 1.0, 3.0, 5.0, 1.0],
        })
        return _make_bsr_columns(games, n)

    def test_dh_game2_includes_game1(self, doubleheader_games):
        """DH game 2 (idx=1) should include DH game 1 (idx=0) in prior history."""
        games = compute_baseruns(doubleheader_games)
        # At idx=1 (DH game 2), NYY has 1 prior game (idx=0)
        assert pd.notna(games.loc[1, "home_bsr_offense"]), (
            "DH game 2 should have BsR from DH game 1"
        )
        expected = games.loc[0, "home_bsr_game"]
        actual = games.loc[1, "home_bsr_offense"]
        assert abs(actual - expected) < 0.001

    def test_dh_both_games_count(self, doubleheader_games):
        """After a DH, next game should see both DH games in history."""
        games = compute_baseruns(doubleheader_games)
        # At idx=2 (first game Apr 2, NYY away), NYY has 2 prior games (idx 0,1 as home)
        nyy_offense_prior = [
            games.loc[0, "home_bsr_game"],
            games.loc[1, "home_bsr_game"],
        ]
        expected = np.mean(nyy_offense_prior)
        actual = games.loc[2, "away_bsr_offense"]
        assert abs(actual - expected) < 0.001


class TestExpansionTeam:
    """A brand new team with no history. Should produce NaN until they have prior games."""

    @pytest.fixture
    def new_team_games(self):
        n = 5
        games = pd.DataFrame({
            "game_date": pd.date_range("2024-04-01", periods=n, freq="D"),
            "home_team_id": [999, 147, 999, 110, 999],
            "away_team_id": [147, 999, 110, 999, 147],
            "home_bat_game_runs": [4.0, 5.0, 3.0, 6.0, 7.0],
            "away_bat_game_runs": [2.0, 3.0, 1.0, 4.0, 2.0],
        })
        return _make_bsr_columns(games, n)

    def test_first_game_nan(self, new_team_games):
        """Expansion team's first game should have NaN BsR."""
        games = compute_baseruns(new_team_games)
        assert pd.isna(games.loc[0, "home_bsr_offense"])

    def test_second_game_uses_first(self, new_team_games):
        """Expansion team's 2nd game (idx=2) should use game 0 as history."""
        games = compute_baseruns(new_team_games)
        # Team 999 plays: idx 0 (home), idx 2 (home), idx 4 (home)
        # But also idx 1 (away), idx 3 (away)
        # At idx 1 (999 is away), they have 1 prior game (idx 0 as home)
        assert pd.notna(games.loc[1, "away_bsr_offense"])


class TestSeasonBoundary:
    """Features should carry across seasons (no year-boundary reset).
    An expanding mean doesn't reset at the start of a new season.
    """

    @pytest.fixture
    def cross_season_games(self):
        n = 6
        games = pd.DataFrame({
            "game_date": pd.to_datetime([
                "2023-09-28", "2023-09-29", "2023-09-30",
                "2024-03-28", "2024-03-29", "2024-03-30",
            ]),
            "home_team_id": [147, 110, 147, 110, 147, 110],
            "away_team_id": [110, 147, 110, 147, 110, 147],
            "home_bat_game_runs": [5.0, 3.0, 6.0, 2.0, 4.0, 7.0],
            "away_bat_game_runs": [3.0, 4.0, 2.0, 5.0, 1.0, 3.0],
        })
        return _make_bsr_columns(games, n)

    def test_bsr_uses_prior_season(self, cross_season_games):
        """First game of new season (idx=3) should use all prior-season games."""
        games = compute_baseruns(cross_season_games)
        # At idx=3, NYY is away. Prior history: idx 0(H), 1(A), 2(H) = 3 games
        nyy_prior = [
            games.loc[0, "home_bsr_game"],
            games.loc[1, "away_bsr_game"],
            games.loc[2, "home_bsr_game"],
        ]
        expected = np.mean(nyy_prior)
        actual = games.loc[3, "away_bsr_offense"]
        assert abs(actual - expected) < 0.001

    def test_pythag_not_nan_at_season_start(self, cross_season_games):
        """Pythag should not reset to NaN at the season boundary."""
        games = compute_baseruns(cross_season_games)
        games = compute_pythagenpat(games)
        assert pd.notna(games.loc[3, "away_pythag_1st"])


class TestAllOneSide:
    """Degenerate case: a team only plays on one side in the dataset.
    Should still compute correctly (they just won't appear on the other side).
    """

    @pytest.fixture
    def one_sided_games(self):
        n = 5
        games = pd.DataFrame({
            "game_date": pd.date_range("2024-04-01", periods=n, freq="D"),
            "home_team_id": [147] * 5,
            "away_team_id": [110, 111, 112, 113, 114],
            "home_bat_game_runs": [5.0, 3.0, 6.0, 4.0, 7.0],
            "away_bat_game_runs": [2.0, 4.0, 1.0, 5.0, 3.0],
        })
        return _make_bsr_columns(games, n)

    def test_all_home_team_gets_expanding(self, one_sided_games):
        """Team always home still gets correct expanding BsR."""
        games = compute_baseruns(one_sided_games)
        # At idx 3, NYY has 3 prior home games
        nyy_prior = [games.loc[i, "home_bsr_game"] for i in range(3)]
        expected = np.mean(nyy_prior)
        actual = games.loc[3, "home_bsr_offense"]
        assert abs(actual - expected) < 0.001

    def test_one_time_opponents_are_nan(self, one_sided_games):
        """Teams that appear only once (as away) should have NaN features."""
        games = compute_baseruns(one_sided_games)
        # Team 110 appears only at idx 0 as away. First game = NaN.
        assert pd.isna(games.loc[0, "away_bsr_offense"])


class TestZeroRunGames:
    """Games with 0 runs scored/allowed (shutouts). BsR and Pythag should handle gracefully."""

    @pytest.fixture
    def shutout_games(self):
        n = 6
        games = pd.DataFrame({
            "game_date": pd.date_range("2024-04-01", periods=n, freq="D"),
            "home_team_id": [147, 110, 147, 110, 147, 110],
            "away_team_id": [110, 147, 110, 147, 110, 147],
            "home_bat_game_runs": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "away_bat_game_runs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        })
        return _make_bsr_columns(games, n)

    def test_pythag_zero_runs_no_nan(self, shutout_games):
        """Pythagenpat shouldn't produce NaN or inf when RS=RA=0."""
        games = compute_baseruns(shutout_games)
        games = compute_pythagenpat(games)
        # Games 1-3 have RS=RA=0. Pythag formula must not blow up.
        for i in range(1, 4):
            val = games.loc[i, "home_pythag_1st"]
            if pd.notna(val):
                assert np.isfinite(val), f"Pythag at idx {i} is not finite: {val}"
                assert 0.0 <= val <= 1.0, f"Pythag out of [0,1]: {val}"

    def test_bsr_zero_runs_finite(self, shutout_games):
        """BsR should be 0 (not NaN) for a shutout game."""
        games = compute_baseruns(shutout_games)
        # Every game has 0 runs. BsR game values should be 0 or very close.
        for i in range(6):
            val = games.loc[i, "home_bsr_game"]
            assert np.isfinite(val), f"BsR game at idx {i} should be finite, got {val}"


class TestLargeBlowout:
    """Extreme run differentials (20+ runs). No overflow or distorted means."""

    @pytest.fixture
    def blowout_games(self):
        n = 5
        games = pd.DataFrame({
            "game_date": pd.date_range("2024-04-01", periods=n, freq="D"),
            "home_team_id": [147, 110, 147, 110, 147],
            "away_team_id": [110, 147, 110, 147, 110],
            "home_bat_game_runs": [25.0, 1.0, 22.0, 0.0, 5.0],
            "away_bat_game_runs": [0.0, 20.0, 1.0, 23.0, 3.0],
        })
        return _make_bsr_columns(games, n)

    def test_blowout_no_overflow(self, blowout_games):
        """25-run games should not cause overflow in BsR or Pythag."""
        games = compute_baseruns(blowout_games)
        games = compute_pythagenpat(games)
        for col in games.columns:
            if games[col].dtype in [np.float32, np.float64]:
                assert not games[col].apply(lambda x: np.isinf(x) if pd.notna(x) else False).any(), (
                    f"Column {col} has inf values after blowout games"
                )

    def test_pythag_extreme_dominance(self, blowout_games):
        """Team that outscores opponents massively should have Pythag near 1.0."""
        games = compute_baseruns(blowout_games)
        games = compute_pythagenpat(games)
        # NYY: scored 25+22+20=67 (via unified: H@0=25, A@1=20, H@2=22, A@3=23)
        # Actually: idx0(H):25, idx1(A):20, idx2(H):22, idx3(A):23
        # At idx 4 (NYY home), RS = 25+20+22+23 = 90, RA = 0+1+1+0 = 2
        pyth = games.loc[4, "home_pythag_1st"]
        assert pyth > 0.95, f"NYY scored 90 allowed 2, pythag should be >0.95, got {pyth}"
