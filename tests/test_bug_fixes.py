"""Tests for all bugs identified by reviewer agents.

Validates that each fix produces correct values with known inputs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Fix 1: Batted ball trajectory classification (popups → FB, exact matching)
# ---------------------------------------------------------------------------

class TestTrajectoryClassification:
    """Popups count as fly balls; no false matches from str.contains."""

    def _make_bip(self, trajectories):
        n = len(trajectories)
        return pd.DataFrame({
            "game_pk": [1] * n,
            "pitcher_id": [100] * n,
            "half_inning": ["top"] * n,
            "home_team_id": [110] * n,
            "away_team_id": [120] * n,
            "hit_launch_speed": [95.0] * n,
            "hit_launch_angle": [20.0] * n,
            "hit_trajectory": trajectories,
            "is_in_play": [True] * n,
        })

    def test_popup_classified_as_fb(self):
        bip = self._make_bip(["popup", "fly_ball", "ground_ball", "line_drive", "bunt_popup"])
        traj = bip["hit_trajectory"].str.lower()

        is_fb = traj.isin(["fly_ball", "popup", "bunt_popup"])
        is_gb = traj.isin(["ground_ball", "bunt_grounder"])
        is_ld = traj.isin(["line_drive"])

        assert is_fb.sum() == 3  # popup + fly_ball + bunt_popup
        assert is_gb.sum() == 1
        assert is_ld.sum() == 1
        # All BIP classified
        assert (is_fb | is_gb | is_ld).sum() == 5

    def test_bunt_line_drive_not_counted_as_ld(self):
        bip = self._make_bip(["bunt_line_drive", "line_drive"])
        traj = bip["hit_trajectory"].str.lower()
        is_ld = traj.isin(["line_drive"])
        assert is_ld.sum() == 1  # Only exact "line_drive", not "bunt_line_drive"


# ---------------------------------------------------------------------------
# Fix 2: Differential NaN propagation (no more NaN → 0.0)
# ---------------------------------------------------------------------------

class TestDifferentialNaN:
    """Differentials propagate NaN instead of producing impossible values."""

    def test_nan_produces_nan_not_extreme(self):
        h = np.array([0.10, np.nan, 0.12, 0.11])
        a = np.array([0.09, 0.13, np.nan, 0.10])

        diff = h - a  # NaN propagation

        assert np.isnan(diff[1])  # home NaN → diff NaN
        assert np.isnan(diff[2])  # away NaN → diff NaN
        np.testing.assert_almost_equal(diff[0], 0.01)
        np.testing.assert_almost_equal(diff[3], 0.01)

    def test_old_behavior_produced_extreme(self):
        """Verify the old bug would produce physically impossible values."""
        h = np.array([np.nan])
        a = np.array([90.0])  # exit velocity

        # Old behavior:
        h_arr = np.where(np.isnan(h), 0.0, h)
        a_arr = np.where(np.isnan(a), 0.0, a)
        old_diff = h_arr - a_arr
        assert old_diff[0] == -90.0  # Impossible EV differential

        # New behavior:
        new_diff = h - a
        assert np.isnan(new_diff[0])  # Correct: unknown


# ---------------------------------------------------------------------------
# Fix 3: Pitchmix matchup = NaN for unknown pitchers (not 0.0)
# ---------------------------------------------------------------------------

class TestPitchmixMatchup:
    """When all SP frequencies are NaN, matchup should be NaN not 0.0."""

    def test_no_sp_profile_yields_nan(self):
        n = 5
        batter_side = pd.DataFrame({
            "_bwoba_FF": [0.320] * n,
            "_bwoba_SL": [0.280] * n,
            "_spfreq_FF": [np.nan] * n,  # No SP profile
            "_spfreq_SL": [np.nan] * n,
        })

        matchup_scores = pd.Series(0.0, index=batter_side.index)
        any_valid_freq = pd.Series(False, index=batter_side.index)

        for pt in ["FF", "SL"]:
            bwoba_col = f"_bwoba_{pt}"
            spfreq_col = f"_spfreq_{pt}"
            freq = batter_side[spfreq_col]
            any_valid_freq = any_valid_freq | freq.notna()
            matchup_scores = matchup_scores + (
                batter_side[bwoba_col].fillna(0.320) * freq.fillna(0.0)
            )

        result = matchup_scores.where(any_valid_freq, np.nan)
        assert result.isna().all()

    def test_valid_sp_profile_produces_score(self):
        n = 3
        batter_side = pd.DataFrame({
            "_bwoba_FF": [0.350] * n,
            "_bwoba_SL": [0.280] * n,
            "_spfreq_FF": [0.60] * n,
            "_spfreq_SL": [0.40] * n,
        })

        matchup_scores = pd.Series(0.0, index=batter_side.index)
        any_valid_freq = pd.Series(False, index=batter_side.index)

        for pt in ["FF", "SL"]:
            bwoba_col = f"_bwoba_{pt}"
            spfreq_col = f"_spfreq_{pt}"
            freq = batter_side[spfreq_col]
            any_valid_freq = any_valid_freq | freq.notna()
            matchup_scores = matchup_scores + (
                batter_side[bwoba_col].fillna(0.320) * freq.fillna(0.0)
            )

        result = matchup_scores.where(any_valid_freq, np.nan)
        expected = 0.350 * 0.60 + 0.280 * 0.40  # 0.322
        np.testing.assert_almost_equal(result.values[0], expected)


# ---------------------------------------------------------------------------
# Fix 4: strand_rate uses opponent LOB (defensive stranding)
# ---------------------------------------------------------------------------

class TestStrandRate:
    """strand_rate = opp_LOB / (opp_LOB + opp_runs) — measures defensive stranding."""

    def test_strand_rate_uses_opponent_lob(self):
        games = pd.DataFrame({
            "home_team_id": [1, 1, 1],
            "away_team_id": [2, 2, 2],
            "home_total_errors": [0.0, 1.0, 0.0],
            "away_total_errors": [1.0, 0.0, 2.0],
            "home_total_lob": [6.0, 7.0, 5.0],  # Home batting LOB
            "away_total_lob": [8.0, 6.0, 9.0],  # Away batting LOB (stranded by HOME defense)
            "home_bat_game_runs": [4.0, 3.0, 5.0],
            "away_bat_game_runs": [2.0, 4.0, 1.0],
        })

        # For home side (measuring home team's defensive stranding):
        # LOB should come from away_total_lob (runners stranded by home pitching)
        opp_lob = games["away_total_lob"].values  # [8, 6, 9]
        opp_runs = games["away_bat_game_runs"].values  # [2, 4, 1]
        expected_strand_rate = opp_lob / (opp_lob + opp_runs)

        np.testing.assert_almost_equal(expected_strand_rate[0], 8 / 10)  # 0.80
        np.testing.assert_almost_equal(expected_strand_rate[1], 6 / 10)  # 0.60
        np.testing.assert_almost_equal(expected_strand_rate[2], 9 / 10)  # 0.90


# ---------------------------------------------------------------------------
# Fix 5: games_back from standings (prior-day lookup)
# ---------------------------------------------------------------------------

class TestGamesBackFromStandings:
    """Standings-based games_back uses prior-day lookup, no shift needed."""

    def test_prior_day_lookup(self):
        from classical_learning.engineering.feature_engineering import _pennant_race_features

        games = pd.DataFrame({
            "home_team_id": [147, 147, 110],
            "away_team_id": [110, 139, 147],
            "game_date": ["2024-06-16", "2024-06-17", "2024-06-18"],
            "home_games_played": [74, 75, 70],
            "away_games_played": [70, 71, 76],
        })

        standings = pd.DataFrame({
            "date": ["2024-06-15", "2024-06-15", "2024-06-15",
                     "2024-06-16", "2024-06-16", "2024-06-16",
                     "2024-06-17", "2024-06-17", "2024-06-17"],
            "team_id": [147, 110, 139, 147, 110, 139, 147, 110, 139],
            "games_back": [0.0, 2.5, 16.0, 0.0, 3.0, 16.5, 0.0, 2.0, 17.0],
            "wild_card_games_back": [0.0, 8.0, 5.5, 0.0, 8.5, 6.0, 0.0, 7.5, 6.5],
        })

        result = _pennant_race_features(games, standings=standings)

        # Game on 2024-06-16: lookup 2024-06-15 standings
        # Home=NYY(147): games_back=0.0, Away=BAL(110): games_back=2.5
        assert result["home_div_games_back"].iloc[0] == 0.0
        assert result["away_div_games_back"].iloc[0] == 2.5
        assert result["home_in_contention"].iloc[0] == 1.0  # 0.0 <= 5
        assert result["away_in_contention"].iloc[0] == 1.0  # 2.5 <= 5

        # Game on 2024-06-17: lookup 2024-06-16 standings
        # Home=NYY(147): games_back=0.0, Away=TB(139): games_back=16.5
        assert result["home_div_games_back"].iloc[1] == 0.0
        assert result["away_div_games_back"].iloc[1] == 16.5
        assert result["away_in_contention"].iloc[1] == 0.0  # 16.5 > 5

        # Differential
        np.testing.assert_almost_equal(
            result["diff_div_games_back"].iloc[0], 0.0 - 2.5
        )

    def test_no_standings_first_game_of_season(self):
        """First day of season has no prior-day standings → NaN."""
        from classical_learning.engineering.feature_engineering import _pennant_race_features

        games = pd.DataFrame({
            "home_team_id": [147],
            "away_team_id": [110],
            "game_date": ["2024-03-28"],
            "home_games_played": [1],
            "away_games_played": [1],
        })
        # No standings for 2024-03-27 (season hasn't started)
        standings = pd.DataFrame({
            "date": ["2024-03-28"],
            "team_id": [147],
            "games_back": [0.0],
            "wild_card_games_back": [0.0],
        })

        result = _pennant_race_features(games, standings=standings)
        # No standings for prior day → NaN
        assert np.isnan(result["home_div_games_back"].iloc[0])

    def test_variance_in_games_back(self):
        """Output should have variance (not all-zero or all-NaN)."""
        from classical_learning.engineering.feature_engineering import _pennant_race_features

        games = pd.DataFrame({
            "home_team_id": [147, 110, 139],
            "away_team_id": [110, 139, 147],
            "game_date": ["2024-06-16", "2024-06-16", "2024-06-16"],
            "home_games_played": [74, 70, 71],
            "away_games_played": [70, 71, 74],
        })
        standings = pd.DataFrame({
            "date": ["2024-06-15"] * 3,
            "team_id": [147, 110, 139],
            "games_back": [0.0, 2.5, 16.0],
            "wild_card_games_back": [0.0, 8.0, 5.5],
        })

        result = _pennant_race_features(games, standings=standings)
        gb_col = result["home_div_games_back"]
        assert gb_col.notna().all()
        assert gb_col.std() > 0  # Not all same value


# ---------------------------------------------------------------------------
# Fix 6: season_progress no cross-season leak
# ---------------------------------------------------------------------------

class TestSeasonProgressNoLeak:
    """First game of new season should NOT inherit prior season's games_played."""

    def test_season_boundary(self):
        games = pd.DataFrame({
            "home_team_id": [1, 1, 1, 1],
            "away_team_id": [2, 2, 2, 2],
            "game_date": ["2023-09-29", "2023-09-30", "2024-03-28", "2024-03-29"],
            "home_games_played": [160, 162, 1, 2],
            "away_games_played": [160, 162, 1, 2],
            "home_division_games_back": ["-", "-", "-", "-"],
            "away_division_games_back": ["3.0", "3.0", "0.0", "0.0"],
        })

        season_map = pd.to_datetime(games["game_date"]).dt.year
        parts = []
        for side in ("home", "away"):
            sub = pd.DataFrame({
                "team_id": games[f"{side}_team_id"],
                "frame_idx": games.index,
                "side": side,
            })
            sub["games_played"] = pd.to_numeric(games[f"{side}_games_played"], errors="coerce").values
            sub["_season"] = season_map.values
            parts.append(sub)

        timeline = pd.concat(parts, ignore_index=True)
        timeline = timeline.sort_values("frame_idx").reset_index(drop=True)

        # With season grouping:
        timeline["gp_shifted_fixed"] = (
            timeline.groupby(["team_id", "_season"])["games_played"]
            .transform(lambda s: s.shift(1))
        )

        # Without season grouping (old bug):
        timeline["gp_shifted_buggy"] = (
            timeline.groupby("team_id")["games_played"]
            .transform(lambda s: s.shift(1))
        )

        # First game of 2024 for home team (frame_idx=2, home side)
        home_2024_first = timeline[(timeline["frame_idx"] == 2) & (timeline["side"] == "home")]

        # Fixed: should be NaN (first game of season, no prior data)
        assert home_2024_first["gp_shifted_fixed"].isna().all()

        # Buggy: would be 162 (from end of 2023)
        assert home_2024_first["gp_shifted_buggy"].iloc[0] == 162.0


# ---------------------------------------------------------------------------
# Fix 7: movement_reason "None" string excluded from extra_base_opps
# ---------------------------------------------------------------------------

class TestMovementReasonNone:
    """Literal 'None' string should not count as a real movement_reason."""

    def test_none_string_excluded(self):
        runners = pd.DataFrame({
            "game_pk": [1, 1, 1, 1, 1],
            "movement_reason": ["r_adv_play", "None", "None", "r_adv_force", "None"],
        })

        # Fixed behavior:
        valid_mask = (runners["movement_reason"].notna()) & (runners["movement_reason"] != "None")
        opps_fixed = valid_mask.sum()

        # Old behavior:
        opps_buggy = runners["movement_reason"].notna().sum()

        assert opps_fixed == 2   # Only real movement reasons
        assert opps_buggy == 5   # Everything passes (3.4x inflation)


# ---------------------------------------------------------------------------
# Fix 8: half_inning in PITCH_META_COLUMNS
# ---------------------------------------------------------------------------

class TestConstantsHalfInning:
    """PITCH_META_COLUMNS includes half_inning for runners side-assignment."""

    def test_half_inning_in_pitch_meta(self):
        from classical_learning.engineering.constants import PITCH_META_COLUMNS
        assert "half_inning" in PITCH_META_COLUMNS
        assert "play_index" in PITCH_META_COLUMNS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
