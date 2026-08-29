"""Tests for pregame/trading/synthetic.py — synthetic feature row construction."""

import numpy as np
import pandas as pd
import pytest

from classical_learning.trading.synthetic import (
    build_synthetic_row,
    _find_latest_team_row,
    _is_team_feature,
    _extract_side_features,
    _recompute_derived,
)


@pytest.fixture
def mini_features():
    """Minimal feature DataFrame with 4 games for testing."""
    return pd.DataFrame([
        # NYY home vs BOS, June 1
        {
            "game_pk": 1001, "game_date": "2026-06-01",
            "home_team_abbr": "NYY", "away_team_abbr": "BOS",
            "home_team_id": 147, "away_team_id": 111,
            "home_elo": 1520.0, "away_elo": 1510.0,
            "home_ewma_avg": 0.260, "away_ewma_avg": 0.250,
            "home_roll10_avg": 0.255, "away_roll10_avg": 0.248,
            "home_all_ewma_era": 3.80, "away_all_ewma_era": 4.10,
            "home_win_streak": 3, "away_win_streak": -1,
            "home_days_rest": 1, "away_days_rest": 2,
            "home_srs": 1.5, "away_srs": 0.8,
            "home_wolfe": 0.55, "away_wolfe": 0.52,
            "home_pythag_1st": 0.56, "away_pythag_1st": 0.51,
            "home_pythag_2nd": 0.54, "away_pythag_2nd": 0.50,
            "home_bsr_offense": 4.5, "away_bsr_offense": 4.2,
            "home_bsr_defense": 3.8, "away_bsr_defense": 4.0,
            "sp_home_season_era": 3.50, "sp_away_season_era": 4.20,
            "sp_home_id": 9001, "sp_away_id": 9002,
            "probable_pitcher_home_id": 9001, "probable_pitcher_away_id": 9002,
            "home_league_id": 103, "away_league_id": 103,
            "home_division_id": 201, "away_division_id": 201,
            "venue_id": 3313,
        },
        # NYY away at TOR, June 15 (NYY more recent, but as AWAY)
        {
            "game_pk": 1002, "game_date": "2026-06-15",
            "home_team_abbr": "TOR", "away_team_abbr": "NYY",
            "home_team_id": 141, "away_team_id": 147,
            "home_elo": 1480.0, "away_elo": 1530.0,
            "home_ewma_avg": 0.245, "away_ewma_avg": 0.265,
            "home_roll10_avg": 0.242, "away_roll10_avg": 0.260,
            "home_all_ewma_era": 4.20, "away_all_ewma_era": 3.70,
            "home_win_streak": -2, "away_win_streak": 5,
            "home_days_rest": 1, "away_days_rest": 1,
            "home_srs": 0.3, "away_srs": 1.8,
            "home_wolfe": 0.48, "away_wolfe": 0.57,
            "home_pythag_1st": 0.49, "away_pythag_1st": 0.58,
            "home_pythag_2nd": 0.47, "away_pythag_2nd": 0.56,
            "home_bsr_offense": 4.0, "away_bsr_offense": 4.7,
            "home_bsr_defense": 4.3, "away_bsr_defense": 3.6,
            "sp_home_season_era": 4.50, "sp_away_season_era": 3.20,
            "sp_home_id": 9003, "sp_away_id": 9001,
            "probable_pitcher_home_id": 9003, "probable_pitcher_away_id": 9001,
            "home_league_id": 103, "away_league_id": 103,
            "home_division_id": 202, "away_division_id": 201,
            "venue_id": 14,
        },
        # BOS away at TB, June 20 (BOS more recent, as AWAY)
        {
            "game_pk": 1003, "game_date": "2026-06-20",
            "home_team_abbr": "TB", "away_team_abbr": "BOS",
            "home_team_id": 139, "away_team_id": 111,
            "home_elo": 1470.0, "away_elo": 1515.0,
            "home_ewma_avg": 0.240, "away_ewma_avg": 0.258,
            "home_roll10_avg": 0.238, "away_roll10_avg": 0.255,
            "home_all_ewma_era": 4.50, "away_all_ewma_era": 3.95,
            "home_win_streak": -3, "away_win_streak": 2,
            "home_days_rest": 2, "away_days_rest": 1,
            "home_srs": -0.5, "away_srs": 1.2,
            "home_wolfe": 0.46, "away_wolfe": 0.54,
            "home_pythag_1st": 0.47, "away_pythag_1st": 0.53,
            "home_pythag_2nd": 0.46, "away_pythag_2nd": 0.52,
            "home_bsr_offense": 3.8, "away_bsr_offense": 4.4,
            "home_bsr_defense": 4.5, "away_bsr_defense": 3.9,
            "sp_home_season_era": 5.00, "sp_away_season_era": 3.80,
            "sp_home_id": 9004, "sp_away_id": 9002,
            "probable_pitcher_home_id": 9004, "probable_pitcher_away_id": 9002,
            "home_league_id": 103, "away_league_id": 103,
            "home_division_id": 202, "away_division_id": 201,
            "venue_id": 12,
        },
        # NYY home vs CLE, June 25 (NYY most recent HOME game)
        {
            "game_pk": 1004, "game_date": "2026-06-25",
            "home_team_abbr": "NYY", "away_team_abbr": "CLE",
            "home_team_id": 147, "away_team_id": 114,
            "home_elo": 1540.0, "away_elo": 1505.0,
            "home_ewma_avg": 0.270, "away_ewma_avg": 0.252,
            "home_roll10_avg": 0.268, "away_roll10_avg": 0.249,
            "home_all_ewma_era": 3.60, "away_all_ewma_era": 4.00,
            "home_win_streak": 4, "away_win_streak": -1,
            "home_days_rest": 1, "away_days_rest": 2,
            "home_srs": 2.0, "away_srs": 0.5,
            "home_wolfe": 0.58, "away_wolfe": 0.50,
            "home_pythag_1st": 0.59, "away_pythag_1st": 0.50,
            "home_pythag_2nd": 0.57, "away_pythag_2nd": 0.49,
            "home_bsr_offense": 4.9, "away_bsr_offense": 4.1,
            "home_bsr_defense": 3.5, "away_bsr_defense": 4.2,
            "sp_home_season_era": 3.10, "sp_away_season_era": 4.00,
            "sp_home_id": 9005, "sp_away_id": 9006,
            "probable_pitcher_home_id": 9005, "probable_pitcher_away_id": 9006,
            "home_league_id": 103, "away_league_id": 103,
            "home_division_id": 201, "away_division_id": 202,
            "venue_id": 3313,
        },
    ])


class TestFindLatestTeamRow:
    def test_find_any_side(self, mini_features):
        row = _find_latest_team_row("NYY", mini_features)
        assert row["game_date"] == "2026-06-25"

    def test_find_home_side(self, mini_features):
        row = _find_latest_team_row("NYY", mini_features, side="home")
        assert row["game_date"] == "2026-06-25"
        assert row["home_team_abbr"] == "NYY"

    def test_find_away_side(self, mini_features):
        row = _find_latest_team_row("NYY", mini_features, side="away")
        assert row["game_date"] == "2026-06-15"
        assert row["away_team_abbr"] == "NYY"

    def test_find_bos_away(self, mini_features):
        row = _find_latest_team_row("BOS", mini_features, side="away")
        assert row["game_date"] == "2026-06-20"

    def test_unknown_team_returns_none(self, mini_features):
        assert _find_latest_team_row("XXX", mini_features) is None


class TestIsTeamFeature:
    def test_rolling_stats(self):
        assert _is_team_feature("roll5_avg") is True
        assert _is_team_feature("roll10_era") is True
        assert _is_team_feature("roll20_whip") is True

    def test_ewma(self):
        assert _is_team_feature("ewma_avg") is True
        assert _is_team_feature("all_ewma_era") is True

    def test_ratings(self):
        assert _is_team_feature("elo") is True
        assert _is_team_feature("srs") is True
        assert _is_team_feature("wolfe") is True
        assert _is_team_feature("pythag_1st") is True

    def test_momentum(self):
        assert _is_team_feature("win_streak") is True
        assert _is_team_feature("days_rest") is True
        assert _is_team_feature("games_last_7d") is True

    def test_raw_game_stats_excluded(self):
        assert _is_team_feature("bat_game_ab") is False
        assert _is_team_feature("pit_game_so") is False
        assert _is_team_feature("team_id") is False
        assert _is_team_feature("team_abbr") is False


class TestBuildSyntheticRow:
    def test_uses_fresh_data_not_stale_matchup(self, mini_features):
        """When building BOS@NYY, should use NYY's June 25 data (vs CLE),
        not their June 1 data (last time NYY hosted BOS)."""
        row = build_synthetic_row("BOS", "NYY", mini_features)
        assert row is not None

        # NYY's home features should come from their June 25 game (vs CLE)
        assert row["home_elo"].iloc[0] == pytest.approx(1540.0)
        assert row["home_ewma_avg"].iloc[0] == pytest.approx(0.270)
        assert row["home_win_streak"].iloc[0] == 4

        # BOS's away features should come from their June 20 game (vs TB)
        assert row["away_elo"].iloc[0] == pytest.approx(1515.0)
        assert row["away_ewma_avg"].iloc[0] == pytest.approx(0.258)
        assert row["away_win_streak"].iloc[0] == 2

    def test_derived_features_recomputed(self, mini_features):
        row = build_synthetic_row("BOS", "NYY", mini_features)
        assert row is not None

        # elo_diff = home - away
        assert row["elo_diff"].iloc[0] == pytest.approx(1540.0 - 1515.0)
        # srs_diff
        assert row["srs_diff"].iloc[0] == pytest.approx(2.0 - 1.2)
        # sp_era_diff = away - home (convention in feature_engineering.py)
        assert row["sp_era_diff"].iloc[0] == pytest.approx(3.80 - 3.10)

    def test_elo_prob_includes_home_advantage(self, mini_features):
        row = build_synthetic_row("BOS", "NYY", mini_features)
        h_elo, a_elo = 1540.0, 1515.0
        expected = 1.0 / (1.0 + 10.0 ** ((a_elo - (h_elo + 24.0)) / 400.0))
        assert row["elo_prob"].iloc[0] == pytest.approx(expected, rel=1e-4)

    def test_unknown_team_returns_none(self, mini_features):
        assert build_synthetic_row("XXX", "NYY", mini_features) is None
        assert build_synthetic_row("BOS", "XXX", mini_features) is None

    def test_game_info_fills_context(self, mini_features):
        game_info = {
            "venue_id": 3313,
            "day_night": "night",
            "game_number": 1,
            "probable_pitcher_home_id": 9005,
            "probable_pitcher_away_id": 9002,
            "home_league_id": 103,
            "away_league_id": 103,
            "home_division_id": 201,
            "away_division_id": 201,
        }
        row = build_synthetic_row("BOS", "NYY", mini_features, game_info=game_info)
        assert row is not None
        assert row["is_night_game"].iloc[0] == 1.0
        assert row["is_doubleheader"].iloc[0] == 0.0
        assert row["is_same_league"].iloc[0] == 1.0
        assert row["is_same_division"].iloc[0] == 1.0

    def test_sp_features_from_pitcher_last_start(self, mini_features):
        """SP features should come from the specific pitcher's last start."""
        game_info = {
            "venue_id": 3313,
            "probable_pitcher_home_id": 9001,  # NYY SP from game 1001 and 1002
            "probable_pitcher_away_id": 9002,  # BOS SP from game 1001 and 1003
        }
        row = build_synthetic_row("BOS", "NYY", mini_features, game_info=game_info)
        assert row is not None
        # SP 9001 last started in game 1002 (as away) with ERA 3.20
        assert row["sp_home_season_era"].iloc[0] == pytest.approx(3.20)
        # SP 9002 last started in game 1003 (as away) with ERA 3.80
        assert row["sp_away_season_era"].iloc[0] == pytest.approx(3.80)
