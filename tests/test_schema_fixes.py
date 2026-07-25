"""Tests for three confirmed S3 schema mismatches (written before fixes).

Bug 1: _extract_game_metadata omits home/away league_id and division_id from
       meta_cols, so _schedule_context always falls back to NaN.
       Raw S3 parquet HAS those columns populated.

Bug 2: pitch_level_features.py was filtering on `at_bat_event` (title-case in
       S3: "Strikeout", "Walk") but frozensets use snake_case (from `event_type`
       column: "strikeout", "walk"). Fix: switch all event filtering to
       `event_type`, which is snake_case and already in PITCH_LEVEL_COLUMNS.

Bug 3: PITCH_LEVEL_COLUMNS listed "inning_half" but S3 column is "half_inning".
       pitch_level_features.py filtered on "inning_half" which is always absent
       → home/away wOBA and pitchmix matchup score all NaN.

Run: conda run -n pred python -m pytest tests/test_schema_fixes.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Bug 1: game_builder._extract_game_metadata drops league/division columns
# ---------------------------------------------------------------------------

class TestLeagueDivisionPassthrough:
    """_extract_game_metadata must pass through home/away league_id and
    division_id so that _schedule_context can compute is_same_league /
    is_same_division instead of producing NaN."""

    def _make_pitches(self):
        return pd.DataFrame({
            "game_pk": [1001, 1001, 1002, 1002],
            "game_date": ["2024-06-15"] * 4,
            "game_datetime_utc": ["2024-06-15T18:00:00Z"] * 4,
            "home_team_id": [147, 147, 112, 112],
            "away_team_id": [111, 111, 158, 158],
            "home_team_name": ["Yankees", "Yankees", "Cubs", "Cubs"],
            "away_team_name": ["Red Sox", "Red Sox", "Brewers", "Brewers"],
            "home_team_abbr": ["NYY", "NYY", "CHC", "CHC"],
            "away_team_abbr": ["BOS", "BOS", "MIL", "MIL"],
            "home_league_id": [103, 103, 104, 104],
            "away_league_id": [103, 103, 104, 104],
            "home_division_id": [201, 201, 205, 205],
            "away_division_id": [201, 201, 205, 205],
            "venue_id": [3313, 3313, 17, 17],
            "venue_name": ["Yankee Stadium"] * 2 + ["Wrigley Field"] * 2,
            "venue_latitude": [40.83] * 4,
            "venue_longitude": [-73.93] * 4,
            "venue_capacity": [54251] * 4,
            "venue_roof_type": ["outdoor"] * 4,
            "weather_condition": ["Clear"] * 4,
            "weather_temp": [75.0] * 4,
            "weather_wind": ["5 mph"] * 4,
            "day_night": ["night"] * 4,
            "attendance": [45000.0] * 4,
            "probable_pitcher_home_id": [123] * 4,
            "probable_pitcher_away_id": [456] * 4,
            "umpire_hp": ["Joe West"] * 4,
            "game_type_code": ["R"] * 4,
            "double_header": ["N"] * 4,
            "game_number": [1] * 4,
        })

    def test_extract_game_metadata_includes_league_columns(self):
        """_extract_game_metadata must include home_league_id and away_league_id."""
        from pregame.engineering.game_builder import _extract_game_metadata

        pitches = self._make_pitches()
        meta = _extract_game_metadata(pitches)

        assert "home_league_id" in meta.columns, (
            "home_league_id missing from game_meta — _schedule_context will NaN out"
        )
        assert "away_league_id" in meta.columns, (
            "away_league_id missing from game_meta"
        )

    def test_extract_game_metadata_includes_division_columns(self):
        """_extract_game_metadata must include home_division_id and away_division_id."""
        from pregame.engineering.game_builder import _extract_game_metadata

        pitches = self._make_pitches()
        meta = _extract_game_metadata(pitches)

        assert "home_division_id" in meta.columns, (
            "home_division_id missing from game_meta"
        )
        assert "away_division_id" in meta.columns, (
            "away_division_id missing from game_meta"
        )

    def test_extract_game_metadata_league_values_non_null(self):
        """League IDs must be non-null in the extracted metadata."""
        from pregame.engineering.game_builder import _extract_game_metadata

        pitches = self._make_pitches()
        meta = _extract_game_metadata(pitches)

        assert meta["home_league_id"].notna().all(), (
            "home_league_id has unexpected NaN in extracted metadata"
        )
        assert meta["away_league_id"].notna().all(), (
            "away_league_id has unexpected NaN in extracted metadata"
        )

    def test_schedule_context_produces_non_null_flags_when_ids_present(self):
        """When the game_frame has populated league/division IDs, is_same_league
        and is_same_division must not be NaN (regression gate)."""
        from pregame.engineering.feature_engineering import _schedule_context

        game_frame = pd.DataFrame({
            "game_pk": [1001, 1002, 1003],
            "game_date": pd.to_datetime(["2024-06-15"] * 3),
            "home_team_id": [147, 112, 147],
            "away_team_id": [111, 158, 119],
            "home_league_id": [103, 104, 103],
            "away_league_id": [103, 104, 104],
            "home_division_id": [201, 205, 201],
            "away_division_id": [201, 205, 206],
            "elo_prob": [0.55, 0.48, 0.62],
            "consensus_home_win_prob": [0.54, 0.47, 0.61],
        })
        result = _schedule_context(game_frame)

        assert result["is_same_league"].notna().all(), (
            "is_same_league is NaN even though league_id columns are present"
        )
        assert result["is_same_division"].notna().all(), (
            "is_same_division is NaN even though division_id columns are present"
        )
        # Spot checks: NYY vs BOS = same division
        assert result.loc[result["game_pk"] == 1001, "is_same_division"].iloc[0] == 1.0
        # CHC vs MIL = same division
        assert result.loc[result["game_pk"] == 1002, "is_same_division"].iloc[0] == 1.0
        # NYY vs LAD = interleague
        assert result.loc[result["game_pk"] == 1003, "is_same_league"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Bug 2: at_bat_event is title-case in S3 ("Strikeout" not "strikeout")
# ---------------------------------------------------------------------------

def _build_pitches_snake_case(n_history_games: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build pitch and game fixtures with snake_case event_type values
    (matching the actual S3 event_type column format) and half_inning column,
    with enough history for rolling windows to produce non-NaN values."""
    HOME_TEAM = 147
    AWAY_TEAM = 111
    PITCHER_H = 1001
    PITCHER_A = 2001
    BATTER_H1, BATTER_H2 = 3001, 3002
    BATTER_A1, BATTER_A2 = 4001, 4002

    pitch_rows = []
    game_rows = []

    for g in range(n_history_games):
        gp = 9000 + g
        gd = f"2024-{(g // 28) + 4:02d}-{(g % 28) + 1:02d}"
        game_rows.append({
            "game_pk": gp,
            "game_date": gd,
            "season": 2024,
            "game_type_code": "R",
            "home_team_id": HOME_TEAM,
            "away_team_id": AWAY_TEAM,
            "probable_pitcher_home_id": PITCHER_H,
            "probable_pitcher_away_id": PITCHER_A,
        })

        # Away pitcher (PITCHER_A) faces home batters in bottom half
        for ab_idx, (batter, event) in enumerate([
            (BATTER_H1, "strikeout"),
            (BATTER_H2, "walk"),
            (BATTER_H1, "single"),
            (BATTER_H2, "home_run"),
            (BATTER_H1, "strikeout"),
            (BATTER_H2, "field_out"),
        ]):
            # Terminal pitch for at-bat
            pitch_rows.append({
                "game_pk": gp,
                "season": 2024,
                "game_date": gd,
                "game_type_code": "R",
                "home_team_id": HOME_TEAM,
                "away_team_id": AWAY_TEAM,
                "pitcher_id": PITCHER_A,
                "batter_id": batter,
                "is_pitch": True,
                "release_speed": 93.0,
                "coord_x0": -1.5,
                "coord_z0": 5.8,
                "pitch_type": "FF",
                "bat_side_code": "R",
                "pitch_hand_code": "R",
                "event_type": event,
                "at_bat_index": ab_idx,
                "pitch_number": 3,
                "inning": 1,
                "half_inning": "bottom",
                "cum_outs": ab_idx % 3,
                "pre_on_first_id": None,
                "pre_on_second_id": None,
                "pre_on_third_id": None,
            })

        # Home pitcher (PITCHER_H) faces away batters in top half
        for ab_idx, (batter, event) in enumerate([
            (BATTER_A1, "strikeout"),
            (BATTER_A2, "walk"),
            (BATTER_A1, "single"),
            (BATTER_A2, "field_out"),
        ]):
            pitch_rows.append({
                "game_pk": gp,
                "season": 2024,
                "game_date": gd,
                "game_type_code": "R",
                "home_team_id": HOME_TEAM,
                "away_team_id": AWAY_TEAM,
                "pitcher_id": PITCHER_H,
                "batter_id": batter,
                "is_pitch": True,
                "release_speed": 91.0,
                "coord_x0": -1.2,
                "coord_z0": 5.5,
                "pitch_type": "FF",
                "bat_side_code": "L",
                "pitch_hand_code": "R",
                "event_type": event,
                "at_bat_index": ab_idx,
                "pitch_number": 3,
                "inning": 1,
                "half_inning": "top",
                "cum_outs": ab_idx % 3,
                "pre_on_first_id": None,
                "pre_on_second_id": None,
                "pre_on_third_id": None,
            })

    pitches = pd.DataFrame(pitch_rows)
    pitches["game_date"] = pd.to_datetime(pitches["game_date"])
    games = pd.DataFrame(game_rows)
    games["game_date"] = pd.to_datetime(games["game_date"])
    return pitches, games


class TestEventTypeColumn:
    """pitch_level_features.py must use event_type (snake_case) not at_bat_event.

    Real S3 data: event_type='strikeout', 'walk', 'home_run' (snake_case).
    at_bat_event='Strikeout', 'Walk', 'Home Run' (title-case, human-readable).
    Fix: all event filtering uses event_type, which already has the right format.
    """

    def test_kpct_features_non_null_with_event_type(self):
        """K% features must be non-NaN when using event_type column."""
        from pregame.engineering.pitch_level_features import compute_pitch_level_features

        pitches, games = _build_pitches_snake_case(n_history_games=12)
        result = compute_pitch_level_features(pitches, games)

        kpct_cols = [c for c in result.columns if "kpct" in c]
        assert len(kpct_cols) > 0, "No kpct columns produced"

        last_game = result.iloc[-1]
        non_null = sum(1 for c in kpct_cols if pd.notna(last_game[c]))
        assert non_null > 0, (
            f"All kpct features NaN for last game — check event_type column usage. "
            f"Columns: {kpct_cols}"
        )

    def test_bbpct_features_non_null_with_event_type(self):
        """BB% features must be non-NaN when using event_type column."""
        from pregame.engineering.pitch_level_features import compute_pitch_level_features

        pitches, games = _build_pitches_snake_case(n_history_games=12)
        result = compute_pitch_level_features(pitches, games)

        bbpct_cols = [c for c in result.columns if "bbpct" in c]
        assert len(bbpct_cols) > 0, "No bbpct columns produced"

        last_game = result.iloc[-1]
        non_null = sum(1 for c in bbpct_cols if pd.notna(last_game[c]))
        assert non_null > 0, (
            f"All bbpct features NaN for last game — check event_type column usage. "
            f"Columns: {bbpct_cols}"
        )

    def test_fip_features_non_null_with_event_type(self):
        """FIP features must be non-NaN when using event_type column."""
        from pregame.engineering.pitch_level_features import compute_pitch_level_features

        pitches, games = _build_pitches_snake_case(n_history_games=12)
        result = compute_pitch_level_features(pitches, games)

        fip_cols = [c for c in result.columns if "fip" in c]
        assert len(fip_cols) > 0, "No fip columns produced"

        last_game = result.iloc[-1]
        non_null = sum(1 for c in fip_cols if pd.notna(last_game[c]))
        assert non_null > 0, (
            f"All FIP features NaN for last game — check event_type column usage. "
            f"Columns: {fip_cols}"
        )

    def test_woba_features_non_null_with_event_type(self):
        """Platoon wOBA features must be non-NaN when using event_type column."""
        from pregame.engineering.pitch_level_features import compute_pitch_level_features

        pitches, games = _build_pitches_snake_case(n_history_games=12)
        result = compute_pitch_level_features(pitches, games)

        woba_cols = [c for c in result.columns if "woba" in c]
        assert len(woba_cols) > 0, "No woba columns produced"

        last_game = result.iloc[-1]
        non_null = sum(1 for c in woba_cols if pd.notna(last_game[c]))
        assert non_null > 0, (
            f"All wOBA features NaN for last game — check event_type column usage. "
            f"Columns: {woba_cols}"
        )


# ---------------------------------------------------------------------------
# Bug 3: PITCH_LEVEL_COLUMNS uses "inning_half" but S3 column is "half_inning"
# ---------------------------------------------------------------------------

class TestInningHalfColumnName:
    """PITCH_LEVEL_COLUMNS and pitch_level_features.py must use 'half_inning',
    not 'inning_half'. The actual S3 parquet column is 'half_inning'."""

    def test_pitch_level_columns_contains_half_inning(self):
        """PITCH_LEVEL_COLUMNS must list 'half_inning' (the S3 column name)."""
        from pregame.engineering.constants import PITCH_LEVEL_COLUMNS

        assert "half_inning" in PITCH_LEVEL_COLUMNS, (
            "PITCH_LEVEL_COLUMNS uses 'inning_half' but S3 parquet has 'half_inning'. "
            "This causes the column to always be absent at load time."
        )

    def test_pitch_level_columns_does_not_contain_inning_half(self):
        """'inning_half' must not appear in PITCH_LEVEL_COLUMNS (wrong S3 name)."""
        from pregame.engineering.constants import PITCH_LEVEL_COLUMNS

        assert "inning_half" not in PITCH_LEVEL_COLUMNS, (
            "'inning_half' is present in PITCH_LEVEL_COLUMNS but S3 has 'half_inning'. "
            "Remove the wrong name."
        )

    def test_woba_splits_use_half_inning_for_team_assignment(self):
        """Platoon wOBA home/away team assignment uses 'half_inning' (S3 name)."""
        from pregame.engineering.pitch_level_features import compute_pitch_level_features

        pitches, games = _build_pitches_snake_case(n_history_games=12)
        assert "half_inning" in pitches.columns, (
            "Test fixture must use 'half_inning' (S3 column name)"
        )
        assert "inning_half" not in pitches.columns

        result = compute_pitch_level_features(pitches, games)
        woba_home_cols = [c for c in result.columns if "home_team_woba" in c]
        woba_away_cols = [c for c in result.columns if "away_team_woba" in c]

        # With enough history, at least the final game must have non-NaN team wOBA
        last_game = result.iloc[-1]

        home_non_null = sum(1 for c in woba_home_cols if pd.notna(last_game[c]))
        assert home_non_null > 0, (
            f"All home_team_woba features NaN — code may still reference 'inning_half'. "
            f"Columns: {woba_home_cols}"
        )
        away_non_null = sum(1 for c in woba_away_cols if pd.notna(last_game[c]))
        assert away_non_null > 0, (
            f"All away_team_woba features NaN — code may still reference 'inning_half'. "
            f"Columns: {woba_away_cols}"
        )

    def test_pitchmix_matchup_score_non_null_with_half_inning(self):
        """pitchmix_matchup_score features depend on half_inning for team
        lineup assignment and must be non-NaN after the column-name fix."""
        from pregame.engineering.pitch_level_features import compute_pitch_level_features

        pitches, games = _build_pitches_snake_case(n_history_games=12)
        result = compute_pitch_level_features(pitches, games)

        matchup_cols = [c for c in result.columns if "pitchmix_matchup_score" in c]
        assert len(matchup_cols) > 0, "No pitchmix_matchup_score columns produced"

        last_game = result.iloc[-1]
        non_null = sum(1 for c in matchup_cols if pd.notna(last_game[c]))
        assert non_null > 0, (
            f"All pitchmix_matchup_score features NaN — check half_inning usage. "
            f"Columns: {matchup_cols}"
        )
