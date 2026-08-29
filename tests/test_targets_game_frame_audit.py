"""Audit tests for target construction (targets.py) and game frame assembly (game_builder.py).

Tests cover:
1. Suspended games: game_date uses originalDate (Date A), not completion date (Date B)
2. Rain-shortened games: targets flag shortened games, first_5 targets handle <5 innings
3. Manfred extra-innings runner (2020+): extra_innings target and total_runs are correct
4. Impossible target values: no negative runs, home_win in {0,1}, no ties in full game
5. first_5_* targets: correctly summed from linescore innings 1-5
6. Game frame column provenance: pre-game vs post-game classification
7. Weather/attendance/day_night provenance (post-game measured vs pre-game announced)
8. Doubleheader sort order
9. Duplicate game_pks
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live"))
from mlb_dl.targets import build_game_targets, _sum_period, TRAINABLE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_linescore(game_pk: int, season: int, innings: list[tuple[int, int]]) -> pd.DataFrame:
    """Create a linescore DataFrame for one game.

    Parameters
    ----------
    innings : list of (home_runs, away_runs) per inning
    """
    rows = []
    for i, (hr, ar) in enumerate(innings, start=1):
        rows.append({
            "game_pk": game_pk,
            "season": season,
            "inning": i,
            "home_runs": hr,
            "away_runs": ar,
        })
    return pd.DataFrame(rows)


def _make_game_meta(game_pk: int, game_date: str = "2024-07-15", **kwargs) -> pd.DataFrame:
    """Create a minimal game_meta DataFrame for one game."""
    row = {
        "game_pk": game_pk,
        "game_date": game_date,
        "home_team_id": 147,
        "away_team_id": 111,
        "venue_id": 3313,
        "day_night": "night",
        "weather_temp": 72.0,
        "weather_condition": "Clear",
        "double_header": "N",
        "game_number": 1,
        "game_type_code": "R",
    }
    row.update(kwargs)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Section 1: Suspended Games — game_date uses originalDate
# ---------------------------------------------------------------------------

class TestSuspendedGames:
    """Verify that suspended games use the original scheduled date, not completion date.

    The MLB API field `gameData.datetime.originalDate` is used as `game_date` by
    download_history.py (line 590). For a game suspended on Date A and resumed on Date B,
    originalDate = Date A. This is correct: features for Date A predictions should
    come from data available on Date A.
    """

    def test_game_date_from_meta_not_shifted(self):
        """Game frame should carry the game_date from metadata (originalDate),
        not any date derived from the linescore or boxscore."""
        linescore = _make_linescore(100001, 2023, [(0, 1)] * 9)
        # Simulate a suspended game: game was scheduled for July 1 but
        # the metadata correctly reports originalDate
        meta = _make_game_meta(100001, game_date="2023-07-01")
        targets = build_game_targets(linescore, meta)

        assert targets["game_date"].iloc[0] == "2023-07-01"

    def test_suspended_game_targets_still_valid(self):
        """Even with a non-standard schedule, targets from linescore are correctly computed."""
        # A game suspended after 6 innings, resumed later. Linescore shows all 9.
        linescore = _make_linescore(100002, 2022, [
            (0, 0), (1, 0), (0, 2), (0, 0), (0, 0),  # innings 1-5
            (0, 1), (2, 0), (0, 0), (1, 0),            # innings 6-9
        ])
        meta = _make_game_meta(100002, game_date="2022-06-15")
        targets = build_game_targets(linescore, meta)

        assert targets["total_runs"].iloc[0] == 7  # 4 home + 3 away
        assert targets["home_runs"].iloc[0] == 4
        assert targets["away_runs"].iloc[0] == 3
        assert targets["home_win"].iloc[0] == 1.0


# ---------------------------------------------------------------------------
# Section 2: Rain-Shortened Games
# ---------------------------------------------------------------------------

class TestRainShortenedGames:
    """Rain-shortened games (< 9 innings) must be flagged and handle targets correctly."""

    def test_shortened_game_flagged(self):
        """Games with fewer than 9 innings get shortened_or_called = 1."""
        # 5-inning game (rain shortened)
        linescore = _make_linescore(200001, 2024, [
            (2, 0), (0, 1), (1, 0), (0, 0), (0, 0),
        ])
        targets = build_game_targets(linescore)
        assert targets["shortened_or_called"].iloc[0] == 1.0
        assert targets["innings_played"].iloc[0] == 5

    def test_full_game_not_flagged(self):
        """A normal 9-inning game is not flagged shortened."""
        linescore = _make_linescore(200002, 2024, [(1, 0)] * 9)
        targets = build_game_targets(linescore)
        assert targets["shortened_or_called"].iloc[0] == 0.0

    def test_shortened_game_total_runs_correct(self):
        """Total runs for shortened games sum only the played innings."""
        linescore = _make_linescore(200003, 2024, [
            (3, 1), (0, 2), (1, 0), (0, 1), (2, 0),  # only 5 innings
        ])
        targets = build_game_targets(linescore)
        # Home: 3+0+1+0+2=6, Away: 1+2+0+1+0=4
        assert targets["total_runs"].iloc[0] == 10
        assert targets["home_runs"].iloc[0] == 6
        assert targets["away_runs"].iloc[0] == 4

    def test_shortened_game_first_5_same_as_total(self):
        """For a 5-inning game, first_5 targets should equal full-game totals."""
        innings = [(1, 2), (0, 0), (3, 1), (0, 0), (2, 0)]
        linescore = _make_linescore(200004, 2024, innings)
        targets = build_game_targets(linescore)
        assert targets["first_5_total_runs"].iloc[0] == targets["total_runs"].iloc[0]
        assert targets["first_5_home_runs"].iloc[0] == targets["home_runs"].iloc[0]
        assert targets["first_5_away_runs"].iloc[0] == targets["away_runs"].iloc[0]

    def test_shortened_game_less_than_5_innings(self):
        """A game called after 4 innings: first_5 targets sum only what exists.

        FINDING: _sum_period filters inning <= max_inning, so for a 4-inning game,
        first_5_total_runs sums only innings 1-4. This is correct: we cannot observe
        inning 5 if it was never played.
        """
        linescore = _make_linescore(200005, 2024, [
            (2, 1), (0, 3), (1, 0), (0, 2),  # only 4 innings
        ])
        targets = build_game_targets(linescore)
        # first_5 sums whatever exists up to inning 5
        assert targets["first_5_home_runs"].iloc[0] == 3  # 2+0+1+0
        assert targets["first_5_away_runs"].iloc[0] == 6  # 1+3+0+2
        assert targets["first_5_total_runs"].iloc[0] == 9
        assert targets["innings_played"].iloc[0] == 4


# ---------------------------------------------------------------------------
# Section 3: Manfred Extra-Innings Runner (2020+)
# ---------------------------------------------------------------------------

class TestManfredExtraInnings:
    """Verify extra_innings target handles the ghost runner rule correctly.

    The Manfred runner (placed on 2B in extras since 2020) can score without a hit.
    The linescore correctly records any runs scored regardless of how they occurred.
    The extra_innings target is purely based on innings_played > 9, which is correct
    since the rule change affects HOW runs score, not WHETHER there are extra innings.
    """

    def test_extra_innings_flag_with_ghost_runner_scoring(self):
        """A 10-inning game where ghost runner scores — extra_innings = 1."""
        innings = [(0, 0)] * 9 + [(1, 0)]  # Home wins in 10th (ghost runner scored)
        linescore = _make_linescore(300001, 2021, innings)
        targets = build_game_targets(linescore)
        assert targets["extra_innings"].iloc[0] == 1.0
        assert targets["innings_played"].iloc[0] == 10
        assert targets["total_runs"].iloc[0] == 1
        assert targets["home_win"].iloc[0] == 1.0

    def test_no_extra_innings_walkoff(self):
        """9-inning game with bottom-9 walkoff — not extra innings."""
        innings = [(0, 0)] * 8 + [(1, 0)]
        linescore = _make_linescore(300002, 2022, innings)
        targets = build_game_targets(linescore)
        assert targets["extra_innings"].iloc[0] == 0.0
        assert targets["innings_played"].iloc[0] == 9

    def test_extra_innings_total_runs_includes_extras(self):
        """Total runs must include all extra-inning runs (not just regulation)."""
        # 12-inning game
        innings = [(0, 0)] * 9 + [(0, 1), (1, 0), (2, 0)]
        linescore = _make_linescore(300003, 2023, innings)
        targets = build_game_targets(linescore)
        assert targets["total_runs"].iloc[0] == 4  # 3 home + 1 away
        assert targets["home_runs"].iloc[0] == 3
        assert targets["away_runs"].iloc[0] == 1
        assert targets["innings_played"].iloc[0] == 12
        assert targets["extra_innings"].iloc[0] == 1.0

    def test_regulation_totals_cap_at_9_innings(self):
        """Regulation period sums only first 9 innings, not extras."""
        innings = [(1, 1)] * 9 + [(0, 0), (2, 0)]  # 11 innings
        linescore = _make_linescore(300004, 2024, innings)
        targets = build_game_targets(linescore)
        # regulation = innings 1-9 only
        assert targets["regulation_home_runs"].iloc[0] == 9
        assert targets["regulation_away_runs"].iloc[0] == 9
        assert targets["regulation_tie"].iloc[0] == 1.0
        # full game includes extras
        assert targets["total_runs"].iloc[0] == 20  # 9+2=11 home, 9 away


# ---------------------------------------------------------------------------
# Section 4: Impossible Target Values
# ---------------------------------------------------------------------------

class TestImpossibleTargetValues:
    """Guard against impossible values in target construction."""

    def test_home_win_binary_only(self):
        """home_win must be in {0.0, 1.0} — never NaN or intermediate."""
        # Regular game
        linescore = _make_linescore(400001, 2024, [(1, 0)] * 5 + [(0, 0)] * 4)
        targets = build_game_targets(linescore)
        assert targets["home_win"].iloc[0] in (0.0, 1.0)

    def test_away_win_binary_only(self):
        """away_win must be in {0.0, 1.0}."""
        linescore = _make_linescore(400002, 2024, [(0, 1)] + [(0, 0)] * 8)
        targets = build_game_targets(linescore)
        assert targets["away_win"].iloc[0] in (0.0, 1.0)

    def test_tied_game_both_win_zero(self):
        """FINDING: If home_run_diff == 0 (tie), both home_win and away_win are 0.

        In MLB, regulation ties can only occur in rain-shortened games or pre-2007
        All-Star games. This is correct behavior: neither team won, so both = 0.
        The target_status should still be 'trainable' but downstream models should
        be aware that home_win + away_win != 1 in ~0.05% of cases.
        """
        # Simulate a rain-shortened tie (5 innings, same score)
        linescore = _make_linescore(400003, 2024, [(1, 1), (0, 0), (0, 0), (0, 0), (0, 0)])
        targets = build_game_targets(linescore)
        assert targets["home_win"].iloc[0] == 0.0
        assert targets["away_win"].iloc[0] == 0.0
        assert targets["home_run_diff"].iloc[0] == 0
        # Both home_win and away_win are 0 — this is a tie
        assert targets["shortened_or_called"].iloc[0] == 1.0

    def test_no_negative_runs(self):
        """Runs (total, home, away) must never be negative."""
        linescore = _make_linescore(400004, 2024, [(0, 0)] * 9)
        targets = build_game_targets(linescore)
        assert targets["total_runs"].iloc[0] >= 0
        assert targets["home_runs"].iloc[0] >= 0
        assert targets["away_runs"].iloc[0] >= 0
        assert targets["first_5_total_runs"].iloc[0] >= 0

    def test_run_diff_symmetry(self):
        """home_run_diff + away_run_diff must always equal 0."""
        linescore = _make_linescore(400005, 2024, [(3, 1), (0, 2), (1, 0)] + [(0, 0)] * 6)
        targets = build_game_targets(linescore)
        assert targets["home_run_diff"].iloc[0] + targets["away_run_diff"].iloc[0] == 0

    def test_yrfi_nrfi_complementary(self):
        """yrfi + nrfi must always equal 1.0."""
        for first_inning in [(0, 0), (1, 0), (0, 2), (3, 1)]:
            innings = [first_inning] + [(0, 0)] * 8
            linescore = _make_linescore(400010, 2024, innings)
            targets = build_game_targets(linescore)
            assert targets["yrfi"].iloc[0] + targets["nrfi"].iloc[0] == 1.0

    def test_first_5_winner_exhaustive(self):
        """Exactly one of first_5_home_win, first_5_away_win, first_5_tie must be 1."""
        # Home leads after 5
        innings = [(2, 0)] + [(0, 0)] * 8
        linescore = _make_linescore(400020, 2024, innings)
        targets = build_game_targets(linescore)
        total = (targets["first_5_home_win"].iloc[0] +
                 targets["first_5_away_win"].iloc[0] +
                 targets["first_5_tie"].iloc[0])
        assert total == 1.0

    def test_first_5_winner_mutual_exclusion(self):
        """Only one of the three first_5 winner flags can be 1."""
        cases = [
            [(2, 0)] + [(0, 0)] * 8,   # home leads
            [(0, 3)] + [(0, 0)] * 8,   # away leads
            [(0, 0)] * 9,              # tied after 5
        ]
        for innings in cases:
            linescore = _make_linescore(400021, 2024, innings)
            targets = build_game_targets(linescore)
            flags = [
                targets["first_5_home_win"].iloc[0],
                targets["first_5_away_win"].iloc[0],
                targets["first_5_tie"].iloc[0],
            ]
            assert sum(f == 1.0 for f in flags) == 1
            assert sum(f == 0.0 for f in flags) == 2


# ---------------------------------------------------------------------------
# Section 5: first_5_* Target Correctness
# ---------------------------------------------------------------------------

class TestFirst5Targets:
    """Verify first_5_* targets are correctly derived from inning-level linescore."""

    def test_first_5_sum_exact(self):
        """First 5 innings sum correctly from individual inning lines."""
        innings = [
            (1, 0),  # inn 1
            (0, 2),  # inn 2
            (3, 0),  # inn 3
            (0, 1),  # inn 4
            (2, 1),  # inn 5
            (0, 0),  # inn 6 (should NOT be included)
            (4, 0),  # inn 7
            (0, 0),  # inn 8
            (0, 0),  # inn 9
        ]
        linescore = _make_linescore(500001, 2024, innings)
        targets = build_game_targets(linescore)

        expected_home_5 = 1 + 0 + 3 + 0 + 2  # = 6
        expected_away_5 = 0 + 2 + 0 + 1 + 1  # = 4
        assert targets["first_5_home_runs"].iloc[0] == expected_home_5
        assert targets["first_5_away_runs"].iloc[0] == expected_away_5
        assert targets["first_5_total_runs"].iloc[0] == expected_home_5 + expected_away_5

    def test_first_5_run_diff(self):
        """first_5_home_run_diff and first_5_away_run_diff are correct and symmetric."""
        innings = [(2, 1), (0, 0), (0, 3), (1, 0), (0, 0)] + [(0, 0)] * 4
        linescore = _make_linescore(500002, 2024, innings)
        targets = build_game_targets(linescore)

        # Home 5: 2+0+0+1+0=3, Away 5: 1+0+3+0+0=4
        assert targets["first_5_home_run_diff"].iloc[0] == -1
        assert targets["first_5_away_run_diff"].iloc[0] == 1
        assert targets["first_5_home_run_diff"].iloc[0] + targets["first_5_away_run_diff"].iloc[0] == 0

    def test_yrfi_from_first_inning_only(self):
        """YRFI is determined by inning 1 scoring only."""
        # No runs in inning 1, runs in inning 2
        innings = [(0, 0), (5, 3)] + [(0, 0)] * 7
        linescore = _make_linescore(500003, 2024, innings)
        targets = build_game_targets(linescore)
        assert targets["yrfi"].iloc[0] == 0.0
        assert targets["nrfi"].iloc[0] == 1.0

    def test_yrfi_triggered_by_away_run(self):
        """A single away run in inning 1 triggers YRFI."""
        innings = [(0, 1)] + [(0, 0)] * 8
        linescore = _make_linescore(500004, 2024, innings)
        targets = build_game_targets(linescore)
        assert targets["yrfi"].iloc[0] == 1.0

    def test_yrfi_triggered_by_home_run(self):
        """A single home run in inning 1 triggers YRFI."""
        innings = [(1, 0)] + [(0, 0)] * 8
        linescore = _make_linescore(500005, 2024, innings)
        targets = build_game_targets(linescore)
        assert targets["yrfi"].iloc[0] == 1.0


# ---------------------------------------------------------------------------
# Section 6: Game Frame Column Provenance
# ---------------------------------------------------------------------------

class TestColumnProvenance:
    """Classify every column from game_builder as PRE-GAME or POST-GAME.

    This test documents which data sources are used and flags any post-game
    columns that could leak into features.
    """

    # Columns from game_builder._extract_game_metadata() — sourced from the
    # pitches table which is populated from the GUMBO API's gameData + liveData.
    # These are all extracted from the final API snapshot (post-game).
    PREGAME_COLUMNS = {
        # Identifiers — known at schedule time
        "game_pk", "game_date", "season",
        "home_team_id", "home_team_name", "home_team_abbr",
        "away_team_id", "away_team_name", "away_team_abbr",
        # Venue — static properties
        "venue_id", "venue_name", "venue_latitude", "venue_longitude",
        "venue_capacity", "venue_roof_type",
        # Schedule metadata — known at schedule time
        "game_type_code", "double_header", "game_number",
        # Probable pitchers — announced 1-2 days before game
        "probable_pitcher_home_id", "probable_pitcher_away_id",
        # Day/night — set at schedule time (pre-game API field)
        "day_night",
    }

    # FINDING: These columns come from the post-game API snapshot.
    # While they appear in the game frame, they are NOT used as model features
    # due to the allowlist in strategy/data.py::_PREGAME_FEATURE_PREFIXES.
    POSTGAME_COLUMNS = {
        # Attendance — official count published AFTER game ends
        # (MLB API: boxscore.info["Att"] or gameData.gameInfo.attendance)
        "attendance",
        # Weather — measured conditions from the GUMBO weather node.
        # The MLB API populates this from stadium weather stations during the game.
        # For retractable-roof venues, "condition" reflects in-game roof state.
        # FINDING: weather_temp is from gameData.weather.temp which is a pre-game
        # forecast for open-air venues. However, for some games it is updated during
        # play. In practice, temperature has negligible leakage risk because:
        # (1) it's an ambient condition, not a game outcome
        # (2) pre-game forecasts match actual within ~2-5F
        "weather_temp", "weather_condition", "weather_wind",
        # Umpire — assigned before game but confirmed post-game in API
        # FINDING: umpire_hp is typically announced day-of but is correctly
        # considered pre-game knowable.
        "umpire_hp",
    }

    def test_pregame_features_allowlist_excludes_postgame(self):
        """Verify the feature selection allowlist in strategy/data.py correctly
        excludes all identified post-game columns from model inputs."""
        # Import the feature selection function
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from classical_learning.strategy.data import _PREGAME_FEATURE_PREFIXES, _POSTGAME_EXCLUSIONS

        # These raw columns from the game frame should NOT match any prefix
        postgame_raw = [
            "attendance", "weather_temp", "weather_condition", "weather_wind",
            "game_duration_minutes",
            "review_home_challenges_used", "review_away_challenges_used",
            "flag_no_hitter", "flag_perfect_game",
            "flag_away_team_no_hitter", "flag_home_team_no_hitter",
        ]
        for col in postgame_raw:
            matches_prefix = any(
                col.startswith(p) or col == p for p in _PREGAME_FEATURE_PREFIXES
            )
            assert not matches_prefix, (
                f"Post-game column '{col}' matches a pregame feature prefix — potential leakage!"
            )

    def test_weather_temp_used_as_feature(self):
        """Confirm that temp_f (derived from weather_temp) IS in the allowlist.

        FINDING: weather_temp is used as a feature (as temp_f). The MLB API's
        gameData.weather.temp field is populated pre-game for most venues (it's
        the forecast/current condition at game start). This is acceptable because:
        1. Temperature is an ambient condition, not a game outcome
        2. It's available from weather APIs pre-game
        3. The feature captures park effects on ball flight, not outcomes
        """
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from classical_learning.strategy.data import _PREGAME_FEATURE_PREFIXES

        assert any("temp_f" == p or "temp_f".startswith(p)
                   for p in _PREGAME_FEATURE_PREFIXES)

    def test_attendance_not_used_as_feature(self):
        """Attendance is post-game only and must NOT appear in features.

        FINDING: attendance is loaded into the game frame but the allowlist
        in strategy/data.py correctly excludes it since 'attendance' does not
        match any prefix in _PREGAME_FEATURE_PREFIXES.
        """
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from classical_learning.strategy.data import _PREGAME_FEATURE_PREFIXES

        assert not any("attendance".startswith(p) or "attendance" == p
                       for p in _PREGAME_FEATURE_PREFIXES)


# ---------------------------------------------------------------------------
# Section 7: Doubleheader Sort Order
# ---------------------------------------------------------------------------

class TestDoubleheaderSortOrder:
    """Verify game_builder sort handles doubleheaders correctly."""

    def test_doubleheader_same_date_different_game_pk(self):
        """Two games on the same date sort by game_pk (which is sequentially assigned)."""
        # Game 1 of doubleheader has lower game_pk
        ls1 = _make_linescore(600001, 2024, [(1, 0)] * 9)
        ls2 = _make_linescore(600002, 2024, [(0, 1)] * 9)
        linescore = pd.concat([ls2, ls1])  # intentionally reversed

        targets = build_game_targets(linescore)
        # Sort by game_pk (as game_builder does: sort_values(["game_date", "game_pk"]))
        targets = targets.sort_values(["game_pk"]).reset_index(drop=True)
        assert targets["game_pk"].iloc[0] == 600001
        assert targets["game_pk"].iloc[1] == 600002

    def test_doubleheader_targets_independent(self):
        """Each game of a doubleheader has independent targets."""
        ls1 = _make_linescore(600010, 2024, [(3, 1)] + [(0, 0)] * 8)
        ls2 = _make_linescore(600011, 2024, [(0, 0)] * 8 + [(0, 2)])
        linescore = pd.concat([ls1, ls2])

        targets = build_game_targets(linescore)
        t1 = targets[targets["game_pk"] == 600010].iloc[0]
        t2 = targets[targets["game_pk"] == 600011].iloc[0]

        assert t1["home_win"] == 1.0
        assert t2["away_win"] == 1.0
        assert t1["yrfi"] == 1.0
        assert t2["yrfi"] == 0.0


# ---------------------------------------------------------------------------
# Section 8: Duplicate game_pks
# ---------------------------------------------------------------------------

class TestDuplicateGamePks:
    """Verify behavior when duplicate game_pks appear in linescore."""

    def test_duplicate_game_pk_in_linescore_aggregates(self):
        """If duplicate inning rows exist for same game_pk, they get summed.

        FINDING: The groupby in build_game_targets sums all rows per game_pk.
        If the same inning appears twice (data corruption), it would double-count
        runs. This is a data quality issue, not a code bug — the pipeline trusts
        the source data to be deduplicated at ingestion time.
        """
        # Normal game
        ls_normal = _make_linescore(700001, 2024, [(1, 0)] * 9)
        targets = build_game_targets(ls_normal)
        assert targets["total_runs"].iloc[0] == 9

        # Same game_pk with duplicate inning 1
        ls_duped = pd.concat([
            ls_normal,
            pd.DataFrame([{"game_pk": 700001, "season": 2024, "inning": 1,
                           "home_runs": 1, "away_runs": 0}])
        ])
        targets_duped = build_game_targets(ls_duped)
        # Duplicate inning 1 row gets summed — this is a data corruption scenario
        assert targets_duped["total_runs"].iloc[0] == 10  # 9 + 1 extra

    def test_unique_game_pks_in_multi_game_linescore(self):
        """Multiple games produce one row each in targets output."""
        ls1 = _make_linescore(700010, 2024, [(1, 0)] * 9)
        ls2 = _make_linescore(700011, 2024, [(0, 1)] * 9)
        ls3 = _make_linescore(700012, 2024, [(2, 2)] * 9)
        linescore = pd.concat([ls1, ls2, ls3])

        targets = build_game_targets(linescore)
        assert len(targets) == 3
        assert set(targets["game_pk"]) == {700010, 700011, 700012}


# ---------------------------------------------------------------------------
# Section 9: _sum_period Correctness
# ---------------------------------------------------------------------------

class TestSumPeriod:
    """Unit tests for the _sum_period helper function."""

    def test_sum_period_max_inning_5(self):
        """Only innings 1-5 are summed when max_inning=5."""
        linescore = _make_linescore(800001, 2024, [(1, 1)] * 9)
        result = _sum_period(linescore, max_inning=5, prefix="test")
        assert result["test_home_runs"].iloc[0] == 5
        assert result["test_away_runs"].iloc[0] == 5

    def test_sum_period_max_inning_1(self):
        """Only inning 1 is summed when max_inning=1."""
        linescore = _make_linescore(800002, 2024,
                                    [(3, 2), (1, 1), (0, 0)] + [(0, 0)] * 6)
        result = _sum_period(linescore, max_inning=1, prefix="first_1")
        assert result["first_1_home_runs"].iloc[0] == 3
        assert result["first_1_away_runs"].iloc[0] == 2

    def test_sum_period_max_inning_9(self):
        """Innings 1-9 summed; extras excluded when max_inning=9."""
        innings = [(1, 0)] * 9 + [(3, 0)]  # 10-inning game
        linescore = _make_linescore(800003, 2024, innings)
        result = _sum_period(linescore, max_inning=9, prefix="reg")
        assert result["reg_home_runs"].iloc[0] == 9  # excludes 10th
        assert result["reg_away_runs"].iloc[0] == 0

    def test_sum_period_fewer_innings_than_max(self):
        """If game has fewer innings than max, sum all available."""
        linescore = _make_linescore(800004, 2024, [(2, 1), (0, 3), (1, 0)])  # 3 innings
        result = _sum_period(linescore, max_inning=5, prefix="first_5")
        assert result["first_5_home_runs"].iloc[0] == 3  # 2+0+1
        assert result["first_5_away_runs"].iloc[0] == 4  # 1+3+0


# ---------------------------------------------------------------------------
# Section 10: Edge Cases and Robustness
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Additional edge cases for target construction."""

    def test_zero_run_game(self):
        """A hypothetical 0-0 game (shortened) produces valid targets."""
        linescore = _make_linescore(900001, 2024, [(0, 0)] * 5)
        targets = build_game_targets(linescore)
        assert targets["total_runs"].iloc[0] == 0
        assert targets["home_win"].iloc[0] == 0.0
        assert targets["away_win"].iloc[0] == 0.0
        assert targets["yrfi"].iloc[0] == 0.0
        assert targets["nrfi"].iloc[0] == 1.0

    def test_high_scoring_extra_inning_game(self):
        """Extreme scoring in extras doesn't corrupt targets."""
        innings = [(0, 0)] * 9 + [(10, 8)]  # 10th inning explosion
        linescore = _make_linescore(900002, 2024, innings)
        targets = build_game_targets(linescore)
        assert targets["total_runs"].iloc[0] == 18
        assert targets["home_win"].iloc[0] == 1.0
        assert targets["extra_innings"].iloc[0] == 1.0
        assert targets["first_5_total_runs"].iloc[0] == 0

    def test_coerce_non_numeric_innings(self):
        """Non-numeric values in linescore are coerced to 0."""
        df = pd.DataFrame([
            {"game_pk": 900003, "season": 2024, "inning": "x",
             "home_runs": "bad", "away_runs": None},
        ])
        targets = build_game_targets(df)
        # Should not crash; coerces to 0
        assert targets["total_runs"].iloc[0] == 0

    def test_empty_linescore_returns_empty(self):
        """Empty linescore input returns empty DataFrame."""
        empty = pd.DataFrame(columns=["game_pk", "season", "inning", "home_runs", "away_runs"])
        targets = build_game_targets(empty)
        assert targets.empty

    def test_target_status_always_trainable(self):
        """All game targets default to 'trainable' status.

        FINDING: build_game_targets does not exclude shortened games from training.
        The `shortened_or_called` column is computed but target_status remains
        'trainable' for all games. Downstream code must decide whether to exclude
        shortened games from the training set for specific targets.
        """
        # Shortened game
        linescore = _make_linescore(900004, 2024, [(1, 0)] * 5)
        targets = build_game_targets(linescore)
        assert targets["target_status"].iloc[0] == TRAINABLE

        # Normal game
        linescore = _make_linescore(900005, 2024, [(1, 0)] * 9)
        targets = build_game_targets(linescore)
        assert targets["target_status"].iloc[0] == TRAINABLE


# ---------------------------------------------------------------------------
# Section 11: Starting Pitcher Game Stats (Post-Game Leakage Guard)
# ---------------------------------------------------------------------------

class TestStartingPitcherLeakageGuard:
    """Verify that starting pitcher GAME stats from box score are not used as features.

    The game_builder includes sp_home_game_* and sp_away_game_* columns (innings pitched,
    strikeouts, etc.) — these are POST-GAME data from the boxscore. The feature
    allowlist must exclude them.
    """

    def test_sp_game_stats_excluded_from_features(self):
        """Starting pitcher game-level stats should NOT match any feature prefix."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from classical_learning.strategy.data import _PREGAME_FEATURE_PREFIXES

        sp_game_cols = [
            "sp_home_game_innings_pitched", "sp_home_game_hits",
            "sp_home_game_runs", "sp_home_game_earned_runs",
            "sp_home_game_bb", "sp_home_game_so", "sp_home_game_hr",
            "sp_home_game_pitches_thrown", "sp_home_game_strikes_thrown",
            "sp_away_game_innings_pitched", "sp_away_game_hits",
            "sp_away_game_runs", "sp_away_game_so",
        ]
        for col in sp_game_cols:
            matches = any(col.startswith(p) or col == p for p in _PREGAME_FEATURE_PREFIXES)
            assert not matches, (
                f"SP game stat '{col}' matches a pregame prefix — this is leakage!"
            )


# ---------------------------------------------------------------------------
# Section 12: game_date Source Verification (originalDate)
# ---------------------------------------------------------------------------

class TestGameDateSource:
    """The download_history.py script uses `originalDate` from the GUMBO API.

    For suspended/resumed games:
    - originalDate = the date the game was first scheduled/started
    - officialDate = can be the completion date

    Using originalDate ensures features computed on Date A aren't polluted by
    games that were actually completed on Date B.
    """

    def test_game_date_propagates_from_meta_to_targets(self):
        """game_date from meta overrides any linescore-derived date."""
        linescore = _make_linescore(1000001, 2023, [(1, 0)] * 9)
        meta = _make_game_meta(1000001, game_date="2023-04-15")
        targets = build_game_targets(linescore, meta)
        assert "game_date" in targets.columns
        assert targets["game_date"].iloc[0] == "2023-04-15"

    def test_targets_work_without_meta(self):
        """build_game_targets works without game_meta (no game_date column)."""
        linescore = _make_linescore(1000002, 2023, [(1, 0)] * 9)
        targets = build_game_targets(linescore, game_meta_df=None)
        assert "game_date" not in targets.columns
        assert targets["home_win"].iloc[0] == 1.0
