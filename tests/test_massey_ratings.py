"""Tests for classical_learning/engineering/massey_ratings.py.

Validates:
1. Correctness of cumulative margin computation
2. Massey normal equation solution (sum-to-zero constraint)
3. Home advantage coefficient extraction
4. Temporal safety (no lookahead)
5. Disconnected schedule handling
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "classical_learning"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deep_learning"))

from engineering.massey_ratings import (
    MasseyDesign,
    MasseyFit,
    prepare_linescore_cumulative,
    fit_massey_inning,
    build_massey_season_ratings,
    build_pregame_massey_features,
    _team_components,
    MASSEY_TARGETS,
)


def _make_linescore(games_data: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build linescore + meta from compact game specs.

    Each game_spec: {game_pk, season, game_date, home, away, innings: [(h_runs, a_runs), ...]}
    """
    ls_rows = []
    meta_rows = []
    for g in games_data:
        meta_rows.append({
            "game_pk": g["game_pk"],
            "season": g["season"],
            "game_date": g["game_date"],
            "home_team_id": g["home"],
            "away_team_id": g["away"],
        })
        for inn_idx, (h, a) in enumerate(g["innings"], start=1):
            ls_rows.append({
                "game_pk": g["game_pk"],
                "season": g["season"],
                "inning": inn_idx,
                "home_runs": h,
                "away_runs": a,
            })

    return pd.DataFrame(ls_rows), pd.DataFrame(meta_rows)


class TestPrepareLinescore:
    def test_basic_cumulative(self):
        """Cumulative margins correctly accumulate across innings."""
        games = [{
            "game_pk": 1, "season": 2023, "game_date": "2023-04-01",
            "home": 100, "away": 200,
            "innings": [(2, 0), (0, 1), (1, 0), (0, 0), (0, 2), (0, 0), (1, 0), (0, 0), (0, 0)],
        }]
        ls, meta = _make_linescore(games)
        cum = prepare_linescore_cumulative(ls, meta)

        assert len(cum) == 1
        row = cum.iloc[0]
        # Inning 1: home +2, cum = 2-0 = 2
        assert row["margin_inn1"] == 2.0
        # Inning 2: +0-1, cum = 2-1 = 1
        assert row["margin_inn2"] == 1.0
        # Inning 3: +1-0, cum = 3-1 = 2
        assert row["margin_inn3"] == 2.0
        # Inning 5: +0-2, cum = 3-3 = 0
        assert row["margin_inn5"] == 0.0
        # Final: 4-3 = 1
        assert row["margin_full"] == 1.0

    def test_extra_innings(self):
        """Games with >9 innings: inn9 captures thru 9, full captures all."""
        games = [{
            "game_pk": 1, "season": 2023, "game_date": "2023-04-01",
            "home": 100, "away": 200,
            "innings": [(0, 0)] * 9 + [(1, 0)],  # 10 innings, walk-off
        }]
        ls, meta = _make_linescore(games)
        cum = prepare_linescore_cumulative(ls, meta)
        row = cum.iloc[0]
        assert row["margin_inn9"] == 0.0  # tied through 9
        assert row["margin_full"] == 1.0  # home wins in 10th

    def test_deduplicates_batches(self):
        """Duplicate (game_pk, inning) rows from batch overlap are deduped."""
        games = [{
            "game_pk": 1, "season": 2023, "game_date": "2023-04-01",
            "home": 100, "away": 200,
            "innings": [(1, 0)] * 9,
        }]
        ls, meta = _make_linescore(games)
        # Simulate batch duplication
        ls_dup = pd.concat([ls, ls], ignore_index=True)
        cum = prepare_linescore_cumulative(ls_dup, meta)
        assert len(cum) == 1
        assert cum.iloc[0]["margin_full"] == 9.0


class TestFitMasseyInning:
    def _balanced_season(self, n_teams=4, games_per_pair=3) -> pd.DataFrame:
        """Create a balanced round-robin season: each pair plays both home & away."""
        teams = list(range(100, 100 + n_teams))
        games = []
        game_pk = 1
        day = 0
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                for _ in range(games_per_pair):
                    innings = [(1, 0)] * 5 + [(0, 1)] * 4  # home wins 5-4
                    games.append({
                        "game_pk": game_pk,
                        "season": 2023,
                        "game_date": f"2023-04-{1 + day:02d}",
                        "home": teams[i],
                        "away": teams[j],
                        "innings": innings,
                    })
                    game_pk += 1
                    day = (day + 1) % 28
        ls, meta = _make_linescore(games)
        return prepare_linescore_cumulative(ls, meta)

    def test_sum_to_zero(self):
        """Team ratings sum to zero (Massey constraint)."""
        cum = self._balanced_season()
        fit = fit_massey_inning(cum, "full", season=2023)
        team_ratings = fit.ratings["massey_full"].values
        assert abs(team_ratings.sum()) < 1e-8

    def test_home_advantage_positive(self):
        """Home advantage coefficient is positive for balanced home-wins schedule."""
        cum = self._balanced_season()
        fit = fit_massey_inning(cum, "full", season=2023)
        ha = fit.coefficients.get("home_advantage", 0.0)
        assert ha > 0, f"Expected positive HA, got {ha}"

    def test_no_home_advantage_design(self):
        """Design without HA produces no home_advantage coefficient."""
        cum = self._balanced_season()
        design = MasseyDesign("test_no_ha", include_home_advantage=False)
        fit = fit_massey_inning(cum, "full", design=design, season=2023)
        assert "home_advantage" not in fit.coefficients

    def test_stronger_team_rated_higher(self):
        """Team that always wins is rated highest."""
        # A beats B by large margin, B beats C by small margin, A beats C
        # Each matchup alternates home/away to isolate team strength from HA
        games = []
        pk = 1
        for _ in range(5):
            # A at home vs B: 8-1
            games.append({
                "game_pk": pk, "season": 2023,
                "game_date": f"2023-04-{pk:02d}",
                "home": 100, "away": 200,
                "innings": [(1, 0)] * 8 + [(0, 1)],
            })
            pk += 1
            # A at away vs B: A still wins 7-2
            games.append({
                "game_pk": pk, "season": 2023,
                "game_date": f"2023-04-{pk:02d}",
                "home": 200, "away": 100,
                "innings": [(0, 1)] * 7 + [(1, 0)] * 2,
            })
            pk += 1
        for _ in range(5):
            # B at home vs C: 5-4
            games.append({
                "game_pk": pk, "season": 2023,
                "game_date": f"2023-04-{pk:02d}",
                "home": 200, "away": 300,
                "innings": [(1, 0)] * 5 + [(0, 1)] * 4,
            })
            pk += 1
            # B at away vs C: B still wins 4-3
            games.append({
                "game_pk": pk, "season": 2023,
                "game_date": f"2023-04-{pk:02d}",
                "home": 300, "away": 200,
                "innings": [(0, 1)] * 4 + [(1, 0)] * 3 + [(0, 0)] * 2,
            })
            pk += 1
        for _ in range(5):
            # A at home vs C: 6-1
            games.append({
                "game_pk": pk, "season": 2023,
                "game_date": f"2023-04-{pk:02d}",
                "home": 100, "away": 300,
                "innings": [(1, 0)] * 6 + [(0, 1)] + [(0, 0)] * 2,
            })
            pk += 1
            # A at away vs C: 5-2
            games.append({
                "game_pk": pk, "season": 2023,
                "game_date": f"2023-04-{pk:02d}",
                "home": 300, "away": 100,
                "innings": [(0, 1)] * 5 + [(1, 0)] * 2 + [(0, 0)] * 2,
            })
            pk += 1
        ls, meta = _make_linescore(games)
        cum = prepare_linescore_cumulative(ls, meta)
        fit = fit_massey_inning(cum, "full", season=2023)
        r = fit.ratings.set_index("team_id")["massey_full"]
        assert r[100] > r[200] > r[300]

    def test_insufficient_games(self):
        """Returns empty fit when games < min_games."""
        games = [{
            "game_pk": 1, "season": 2023, "game_date": "2023-04-01",
            "home": 100, "away": 200,
            "innings": [(1, 0)] * 9,
        }]
        ls, meta = _make_linescore(games)
        cum = prepare_linescore_cumulative(ls, meta)
        design = MasseyDesign("test", min_games=5)
        fit = fit_massey_inning(cum, "full", design=design, season=2023)
        assert fit.ratings.empty

    def test_inning_targets_differ(self):
        """Per-inning ratings differ when scoring patterns change."""
        # Team A dominates early, B dominates late
        games = []
        for pk in range(1, 21):
            games.append({
                "game_pk": pk, "season": 2023,
                "game_date": f"2023-04-{pk:02d}",
                "home": 100, "away": 200,
                "innings": [(3, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                            (0, 2), (0, 2), (0, 2), (0, 2)],  # A: 3, B: 8 → B wins
            })
        for pk in range(21, 41):
            games.append({
                "game_pk": pk, "season": 2023,
                "game_date": f"2023-04-{pk:02d}",
                "home": 200, "away": 100,
                "innings": [(0, 3), (0, 0), (0, 0), (0, 0), (0, 0),
                            (2, 0), (2, 0), (2, 0), (2, 0)],  # A: 3, B: 8 again
            })
        ls, meta = _make_linescore(games)
        cum = prepare_linescore_cumulative(ls, meta)

        fit_inn1 = fit_massey_inning(cum, "inn1", season=2023)
        fit_full = fit_massey_inning(cum, "full", season=2023)

        r1 = fit_inn1.ratings.set_index("team_id")["massey_inn1"]
        rf = fit_full.ratings.set_index("team_id")["massey_full"]

        # A is better in inning 1, B is better overall
        assert r1[100] > r1[200], "Team A should dominate inning 1"
        assert rf[200] > rf[100], "Team B should dominate full game"


class TestTemporalSafety:
    def test_no_lookahead(self):
        """Pregame features for date D use only games before D."""
        # 20 games then a change — team 100 gets dominant
        games = []
        for pk in range(1, 21):
            games.append({
                "game_pk": pk, "season": 2023,
                "game_date": "2023-04-01",
                "home": 100 + (pk % 4), "away": 100 + ((pk + 1) % 4),
                "innings": [(1, 0)] * 5 + [(0, 1)] * 4,  # 5-4
            })
        # Day 2: team 100 blows out everyone
        for pk in range(21, 51):
            games.append({
                "game_pk": pk, "season": 2023,
                "game_date": "2023-04-02",
                "home": 100, "away": 101 + (pk % 3),
                "innings": [(5, 0)] * 9,  # 45-0
            })
        # Day 3: more games
        for pk in range(51, 81):
            games.append({
                "game_pk": pk, "season": 2023,
                "game_date": "2023-04-03",
                "home": 100, "away": 101 + (pk % 3),
                "innings": [(5, 0)] * 9,
            })

        ls, meta = _make_linescore(games)
        cum = prepare_linescore_cumulative(ls, meta)

        features = build_pregame_massey_features(
            cum, targets=["full"], min_prior_games=10
        )

        # Day 2 features should only use Day 1 data (balanced → near zero diffs)
        day2 = features[features["game_date"] == pd.Timestamp("2023-04-02")]
        if not day2.empty and "diff_massey_full" in day2.columns:
            max_diff_day2 = day2["diff_massey_full"].abs().max()
            # Day 1 was balanced → diffs should be small
            assert max_diff_day2 < 2.0, f"Day 2 used lookahead? max_diff={max_diff_day2}"


class TestDisconnectedSchedule:
    def test_two_components(self):
        """Handles disconnected schedule with separate sum-to-zero per component."""
        # Conference A: teams 100, 101 play each other (alternating home/away)
        # Conference B: teams 200, 201 play each other
        # No crossover
        games = []
        for pk in range(1, 11):
            games.append({
                "game_pk": pk, "season": 2023,
                "game_date": f"2023-04-{pk:02d}",
                "home": 100, "away": 101,
                "innings": [(2, 0)] * 9,
            })
        for pk in range(11, 21):
            games.append({
                "game_pk": pk, "season": 2023,
                "game_date": f"2023-04-{pk:02d}",
                "home": 101, "away": 100,
                "innings": [(0, 2)] * 9,  # 100 still wins
            })
        for pk in range(21, 31):
            games.append({
                "game_pk": pk, "season": 2023,
                "game_date": f"2023-04-{pk:02d}",
                "home": 200, "away": 201,
                "innings": [(1, 0)] * 9,
            })
        for pk in range(31, 41):
            games.append({
                "game_pk": pk, "season": 2023,
                "game_date": f"2023-04-{pk:02d}",
                "home": 201, "away": 200,
                "innings": [(0, 1)] * 9,  # 200 still wins
            })

        ls, meta = _make_linescore(games)
        cum = prepare_linescore_cumulative(ls, meta)
        # Use no-HA design so constraint is purely team-based
        design = MasseyDesign("test_no_ha", include_home_advantage=False)
        fit = fit_massey_inning(cum, "full", design=design, season=2023)

        assert len(fit.components) == 2
        rating_col = f"{design.name}_full"
        r = fit.ratings.set_index("team_id")[rating_col]
        # Within each component: winner > loser
        assert r[100] > r[101]
        assert r[200] > r[201]
        # Each component sums to zero independently
        assert abs(r[100] + r[101]) < 1e-8
        assert abs(r[200] + r[201]) < 1e-8
