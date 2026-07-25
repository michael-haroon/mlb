"""Tests for pregame.engineering.pitch_level_features.

Covers five feature families — TTO velocity decay, K-BB% splits, FIP splits,
platoon wOBA, and pitch-mix matchup — via the public entry point
`compute_pitch_level_features(pitches_raw, game_frame)`.

Key invariants tested:
  - shift(1) no-leakage: rolling features for game N never include game N data
  - game_type_code=="R" filter: spring training excluded
  - First-game NaN: no prior data => NaN, not 0.0
  - Missing pitcher: NaN starter => NaN features, no crash
  - Correct team assignment (home vs away)
  - Formula correctness (FIP, velo decay)
  - Output shape, column completeness, float32 dtype

NOTE: The module's _compute_woba_splits has a bug on pandas 3.0: it does not
guard against empty sub-DataFrames when a pitch_hand_code split has 0 rows.
groupby().apply(include_groups=False) on an empty DF returns a DataFrame (not
a Series) which cannot be assigned to a single column. Tests work around this
by always including both LHP and RHP pitch data in fixtures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pregame.engineering.pitch_level_features import (
    FIP_CONSTANT,
    compute_pitch_level_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HOME_TEAM = 147  # NYY
_AWAY_TEAM = 111  # BOS
_PITCHER_HOME = 100001  # RHP
_PITCHER_AWAY = 200001  # RHP
_PITCHER_HOME_LHP = 100002  # LHP
_PITCHER_AWAY_LHP = 200002  # LHP


def _make_pitch_row(
    game_pk: int,
    season: int,
    game_date: str,
    pitcher_id: int,
    batter_id: int,
    at_bat_index: int,
    pitch_number: int,
    *,
    is_pitch: bool = True,
    release_speed: float = 92.0,
    coord_x0: float = -1.5,
    coord_z0: float = 5.8,
    pitch_type: str = "FF",
    bat_side_code: str = "R",
    pitch_hand_code: str = "R",
    event_type: str | None = None,
    inning: int = 1,
    half_inning: str = "top",
    cum_outs: int = 0,
    home_team_id: int = _HOME_TEAM,
    away_team_id: int = _AWAY_TEAM,
    game_type_code: str = "R",
) -> dict:
    """Build a single pitch row with all required PITCH_LEVEL_COLUMNS."""
    return {
        "game_pk": game_pk,
        "season": season,
        "game_date": game_date,
        "game_type_code": game_type_code,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "pitcher_id": pitcher_id,
        "batter_id": batter_id,
        "is_pitch": is_pitch,
        "release_speed": release_speed,
        "coord_x0": coord_x0,
        "coord_z0": coord_z0,
        "pitch_type": pitch_type,
        "bat_side_code": bat_side_code,
        "pitch_hand_code": pitch_hand_code,
        "event_type": event_type,
        "at_bat_index": at_bat_index,
        "pitch_number": pitch_number,
        "inning": inning,
        "half_inning": half_inning,
        "cum_outs": cum_outs,
        "pre_on_first_id": np.nan,
        "pre_on_second_id": np.nan,
        "pre_on_third_id": np.nan,
    }


def _make_game_frame(games: list[dict]) -> pd.DataFrame:
    """Build a game_frame from a list of dicts with game_pk, game_date, etc."""
    return pd.DataFrame(games)


def _make_pa_pitches(
    game_pk: int,
    season: int,
    game_date: str,
    pitcher_id: int,
    batter_id: int,
    at_bat_index: int,
    outcome_event: str,
    *,
    num_pitches: int = 3,
    bat_side_code: str = "R",
    pitch_hand_code: str = "R",
    inning: int = 1,
    half_inning: str = "top",
    pitch_type: str = "FF",
    release_speed: float = 92.0,
    home_team_id: int = _HOME_TEAM,
    away_team_id: int = _AWAY_TEAM,
    game_type_code: str = "R",
) -> list[dict]:
    """Generate pitches for one complete plate appearance (PA).

    Only the last pitch has at_bat_event set (mimicking real data).
    """
    rows = []
    for pn in range(1, num_pitches + 1):
        event = outcome_event if pn == num_pitches else None
        rows.append(_make_pitch_row(
            game_pk=game_pk,
            season=season,
            game_date=game_date,
            pitcher_id=pitcher_id,
            batter_id=batter_id,
            at_bat_index=at_bat_index,
            pitch_number=pn,
            is_pitch=True,
            release_speed=release_speed,
            bat_side_code=bat_side_code,
            pitch_hand_code=pitch_hand_code,
            event_type=event,
            inning=inning,
            half_inning=half_inning,
            pitch_type=pitch_type,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            game_type_code=game_type_code,
        ))
    return rows


def _generate_start_pitches(
    game_pk: int,
    season: int,
    game_date: str,
    pitcher_id: int,
    total_pitches: int = 100,
    *,
    early_velo: float = 95.0,
    mid_velo: float = 93.0,
    late_velo: float = 90.0,
    coord_x0: float = -1.5,
    coord_z0: float = 5.8,
    pitch_type: str = "FF",
    pitch_hand_code: str = "R",
    half_inning: str = "top",
    home_team_id: int = _HOME_TEAM,
    away_team_id: int = _AWAY_TEAM,
) -> list[dict]:
    """Generate N pitches for a single start with controlled velo profile.

    Pitches 1-25: early_velo, 26-74: mid_velo, 75+: late_velo.
    All are is_pitch=True, no at_bat_event (TTO only needs release data).
    """
    rows = []
    at_bat_idx = 0
    for seq in range(1, total_pitches + 1):
        if seq <= 25:
            velo = early_velo
        elif seq >= 75:
            velo = late_velo
        else:
            velo = mid_velo

        # Advance at_bat_index every ~4 pitches (realistic PA length).
        if seq > 1 and (seq - 1) % 4 == 0:
            at_bat_idx += 1
        pn = ((seq - 1) % 4) + 1

        rows.append(_make_pitch_row(
            game_pk=game_pk,
            season=season,
            game_date=game_date,
            pitcher_id=pitcher_id,
            batter_id=300 + (at_bat_idx % 9),  # cycle through 9 batters
            at_bat_index=at_bat_idx,
            pitch_number=pn,
            is_pitch=True,
            release_speed=velo,
            coord_x0=coord_x0,
            coord_z0=coord_z0,
            pitch_type=pitch_type,
            bat_side_code="R",
            pitch_hand_code=pitch_hand_code,
            event_type=None,
            inning=1 + at_bat_idx // 3,
            half_inning=half_inning,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
        ))
    return rows


def _add_both_hand_filler_pitches(
    rows: list[dict],
    game_pks_and_dates: list[tuple[int, str]],
    at_bat_offset: int = 900,
) -> None:
    """Add LHP and RHP plate appearances to prevent empty-hand-split bug.

    The module's _compute_woba_splits crashes on pandas 3.0 if either L or R hand
    sub has 0 rows (groupby.apply(include_groups=False) returns a DataFrame instead
    of a Series for empty DataFrames). Adding 3 PAs per hand per game avoids this
    without affecting rolling feature values (they require min_periods=10+).
    """
    for game_pk, game_date in game_pks_and_dates:
        # LHP filler (3 PAs per game)
        for i in range(3):
            rows.extend(_make_pa_pitches(
                game_pk=game_pk,
                season=2025,
                game_date=game_date,
                pitcher_id=_PITCHER_HOME_LHP,
                batter_id=950 + i,
                at_bat_index=at_bat_offset + i,
                outcome_event="field_out",
                bat_side_code="R",
                pitch_hand_code="L",
                half_inning="top",
                num_pitches=2,
            ))
        # RHP filler (3 PAs per game)
        for i in range(3):
            rows.extend(_make_pa_pitches(
                game_pk=game_pk,
                season=2025,
                game_date=game_date,
                pitcher_id=_PITCHER_AWAY,
                batter_id=960 + i,
                at_bat_index=at_bat_offset + 10 + i,
                outcome_event="field_out",
                bat_side_code="L",
                pitch_hand_code="R",
                half_inning="top",
                num_pitches=2,
            ))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_GAME_DATES = [
    (1001, "2025-04-01"),
    (1002, "2025-04-06"),
    (1003, "2025-04-11"),
]


@pytest.fixture
def three_game_frame() -> pd.DataFrame:
    """Three regular-season games with known pitchers."""
    return _make_game_frame([
        {
            "game_pk": 1001, "game_date": "2025-04-01",
            "home_team_id": _HOME_TEAM, "away_team_id": _AWAY_TEAM,
            "probable_pitcher_home_id": _PITCHER_HOME,
            "probable_pitcher_away_id": _PITCHER_AWAY,
        },
        {
            "game_pk": 1002, "game_date": "2025-04-06",
            "home_team_id": _HOME_TEAM, "away_team_id": _AWAY_TEAM,
            "probable_pitcher_home_id": _PITCHER_HOME,
            "probable_pitcher_away_id": _PITCHER_AWAY,
        },
        {
            "game_pk": 1003, "game_date": "2025-04-11",
            "home_team_id": _HOME_TEAM, "away_team_id": _AWAY_TEAM,
            "probable_pitcher_home_id": _PITCHER_HOME,
            "probable_pitcher_away_id": _PITCHER_AWAY,
        },
    ])


@pytest.fixture
def kbb_pitches_strikeout_lhh_walk_rhh() -> pd.DataFrame:
    """Pitcher _PITCHER_HOME: strikes out all LHH, walks all RHH.

    3 games x 5 PAs per hand per game = 30 total PAs.
    Includes LHP filler to prevent empty-hand-split crash.
    """
    rows = []
    for game_pk, game_date in _GAME_DATES:
        ab_idx = 0
        # 5 LHH strikeouts per game
        for _ in range(5):
            rows.extend(_make_pa_pitches(
                game_pk=game_pk, season=2025, game_date=game_date,
                pitcher_id=_PITCHER_HOME, batter_id=500 + ab_idx,
                at_bat_index=ab_idx, outcome_event="strikeout",
                bat_side_code="L", pitch_hand_code="R",
                half_inning="top",
            ))
            ab_idx += 1
        # 5 RHH walks per game
        for _ in range(5):
            rows.extend(_make_pa_pitches(
                game_pk=game_pk, season=2025, game_date=game_date,
                pitcher_id=_PITCHER_HOME, batter_id=600 + ab_idx,
                at_bat_index=ab_idx, outcome_event="walk",
                bat_side_code="R", pitch_hand_code="R",
                half_inning="top",
            ))
            ab_idx += 1

    _add_both_hand_filler_pitches(rows, _GAME_DATES)
    return pd.DataFrame(rows)


@pytest.fixture
def fip_pitches_known_stats() -> pd.DataFrame:
    """Pitcher _PITCHER_HOME vs all RHH across 3 games with known FIP components.

    Game 1: 3 K, 1 BB (walk), 1 HR, 4 field_out => outs=3+4=7, IP=7/3
    Game 2: 3 K, 1 BB (walk), 0 HR, 5 field_out => outs=3+5=8, IP=8/3
    Game 3: just a few PAs (these should NOT leak into game 3 features)

    Expected for game 3 (roll5, shift(1), uses games 1+2):
      HR=1, BB=2, K=6, IP=15/3=5
      FIP = (13*1 + 3*2 - 2*6) / 5 + 3.10 = 7/5 + 3.10 = 4.50
    """
    rows = []

    # Game 1: 3K + 1BB + 1HR + 4 field_out = 9 PAs vs RHH
    game1_events = (
        ["strikeout"] * 3 + ["walk"] * 1 + ["home_run"] * 1 + ["field_out"] * 4
    )
    for ab_idx, event in enumerate(game1_events):
        rows.extend(_make_pa_pitches(
            game_pk=1001, season=2025, game_date="2025-04-01",
            pitcher_id=_PITCHER_HOME, batter_id=700 + ab_idx,
            at_bat_index=ab_idx, outcome_event=event,
            bat_side_code="R", pitch_hand_code="R",
            half_inning="top",
        ))

    # Game 2: 3K + 1BB + 0HR + 5 field_out = 9 PAs vs RHH
    game2_events = (
        ["strikeout"] * 3 + ["walk"] * 1 + ["field_out"] * 5
    )
    for ab_idx, event in enumerate(game2_events):
        rows.extend(_make_pa_pitches(
            game_pk=1002, season=2025, game_date="2025-04-06",
            pitcher_id=_PITCHER_HOME, batter_id=800 + ab_idx,
            at_bat_index=ab_idx + 20, outcome_event=event,
            bat_side_code="R", pitch_hand_code="R",
            half_inning="top",
        ))

    # Game 3: 2K + 2HR (dramatically different — should NOT leak)
    game3_events = ["strikeout"] * 2 + ["home_run"] * 2
    for ab_idx, event in enumerate(game3_events):
        rows.extend(_make_pa_pitches(
            game_pk=1003, season=2025, game_date="2025-04-11",
            pitcher_id=_PITCHER_HOME, batter_id=900 + ab_idx,
            at_bat_index=ab_idx + 40, outcome_event=event,
            bat_side_code="R", pitch_hand_code="R",
            half_inning="top",
        ))

    _add_both_hand_filler_pitches(rows, _GAME_DATES)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: No-leakage (shift(1) enforcement)
# ---------------------------------------------------------------------------

class TestNoLeakage:
    """Verify that rolling features for game N never include data from game N."""

    def test_kbb_shift1_excludes_current_game(self, three_game_frame):
        """Pitcher has 100% K vs LHH in games 1-2, then 0% K in game 3.

        Game 3 feature should reflect games 1-2 (kpct=1.0), NOT game 3 data.
        """
        rows = []
        # Games 1-2: all LHH strikeouts
        for game_pk, game_date in [(1001, "2025-04-01"), (1002, "2025-04-06")]:
            for ab_idx in range(5):
                rows.extend(_make_pa_pitches(
                    game_pk=game_pk, season=2025, game_date=game_date,
                    pitcher_id=_PITCHER_HOME, batter_id=500 + ab_idx,
                    at_bat_index=ab_idx, outcome_event="strikeout",
                    bat_side_code="L", pitch_hand_code="R",
                    half_inning="top",
                ))

        # Game 3: all LHH walks (opposite behavior)
        for ab_idx in range(5):
            rows.extend(_make_pa_pitches(
                game_pk=1003, season=2025, game_date="2025-04-11",
                pitcher_id=_PITCHER_HOME, batter_id=550 + ab_idx,
                at_bat_index=ab_idx + 20, outcome_event="walk",
                bat_side_code="L", pitch_hand_code="R",
                half_inning="top",
            ))

        _add_both_hand_filler_pitches(rows, _GAME_DATES)
        pitches = pd.DataFrame(rows)
        result = compute_pitch_level_features(pitches, three_game_frame)

        # Game 3's K% vs LHH should be 1.0 (from games 1-2), not 0.0 (game 3)
        game3_row = result[result["game_pk"] == 1003]
        kpct_val = game3_row["home_sp_kpct_vs_lhh_roll5"].values[0]
        assert kpct_val == pytest.approx(1.0, abs=1e-4), (
            f"Leakage detected: kpct={kpct_val}, expected 1.0 from games 1-2 only"
        )


# ---------------------------------------------------------------------------
# Test 2: TTO velocity decay correctness
# ---------------------------------------------------------------------------

class TestTTOVeloDecay:
    """Verify velocity decay feature reflects the correct sign and magnitude."""

    def test_negative_velo_decay_from_prior_starts(self):
        """Pitcher throws 95mph early, 90mph late => decay = -5.

        With 3 starts, game 3 feature should reflect mean decay of starts 1-2.
        """
        pitches_rows = []
        # Game 1: 100 pitches with high-early/low-late profile
        pitches_rows.extend(_generate_start_pitches(
            game_pk=1001, season=2025, game_date="2025-04-01",
            pitcher_id=_PITCHER_HOME, total_pitches=100,
            early_velo=95.0, mid_velo=93.0, late_velo=90.0,
        ))
        # Game 2: identical profile
        pitches_rows.extend(_generate_start_pitches(
            game_pk=1002, season=2025, game_date="2025-04-06",
            pitcher_id=_PITCHER_HOME, total_pitches=100,
            early_velo=95.0, mid_velo=93.0, late_velo=90.0,
        ))
        # Game 3: 100 pitches — flat velo (shouldn't matter; we check game 3 feature)
        pitches_rows.extend(_generate_start_pitches(
            game_pk=1003, season=2025, game_date="2025-04-11",
            pitcher_id=_PITCHER_HOME, total_pitches=100,
            early_velo=92.0, mid_velo=92.0, late_velo=92.0,
        ))

        # LHP filler to prevent empty-hand-split crash
        _add_both_hand_filler_pitches(pitches_rows, _GAME_DATES)

        pitches = pd.DataFrame(pitches_rows)
        game_frame = _make_game_frame([
            {
                "game_pk": 1001, "game_date": "2025-04-01",
                "home_team_id": _HOME_TEAM, "away_team_id": _AWAY_TEAM,
                "probable_pitcher_home_id": _PITCHER_HOME,
                "probable_pitcher_away_id": _PITCHER_AWAY,
            },
            {
                "game_pk": 1002, "game_date": "2025-04-06",
                "home_team_id": _HOME_TEAM, "away_team_id": _AWAY_TEAM,
                "probable_pitcher_home_id": _PITCHER_HOME,
                "probable_pitcher_away_id": _PITCHER_AWAY,
            },
            {
                "game_pk": 1003, "game_date": "2025-04-11",
                "home_team_id": _HOME_TEAM, "away_team_id": _AWAY_TEAM,
                "probable_pitcher_home_id": _PITCHER_HOME,
                "probable_pitcher_away_id": _PITCHER_AWAY,
            },
        ])

        result = compute_pitch_level_features(pitches, game_frame)
        game3 = result[result["game_pk"] == 1003]

        decay_val = game3["home_sp_tto_velo_decay_roll5"].values[0]

        # Expected: mean(-5.0, -5.0) = -5.0
        assert decay_val < 0, f"velo_decay should be negative, got {decay_val}"
        assert decay_val == pytest.approx(-5.0, abs=0.5), (
            f"Expected velo_decay ~ -5.0, got {decay_val}"
        )


# ---------------------------------------------------------------------------
# Test 3: K-BB% handedness split correctness
# ---------------------------------------------------------------------------

class TestKBBHandednessSplits:
    """Verify K% splits correctly by batter handedness."""

    def test_kpct_lhh_high_rhh_low(
        self, three_game_frame, kbb_pitches_strikeout_lhh_walk_rhh
    ):
        """Pitcher Ks all LHH, walks all RHH => kpct_vs_lhh ~ 1.0, kpct_vs_rhh ~ 0.0."""
        result = compute_pitch_level_features(
            kbb_pitches_strikeout_lhh_walk_rhh, three_game_frame
        )
        game3 = result[result["game_pk"] == 1003]

        kpct_lhh = game3["home_sp_kpct_vs_lhh_roll5"].values[0]
        kpct_rhh = game3["home_sp_kpct_vs_rhh_roll5"].values[0]

        assert kpct_lhh == pytest.approx(1.0, abs=0.01), (
            f"Expected kpct_vs_lhh=1.0, got {kpct_lhh}"
        )
        assert kpct_rhh == pytest.approx(0.0, abs=0.01), (
            f"Expected kpct_vs_rhh=0.0, got {kpct_rhh}"
        )

    def test_bbpct_lhh_low_rhh_high(
        self, three_game_frame, kbb_pitches_strikeout_lhh_walk_rhh
    ):
        """Pitcher Ks all LHH, walks all RHH => bbpct_vs_lhh ~ 0.0, bbpct_vs_rhh ~ 1.0."""
        result = compute_pitch_level_features(
            kbb_pitches_strikeout_lhh_walk_rhh, three_game_frame
        )
        game3 = result[result["game_pk"] == 1003]

        bbpct_lhh = game3["home_sp_bbpct_vs_lhh_roll5"].values[0]
        bbpct_rhh = game3["home_sp_bbpct_vs_rhh_roll5"].values[0]

        assert bbpct_lhh == pytest.approx(0.0, abs=0.01)
        assert bbpct_rhh == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Test 4: FIP splits formula correctness
# ---------------------------------------------------------------------------

class TestFIPFormula:
    """Verify FIP = (13*HR + 3*BB - 2*K) / IP + FIP_CONSTANT."""

    def test_fip_vs_rhh_known_value(self, three_game_frame, fip_pitches_known_stats):
        """Game 1: HR=1, BB=1, K=3, outs=7; Game 2: HR=0, BB=1, K=3, outs=8.

        Game 3 feature (roll5, shift(1)):
          HR=1, BB=2, K=6, outs=15, IP=5
          FIP = (13*1 + 3*2 - 2*6)/5 + 3.10 = 7/5 + 3.10 = 4.50
        """
        result = compute_pitch_level_features(
            fip_pitches_known_stats, three_game_frame
        )
        game3 = result[result["game_pk"] == 1003]

        fip_val = game3["home_sp_fip_vs_rhh_roll5"].values[0]
        expected_fip = (13 * 1 + 3 * 2 - 2 * 6) / 5.0 + FIP_CONSTANT  # 4.50
        assert fip_val == pytest.approx(expected_fip, abs=0.01), (
            f"Expected FIP={expected_fip:.3f}, got {fip_val:.3f}"
        )

    def test_fip_excludes_game3_data(self, three_game_frame, fip_pitches_known_stats):
        """Game 3 has 2 HR which would inflate FIP if leaked. Verify no leakage."""
        result = compute_pitch_level_features(
            fip_pitches_known_stats, three_game_frame
        )
        game3 = result[result["game_pk"] == 1003]

        fip_val = game3["home_sp_fip_vs_rhh_roll5"].values[0]
        # If game 3's 2 HR leaked: HR=3, BB=2, K=8, outs=19, IP=19/3
        # FIP would be (39+6-16)/(19/3) + 3.10 = 29/(6.33) + 3.10 ~ 7.68
        # Clean value should be 4.50
        clean_fip = 4.50
        assert fip_val == pytest.approx(clean_fip, abs=0.01)


# ---------------------------------------------------------------------------
# Test 5: Platoon wOBA correct team assignment (home vs away)
# ---------------------------------------------------------------------------

class TestPlatoonWOBATeamAssignment:
    """Home batters (bottom) hitting singles vs RHP should have higher wOBA
    than away batters (top) striking out vs RHP."""

    def test_home_woba_higher_than_away(self):
        """Home batters all single vs RHP; away batters all strike out vs RHP.

        The rolling wOBA window is 100 PA with min_periods=20. However, after
        drop_duplicates(["batter_id", "pitch_hand_code", "game_pk"]) in the module,
        only 1 PA per batter per game is kept. We need 25+ unique batters per side
        across enough games so that each batter accumulates 20+ PAs. With the
        dedup constraint, we use many games with many unique batters to exceed
        min_periods per batter.

        Alternative: use many games (8+) with the same batter pool so each batter
        gets 8 PAs total — but min_periods=20 means we'd still be NaN. Instead,
        we test at the 200pa window with min_periods=40 which is even harder.

        The correct approach: after drop_duplicates, each batter gets at most 1 PA
        per game. With 3 games, max 3 PAs per batter — far below min_periods=20.
        So wOBA will always be NaN with only 3 games of test data.

        For a meaningful test, we need to create enough games. We'll create 25 games
        with the same batter pool, giving each batter 25 PAs (above min_periods=20).
        """
        rows = []
        home_batters = list(range(301, 311))  # 10 home batters
        away_batters = list(range(401, 411))  # 10 away batters

        # 25 games so each batter accumulates 25 PAs (> min_periods=20 for roll100)
        games = [(2000 + i, f"2025-04-{i+1:02d}") for i in range(25)]

        ab_idx_home = 0
        ab_idx_away = 500

        for game_pk, game_date in games:
            for batter in home_batters:
                # 1 PA per batter per game — will survive drop_duplicates
                rows.extend(_make_pa_pitches(
                    game_pk=game_pk, season=2025, game_date=game_date,
                    pitcher_id=_PITCHER_AWAY, batter_id=batter,
                    at_bat_index=ab_idx_home, outcome_event="single",
                    bat_side_code="L", pitch_hand_code="R",
                    half_inning="bottom",
                    num_pitches=2,
                ))
                ab_idx_home += 1

            for batter in away_batters:
                rows.extend(_make_pa_pitches(
                    game_pk=game_pk, season=2025, game_date=game_date,
                    pitcher_id=_PITCHER_HOME, batter_id=batter,
                    at_bat_index=ab_idx_away, outcome_event="strikeout",
                    bat_side_code="R", pitch_hand_code="R",
                    half_inning="top",
                    num_pitches=3,
                ))
                ab_idx_away += 1

        # LHP+RHP filler
        _add_both_hand_filler_pitches(rows, games)

        pitches = pd.DataFrame(rows)
        game_frame = _make_game_frame([
            {
                "game_pk": gp, "game_date": gd,
                "home_team_id": _HOME_TEAM, "away_team_id": _AWAY_TEAM,
                "probable_pitcher_home_id": _PITCHER_HOME,
                "probable_pitcher_away_id": _PITCHER_AWAY,
            }
            for gp, gd in games
        ])

        result = compute_pitch_level_features(pitches, game_frame)
        # Check last game (game 25) — all batters should have 24 prior PAs
        last_game_pk = games[-1][0]
        last_game = result[result["game_pk"] == last_game_pk]

        home_woba = last_game["home_team_woba_vs_rhp_roll100pa"].values[0]
        away_woba = last_game["away_team_woba_vs_rhp_roll100pa"].values[0]

        # Home batters hit singles (wOBA weight 0.880); away batters strike out (0.0).
        assert not np.isnan(home_woba), (
            "home_team_woba should not be NaN with 24 prior PAs per batter (> min_periods=20)"
        )
        # away_woba should be 0.0 or NaN (all strikeouts = 0 woba numerator)
        if not np.isnan(away_woba):
            assert home_woba > away_woba, (
                f"Expected home_woba ({home_woba:.3f}) > away_woba ({away_woba:.3f})"
            )
        assert home_woba > 0.5, (
            f"Home wOBA should be substantial (all singles ~ 0.880), got {home_woba:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 6: NaN on first game (no prior data)
# ---------------------------------------------------------------------------

class TestFirstGameNaN:
    """A pitcher's very first game should produce NaN features, not 0.0."""

    def test_first_game_features_are_nan(self):
        """Single-game pitcher: all rolling features must be NaN."""
        rows = []
        # One game with 5 PAs for the pitcher vs LHH
        for ab_idx in range(5):
            rows.extend(_make_pa_pitches(
                game_pk=1001, season=2025, game_date="2025-04-01",
                pitcher_id=_PITCHER_HOME, batter_id=500 + ab_idx,
                at_bat_index=ab_idx, outcome_event="strikeout",
                bat_side_code="L", pitch_hand_code="R",
                half_inning="top",
            ))

        # Also add TTO-relevant pitches (100 pitches for the start)
        rows.extend(_generate_start_pitches(
            game_pk=1001, season=2025, game_date="2025-04-01",
            pitcher_id=_PITCHER_HOME, total_pitches=100,
        ))

        # LHP filler
        _add_both_hand_filler_pitches(rows, [(1001, "2025-04-01")])

        pitches = pd.DataFrame(rows)
        game_frame = _make_game_frame([{
            "game_pk": 1001, "game_date": "2025-04-01",
            "home_team_id": _HOME_TEAM, "away_team_id": _AWAY_TEAM,
            "probable_pitcher_home_id": _PITCHER_HOME,
            "probable_pitcher_away_id": _PITCHER_AWAY,
        }])

        result = compute_pitch_level_features(pitches, game_frame)

        # All rolling SP features for game 1 should be NaN (no prior data).
        sp_cols = [c for c in result.columns if c.startswith("home_sp_") and "roll" in c]
        assert len(sp_cols) > 0, "Expected at least some home SP rolling columns"

        for col in sp_cols:
            val = result[col].values[0]
            assert np.isnan(val), (
                f"First game feature {col} should be NaN, got {val}"
            )


# ---------------------------------------------------------------------------
# Test 7: game_type_code filter
# ---------------------------------------------------------------------------

class TestGameTypeFilter:
    """Spring training pitches (game_type_code='S') must not influence features."""

    def test_spring_training_excluded(self):
        """Pitcher dominates in spring training but is mediocre in regular season.

        Feature should reflect only regular-season data.
        """
        rows = []

        # Spring training game (game_type_code="S"): pitcher Ks everyone vs LHH
        for ab_idx in range(10):
            rows.extend(_make_pa_pitches(
                game_pk=9001, season=2025, game_date="2025-03-10",
                pitcher_id=_PITCHER_HOME, batter_id=500 + ab_idx,
                at_bat_index=ab_idx, outcome_event="strikeout",
                bat_side_code="L", pitch_hand_code="R",
                half_inning="top",
                game_type_code="S",  # Spring training
            ))

        # Regular season game 1: pitcher walks everyone vs LHH
        for ab_idx in range(5):
            rows.extend(_make_pa_pitches(
                game_pk=1001, season=2025, game_date="2025-04-01",
                pitcher_id=_PITCHER_HOME, batter_id=600 + ab_idx,
                at_bat_index=ab_idx + 20, outcome_event="walk",
                bat_side_code="L", pitch_hand_code="R",
                half_inning="top",
                game_type_code="R",
            ))

        # Regular season game 2: pitcher walks everyone vs LHH
        for ab_idx in range(5):
            rows.extend(_make_pa_pitches(
                game_pk=1002, season=2025, game_date="2025-04-06",
                pitcher_id=_PITCHER_HOME, batter_id=700 + ab_idx,
                at_bat_index=ab_idx + 40, outcome_event="walk",
                bat_side_code="L", pitch_hand_code="R",
                half_inning="top",
                game_type_code="R",
            ))

        # Regular season game 3 (target)
        for ab_idx in range(3):
            rows.extend(_make_pa_pitches(
                game_pk=1003, season=2025, game_date="2025-04-11",
                pitcher_id=_PITCHER_HOME, batter_id=800 + ab_idx,
                at_bat_index=ab_idx + 60, outcome_event="field_out",
                bat_side_code="L", pitch_hand_code="R",
                half_inning="top",
                game_type_code="R",
            ))

        # LHP filler for regular-season games only
        _add_both_hand_filler_pitches(rows, _GAME_DATES)

        pitches = pd.DataFrame(rows)
        game_frame = _make_game_frame([
            {
                "game_pk": 1001, "game_date": "2025-04-01",
                "home_team_id": _HOME_TEAM, "away_team_id": _AWAY_TEAM,
                "probable_pitcher_home_id": _PITCHER_HOME,
                "probable_pitcher_away_id": _PITCHER_AWAY,
            },
            {
                "game_pk": 1002, "game_date": "2025-04-06",
                "home_team_id": _HOME_TEAM, "away_team_id": _AWAY_TEAM,
                "probable_pitcher_home_id": _PITCHER_HOME,
                "probable_pitcher_away_id": _PITCHER_AWAY,
            },
            {
                "game_pk": 1003, "game_date": "2025-04-11",
                "home_team_id": _HOME_TEAM, "away_team_id": _AWAY_TEAM,
                "probable_pitcher_home_id": _PITCHER_HOME,
                "probable_pitcher_away_id": _PITCHER_AWAY,
            },
        ])

        result = compute_pitch_level_features(pitches, game_frame)
        game3 = result[result["game_pk"] == 1003]

        kpct_lhh = game3["home_sp_kpct_vs_lhh_roll5"].values[0]

        # If spring training were included, kpct would be > 0 (10K from ST).
        # With only regular-season data (all walks), kpct should be 0.0.
        assert kpct_lhh == pytest.approx(0.0, abs=0.01), (
            f"Spring training data leaked: kpct_vs_lhh={kpct_lhh}, expected 0.0"
        )


# ---------------------------------------------------------------------------
# Test 8: Output shape and column count
# ---------------------------------------------------------------------------

class TestOutputShape:
    """Verify output has correct shape and expected column names."""

    def test_output_rows_match_game_frame(self, three_game_frame):
        """Output must have same row count as game_frame."""
        rows = []
        for game_pk, game_date in _GAME_DATES:
            for ab_idx in range(3):
                rows.extend(_make_pa_pitches(
                    game_pk=game_pk, season=2025, game_date=game_date,
                    pitcher_id=_PITCHER_HOME, batter_id=500 + ab_idx,
                    at_bat_index=ab_idx, outcome_event="strikeout",
                    bat_side_code="R", pitch_hand_code="R",
                    half_inning="top",
                ))
        _add_both_hand_filler_pitches(rows, _GAME_DATES)
        pitches = pd.DataFrame(rows)

        result = compute_pitch_level_features(pitches, three_game_frame)
        assert len(result) == len(three_game_frame), (
            f"Output rows ({len(result)}) != game_frame rows ({len(three_game_frame)})"
        )

    def test_expected_columns_present(self, three_game_frame):
        """All expected feature column families should be in the output."""
        rows = []
        for game_pk, game_date in _GAME_DATES:
            rows.extend(_generate_start_pitches(
                game_pk=game_pk, season=2025, game_date=game_date,
                pitcher_id=_PITCHER_HOME, total_pitches=100,
            ))
            for ab_idx in range(5):
                rows.extend(_make_pa_pitches(
                    game_pk=game_pk, season=2025, game_date=game_date,
                    pitcher_id=_PITCHER_HOME, batter_id=500 + ab_idx,
                    at_bat_index=ab_idx + 30, outcome_event="strikeout",
                    bat_side_code="R", pitch_hand_code="R",
                    half_inning="top",
                ))
        _add_both_hand_filler_pitches(rows, _GAME_DATES)
        pitches = pd.DataFrame(rows)

        result = compute_pitch_level_features(pitches, three_game_frame)

        # TTO columns
        for side in ("home", "away"):
            for stat in ("velo_decay", "release_x_std", "release_z_std"):
                for w in (5, 10):
                    col = f"{side}_sp_tto_{stat}_roll{w}"
                    assert col in result.columns, f"Missing TTO column: {col}"

        # K-BB% columns
        for side in ("home", "away"):
            for metric in ("kpct", "bbpct", "kbb_diff"):
                for hand in ("lhh", "rhh"):
                    for w in (5, 10):
                        col = f"{side}_sp_{metric}_vs_{hand}_roll{w}"
                        assert col in result.columns, f"Missing K-BB column: {col}"

        # FIP columns
        for side in ("home", "away"):
            for hand in ("lhh", "rhh"):
                for w in (5, 10):
                    col = f"{side}_sp_fip_vs_{hand}_roll{w}"
                    assert col in result.columns, f"Missing FIP column: {col}"

        # Platoon wOBA columns
        for side in ("home", "away"):
            for hand in ("lhp", "rhp"):
                for window in (100, 200):
                    col = f"{side}_team_woba_vs_{hand}_roll{window}pa"
                    assert col in result.columns, f"Missing wOBA column: {col}"

        # Pitch-mix matchup columns
        for side in ("home", "away"):
            col = f"{side}_team_pitchmix_matchup_score_roll10"
            assert col in result.columns, f"Missing pitchmix column: {col}"

        # game_frame original columns should also be present
        for col in three_game_frame.columns:
            assert col in result.columns, f"Missing game_frame column: {col}"


# ---------------------------------------------------------------------------
# Test 9: Float32 dtype
# ---------------------------------------------------------------------------

class TestFloat32Dtype:
    """All feature output columns (except identifiers) must be float32."""

    def test_feature_columns_are_float32(self, three_game_frame):
        """Verify numeric feature columns are float32 for memory efficiency."""
        rows = []
        for game_pk, game_date in _GAME_DATES:
            rows.extend(_generate_start_pitches(
                game_pk=game_pk, season=2025, game_date=game_date,
                pitcher_id=_PITCHER_HOME, total_pitches=100,
            ))
            for ab_idx in range(5):
                rows.extend(_make_pa_pitches(
                    game_pk=game_pk, season=2025, game_date=game_date,
                    pitcher_id=_PITCHER_HOME, batter_id=500 + ab_idx,
                    at_bat_index=ab_idx + 30, outcome_event="strikeout",
                    bat_side_code="R", pitch_hand_code="R",
                    half_inning="top",
                ))
        _add_both_hand_filler_pitches(rows, _GAME_DATES)
        pitches = pd.DataFrame(rows)

        result = compute_pitch_level_features(pitches, three_game_frame)

        # Identify new feature columns (not in original game_frame)
        new_cols = set(result.columns) - set(three_game_frame.columns)

        # Exclude game_pk which is an integer identifier
        feature_cols = [c for c in new_cols if c != "game_pk"]
        assert len(feature_cols) > 0, "Expected new feature columns"

        for col in feature_cols:
            assert result[col].dtype == np.float32, (
                f"Column {col} has dtype {result[col].dtype}, expected float32"
            )


# ---------------------------------------------------------------------------
# Test 10: Missing pitcher (NaN) does not crash
# ---------------------------------------------------------------------------

class TestMissingPitcher:
    """If probable_pitcher is NaN (TBD), features should be NaN, not crash."""

    def test_nan_pitcher_produces_nan_features(self):
        """Game with NaN home pitcher should return NaN for home SP features."""
        rows = []
        # Need at least some pitch data so the function doesn't short-circuit
        for ab_idx in range(3):
            rows.extend(_make_pa_pitches(
                game_pk=1001, season=2025, game_date="2025-04-01",
                pitcher_id=_PITCHER_AWAY, batter_id=500 + ab_idx,
                at_bat_index=ab_idx, outcome_event="field_out",
                bat_side_code="R", pitch_hand_code="R",
                half_inning="top",
            ))
        _add_both_hand_filler_pitches(rows, [(1001, "2025-04-01")])
        pitches = pd.DataFrame(rows)

        game_frame = _make_game_frame([{
            "game_pk": 1001, "game_date": "2025-04-01",
            "home_team_id": _HOME_TEAM, "away_team_id": _AWAY_TEAM,
            "probable_pitcher_home_id": np.nan,  # TBD starter
            "probable_pitcher_away_id": _PITCHER_AWAY,
        }])

        # Should not raise
        result = compute_pitch_level_features(pitches, game_frame)

        # All home SP features should be NaN
        home_sp_cols = [
            c for c in result.columns
            if c.startswith("home_sp_") and "roll" in c
        ]
        for col in home_sp_cols:
            val = result[col].values[0]
            assert np.isnan(val), (
                f"Column {col} should be NaN for missing pitcher, got {val}"
            )

    def test_nan_pitcher_does_not_affect_other_side(self):
        """NaN home pitcher should not corrupt away pitcher features."""
        rows = []
        # Away pitcher has multiple games of data
        for game_pk, game_date in _GAME_DATES:
            for ab_idx in range(5):
                rows.extend(_make_pa_pitches(
                    game_pk=game_pk, season=2025, game_date=game_date,
                    pitcher_id=_PITCHER_AWAY, batter_id=500 + ab_idx,
                    at_bat_index=ab_idx, outcome_event="strikeout",
                    bat_side_code="R", pitch_hand_code="R",
                    half_inning="bottom",
                ))
        _add_both_hand_filler_pitches(rows, _GAME_DATES)
        pitches = pd.DataFrame(rows)

        game_frame = _make_game_frame([
            {
                "game_pk": gp, "game_date": gd,
                "home_team_id": _HOME_TEAM, "away_team_id": _AWAY_TEAM,
                "probable_pitcher_home_id": np.nan,
                "probable_pitcher_away_id": _PITCHER_AWAY,
            }
            for gp, gd in _GAME_DATES
        ])

        result = compute_pitch_level_features(pitches, game_frame)
        game3 = result[result["game_pk"] == 1003]

        # Away SP has 2 prior games of strikeouts => kpct should be non-NaN
        away_kpct = game3["away_sp_kpct_vs_rhh_roll5"].values[0]
        # With only 2 games meeting min_periods=2, game 3 should have a valid value
        assert not np.isnan(away_kpct), (
            "Away SP kpct should be valid despite NaN home pitcher"
        )
        assert away_kpct == pytest.approx(1.0, abs=0.01), (
            f"Away SP kpct should be ~1.0 (all Ks), got {away_kpct}"
        )
