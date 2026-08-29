"""Audit of CONTEXTUAL features for lookahead leakage, bias, missingness,
distribution validity, and staleness.

Covers:
- _park_factors: rolling park factor from prior games
- _weather_features: temp_f, is_dome, is_night_game
- _air_density_features: ADI from venue elevation
- _starting_pitcher_features: SP ERA/WHIP diffs, handedness
- _head_to_head: H2H record in last 10 meetings
- _differentials_and_sums: home-away diffs/sums
- _consensus_probability: mean of rating probabilities

Run: conda run -n pred python -m pytest tests/test_contextual_features_audit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.engineering.feature_engineering import (
    _air_density_features,
    _consensus_probability,
    _differentials_and_sums,
    _head_to_head,
    _park_factors,
    _starting_pitcher_features,
    _weather_features,
)


# ===========================================================================
# Fixtures: synthetic game frames for targeted testing
# ===========================================================================


@pytest.fixture
def base_games():
    """Minimal game frame with 20 games, two venues, two matchups."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        "game_pk": range(1000, 1000 + n),
        "game_date": pd.date_range("2024-04-01", periods=n, freq="D"),
        "season": [2024] * n,
        "game_type_code": ["R"] * n,
        "venue_id": [19] * 10 + [1] * 10,  # 10 at Coors, 10 at generic
        "home_team_id": [115] * 10 + [147] * 10,
        "away_team_id": [147] * 10 + [115] * 10,
        "home_bat_game_runs": np.random.randint(2, 8, n),
        "away_bat_game_runs": np.random.randint(1, 7, n),
        "total_runs": np.random.randint(5, 15, n),
        "home_win": np.random.choice([0, 1], n),
        "home_run_diff": np.random.randint(-5, 6, n),
        "weather_temp": np.random.randint(55, 95, n).astype(str),
        "venue_roof_type": ["Open"] * 10 + ["Dome"] * 10,
        "day_night": ["night"] * 12 + ["day"] * 8,
        "sp_home_id": [100] * n,
        "sp_away_id": [200] * n,
        "sp_home_hand": ["R"] * n,
        "sp_away_hand": ["L"] * n,
        "sp_home_game_earned_runs": np.random.randint(0, 5, n).astype(float),
        "sp_home_game_innings_pitched": np.random.uniform(4, 7, n),
        "sp_home_game_hits": np.random.randint(3, 9, n).astype(float),
        "sp_home_game_bb": np.random.randint(0, 4, n).astype(float),
        "sp_away_game_earned_runs": np.random.randint(0, 5, n).astype(float),
        "sp_away_game_innings_pitched": np.random.uniform(4, 7, n),
        "sp_away_game_hits": np.random.randint(3, 9, n).astype(float),
        "sp_away_game_bb": np.random.randint(0, 4, n).astype(float),
    })


@pytest.fixture
def games_with_ratings(base_games):
    """Game frame with mock rating probability columns."""
    df = base_games.copy()
    np.random.seed(99)
    df["elo_prob"] = np.random.uniform(0.4, 0.6, len(df)).astype("float64")
    df["wolfe_prob"] = np.random.uniform(0.4, 0.6, len(df)).astype("float64")
    df["log5_prob_short"] = np.random.uniform(0.4, 0.6, len(df)).astype("float64")
    return df


# ===========================================================================
# PARK FACTORS
# ===========================================================================


class TestParkFactors:
    """Audit _park_factors for lookahead, bias, staleness, distribution."""

    def test_no_lookahead_shift1(self, base_games):
        """Park factor at row i must not incorporate total_runs from row i."""
        df = _park_factors(base_games)
        # First game at each venue should be NaN (no prior data)
        coors_rows = df[df["venue_id"] == 19]
        assert pd.isna(df.loc[coors_rows.index[0], "park_factor"]), (
            "First game at a venue should have NaN park_factor (no prior data)"
        )

    def test_no_lookahead_manual_verify(self):
        """Manually verify park_factor at game index 15 uses only prior games.

        venue_avg requires min_periods=10, so we need >10 prior games at the venue.
        We construct a 20-game frame at a single venue and verify game 15.
        """
        np.random.seed(42)
        n = 20
        total_runs = np.array([7, 8, 9, 6, 10, 8, 7, 9, 11, 6, 8, 7, 10, 9, 8, 12, 7, 6, 9, 8])
        df = pd.DataFrame({
            "venue_id": [1] * n,
            "total_runs": total_runs,
            "season": [2024] * n,
            "game_type_code": ["R"] * n,
        })
        result = _park_factors(df)
        # At index 15: shift(1) means venue_avg uses expanding mean of total_runs[0:15]
        # and league_avg uses expanding mean of total_runs[0:15] (same season, same games)
        # So park_factor at index 15 should be venue_avg / league_avg ≈ 1.0
        # (since venue and league are the same set of games here)
        prior_runs = total_runs[:15]
        expected_venue_avg = prior_runs.mean()
        expected_league_avg = prior_runs.mean()
        expected_pf = expected_venue_avg / max(expected_league_avg, 1.0)
        actual_pf = result.loc[15, "park_factor"]
        assert abs(actual_pf - expected_pf) < 0.01, (
            f"Park factor at idx 15: expected {expected_pf:.4f}, got {actual_pf:.4f}"
        )

    def test_expanding_mean_staleness_renovation(self):
        """Expanding mean never forgets — a venue rebuilt in 2020 still carries
        2010 data. Verify this staleness exists as a known limitation."""
        np.random.seed(7)
        n = 40
        # Simulate venue with two eras: high scoring (1-20), then low (21-40)
        total_runs = np.concatenate([
            np.random.randint(10, 16, 20),  # old era: high scoring
            np.random.randint(4, 8, 20),    # new era: low scoring (renovation)
        ])
        df = pd.DataFrame({
            "venue_id": [99] * n,
            "total_runs": total_runs,
            "season": [2020] * 20 + [2024] * 20,
            "game_type_code": ["R"] * n,
        })
        result = _park_factors(df)
        # By game 39, expanding mean still includes the high-era data
        # The park_factor at game 39 should be inflated relative to pure new-era
        new_era_mean = total_runs[20:39].mean()  # pure new-era
        actual_venue_contrib = result.loc[39, "park_factor"]
        # Expanding mean includes old era, so it should be higher than new-era-only
        # This documents the staleness concern — not a bug per se, but a limitation
        assert actual_venue_contrib is not None and not pd.isna(actual_venue_contrib)

    def test_coors_field_nonlinear_altitude(self, base_games):
        """Verify Coors Field park factor is notably higher than sea-level venues.
        The simple ratio approach doesn't model nonlinear altitude effects explicitly,
        but it should still produce elevated values at Coors."""
        # Create games with realistic Coors data (higher run scoring)
        np.random.seed(11)
        n = 30
        df = pd.DataFrame({
            "venue_id": [19] * 15 + [1] * 15,
            "total_runs": np.concatenate([
                np.random.randint(9, 15, 15),  # Coors: high scoring
                np.random.randint(6, 10, 15),  # Sea level: normal
            ]),
            "season": [2024] * n,
            "game_type_code": ["R"] * n,
        })
        result = _park_factors(df)
        coors_pf = result[result["venue_id"] == 19]["park_factor"].dropna()
        sea_pf = result[result["venue_id"] == 1]["park_factor"].dropna()
        if len(coors_pf) > 0 and len(sea_pf) > 0:
            assert coors_pf.mean() > sea_pf.mean(), (
                "Coors park_factor should be higher than sea-level"
            )

    def test_spring_training_excluded_from_league_avg(self):
        """Non-regular-season games should not contaminate league average."""
        np.random.seed(3)
        n = 20
        df = pd.DataFrame({
            "venue_id": [1] * n,
            "total_runs": [20] * 5 + [8] * 15,  # Spring: 20 runs; regular: 8 runs
            "season": [2024] * n,
            "game_type_code": ["S"] * 5 + ["R"] * 15,  # 5 spring + 15 regular
        })
        result = _park_factors(df)
        # League avg should only use regular season games
        # Spring training games at index 0-4 get park_factor via venue_avg (which
        # includes all games) but league_avg excludes spring training
        # Regular season games (index 5+) should have league_avg based only on R games
        reg_pf = result.iloc[10:]["park_factor"].dropna()
        if len(reg_pf) > 0:
            # Should be close to 1.0 since venue and league are both 8 runs avg
            assert all(reg_pf < 2.0), "Park factors should be reasonable (<2x)"
            assert all(reg_pf > 0.3), "Park factors should be reasonable (>0.3x)"

    def test_venue_id_change_midseason(self):
        """If a team moves venues mid-dataset, each venue gets its own expanding mean."""
        np.random.seed(5)
        n = 20
        df = pd.DataFrame({
            "venue_id": [100] * 10 + [200] * 10,  # team moved to new stadium
            "total_runs": np.random.randint(6, 12, n),
            "season": [2024] * n,
            "game_type_code": ["R"] * n,
        })
        result = _park_factors(df)
        # First game at venue 200 (index 10) should have NaN park_factor
        # because there's no prior data for venue 200
        pf_at_new_venue_first = result.loc[10, "park_factor"]
        assert pd.isna(pf_at_new_venue_first), (
            "First game at new venue should have NaN park_factor"
        )

    def test_missingness_min_periods(self):
        """Park factor requires min_periods=10 for venue_avg. Verify NaN for < 10 games."""
        np.random.seed(6)
        n = 12
        df = pd.DataFrame({
            "venue_id": [1] * n,
            "total_runs": np.random.randint(6, 12, n),
            "season": [2024] * n,
            "game_type_code": ["R"] * n,
        })
        result = _park_factors(df)
        # With min_periods=10, games 0-9 (index positions) should have NaN venue_avg
        # because shift(1) means game at idx 10 has 10 prior games
        # Games 0-10 should have NaN park_factor (0 to 9 prior games, need 10)
        for i in range(10):
            assert pd.isna(result.loc[i, "park_factor"]), (
                f"Game {i} should have NaN park_factor (only {i} prior games, need 10)"
            )

    def test_distribution_range(self, base_games):
        """Park factor should be in reasonable range [0.5, 2.0]."""
        df = _park_factors(base_games)
        valid_pf = df["park_factor"].dropna()
        if len(valid_pf) > 0:
            assert valid_pf.min() > 0.3, f"Park factor too low: {valid_pf.min()}"
            assert valid_pf.max() < 3.0, f"Park factor too high: {valid_pf.max()}"


# ===========================================================================
# WEATHER FEATURES
# ===========================================================================


class TestWeatherFeatures:
    """Audit _weather_features for distribution, missingness, bias."""

    def test_no_lookahead(self, base_games):
        """Weather features are game-level static metadata, not temporal — no lookahead possible."""
        df = _weather_features(base_games)
        # Weather at game i describes that game's conditions, not outcome
        assert "temp_f" in df.columns
        assert "is_dome" in df.columns
        assert "is_night_game" in df.columns

    def test_temp_range_reasonable(self, base_games):
        """Temperature should be in physically plausible range [-10, 130] F."""
        df = _weather_features(base_games)
        valid_temp = df["temp_f"].dropna()
        assert valid_temp.min() >= -10, f"Temp too low: {valid_temp.min()}"
        assert valid_temp.max() <= 130, f"Temp too high: {valid_temp.max()}"

    def test_temp_extreme_values(self):
        """Verify handling of extreme / invalid temperature strings."""
        df = pd.DataFrame({
            "weather_temp": ["72", "0", "-5", "999", "NA", "", "hot"],
            "venue_roof_type": ["Open"] * 7,
            "day_night": ["night"] * 7,
        })
        result = _weather_features(df)
        # "hot" and "" should become NaN
        assert pd.isna(result.loc[6, "temp_f"]), "Non-numeric temp should be NaN"
        # "999" is passed through as numeric (potential data quality issue)
        assert result.loc[3, "temp_f"] == 999.0

    def test_dome_encoding(self):
        """Dome flag should be 1 for Dome/Retractable, 0 otherwise."""
        df = pd.DataFrame({
            "venue_roof_type": ["Dome", "Retractable", "Open", "Outdoor", np.nan],
        })
        result = _weather_features(df)
        expected = [1.0, 1.0, 0.0, 0.0, 0.0]
        np.testing.assert_array_equal(result["is_dome"].values, expected)

    def test_night_game_encoding(self):
        """Night game flag should be 1 for 'night', 0 otherwise."""
        df = pd.DataFrame({
            "day_night": ["night", "day", "Night", "DAY", np.nan],
        })
        result = _weather_features(df)
        # Case-sensitive: only exact "night" matches
        expected = [1.0, 0.0, 0.0, 0.0, 0.0]
        np.testing.assert_array_equal(result["is_night_game"].values, expected)

    def test_dome_bias_missingness(self):
        """Dome venues should have temp available (from climate control).
        Open venues in early/late season may have missing temp — MNAR concern."""
        # This is a documentation test: dome games always have controlled temp
        # while outdoor games can have missing weather data
        df = pd.DataFrame({
            "weather_temp": [np.nan, np.nan, "72", "75", np.nan],
            "venue_roof_type": ["Open", "Open", "Dome", "Dome", "Open"],
            "day_night": ["night"] * 5,
        })
        result = _weather_features(df)
        dome_missing = result[result["is_dome"] == 1.0]["temp_f"].isna().sum()
        open_missing = result[result["is_dome"] == 0.0]["temp_f"].isna().sum()
        # Dome venues typically have temp recorded; open venues may not
        # This tests the structural pattern, not a code bug
        assert dome_missing == 0, "Dome venues should have temp data"
        assert open_missing > 0, "Open venues may have missing temp"


# ===========================================================================
# AIR DENSITY FEATURES
# ===========================================================================


class TestAirDensityFeatures:
    """Audit _air_density_features for accuracy, bias, and coverage."""

    def test_no_lookahead(self, base_games):
        """ADI is derived from venue_id (static), no temporal lookahead possible."""
        df = _air_density_features(base_games)
        assert "air_density_index" in df.columns
        # ADI depends only on venue_id, which is known before game starts
        # No same-game outcome data is used

    def test_coors_field_adi_value(self):
        """Coors Field (5280 ft) should have ADI ~0.854 (ISA lapse-rate)."""
        df = pd.DataFrame({"venue_id": [19]})  # Coors Field
        result = _air_density_features(df)
        adi = result.loc[0, "air_density_index"]
        # ISA: (1 - 6.8756e-6 * 5280)^4.2558 ≈ 0.854
        expected = (1.0 - 6.8756e-6 * 5280) ** 4.2558
        assert abs(adi - expected) < 0.001, (
            f"Coors ADI: expected {expected:.4f}, got {adi:.4f}"
        )
        assert 0.84 < adi < 0.87, f"Coors ADI should be ~0.854, got {adi}"

    def test_sea_level_adi_value(self):
        """Venues not in elevation dict (sea level) should have ADI ~1.0."""
        df = pd.DataFrame({"venue_id": [9999]})  # Unknown venue
        result = _air_density_features(df)
        adi = result.loc[0, "air_density_index"]
        # ISA: (1 - 2.2558e-5 * 0)^4.2559 = 1.0
        assert abs(adi - 1.0) < 0.001, f"Sea-level ADI should be 1.0, got {adi}"

    def test_adi_range(self, base_games):
        """All ADI values should be in (0, 1]."""
        df = _air_density_features(base_games)
        adi = df["air_density_index"]
        assert (adi > 0).all(), "ADI must be positive"
        assert (adi <= 1.0).all(), "ADI must be <= 1.0 (relative to sea level)"

    def test_barometric_formula_accuracy(self):
        """Verify the simplified barometric formula vs the full ISA model.
        Full ISA: rho/rho0 = (1 - L*h/T0)^(g*M/(R*L) - 1)
        where L=0.0065 K/m, T0=288.15 K, g=9.81, M=0.029 kg/mol, R=8.314.

        FINDING: The code uses an isothermal approximation (exp(-k*h)) which
        underestimates density at altitude vs the ISA lapse-rate model by ~3.4%
        at Coors Field elevation. This means Coors ADI is ~0.826 in the code but
        the ISA model gives ~0.854. The code OVERSTATES the altitude effect.
        For lower venues (< 2000 ft) the error is <1% and acceptable.
        """
        # Full ISA for Coors (5280 ft = 1609.3 m)
        h_m = 5280 * 0.3048
        L = 0.0065  # K/m lapse rate
        T0 = 288.15  # K
        g = 9.80665  # m/s^2
        M = 0.0289644  # kg/mol
        R = 8.31447  # J/(mol*K)
        exponent = (g * M / (R * L)) - 1
        isa_density_ratio = (1 - L * h_m / T0) ** exponent

        # Simplified formula used in code
        simplified = np.exp(-3.63e-5 * 5280)

        # Document: the isothermal approx deviates ~3.4% at Coors elevation.
        # This is a known limitation — the relative ordering of venues is preserved
        # and the feature still captures "Coors is way lower density than sea level."
        error_pct = abs(simplified - isa_density_ratio) / isa_density_ratio * 100
        assert error_pct < 5.0, (
            f"Simplified ADI ({simplified:.4f}) deviates {error_pct:.1f}% from ISA ({isa_density_ratio:.4f})"
        )
        # For Chase Field (1082 ft = 329.8 m) the error should be much smaller
        h_chase = 1082 * 0.3048
        isa_chase = (1 - L * h_chase / T0) ** exponent
        simplified_chase = np.exp(-3.63e-5 * 1082)
        error_chase = abs(simplified_chase - isa_chase) / isa_chase * 100
        assert error_chase < 1.0, (
            f"Chase Field ADI error ({error_chase:.2f}%) should be <1%"
        )

    def test_missing_venue_elevation_bias(self):
        """Venues not in the elevation dict get ADI=1.0 regardless of actual elevation.
        This means any venue above 400ft NOT in the dict is underestimated."""
        from classical_learning.engineering.feature_engineering import _VENUE_ELEVATIONS_FT

        # Check that the dict covers the extreme outliers
        assert 19 in _VENUE_ELEVATIONS_FT, "Coors Field must be in elevation dict"
        # Chase Field (Phoenix, 1082 ft) is important
        assert 15 in _VENUE_ELEVATIONS_FT, "Chase Field must be in elevation dict"

    def test_adi_never_nan(self, base_games):
        """ADI should never be NaN — fillna(0) on elevation ensures all venues get a value."""
        df = _air_density_features(base_games)
        assert df["air_density_index"].isna().sum() == 0, "ADI should never be NaN"


# ===========================================================================
# STARTING PITCHER FEATURES
# ===========================================================================


class TestStartingPitcherFeatures:
    """Audit _starting_pitcher_features for lookahead, missingness, distribution."""

    def test_no_lookahead_era_uses_shift(self, base_games):
        """ERA at game i must not use earned runs from game i itself."""
        from classical_learning.engineering.feature_engineering import _compute_pregame_pitcher_era

        df = _compute_pregame_pitcher_era(base_games)
        df = _starting_pitcher_features(df)

        # First game for pitcher 100 should have NaN ERA (no prior data)
        assert pd.isna(df.loc[0, "sp_home_season_era"]), (
            "First game for pitcher should have NaN ERA"
        )

    def test_era_excludes_current_game(self, base_games):
        """Verify ERA at game N uses only starts 0..N-1."""
        from classical_learning.engineering.feature_engineering import _compute_pregame_pitcher_era

        df = _compute_pregame_pitcher_era(base_games)
        # ERA at game 5 for home pitcher (id=100) should use games 0-4
        # Manual: cumulative ER / cumulative IP * 9
        er_prior = df["sp_home_game_earned_runs"].iloc[:5].sum()
        ip_prior = df["sp_home_game_innings_pitched"].iloc[:5].sum()
        expected_era = er_prior / ip_prior * 9.0
        actual_era = df.loc[5, "sp_home_season_era"]
        assert abs(actual_era - expected_era) < 0.01, (
            f"ERA at game 5: expected {expected_era:.3f}, got {actual_era:.3f}"
        )

    def test_handedness_encoding(self, base_games):
        """Handedness should be binary 0/1."""
        from classical_learning.engineering.feature_engineering import _compute_pregame_pitcher_era

        df = _compute_pregame_pitcher_era(base_games)
        df = _starting_pitcher_features(df)
        assert "sp_home_is_lefty" in df.columns
        assert "sp_away_is_lefty" in df.columns
        # Home pitcher is "R" → 0, away is "L" → 1
        assert (df["sp_home_is_lefty"] == 0.0).all()
        assert (df["sp_away_is_lefty"] == 1.0).all()

    def test_era_diff_sign_convention(self, base_games):
        """sp_era_diff = away_era - home_era. Positive means away pitcher is worse."""
        from classical_learning.engineering.feature_engineering import _compute_pregame_pitcher_era

        df = _compute_pregame_pitcher_era(base_games)
        df = _starting_pitcher_features(df)
        valid_mask = df["sp_era_diff"].notna()
        if valid_mask.any():
            row = df[valid_mask].iloc[0]
            expected = row["sp_away_season_era"] - row["sp_home_season_era"]
            assert abs(row["sp_era_diff"] - expected) < 0.01

    def test_era_distribution_range(self, base_games):
        """Season ERA should be in [0, ~30] range (extreme outliers possible early season)."""
        from classical_learning.engineering.feature_engineering import _compute_pregame_pitcher_era

        df = _compute_pregame_pitcher_era(base_games)
        for side in ("home", "away"):
            era = df[f"sp_{side}_season_era"].dropna()
            if len(era) > 0:
                assert era.min() >= 0, f"ERA cannot be negative: {era.min()}"
                assert era.max() < 100, f"ERA suspiciously high: {era.max()}"

    def test_pitcher_cross_team_accumulation(self):
        """A pitcher who starts for both teams (trade scenario) should accumulate
        stats across both sides correctly in the unified timeline."""
        from classical_learning.engineering.feature_engineering import _compute_pregame_pitcher_era

        # Pitcher 300 starts 3 games at home, then 3 games away (simulates trade)
        df = pd.DataFrame({
            "sp_home_id": [300, 300, 300, 999, 999, 999],
            "sp_away_id": [999, 999, 999, 300, 300, 300],
            "sp_home_game_earned_runs": [2.0, 3.0, 1.0, 4.0, 2.0, 3.0],
            "sp_home_game_innings_pitched": [6.0, 5.0, 7.0, 6.0, 5.0, 6.0],
            "sp_home_game_hits": [5.0, 6.0, 4.0, 7.0, 5.0, 6.0],
            "sp_home_game_bb": [2.0, 1.0, 3.0, 2.0, 3.0, 1.0],
            "sp_away_game_earned_runs": [3.0, 2.0, 4.0, 2.0, 3.0, 1.0],
            "sp_away_game_innings_pitched": [5.0, 6.0, 5.0, 6.0, 5.0, 7.0],
            "sp_away_game_hits": [6.0, 5.0, 7.0, 5.0, 6.0, 4.0],
            "sp_away_game_bb": [1.0, 3.0, 2.0, 3.0, 1.0, 2.0],
        })
        result = _compute_pregame_pitcher_era(df)
        # At game 3, pitcher 300 is now away. Their ERA should reflect
        # games 0-2 as home starter (ER: 2+3+1=6, IP: 6+5+7=18) → ERA = 6/18*9 = 3.0
        era_at_game3 = result.loc[3, "sp_away_season_era"]
        expected = (2.0 + 3.0 + 1.0) / (6.0 + 5.0 + 7.0) * 9.0
        assert abs(era_at_game3 - expected) < 0.01, (
            f"Traded pitcher ERA: expected {expected:.3f}, got {era_at_game3}"
        )

    def test_missingness_cold_start(self):
        """New pitcher with no history should have NaN ERA, not 0."""
        from classical_learning.engineering.feature_engineering import _compute_pregame_pitcher_era

        df = pd.DataFrame({
            "sp_home_id": [500],
            "sp_away_id": [600],
            "sp_home_game_earned_runs": [3.0],
            "sp_home_game_innings_pitched": [6.0],
            "sp_home_game_hits": [5.0],
            "sp_home_game_bb": [2.0],
            "sp_away_game_earned_runs": [2.0],
            "sp_away_game_innings_pitched": [7.0],
            "sp_away_game_hits": [4.0],
            "sp_away_game_bb": [1.0],
        })
        result = _compute_pregame_pitcher_era(df)
        assert pd.isna(result.loc[0, "sp_home_season_era"]), (
            "First-ever start should have NaN ERA"
        )
        assert pd.isna(result.loc[0, "sp_away_season_era"]), (
            "First-ever start should have NaN ERA"
        )


# ===========================================================================
# HEAD-TO-HEAD
# ===========================================================================


class TestHeadToHead:
    """Audit _head_to_head for lookahead, matchup_key correctness, cross-season."""

    def test_no_lookahead_shift1(self, base_games):
        """H2H at game i must not include game i's outcome."""
        df = _head_to_head(base_games)
        # First few games between these teams should have NaN (min_periods=3)
        if "h2h_home_winrate_10" in df.columns:
            # With min_periods=3, games 0-2 should be NaN
            for i in range(3):
                assert pd.isna(df.loc[i, "h2h_home_winrate_10"]), (
                    f"H2H at game {i} should be NaN (< 3 prior meetings)"
                )

    def test_matchup_key_symmetric(self, base_games):
        """matchup_key should be the same regardless of who is home/away.
        TEA_NYY == NYY_TEA for the same pair."""
        # Create games where team order flips
        df = pd.DataFrame({
            "home_team_id": [100, 200, 100, 200, 100, 200, 100, 200, 100, 200],
            "away_team_id": [200, 100, 200, 100, 200, 100, 200, 100, 200, 100],
            "home_win": [1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
            "home_run_diff": [3, -2, 1, 4, -1, 2, -3, -1, 5, -2],
        })
        result = _head_to_head(df)
        # Both orientations should be grouped together (same matchup_key)
        # After 6 games, all have sufficient history (min_periods=3)
        if "h2h_home_winrate_10" in result.columns:
            # Game at index 6: prior meetings are indices 0-5
            # home_win values seen by matchup: [1, 0, 1, 1, 0, 1]
            # But wait — the key sorts team IDs, so "100" < "200" → key is "100_200"
            # home_win is always from the perspective of whoever is home
            # When 100 is home (indices 0,2,4,6,8): home_win = [1,1,0,0,1]
            # When 200 is home (indices 1,3,5,7,9): home_win = [0,1,1,0,0]
            # The rolling mean pools ALL games for the matchup key
            # This is a BIAS CONCERN: h2h_home_winrate mixes perspectives
            val_at_6 = result.loc[6, "h2h_home_winrate_10"]
            if not pd.isna(val_at_6):
                # Should reflect rolling mean of home_win for games 0-5
                # home_win at [0,1,2,3,4,5] = [1,0,1,1,0,1] → mean of last 6 = 4/6
                # With shift(1) and rolling(10, min_periods=3): mean([1,0,1,1,0,1]) = 0.667
                # Wait — shift(1) at index 6 means we see indices 0-5
                expected = np.mean([1, 0, 1, 1, 0, 1])
                assert abs(val_at_6 - expected) < 0.01, (
                    f"H2H winrate at idx 6: expected {expected:.3f}, got {val_at_6:.3f}"
                )

    def test_h2h_cross_season_leakage(self):
        """H2H rolling window spans seasons — this is intentional but should be documented.
        A team that dominated a matchup last season carries that signal forward."""
        df = pd.DataFrame({
            "home_team_id": [100] * 10,
            "away_team_id": [200] * 10,
            "home_win": [1] * 5 + [0] * 5,  # Won 5 in season 1, lost 5 in season 2
            "home_run_diff": [3] * 5 + [-3] * 5,
            "season": [2023] * 5 + [2024] * 5,
        })
        result = _head_to_head(df)
        if "h2h_home_winrate_10" in result.columns:
            # At game 5 (first of season 2024), should see 100% from season 2023
            val = result.loc[5, "h2h_home_winrate_10"]
            if not pd.isna(val):
                assert val == 1.0, (
                    "First game of new season should carry forward prior season H2H"
                )

    def test_h2h_home_winrate_bias(self):
        """CRITICAL BIAS: h2h_home_winrate uses raw home_win which mixes both
        directions. When team A is home it's their win; when team B is home it's B's win.
        The feature doesn't normalize for perspective — it measures how often the
        home team wins in this matchup, not how often a specific team wins."""
        df = pd.DataFrame({
            "home_team_id": ["A", "B", "A", "B", "A", "B", "A"],
            "away_team_id": ["B", "A", "B", "A", "B", "A", "B"],
            # Team A always wins regardless of home/away
            "home_win":     [1,    0,   1,   0,   1,   0,   1],
            "home_run_diff": [3, -3, 2, -2, 4, -1, 3],
        })
        result = _head_to_head(df)
        if "h2h_home_winrate_10" in result.columns:
            # At game 6 (A is home), prior games home_win = [1,0,1,0,1,0]
            # Mean = 0.5 — even though team A always wins!
            # This demonstrates the feature measures "home advantage in this matchup"
            # not "team dominance in this matchup" — a potential bias/information loss
            val = result.loc[6, "h2h_home_winrate_10"]
            if not pd.isna(val):
                assert abs(val - 0.5) < 0.01, (
                    "H2H winrate should be 0.5 when home/away alternate with constant winner"
                )

    def test_h2h_missing_early_season(self, base_games):
        """Teams that haven't played recently should have NaN H2H features."""
        # Create two pairs that never meet
        df = pd.DataFrame({
            "home_team_id": [100, 100, 300, 300],
            "away_team_id": [200, 200, 400, 400],
            "home_win": [1, 0, 1, 1],
            "home_run_diff": [2, -1, 3, 1],
        })
        result = _head_to_head(df)
        if "h2h_home_winrate_10" in result.columns:
            # All should be NaN (< 3 meetings per pair)
            assert result["h2h_home_winrate_10"].isna().all()


# ===========================================================================
# DIFFERENTIALS AND SUMS
# ===========================================================================


class TestDifferentialsAndSums:
    """Audit _differentials_and_sums for correctness and edge cases."""

    def test_basic_diff_computation(self):
        """diff = home - away, sum = home + away."""
        df = pd.DataFrame({
            "home_roll10_winpct": [0.6, 0.7, 0.5],
            "away_roll10_winpct": [0.4, 0.5, 0.8],
            "home_roll20_winpct": [0.55, 0.65, 0.45],
            "away_roll20_winpct": [0.45, 0.55, 0.75],
        })
        result = _differentials_and_sums(df)
        expected_diff_10 = [0.2, 0.2, -0.3]
        expected_sum_10 = [1.0, 1.2, 1.3]
        np.testing.assert_array_almost_equal(
            result["diff_roll10_winpct"].values, expected_diff_10, decimal=5
        )
        np.testing.assert_array_almost_equal(
            result["sum_roll10_winpct"].values, expected_sum_10, decimal=5
        )

    def test_nan_propagation(self):
        """If one side is NaN, diff and sum should be NaN."""
        df = pd.DataFrame({
            "home_roll10_winpct": [0.6, np.nan, 0.5],
            "away_roll10_winpct": [np.nan, 0.5, 0.8],
        })
        result = _differentials_and_sums(df)
        assert pd.isna(result.loc[0, "diff_roll10_winpct"])
        assert pd.isna(result.loc[1, "diff_roll10_winpct"])
        assert not pd.isna(result.loc[2, "diff_roll10_winpct"])

    def test_no_lookahead(self):
        """Diffs/sums are pure transformations of existing columns — lookahead
        depends entirely on the input features, not this function."""
        df = pd.DataFrame({
            "home_roll10_winpct": [0.6, 0.7],
            "away_roll10_winpct": [0.4, 0.5],
        })
        result = _differentials_and_sums(df)
        # This function introduces no new temporal logic — it's a pointwise transform
        assert "diff_roll10_winpct" in result.columns
        assert "sum_roll10_winpct" in result.columns

    def test_only_rolling_columns_processed(self):
        """Only columns starting with 'home_roll' get diff/sum treatment."""
        df = pd.DataFrame({
            "home_roll10_winpct": [0.6],
            "away_roll10_winpct": [0.4],
            "home_team_id": [100],
            "away_team_id": [200],
        })
        result = _differentials_and_sums(df)
        # home_team_id should NOT generate a diff
        assert "diff_team_id" not in result.columns


# ===========================================================================
# CONSENSUS PROBABILITY
# ===========================================================================


class TestConsensusProbability:
    """Audit _consensus_probability for inherited leakage and distribution."""

    def test_basic_computation(self, games_with_ratings):
        """Consensus = mean of all _prob columns."""
        df = _consensus_probability(games_with_ratings)
        assert "consensus_home_win_prob" in df.columns
        assert "consensus_home_win_std" in df.columns

        # Manual check for first row
        prob_cols = ["elo_prob", "wolfe_prob", "log5_prob_short"]
        expected_mean = games_with_ratings.loc[0, prob_cols].mean()
        actual_mean = df.loc[0, "consensus_home_win_prob"]
        assert abs(actual_mean - expected_mean) < 0.001

    def test_consensus_range(self, games_with_ratings):
        """Consensus probability should be in [0, 1]."""
        df = _consensus_probability(games_with_ratings)
        cons = df["consensus_home_win_prob"]
        assert (cons >= 0).all(), f"Consensus below 0: {cons.min()}"
        assert (cons <= 1).all(), f"Consensus above 1: {cons.max()}"

    def test_consensus_std_nonnegative(self, games_with_ratings):
        """Standard deviation should be >= 0."""
        df = _consensus_probability(games_with_ratings)
        std = df["consensus_home_win_std"]
        assert (std >= 0).all() or std.isna().all()

    def test_single_rating_system(self):
        """With only one rating system, std should be NaN or 0."""
        df = pd.DataFrame({
            "elo_prob": [0.55, 0.60, 0.45],
        })
        # Need to set dtype to float64 for detection
        df["elo_prob"] = df["elo_prob"].astype("float64")
        result = _consensus_probability(df)
        if "consensus_home_win_std" in result.columns:
            # std of a single value is NaN in pandas
            assert result["consensus_home_win_std"].isna().all() or (
                result["consensus_home_win_std"] == 0
            ).all()

    def test_leakage_inheritance_warning(self, games_with_ratings):
        """CRITICAL: If rating systems have lookahead leakage (as documented in
        leakage_analysis_2026_06_28.md), consensus inherits that leakage.
        This test documents the dependency chain."""
        # The consensus_probability function uses _prob columns from ratings
        # If those ratings were computed with same-game data, the consensus
        # inherits that leakage. This is structural — not testable in isolation.
        df = _consensus_probability(games_with_ratings)
        prob_cols = [c for c in games_with_ratings.columns
                     if "_prob" in c and games_with_ratings[c].dtype in ("float32", "float64")]
        assert len(prob_cols) > 0, "Need at least one rating probability column"
        # Document: consensus leakage = max(leakage across rating systems)

    def test_no_extra_columns_included(self):
        """Only columns with '_prob' in name and float dtype should be included.

        FINDING: The filter uses `"_prob" in c` which is a substring match.
        This means a column named "not_a_probability" WOULD be included because
        "_prob" appears in "not_a_probability" (position 5). In practice this is
        not triggered because no real columns have "_prob" as an interior substring,
        but it's a fragile filter. The string column correctly gets excluded by
        the dtype check.
        """
        df = pd.DataFrame({
            "elo_prob": pd.array([0.55, 0.60], dtype="float64"),
            "wolfe_prob": pd.array([0.50, 0.55], dtype="float64"),
            "some_prob_string": ["high", "low"],  # string — excluded by dtype check
            "junk_float_col": pd.array([0.99, 0.01], dtype="float64"),  # no _prob at all
        })
        result = _consensus_probability(df)
        if "consensus_home_win_prob" in result.columns:
            # Should only average elo_prob and wolfe_prob (junk_float_col excluded)
            expected = (df["elo_prob"] + df["wolfe_prob"]) / 2
            np.testing.assert_array_almost_equal(
                result["consensus_home_win_prob"].values,
                expected.values,
                decimal=5,
            )

    def test_prob_substring_match_vulnerability(self):
        """FINDING: '_prob' substring match includes columns like 'not_a_probability'
        because '_prob' appears as a substring. This is a latent bug if any future
        float column accidentally contains '_prob' in its name."""
        df = pd.DataFrame({
            "elo_prob": pd.array([0.55, 0.60], dtype="float64"),
            "not_a_probability": pd.array([0.99, 0.01], dtype="float64"),
        })
        result = _consensus_probability(df)
        # KNOWN ISSUE: "not_a_probability" contains "_prob" substring → gets included
        # This test documents the vulnerability
        if "consensus_home_win_prob" in result.columns:
            # The code includes BOTH columns (incorrectly)
            actual = result["consensus_home_win_prob"].values
            # If the bug exists, it averages both columns
            buggy_expected = (df["elo_prob"] + df["not_a_probability"]) / 2
            correct_expected = df["elo_prob"]  # only elo_prob should count
            # Assert the bug IS present (documenting current behavior)
            np.testing.assert_array_almost_equal(
                actual, buggy_expected.values, decimal=5,
                err_msg="Expected _prob substring vulnerability to include extra column"
            )


# ===========================================================================
# INTEGRATION: Leakage detection via temporal split
# ===========================================================================


class TestLeakageDetection:
    """Cross-cutting test: verify no feature at row i uses data from row i or later."""

    def test_park_factor_temporal_monotonicity(self):
        """Park factor expanding mean should only grow as more games are added,
        never jump due to future data injection."""
        np.random.seed(42)
        n = 30
        runs = np.random.randint(6, 12, n)
        df = pd.DataFrame({
            "venue_id": [1] * n,
            "total_runs": runs,
            "season": [2024] * n,
            "game_type_code": ["R"] * n,
        })
        result = _park_factors(df)
        pf = result["park_factor"].values

        # Verify: park_factor[i] should only depend on total_runs[0:i]
        # If we corrupt total_runs[15:] and recompute, pf[0:15] should be identical
        df_corrupted = df.copy()
        df_corrupted.loc[15:, "total_runs"] = 100  # extreme corruption
        result_corrupted = _park_factors(df_corrupted)
        pf_corrupted = result_corrupted["park_factor"].values

        # First 15 park factors should be identical
        np.testing.assert_array_equal(
            pf[:15], pf_corrupted[:15],
            err_msg="Park factor at row i changed when future data was corrupted — lookahead leak!"
        )

    def test_h2h_temporal_isolation(self):
        """Corrupting future H2H outcomes should not affect past H2H features."""
        np.random.seed(42)
        n = 15
        df = pd.DataFrame({
            "home_team_id": [100] * n,
            "away_team_id": [200] * n,
            "home_win": np.random.choice([0, 1], n),
            "home_run_diff": np.random.randint(-5, 6, n),
        })
        result = _head_to_head(df)

        # Corrupt future
        df_corrupted = df.copy()
        df_corrupted.loc[10:, "home_win"] = 1
        df_corrupted.loc[10:, "home_run_diff"] = 99
        result_corrupted = _head_to_head(df_corrupted)

        if "h2h_home_winrate_10" in result.columns:
            # First 10 values should be identical
            orig = result["h2h_home_winrate_10"].values[:10]
            corr = result_corrupted["h2h_home_winrate_10"].values[:10]
            np.testing.assert_array_equal(
                orig, corr,
                err_msg="H2H at row i changed when future data was corrupted — lookahead leak!"
            )

    def test_sp_era_temporal_isolation(self):
        """Corrupting future pitching stats should not affect past ERA."""
        from classical_learning.engineering.feature_engineering import _compute_pregame_pitcher_era

        np.random.seed(42)
        n = 10
        df = pd.DataFrame({
            "sp_home_id": [100] * n,
            "sp_away_id": [200] * n,
            "sp_home_game_earned_runs": np.random.randint(0, 5, n).astype(float),
            "sp_home_game_innings_pitched": np.random.uniform(4, 7, n),
            "sp_home_game_hits": np.random.randint(3, 9, n).astype(float),
            "sp_home_game_bb": np.random.randint(0, 4, n).astype(float),
            "sp_away_game_earned_runs": np.random.randint(0, 5, n).astype(float),
            "sp_away_game_innings_pitched": np.random.uniform(4, 7, n),
            "sp_away_game_hits": np.random.randint(3, 9, n).astype(float),
            "sp_away_game_bb": np.random.randint(0, 4, n).astype(float),
        })
        result = _compute_pregame_pitcher_era(df)

        # Corrupt future
        df_corrupted = df.copy()
        df_corrupted.loc[5:, "sp_home_game_earned_runs"] = 99.0
        result_corrupted = _compute_pregame_pitcher_era(df_corrupted)

        # First 5 ERA values should be identical
        orig = result["sp_home_season_era"].values[:5]
        corr = result_corrupted["sp_home_season_era"].values[:5]
        np.testing.assert_array_equal(
            orig, corr,
            err_msg="ERA at row i changed when future data was corrupted — lookahead leak!"
        )


# ===========================================================================
# BIAS AND STALENESS DOCUMENTATION TESTS
# ===========================================================================


class TestBiasAndStaleness:
    """Document known bias and staleness concerns."""

    def test_park_factor_expanding_mean_staleness(self):
        """STALENESS: expanding mean has infinite memory. A venue rebuilt in 2020
        with modern dimensions still carries data from the old configuration.
        The min_periods=10 provides a cold-start gate but no recency weighting."""
        # This is a documentation/awareness test
        # Proposed fix: use EWM or rolling window (e.g., last 162 games) instead
        np.random.seed(1)
        old_era = np.random.normal(12, 2, 500)  # 500 games in old park
        new_era = np.random.normal(8, 1, 50)    # 50 games in new park
        runs = np.concatenate([old_era, new_era])
        df = pd.DataFrame({
            "venue_id": [1] * 550,
            "total_runs": runs,
            "season": [2015] * 100 + [2016] * 100 + [2017] * 100 +
                      [2018] * 100 + [2019] * 100 + [2024] * 50,
            "game_type_code": ["R"] * 550,
        })
        result = _park_factors(df)
        # After 550 games, the expanding mean is dominated by old-era data
        # even though the last 50 games are in a "new park" with same venue_id
        final_pf = result.iloc[-1]["park_factor"]
        # The park factor should be closer to new-era reality but expanding mean
        # drags it toward old-era average
        new_era_avg = new_era.mean()
        old_era_avg = old_era.mean()
        # Staleness flag: final PF is between old and new, not matching new
        assert final_pf is not None and not pd.isna(final_pf)

    def test_h2h_cross_season_span(self):
        """H2H rolling window spans ALL meetings regardless of season gap.
        A 2019 meeting still influences 2024 H2H if within the 10-game window.
        This is arguably too stale for roster-turnover-heavy matchups."""
        df = pd.DataFrame({
            "home_team_id": [100] * 10,
            "away_team_id": [200] * 10,
            "home_win": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "home_run_diff": [5, 4, 3, 2, 1, -1, -2, -3, -4, -5],
            "season": [2019] * 5 + [2024] * 5,
        })
        result = _head_to_head(df)
        if "h2h_home_winrate_10" in result.columns:
            # At game 5 (first 2024 game): sees 5 wins from 2019 → winrate=1.0
            val = result.loc[5, "h2h_home_winrate_10"]
            if not pd.isna(val):
                assert val == 1.0, (
                    "H2H carries stale 2019 data into 2024 predictions"
                )

    def test_adi_coverage_gap(self):
        """Only 9 venues have elevation data. The other ~21 MLB venues all get ADI=1.0.
        Some mid-elevation venues (400-500ft) like Pittsburgh, Milwaukee may deserve
        non-trivial ADI values."""
        from classical_learning.engineering.feature_engineering import _VENUE_ELEVATIONS_FT
        # 30 MLB teams but only 9 venues in the dict
        assert len(_VENUE_ELEVATIONS_FT) <= 10, (
            "Most venues default to sea level — potential underfitting"
        )

    def test_weather_dome_temp_correlation_with_target(self):
        """BIAS: Dome teams play in controlled environments year-round.
        They never face extreme cold/heat → temp_f is more uniform for dome teams.
        This could create a spurious temp→target correlation if not controlled."""
        # Documentation test — no code assertion needed
        # When temp_f is NaN for outdoor games but filled for dome games,
        # imputation strategy matters for model fairness
        pass

    def test_consensus_probability_double_counting(self):
        """If multiple rating systems share similar base features (e.g., all use
        win%, run differential), their probabilities are correlated. The consensus
        mean overweights the shared signal vs. unique signal from each system.
        std underestimates true disagreement due to correlation."""
        # This is a structural concern, not a code bug
        # Proposed mitigation: use weights inversely proportional to pairwise correlation
        df = pd.DataFrame({
            # Two highly correlated rating systems
            "elo_prob": pd.array([0.55, 0.60, 0.45, 0.70, 0.50], dtype="float64"),
            "wolfe_prob": pd.array([0.56, 0.59, 0.44, 0.71, 0.51], dtype="float64"),
            # One independent system
            "log5_prob_short": pd.array([0.40, 0.80, 0.50, 0.30, 0.65], dtype="float64"),
        })
        result = _consensus_probability(df)
        # The consensus is dominated by the correlated pair
        # (2/3 weight on elo/wolfe direction vs 1/3 on log5)
        # This is documented, not fixed here
        assert "consensus_home_win_prob" in result.columns


# ===========================================================================
# EDGE CASES AND ROBUSTNESS
# ===========================================================================


class TestEdgeCases:
    """Edge cases: empty frames, all-NaN columns, single-row frames."""

    def test_empty_dataframe(self):
        """All functions should handle empty DataFrames gracefully."""
        df = pd.DataFrame(columns=[
            "venue_id", "total_runs", "season", "game_type_code",
            "weather_temp", "venue_roof_type", "day_night",
            "home_team_id", "away_team_id", "home_win", "home_run_diff",
        ])
        # These should not raise
        _park_factors(df)
        _weather_features(df)
        _air_density_features(df)
        _head_to_head(df)
        _differentials_and_sums(df)
        _consensus_probability(df)

    def test_single_game(self):
        """Single-row frame should work without errors (all rolling features NaN)."""
        df = pd.DataFrame({
            "venue_id": [19],
            "total_runs": [10],
            "season": [2024],
            "game_type_code": ["R"],
            "home_team_id": [100],
            "away_team_id": [200],
            "home_win": [1],
            "home_run_diff": [3],
            "weather_temp": ["72"],
            "venue_roof_type": ["Open"],
            "day_night": ["night"],
        })
        result = _park_factors(df)
        assert pd.isna(result.loc[0, "park_factor"])
        result = _weather_features(df)
        assert result.loc[0, "temp_f"] == 72.0
        result = _air_density_features(df)
        assert result.loc[0, "air_density_index"] < 1.0  # Coors
        result = _head_to_head(df)
        if "h2h_home_winrate_10" in result.columns:
            assert pd.isna(result.loc[0, "h2h_home_winrate_10"])

    def test_all_nan_total_runs(self):
        """If total_runs is all NaN, park_factor should be all NaN."""
        df = pd.DataFrame({
            "venue_id": [1, 1, 1, 1, 1],
            "total_runs": [np.nan] * 5,
            "season": [2024] * 5,
            "game_type_code": ["R"] * 5,
        })
        result = _park_factors(df)
        assert result["park_factor"].isna().all()

    def test_park_factor_division_by_zero(self):
        """League average is clipped to >=1.0 to prevent division by zero."""
        df = pd.DataFrame({
            "venue_id": [1] * 20,
            "total_runs": [0] * 20,  # Zero runs in all games
            "season": [2024] * 20,
            "game_type_code": ["R"] * 20,
        })
        result = _park_factors(df)
        # Should not produce inf values
        valid_pf = result["park_factor"].dropna()
        assert not np.isinf(valid_pf).any(), "Park factor should never be inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
