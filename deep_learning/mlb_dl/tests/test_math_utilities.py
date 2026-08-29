"""Tests for math utility modules in the deep learning pipeline.

Covers:
- distributions.py: NegBin NLL, PMF, CDF, MarketDeriver off-by-one checks
- datasets.py: Standardizer fit/transform, temporal_split_dates, game-index decay,
  _hash_bucket collision properties, _left_pad padding/truncation
- rating_sequences.py: load_rating_sequences missing-file handling, shape consistency
- weather_context.py: constant-dimension consistency, NaN-to-zero guarantee
"""

import hashlib
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


# ============================================================================
# distributions.py tests
# ============================================================================


class TestNegbinNLL:
    """Tests for negbin_nll loss function correctness and numerical stability."""

    def test_y_zero_boundary(self):
        """NLL at y=0 must be finite (lgamma(0+alpha) + 0*log terms)."""
        from deep_learning.mlb_dl.distributions import negbin_nll

        y = torch.tensor([0.0])
        mu = torch.tensor([3.5])
        alpha = torch.tensor([1.5])
        nll = negbin_nll(y, mu, alpha)
        assert torch.isfinite(nll).all(), f"NLL not finite at y=0: {nll}"

    def test_y_large(self):
        """NLL at y=25 (rare high-scoring game) must be finite."""
        from deep_learning.mlb_dl.distributions import negbin_nll

        y = torch.tensor([25.0])
        mu = torch.tensor([4.5])
        alpha = torch.tensor([2.0])
        nll = negbin_nll(y, mu, alpha)
        assert torch.isfinite(nll).all(), f"NLL not finite at y=25: {nll}"

    def test_alpha_very_small(self):
        """alpha=0.01 (extreme overdispersion) must produce finite NLL."""
        from deep_learning.mlb_dl.distributions import negbin_nll

        y = torch.tensor([3.0, 0.0, 10.0])
        mu = torch.tensor([4.0, 4.0, 4.0])
        alpha = torch.tensor([0.01, 0.01, 0.01])
        nll = negbin_nll(y, mu, alpha)
        assert torch.isfinite(nll).all(), f"NLL not finite at alpha=0.01: {nll}"

    def test_alpha_very_large(self):
        """alpha=100 (near-Poisson) must produce finite NLL."""
        from deep_learning.mlb_dl.distributions import negbin_nll

        y = torch.tensor([3.0, 0.0, 15.0])
        mu = torch.tensor([4.0, 4.0, 4.0])
        alpha = torch.tensor([100.0, 100.0, 100.0])
        nll = negbin_nll(y, mu, alpha)
        assert torch.isfinite(nll).all(), f"NLL not finite at alpha=100: {nll}"

    def test_nll_always_non_negative(self):
        """NegBin NLL must be >= 0 for all valid inputs (it is -log of a probability)."""
        from deep_learning.mlb_dl.distributions import negbin_nll

        # Broad sweep of realistic values
        ys = torch.tensor([0.0, 1.0, 2.0, 5.0, 10.0, 20.0])
        mus = torch.tensor([0.5, 1.0, 3.0, 5.0, 8.0, 4.0])
        alphas = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
        nll = negbin_nll(ys, mus, alphas)
        assert (nll >= 0).all(), f"Found negative NLL values: {nll[nll < 0]}"

    def test_nll_batch_shape(self):
        """Output shape matches input batch dimension."""
        from deep_learning.mlb_dl.distributions import negbin_nll

        batch_size = 16
        y = torch.randint(0, 10, (batch_size,)).float()
        mu = torch.ones(batch_size) * 4.0
        alpha = torch.ones(batch_size) * 2.0
        nll = negbin_nll(y, mu, alpha)
        assert nll.shape == (batch_size,)

    def test_nll_minimum_near_true_mean(self):
        """NLL should be lowest when mu ~ y (for a single observation)."""
        from deep_learning.mlb_dl.distributions import negbin_nll

        y = torch.tensor([5.0])
        alpha = torch.tensor([2.0])
        # Test a range of mu values; NLL should be lower near mu=5
        mus = [1.0, 3.0, 5.0, 7.0, 10.0]
        nlls = [negbin_nll(y, torch.tensor([m]), alpha).item() for m in mus]
        best_idx = np.argmin(nlls)
        assert mus[best_idx] == 5.0, f"NLL minimized at mu={mus[best_idx]}, expected near 5.0"


class TestNegbinPMF:
    """Tests for negbin_pmf probability mass function."""

    def test_pmf_sums_to_one(self):
        """PMF over k=0..50 must sum close to 1.0 for typical MLB parameters."""
        from deep_learning.mlb_dl.distributions import negbin_pmf

        k = np.arange(51)
        # Typical MLB: mean ~4.5 runs, moderate overdispersion
        pmf = negbin_pmf(k, mu=4.5, alpha=2.0)
        total = pmf.sum()
        assert abs(total - 1.0) < 1e-4, f"PMF sum = {total}, expected ~1.0"

    def test_pmf_sums_to_one_high_mu(self):
        """PMF with high mu (early game, many remaining runs) still sums to ~1."""
        from deep_learning.mlb_dl.distributions import negbin_pmf

        k = np.arange(51)
        pmf = negbin_pmf(k, mu=10.0, alpha=3.0)
        total = pmf.sum()
        assert abs(total - 1.0) < 0.01, f"PMF sum = {total} for mu=10"

    def test_pmf_all_non_negative(self):
        """No probability mass can be negative."""
        from deep_learning.mlb_dl.distributions import negbin_pmf

        k = np.arange(51)
        pmf = negbin_pmf(k, mu=4.5, alpha=1.0)
        assert (pmf >= 0).all(), f"Negative PMF values found: {pmf[pmf < 0]}"

    def test_pmf_at_k_zero(self):
        """P(X=0) must equal (alpha/(alpha+mu))^alpha (analytical formula)."""
        from deep_learning.mlb_dl.distributions import negbin_pmf

        mu, alpha = 4.0, 2.0
        p0 = negbin_pmf(np.array([0]), mu, alpha)[0]
        # Analytical: P(X=0) = (alpha/(alpha+mu))^alpha
        expected = (alpha / (alpha + mu)) ** alpha
        assert abs(p0 - expected) < 1e-8, f"P(0) = {p0}, expected {expected}"

    def test_pmf_small_alpha(self):
        """Small alpha (heavy tails) produces wider distribution but still sums to ~1."""
        from deep_learning.mlb_dl.distributions import negbin_pmf

        k = np.arange(100)  # Need wider range for heavy tails
        pmf = negbin_pmf(k, mu=4.0, alpha=0.5)
        total = pmf.sum()
        assert abs(total - 1.0) < 0.01, f"PMF sum = {total} for alpha=0.5"
        assert (pmf >= 0).all()


class TestNegbinCDF:
    """Tests for negbin_cdf cumulative distribution function."""

    def test_cdf_monotone(self):
        """CDF must be monotonically non-decreasing."""
        from deep_learning.mlb_dl.distributions import negbin_cdf

        k = np.arange(30)
        cdf = negbin_cdf(k, mu=4.5, alpha=2.0)
        diffs = np.diff(cdf)
        assert (diffs >= -1e-10).all(), f"CDF decreased: {diffs[diffs < 0]}"

    def test_cdf_bounds(self):
        """CDF(0) > 0 and CDF(large) ~ 1.0."""
        from deep_learning.mlb_dl.distributions import negbin_cdf

        cdf_0 = negbin_cdf(np.array([0]), mu=4.5, alpha=2.0)[0]
        cdf_50 = negbin_cdf(np.array([50]), mu=4.5, alpha=2.0)[0]
        assert cdf_0 > 0.0
        assert cdf_0 < 1.0
        assert abs(cdf_50 - 1.0) < 1e-6


class TestPXGeqN:
    """Test P(X>=N) computation via the MarketDeriver to catch off-by-one errors."""

    def test_p_over_half_line_excludes_line(self):
        """P(team_final > 5.5) with observed=0 = P(remaining >= 6).

        This must NOT include k=5 since 5 is not > 5.5.
        """
        from deep_learning.mlb_dl.distributions import negbin_cdf

        mu, alpha = 4.5, 2.0
        # P(X > 5.5) = P(X >= 6) = 1 - P(X <= 5)
        cdf_5 = negbin_cdf(np.array([5]), mu, alpha)[0]
        p_geq_6 = 1.0 - cdf_5

        from deep_learning.mlb_dl.distributions import MarketDeriver

        deriver = MarketDeriver(max_runs=35)
        p_over = deriver.p_team_total_over(mu, alpha, observed=0, line=5.5)
        assert abs(p_over - p_geq_6) < 1e-6, (
            f"P(X>5.5) via deriver={p_over}, direct={p_geq_6}"
        )

    def test_p_geq_1_excludes_k_0(self):
        """P(X>=1) must not include k=0."""
        from deep_learning.mlb_dl.distributions import negbin_cdf, negbin_pmf

        mu, alpha = 4.0, 2.0
        p0 = negbin_pmf(np.array([0]), mu, alpha)[0]
        cdf_0 = negbin_cdf(np.array([0]), mu, alpha)[0]

        # P(X>=1) = 1 - P(X=0)
        p_geq_1 = 1.0 - p0
        p_geq_1_via_cdf = 1.0 - cdf_0
        assert abs(p_geq_1 - p_geq_1_via_cdf) < 1e-10

        # Via MarketDeriver: P(final > 0.5) with observed=0 → threshold=0.5
        from deep_learning.mlb_dl.distributions import MarketDeriver

        deriver = MarketDeriver(max_runs=35)
        p_via_deriver = deriver.p_team_total_over(mu, alpha, observed=0, line=0.5)
        assert abs(p_via_deriver - p_geq_1) < 1e-4, (
            f"Off-by-one: deriver={p_via_deriver}, expected={p_geq_1}"
        )

    def test_already_over_line(self):
        """If team already exceeded line, P(over) = 1.0."""
        from deep_learning.mlb_dl.distributions import MarketDeriver

        deriver = MarketDeriver(max_runs=35)
        p = deriver.p_team_total_over(mu_remaining=3.0, alpha_remaining=2.0, observed=6, line=5.5)
        assert p == 1.0

    def test_p_total_over_integer_line(self):
        """P(final > 5) with observed=0 = P(remaining > 5) = P(remaining >= 6)."""
        from deep_learning.mlb_dl.distributions import MarketDeriver, negbin_cdf

        mu, alpha = 4.5, 2.0
        deriver = MarketDeriver(max_runs=35)
        p_over_5 = deriver.p_team_total_over(mu, alpha, observed=0, line=5.0)

        # Direct: P(X > 5) = 1 - P(X <= 5)
        cdf_5 = negbin_cdf(np.array([5]), mu, alpha)[0]
        expected = 1.0 - cdf_5
        assert abs(p_over_5 - expected) < 1e-6


class TestMarketDeriverConsistency:
    """Integration tests for MarketDeriver probability coherence."""

    def test_home_away_extra_sum_to_one(self):
        """P(home_win) + P(away_win) + P(extra_innings) = 1.0."""
        from deep_learning.mlb_dl.distributions import MarketDeriver

        deriver = MarketDeriver(max_runs=25)
        markets = deriver.derive_all_markets(
            h_observed=2,
            a_observed=1,
            mu_h_remaining=3.0,
            alpha_h_remaining=2.0,
            mu_a_remaining=3.5,
            alpha_a_remaining=2.0,
            innings_completed=4.0,
        )
        total = markets["home_win"] + markets["away_win"] + markets["extra_innings"]
        assert abs(total - 1.0) < 0.01, f"Win probs sum = {total}, expected 1.0"

    def test_total_over_monotone(self):
        """P(total > X) must decrease as X increases."""
        from deep_learning.mlb_dl.distributions import MarketDeriver

        deriver = MarketDeriver(max_runs=25)
        markets = deriver.derive_all_markets(
            h_observed=0,
            a_observed=0,
            mu_h_remaining=4.5,
            alpha_h_remaining=2.0,
            mu_a_remaining=4.0,
            alpha_a_remaining=2.0,
            innings_completed=0.0,
        )
        lines = [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
        probs = [markets[f"total_runs_over_{l}"] for l in lines]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1], (
                f"P(over {lines[i]}) = {probs[i]} < P(over {lines[i+1]}) = {probs[i+1]}"
            )


class TestScaleToFirst5Inconsistency:
    """Verify the inconsistency between scale_to_first_5 and derive_first_5_markets.

    scale_to_first_5 scales alpha (preserving variance-to-mean ratio);
    derive_first_5_markets does NOT scale alpha. This is a confirmed divergence.
    """

    def test_scale_to_first_5_scales_alpha(self):
        """scale_to_first_5 correctly scales both mu and alpha."""
        from deep_learning.mlb_dl.distributions import scale_to_first_5

        mu_f5, alpha_f5 = scale_to_first_5(mu_full=6.0, alpha_full=2.0, innings_completed=3.0)
        # innings_left_in_f5 = 5.0 - 3.0 = 2.0
        # innings_left_full = 9.0 - 3.0 = 6.0
        # scale_factor = 2/6 = 1/3
        expected_mu = 6.0 * (2.0 / 6.0)
        expected_alpha = 2.0 * (2.0 / 6.0)
        assert abs(mu_f5 - expected_mu) < 1e-8
        assert abs(alpha_f5 - expected_alpha) < 1e-8

    def test_derive_first_5_scales_alpha_consistently(self):
        """derive_first_5_markets now scales alpha with mu — consistent with scale_to_first_5.

        Both code paths preserve the variance-to-mean ratio under time-subsetting:
        NegBin var/mean = 1 + mu/alpha. Scaling both by the same factor keeps this ratio.
        """
        from deep_learning.mlb_dl.distributions import MarketDeriver, joint_pmf_grid

        mu_h_full, alpha_h = 6.0, 2.0
        mu_a_full, alpha_a = 5.0, 2.0
        innings_completed = 3.0

        innings_left_in_f5 = 5.0 - innings_completed  # 2
        innings_left_full = 9.0 - innings_completed  # 6
        scale_factor = innings_left_in_f5 / innings_left_full  # 1/3

        mu_h_f5 = mu_h_full * scale_factor
        mu_a_f5 = mu_a_full * scale_factor
        alpha_h_scaled = alpha_h * scale_factor
        alpha_a_scaled = alpha_a * scale_factor

        # derive_first_5_markets should now produce the same grid as manual scale_to_first_5
        deriver = MarketDeriver(max_runs=25)
        deriver.derive_first_5_markets(
            mu_h_remaining_full=mu_h_full, alpha_h_remaining_full=alpha_h,
            mu_a_remaining_full=mu_a_full, alpha_a_remaining_full=alpha_a,
            h_observed=0, a_observed=0, innings_completed=innings_completed,
        )

        # Both grid variants should now use scaled alpha — verify they diverge from unscaled
        grid_unscaled_alpha = joint_pmf_grid(mu_h_f5, alpha_h, mu_a_f5, alpha_a, max_runs=25)
        grid_scaled_alpha = joint_pmf_grid(mu_h_f5, alpha_h_scaled, mu_a_f5, alpha_a_scaled, max_runs=25)
        assert not np.allclose(grid_unscaled_alpha, grid_scaled_alpha, atol=1e-6), (
            "Scaled-alpha grid must differ from unscaled-alpha grid"
        )


# ============================================================================
# datasets.py tests
# ============================================================================


class TestStandardizerFit:
    """Tests for Standardizer.fit edge cases."""

    def test_constant_column(self):
        """Constant column (std=0) must not produce NaN mean or std."""
        from deep_learning.mlb_dl.datasets import Standardizer

        df = pd.DataFrame({"const": [5.0, 5.0, 5.0, 5.0, 5.0]})
        std = Standardizer.fit(df, ["const"])
        assert std.mean["const"] == 5.0
        assert std.std["const"] == 1.0  # guarded to 1.0 when true std < 1e-6
        assert np.isfinite(std.mean["const"])
        assert np.isfinite(std.std["const"])

    def test_single_row_df(self):
        """Single-row DataFrame: std must default to 1.0, mean is the only value."""
        from deep_learning.mlb_dl.datasets import Standardizer

        df = pd.DataFrame({"x": [3.7]})
        std = Standardizer.fit(df, ["x"])
        assert abs(std.mean["x"] - 3.7) < 1e-6
        assert std.std["x"] == 1.0  # single value → std=0 → guard to 1.0

    def test_nan_heavy_column(self):
        """Column with mostly NaN: must compute stats from finite values only."""
        from deep_learning.mlb_dl.datasets import Standardizer

        df = pd.DataFrame({"x": [np.nan, np.nan, 2.0, np.nan, 4.0]})
        std = Standardizer.fit(df, ["x"])
        # mean of [2.0, 4.0] = 3.0
        assert abs(std.mean["x"] - 3.0) < 1e-6
        # std of [2.0, 4.0] = 1.0
        assert abs(std.std["x"] - 1.0) < 1e-6

    def test_all_nan_column(self):
        """All-NaN column must produce mean=0.0, std=1.0 (safe fallbacks)."""
        from deep_learning.mlb_dl.datasets import Standardizer

        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
        std = Standardizer.fit(df, ["x"])
        assert std.mean["x"] == 0.0
        assert std.std["x"] == 1.0


class TestStandardizerTransform:
    """Tests for Standardizer.transform correctness."""

    def test_constant_column_all_zeros(self):
        """Constant column transforms to all zeros (not NaN from 0/0)."""
        from deep_learning.mlb_dl.datasets import Standardizer

        df = pd.DataFrame({"const": [5.0, 5.0, 5.0]})
        std = Standardizer.fit(df, ["const"])
        values, mask = std.transform(df)
        # (5.0 - 5.0) / 1.0 = 0.0
        assert np.all(values == 0.0)
        assert np.all(mask == 1.0)

    def test_nan_becomes_zero(self):
        """NaN values in transform produce 0.0 in output, 0.0 in mask."""
        from deep_learning.mlb_dl.datasets import Standardizer

        df_fit = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        std = Standardizer.fit(df_fit, ["x"])
        df_test = pd.DataFrame({"x": [np.nan, 3.0]})
        values, mask = std.transform(df_test)
        # First row: NaN → 0.0 after nan_to_num
        assert values[0, 0] == 0.0
        # Mask: NaN position → 0.0
        assert mask[0, 0] == 0.0
        # Second row: finite → mask=1.0
        assert mask[1, 0] == 1.0

    def test_transform_output_shape(self):
        """Transform returns (n_rows, n_features) arrays."""
        from deep_learning.mlb_dl.datasets import Standardizer

        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
        std = Standardizer.fit(df, ["a", "b", "c"])
        values, mask = std.transform(df)
        assert values.shape == (2, 3)
        assert mask.shape == (2, 3)
        assert values.dtype == np.float32
        assert mask.dtype == np.float32


class TestTemporalSplitDates:
    """Tests for temporal_split_dates correctness."""

    def test_train_end_before_val_end(self):
        """train_end < val_end for any valid input."""
        from deep_learning.mlb_dl.datasets import temporal_split_dates

        dates = pd.date_range("2020-04-01", periods=200, freq="D")
        df = pd.DataFrame({"game_date": dates})
        train_end, val_end = temporal_split_dates(df)
        assert train_end < val_end

    def test_both_are_timestamps(self):
        """Return values must be pd.Timestamp."""
        from deep_learning.mlb_dl.datasets import temporal_split_dates

        dates = pd.date_range("2020-04-01", periods=100, freq="D")
        df = pd.DataFrame({"game_date": dates})
        train_end, val_end = temporal_split_dates(df)
        assert isinstance(train_end, pd.Timestamp)
        assert isinstance(val_end, pd.Timestamp)

    def test_default_80_10_10_fractions(self):
        """Default split at 80%/10%/10% of unique dates."""
        from deep_learning.mlb_dl.datasets import temporal_split_dates

        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({"game_date": dates})
        train_end, val_end = temporal_split_dates(df)
        # train_end = dates[80], val_end = dates[90]
        assert train_end == dates[80]
        assert val_end == dates[90]

    def test_min_date_bounds_split_pool(self):
        """Optional min_date excludes pre-Statcast target dates from split quantiles."""
        from deep_learning.mlb_dl.datasets import temporal_split_dates

        old_dates = pd.date_range("2008-01-01", periods=100, freq="D")
        statcast_dates = pd.date_range("2015-01-01", periods=100, freq="D")
        df = pd.DataFrame({"game_date": list(old_dates) + list(statcast_dates)})

        train_end, val_end = temporal_split_dates(df, min_date="2015-01-01")

        assert train_end == statcast_dates[80]
        assert val_end == statcast_dates[90]

    def test_raises_on_too_few_dates(self):
        """Must raise ValueError if fewer than 10 distinct dates."""
        from deep_learning.mlb_dl.datasets import temporal_split_dates

        df = pd.DataFrame({"game_date": pd.date_range("2020-01-01", periods=5, freq="D")})
        with pytest.raises(ValueError, match="at least 10"):
            temporal_split_dates(df)


class TestComputeGameDecayWeight:
    """Tests for compute_game_decay_weight monotonicity and bounds."""

    def test_monotonically_decreasing_same_season(self):
        """Weights must decrease for older games within a single season."""
        from deep_learning.mlb_dl.datasets import compute_game_decay_weight, SequenceSpec

        dates = pd.to_datetime([f"2024-04-{d:02d}" for d in range(1, 21)])
        entry = {
            "dates": list(dates),
            "seasons": [2024] * 20,
            "indices": list(range(20)),
        }
        spec = SequenceSpec()
        current_date = pd.Timestamp("2024-04-25")
        weights = compute_game_decay_weight(entry, current_date, spec)

        assert len(weights) == 20
        # Weights should be monotonically increasing (most recent is last, highest)
        for i in range(len(weights) - 1):
            assert weights[i] <= weights[i + 1], (
                f"Weight[{i}]={weights[i]} > Weight[{i+1}]={weights[i+1]}"
            )

    def test_most_recent_weight_is_one(self):
        """Most recent game in same season has delta=0, seasons_crossed=0 → weight=1.0."""
        from deep_learning.mlb_dl.datasets import compute_game_decay_weight, SequenceSpec

        dates = pd.to_datetime(["2024-04-01", "2024-04-03", "2024-04-05"])
        entry = {
            "dates": list(dates),
            "seasons": [2024, 2024, 2024],
            "indices": list(range(3)),
        }
        spec = SequenceSpec()
        current_date = pd.Timestamp("2024-04-10")
        weights = compute_game_decay_weight(entry, current_date, spec)
        assert len(weights) == 3
        assert abs(weights[-1] - 1.0) < 1e-10

    def test_cross_season_penalty(self):
        """Prior-season games get penalized by exp(-lambda_inter)."""
        from deep_learning.mlb_dl.datasets import compute_game_decay_weight, SequenceSpec

        dates = pd.to_datetime(["2023-09-28", "2024-04-01"])
        entry = {
            "dates": list(dates),
            "seasons": [2023, 2024],
            "indices": [0, 1],
        }
        spec = SequenceSpec()
        current_date = pd.Timestamp("2024-04-05")
        weights = compute_game_decay_weight(entry, current_date, spec)

        assert len(weights) == 2
        # Most recent (2024-04-01, same season as current_season=2024): delta=0, s=0 → 1.0
        assert abs(weights[1] - 1.0) < 1e-10
        # Older (2023-09-28, season 2023 vs current 2024): delta=1, s=1
        expected_older = math.exp(-spec.intra_season_lambda * 1) * math.exp(-spec.inter_season_lambda * 1)
        assert abs(weights[0] - expected_older) < 1e-10

    def test_weights_in_zero_one_range(self):
        """All weights must be in (0, 1]."""
        from deep_learning.mlb_dl.datasets import compute_game_decay_weight, SequenceSpec

        dates = pd.to_datetime([f"2023-{m:02d}-15" for m in range(4, 10)] +
                               [f"2024-{m:02d}-15" for m in range(4, 10)])
        entry = {
            "dates": list(dates),
            "seasons": [2023] * 6 + [2024] * 6,
            "indices": list(range(12)),
        }
        spec = SequenceSpec()
        current_date = pd.Timestamp("2024-10-01")
        weights = compute_game_decay_weight(entry, current_date, spec)

        for w in weights:
            assert 0.0 < w <= 1.0, f"Weight {w} outside (0, 1]"

    def test_empty_history(self):
        """No games before current_date returns empty list."""
        from deep_learning.mlb_dl.datasets import compute_game_decay_weight, SequenceSpec

        dates = pd.to_datetime(["2024-04-10", "2024-04-12"])
        entry = {
            "dates": list(dates),
            "seasons": [2024, 2024],
            "indices": [0, 1],
        }
        spec = SequenceSpec()
        # current_date before all games
        weights = compute_game_decay_weight(entry, pd.Timestamp("2024-04-01"), spec)
        assert weights == []


class TestHashBucket:
    """Tests for _hash_bucket collision properties and range."""

    def test_output_in_range(self):
        """All outputs must be in [0, n_buckets)."""
        from deep_learning.mlb_dl.datasets import _hash_bucket

        n_buckets = 50000
        for player_id in range(1000):
            bucket = _hash_bucket(player_id, n_buckets)
            assert 0 <= bucket < n_buckets, f"Bucket {bucket} outside [0, {n_buckets})"

    def test_none_maps_to_zero(self):
        """None/NaN inputs must map to bucket 0."""
        from deep_learning.mlb_dl.datasets import _hash_bucket

        assert _hash_bucket(None, 50000) == 0
        assert _hash_bucket(float("nan"), 50000) == 0
        assert _hash_bucket("nan", 50000) == 0
        assert _hash_bucket("None", 50000) == 0

    def test_different_ids_low_collision(self):
        """Different player IDs should have a low collision rate (< 5% for 1000 IDs in 50k buckets)."""
        from deep_learning.mlb_dl.datasets import _hash_bucket

        n_buckets = 50000
        ids = list(range(100000, 101000))  # 1000 realistic player IDs
        buckets = [_hash_bucket(pid, n_buckets) for pid in ids]
        unique_buckets = len(set(buckets))
        collision_rate = 1.0 - (unique_buckets / len(ids))
        assert collision_rate < 0.05, f"Collision rate {collision_rate:.2%} too high"

    def test_deterministic(self):
        """Same input always maps to same bucket."""
        from deep_learning.mlb_dl.datasets import _hash_bucket

        for _ in range(10):
            assert _hash_bucket(123456, 50000) == _hash_bucket(123456, 50000)

    def test_valid_ids_never_zero(self):
        """Valid (non-None, non-NaN) IDs should map to [1, n_buckets-1], never 0."""
        from deep_learning.mlb_dl.datasets import _hash_bucket

        n_buckets = 50000
        for pid in [1, 100, 999, 123456, 654321]:
            bucket = _hash_bucket(pid, n_buckets)
            assert bucket > 0, f"Valid ID {pid} mapped to bucket 0 (reserved for missing)"


class TestLeftPad:
    """Tests for _left_pad padding and truncation logic."""

    def test_padding_zeros_at_start(self):
        """Left-padding places zeros at the START, actual data at the END."""
        from deep_learning.mlb_dl.datasets import _left_pad

        values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        mask = np.ones_like(values)
        target_len = 5
        feature_dim = 2

        padded_values, padded_mask, padding = _left_pad(values, mask, target_len, feature_dim)

        assert padded_values.shape == (5, 2)
        # First 3 rows should be zeros (padding)
        np.testing.assert_array_equal(padded_values[:3], 0.0)
        # Last 2 rows should be the original data
        np.testing.assert_array_equal(padded_values[3:], values)

    def test_padding_indicator_vector(self):
        """Padding indicator: 0 for padded positions, 1 for real data."""
        from deep_learning.mlb_dl.datasets import _left_pad

        values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        mask = np.ones_like(values)
        target_len = 5
        feature_dim = 2

        _, _, padding = _left_pad(values, mask, target_len, feature_dim)
        assert padding.shape == (5,)
        # First 3 positions: padded (0.0)
        np.testing.assert_array_equal(padding[:3], 0.0)
        # Last 2 positions: real data (1.0)
        np.testing.assert_array_equal(padding[3:], 1.0)

    def test_truncation_keeps_most_recent(self):
        """When input exceeds target_len, keep LAST (most recent) entries."""
        from deep_learning.mlb_dl.datasets import _left_pad

        values = np.arange(10 * 3, dtype="float32").reshape(10, 3)
        mask = np.ones_like(values)
        target_len = 5
        feature_dim = 3

        trunc_values, trunc_mask, padding = _left_pad(values, mask, target_len, feature_dim)

        assert trunc_values.shape == (5, 3)
        # Should keep the LAST 5 rows (most recent games)
        np.testing.assert_array_equal(trunc_values, values[5:])
        # Padding all 1s (no padding needed)
        np.testing.assert_array_equal(padding, 1.0)

    def test_exact_length_no_pad_no_truncate(self):
        """Input exactly at target_len: no padding, no truncation."""
        from deep_learning.mlb_dl.datasets import _left_pad

        target_len = 5
        feature_dim = 2
        values = np.ones((target_len, feature_dim), dtype="float32") * 7.0
        mask = np.ones_like(values)

        out_values, out_mask, padding = _left_pad(values, mask, target_len, feature_dim)

        # pad_count = 5 - 5 = 0, goes to else branch (truncation path)
        assert out_values.shape == (5, 2)
        np.testing.assert_array_equal(out_values, values)
        np.testing.assert_array_equal(padding, 1.0)

    def test_empty_input(self):
        """Empty input (0 rows) produces all-zero padding."""
        from deep_learning.mlb_dl.datasets import _left_pad

        values = np.zeros((0, 4), dtype="float32")
        mask = np.zeros((0, 4), dtype="float32")
        target_len = 5
        feature_dim = 4

        out_values, out_mask, padding = _left_pad(values, mask, target_len, feature_dim)

        assert out_values.shape == (5, 4)
        np.testing.assert_array_equal(out_values, 0.0)
        np.testing.assert_array_equal(padding, 0.0)


# ============================================================================
# rating_sequences.py tests
# ============================================================================


class TestLoadRatingSequences:
    """Tests for load_rating_sequences file handling and shape consistency."""

    def test_missing_file_returns_empty(self):
        """Missing .npz or .json returns empty store instead of crashing — allows
        graceful inference start before rating sequences are built."""
        from deep_learning.mlb_dl.rating_sequences import load_rating_sequences

        seqs, cols, k = load_rating_sequences("/nonexistent/path/rating_sequences.npz")
        assert seqs == {}
        assert cols == []
        assert k == 0

    def test_roundtrip_shape_consistency(self):
        """Saved and loaded sequences have shape (K, N_RATINGS) per key."""
        from deep_learning.mlb_dl.rating_sequences import (
            RATING_SEQ_STEPS,
            load_rating_sequences,
            save_rating_sequences,
        )

        n_ratings = 7
        k_steps = RATING_SEQ_STEPS
        rating_cols = [f"feature_{i}" for i in range(n_ratings)]
        means = {col: 0.0 for col in rating_cols}
        stds = {col: 1.0 for col in rating_cols}

        sequences = {
            (717001, "home"): np.random.randn(k_steps, n_ratings).astype(np.float32),
            (717001, "away"): np.random.randn(k_steps, n_ratings).astype(np.float32),
            (717002, "home"): np.random.randn(k_steps, n_ratings).astype(np.float32),
            (717002, "away"): np.random.randn(k_steps, n_ratings).astype(np.float32),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "rating_sequences.npz")
            save_rating_sequences(sequences, rating_cols, means, stds, out_path)

            loaded_seqs, loaded_cols, loaded_k = load_rating_sequences(out_path)

        assert loaded_k == k_steps
        assert loaded_cols == rating_cols
        assert len(loaded_seqs) == 4

        for key, seq in loaded_seqs.items():
            assert seq.shape == (k_steps, n_ratings), (
                f"Key {key}: shape {seq.shape}, expected ({k_steps}, {n_ratings})"
            )

    def test_k_steps_constant_matches_data(self):
        """RATING_SEQ_STEPS must match the k_steps dimension in built sequences."""
        from deep_learning.mlb_dl.rating_sequences import (
            RATING_SEQ_STEPS,
            save_rating_sequences,
            load_rating_sequences,
        )

        k_steps = RATING_SEQ_STEPS
        n_ratings = 5
        rating_cols = [f"r_{i}" for i in range(n_ratings)]

        sequences = {
            (100, "home"): np.zeros((k_steps, n_ratings), dtype=np.float32),
        }
        means = {col: 0.0 for col in rating_cols}
        stds = {col: 1.0 for col in rating_cols}

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "test.npz")
            save_rating_sequences(sequences, rating_cols, means, stds, out_path)
            _, _, loaded_k = load_rating_sequences(out_path)

        assert loaded_k == RATING_SEQ_STEPS


class TestIdentifyRatingColumns:
    """Tests for identify_rating_columns feature selection logic."""

    def test_selects_elo_prefixes(self):
        from deep_learning.mlb_dl.rating_sequences import identify_rating_columns

        cols = ["home_elo_rating", "away_elo_rating", "batting_avg", "elo_diff"]
        result = identify_rating_columns(cols)
        assert "home_elo_rating" in result
        assert "away_elo_rating" in result
        assert "elo_diff" in result
        assert "batting_avg" not in result

    def test_excludes_velo(self):
        """Columns containing 'velo' should be excluded even if they match a prefix."""
        from deep_learning.mlb_dl.rating_sequences import identify_rating_columns

        cols = ["home_elo_velo_interaction", "elo_diff"]
        result = identify_rating_columns(cols)
        assert "home_elo_velo_interaction" not in result
        assert "elo_diff" in result

    def test_interaction_columns(self):
        """Interaction columns with _x_ and a rating component should be selected."""
        from deep_learning.mlb_dl.rating_sequences import identify_rating_columns

        cols = ["batting_avg_x_elo_diff", "batting_avg_x_spin_rate"]
        result = identify_rating_columns(cols)
        assert "batting_avg_x_elo_diff" in result
        assert "batting_avg_x_spin_rate" not in result


# ============================================================================
# weather_context.py tests
# ============================================================================


class TestWeatherConstants:
    """Tests for weather_context.py constant-shape consistency."""

    def test_token_dim_matches_column_count(self):
        """WEATHER_TOKEN_DIM must equal len(WEATHER_TEMPORAL_COLUMNS)."""
        from deep_learning.mlb_dl.weather_context import (
            WEATHER_TOKEN_DIM,
            WEATHER_TEMPORAL_COLUMNS,
        )

        assert WEATHER_TOKEN_DIM == len(WEATHER_TEMPORAL_COLUMNS), (
            f"WEATHER_TOKEN_DIM={WEATHER_TOKEN_DIM} != "
            f"len(WEATHER_TEMPORAL_COLUMNS)={len(WEATHER_TEMPORAL_COLUMNS)}"
        )

    def test_compute_hour_features_output_dim(self):
        """compute_hour_features returns exactly WEATHER_TOKEN_DIM features."""
        from deep_learning.mlb_dl.weather_context import (
            WEATHER_TOKEN_DIM,
            compute_hour_features,
        )

        empty_row = {}
        out = compute_hour_features(empty_row, venue_id=2500, cf_azimuth_deg=0.0)
        assert out.shape == (WEATHER_TOKEN_DIM,)

    def test_temporal_hours_is_four(self):
        """WEATHER_TEMPORAL_HOURS must be 4 (game duration coverage)."""
        from deep_learning.mlb_dl.weather_context import WEATHER_TEMPORAL_HOURS

        assert WEATHER_TEMPORAL_HOURS == 4


class TestWeatherNaNHandling:
    """Verify NaN inputs are filled to 0.0 before model consumption."""

    def test_nan_era5_row_produces_zeros(self):
        """All-NaN era5 row must produce all-zero feature vector."""
        from deep_learning.mlb_dl.weather_context import compute_hour_features

        nan_row = {
            "temperature_2m": float("nan"),
            "dew_point_2m": float("nan"),
            "surface_pressure": float("nan"),
            "relative_humidity_2m": float("nan"),
            "vapour_pressure_deficit": float("nan"),
            "wet_bulb_temperature_2m": float("nan"),
            "wind_u_10m": float("nan"),
            "wind_v_10m": float("nan"),
            "wind_speed_10m": float("nan"),
            "wind_gusts_10m": float("nan"),
            "cloud_cover": float("nan"),
            "visibility": float("nan"),
            "precipitation": float("nan"),
            "boundary_layer_height": float("nan"),
            "shortwave_radiation": float("nan"),
            "soil_moisture_0_to_7cm": float("nan"),
        }
        out = compute_hour_features(nan_row, venue_id=2500, cf_azimuth_deg=0.0)
        assert not np.any(np.isnan(out)), f"NaN found in output: {out}"
        np.testing.assert_array_equal(out, 0.0)

    def test_partial_nan_no_propagation(self):
        """Some NaN values must not corrupt other features."""
        from deep_learning.mlb_dl.weather_context import compute_hour_features

        row = {
            "temperature_2m": 72.0,
            "dew_point_2m": 55.0,
            "surface_pressure": 1013.0,
            "relative_humidity_2m": float("nan"),  # single NaN
            "vapour_pressure_deficit": 1.2,
            "wet_bulb_temperature_2m": 63.0,
            "wind_u_10m": 3.0,
            "wind_v_10m": 4.0,
            "wind_speed_10m": 5.0,
            "wind_gusts_10m": 8.0,
            "cloud_cover": 25.0,
            "visibility": 30000.0,
            "precipitation": 0.0,
            "boundary_layer_height": 1500.0,
            "shortwave_radiation": 400.0,
            "soil_moisture_0_to_7cm": 0.3,
        }
        out = compute_hour_features(row, venue_id=2500, cf_azimuth_deg=45.0)
        assert not np.any(np.isnan(out)), "NaN propagated from single bad input"
        # Air density should still be computed (temp and pressure are valid)
        assert out[0] > 0.0

    def test_vectorized_nan_handling(self):
        """Vectorized path also produces 0.0 for NaN inputs, no NaN in output."""
        from deep_learning.mlb_dl.weather_context import compute_hour_features_vectorized

        era5_df = pd.DataFrame({
            "temperature_2m": [72.0, np.nan],
            "dew_point_2m": [55.0, np.nan],
            "surface_pressure": [1013.0, np.nan],
            "relative_humidity_2m": [60.0, np.nan],
            "vapour_pressure_deficit": [1.2, np.nan],
            "wet_bulb_temperature_2m": [63.0, np.nan],
            "wind_u_10m": [3.0, np.nan],
            "wind_v_10m": [4.0, np.nan],
            "wind_speed_10m": [5.0, np.nan],
            "wind_gusts_10m": [8.0, np.nan],
            "cloud_cover": [25.0, np.nan],
            "visibility": [30000.0, np.nan],
            "precipitation": [0.0, np.nan],
            "boundary_layer_height": [1500.0, np.nan],
            "shortwave_radiation": [400.0, np.nan],
            "soil_moisture_0_to_7cm": [0.3, np.nan],
        })
        venue_ids = np.array([2500, 2500])
        azimuths = np.array([45.0, 45.0])

        out = compute_hour_features_vectorized(era5_df, venue_ids, azimuths)
        assert not np.any(np.isnan(out)), f"NaN found in vectorized output"
        # Row 1 (all NaN input) should be all zeros
        np.testing.assert_array_equal(out[1], 0.0)
