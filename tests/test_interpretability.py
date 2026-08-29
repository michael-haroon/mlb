"""Tests for interpretability methods: H-statistic, ALE, TreeSHAP.

Three categories:
  1. Mathematical identity tests — verify the methods satisfy known theoretical
     properties on analytically tractable functions.
  2. Empirical recovery tests — verify methods recover ground-truth signal on
     synthetic data with known interactions and response shapes.
  3. Code correctness tests — verify API contracts, edge cases, and consistency.

Run: conda run -n pred python -m pytest tests/test_interpretability.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def additive_regression_data():
    """Dataset where y = x0 + x1 + noise (NO interaction).

    H-stat should be ~0 for (x0, x1) since effects are purely additive.
    ALE for x0 should be approximately linear with slope 1.
    """
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.standard_normal((n, 4))
    y = X[:, 0] + X[:, 1] + rng.normal(0, 0.1, n)
    years = np.repeat(np.arange(2015, 2025), n // 10)[:n]
    return pd.DataFrame(X, columns=["x0", "x1", "x2", "x3"]), pd.Series(y), pd.Series(years)


@pytest.fixture
def interaction_regression_data():
    """Dataset where y = x0 * x1 + noise (STRONG interaction).

    H-stat for (x0, x1) should be substantially > 0.
    H-stat for (x0, x2) should be ~0 (x2 is noise).
    """
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.standard_normal((n, 4))
    y = X[:, 0] * X[:, 1] + rng.normal(0, 0.1, n)
    years = np.repeat(np.arange(2015, 2025), n // 10)[:n]
    return pd.DataFrame(X, columns=["x0", "x1", "x2", "x3"]), pd.Series(y), pd.Series(years)


@pytest.fixture
def threshold_data():
    """Dataset with threshold effect: y = I(x0 > 0) + noise.

    ALE should show a step function around x0=0.
    """
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] > 0).astype(float) + rng.normal(0, 0.05, n)
    years = np.repeat(np.arange(2015, 2025), n // 10)[:n]
    return pd.DataFrame(X, columns=["x0", "x1", "x2"]), pd.Series(y), pd.Series(years)


@pytest.fixture
def saturating_data():
    """Dataset with saturating effect: y = tanh(x0) + noise.

    ALE should show diminishing returns at extremes (non-monotone derivative).
    """
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.standard_normal((n, 3))
    y = np.tanh(X[:, 0]) + rng.normal(0, 0.05, n)
    years = np.repeat(np.arange(2015, 2025), n // 10)[:n]
    return pd.DataFrame(X, columns=["x0", "x1", "x2"]), pd.Series(y), pd.Series(years)


@pytest.fixture
def classification_data():
    """Binary classification with known feature importance ordering.

    Interaction coefficient (2.0) is deliberately large relative to main effects
    so that H-stat can detect it above tree-approximation noise.
    """
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.standard_normal((n, 5))
    logit = 1.0 * X[:, 0] + 0.5 * X[:, 1] + 2.0 * X[:, 0] * X[:, 1]
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(int)
    years = np.repeat(np.arange(2015, 2025), n // 10)[:n]
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(5)]), pd.Series(y), pd.Series(years)


def _fit_regressor(X, y, n_estimators=200):
    """Fit a BaggingRegressor on the full dataset (for unit tests, not CV)."""
    base = DecisionTreeRegressor(max_features=1, min_weight_fraction_leaf=0.02)
    clf = BaggingRegressor(
        estimator=base, n_estimators=n_estimators,
        max_features=1.0, max_samples=1.0,
        n_jobs=-1, random_state=42,
    )
    clf.fit(X, y)
    return clf


def _fit_classifier(X, y, n_estimators=200):
    """Fit a BaggingClassifier on the full dataset (for unit tests, not CV)."""
    base = DecisionTreeClassifier(
        criterion="entropy", max_features=1,
        class_weight="balanced", min_weight_fraction_leaf=0.02)
    clf = BaggingClassifier(
        estimator=base, n_estimators=n_estimators,
        max_features=1.0, max_samples=1.0,
        n_jobs=-1, random_state=42,
    )
    clf.fit(X, y)
    return clf


# ─────────────────────────────────────────────────────────────────────────────
#  1. MATHEMATICAL IDENTITY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestHStatMathematical:
    """H-stat mathematical properties derived from its definition."""

    def test_h_bounded_01(self, additive_regression_data):
        """H² ∈ [0, 1] by construction (ratio of variances)."""
        from classical_learning.analysis.interpretability import h_statistic_pair

        X, y, _ = additive_regression_data
        model = _fit_regressor(X.values, y.values)
        h_sq = h_statistic_pair(model, X.values, 0, 1, n_grid=15,
                                is_classifier=False, subsample=500)
        assert 0.0 <= h_sq <= 1.0

    def test_additive_function_h_near_zero(self, additive_regression_data):
        """For f(x) = g(x0) + h(x1), H²(x0, x1) = 0 exactly.

        With finite trees and noise, we allow tolerance.
        Mathematical proof: if PD_ij = PD_i + PD_j, the numerator is identically 0.
        """
        from classical_learning.analysis.interpretability import h_statistic_pair

        X, y, _ = additive_regression_data
        model = _fit_regressor(X.values, y.values, n_estimators=500)
        h_sq = h_statistic_pair(model, X.values, 0, 1, n_grid=20,
                                is_classifier=False, subsample=800)
        # For a well-fit additive function, H should be very small
        assert h_sq < 0.05, f"H²={h_sq:.4f} too high for additive function"

    def test_multiplicative_function_h_large(self, interaction_regression_data):
        """For f(x) = x0 * x1, H²(x0, x1) should be substantial.

        Mathematical: PD_ij ≠ PD_i + PD_j when the true function is multiplicative.
        PD_i(x0) = x0 * E[x1] = 0 (since E[x1]=0 for standard normal).
        PD_ij(x0, x1) = x0 * x1.
        Residual = x0*x1 - 0 - 0 = x0*x1 → H² ≈ 1.
        """
        from classical_learning.analysis.interpretability import h_statistic_pair

        X, y, _ = interaction_regression_data
        model = _fit_regressor(X.values, y.values, n_estimators=500)
        h_sq = h_statistic_pair(model, X.values, 0, 1, n_grid=20,
                                is_classifier=False, subsample=800)
        assert h_sq > 0.3, f"H²={h_sq:.4f} too low for x0*x1 interaction"

    def test_noise_features_h_near_zero(self, interaction_regression_data):
        """H²(x0, x2) ≈ 0 when x2 is pure noise (no interaction with x0)."""
        from classical_learning.analysis.interpretability import h_statistic_pair

        X, y, _ = interaction_regression_data
        model = _fit_regressor(X.values, y.values, n_estimators=500)
        h_sq = h_statistic_pair(model, X.values, 0, 2, n_grid=20,
                                is_classifier=False, subsample=800)
        assert h_sq < 0.15, f"H²={h_sq:.4f} too high for noise pair"

    def test_h_symmetric(self, interaction_regression_data):
        """H²(i,j) = H²(j,i) — the statistic is symmetric by definition."""
        from classical_learning.analysis.interpretability import h_statistic_pair

        X, y, _ = interaction_regression_data
        model = _fit_regressor(X.values, y.values)
        h_ij = h_statistic_pair(model, X.values, 0, 1, n_grid=15,
                                is_classifier=False, subsample=500)
        h_ji = h_statistic_pair(model, X.values, 1, 0, n_grid=15,
                                is_classifier=False, subsample=500)
        assert h_ij == pytest.approx(h_ji, abs=0.02)


class TestALEMathematical:
    """ALE mathematical properties from its definition."""

    def test_ale_is_centered(self, additive_regression_data):
        """ALE is centered: weighted mean = 0 by construction.

        This follows from the centering step in the algorithm.
        """
        from classical_learning.analysis.interpretability import ale_1d

        X, y, _ = additive_regression_data
        model = _fit_regressor(X.values, y.values)
        centers, values = ale_1d(model, X.values, 0, n_bins=30, is_classifier=False)

        # Weighted mean should be ~0 (exact 0 if we weight by bin counts)
        assert abs(values.mean()) < 0.05, f"ALE not centered: mean={values.mean():.4f}"

    def test_ale_linear_function_is_linear(self, additive_regression_data):
        """For y = x0 + ..., ALE(x0) should be approximately linear.

        Mathematical: local effect in each bin ≈ Δx (constant), so accumulated
        values form a line. Deviation comes from tree approximation error.
        """
        from classical_learning.analysis.interpretability import ale_1d

        X, y, _ = additive_regression_data
        model = _fit_regressor(X.values, y.values, n_estimators=500)
        centers, values = ale_1d(model, X.values, 0, n_bins=30, is_classifier=False)

        # Fit linear regression to ALE curve
        if len(centers) > 2:
            slope, intercept = np.polyfit(centers, values, 1)
            residuals = values - (slope * centers + intercept)
            r_squared = 1 - np.var(residuals) / np.var(values)
            assert r_squared > 0.9, f"ALE not linear: R²={r_squared:.3f}"

    def test_ale_noise_feature_flat(self, additive_regression_data):
        """ALE for noise feature (x2, x3) should be approximately flat (range ≈ 0)."""
        from classical_learning.analysis.interpretability import ale_1d

        X, y, _ = additive_regression_data
        model = _fit_regressor(X.values, y.values, n_estimators=500)
        _, values = ale_1d(model, X.values, 2, n_bins=30, is_classifier=False)

        ale_range = values.max() - values.min()
        # Noise feature should have negligible effect
        assert ale_range < 0.3, f"Noise feature ALE range={ale_range:.4f} too large"

    def test_ale_monotone_for_monotone_function(self, additive_regression_data):
        """For y = x0 + ... (monotone in x0), ALE(x0) should be monotone."""
        from classical_learning.analysis.interpretability import ale_1d

        X, y, _ = additive_regression_data
        model = _fit_regressor(X.values, y.values, n_estimators=500)
        centers, values = ale_1d(model, X.values, 0, n_bins=20, is_classifier=False)

        # Allow small non-monotonicities from tree approximation
        diffs = np.diff(values)
        n_decreasing = (diffs < -0.01).sum()
        # At most 10% of steps should be decreasing for a monotone function
        assert n_decreasing <= len(diffs) * 0.1, \
            f"Too many decreasing steps ({n_decreasing}/{len(diffs)})"


class TestSHAPMathematical:
    """SHAP mathematical properties from Shapley axiom guarantees."""

    def test_shap_efficiency(self, additive_regression_data):
        """Efficiency axiom: sum of SHAP values = f(x) - E[f(x)].

        For every prediction, the SHAP values must sum to the difference
        between that prediction and the base value (expected prediction).
        """
        from classical_learning.analysis.interpretability import _tree_shap_values

        X, y, _ = additive_regression_data
        model = _fit_regressor(X.values, y.values)
        shap_vals = _tree_shap_values(model, X.values[:100], is_classifier=False)

        predictions = model.predict(X.values[:100])
        base_value = model.predict(X.values).mean()  # approximate expected value

        # sum(SHAP) ≈ prediction - base_value (within tree precision)
        shap_sums = shap_vals.sum(axis=1)
        expected_diffs = predictions - base_value
        np.testing.assert_allclose(shap_sums, expected_diffs, atol=0.1)

    def test_shap_null_player(self, additive_regression_data):
        """Null player axiom: noise features get ~0 SHAP importance.

        If feature j never changes the prediction, φ_j = 0.
        """
        from classical_learning.analysis.interpretability import _tree_shap_values

        X, y, _ = additive_regression_data
        model = _fit_regressor(X.values, y.values, n_estimators=500)
        shap_vals = _tree_shap_values(model, X.values[:200], is_classifier=False)

        # x2, x3 are noise — their mean |SHAP| should be much smaller than x0, x1
        signal_imp = np.abs(shap_vals[:, :2]).mean()
        noise_imp = np.abs(shap_vals[:, 2:]).mean()
        assert noise_imp < signal_imp * 0.15, \
            f"Noise SHAP ({noise_imp:.4f}) too large vs signal ({signal_imp:.4f})"

    def test_shap_symmetry(self, additive_regression_data):
        """Symmetry axiom: features with equal contribution get equal SHAP.

        Since y = x0 + x1 + noise, x0 and x1 have identical roles.
        """
        from classical_learning.analysis.interpretability import _tree_shap_values

        X, y, _ = additive_regression_data
        model = _fit_regressor(X.values, y.values, n_estimators=500)
        shap_vals = _tree_shap_values(model, X.values[:500], is_classifier=False)

        imp_x0 = np.abs(shap_vals[:, 0]).mean()
        imp_x1 = np.abs(shap_vals[:, 1]).mean()
        # Should be within 20% of each other (both contribute equally)
        ratio = imp_x0 / (imp_x1 + 1e-10)
        assert 0.7 < ratio < 1.3, f"Symmetry violation: ratio={ratio:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
#  2. EMPIRICAL RECOVERY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestHStatRecovery:
    """H-stat correctly identifies interaction pairs on synthetic data."""

    def test_ranks_true_interaction_above_noise(self, interaction_regression_data):
        """H²(x0, x1) > H²(x0, x2) > ~0 when y = x0*x1."""
        from classical_learning.analysis.interpretability import h_statistic_pair

        X, y, _ = interaction_regression_data
        model = _fit_regressor(X.values, y.values, n_estimators=300)

        h_01 = h_statistic_pair(model, X.values, 0, 1, n_grid=15,
                                is_classifier=False, subsample=500)
        h_02 = h_statistic_pair(model, X.values, 0, 2, n_grid=15,
                                is_classifier=False, subsample=500)
        h_23 = h_statistic_pair(model, X.values, 2, 3, n_grid=15,
                                is_classifier=False, subsample=500)

        assert h_01 > h_02, f"True interaction not ranked first: H01={h_01}, H02={h_02}"
        assert h_01 > h_23, f"True interaction not ranked first: H01={h_01}, H23={h_23}"

    def test_classification_interaction(self, classification_data):
        """Recovers x0*x1 interaction in binary classification (logistic DGP)."""
        from classical_learning.analysis.interpretability import h_statistic_pair

        X, y, _ = classification_data
        model = _fit_classifier(X.values, y.values, n_estimators=300)

        h_01 = h_statistic_pair(model, X.values, 0, 1, n_grid=15,
                                is_classifier=True, subsample=500)
        h_34 = h_statistic_pair(model, X.values, 3, 4, n_grid=15,
                                is_classifier=True, subsample=500)

        assert h_01 > h_34, f"Interaction not recovered: H01={h_01:.4f}, H34={h_34:.4f}"

    def test_top_pairs_api(self, interaction_regression_data):
        """h_statistic_top_pairs returns correct format and ranking."""
        from classical_learning.analysis.interpretability import h_statistic_top_pairs

        X, y, years = interaction_regression_data
        result = h_statistic_top_pairs(
            X, y, years, top_n_features=4, n_grid=10, subsample=300,
            n_estimators=200, regression=True)

        assert isinstance(result, pd.DataFrame)
        assert "feature_i" in result.columns
        assert "feature_j" in result.columns
        assert "h_squared" in result.columns
        assert len(result) == 6  # C(4,2) = 6 pairs

        # Top pair should involve x0 and x1
        top = result.iloc[0]
        top_pair = {top["feature_i"], top["feature_j"]}
        assert top_pair == {"x0", "x1"}, f"Top pair is {top_pair}, expected {{x0, x1}}"


class TestALERecovery:
    """ALE recovers known response shapes."""

    def test_threshold_shows_step(self, threshold_data):
        """ALE for y = I(x0 > 0) should show a step around x0=0."""
        from classical_learning.analysis.interpretability import ale_1d

        X, y, _ = threshold_data
        model = _fit_regressor(X.values, y.values, n_estimators=500)
        centers, values = ale_1d(model, X.values, 0, n_bins=40, is_classifier=False)

        # ALE should have most of its range concentrated around x0=0
        left_of_zero = values[centers < -0.5].mean() if (centers < -0.5).any() else 0
        right_of_zero = values[centers > 0.5].mean() if (centers > 0.5).any() else 0
        step_size = right_of_zero - left_of_zero
        assert step_size > 0.5, f"Step too small: {step_size:.3f}"

    def test_saturation_shows_diminishing_returns(self, saturating_data):
        """ALE for y = tanh(x0) should show flattening at extremes."""
        from classical_learning.analysis.interpretability import ale_1d

        X, y, _ = saturating_data
        model = _fit_regressor(X.values, y.values, n_estimators=500)
        centers, values = ale_1d(model, X.values, 0, n_bins=30, is_classifier=False)

        # The derivative (local slope) should decrease at extremes
        if len(centers) > 10:
            diffs = np.diff(values) / np.diff(centers)
            # Central slope should be larger than extreme slopes
            mid = len(diffs) // 2
            quarter = len(diffs) // 4
            central_slope = np.abs(diffs[mid - quarter:mid + quarter]).mean()
            extreme_slope = (np.abs(diffs[:quarter]).mean() +
                           np.abs(diffs[-quarter:]).mean()) / 2
            assert central_slope > extreme_slope * 0.8, \
                f"No saturation detected: central={central_slope:.3f}, extreme={extreme_slope:.3f}"

    def test_ale_all_features_api(self, additive_regression_data):
        """ale_all_features returns correct format."""
        from classical_learning.analysis.interpretability import ale_all_features

        X, y, years = additive_regression_data
        result = ale_all_features(X, y, years, n_bins=20, n_estimators=100,
                                  regression=True)

        assert isinstance(result, dict)
        assert set(result.keys()) == set(X.columns)
        for fname, data in result.items():
            assert "centers" in data
            assert "ale" in data
            assert "range" in data
            assert "monotone" in data
            assert len(data["centers"]) == len(data["ale"])

    def test_ale_range_ordering(self, additive_regression_data):
        """Signal features (x0, x1) should have larger ALE range than noise (x2, x3)."""
        from classical_learning.analysis.interpretability import ale_all_features

        X, y, years = additive_regression_data
        result = ale_all_features(X, y, years, n_bins=20, n_estimators=300,
                                  regression=True)

        signal_range = max(result["x0"]["range"], result["x1"]["range"])
        noise_range = max(result["x2"]["range"], result["x3"]["range"])
        assert signal_range > noise_range * 2, \
            f"Signal range ({signal_range:.4f}) not >> noise range ({noise_range:.4f})"


class TestSHAPRecovery:
    """SHAP recovers known importance ordering and interactions."""

    def test_importance_ordering(self, interaction_regression_data):
        """For y = x0*x1, SHAP should rank x0 and x1 above x2, x3."""
        from classical_learning.analysis.interpretability import _tree_shap_values

        X, y, _ = interaction_regression_data
        model = _fit_regressor(X.values, y.values, n_estimators=300)
        shap_vals = _tree_shap_values(model, X.values[:500], is_classifier=False)

        mean_abs = np.abs(shap_vals).mean(axis=0)
        # x0 and x1 should be top 2
        top_2 = np.argsort(mean_abs)[-2:]
        assert set(top_2) == {0, 1}, f"Top 2 features are {top_2}, expected {{0, 1}}"

    def test_shap_importance_api(self, interaction_regression_data):
        """shap_importance returns correct format."""
        from classical_learning.analysis.interpretability import shap_importance

        X, y, years = interaction_regression_data
        result = shap_importance(X, y, years, n_estimators=100, regression=True)

        assert "global_importance" in result
        assert "shap_values" in result
        assert isinstance(result["global_importance"], pd.DataFrame)
        assert "mean_abs_shap" in result["global_importance"].columns
        assert result["shap_values"].shape[1] == X.shape[1]

    def test_shap_interactions_detected(self, interaction_regression_data):
        """SHAP interaction values detect x0*x1 interaction."""
        from classical_learning.analysis.interpretability import shap_importance

        X, y, years = interaction_regression_data
        result = shap_importance(
            X, y, years, n_estimators=200, regression=True,
            compute_interactions=True, interaction_top_n=4,
            subsample_interactions=200)

        assert "interactions" in result
        df = result["interactions"]
        top = df.iloc[0]
        top_pair = {top["feature_i"], top["feature_j"]}
        assert top_pair == {"x0", "x1"}, f"Top interaction pair: {top_pair}"

    def test_classification_shap(self, classification_data):
        """SHAP works for classification targets."""
        from classical_learning.analysis.interpretability import shap_importance

        X, y, years = classification_data
        result = shap_importance(X, y, years, n_estimators=100, regression=False)

        imp = result["global_importance"]
        # x0 should be top (coefficient=2 in logit)
        assert imp.index[0] == "x0", f"Top feature is {imp.index[0]}, expected x0"


# ─────────────────────────────────────────────────────────────────────────────
#  3. CODE CORRECTNESS / EDGE CASE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases and API contracts."""

    def test_h_stat_constant_feature(self):
        """H-stat handles constant features gracefully (all same value)."""
        from classical_learning.analysis.interpretability import h_statistic_pair

        rng = np.random.default_rng(42)
        n = 500
        X = np.column_stack([rng.standard_normal(n), np.ones(n), rng.standard_normal(n)])
        y = X[:, 0] + rng.normal(0, 0.1, n)
        model = _fit_regressor(X, y)

        # Feature 1 is constant — H-stat should be 0 or handle gracefully
        h_sq = h_statistic_pair(model, X, 0, 1, n_grid=10, is_classifier=False)
        assert np.isfinite(h_sq)
        assert h_sq >= 0

    def test_ale_few_unique_values(self):
        """ALE handles features with very few unique values."""
        from classical_learning.analysis.interpretability import ale_1d

        rng = np.random.default_rng(42)
        n = 500
        # Binary feature
        X = np.column_stack([
            rng.choice([0, 1], n),
            rng.standard_normal(n),
        ])
        y = X[:, 0] + rng.normal(0, 0.1, n)
        model = _fit_regressor(X, y)

        centers, values = ale_1d(model, X, 0, n_bins=40, is_classifier=False)
        # Should not crash and return something sensible
        assert len(centers) > 0
        assert len(centers) == len(values)

    def test_ale_all_nan_column_handled(self):
        """ALE gracefully handles a feature that was all-NaN (filled to constant)."""
        from classical_learning.analysis.interpretability import ale_1d

        rng = np.random.default_rng(42)
        n = 500
        X = np.column_stack([
            rng.standard_normal(n),
            np.zeros(n),  # simulates all-NaN filled to median=0
        ])
        y = X[:, 0] + rng.normal(0, 0.1, n)
        model = _fit_regressor(X, y)

        centers, values = ale_1d(model, X, 1, n_bins=40, is_classifier=False)
        assert len(centers) >= 1

    def test_shap_single_tree(self):
        """SHAP works with a single-tree model."""
        from classical_learning.analysis.interpretability import _tree_shap_values

        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 3))
        y = X[:, 0] + rng.normal(0, 0.1, n)

        base = DecisionTreeRegressor(max_features=1, min_weight_fraction_leaf=0.02)
        model = BaggingRegressor(estimator=base, n_estimators=1, n_jobs=1, random_state=42)
        model.fit(X, y)

        shap_vals = _tree_shap_values(model, X[:10], is_classifier=False)
        assert shap_vals.shape == (10, 3)

    def test_h_stat_subsample_reproducible(self, additive_regression_data):
        """H-stat with same seed gives same result (deterministic subsampling)."""
        from classical_learning.analysis.interpretability import h_statistic_pair

        X, y, _ = additive_regression_data
        model = _fit_regressor(X.values, y.values)

        h1 = h_statistic_pair(model, X.values, 0, 1, n_grid=10,
                              is_classifier=False, subsample=300)
        h2 = h_statistic_pair(model, X.values, 0, 1, n_grid=10,
                              is_classifier=False, subsample=300)
        assert h1 == h2

    def test_ale_cv_returns_dataframe(self, additive_regression_data):
        """ale_cv returns properly formatted DataFrame."""
        from classical_learning.analysis.interpretability import ale_cv

        X, y, years = additive_regression_data
        result = ale_cv(X, y, years, n_bins=15, n_estimators=50,
                       regression=True, feature_subset=["x0", "x2"])

        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "ale_range_mean" in result.columns
        assert "monotone_frac" in result.columns
        assert len(result) == 2

    def test_shap_cv_returns_dataframe(self, additive_regression_data):
        """shap_cv returns properly formatted DataFrame."""
        from classical_learning.analysis.interpretability import shap_cv

        X, y, years = additive_regression_data
        result = shap_cv(X, y, years, n_estimators=50, regression=True)

        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "shap_mean" in result.columns
        assert "shap_std" in result.columns
        assert len(result) == X.shape[1]


class TestCrossMethodConsistency:
    """Verify that the three methods give consistent signals."""

    def test_h_stat_and_shap_interaction_agree(self, interaction_regression_data):
        """Both H-stat and SHAP interactions should identify (x0, x1) as top pair."""
        from classical_learning.analysis.interpretability import (
            h_statistic_pair, shap_importance)

        X, y, years = interaction_regression_data
        model = _fit_regressor(X.values, y.values, n_estimators=300)

        # H-stat
        h_01 = h_statistic_pair(model, X.values, 0, 1, n_grid=15,
                                is_classifier=False, subsample=500)
        h_02 = h_statistic_pair(model, X.values, 0, 2, n_grid=15,
                                is_classifier=False, subsample=500)

        # SHAP interactions
        result = shap_importance(
            X, y, years, n_estimators=300, regression=True,
            compute_interactions=True, interaction_top_n=4,
            subsample_interactions=200)
        shap_top = result["interactions"].iloc[0]
        shap_pair = {shap_top["feature_i"], shap_top["feature_j"]}

        # Both should agree on (x0, x1) as the strongest interaction
        assert h_01 > h_02, "H-stat should rank (0,1) above (0,2)"
        assert shap_pair == {"x0", "x1"}, f"SHAP top pair: {shap_pair}"

    def test_ale_range_correlates_with_shap(self, additive_regression_data):
        """Features with large ALE range should have large mean |SHAP|."""
        from classical_learning.analysis.interpretability import (
            ale_all_features, _tree_shap_values)

        X, y, years = additive_regression_data
        model = _fit_regressor(X.values, y.values, n_estimators=300)

        # ALE
        ale_result = ale_all_features(X, y, years, n_bins=20,
                                     n_estimators=300, regression=True)
        ale_ranges = [ale_result[c]["range"] for c in X.columns]

        # SHAP
        shap_vals = _tree_shap_values(model, X.values[:500], is_classifier=False)
        shap_imp = np.abs(shap_vals).mean(axis=0)

        # Rank correlation should be positive (both capture feature importance)
        from scipy.stats import spearmanr
        corr, _ = spearmanr(ale_ranges, shap_imp)
        assert corr > 0.5, f"ALE-SHAP rank correlation too low: {corr:.3f}"
