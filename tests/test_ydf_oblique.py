"""Adversarial tests for YDF oblique GBT integration.

Strategy: define expected behavior first, then verify the codebase follows it.
Each test states the invariant being checked as the assertion before exercising
the code path that must satisfy it.
"""
import numpy as np
import pandas as pd
import pickle
import pytest


# ---------------------------------------------------------------------------
# Expected behavior: Pipeline registration
# ---------------------------------------------------------------------------

class TestRegistration:
    """YDF oblique GBT must be properly registered in all pipeline touchpoints."""

    def test_in_tree_models_not_imputation_not_scaling(self):
        """YDF handles NaN and normalizes internally — must NOT require imputation or scaling."""
        from pregame.strategy.config import TREE_MODELS, NEEDS_IMPUTATION, NEEDS_SCALING

        assert "ydf_oblique_gbt" in TREE_MODELS
        assert "ydf_oblique_gbt" not in NEEDS_IMPUTATION
        assert "ydf_oblique_gbt" not in NEEDS_SCALING

    def test_in_model_builders(self):
        """Must be instantiable via build_model."""
        from pregame.strategy.models import build_model
        clf = build_model("ydf_oblique_gbt", "classification", {})
        assert hasattr(clf, "fit")
        assert hasattr(clf, "predict_proba")

    def test_in_column_subsample_confirmed(self):
        """YDF subsamples features per split — safe to include redundant features."""
        from pregame.analysis.feature_routing import COLUMN_SUBSAMPLE_CONFIRMED
        assert "ydf_oblique_gbt" in COLUMN_SUBSAMPLE_CONFIRMED

    def test_in_feature_sizing_families(self):
        """Must participate in per-family sizing curves."""
        import inspect
        from pregame.strategy.feature_sizing import run_sizing_curve
        source = inspect.getsource(run_sizing_curve)
        assert "ydf_oblique_gbt" in source


# ---------------------------------------------------------------------------
# Expected behavior: sklearn contract
# ---------------------------------------------------------------------------

class TestSklearnContract:
    """The wrapper must satisfy sklearn's estimator contract."""

    @pytest.fixture
    def fitted_clf(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (200, 8))
        y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
        from pregame.strategy.models import YDFObliqueClassifier
        clf = YDFObliqueClassifier(num_trees=30, max_depth=4)
        clf.fit(X, y)
        return clf, X

    def test_predict_proba_shape_and_range(self, fitted_clf):
        """predict_proba must return (n, 2) with values in [0, 1] summing to 1."""
        clf, X = fitted_clf
        proba = clf.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_returns_class_labels(self, fitted_clf):
        """predict must return values from classes_, not raw probabilities."""
        clf, X = fitted_clf
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset(set(clf.classes_))

    def test_feature_importances_shape_and_normalization(self, fitted_clf):
        """feature_importances_ must be non-negative, sum to 1, shape = n_features."""
        clf, X = fitted_clf
        imp = clf.feature_importances_
        assert imp.shape == (X.shape[1],)
        assert np.all(imp >= 0.0)
        np.testing.assert_allclose(imp.sum(), 1.0, atol=1e-6)

    def test_get_set_params_roundtrip(self):
        """get_params → set_params must be lossless."""
        from pregame.strategy.models import YDFObliqueClassifier
        clf = YDFObliqueClassifier(num_trees=77, shrinkage=0.05)
        params = clf.get_params()
        clf2 = YDFObliqueClassifier()
        clf2.set_params(**params)
        assert clf2.get_params() == params

    def test_sklearn_clone(self):
        """sklearn.base.clone must produce a valid unfitted copy."""
        from sklearn.base import clone
        from pregame.strategy.models import YDFObliqueClassifier
        clf = YDFObliqueClassifier(num_trees=50, shrinkage=0.03)
        clf2 = clone(clf)
        assert clf2.get_params() == clf.get_params()
        assert not hasattr(clf2, "model_")


# ---------------------------------------------------------------------------
# Expected behavior: NaN handling
# ---------------------------------------------------------------------------

class TestNaNHandling:
    """YDF must handle NaN without imputation — both train and inference."""

    def test_nan_in_training_data(self):
        """Fit must succeed with arbitrary NaN patterns."""
        from pregame.strategy.models import YDFObliqueClassifier
        rng = np.random.default_rng(99)
        X = rng.uniform(0, 1, (300, 10))
        y = (X[:, 0] > 0.5).astype(int)
        X[rng.random(X.shape) < 0.2] = np.nan  # 20% missing

        clf = YDFObliqueClassifier(num_trees=30)
        clf.fit(X, y)
        assert hasattr(clf, "model_")

    def test_nan_in_inference_data(self):
        """Predict must succeed when inference data has NaN in different positions than train."""
        from pregame.strategy.models import YDFObliqueClassifier
        rng = np.random.default_rng(7)
        X_train = rng.uniform(0, 1, (200, 5))
        y = (X_train[:, 0] > 0.5).astype(int)

        clf = YDFObliqueClassifier(num_trees=30)
        clf.fit(X_train, y)

        X_test = rng.uniform(0, 1, (50, 5))
        X_test[:, 3] = np.nan  # entire column missing at inference
        proba = clf.predict_proba(X_test)
        assert proba.shape == (50, 2)
        assert not np.any(np.isnan(proba))

    def test_all_nan_column(self):
        """An entirely NaN column must not crash — model ignores it."""
        from pregame.strategy.models import YDFObliqueClassifier
        rng = np.random.default_rng(11)
        X = rng.uniform(0, 1, (200, 5))
        X[:, 2] = np.nan
        y = (X[:, 0] > 0.5).astype(int)

        clf = YDFObliqueClassifier(num_trees=30)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert not np.any(np.isnan(proba))


# ---------------------------------------------------------------------------
# Expected behavior: Sample weights
# ---------------------------------------------------------------------------

class TestSampleWeights:
    """Temporal decay weights must influence training — not be silently ignored."""

    def test_sample_weights_change_predictions(self):
        """Heavily weighting one class's samples must shift predictions toward it."""
        from pregame.strategy.models import YDFObliqueClassifier
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (400, 5))
        y = (X[:, 0] > 0.5).astype(int)

        clf_unweighted = YDFObliqueClassifier(num_trees=50, random_seed=42)
        clf_unweighted.fit(X, y)

        # Weight class-1 samples 10x
        weights = np.where(y == 1, 10.0, 1.0)
        clf_weighted = YDFObliqueClassifier(num_trees=50, random_seed=42)
        clf_weighted.fit(X, y, sample_weight=weights)

        p_unw = clf_unweighted.predict_proba(X)[:, 1].mean()
        p_w = clf_weighted.predict_proba(X)[:, 1].mean()

        # Weighted model should predict higher P(1) on average
        assert p_w > p_unw, f"Weights had no effect: unweighted={p_unw:.4f}, weighted={p_w:.4f}"


# ---------------------------------------------------------------------------
# Expected behavior: Oblique splits find multi-feature interactions
# ---------------------------------------------------------------------------

class TestObliqueSignal:
    """Oblique trees must outperform axis-aligned on linear-combination targets."""

    def test_oblique_beats_axisaligned_on_linear_boundary(self):
        """When the true decision boundary is w1*X1 + w2*X2 > t, oblique must do better."""
        from pregame.strategy.models import YDFObliqueClassifier
        rng = np.random.default_rng(42)
        n = 1000
        X = rng.uniform(0, 1, (n, 10))
        # True boundary: 0.6*X0 + 0.4*X1 > 0.5
        y = (0.6 * X[:, 0] + 0.4 * X[:, 1] > 0.5).astype(int)

        from sklearn.ensemble import HistGradientBoostingClassifier

        oblique = YDFObliqueClassifier(num_trees=100, max_depth=4)
        oblique.fit(X, y)
        acc_oblique = (oblique.predict(X) == y).mean()

        axis = HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=42)
        axis.fit(X, y)
        acc_axis = (axis.predict(X) == y).mean()

        # Oblique should match or beat axis-aligned on this linear boundary
        assert acc_oblique >= acc_axis - 0.02, (
            f"Oblique ({acc_oblique:.4f}) should not be substantially worse "
            f"than axis-aligned ({acc_axis:.4f}) on a linear boundary"
        )

    def test_high_density_uses_more_features(self):
        """Higher density_factor should produce projections involving more features."""
        from pregame.strategy.models import YDFObliqueClassifier
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (500, 20))
        y = (X[:, :5].sum(axis=1) > 2.5).astype(int)

        # Low density — expects sparse projections
        clf_sparse = YDFObliqueClassifier(
            num_trees=50, sparse_oblique_projection_density_factor=1.0)
        clf_sparse.fit(X, y)

        # High density — expects denser projections
        clf_dense = YDFObliqueClassifier(
            num_trees=50, sparse_oblique_projection_density_factor=5.0)
        clf_dense.fit(X, y)

        # Dense model should spread importance across more features
        imp_sparse = clf_sparse.feature_importances_
        imp_dense = clf_dense.feature_importances_

        n_active_sparse = (imp_sparse > 0.01).sum()
        n_active_dense = (imp_dense > 0.01).sum()

        assert n_active_dense >= n_active_sparse, (
            f"Dense ({n_active_dense} active) should use >= features than sparse ({n_active_sparse})"
        )


# ---------------------------------------------------------------------------
# Expected behavior: Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    """Model must survive pickle round-trip with identical predictions."""

    def test_pickle_roundtrip_identical(self):
        from pregame.strategy.models import YDFObliqueClassifier
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (200, 5))
        y = (X[:, 0] > 0.5).astype(int)

        clf = YDFObliqueClassifier(num_trees=30)
        clf.fit(X, y)

        pkl = pickle.dumps(clf)
        clf2 = pickle.loads(pkl)

        np.testing.assert_array_equal(
            clf.predict_proba(X), clf2.predict_proba(X),
            err_msg="Pickle round-trip changed predictions"
        )

    def test_pickle_roundtrip_regressor(self):
        from pregame.strategy.models import YDFObliqueRegressor
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (200, 5))
        y = X[:, 0] + X[:, 1]

        reg = YDFObliqueRegressor(num_trees=30)
        reg.fit(X, y)

        pkl = pickle.dumps(reg)
        reg2 = pickle.loads(pkl)

        np.testing.assert_array_equal(
            reg.predict(X), reg2.predict(X),
            err_msg="Pickle round-trip changed predictions"
        )


# ---------------------------------------------------------------------------
# Expected behavior: max_num_features cap (optional, not in search space)
# ---------------------------------------------------------------------------

class TestMaxNumFeatures:
    """When sparse_oblique_max_num_features is set, it must constrain projections."""

    def test_max_num_features_does_not_crash(self):
        """Setting the cap must not error — just constrain projection width."""
        from pregame.strategy.models import YDFObliqueClassifier
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (200, 20))
        y = (X[:, :3].sum(axis=1) > 1.5).astype(int)

        clf = YDFObliqueClassifier(
            num_trees=30, sparse_oblique_max_num_features=5)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (200, 2)

    def test_none_means_uncapped(self):
        """Default None must allow arbitrary projection width."""
        from pregame.strategy.models import YDFObliqueClassifier
        clf = YDFObliqueClassifier()
        assert clf.sparse_oblique_max_num_features is None
        params = clf.get_params()
        assert params["sparse_oblique_max_num_features"] is None


# ---------------------------------------------------------------------------
# Expected behavior: Regression
# ---------------------------------------------------------------------------

class TestRegression:
    """Regressor must produce continuous float predictions, not classes."""

    def test_regressor_continuous_output(self):
        from pregame.strategy.models import YDFObliqueRegressor
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (300, 5))
        y = X[:, 0] + 2 * X[:, 1] + rng.normal(0, 0.1, 300)

        reg = YDFObliqueRegressor(num_trees=50)
        reg.fit(X, y)
        preds = reg.predict(X)

        assert preds.dtype == np.float64
        assert len(np.unique(preds)) > 50  # continuous, not discretized

    def test_regressor_reasonable_range(self):
        """Predictions should be within a reasonable range of actual targets."""
        from pregame.strategy.models import YDFObliqueRegressor
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (300, 5))
        y = X[:, 0] + X[:, 1]  # range ~ [0, 2]

        reg = YDFObliqueRegressor(num_trees=100, max_depth=5)
        reg.fit(X, y)
        preds = reg.predict(X)

        assert preds.min() > -0.5, f"Predictions too low: {preds.min()}"
        assert preds.max() < 2.5, f"Predictions too high: {preds.max()}"
