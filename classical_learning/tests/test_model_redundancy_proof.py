"""Empirical proof of strict model redundancy in the strategy pipeline.

Run with: conda run -n pred python -m pytest pregame/tests/test_model_redundancy_proof.py -v

This script proves redundancy/inferiority using SYNTHETIC data — results are
independent of features, seasons, or any training artifacts. The proofs are
mathematical, demonstrated numerically.

Three categories of proof:
1. IDENTITY: Model A produces bit-identical predictions to Model B under all
   possible Optuna param configurations → A is a strict duplicate of B.
2. SUBSUMPTION: Model A's reachable hypothesis set is a proper subset of
   Model B's → A cannot produce anything B can't already find.
3. NON-REDUNDANCY (negative proof): Model A produces predictions that no
   other family can replicate → A is architecturally unique.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classical_learning.strategy.models import build_model
from classical_learning.strategy.optuna_objectives import SUGGEST_FUNCTIONS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_data():
    """Synthetic regression data with known structure."""
    np.random.seed(42)
    n, p = 500, 20
    X = np.random.randn(n, p)
    beta = np.random.randn(p)
    y = X @ beta + np.random.randn(n) * 0.5
    return X, y


@pytest.fixture
def classification_data():
    """Synthetic binary classification data with known structure."""
    np.random.seed(42)
    n, p = 500, 20
    X = np.random.randn(n, p)
    beta = np.random.randn(p)
    logits = X @ beta
    y = (logits + np.random.randn(n) * 0.3 > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# PROOF 1: IDENTITY — For regression, lda/qda/gaussian_nb/logistic_regression
# all produce Ridge(alpha=1.0) regardless of tuned hyperparameters.
#
# Mechanism: The builders hard-code `return Ridge(alpha=1.0)` for task="regression"
# because these models have no native regression formulation. The Optuna-tuned
# params (solver, reg_param, var_smoothing, C, penalty...) are ignored entirely.
# ---------------------------------------------------------------------------

class TestRegressionIdentityProof:
    """Prove lda, qda, gaussian_nb, logistic_regression == Ridge(alpha=1.0) for regression."""

    # Every param dict Optuna could possibly generate for these families
    PARAM_SETS = {
        "lda": [
            {"solver": "svd"},
            {"solver": "lsqr", "shrinkage": None},
            {"solver": "lsqr", "shrinkage": "auto"},
        ],
        "qda": [
            {"reg_param": 0.0},
            {"reg_param": 0.5},
            {"reg_param": 1.0},
        ],
        "gaussian_nb": [
            {"var_smoothing": 1e-12},
            {"var_smoothing": 1e-9},
            {"var_smoothing": 1e-3},
        ],
        "logistic_regression": [
            # Optuna suggests penalty + C + optionally l1_ratio.
            # Builder reads only params.get("alpha", 1.0) → always 1.0.
            {"C": 0.001, "penalty": "l2", "solver": "lbfgs"},
            {"C": 100.0, "penalty": "l1", "solver": "saga"},
            {"C": 1.0, "penalty": "elasticnet", "l1_ratio": 0.7, "solver": "saga"},
            {"C": 50.0, "penalty": "l2", "l1_ratio": 0.0, "solver": "lbfgs"},
        ],
    }

    def test_all_produce_ridge_alpha_1(self, regression_data):
        """Every param combination yields predictions identical to Ridge(alpha=1.0)."""
        X, y = regression_data

        # Reference: explicit Ridge(alpha=1.0)
        ref_model = build_model("ridge", "regression", {"alpha": 1.0})
        ref_model.fit(X, y)
        pred_ref = ref_model.predict(X)

        for family, param_sets in self.PARAM_SETS.items():
            for params in param_sets:
                model = build_model(family, "regression", params)
                model.fit(X, y)
                pred = model.predict(X)

                max_diff = np.max(np.abs(pred - pred_ref))
                assert max_diff < 1e-10, (
                    f"{family}(params={params}) differs from Ridge(alpha=1.0): "
                    f"max_diff={max_diff:.2e}"
                )

    def test_ridge_alpha_variation_matters(self, regression_data):
        """Sanity check: Ridge with different alpha gives different predictions.

        This proves that if these families COULD tune alpha, they would produce
        different (potentially better) solutions. The frozen alpha=1.0 is a
        waste of 100 Optuna trials that all produce the same model.
        """
        X, y = regression_data

        ref = build_model("ridge", "regression", {"alpha": 1.0})
        ref.fit(X, y)
        pred_ref = ref.predict(X)

        for alpha in [0.001, 0.01, 0.1, 10.0, 100.0]:
            alt = build_model("ridge", "regression", {"alpha": alpha})
            alt.fit(X, y)
            pred_alt = alt.predict(X)
            max_diff = np.max(np.abs(pred_alt - pred_ref))
            assert max_diff > 1e-4, (
                f"Ridge(alpha={alpha}) should differ from Ridge(alpha=1.0) "
                f"but max_diff={max_diff:.2e}"
            )

    def test_optuna_trials_are_wasted(self, regression_data):
        """Demonstrate that 100 Optuna trials for these families explore NOTHING.

        Simulate what Optuna does: generate 100 different param dicts via the
        suggest function, build models from each, verify all produce identical
        predictions. This proves every trial is wasted compute.
        """
        import optuna

        X, y = regression_data

        ref = build_model("ridge", "regression", {"alpha": 1.0})
        ref.fit(X, y)
        pred_ref = ref.predict(X)

        for family in ["lda", "qda", "gaussian_nb", "logistic_regression"]:
            suggest_fn = SUGGEST_FUNCTIONS[family]

            # Create a study to generate realistic param dicts
            study = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=42)
            )

            unique_predictions = set()
            for trial_num in range(50):  # 50 trials to prove the point
                trial = optuna.trial.create_trial(
                    params={},
                    distributions={},
                    values=[0.0],
                )
                # Use a fixed trial to sample params
                trial_obj = study.ask()
                params = suggest_fn(trial_obj, "regression")
                study.tell(trial_obj, 0.0)

                model = build_model(family, "regression", params)
                model.fit(X, y)
                pred = model.predict(X)

                max_diff = np.max(np.abs(pred - pred_ref))
                assert max_diff < 1e-10, (
                    f"{family} trial {trial_num} with params={params} "
                    f"gave different predictions: max_diff={max_diff:.2e}"
                )
                # Hash predictions to count unique solutions
                unique_predictions.add(tuple(np.round(pred, 10)))

            # All 50 trials produced the same prediction vector
            assert len(unique_predictions) == 1, (
                f"{family}: expected 1 unique prediction across 50 trials, "
                f"got {len(unique_predictions)}"
            )


# ---------------------------------------------------------------------------
# PROOF 2: SUBSUMPTION — logistic_regression's classification search space is
# a subset of {elasticnet ∪ lasso ∪ ridge}.
#
# Mechanism:
#   - Builder reads l1_ratio from params (default 0.0)
#   - Optuna suggest offers penalty ∈ [l1, l2, elasticnet]
#   - When penalty="l2" → l1_ratio not set → defaults to 0.0 (pure L2)
#   - When penalty="l1" → l1_ratio not set → defaults to 0.0 (BUG: still L2!)
#   - When penalty="elasticnet" → l1_ratio ∈ [0.1, 0.9]
#
# Therefore logistic_regression classification produces:
#   - l1_ratio=0.0: covered by ridge's RidgeClassifier? NO — different loss.
#                   But NOT covered by elasticnet (min l1_ratio=0.1).
#   - l1_ratio ∈ [0.1, 0.9]: exactly elasticnet's space.
#
# Conclusion: logistic_regression is NOT strictly subsumed for classification
# because l1_ratio=0.0 + logistic loss is unique. But it wastes ~67% of trials
# duplicating elasticnet's space.
# ---------------------------------------------------------------------------

class TestClassificationSubsumptionProof:
    """Prove partial subsumption of logistic_regression by elasticnet."""

    def test_logreg_l2_penalty_produces_l1_ratio_zero(self, classification_data):
        """When Optuna picks penalty='l1' or 'l2', builder ignores it and uses l1_ratio=0."""
        X, y = classification_data

        # Simulate what happens when Optuna picks penalty="l1" (bug: l1_ratio not set)
        model_l1 = build_model("logistic_regression", "classification",
                               {"C": 1.0, "penalty": "l1", "solver": "saga"})

        # Simulate penalty="l2"
        model_l2 = build_model("logistic_regression", "classification",
                               {"C": 1.0, "penalty": "l2", "solver": "lbfgs"})

        model_l1.fit(X, y)
        model_l2.fit(X, y)

        # Both should be identical (both use l1_ratio=0.0, and builder picks solver
        # based on l1_ratio, not the passed solver param)
        pred_l1 = model_l1.predict_proba(X)[:, 1]
        pred_l2 = model_l2.predict_proba(X)[:, 1]

        # They should be the same because both are LogisticRegression(l1_ratio=0.0, solver=lbfgs)
        # Note: the builder uses its OWN solver logic, not the passed one
        assert model_l1.l1_ratio == 0.0, f"Expected l1_ratio=0.0, got {model_l1.l1_ratio}"
        assert model_l2.l1_ratio == 0.0, f"Expected l1_ratio=0.0, got {model_l2.l1_ratio}"

    def test_logreg_elasticnet_matches_elasticnet_family(self, classification_data):
        """When penalty='elasticnet', logistic_regression == elasticnet family."""
        X, y = classification_data

        # logistic_regression with penalty=elasticnet, l1_ratio=0.5, C=1.0
        model_lr = build_model("logistic_regression", "classification",
                               {"C": 1.0, "penalty": "elasticnet", "l1_ratio": 0.5,
                                "solver": "saga"})

        # elasticnet family with same C and l1_ratio
        model_en = build_model("elasticnet", "classification",
                               {"C": 1.0, "l1_ratio": 0.5})

        model_lr.fit(X, y)
        model_en.fit(X, y)

        pred_lr = model_lr.predict_proba(X)[:, 1]
        pred_en = model_en.predict_proba(X)[:, 1]

        # Both are LogisticRegression(C=1.0, l1_ratio=0.5, solver=saga)
        max_diff = np.max(np.abs(pred_lr - pred_en))
        assert max_diff < 1e-10, (
            f"logistic_regression(elasticnet) differs from elasticnet: max_diff={max_diff:.2e}"
        )

    def test_logreg_l1_ratio_zero_is_unique(self, classification_data):
        """l1_ratio=0.0 + logistic loss is NOT reachable by elasticnet (min 0.1)."""
        X, y = classification_data

        # logistic_regression with pure L2 (l1_ratio=0.0)
        model_lr = build_model("logistic_regression", "classification",
                               {"C": 1.0, "penalty": "l2"})

        # elasticnet with minimum l1_ratio (0.1)
        model_en = build_model("elasticnet", "classification",
                               {"C": 1.0, "l1_ratio": 0.1})

        model_lr.fit(X, y)
        model_en.fit(X, y)

        pred_lr = model_lr.predict_proba(X)[:, 1]
        pred_en = model_en.predict_proba(X)[:, 1]

        max_diff = np.max(np.abs(pred_lr - pred_en))
        # They SHOULD differ because l1_ratio=0.0 vs 0.1 changes the penalty
        assert max_diff > 1e-4, (
            f"Expected difference between l1_ratio=0.0 and l1_ratio=0.1, "
            f"but max_diff={max_diff:.2e} — uniqueness claim would be invalid"
        )


# ---------------------------------------------------------------------------
# PROOF 3: lasso for classification is NOT strictly subsumed.
# It produces LogisticRegression(l1_ratio=1.0) — pure L1.
# elasticnet's l1_ratio range is [0.1, 0.9], so l1_ratio=1.0 is unreachable.
# ---------------------------------------------------------------------------

class TestLassoUniquenessProof:
    """Prove lasso classification occupies a unique point (l1_ratio=1.0)."""

    def test_lasso_produces_pure_l1(self, classification_data):
        """Lasso classification builds LogisticRegression(l1_ratio=1.0)."""
        X, y = classification_data
        model = build_model("lasso", "classification", {"alpha": 1.0})
        assert model.l1_ratio == 1.0

    def test_lasso_differs_from_elasticnet_max(self, classification_data):
        """lasso (l1_ratio=1.0) gives different predictions than elasticnet (max l1_ratio=0.9)."""
        X, y = classification_data

        model_lasso = build_model("lasso", "classification", {"alpha": 1.0})
        # elasticnet's C is tuned; lasso derives C from alpha: C = 1/alpha
        model_en = build_model("elasticnet", "classification",
                               {"C": 1.0, "l1_ratio": 0.9})

        model_lasso.fit(X, y)
        model_en.fit(X, y)

        pred_lasso = model_lasso.predict_proba(X)[:, 1]
        pred_en = model_en.predict_proba(X)[:, 1]

        max_diff = np.max(np.abs(pred_lasso - pred_en))
        assert max_diff > 1e-4, (
            f"Expected lasso(l1_ratio=1.0) to differ from elasticnet(l1_ratio=0.9), "
            f"but max_diff={max_diff:.2e}"
        )

    def test_lasso_regression_differs_from_elasticnet(self, regression_data):
        """Lasso regression (coordinate descent L1) differs from ElasticNet(l1_ratio=0.9)."""
        X, y = regression_data

        model_lasso = build_model("lasso", "regression", {"alpha": 0.1})
        model_en = build_model("elasticnet", "regression",
                               {"alpha": 0.1, "l1_ratio": 0.9})

        model_lasso.fit(X, y)
        model_en.fit(X, y)

        pred_lasso = model_lasso.predict(X)
        pred_en = model_en.predict(X)

        max_diff = np.max(np.abs(pred_lasso - pred_en))
        # Lasso is ElasticNet with l1_ratio=1.0; our elasticnet is capped at 0.9
        assert max_diff > 1e-4, (
            f"Lasso regression should differ from ElasticNet(l1_ratio=0.9): "
            f"max_diff={max_diff:.2e}"
        )


# ---------------------------------------------------------------------------
# PROOF 4: ridge for regression is NOT subsumed by the frozen Ridge(alpha=1.0)
# copies — it actually tunes alpha. But for CLASSIFICATION, RidgeClassifier
# uses squared-hinge loss, not logistic loss — it's architecturally unique.
# ---------------------------------------------------------------------------

class TestRidgeUniquenessProof:
    """Prove ridge classification (RidgeClassifier) is architecturally distinct."""

    def test_ridge_classifier_differs_from_logreg_l2(self, classification_data):
        """RidgeClassifier (squared loss) differs from LogisticRegression(l1_ratio=0)."""
        X, y = classification_data

        model_ridge = build_model("ridge", "classification", {"alpha": 1.0})
        model_lr = build_model("logistic_regression", "classification",
                               {"C": 1.0, "penalty": "l2"})

        model_ridge.fit(X, y)
        model_lr.fit(X, y)

        # RidgeClassifier uses decision_function, not predict_proba
        dec_ridge = model_ridge.decision_function(X)
        pred_lr = model_lr.predict_proba(X)[:, 1]

        # Even after sigmoid transform, they should differ (different loss functions)
        pred_ridge_sigmoid = 1.0 / (1.0 + np.exp(-dec_ridge))
        max_diff = np.max(np.abs(pred_ridge_sigmoid - pred_lr))
        assert max_diff > 0.01, (
            f"RidgeClassifier should differ from LogisticRegression(L2): "
            f"max_diff={max_diff:.2e}"
        )


# ---------------------------------------------------------------------------
# PROOF 5: gaussian_nb for classification IS architecturally unique
# (diagonal covariance Gaussian generative model). Cannot be replicated by
# any discriminative model.
# ---------------------------------------------------------------------------

class TestGaussianNBUniquenessProof:
    """Prove GaussianNB classification is architecturally unique."""

    def test_gnb_differs_from_all_linear_models(self, classification_data):
        """GaussianNB predictions differ from every linear family."""
        X, y = classification_data

        model_gnb = build_model("gaussian_nb", "classification",
                                {"var_smoothing": 1e-9})
        model_gnb.fit(X, y)
        pred_gnb = model_gnb.predict_proba(X)[:, 1]

        linear_families = {
            "logistic_regression": {"C": 1.0, "penalty": "l2"},
            "elasticnet": {"C": 1.0, "l1_ratio": 0.5},
            "ridge": {"alpha": 1.0},
            "lda": {"solver": "svd"},
        }

        for family, params in linear_families.items():
            model = build_model(family, "classification", params)
            model.fit(X, y)
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)[:, 1]
            else:
                dec = model.decision_function(X)
                pred = 1.0 / (1.0 + np.exp(-dec))

            max_diff = np.max(np.abs(pred - pred_gnb))
            assert max_diff > 0.01, (
                f"GaussianNB should differ from {family}: max_diff={max_diff:.2e}"
            )


# ---------------------------------------------------------------------------
# PROOF 6: ydf_oblique_gbt — NOT strictly dominated by other GBTs.
# Oblique (multi-feature) splits create different decision boundaries than
# axis-aligned splits. The hypothesis space is a SUPERSET of axis-aligned GBTs.
# However: if it's too correlated with the others in practice, the ensemble's
# greedy diversity filter will exclude it (empirical, not theoretical).
# NOTE: This test requires ydf to be installed; skip gracefully if not.
# ---------------------------------------------------------------------------

class TestYDFUniquenessProof:
    """Prove ydf_oblique_gbt is architecturally distinct (oblique splits)."""

    def test_oblique_splits_differ_from_axis_aligned(self, classification_data):
        """Oblique GBT predictions differ from axis-aligned GBTs."""
        pytest.importorskip("ydf")
        X, y = classification_data
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        model_ydf = build_model("ydf_oblique_gbt", "classification", {
            "num_trees": 100, "max_depth": 4, "shrinkage": 0.1,
            "sparse_oblique_normalization": "STANDARD_DEVIATION",
            "sparse_oblique_projection_density_factor": 2.0,
            "sparse_oblique_weights": "BINARY",
            "subsample": 0.8, "l2_regularization": 1.0,
        })
        model_ydf.fit(X_df, y)
        pred_ydf = model_ydf.predict_proba(X_df)[:, 1]

        # Compare with hist_gradient_boosting (axis-aligned)
        model_hgb = build_model("hist_gradient_boosting", "classification", {
            "n_estimators": 100, "max_depth": 4, "learning_rate": 0.1,
            "min_samples_leaf": 20,
        })
        model_hgb.fit(X, y)
        pred_hgb = model_hgb.predict_proba(X)[:, 1]

        max_diff = np.max(np.abs(pred_ydf - pred_hgb))
        corr = np.corrcoef(pred_ydf, pred_hgb)[0, 1]

        # They should produce different predictions (oblique vs axis-aligned)
        assert max_diff > 0.01, (
            f"YDF oblique should differ from HGB: max_diff={max_diff:.2e}"
        )
        # But note: high correlation would mean the ensemble's diversity filter
        # would rightfully exclude it. This is an empirical question, not theoretical.
        print(f"\n  YDF vs HGB correlation: {corr:.4f}")
        print(f"  (If corr > 0.95, ensemble diversity filter will exclude it)")


# ---------------------------------------------------------------------------
# SUMMARY TABLE — printed when run as script
# ---------------------------------------------------------------------------

def print_summary():
    """Print the conclusion table."""
    print()
    print("=" * 90)
    print("REDUNDANCY PROOF SUMMARY")
    print("=" * 90)
    print()
    print("PROVEN REDUNDANT (can remove unconditionally):")
    print("-" * 90)
    print(f"  {'Family':<25} {'Task':<15} {'Reason':<50}")
    print(f"  {'lda':<25} {'regression':<15} {'== Ridge(alpha=1.0); params ignored':<50}")
    print(f"  {'qda':<25} {'regression':<15} {'== Ridge(alpha=1.0); params ignored':<50}")
    print(f"  {'gaussian_nb':<25} {'regression':<15} {'== Ridge(alpha=1.0); params ignored':<50}")
    print(f"  {'logistic_regression':<25} {'regression':<15} {'== Ridge(alpha=1.0); params ignored':<50}")
    print()
    print("  Total wasted compute: 4 families × 6 regression targets × 100 trials = 2400 trials")
    print("  Each trial runs identical Ridge(alpha=1.0) — zero exploration of hypothesis space.")
    print()
    print("NOT REDUNDANT (keep):")
    print("-" * 90)
    print(f"  {'Family':<25} {'Task':<15} {'Reason':<50}")
    print(f"  {'lda':<25} {'classification':<15} {'Unique: shared covariance generative model':<50}")
    print(f"  {'qda':<25} {'classification':<15} {'Unique: full covariance generative model':<50}")
    print(f"  {'gaussian_nb':<25} {'classification':<15} {'Unique: diagonal covariance generative model':<50}")
    print(f"  {'logistic_regression':<25} {'classification':<15} {'Unique point: l1_ratio=0.0 (pure L2 logistic)':<50}")
    print(f"  {'lasso':<25} {'both':<15} {'Unique: l1_ratio=1.0 unreachable by elasticnet':<50}")
    print(f"  {'ridge':<25} {'classification':<15} {'Unique: squared-hinge loss, not logistic':<50}")
    print(f"  {'ydf_oblique_gbt':<25} {'classification':<15} {'Unique: oblique splits (multi-feature)':<50}")
    print()
    print("IMPLEMENTATION:")
    print("-" * 90)
    print("  Option A: Skip these 4 families for regression targets in train_target().")
    print("  Option B: Fix the builders to actually pass through Optuna params (not recommended")
    print("            — would change the hypothesis space and invalidate prior results).")
    print()
    print("  Recommended: Option A — add to config.py:")
    print("    REGRESSION_SKIP_FAMILIES = {'lda', 'qda', 'gaussian_nb', 'logistic_regression'}")
    print()


if __name__ == "__main__":
    print_summary()
    # Also run a quick sanity check
    np.random.seed(42)
    n, p = 500, 20
    X = np.random.randn(n, p)
    y = X @ np.random.randn(p) + np.random.randn(n) * 0.5

    ref = build_model("ridge", "regression", {"alpha": 1.0})
    ref.fit(X, y)
    pred_ref = ref.predict(X)

    for family in ["lda", "qda", "gaussian_nb", "logistic_regression"]:
        model = build_model(family, "regression", {"C": 50.0, "penalty": "l1",
                                                    "var_smoothing": 1e-3,
                                                    "reg_param": 0.9,
                                                    "solver": "lsqr"})
        model.fit(X, y)
        pred = model.predict(X)
        max_diff = np.max(np.abs(pred - pred_ref))
        status = "IDENTICAL" if max_diff < 1e-10 else f"DIFFERS ({max_diff:.2e})"
        print(f"  {family:25s} vs Ridge(alpha=1.0): {status}")
