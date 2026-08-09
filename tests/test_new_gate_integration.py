"""Integration tests for filter_features_v2() — PCA-validated gating system.

Run: conda run -n pred python -m pytest tests/test_new_gate_integration.py -v

Tests define expected behavior BEFORE implementation exists.
Each test verifies one specific aspect of the gate's logic.
"""
import numpy as np
import pandas as pd
import pytest

from pregame.analysis.feature_importance import (
    filter_features_v2,
    feature_score,
    feature_score_clt,
    EB_PRIORS,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_fold_raw(n_features=10, n_folds=8, values=None):
    """Create a fake fold-structured raw DataFrame (n_folds x n_features)."""
    feats = [f"feat_{i}" for i in range(n_features)]
    if values is None:
        values = np.random.default_rng(42).normal(0.01, 0.003, (n_folds, n_features))
    return pd.DataFrame(values, columns=feats)


def _make_tree_raw(n_features=10, n_trees=1000, values=None):
    """Create a fake tree-level raw DataFrame (n_trees x n_features)."""
    feats = [f"feat_{i}" for i in range(n_features)]
    if values is None:
        values = np.random.default_rng(42).normal(0.01, 0.005, (n_trees, n_features))
    return pd.DataFrame(values, columns=feats)


def _make_cluster_raw(n_clusters=3, n_folds=8, values=None):
    """CFI_MDA: fold-structured cluster importance (n_folds x n_clusters)."""
    cols = [f"cluster_{i}" for i in range(n_clusters)]
    if values is None:
        values = np.random.default_rng(42).normal(0.01, 0.003, (n_folds, n_clusters))
    return pd.DataFrame(values, columns=cols)


def _make_cfi_mdi_raw(n_clusters=3, n_trees=1000, values=None):
    """CFI_MDI: per-tree cluster importance (n_trees x n_clusters)."""
    cols = [f"cluster_{i}" for i in range(n_clusters)]
    if values is None:
        values = np.random.default_rng(42).normal(0.01, 0.005, (n_trees, n_clusters))
    return pd.DataFrame(values, columns=cols)


def _all_eligible_crosscheck():
    """PCA cross-check where all methods are eligible."""
    return {
        "SFI": {"tau": 0.5, "p_value": 0.001},
        "DESUB_MDA": {"tau": 0.4, "p_value": 0.01},
        "PCA_MDA": {"tau": 0.6, "p_value": 0.0},
        "RESID_MDA": {"tau": 0.3, "p_value": 0.02},
        "MDI": {"tau": 0.4, "p_value": 0.005},
        "CFI_MDI": {"tau": 0.7, "p_value": 0.0},
        "CFI_MDA": {"tau": 0.5, "p_value": 0.0},
    }


def _only_cfi_eligible_crosscheck():
    """PCA cross-check where only CFI methods are eligible (like extra_innings)."""
    return {
        "SFI": {"tau": 0.1, "p_value": 0.3},
        "DESUB_MDA": {"tau": -0.2, "p_value": 0.01},
        "PCA_MDA": {"tau": -0.02, "p_value": 0.8},
        "RESID_MDA": {"tau": -0.23, "p_value": 0.01},
        "MDI": {"tau": 0.02, "p_value": 0.86},
        "CFI_MDI": {"tau": 0.63, "p_value": 0.0},
        "CFI_MDA": {"tau": 0.30, "p_value": 0.0},
    }


def _standard_clusters(n_features=10, n_clusters=3):
    """Map features to clusters."""
    clusters = {}
    for i in range(n_clusters):
        start = i * (n_features // n_clusters)
        end = start + (n_features // n_clusters) if i < n_clusters - 1 else n_features
        clusters[f"cluster_{i}"] = [f"feat_{j}" for j in range(start, end)]
    return clusters


# ─── Test 1: Output schema ───────────────────────────────────────────────────

class TestGateOutputSchema:
    """New gate must produce all columns that feature_routing.py expects."""

    def test_required_columns_present(self):
        n_feat, n_folds, n_trees = 10, 8, 1000
        clusters = _standard_clusters(n_feat, 3)
        report = filter_features_v2(
            sfi_raw=_make_fold_raw(n_feat, n_folds),
            desub_mda_raw=_make_fold_raw(n_feat, n_folds),
            pca_mda_raw=_make_fold_raw(n_feat, n_folds),
            resid_mda_raw=_make_fold_raw(n_feat, n_folds),
            mdi_raw=_make_tree_raw(n_feat, n_trees),
            cfi_mda_raw=_make_cluster_raw(len(clusters), n_folds),
            cfi_mdi_raw=_make_cfi_mdi_raw(len(clusters), n_trees),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_all_eligible_crosscheck(),
        )
        required = [
            "mdi_passes", "sfi_passes", "desub_mda_passes",
            "pca_mda_passes", "resid_mda_passes", "cfi_mda_cluster_passes",
            "tier", "composite_rank",
        ]
        for col in required:
            assert col in report.columns, f"Missing column: {col}"
        assert report.index.name == "feature"
        assert set(report["tier"].dropna().unique()) <= {"ACCEPTED", "NEEDS SPECIFICATION", "REJECTED"}


# ─── Test 2: PCA cross-check eligibility ─────────────────────────────────────

class TestPCACrosscheckEligibility:

    def test_ineligible_method_excluded(self):
        """Method with tau<0 or p>0.05 → excluded from gating."""
        n_feat, n_folds, n_trees = 10, 8, 1000
        clusters = _standard_clusters(n_feat, 3)

        crosscheck = _all_eligible_crosscheck()
        crosscheck["SFI"] = {"tau": -0.2, "p_value": 0.03}

        strongly_positive = np.full((n_folds, n_feat), 0.05)
        sfi_negative = np.full((n_folds, n_feat), -0.8)

        report = filter_features_v2(
            sfi_raw=pd.DataFrame(sfi_negative, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(strongly_positive, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(strongly_positive, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(strongly_positive, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=_make_tree_raw(n_feat, n_trees),
            cfi_mda_raw=_make_cluster_raw(len(clusters), n_folds),
            cfi_mdi_raw=_make_cfi_mdi_raw(len(clusters), n_trees),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=crosscheck,
        )
        assert (report["sfi_passes"].isna()).all(), "Ineligible SFI should be NaN"
        assert report["tier"].iloc[0] != "REJECTED", "SFI (ineligible) should not cause rejection"


# ─── Test 3: Only CFI eligible → cluster-only gating ─────────────────────────

class TestOnlyCFIEligible:

    def test_features_get_needs_spec_when_no_per_feature_methods(self):
        """When only CFI methods are eligible, individual features → NEEDS_SPECIFICATION."""
        n_feat, n_folds, n_trees = 10, 8, 1000
        clusters = _standard_clusters(n_feat, 3)

        strongly_positive_clusters = np.full((n_folds, len(clusters)), 0.05)
        report = filter_features_v2(
            sfi_raw=_make_fold_raw(n_feat, n_folds),
            desub_mda_raw=_make_fold_raw(n_feat, n_folds),
            pca_mda_raw=_make_fold_raw(n_feat, n_folds),
            resid_mda_raw=_make_fold_raw(n_feat, n_folds),
            mdi_raw=_make_tree_raw(n_feat, n_trees),
            cfi_mda_raw=pd.DataFrame(strongly_positive_clusters, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            cfi_mdi_raw=_make_cfi_mdi_raw(len(clusters), n_trees),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_only_cfi_eligible_crosscheck(),
        )
        non_vetoed = report[report["tier"] != "REJECTED"]
        assert (non_vetoed["tier"] == "NEEDS SPECIFICATION").all()


# ─── Test 4: Conservative union — all eligible reject ─────────────────────────

class TestConservativeUnionAllReject:

    def test_all_eligible_reject_yields_rejected(self):
        """Feature rejected by ALL eligible per-feature tests → REJECTED."""
        n_feat, n_folds, n_trees = 5, 8, 1000
        clusters = _standard_clusters(n_feat, 2)

        rng = np.random.default_rng(99)
        all_negative = rng.normal(-1.0, 0.01, (n_folds, n_feat))
        mdi_below_null = rng.normal(0.001, 0.0001, (n_trees, n_feat))

        positive_clusters = np.full((n_folds, len(clusters)), 0.05)
        positive_cfi_mdi = np.full((n_trees, len(clusters)), 0.01)

        report = filter_features_v2(
            sfi_raw=pd.DataFrame(all_negative, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(all_negative, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(all_negative, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(all_negative, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=pd.DataFrame(mdi_below_null, columns=[f"feat_{i}" for i in range(n_feat)]),
            cfi_mda_raw=pd.DataFrame(positive_clusters, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            cfi_mdi_raw=pd.DataFrame(positive_cfi_mdi, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_all_eligible_crosscheck(),
        )
        assert (report["tier"] == "REJECTED").all()


# ─── Test 5: Conservative union — one eligible passes ─────────────────────────

class TestConservativeUnionOnePass:

    def test_one_pass_prevents_rejection(self):
        """Feature passes ONE eligible test → NOT REJECTED (conservative)."""
        n_feat, n_folds, n_trees = 5, 8, 1000
        clusters = _standard_clusters(n_feat, 2)

        all_negative = np.full((n_folds, n_feat), -0.05)
        pca_positive = np.full((n_folds, n_feat), 0.05)
        positive_clusters = np.full((n_folds, len(clusters)), 0.05)
        positive_cfi_mdi = np.full((n_trees, len(clusters)), 0.01)

        report = filter_features_v2(
            sfi_raw=pd.DataFrame(all_negative, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(all_negative, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(pca_positive, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(all_negative, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=_make_tree_raw(n_feat, n_trees),
            cfi_mda_raw=pd.DataFrame(positive_clusters, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            cfi_mdi_raw=pd.DataFrame(positive_cfi_mdi, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_all_eligible_crosscheck(),
        )
        assert (report["tier"] != "REJECTED").all()


# ─── Test 6: CFI_MDI cluster gate (CLT z-test) ──────────────────────────────

class TestCFIMDIClusterGate:

    def test_significantly_negative_cluster_gated(self):
        """CFI_MDI CLT z-test: cluster mean significantly < 0 → gated."""
        n_feat, n_folds, n_trees = 6, 8, 1000
        clusters = {"cluster_0": [f"feat_{i}" for i in range(3)],
                    "cluster_1": [f"feat_{i}" for i in range(3, 6)]}

        cfi_mdi_vals = np.column_stack([
            np.random.default_rng(42).normal(-0.05, 0.01, n_trees),
            np.random.default_rng(43).normal(0.05, 0.01, n_trees),
        ])
        cfi_mda_vals = np.column_stack([
            np.full(n_folds, -0.02),
            np.full(n_folds, 0.05),
        ])

        positive_feats = np.full((n_folds, n_feat), 0.05)
        report = filter_features_v2(
            sfi_raw=pd.DataFrame(positive_feats, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(positive_feats, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(positive_feats, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(positive_feats, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=_make_tree_raw(n_feat, n_trees),
            cfi_mda_raw=pd.DataFrame(cfi_mda_vals, columns=["cluster_0", "cluster_1"]),
            cfi_mdi_raw=pd.DataFrame(cfi_mdi_vals, columns=["cluster_0", "cluster_1"]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_all_eligible_crosscheck(),
        )
        cluster_0_feats = report.loc[["feat_0", "feat_1", "feat_2"]]
        cluster_1_feats = report.loc[["feat_3", "feat_4", "feat_5"]]
        assert (cluster_0_feats["tier"] == "REJECTED").all()
        assert (cluster_1_feats["tier"] != "REJECTED").all()


# ─── Test 7: CFI_MDA fold-count gate ─────────────────────────────────────────

class TestCFIMDAFoldCount:

    def test_zero_to_two_positive_gates_cluster(self):
        """CFI_MDA: 0-2 positive folds → cluster gated out."""
        n_feat, n_folds, n_trees = 6, 8, 1000
        clusters = {"cluster_0": [f"feat_{i}" for i in range(3)],
                    "cluster_1": [f"feat_{i}" for i in range(3, 6)]}

        cfi_mda_vals = np.column_stack([
            np.array([-0.01, -0.02, 0.01, -0.01, -0.03, -0.02, -0.01, -0.01]),
            np.array([0.05, 0.04, 0.06, 0.03, 0.05, 0.04, 0.05, 0.06]),
        ])
        cfi_mdi_negative = np.column_stack([
            np.random.default_rng(42).normal(-0.01, 0.005, n_trees),
            np.random.default_rng(43).normal(0.05, 0.01, n_trees),
        ])

        positive_feats = np.full((n_folds, n_feat), 0.05)
        report = filter_features_v2(
            sfi_raw=pd.DataFrame(positive_feats, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(positive_feats, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(positive_feats, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(positive_feats, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=_make_tree_raw(n_feat, n_trees),
            cfi_mda_raw=pd.DataFrame(cfi_mda_vals, columns=["cluster_0", "cluster_1"]),
            cfi_mdi_raw=pd.DataFrame(cfi_mdi_negative, columns=["cluster_0", "cluster_1"]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_all_eligible_crosscheck(),
        )
        cluster_0_feats = report.loc[["feat_0", "feat_1", "feat_2"]]
        assert (cluster_0_feats["tier"] == "REJECTED").all()


# ─── Test 8: Dual cluster veto ───────────────────────────────────────────────

class TestDualClusterVeto:

    def test_both_condemn_vetoes(self):
        """Both CFI_MDI (CLT) AND CFI_MDA (fold≤2) condemn → cluster_vetoed."""
        n_feat, n_folds, n_trees = 3, 8, 1000
        clusters = {"cluster_0": [f"feat_{i}" for i in range(3)]}

        cfi_mda_all_neg = np.full((n_folds, 1), -0.01)
        cfi_mdi_neg = np.random.default_rng(42).normal(-0.05, 0.01, (n_trees, 1))

        positive_feats = np.full((n_folds, n_feat), 0.05)
        report = filter_features_v2(
            sfi_raw=pd.DataFrame(positive_feats, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(positive_feats, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(positive_feats, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(positive_feats, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=_make_tree_raw(n_feat, n_trees),
            cfi_mda_raw=pd.DataFrame(cfi_mda_all_neg, columns=["cluster_0"]),
            cfi_mdi_raw=pd.DataFrame(cfi_mdi_neg, columns=["cluster_0"]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_all_eligible_crosscheck(),
        )
        assert (report["tier"] == "REJECTED").all()


# ─── Test 8b: CFI_MDI-only cluster importance is noise ────────────────────────

class TestCFIMDIOnlyIsNoise:

    def test_cfi_mdi_only_cluster_vetoed(self):
        """Cluster with positive CFI_MDI but CFI_MDA fold-count ≤ 2 → vetoed.

        Same principle as per-feature MDI: tree-level splitting importance
        alone cannot validate a cluster if fold-structured permutation
        importance says it's dead.
        """
        n_feat, n_folds, n_trees = 3, 8, 1000
        clusters = {"cluster_0": [f"feat_{i}" for i in range(3)]}

        # CFI_MDA: 0 positive folds (condemns)
        cfi_mda_all_neg = np.full((n_folds, 1), -0.01)
        # CFI_MDI: clearly POSITIVE (does NOT condemn — tree says "important")
        rng = np.random.default_rng(42)
        cfi_mdi_positive = rng.normal(0.05, 0.01, (n_trees, 1))

        positive_feats = np.full((n_folds, n_feat), 0.05)
        report = filter_features_v2(
            sfi_raw=pd.DataFrame(positive_feats, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(positive_feats, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(positive_feats, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(positive_feats, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=_make_tree_raw(n_feat, n_trees),
            cfi_mda_raw=pd.DataFrame(cfi_mda_all_neg, columns=["cluster_0"]),
            cfi_mdi_raw=pd.DataFrame(cfi_mdi_positive, columns=["cluster_0"]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_all_eligible_crosscheck(),
        )
        assert report["cluster_vetoed"].all(), (
            "Cluster passing only CFI_MDI (tree-level) but failing CFI_MDA "
            "(fold-structured) should be vetoed"
        )
        assert (report["tier"] == "REJECTED").all()

    def test_cfi_mda_positive_prevents_veto(self):
        """Cluster with both CFI_MDI and CFI_MDA positive → NOT vetoed.

        When the fold-structured method also supports the cluster, it's real.
        """
        n_feat, n_folds, n_trees = 3, 8, 1000
        clusters = {"cluster_0": [f"feat_{i}" for i in range(3)]}

        # CFI_MDA: 7 positive folds (keeps)
        cfi_mda_positive = np.full((n_folds, 1), 0.05)
        # CFI_MDI: positive
        rng = np.random.default_rng(42)
        cfi_mdi_positive = rng.normal(0.05, 0.01, (n_trees, 1))

        positive_feats = np.full((n_folds, n_feat), 0.05)
        report = filter_features_v2(
            sfi_raw=pd.DataFrame(positive_feats, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(positive_feats, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(positive_feats, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(positive_feats, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=_make_tree_raw(n_feat, n_trees),
            cfi_mda_raw=pd.DataFrame(cfi_mda_positive, columns=["cluster_0"]),
            cfi_mdi_raw=pd.DataFrame(cfi_mdi_positive, columns=["cluster_0"]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_all_eligible_crosscheck(),
        )
        assert not report["cluster_vetoed"].any()


# ─── Test 8c: Trend rescue — growing features/clusters saved from veto ───────

class TestTrendRescue:

    def test_significant_uptrend_rescues_vetoed_cluster(self):
        """Cluster with monotonic growth and last fold positive → rescued.

        CFI_MDA fold-count is ≤ 2 (only last 2 positive) but there's a
        significant upward trend (Kendall tau p ≤ 0.05) and last fold > 0.
        Feature should NOT be rejected — it's growing into relevance.
        """
        n_feat, n_folds, n_trees = 3, 8, 1000
        clusters = {"cluster_0": [f"feat_{i}" for i in range(3)]}

        # CFI_MDA: strong monotonic growth, only last 2 folds positive
        # tau should be ~1.0 (perfectly monotonic)
        cfi_mda_growing = np.array([
            [-0.10], [-0.08], [-0.06], [-0.04], [-0.02], [-0.01], [0.01], [0.03]
        ])
        rng = np.random.default_rng(42)
        cfi_mdi_positive = rng.normal(0.05, 0.01, (n_trees, 1))

        # Per-feature data: also growing (mirrors cluster trajectory)
        feat_growing = np.array([
            [-0.10, -0.10, -0.10],
            [-0.08, -0.08, -0.08],
            [-0.06, -0.06, -0.06],
            [-0.04, -0.04, -0.04],
            [-0.02, -0.02, -0.02],
            [-0.01, -0.01, -0.01],
            [0.01, 0.01, 0.01],
            [0.03, 0.03, 0.03],
        ])

        report = filter_features_v2(
            sfi_raw=pd.DataFrame(feat_growing, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(feat_growing, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(feat_growing, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(feat_growing, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=_make_tree_raw(n_feat, n_trees),
            cfi_mda_raw=pd.DataFrame(cfi_mda_growing, columns=["cluster_0"]),
            cfi_mdi_raw=pd.DataFrame(cfi_mdi_positive, columns=["cluster_0"]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_all_eligible_crosscheck(),
        )
        assert (report["tier"] != "REJECTED").all(), (
            f"Growing features with significant uptrend should be rescued. "
            f"Got: {dict(report['tier'].value_counts())}"
        )
        assert (report["trend_rescue"] == "significant").all()

    def test_heuristic_uptrend_flags_for_review(self):
        """Positive slope + last fold > 0 but NOT significant → rescued with flag.

        Theil-Sen slope > 0 and last fold positive, but Kendall tau p > 0.05.
        Feature rescued but marked for manual review.
        """
        n_feat, n_folds, n_trees = 3, 8, 1000
        clusters = {"cluster_0": [f"feat_{i}" for i in range(3)]}

        # CFI_MDA: noisy but upward drift, last fold positive
        # tau=0.5 at n=8 → p≈0.109, NOT significant
        # Theil-Sen slope=0.0088 > 0
        cfi_mda_noisy_up = np.array([
            [-0.05], [-0.02], [-0.08], [-0.01], [-0.06], [-0.03], [-0.005], [0.02]
        ])
        rng = np.random.default_rng(42)
        cfi_mdi_positive = rng.normal(0.05, 0.01, (n_trees, 1))

        # Per-feature: same noisy uptrend
        feat_noisy_up = np.array([
            [-0.05, -0.05, -0.05],
            [-0.02, -0.02, -0.02],
            [-0.08, -0.08, -0.08],
            [-0.01, -0.01, -0.01],
            [-0.06, -0.06, -0.06],
            [-0.03, -0.03, -0.03],
            [-0.005, -0.005, -0.005],
            [0.02, 0.02, 0.02],
        ])

        report = filter_features_v2(
            sfi_raw=pd.DataFrame(feat_noisy_up, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(feat_noisy_up, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(feat_noisy_up, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(feat_noisy_up, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=_make_tree_raw(n_feat, n_trees),
            cfi_mda_raw=pd.DataFrame(cfi_mda_noisy_up, columns=["cluster_0"]),
            cfi_mdi_raw=pd.DataFrame(cfi_mdi_positive, columns=["cluster_0"]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_all_eligible_crosscheck(),
        )
        assert (report["tier"] != "REJECTED").all(), (
            f"Heuristic uptrend with last fold positive should be rescued. "
            f"Got: {dict(report['tier'].value_counts())}"
        )
        assert (report["trend_rescue"] == "heuristic").all()

    def test_no_rescue_when_last_fold_negative(self):
        """Upward trend but last fold ≤ null → NO rescue (not currently relevant)."""
        n_feat, n_folds, n_trees = 3, 8, 1000
        clusters = {"cluster_0": [f"feat_{i}" for i in range(3)]}

        # Growing but last fold is still below null for all methods
        # DESUB/PCA/RESID null=0, SFI null=log(0.5)≈-0.693
        cfi_mda_growing_not_there = np.array([
            [-0.20], [-0.18], [-0.15], [-0.12], [-0.09], [-0.06], [-0.03], [-0.01]
        ])
        rng = np.random.default_rng(42)
        cfi_mdi_positive = rng.normal(0.05, 0.01, (n_trees, 1))

        # Per-feature: all methods have last fold below their null
        # SFI null=-0.693, so use values well below that
        feat_growing_neg = np.array([
            [-1.5, -1.5, -1.5],
            [-1.4, -1.4, -1.4],
            [-1.3, -1.3, -1.3],
            [-1.2, -1.2, -1.2],
            [-1.1, -1.1, -1.1],
            [-1.0, -1.0, -1.0],
            [-0.9, -0.9, -0.9],
            [-0.8, -0.8, -0.8],
        ])

        report = filter_features_v2(
            sfi_raw=pd.DataFrame(feat_growing_neg, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(feat_growing_neg, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(feat_growing_neg, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(feat_growing_neg, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=_make_tree_raw(n_feat, n_trees),
            cfi_mda_raw=pd.DataFrame(cfi_mda_growing_not_there, columns=["cluster_0"]),
            cfi_mdi_raw=pd.DataFrame(cfi_mdi_positive, columns=["cluster_0"]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_all_eligible_crosscheck(),
        )
        assert (report["tier"] == "REJECTED").all(), (
            f"Last fold negative → no rescue even with uptrend. "
            f"Got: {dict(report['tier'].value_counts())}"
        )

    def test_no_rescue_flat_or_declining(self):
        """Declining trend with last fold positive → NO rescue."""
        n_feat, n_folds, n_trees = 3, 8, 1000
        clusters = {"cluster_0": [f"feat_{i}" for i in range(3)]}

        # Declining: only last fold is barely positive (spike), overall tau < 0
        # Fold-count: 1 positive → vetoed
        cfi_mda_declining = np.array([
            [-0.01], [-0.02], [-0.03], [-0.04], [-0.05], [-0.06], [-0.07], [0.001]
        ])
        rng = np.random.default_rng(42)
        cfi_mdi_positive = rng.normal(0.05, 0.01, (n_trees, 1))

        feat_declining = np.tile(cfi_mda_declining, (1, n_feat))

        report = filter_features_v2(
            sfi_raw=pd.DataFrame(feat_declining, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(feat_declining, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(feat_declining, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(feat_declining, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=_make_tree_raw(n_feat, n_trees),
            cfi_mda_raw=pd.DataFrame(cfi_mda_declining, columns=["cluster_0"]),
            cfi_mdi_raw=pd.DataFrame(cfi_mdi_positive, columns=["cluster_0"]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_all_eligible_crosscheck(),
        )
        assert (report["tier"] == "REJECTED").all(), (
            f"Declining trend → no rescue even with last fold barely positive. "
            f"Got: {dict(report['tier'].value_counts())}"
        )

    def test_mdi_only_with_significant_trend_rescued(self):
        """MDI-only feature with significant uptrend → rescued.

        Trend rescue applies to MDI-only rejected features too,
        not just cluster-vetoed ones.
        """
        n_feat, n_folds, n_trees = 3, 8, 1000
        clusters = _standard_clusters(n_feat, 2)

        # Per-feature: monotonic growth from below null to above null
        # SFI null = log(0.5) ≈ -0.693; DESUB/PCA/RESID null = 0
        # Make DESUB/PCA/RESID grow from negative to positive
        feat_growing = np.array([
            [-0.10, -0.10, -0.10],
            [-0.08, -0.08, -0.08],
            [-0.05, -0.05, -0.05],
            [-0.03, -0.03, -0.03],
            [-0.01, -0.01, -0.01],
            [0.00, 0.00, 0.00],
            [0.02, 0.02, 0.02],
            [0.04, 0.04, 0.04],
        ])
        # SFI: well below null (all < -0.693) → rejects
        sfi_below = np.full((n_folds, n_feat), -1.5)
        rng = np.random.default_rng(55)
        mdi_high = rng.normal(0.5, 0.01, (n_trees, n_feat))

        positive_clusters = np.full((n_folds, len(clusters)), 0.05)
        positive_cfi_mdi = np.full((n_trees, len(clusters)), 0.01)

        report = filter_features_v2(
            sfi_raw=pd.DataFrame(sfi_below, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(feat_growing, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(feat_growing, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(feat_growing, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=pd.DataFrame(mdi_high, columns=[f"feat_{i}" for i in range(n_feat)]),
            cfi_mda_raw=pd.DataFrame(positive_clusters, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            cfi_mdi_raw=pd.DataFrame(positive_cfi_mdi, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_all_eligible_crosscheck(),
        )
        # Growing feature with last fold > null should be rescued
        assert (report["tier"] != "REJECTED").all(), (
            f"MDI-only feature with significant uptrend should be rescued. "
            f"Got: {dict(report['tier'].value_counts())}"
        )

    def test_cluster_vetoed_but_per_feature_growing(self):
        """Feature in dead cluster but individually growing → rescued by per-feature trend.

        Cluster CFI_MDA is flat/dead (no cluster rescue), but the individual
        feature's PCA_MDA trajectory shows significant growth. Per-feature
        rescue applies to cluster-vetoed features too.
        """
        n_feat, n_folds, n_trees = 3, 8, 1000
        clusters = {"cluster_0": [f"feat_{i}" for i in range(3)]}

        # Cluster: dead (0 positive folds, no trend — flat negative)
        cfi_mda_dead = np.full((n_folds, 1), -0.02)
        rng = np.random.default_rng(42)
        cfi_mdi_positive = rng.normal(0.05, 0.01, (n_trees, 1))

        # Per-feature: monotonic growth into positive territory
        feat_growing = np.array([
            [-0.10, -0.10, -0.10],
            [-0.08, -0.08, -0.08],
            [-0.06, -0.06, -0.06],
            [-0.04, -0.04, -0.04],
            [-0.02, -0.02, -0.02],
            [-0.01, -0.01, -0.01],
            [0.01, 0.01, 0.01],
            [0.03, 0.03, 0.03],
        ])
        sfi_below = np.full((n_folds, n_feat), -1.5)

        report = filter_features_v2(
            sfi_raw=pd.DataFrame(sfi_below, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(feat_growing, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(feat_growing, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(feat_growing, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=_make_tree_raw(n_feat, n_trees),
            cfi_mda_raw=pd.DataFrame(cfi_mda_dead, columns=["cluster_0"]),
            cfi_mdi_raw=pd.DataFrame(cfi_mdi_positive, columns=["cluster_0"]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_all_eligible_crosscheck(),
        )
        assert (report["tier"] != "REJECTED").all(), (
            f"Feature with growing per-feature trend should be rescued even in dead cluster. "
            f"Got: {dict(report['tier'].value_counts())}"
        )
        assert (report["trend_rescue"].notna()).all()

    def test_all_reject_rescued_by_sfi_trend(self):
        """All eligible methods reject, but SFI trajectory shows growth → rescued.

        Verifies rescue applies to the all-eligible-reject path (not just
        MDI-only or cluster-veto).
        """
        n_feat, n_folds, n_trees = 3, 8, 1000
        clusters = _standard_clusters(n_feat, 2)

        rng = np.random.default_rng(33)
        # DESUB/PCA/RESID: constant negative → reject
        negative_folds = rng.normal(-0.5, 0.01, (n_folds, n_feat))
        # MDI: below null → also rejects
        mdi_low = rng.normal(0.001, 0.0001, (n_trees, n_feat))
        # SFI: monotonic growth crossing null (log(0.5)≈-0.693)
        sfi_growing = np.array([
            [-1.2, -1.2, -1.2],
            [-1.1, -1.1, -1.1],
            [-0.95, -0.95, -0.95],
            [-0.85, -0.85, -0.85],
            [-0.75, -0.75, -0.75],
            [-0.70, -0.70, -0.70],
            [-0.60, -0.60, -0.60],
            [-0.50, -0.50, -0.50],
        ])

        positive_clusters = np.full((n_folds, len(clusters)), 0.05)
        positive_cfi_mdi = np.full((n_trees, len(clusters)), 0.01)

        report = filter_features_v2(
            sfi_raw=pd.DataFrame(sfi_growing, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(negative_folds, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(negative_folds, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(negative_folds, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=pd.DataFrame(mdi_low, columns=[f"feat_{i}" for i in range(n_feat)]),
            cfi_mda_raw=pd.DataFrame(positive_clusters, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            cfi_mdi_raw=pd.DataFrame(positive_cfi_mdi, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_all_eligible_crosscheck(),
        )
        assert (report["tier"] != "REJECTED").all(), (
            f"All-reject with SFI growing above null should be rescued. "
            f"Got: {dict(report['tier'].value_counts())}"
        )
        assert (report["trend_rescue"].notna()).all()


# ─── Test 9: Feature in vetoed cluster → REJECTED ────────────────────────────

class TestBestInWorthlessCluster:

    def test_strong_feature_in_vetoed_cluster_rejected(self):
        """Feature with strong individual scores in a vetoed cluster → REJECTED."""
        n_feat, n_folds, n_trees = 3, 8, 1000
        clusters = {"cluster_0": [f"feat_{i}" for i in range(3)]}

        strong_positive = np.full((n_folds, n_feat), 0.10)
        cfi_mda_dead = np.full((n_folds, 1), -0.03)
        cfi_mdi_dead = np.random.default_rng(42).normal(-0.10, 0.01, (n_trees, 1))

        report = filter_features_v2(
            sfi_raw=pd.DataFrame(strong_positive, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(strong_positive, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(strong_positive, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(strong_positive, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=_make_tree_raw(n_feat, n_trees),
            cfi_mda_raw=pd.DataFrame(cfi_mda_dead, columns=["cluster_0"]),
            cfi_mdi_raw=pd.DataFrame(cfi_mdi_dead, columns=["cluster_0"]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_all_eligible_crosscheck(),
        )
        assert (report["tier"] == "REJECTED").all()


# ─── Test 10: MDI gates when eligible ────────────────────────────────────────

class TestMDIGatesWhenEligible:

    def test_mdi_participates_in_union(self):
        """For targets where MDI is PCA-eligible, it participates in gating."""
        n_feat, n_folds, n_trees = 5, 8, 1000
        clusters = _standard_clusters(n_feat, 2)

        all_negative = np.full((n_folds, n_feat), -0.05)
        mdi_strongly_positive = np.full((n_trees, n_feat), 0.05)
        positive_clusters = np.full((n_folds, len(clusters)), 0.05)
        positive_cfi_mdi = np.full((n_trees, len(clusters)), 0.01)

        report = filter_features_v2(
            sfi_raw=pd.DataFrame(all_negative, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(all_negative, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(all_negative, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(all_negative, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=pd.DataFrame(mdi_strongly_positive, columns=[f"feat_{i}" for i in range(n_feat)]),
            cfi_mda_raw=pd.DataFrame(positive_clusters, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            cfi_mdi_raw=pd.DataFrame(positive_cfi_mdi, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_all_eligible_crosscheck(),
        )
        assert (report["tier"] != "REJECTED").all(), "MDI (eligible) passes → conservative union saves features"


# ─── Test 11: MDI excluded when ineligible ───────────────────────────────────

class TestMDIExcludedWhenIneligible:

    def test_mdi_does_not_gate_when_ineligible(self):
        """For extra_innings-like targets where MDI is ineligible, it does nothing."""
        n_feat, n_folds, n_trees = 5, 8, 1000
        clusters = _standard_clusters(n_feat, 2)

        all_negative = np.full((n_folds, n_feat), -0.05)
        mdi_strongly_positive = np.full((n_trees, n_feat), 0.05)
        positive_clusters = np.full((n_folds, len(clusters)), 0.05)
        positive_cfi_mdi = np.full((n_trees, len(clusters)), 0.01)

        report = filter_features_v2(
            sfi_raw=pd.DataFrame(all_negative, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(all_negative, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(all_negative, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(all_negative, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=pd.DataFrame(mdi_strongly_positive, columns=[f"feat_{i}" for i in range(n_feat)]),
            cfi_mda_raw=pd.DataFrame(positive_clusters, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            cfi_mdi_raw=pd.DataFrame(positive_cfi_mdi, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_only_cfi_eligible_crosscheck(),
        )
        assert (report["tier"] == "NEEDS SPECIFICATION").all(), \
            "MDI ineligible + no per-feature methods eligible → NEEDS_SPECIFICATION"


# ─── Test 12: EB CI uses mean ────────────────────────────────────────────────

class TestEBCIUsesMean:

    def test_level_is_mean_not_median(self):
        """Point estimate must be mean(all folds), coherent with SE."""
        vals = np.array([0.20, 0.17, 0.14, 0.11, 0.08, 0.05, 0.02, -0.01])
        result = feature_score(vals, "desub_mda", null=0.0,
                               d0=EB_PRIORS["desub_mda"]["d0"],
                               s0_sq=EB_PRIORS["desub_mda"]["s0_sq"])
        assert result["level"] == pytest.approx(np.mean(vals))
        assert result["level"] != pytest.approx(np.median(vals[-3:]))


# ─── Test 13: Instability detection ──────────────────────────────────────────

class TestInstabilityDetection:

    def test_5x_mad_triggers_instability(self):
        """MAD ratio > 5x → INSTABILITY flag + NEEDS_SPECIFICATION."""
        vals = np.array([0.05, 0.05, 0.05, 0.05, 0.50, -0.40, 0.45, -0.35])
        result = feature_score(vals, "desub_mda", null=0.0,
                               d0=EB_PRIORS["desub_mda"]["d0"],
                               s0_sq=EB_PRIORS["desub_mda"]["s0_sq"])
        assert result["flag"] == "INSTABILITY"
        assert result["decision"] == "NEEDS_SPECIFICATION"

    def test_4x_mad_no_longer_triggers(self):
        """4x MAD ratio does not trigger — must exceed 5x now."""
        vals = np.array([0.05, 0.04, 0.06, 0.05, 0.075, 0.025, 0.065, 0.035])
        result = feature_score(vals, "desub_mda", null=0.0,
                               d0=EB_PRIORS["desub_mda"]["d0"],
                               s0_sq=EB_PRIORS["desub_mda"]["s0_sq"])
        assert result["flag"] != "INSTABILITY"


# ─── Test 14: MDI-only importance is noise ───────────────────────────────────

class TestMDIOnlyIsNoise:

    def test_mdi_only_feature_rejected(self):
        """Feature passing MDI but failing all other eligible methods → REJECTED.

        MDI alone cannot save a feature — tree-based splitting importance
        is prone to finding spurious structure that permutation/ablation
        methods correctly reject.
        """
        n_feat, n_folds, n_trees = 5, 8, 1000
        clusters = _standard_clusters(n_feat, 2)

        rng = np.random.default_rng(77)
        # Other methods: clearly below their respective nulls
        # SFI null=log(0.5)≈-0.693, DESUB/PCA/RESID null=0
        negative_folds = rng.normal(-1.5, 0.01, (n_folds, n_feat))
        # MDI: clearly above null (1/n_feat=0.2) — the tree finds "importance"
        mdi_high = rng.normal(0.5, 0.01, (n_trees, n_feat))

        positive_clusters = np.full((n_folds, len(clusters)), 0.05)
        positive_cfi_mdi = np.full((n_trees, len(clusters)), 0.01)

        report = filter_features_v2(
            sfi_raw=pd.DataFrame(negative_folds, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(negative_folds, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(negative_folds, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(negative_folds, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=pd.DataFrame(mdi_high, columns=[f"feat_{i}" for i in range(n_feat)]),
            cfi_mda_raw=pd.DataFrame(positive_clusters, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            cfi_mdi_raw=pd.DataFrame(positive_cfi_mdi, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_all_eligible_crosscheck(),
        )
        assert (report["tier"] == "REJECTED").all(), (
            f"MDI-only features must be rejected. Got: {dict(report['tier'].value_counts())}"
        )

    def test_mdi_only_with_some_ineligible_still_rejected(self):
        """MDI-only rule fires even when some non-MDI methods are ineligible.

        If only PCA_MDA + MDI are eligible, and PCA_MDA rejects while MDI
        accepts, the feature should still be REJECTED — MDI alone cannot save.
        """
        n_feat, n_folds, n_trees = 5, 8, 1000
        clusters = _standard_clusters(n_feat, 2)

        rng = np.random.default_rng(88)
        negative_folds = rng.normal(-0.5, 0.01, (n_folds, n_feat))
        mdi_high = rng.normal(0.5, 0.01, (n_trees, n_feat))

        # Only PCA_MDA and MDI eligible
        crosscheck = {
            "SFI": {"tau": 0.1, "p_value": 0.3},
            "DESUB_MDA": {"tau": -0.1, "p_value": 0.5},
            "PCA_MDA": {"tau": 0.6, "p_value": 0.0},
            "RESID_MDA": {"tau": 0.05, "p_value": 0.6},
            "MDI": {"tau": 0.4, "p_value": 0.005},
            "CFI_MDI": {"tau": 0.7, "p_value": 0.0},
            "CFI_MDA": {"tau": 0.5, "p_value": 0.0},
        }

        positive_clusters = np.full((n_folds, len(clusters)), 0.05)
        positive_cfi_mdi = np.full((n_trees, len(clusters)), 0.01)

        report = filter_features_v2(
            sfi_raw=pd.DataFrame(negative_folds, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(negative_folds, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(negative_folds, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(negative_folds, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=pd.DataFrame(mdi_high, columns=[f"feat_{i}" for i in range(n_feat)]),
            cfi_mda_raw=pd.DataFrame(positive_clusters, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            cfi_mdi_raw=pd.DataFrame(positive_cfi_mdi, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=crosscheck,
        )
        assert (report["tier"] == "REJECTED").all(), (
            f"MDI-only features must be rejected even with some methods ineligible. "
            f"Got: {dict(report['tier'].value_counts())}"
        )

    def test_mdi_plus_one_other_passes_not_rejected(self):
        """If MDI AND at least one non-MDI method both pass, feature survives.

        The MDI-only rule only fires when MDI is the SOLE supporter.
        """
        n_feat, n_folds, n_trees = 5, 8, 1000
        clusters = _standard_clusters(n_feat, 2)

        rng = np.random.default_rng(66)
        # PCA_MDA positive, DESUB/SFI/RESID negative
        pca_positive = rng.normal(0.1, 0.01, (n_folds, n_feat))
        others_negative = rng.normal(-0.5, 0.01, (n_folds, n_feat))
        mdi_high = rng.normal(0.5, 0.01, (n_trees, n_feat))

        positive_clusters = np.full((n_folds, len(clusters)), 0.05)
        positive_cfi_mdi = np.full((n_trees, len(clusters)), 0.01)

        report = filter_features_v2(
            sfi_raw=pd.DataFrame(others_negative, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(others_negative, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(pca_positive, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(others_negative, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=pd.DataFrame(mdi_high, columns=[f"feat_{i}" for i in range(n_feat)]),
            cfi_mda_raw=pd.DataFrame(positive_clusters, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            cfi_mdi_raw=pd.DataFrame(positive_cfi_mdi, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=_all_eligible_crosscheck(),
        )
        # Should NOT be rejected — PCA_MDA + MDI both pass
        assert (report["tier"] != "REJECTED").all(), (
            f"Feature with MDI + another method passing should NOT be rejected. "
            f"Got: {dict(report['tier'].value_counts())}"
        )


# ─── Test 14: Ineligible method excluded from ranking ─────────────────────────

class TestIneligibleExcludedFromRank:

    def test_ineligible_method_no_rank(self):
        """Ineligible method contributes nothing to composite_rank."""
        n_feat, n_folds, n_trees = 5, 8, 1000
        clusters = _standard_clusters(n_feat, 2)

        crosscheck = _all_eligible_crosscheck()
        crosscheck["SFI"] = {"tau": -0.1, "p_value": 0.3}

        positive = np.full((n_folds, n_feat), 0.05)
        positive_clusters = np.full((n_folds, len(clusters)), 0.05)
        positive_cfi_mdi = np.full((n_trees, len(clusters)), 0.01)

        report = filter_features_v2(
            sfi_raw=pd.DataFrame(positive, columns=[f"feat_{i}" for i in range(n_feat)]),
            desub_mda_raw=pd.DataFrame(positive, columns=[f"feat_{i}" for i in range(n_feat)]),
            pca_mda_raw=pd.DataFrame(positive, columns=[f"feat_{i}" for i in range(n_feat)]),
            resid_mda_raw=pd.DataFrame(positive, columns=[f"feat_{i}" for i in range(n_feat)]),
            mdi_raw=_make_tree_raw(n_feat, n_trees),
            cfi_mda_raw=pd.DataFrame(positive_clusters, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            cfi_mdi_raw=pd.DataFrame(positive_cfi_mdi, columns=[f"cluster_{i}" for i in range(len(clusters))]),
            clusters=clusters,
            sfi_null=np.log(0.5),
            pca_crosscheck=crosscheck,
        )
        assert report["sfi_passes"].isna().all()
        if "sfi_rank" in report.columns:
            assert report["sfi_rank"].isna().all()
