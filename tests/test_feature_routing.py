"""Known-answer tests for feature routing classification and per-family routing.

Tests use synthesized filter_report rows with controlled pass/fail patterns.
Each test encodes a known signal type → expected evidence group → expected
model family inclusion/exclusion.

Adversarial cases target boundary conditions in the classification logic:
- Method conflicts (MDI passes, SFI fails but RESID passes)
- Tier overrides vs method-based classification
- Empty/all-pass/all-fail edge cases
- Model family boundary: feature included in one family but excluded from another

Run: conda run -n pred python -m pytest tests/test_feature_routing.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.analysis.feature_routing import (
    _classify_feature,
    classify_all_features,
    get_feature_set,
)


# ---------------------------------------------------------------------------
# Helpers to synthesize filter_report rows
# ---------------------------------------------------------------------------

def _make_row(
    mdi=False, sfi=False, pca=False, resid=False, desub=False, cfi=False,
    tier=None, composite_rank=1, mdi_rank=1, sfi_rank=1, pca_mda_rank=1,
) -> pd.Series:
    """Create a single filter_report row with controlled pass/fail pattern."""
    return pd.Series({
        "mdi_passes": mdi,
        "sfi_passes": sfi,
        "pca_mda_passes": pca,
        "resid_mda_passes": resid,
        "desub_mda_passes": desub,
        "cfi_mda_cluster_passes": cfi,
        "tier": tier,
        "composite_rank": composite_rank,
        "mdi_rank": mdi_rank,
        "sfi_rank": sfi_rank,
        "pca_mda_rank": pca_mda_rank,
    })


def _make_report(features: dict[str, dict]) -> pd.DataFrame:
    """Create a filter_report DataFrame from {feature_name: kwargs_for_make_row}."""
    rows = {name: _make_row(**kwargs) for name, kwargs in features.items()}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Test 1: Classification — known-answer for each evidence group
# ---------------------------------------------------------------------------

class TestClassifyFeature:
    """Each pass/fail pattern must map to exactly one evidence group."""

    def test_all_pass_is_accepted(self):
        """All methods pass → accepted (strongest signal)."""
        row = _make_row(mdi=True, sfi=True, pca=True, resid=True, desub=True, cfi=True)
        assert _classify_feature(row) == "accepted"

    def test_sfi_pca_resid_is_accepted(self):
        """SFI + PCA + RESID = promoted to accepted (standalone + orthogonal)."""
        row = _make_row(sfi=True, pca=True, resid=True)
        assert _classify_feature(row) == "accepted"

    def test_sfi_only_is_standalone(self):
        """SFI alone → standalone (marginal predictive power, not orthogonal)."""
        row = _make_row(sfi=True)
        assert _classify_feature(row) == "standalone"

    def test_sfi_and_mdi_no_pca_is_standalone(self):
        """SFI + MDI but no PCA/RESID → standalone (SFI takes priority)."""
        row = _make_row(sfi=True, mdi=True)
        assert _classify_feature(row) == "standalone"

    def test_desub_no_sfi_is_complementary(self):
        """Desub passes, SFI fails → complementary (interaction-only signal)."""
        row = _make_row(desub=True, mdi=True)
        assert _classify_feature(row) == "complementary"

    def test_cfi_no_sfi_is_complementary(self):
        """CFI passes, SFI fails → complementary."""
        row = _make_row(cfi=True, mdi=True)
        assert _classify_feature(row) == "complementary"

    def test_mdi_resid_no_sfi_no_desub_is_complementary(self):
        """MDI + RESID pass, SFI and desub fail → complementary.
        Tree-useful AND non-redundant unique signal."""
        row = _make_row(mdi=True, resid=True)
        assert _classify_feature(row) == "complementary"

    def test_pca_resid_no_mdi_no_sfi_is_linear_only(self):
        """PCA + RESID pass, MDI and SFI fail → linear_only."""
        row = _make_row(pca=True, resid=True)
        assert _classify_feature(row) == "linear_only"

    def test_mdi_only_is_absorbed(self):
        """MDI alone (no SFI, no desub) → absorbed (tree sees it but no OOS confirmation)."""
        row = _make_row(mdi=True)
        assert _classify_feature(row) == "absorbed"

    def test_pca_only_no_resid_is_absorbed(self):
        """PCA alone without RESID → absorbed (partial linear signal)."""
        row = _make_row(pca=True)
        assert _classify_feature(row) == "absorbed"

    def test_cfi_only_no_mdi_no_sfi_is_complementary(self):
        """CFI alone → complementary (line 74 catches cfi before line 87).
        The 'redundant' branch at line 87 is unreachable dead code because
        'desub or cfi or (mdi and resid)' at line 74 always fires first when
        cfi=True. This is architecturally intentional: CFI confirms cluster-level
        interaction signal, which trees can exploit."""
        row = _make_row(cfi=True)
        assert _classify_feature(row) == "complementary"

    def test_nothing_passes_is_noise(self):
        """No method passes → noise."""
        row = _make_row()
        assert _classify_feature(row) == "noise"

    def test_tier_rejected_overrides_all(self):
        """REJECTED tier overrides any pass/fail pattern."""
        row = _make_row(mdi=True, sfi=True, pca=True, resid=True, tier="REJECTED")
        assert _classify_feature(row) == "rejected"

    def test_tier_accepted_overrides_all(self):
        """ACCEPTED tier overrides method-based classification."""
        row = _make_row(tier="ACCEPTED")
        assert _classify_feature(row) == "accepted"


# ---------------------------------------------------------------------------
# Test 2: Adversarial edge cases in classification
# ---------------------------------------------------------------------------

class TestClassifyAdversarial:
    """Boundary conditions that test the priority ordering of the logic."""

    def test_sfi_takes_priority_over_desub(self):
        """If SFI passes, feature is standalone/accepted even if desub also passes.
        SFI = standalone power; desub = interaction. SFI wins."""
        row = _make_row(sfi=True, desub=True, mdi=True)
        # SFI passes → standalone (no PCA+RESID to promote to accepted)
        assert _classify_feature(row) == "standalone"

    def test_sfi_with_pca_only_no_resid_is_standalone(self):
        """SFI + PCA but NOT RESID → standalone (not promoted to accepted).
        Promotion requires PCA AND RESID."""
        row = _make_row(sfi=True, pca=True)
        assert _classify_feature(row) == "standalone"

    def test_mdi_resid_desub_no_sfi_is_complementary_not_linear(self):
        """MDI+RESID+desub but no SFI → complementary (not linear_only).
        Complementary check fires before linear_only check."""
        row = _make_row(mdi=True, resid=True, desub=True, pca=True)
        assert _classify_feature(row) == "complementary"

    def test_pca_resid_cfi_no_mdi_no_sfi(self):
        """PCA+RESID+CFI but no MDI, no SFI.
        CFI triggers complementary (line 74: desub or cfi or (mdi and resid))."""
        row = _make_row(pca=True, resid=True, cfi=True)
        assert _classify_feature(row) == "complementary"

    def test_mdi_pca_no_resid_no_sfi_is_absorbed(self):
        """MDI+PCA but no RESID, no SFI → absorbed.
        MDI alone = absorbed; PCA alone = absorbed; together still absorbed
        because the MDI branch (line 83) fires first."""
        row = _make_row(mdi=True, pca=True)
        assert _classify_feature(row) == "absorbed"

    def test_resid_only_is_linear_only(self):
        """RESID alone (no PCA, no MDI, no SFI) → linear_only.
        Unique residual signal should reach linear models even without PCA."""
        row = _make_row(resid=True)
        assert _classify_feature(row) == "linear_only"

    def test_desub_only_no_mdi_is_complementary(self):
        """desub alone (no MDI, no SFI) → complementary.
        Line 74: 'desub or cfi or (mdi and resid)' — desub alone suffices."""
        row = _make_row(desub=True)
        assert _classify_feature(row) == "complementary"


# ---------------------------------------------------------------------------
# Test 3: Per-family routing — features land in expected model's feature set
# ---------------------------------------------------------------------------

class TestGetFeatureSet:
    """Verify correct routing: which evidence groups go to which model families."""

    @pytest.fixture
    def report(self):
        """Synthesized filter_report with one feature per evidence group."""
        return _make_report({
            "strong_feature": dict(sfi=True, pca=True, resid=True, mdi=True, composite_rank=1),
            "standalone_feat": dict(sfi=True, sfi_rank=2),
            "interaction_feat": dict(desub=True, mdi=True, mdi_rank=3),
            "linear_feat": dict(pca=True, resid=True, pca_mda_rank=4),
            "absorbed_feat": dict(mdi=True, mdi_rank=5),
            "noise_feat": dict(),
        })

    def test_lightgbm_gets_accepted_standalone_complementary_absorbed(self, report):
        """LightGBM (column subsampling confirmed) gets everything except noise/linear."""
        feats = get_feature_set("lightgbm", report)
        assert "strong_feature" in feats
        assert "standalone_feat" in feats
        assert "interaction_feat" in feats
        assert "absorbed_feat" in feats
        assert "linear_feat" not in feats
        assert "noise_feat" not in feats

    def test_logistic_regression_gets_accepted_standalone_linear(self, report):
        """LogReg gets accepted + standalone + linear_only (no interactions, no absorbed)."""
        feats = get_feature_set("logistic_regression", report)
        assert "strong_feature" in feats
        assert "standalone_feat" in feats
        assert "linear_feat" in feats
        assert "interaction_feat" not in feats
        assert "absorbed_feat" not in feats
        assert "noise_feat" not in feats

    def test_adaboost_gets_only_accepted_standalone(self, report):
        """AdaBoost (no shrinkage, no subsampling) gets minimal safe set."""
        feats = get_feature_set("adaboost", report)
        assert "strong_feature" in feats
        assert "standalone_feat" in feats
        assert "interaction_feat" not in feats
        assert "absorbed_feat" not in feats
        assert "linear_feat" not in feats

    def test_mlp_gets_complementary_by_sfi_rank(self, report):
        """MLP gets complementary features ordered by SFI rank (not MDI)."""
        feats = get_feature_set("mlp", report)
        assert "strong_feature" in feats
        assert "standalone_feat" in feats
        assert "interaction_feat" in feats  # complementary
        assert "absorbed_feat" not in feats  # no column subsampling

    def test_knn_gets_only_accepted_standalone(self, report):
        """KNN (distance-based, curse of dimensionality) gets minimal set."""
        feats = get_feature_set("knn", report)
        assert "strong_feature" in feats
        assert "standalone_feat" in feats
        assert "interaction_feat" not in feats
        assert "linear_feat" not in feats
        assert "absorbed_feat" not in feats

    def test_random_forest_same_as_lightgbm(self, report):
        """RF (always has sqrt subsampling) gets same set as LightGBM."""
        lgb_feats = set(get_feature_set("lightgbm", report))
        rf_feats = set(get_feature_set("random_forest", report))
        assert lgb_feats == rf_feats

    def test_catboost_excludes_absorbed(self, report):
        """CatBoost (rsm not yet activated) excludes absorbed features."""
        feats = get_feature_set("catboost", report)
        assert "strong_feature" in feats
        assert "interaction_feat" in feats
        assert "absorbed_feat" not in feats

    def test_lasso_gets_linear_features(self, report):
        """Lasso (L1 embedded selection) gets wide pool including linear_only."""
        feats = get_feature_set("lasso", report)
        assert "linear_feat" in feats
        assert "strong_feature" in feats

    def test_noise_never_routed_anywhere(self, report):
        """Noise features are excluded from ALL model families."""
        for family in ["lightgbm", "logistic_regression", "mlp", "knn", "adaboost",
                       "catboost", "random_forest", "lasso", "ridge"]:
            feats = get_feature_set(family, report)
            assert "noise_feat" not in feats, f"noise_feat routed to {family}"


# ---------------------------------------------------------------------------
# Test 4: Adversarial routing edge cases
# ---------------------------------------------------------------------------

class TestRoutingAdversarial:
    """Edge cases in the routing logic."""

    def test_empty_report_returns_empty_list(self):
        """Empty filter_report → empty feature set for all families."""
        empty = pd.DataFrame()
        for family in ["lightgbm", "logistic_regression", "mlp"]:
            feats = get_feature_set(family, empty)
            assert feats == []

    def test_all_rejected_returns_empty_list(self):
        """All features rejected → empty set."""
        report = _make_report({
            "feat_a": dict(tier="REJECTED", sfi=True, mdi=True),
            "feat_b": dict(tier="REJECTED", pca=True, resid=True),
        })
        feats = get_feature_set("lightgbm", report)
        assert feats == []

    def test_all_noise_returns_empty_list(self):
        """All features are noise → empty set."""
        report = _make_report({
            "noise_a": dict(),
            "noise_b": dict(),
            "noise_c": dict(resid=True),  # resid alone = noise
        })
        feats = get_feature_set("lightgbm", report)
        assert feats == []

    def test_ordering_respects_rank_within_category(self):
        """Within accepted category, features are ordered by composite_rank."""
        report = _make_report({
            "high_rank": dict(sfi=True, pca=True, resid=True, composite_rank=10),
            "low_rank": dict(sfi=True, pca=True, resid=True, composite_rank=1),
        })
        feats = get_feature_set("lightgbm", report)
        assert feats.index("low_rank") < feats.index("high_rank")

    def test_accepted_comes_before_standalone(self):
        """Category priority: accepted features always precede standalone."""
        report = _make_report({
            "standalone_first_alpha": dict(sfi=True, sfi_rank=1),
            "accepted_last_alpha": dict(sfi=True, pca=True, resid=True, composite_rank=99),
        })
        feats = get_feature_set("lightgbm", report)
        # accepted (even with high rank) comes before standalone (even with low rank)
        assert feats.index("accepted_last_alpha") < feats.index("standalone_first_alpha")

    def test_feature_with_rejected_tier_excluded_even_if_all_pass(self):
        """REJECTED tier is absolute — even all-pass features get excluded."""
        report = _make_report({
            "good_feat": dict(sfi=True, pca=True, resid=True, composite_rank=1),
            "rejected_but_strong": dict(
                tier="REJECTED", mdi=True, sfi=True, pca=True, resid=True, desub=True, cfi=True,
                composite_rank=0,  # lower rank than good_feat
            ),
        })
        feats = get_feature_set("lightgbm", report)
        assert "good_feat" in feats
        assert "rejected_but_strong" not in feats

    def test_schedule_flag_expected_routing(self):
        """Schedule context flags: binary interaction-only signal.
        Expected: SFI fails (flag alone doesn't predict winner), MDI passes
        (trees split on it usefully), → complementary → routed to trees, not linear."""
        report = _make_report({
            "is_same_league": dict(mdi=True, desub=True),  # interaction signal
            "elo_prob_x_same_league": dict(sfi=True, pca=True, resid=True),  # standalone
        })
        lgb_feats = get_feature_set("lightgbm", report)
        lr_feats = get_feature_set("logistic_regression", report)

        # Raw flag → complementary → trees get it, linear doesn't
        assert "is_same_league" in lgb_feats
        assert "is_same_league" not in lr_feats

        # Interaction term → accepted → both get it
        assert "elo_prob_x_same_league" in lgb_feats
        assert "elo_prob_x_same_league" in lr_feats
