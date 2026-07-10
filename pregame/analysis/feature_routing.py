"""Evidence-based feature routing to model families.

Routes features based on which de Prado importance tests they pass,
matched to each model's architectural capacity to exploit them.

Routing logic derives from two principles:
1. A feature's pass/fail pattern across tests reveals its signal TYPE
   (standalone, interaction-only, linear-orthogonal, redundant)
2. Each model architecture can exploit specific signal types based on
   its mathematical mechanism (splitting, gradient descent, distance, etc.)

Key architectural distinctions:
- Column subsampling (RF/ET always, LGB/XGB via config) makes redundant
  features safe via stochastic decorrelation across estimators
- AdaBoost lacks shrinkage AND subsampling — interaction-only features
  cause it to overfit via residual reweighting on noise
- MDI rank is a tree-split statistic — irrelevant for gradient-based MLP
- Lasso/ElasticNet do embedded L1 selection — don't pre-filter aggressively
- GaussianNB assumes conditional independence — orthogonal features ideal,
  dependency-implying features (complementary/redundant) are pathological
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Per-feature classification based on pass/fail patterns
# ─────────────────────────────────────────────────────────────────────────────

def _classify_feature(row: pd.Series) -> str:
    """Classify a single feature into an evidence group.

    Uses the pass/fail pattern across all importance methods to determine
    what kind of signal the feature carries:

      accepted     — all methods agree OR SFI+PCA+RESID: proven predictive signal
      complementary — interaction signal (desub/CFI/MDI pass, SFI fails)
      standalone   — SFI passes: standalone predictive power
      linear_only  — PCA+RESID pass, MDI+SFI fail: linear-space signal only
      absorbed     — partial signal, redundant with better features
      redundant    — only cluster-level CFI-MDA passes
      noise        — no method passes
      rejected     — explicitly flagged REJECTED by filter
    """
    if row.get("tier") == "REJECTED":
        return "rejected"
    if row.get("tier") == "ACCEPTED":
        return "accepted"

    mdi = bool(row.get("mdi_passes") is True)
    sfi = bool(row.get("sfi_passes") is True)
    pca = bool(row.get("pca_mda_passes") is True)
    resid = bool(row.get("resid_mda_passes") is True)
    desub = bool(row.get("desub_mda_passes") is True)
    cfi = bool(row.get("cfi_mda_cluster_passes") is True)

    # Standalone signal: SFI passes (feature has predictive power alone)
    if sfi:
        if pca and resid:
            return "accepted"  # promote: standalone + orthogonal = strong
        return "standalone"

    # Tree-exploitable, non-standalone signal. Three paths:
    # 1. desub/CFI pass: specific interaction tests confirm joint effects
    # 2. MDI + RESID pass: tree-useful AND non-redundant (unique signal that
    #    trees can exploit even though desub/CFI didn't fire — those tests
    #    are not omniscient interaction detectors)
    if desub or cfi or (mdi and resid):
        return "complementary"

    # Linear-only signal: PCA+RESID pass but tree methods fail
    if pca and resid and not mdi:
        return "linear_only"

    # MDI passes but no OOS interaction confirmation (desub/CFI both fail).
    # Tree splits see the feature but standalone tests don't confirm it.
    if mdi and not sfi and not desub:
        return "absorbed"

    # Cluster-level only
    if cfi and not mdi and not sfi:
        return "redundant"

    # PCA passes alone (partial linear signal)
    if pca and not resid:
        return "absorbed"

    return "noise"


def classify_all_features(filter_report: pd.DataFrame) -> pd.Series:
    """Classify all features in a filter_report into evidence groups.

    Returns Series(index=feature, values=group_name).
    """
    return filter_report.apply(_classify_feature, axis=1).rename("evidence_group")


# ─────────────────────────────────────────────────────────────────────────────
#  Within-category ordering logic
# ─────────────────────────────────────────────────────────────────────────────

def _order_by_rank(features: list[str], report: pd.DataFrame, rank_col: str) -> list[str]:
    """Order features by a specific rank column (lower = better)."""
    if not features or rank_col not in report.columns:
        return features
    subset = report.loc[[f for f in features if f in report.index], rank_col].dropna()
    ordered = subset.sort_values().index.tolist()
    # Append any features missing from the rank column at the end
    remaining = [f for f in features if f not in ordered]
    return ordered + remaining


# ─────────────────────────────────────────────────────────────────────────────
#  Per-family feature set routing with hierarchical ordering
# ─────────────────────────────────────────────────────────────────────────────

# Models with CONFIRMED column subsampling — safe for absorbed/redundant.
# RF/ET: max_features='sqrt' hardcoded (always active).
# LGB/XGB: colsample_bytree in [0.4, 1.0] Optuna range, default 0.8.
#   NOTE: actual trial values not yet verified — search space includes 1.0.
#   Routing absorbed here is architecturally coherent but remains a hypothesis
#   pending ablation (with vs without absorbed on holdout).
COLUMN_SUBSAMPLE_CONFIRMED = {"random_forest", "extra_trees", "lightgbm", "xgboost"}

# CatBoost (rsm) and HistGB (max_features) are NOT activated yet.
# Decision: defer adding rsm/max_features to Optuna until the ablation on
# RF/ET/LGB/XGB confirms absorbed features are net-positive. Adding new
# Optuna params reopens the trial space and invalidates comparisons.
COLUMN_SUBSAMPLE_DEFERRED = {"catboost", "hist_gradient_boosting"}


def get_feature_set(family: str, filter_report: pd.DataFrame) -> list[str]:
    """Return hierarchically-ordered feature list for this model family.

    Features are layered by category priority, and within each category
    ordered by the method that defines that category's signal strength:

      accepted    → composite_rank (all methods pass, average is meaningful)
      standalone  → SFI rank (defining test for marginal power)
      complementary → MDI rank for trees, SFI rank for MLP
      linear_only → PCA rank (orthogonality is the relevant dimension)
      absorbed    → MDI rank (only for models with column subsampling)

    The returned list encodes this priority: features at the front are
    higher-priority. When S* caps are applied, they cut from the tail.
    """
    groups = classify_all_features(filter_report)

    accepted = filter_report.index[groups == "accepted"].tolist()
    complementary = filter_report.index[groups == "complementary"].tolist()
    standalone = filter_report.index[groups == "standalone"].tolist()
    linear_only = filter_report.index[groups == "linear_only"].tolist()
    absorbed = filter_report.index[groups.isin(["absorbed", "redundant"])].tolist()

    # Order within each category by the relevant method rank
    accepted_ordered = _order_by_rank(accepted, filter_report, "composite_rank")
    standalone_ordered = _order_by_rank(standalone, filter_report, "sfi_rank")
    complementary_by_mdi = _order_by_rank(complementary, filter_report, "mdi_rank")
    complementary_by_sfi = _order_by_rank(complementary, filter_report, "sfi_rank")
    linear_only_ordered = _order_by_rank(linear_only, filter_report, "pca_mda_rank")
    absorbed_ordered = _order_by_rank(absorbed, filter_report, "mdi_rank")

    # ── Tree ensembles WITH confirmed column subsampling ──────────────────────
    # RF/ET: hardcoded sqrt. LGB/XGB: Optuna [0.4, 1.0], default 0.8.
    # Can exploit: interactions (tree splits), redundancy (subsampling decorrelates).
    if family in COLUMN_SUBSAMPLE_CONFIRMED:
        return (accepted_ordered + standalone_ordered
                + complementary_by_mdi + absorbed_ordered)

    # ── Tree ensembles WITHOUT confirmed column subsampling ────────────────────
    # CatBoost (rsm not yet activated), HistGB (max_features not yet activated).
    # Can exploit interactions but redundancy risks substitution effects.
    if family in COLUMN_SUBSAMPLE_DEFERRED:
        return accepted_ordered + standalone_ordered + complementary_by_mdi

    # ── AdaBoost ───────────────────────────────────────────────────────────
    # No shrinkage, no subsampling, sequential reweighting on residuals.
    # Interaction-only features (complementary) are noise sources for stumps.
    # Absorbed features amplify reweighting instability.
    if family == "adaboost":
        return accepted_ordered + standalone_ordered

    # ── MLP ────────────────────────────────────────────────────────────────
    # Dense layers learn interactions via hidden units — complementary is valid.
    # BUT ordering by MDI is wrong (tree-split statistic, irrelevant for MLP).
    # Use SFI (model-agnostic permutation importance) to order complementary.
    # No column subsampling mechanism → redundant features create correlated
    # gradients without adding information.
    if family == "mlp":
        return accepted_ordered + standalone_ordered + complementary_by_sfi

    # ── Lasso / ElasticNet ─────────────────────────────────────────────────
    # L1 component IS the selection mechanism. Don't pre-filter aggressively.
    # Give wider pool: accepted + standalone + linear_only (all of it).
    if family in ("lasso", "elasticnet"):
        return accepted_ordered + standalone_ordered + linear_only_ordered

    # ── Linear models without embedded selection ───────────────────────────
    # Ridge, LogReg, SGD, BaggingLogReg: linear boundary, no interaction capacity.
    # Pre-filtering by PCA rank is appropriate (they can't select on their own).
    if family in ("logistic_regression", "ridge", "sgd", "bagging_logreg"):
        return accepted_ordered + standalone_ordered + linear_only_ordered

    # ── LDA ────────────────────────────────────────────────────────────────
    # Shared covariance matrix across classes, linear boundary.
    # Collinear features → singular Σ → LDA crashes.
    # linear_only (PCA+RESID pass = orthogonal) is safe for covariance estimation.
    if family == "lda":
        return accepted_ordered + standalone_ordered + linear_only_ordered

    # ── QDA ────────────────────────────────────────────────────────────────
    # Per-class covariance matrices — even more sensitive to dimensionality.
    # More features = more parameters per class = singularity risk.
    # Conservative: accepted + standalone only (fewer, proven features).
    if family == "qda":
        return accepted_ordered + standalone_ordered

    # ── GaussianNB ─────────────────────────────────────────────────────────
    # Conditional independence assumption.
    # linear_only (PCA+RESID = orthogonal) ≈ conditional independence = ideal.
    # complementary/absorbed imply dependency → double-counting under NB.
    if family == "gaussian_nb":
        return accepted_ordered + standalone_ordered + linear_only_ordered

    # ── KNN ────────────────────────────────────────────────────────────────
    # Distance-based, curse of dimensionality.
    # Every feature must carry standalone signal strong enough to improve
    # neighbor quality despite the added dimension.
    if family == "knn":
        return accepted_ordered + standalone_ordered

    # Unknown family → conservative (accepted + standalone + complementary)
    log.warning(f"Unknown family {family!r} — using default routing")
    return accepted_ordered + standalone_ordered + complementary_by_mdi


# ─────────────────────────────────────────────────────────────────────────────
#  Full routing orchestration
# ─────────────────────────────────────────────────────────────────────────────

def route_features(filter_report: pd.DataFrame) -> dict:
    """Classify all features and return grouped feature lists.

    Returns dict with:
      'groups': {group_name: [feature_names]}
      'per_family': {family_name: [feature_names]}
      'summary': {group_name: count}
    """
    groups = classify_all_features(filter_report)
    group_dict = {}
    for group_name in groups.unique():
        group_dict[group_name] = sorted(
            filter_report.index[groups == group_name].tolist()
        )

    all_families = [
        "lightgbm", "xgboost", "catboost", "random_forest", "extra_trees",
        "hist_gradient_boosting", "adaboost", "mlp",
        "logistic_regression", "ridge", "lasso", "elasticnet", "sgd",
        "bagging_logreg", "knn", "lda", "qda", "gaussian_nb",
    ]
    per_family = {
        family: get_feature_set(family, filter_report)
        for family in all_families
    }

    summary = {group: len(members) for group, members in group_dict.items()}
    log.info(f"Feature routing summary: {summary}")
    for family, feats in sorted(per_family.items(), key=lambda x: -len(x[1])):
        log.info(f"  {family}: {len(feats)} features")

    return {
        "groups": group_dict,
        "per_family": per_family,
        "summary": summary,
    }


def write_routing_report(output_dir: Path, filter_report: pd.DataFrame) -> Path:
    """Run routing and write routing_report.json to output_dir.

    Returns path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    routing = route_features(filter_report)

    report_path = output_dir / "routing_report.json"
    with open(report_path, "w") as f:
        json.dump(routing, f, indent=2)

    log.info(f"Routing report written to {report_path}")
    return report_path
