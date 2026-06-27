"""Evidence-based feature routing to model families.

Routes features based on which de Prado importance tests they pass,
not a naive top-pct threshold. Each model family gets the features
whose evidence profile matches its assumptions:

  GBDT (lgbm/xgb/catboost) — handles interactions, redundancy is free
  RF_ET (random_forest/extra_trees) — same as GBDT
  WEAK_TREE (adaboost/hist_gb/bagging_logreg) — fragile to noise/redundancy
  LINEAR (logreg/ridge/lasso/elasticnet/sgd) — needs orthogonal signal
  FRAGILE (knn/lda/qda/gaussian_nb) — curse of dimensionality
  NEURAL (mlp) — moderate robustness, needs some signal evidence
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Model family groupings
# ─────────────────────────────────────────────────────────────────────────────

GBDT = {"lightgbm", "xgboost", "catboost"}
RF_ET = {"random_forest", "extra_trees"}
WEAK_TREE = {"adaboost", "hist_gradient_boosting", "bagging_logreg"}
LINEAR = {"logistic_regression", "ridge", "lasso", "elasticnet", "sgd"}
FRAGILE = {"knn", "lda", "qda", "gaussian_nb"}
NEURAL = {"mlp"}


# ─────────────────────────────────────────────────────────────────────────────
#  Per-feature classification based on pass/fail patterns
# ─────────────────────────────────────────────────────────────────────────────

def _classify_feature(row: pd.Series) -> str:
    """Classify a single feature into an evidence group.

    Uses the pass/fail pattern across all importance methods to determine
    what kind of signal the feature carries:

      accepted     — all methods agree: proven predictive signal
      complementary — MDI+desub/CFI pass, SFI fails: interaction signal
      standalone   — SFI passes: standalone predictive power
      linear_only  — PCA+RESID pass, MDI+SFI fail: linear-space signal only
      absorbed     — MDI passes alone: redundant with better features
      redundant    — only cluster-level CFI-MDA passes
      noise        — no method passes
      rejected     — explicitly flagged REJECTED by filter
    """
    if row.get("tier") == "REJECTED":
        return "rejected"
    if row.get("tier") == "ACCEPTED":
        return "accepted"

    # Extract pass/fail signals (coerce NaN → False for missing methods)
    mdi = bool(row.get("mdi_passes") is True)
    sfi = bool(row.get("sfi_passes") is True)
    pca = bool(row.get("pca_mda_passes") is True)
    resid = bool(row.get("resid_mda_passes") is True)
    desub = bool(row.get("desub_mda_passes") is True)
    cfi = bool(row.get("cfi_mda_cluster_passes") is True)

    # Interaction signal: feature works in combination (desub or CFI pass)
    # but fails standalone (SFI fails)
    if (desub or cfi) and not sfi:
        return "complementary"

    # Standalone signal: SFI passes (feature has predictive power alone)
    if sfi:
        if pca and resid:
            return "accepted"  # promote: standalone + orthogonal = strong
        return "standalone"

    # Linear-only signal: PCA+RESID pass but tree methods fail
    if pca and resid and not mdi:
        return "linear_only"

    # MDI-only: tree sees it but no OOS confirmation
    if mdi and not sfi and not desub and not pca:
        return "absorbed"

    # Cluster-level only
    if cfi and not mdi and not sfi:
        return "redundant"

    # PCA passes alone (partial linear signal)
    if pca and not resid:
        return "absorbed"

    # Nothing passes
    return "noise"


def classify_all_features(filter_report: pd.DataFrame) -> pd.Series:
    """Classify all features in a filter_report into evidence groups.

    Returns Series(index=feature, values=group_name).
    """
    return filter_report.apply(_classify_feature, axis=1).rename("evidence_group")


# ─────────────────────────────────────────────────────────────────────────────
#  Per-family feature set routing
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_set(family: str, filter_report: pd.DataFrame) -> list[str]:
    """Return the feature list appropriate for this model family.

    Uses the granular pass/fail patterns from filter_report to determine
    which features each model can responsibly consume.

    Routing logic:
      GBDT/RF_ET:   accepted + complementary + standalone
                    (handles interactions, redundancy is cheap)
      WEAK_TREE:    accepted + standalone
                    (sensitive to noise, needs clean signal)
      NEURAL:       accepted + standalone
                    (moderate robustness, benefits from more features)
      LINEAR:       accepted + standalone + linear_only
                    (needs orthogonal features, PCA-MDA identifies these)
      FRAGILE:      accepted only
                    (curse of dimensionality, only proven features)
    """
    groups = classify_all_features(filter_report)

    accepted = filter_report.index[groups == "accepted"].tolist()
    complementary = filter_report.index[groups == "complementary"].tolist()
    standalone = filter_report.index[groups == "standalone"].tolist()
    linear_only = filter_report.index[groups == "linear_only"].tolist()

    if family in GBDT | RF_ET:
        return sorted(accepted + complementary + standalone)
    elif family in WEAK_TREE | NEURAL:
        return sorted(accepted + standalone)
    elif family in LINEAR:
        return sorted(accepted + standalone + linear_only)
    elif family in FRAGILE:
        return sorted(accepted)
    # Unknown family → conservative (accepted only)
    log.warning(f"Unknown family {family!r} — using accepted features only")
    return sorted(accepted)


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

    all_families = GBDT | RF_ET | WEAK_TREE | LINEAR | FRAGILE | NEURAL
    per_family = {
        family: get_feature_set(family, filter_report)
        for family in sorted(all_families)
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
