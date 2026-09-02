"""Re-run filter_features_v2 + routing from saved raw CSVs.

Reads all raw importance files for each target, runs the new gate,
writes fresh feature_report.csv and routing_report.json, then verifies
the artifacts are readable by the strategy routing logic.
"""
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.analysis.feature_importance import filter_features_v2
from classical_learning.analysis.feature_routing import route_features, write_routing_report

logging.basicConfig(level=logging.WARNING)  # suppress noisy sub-module logs
log = logging.getLogger("regate")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(handler)

IMPORTANCE_BASE = Path(__file__).resolve().parents[1] / "pregame/artifacts/importance"
# Rigorous PCA cross-check (3-method: MDI, MDA, SFI) — authoritative for regime detection
PCA_CROSSCHECK_BASE = Path(__file__).resolve().parents[1] / "data/importance"

TARGETS = [
    "home_win", "yrfi", "extra_innings", "first_5_home_win",
    "home_runs", "away_runs", "total_runs", "home_run_diff",
    "first_5_total_runs", "first_5_home_run_diff",
]


def load_raw(target_dir: Path, name: str) -> pd.DataFrame | None:
    path = target_dir / f"importance_{name}_raw.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df


def process_target(target: str) -> dict:
    target_dir = IMPORTANCE_BASE / target
    filtered_dir = target_dir / "filtered"
    filtered_dir.mkdir(exist_ok=True)

    # ── Load raw files ────────────────────────────────────────────────────
    sfi_raw = load_raw(target_dir, "sfi")
    desub_raw = load_raw(target_dir, "desub_mda")
    pca_raw = load_raw(target_dir, "pca_mda")
    resid_raw = load_raw(target_dir, "resid_mda")
    mdi_raw = load_raw(target_dir, "mdi")
    cfi_mda_raw = load_raw(target_dir, "cfi_mda")

    cfi_mdi_path = target_dir / "importance_cfi_mdi_cluster.csv"
    if not cfi_mdi_path.exists():
        cfi_mdi_path = target_dir / "importance_cfi_mdi_raw.csv"
    cfi_mdi_raw = pd.read_csv(cfi_mdi_path, index_col=0) if cfi_mdi_path.exists() else None

    # ── Load pca_crosscheck (kendall_tau.json) ────────────────────────────
    kt_path = target_dir / "kendall_tau.json"
    if not kt_path.exists():
        return {"target": target, "status": "SKIP: no kendall_tau.json"}
    with open(kt_path) as f:
        pca_crosscheck = json.load(f)

    # ── Load cluster map ──────────────────────────────────────────────────
    cm_path = target_dir / "cluster_map.json"
    if not cm_path.exists():
        return {"target": target, "status": "SKIP: no cluster_map.json"}
    with open(cm_path) as f:
        raw_clusters = json.load(f)
    # Keys may be strings; routing expects int or whatever key type
    clusters = {k: v for k, v in raw_clusters.items()}

    # ── sfi_null from importance_summary.csv ─────────────────────────────
    summary_path = target_dir / "importance_summary.csv"
    sfi_null = None
    if summary_path.exists() and sfi_raw is not None:
        summary = pd.read_csv(summary_path, index_col=0)
        for col in ("null_log_loss", "null_r2"):
            if col in summary.columns:
                sfi_null = float(summary[col].iloc[0])
                break
    if sfi_null is None:
        meta_path = target_dir / "importance_sfi_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                sfi_null = json.load(f).get("sfi_null")

    # ── Run new gate ──────────────────────────────────────────────────────
    filter_report = filter_features_v2(
        sfi_raw=sfi_raw,
        desub_mda_raw=desub_raw,
        pca_mda_raw=pca_raw,
        resid_mda_raw=resid_raw,
        mdi_raw=mdi_raw,
        cfi_mda_raw=cfi_mda_raw,
        cfi_mdi_raw=cfi_mdi_raw,
        clusters=clusters,
        sfi_null=sfi_null,
        pca_crosscheck=pca_crosscheck,
    )

    # ── Load rigorous PCA cross-check for MDA regime detection ─────────────
    rigorous_crosscheck = None
    rigorous_kt_path = PCA_CROSSCHECK_BASE / target / "kendall_tau.json"
    if rigorous_kt_path.exists():
        with open(rigorous_kt_path) as f:
            rigorous_crosscheck = json.load(f)
    mda_importance_dir = PCA_CROSSCHECK_BASE / target

    # ── Write artifacts ───────────────────────────────────────────────────
    filter_report.to_csv(filtered_dir / "feature_report.csv")
    route_result = route_features(
        filter_report,
        pca_crosscheck=rigorous_crosscheck,
        importance_dir=mda_importance_dir,
    )
    write_routing_report(
        filtered_dir, filter_report,
        pca_crosscheck=rigorous_crosscheck,
        importance_dir=mda_importance_dir,
    )

    # ── Collect stats ─────────────────────────────────────────────────────
    tiers = filter_report["tier"].value_counts().to_dict()
    groups = {k: len(v) for k, v in route_result["groups"].items()}
    pf = route_result["per_family"]
    regime = route_result.get("regime", "evidence_group")

    # Check eligibility reflected correctly: ineligible methods should be NaN
    sfi_nan_pct = filter_report["sfi_passes"].isna().mean() * 100
    sfi_true_pct = (filter_report["sfi_passes"] == True).mean() * 100

    return {
        "target": target,
        "status": "OK",
        "regime": regime,
        "tiers": tiers,
        "routing_groups": groups,
        "trees_lgb": len(pf.get("lightgbm", [])),
        "lin_lasso": len(pf.get("lasso", [])),
        "lin_ridge": len(pf.get("ridge", [])),
        "ydf": len(pf.get("ydf_oblique_gbt", [])),
        "sfi_nan_pct": round(sfi_nan_pct, 1),
        "sfi_true_pct": round(sfi_true_pct, 1),
        "all_families": {fam: len(feats) for fam, feats in sorted(pf.items())},
    }


def verify_strategy_reachability(target: str) -> dict:
    """Check that train_target can load and use the fresh artifacts."""
    filtered_dir = IMPORTANCE_BASE / target / "filtered"
    issues = []

    # feature_report.csv readable?
    fr_path = filtered_dir / "feature_report.csv"
    if not fr_path.exists():
        return {"target": target, "reachable": False, "issues": ["feature_report.csv missing"]}
    df = pd.read_csv(fr_path, index_col=0)

    # routing_report.json readable?
    rr_path = filtered_dir / "routing_report.json"
    if not rr_path.exists():
        issues.append("routing_report.json missing")
    else:
        with open(rr_path) as f:
            rr = json.load(f)
        required_keys = {"groups", "per_family", "summary"}
        missing = required_keys - set(rr.keys())
        if missing:
            issues.append(f"routing_report missing keys: {missing}")

    # All 19 families present?
    from classical_learning.analysis.feature_routing import COLUMN_SUBSAMPLE_CONFIRMED, COLUMN_SUBSAMPLE_DEFERRED
    all_fams = COLUMN_SUBSAMPLE_CONFIRMED | COLUMN_SUBSAMPLE_DEFERRED | {
        "adaboost", "mlp", "lasso", "elasticnet", "ridge", "logistic_regression",
        "sgd", "bagging_logreg", "lda", "gaussian_nb", "knn", "qda",
    }
    pf_keys = set(rr["per_family"].keys()) if rr_path.exists() else set()
    missing_fams = all_fams - pf_keys
    if missing_fams:
        issues.append(f"missing families in routing: {sorted(missing_fams)}")

    # tier column has only valid values?
    valid_tiers = {"ACCEPTED", "NEEDS SPECIFICATION", "REJECTED"}
    bad = set(df["tier"].unique()) - valid_tiers
    if bad:
        issues.append(f"unexpected tier values: {bad}")

    # No REJECTED features leaking into any model family list
    rejected = set(df[df["tier"] == "REJECTED"].index)
    if rr_path.exists():
        for fam, feats in rr["per_family"].items():
            leaked = rejected & set(feats)
            if leaked:
                issues.append(f"REJECTED features in {fam}: {leaked}")

    return {
        "target": target,
        "reachable": len(issues) == 0,
        "issues": issues,
    }


def cleanup_nested_dirs():
    """Remove duplicate nested {target}/{target}/ artifact dirs."""
    removed = []
    for target in TARGETS:
        nested = IMPORTANCE_BASE / target / target
        if nested.exists() and nested.is_dir():
            import shutil
            shutil.rmtree(nested)
            removed.append(str(nested))
    return removed


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1: Re-gate + re-route all targets")
    print("=" * 70)

    results = []
    for t in TARGETS:
        log.info(f"\n--- {t} ---")
        r = process_target(t)
        results.append(r)

        if r["status"] != "OK":
            print(f"  {t}: {r['status']}")
            continue

        tiers = r["tiers"]
        groups = r["routing_groups"]
        regime = r.get("regime", "evidence_group")
        print(f"\n{t} [{regime}]:")
        print(f"  Tiers  — ACCEPTED={tiers.get('ACCEPTED',0)}  NEEDS_SPEC={tiers.get('NEEDS SPECIFICATION',0)}  REJECTED={tiers.get('REJECTED',0)}")
        print(f"  Groups — {groups}")
        print(f"  Trees(LGB/YDF): {r['trees_lgb']}/{r['ydf']}  Linear(lasso/ridge): {r['lin_lasso']}/{r['lin_ridge']}")
        print(f"  SFI eligibility: NaN={r['sfi_nan_pct']}%  True={r['sfi_true_pct']}%")

    print("\n" + "=" * 70)
    print("STEP 2: Verify strategy reachability")
    print("=" * 70)

    all_ok = True
    for t in TARGETS:
        v = verify_strategy_reachability(t)
        status = "PASS" if v["reachable"] else "FAIL"
        print(f"  {t}: {status}", end="")
        if v["issues"]:
            print(f"  -- {v['issues']}")
            all_ok = False
        else:
            print()

    print("\n" + "=" * 70)
    print("STEP 3: Remove nested duplicate dirs")
    print("=" * 70)

    removed = cleanup_nested_dirs()
    if removed:
        for p in removed:
            print(f"  Removed: {p}")
    else:
        print("  Nothing to remove.")

    print("\n" + "=" * 70)
    if all_ok:
        print("ALL CHECKS PASSED — artifacts ready for train_target")
    else:
        print("FAILURES DETECTED — see above")
    print("=" * 70)
