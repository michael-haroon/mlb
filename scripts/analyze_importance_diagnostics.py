"""Diagnostic analysis of feature importance results.

Consumes raw per-fold/per-tree importance CSVs from S3 and produces:
1. Cross-test concordance: which features score positive on ALL tests across ALL folds
2. Trend analysis: Kendall's tau + Theil-Sen slope per feature per test
3. Behavioral classification: stable-positive, stable-negative, trending-up,
   trending-down, noisy/oscillating, mixed-signal
4. Cross-target stability: features that are consistently important across targets

Usage:
    conda run -n pred python scripts/analyze_importance_diagnostics.py [--target home_win]
    conda run -n pred python scripts/analyze_importance_diagnostics.py --all

Downloads from S3: pregame/artifacts/importance/{target}/importance_{test}_raw.csv
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, theilslopes, iqr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

BUCKET = "mlb-265753586044-us-east-1-an"
PREFIX = "pregame/artifacts/importance"

TARGETS = [
    "home_win", "yrfi", "first_5_home_win", "extra_innings",
    "home_run_diff", "total_runs", "home_runs", "away_runs",
    "first_5_home_run_diff", "first_5_total_runs",
]

# Tests that produce per-fold raw data (fold × feature matrices)
FOLD_TESTS = ["mda", "sfi", "desub_mda", "pca_mda", "resid_mda"]

# Tests that produce per-tree raw data (tree × feature matrices)
TREE_TESTS = ["mdi"]

# Cluster-level tests
CLUSTER_TESTS = ["cfi_mda"]


def download_csv(s3, target: str, filename: str) -> pd.DataFrame | None:
    """Download a CSV from S3, return DataFrame or None if not found."""
    key = f"{PREFIX}/{target}/{filename}"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()), index_col=0)
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as e:
        log.warning(f"Failed to download {key}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Per-feature trend analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_trend(fold_values: np.ndarray) -> dict:
    """Analyze temporal behavior of a feature's fold-level importance.

    Returns dict with:
        - tau: Kendall's tau (rank correlation with fold index)
        - tau_p: p-value for tau
        - slope: Theil-Sen slope (robust linear trend)
        - cv: coefficient of variation (noise level)
        - behavior: classification string
        - all_positive: bool (all folds > 0)
        - all_negative: bool (all folds < 0)
        - frac_positive: fraction of folds with positive importance
        - mean, median, std, iqr
        - last_fold: value of most recent fold
        - first_fold: value of earliest fold
    """
    vals = np.asarray(fold_values, dtype=np.float64)
    valid = vals[~np.isnan(vals)]
    n = len(valid)

    result = {
        "n_folds": n,
        "n_nan": int(np.isnan(vals).sum()),
    }

    if n < 3:
        result["behavior"] = "insufficient_data"
        return result

    mean = float(np.mean(valid))
    median = float(np.median(valid))
    std = float(np.std(valid, ddof=1))
    result["mean"] = mean
    result["median"] = median
    result["std"] = std
    result["iqr"] = float(iqr(valid))
    result["cv"] = abs(std / mean) if abs(mean) > 1e-15 else float("inf")
    result["all_positive"] = bool((valid > 0).all())
    result["all_negative"] = bool((valid < 0).all())
    result["frac_positive"] = float((valid > 0).mean())
    result["first_fold"] = float(valid[0])
    result["last_fold"] = float(valid[-1])

    # Trend: Kendall's tau (non-parametric rank correlation with fold index)
    x = np.arange(n)
    tau, tau_p = kendalltau(x, valid)
    result["tau"] = float(tau) if not np.isnan(tau) else 0.0
    result["tau_p"] = float(tau_p) if not np.isnan(tau_p) else 1.0

    # Theil-Sen slope (robust to outliers)
    slope, intercept, _, _ = theilslopes(valid, x)
    result["slope"] = float(slope)
    result["intercept"] = float(intercept)

    # Oscillation detection: count sign changes in consecutive differences
    diffs = np.diff(valid)
    sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
    max_sign_changes = max(n - 2, 1)
    result["oscillation_ratio"] = sign_changes / max_sign_changes
    result["n_sign_changes"] = sign_changes

    # Behavioral classification
    behavior = _classify_behavior(
        tau=result["tau"],
        tau_p=result["tau_p"],
        cv=result["cv"],
        all_positive=result["all_positive"],
        all_negative=result["all_negative"],
        frac_positive=result["frac_positive"],
        oscillation_ratio=result["oscillation_ratio"],
        mean=mean,
    )
    result["behavior"] = behavior

    return result


def _classify_behavior(tau, tau_p, cv, all_positive, all_negative,
                       frac_positive, oscillation_ratio, mean) -> str:
    """Classify feature behavior into a diagnostic category.

    Categories (not mutually exclusive in theory, but assigned by priority):
    - stable_positive: all folds positive, low CV, no significant trend
    - stable_negative: all folds negative, low CV
    - trending_up: significant positive tau (p < 0.10), mean may be negative but improving
    - trending_down: significant negative tau (p < 0.10)
    - noisy_positive: mostly positive but high CV or oscillation
    - noisy_negative: mostly negative but high CV or oscillation
    - oscillating: high sign-change ratio (>0.6), no clear direction
    - mixed_signal: some positive, some negative, no clear trend
    - emerging: trending up, last folds positive, early folds negative
    - decaying: trending down, early folds positive, last folds negative
    """
    sig_trend = tau_p < 0.10
    strong_trend = tau_p < 0.05

    # Stable categories (low noise, no trend)
    if all_positive and cv < 1.0 and not sig_trend:
        return "stable_positive"
    if all_negative and cv < 1.0 and not sig_trend:
        return "stable_negative"

    # Strong trend categories
    if strong_trend and tau > 0.3:
        if frac_positive < 0.5 and mean < 0:
            return "emerging"  # currently negative but improving
        return "trending_up"
    if strong_trend and tau < -0.3:
        if frac_positive > 0.5 and mean > 0:
            return "decaying"  # currently positive but deteriorating
        return "trending_down"

    # Oscillating (many sign changes, no clear direction)
    if oscillation_ratio > 0.6 and not sig_trend:
        if frac_positive > 0.6:
            return "noisy_positive"
        elif frac_positive < 0.4:
            return "noisy_negative"
        return "oscillating"

    # Mixed signal
    if 0.3 < frac_positive < 0.7 and not sig_trend:
        return "mixed_signal"

    # Weak trend categories
    if sig_trend and tau > 0:
        return "trending_up"
    if sig_trend and tau < 0:
        return "trending_down"

    # Default based on sign dominance
    if frac_positive > 0.7:
        return "noisy_positive" if cv > 1.0 else "stable_positive"
    if frac_positive < 0.3:
        return "noisy_negative" if cv > 1.0 else "stable_negative"

    return "mixed_signal"


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-test concordance
# ─────────────────────────────────────────────────────────────────────────────

def compute_concordance(test_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """For each feature, check which tests have all-positive folds.

    Returns DataFrame with features as rows and tests as columns,
    values are: 'all_positive', 'mostly_positive', 'mixed', 'mostly_negative', 'all_negative'
    """
    all_features = set()
    for df in test_results.values():
        if df is not None:
            all_features.update(df.columns.tolist())
    all_features = sorted(all_features)

    records = []
    for feat in all_features:
        row = {"feature": feat}
        n_all_positive = 0
        n_tests = 0

        for test_name, df in test_results.items():
            if df is None or feat not in df.columns:
                row[f"{test_name}_status"] = "missing"
                continue

            vals = df[feat].dropna().values
            if len(vals) == 0:
                row[f"{test_name}_status"] = "no_data"
                continue

            n_tests += 1
            frac_pos = (vals > 0).mean()

            if frac_pos == 1.0:
                status = "all_positive"
                n_all_positive += 1
            elif frac_pos >= 0.75:
                status = "mostly_positive"
            elif frac_pos >= 0.25:
                status = "mixed"
            elif frac_pos > 0:
                status = "mostly_negative"
            else:
                status = "all_negative"

            row[f"{test_name}_status"] = status
            row[f"{test_name}_frac_pos"] = frac_pos
            row[f"{test_name}_mean"] = float(vals.mean())

        row["n_tests_all_positive"] = n_all_positive
        row["n_tests_available"] = n_tests
        row["unanimous_positive"] = (n_all_positive == n_tests and n_tests > 0)
        records.append(row)

    return pd.DataFrame(records).set_index("feature").sort_values(
        "n_tests_all_positive", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Main analysis pipeline
# ─────────────────────────────────────────────────────────────────────────────

def analyze_target(s3, target: str, output_dir: Path) -> dict:
    """Full diagnostic analysis for one target."""
    log.info(f"{'='*60}")
    log.info(f"Analyzing: {target}")
    log.info(f"{'='*60}")

    target_dir = output_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)

    # Download all raw importance data
    test_raws = {}
    for test in FOLD_TESTS:
        df = download_csv(s3, target, f"importance_{test}_raw.csv")
        if df is not None:
            test_raws[test] = df
            log.info(f"  {test}: {df.shape[1]} features × {df.shape[0]} folds")
        else:
            log.warning(f"  {test}: NOT FOUND")

    # MDI raw (per-tree)
    mdi_raw = download_csv(s3, target, "importance_mdi_raw.csv")
    if mdi_raw is not None:
        log.info(f"  mdi: {mdi_raw.shape[1]} features × {mdi_raw.shape[0]} trees")

    # CFI-MDA raw
    cfi_mda_raw = download_csv(s3, target, "importance_cfi_mda_raw.csv")
    if cfi_mda_raw is not None:
        log.info(f"  cfi_mda: {cfi_mda_raw.shape[1]} clusters × {cfi_mda_raw.shape[0]} folds")

    if not test_raws:
        log.warning(f"  No raw data found for {target} — skipping")
        return {}

    # ── 1. Per-feature trend analysis (fold-based tests) ──────────────────
    log.info("  Computing per-feature trend analysis...")
    trend_records = []
    for test_name, df in test_raws.items():
        for feat in df.columns:
            vals = df[feat].values
            analysis = analyze_trend(vals)
            analysis["feature"] = feat
            analysis["test"] = test_name
            trend_records.append(analysis)

    trend_df = pd.DataFrame(trend_records)
    trend_df.to_csv(target_dir / "trend_analysis.csv", index=False)

    # ── 2. Cross-test concordance ─────────────────────────────────────────
    log.info("  Computing cross-test concordance...")
    concordance = compute_concordance(test_raws)
    concordance.to_csv(target_dir / "concordance.csv")

    # Features that are all-positive across ALL available tests
    unanimous = concordance[concordance["unanimous_positive"]]
    log.info(f"  Unanimous positive (all folds, all tests): {len(unanimous)} features")
    if len(unanimous) > 0:
        log.info(f"    {unanimous.index.tolist()[:20]}")

    # ── 3. Behavior summary (pivot: feature × test → behavior) ────────────
    log.info("  Building behavior matrix...")
    behavior_pivot = trend_df.pivot(index="feature", columns="test", values="behavior")
    behavior_pivot.to_csv(target_dir / "behavior_matrix.csv")

    # ── 4. Summary statistics ─────────────────────────────────────────────
    behavior_counts = trend_df["behavior"].value_counts().to_dict()
    log.info(f"  Behavior distribution: {behavior_counts}")

    # Features with consistent behavior across tests
    consistent = []
    for feat in behavior_pivot.index:
        behaviors = behavior_pivot.loc[feat].dropna().values
        if len(behaviors) >= 3 and len(set(behaviors)) == 1:
            consistent.append({"feature": feat, "behavior": behaviors[0],
                             "n_tests": len(behaviors)})
    consistent_df = pd.DataFrame(consistent)
    if not consistent_df.empty:
        consistent_df.to_csv(target_dir / "consistent_behavior.csv", index=False)
        log.info(f"  Features with consistent behavior across ≥3 tests: {len(consistent_df)}")

    # ── 5. MDI tree-level diagnostics ─────────────────────────────────────
    mdi_diagnostics = None
    if mdi_raw is not None:
        log.info("  Computing MDI tree-level diagnostics...")
        mdi_diag = []
        for feat in mdi_raw.columns:
            vals = mdi_raw[feat].dropna().values
            if len(vals) < 10:
                continue
            mdi_diag.append({
                "feature": feat,
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "cv": float(vals.std() / vals.mean()) if vals.mean() > 1e-15 else float("inf"),
                "pct_nonzero": float((vals > 0).mean()),
                "p25": float(np.percentile(vals, 25)),
                "p75": float(np.percentile(vals, 75)),
            })
        mdi_diagnostics = pd.DataFrame(mdi_diag).set_index("feature")
        mdi_diagnostics = mdi_diagnostics.sort_values("mean", ascending=False)
        mdi_diagnostics.to_csv(target_dir / "mdi_diagnostics.csv")

    # ── 6. CFI-MDA cluster diagnostics ────────────────────────────────────
    if cfi_mda_raw is not None:
        log.info("  Computing CFI-MDA cluster diagnostics...")
        cfi_diag = []
        for cluster in cfi_mda_raw.columns:
            vals = cfi_mda_raw[cluster].dropna().values
            if len(vals) == 0:
                continue
            analysis = analyze_trend(vals)
            analysis["cluster"] = cluster
            cfi_diag.append(analysis)
        cfi_diag_df = pd.DataFrame(cfi_diag)
        cfi_diag_df.to_csv(target_dir / "cfi_mda_diagnostics.csv", index=False)

    # ── 7. Top features report ────────────────────────────────────────────
    log.info("  Generating top features report...")
    # Rank by number of tests scoring all-positive, then by mean importance
    report = concordance[["n_tests_all_positive", "n_tests_available", "unanimous_positive"]].copy()
    for test_name in FOLD_TESTS:
        mean_col = f"{test_name}_mean"
        if mean_col in concordance.columns:
            report[f"mean_{test_name}"] = concordance[mean_col]
    report.to_csv(target_dir / "top_features_report.csv")

    summary = {
        "target": target,
        "n_features": len(concordance),
        "n_unanimous_positive": int(len(unanimous)),
        "behavior_distribution": behavior_counts,
        "n_consistent_behavior": len(consistent_df) if not consistent_df.empty else 0,
        "tests_available": list(test_raws.keys()),
    }
    with open(target_dir / "diagnostics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=None,
                       help="Single target to analyze (default: all)")
    parser.add_argument("--all", action="store_true",
                       help="Analyze all targets")
    parser.add_argument("--output", default="data/importance_diagnostics",
                       help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")

    targets = TARGETS if (args.all or args.target is None) else [args.target]

    all_summaries = {}
    for target in targets:
        summary = analyze_target(s3, target, output_dir)
        all_summaries[target] = summary

    # Cross-target stability analysis
    if len(targets) > 1:
        log.info("=" * 60)
        log.info("Cross-target stability analysis")
        log.info("=" * 60)

        # Load concordance from each target, find features unanimous across targets
        cross_target = {}
        for target in targets:
            csv_path = output_dir / target / "concordance.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, index_col=0)
                for feat in df.index:
                    if feat not in cross_target:
                        cross_target[feat] = {"n_targets_unanimous": 0,
                                             "n_targets_present": 0,
                                             "targets_unanimous": []}
                    cross_target[feat]["n_targets_present"] += 1
                    if df.loc[feat, "unanimous_positive"]:
                        cross_target[feat]["n_targets_unanimous"] += 1
                        cross_target[feat]["targets_unanimous"].append(target)

        cross_df = pd.DataFrame(cross_target).T
        cross_df = cross_df.sort_values("n_targets_unanimous", ascending=False)
        cross_df.to_csv(output_dir / "cross_target_stability.csv")

        universally_strong = cross_df[cross_df["n_targets_unanimous"] >= 5]
        log.info(f"Features unanimous in ≥5 targets: {len(universally_strong)}")
        if len(universally_strong) > 0:
            for feat in universally_strong.index[:20]:
                row = universally_strong.loc[feat]
                log.info(f"  {feat}: {row['n_targets_unanimous']}/{row['n_targets_present']} targets")

    with open(output_dir / "all_summaries.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    log.info("Done. Results in: %s", output_dir)


if __name__ == "__main__":
    main()
