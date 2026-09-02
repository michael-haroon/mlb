"""De Prado PCA cross-check: eigenvalue rank vs importance rank on PCs.

Thin wrapper around pca_cross_check() in the codebase. Runs all 10 targets
locally, saves results to data/importance/{target}/ and uploads to S3.

For EC2 parallelized version, use scripts/run_pca_crosscheck_ec2.py via
scripts/launch_pca_crosscheck_ec2.sh.

Usage:
    conda run -n pred python scripts/rerun_pca_crosscheck.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.strategy.config import ALL_TARGETS, TARGETS_CLASSIFICATION
from classical_learning.strategy.data import load_features, compute_temporal_weights
from classical_learning.analysis.feature_importance import pca_cross_check

BUCKET = "mlb-265753586044-us-east-1-an"
S3_PREFIX = "artifacts/importance"
FEATURES_PATH = Path("tmp/game_features.parquet")
LOCAL_IMPORTANCE_DIR = Path("data/importance")
DATA_MODE = "2015+"


def main():
    s3 = boto3.client("s3")
    all_results = {}

    for target in ALL_TARGETS:
        print(f"\n{'='*70}")
        print(f"  PCA cross-check: {target}")
        print(f"{'='*70}")

        try:
            X, y, seasons, _ = load_features(FEATURES_PATH, target, DATA_MODE)
        except (ValueError, KeyError) as e:
            print(f"  SKIP: {e}")
            continue

        nan_pct = X.isna().mean()
        valid_cols = nan_pct[nan_pct < 0.95].index.tolist()
        X = X[valid_cols]

        task = "classification" if target in TARGETS_CLASSIFICATION else "regression"
        regression = (task == "regression")
        sample_weight = compute_temporal_weights(seasons)

        print(f"  {X.shape[1]} features, {len(X)} samples, task={task}")

        pca_info, tau_results = pca_cross_check(
            X, y, seasons, sample_weight, regression=regression,
        )

        # Print per-target results
        print(f"  {'Method':<8} {'Tau':>8} {'p':>8}  Verdict")
        print(f"  {'-'*40}")
        for method, res in tau_results.items():
            t = res["tau"]
            pv = res["p_value"]
            if t is not None and pv <= 0.05 and t > 0:
                v = "PASS"
            elif t is not None and pv <= 0.05 and t < 0:
                v = "FAIL"
            else:
                v = "INCONCLUSIVE"
            print(f"  {method:<8} {t:>+.4f} {pv:>8.4f}  {v}")

        # Save locally
        target_dir = LOCAL_IMPORTANCE_DIR / target
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(target_dir / "kendall_tau.json", "w") as f:
            json.dump(tau_results, f, indent=2)
        pca_info.to_csv(target_dir / "pca_cross_check.csv")

        # Upload to S3
        s3.upload_file(str(target_dir / "pca_cross_check.csv"), BUCKET,
                       f"{S3_PREFIX}/{target}/pca_cross_check.csv")
        s3.upload_file(str(target_dir / "kendall_tau.json"), BUCKET,
                       f"{S3_PREFIX}/{target}/kendall_tau.json")

        all_results[target] = tau_results

    # Final summary
    methods = ["MDI", "MDA", "SFI"]
    print(f"\n\n{'='*70}")
    print("  SUMMARY: Eigenvalue rank vs PC importance rank")
    print(f"{'='*70}")

    print(f"\n  {'Target':<24}", end="")
    for m in methods:
        print(f" {'tau':>6} {'p':>6}", end="")
    print()
    print(f"  {'-'*70}")
    for target, results in all_results.items():
        row = f"  {target:<24}"
        for m in methods:
            if m in results and results[m]["tau"] is not None:
                t = results[m]["tau"]
                p = results[m]["p_value"]
                marker = "*" if p <= 0.05 else " "
                row += f" {t:>+.3f} {p:>.3f}{marker}"
            else:
                row += "    N/A    N/A "
        print(row)
    print(f"\n  * = significant (p <= 0.05)")


if __name__ == "__main__":
    main()
