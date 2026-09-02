"""Run ONE interpretability test for ONE target on EC2.

Usage:
    python3.11 scripts/run_interpretability_ec2.py --target home_win --test h_stat
    python3.11 scripts/run_interpretability_ec2.py --target home_win --test ale
    python3.11 scripts/run_interpretability_ec2.py --target home_win --test shap

Tests:
    h_stat  — Friedman's H-statistic for top-N feature pairs (pairwise interactions)
    ale     — Accumulated Local Effects for all features (response shapes)
    shap    — TreeSHAP global importance + interaction matrix

Downloads game_features.parquet from S3. Uses pre-computed MDI ranking from
existing importance results to select top features for interaction analysis.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.strategy.config import TARGETS_CLASSIFICATION
from classical_learning.strategy.data import compute_temporal_weights, load_features
from classical_learning.analysis.interpretability import (
    ale_all_features,
    ale_cv,
    h_statistic_cv,
    h_statistic_top_pairs,
    shap_cv,
    shap_importance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BUCKET = "mlb-265753586044-us-east-1-an"
FEATURES_KEY = "data/features/game_features.parquet"
OUTPUT_PREFIX = "classical_learning/artifacts/interpretability"

VALID_TESTS = ["h_stat", "ale", "shap"]


def download_features(s3) -> Path:
    log.info(f"Downloading s3://{BUCKET}/{FEATURES_KEY} ...")
    obj = s3.get_object(Bucket=BUCKET, Key=FEATURES_KEY)
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.write(obj["Body"].read())
    tmp.close()
    log.info(f"Downloaded to {tmp.name}")
    return Path(tmp.name)


def download_mdi_ranking(s3, target: str) -> pd.DataFrame | None:
    """Try to download pre-computed MDI summary from existing importance results."""
    key = f"classical_learning/artifacts/importance/expanding/{target}/importance_mdi_summary.csv"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        df = pd.read_csv(obj["Body"], index_col=0)
        log.info(f"Loaded pre-computed MDI ranking ({len(df)} features)")
        return df
    except s3.exceptions.NoSuchKey:
        log.info("No pre-computed MDI ranking found — will compute internally")
        return None
    except Exception as e:
        log.warning(f"Failed to load MDI ranking: {e}")
        return None


def upload_artifacts(s3, target: str, test_name: str, artifacts: dict):
    """Upload dict of {filename: dataframe_or_dict_or_ndarray} to S3."""
    prefix = f"{OUTPUT_PREFIX}/{target}"
    count = 0
    for filename, data in artifacts.items():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            if isinstance(data, pd.DataFrame):
                data.to_csv(f.name)
            elif isinstance(data, dict):
                # Serialize dict (may contain numpy arrays)
                serializable = {}
                for k, v in data.items():
                    if isinstance(v, dict):
                        inner = {}
                        for kk, vv in v.items():
                            if isinstance(vv, np.ndarray):
                                inner[kk] = vv.tolist()
                            else:
                                inner[kk] = vv
                        serializable[k] = inner
                    elif isinstance(v, np.ndarray):
                        serializable[k] = v.tolist()
                    else:
                        serializable[k] = v
                with open(f.name, "w") as jf:
                    json.dump(serializable, jf, indent=2, default=str)
                # Change suffix for json
                key = f"{prefix}/{filename}"
                s3.upload_file(f.name, BUCKET, key)
                count += 1
                continue
            else:
                continue
            key = f"{prefix}/{filename}"
            s3.upload_file(f.name, BUCKET, key)
            count += 1
    log.info(f"Uploaded {count} artifacts to s3://{BUCKET}/{prefix}/")


def run_h_stat(X, y, years, sample_weight, regression, mdi_ranking):
    """Run H-statistic analysis."""
    artifacts = {}

    # Single-fold (fast) — top 30 features
    log.info("Running H-stat (single fold, top 30 features)...")
    h_df = h_statistic_top_pairs(
        X, y, years,
        top_n_features=30,
        n_grid=20,
        subsample=500,
        n_estimators=300,
        sample_weight=sample_weight,
        regression=regression,
        mdi_ranking=mdi_ranking,
    )
    artifacts["h_stat_summary.csv"] = h_df

    # CV stability (top 20 features — smaller to be tractable across folds)
    log.info("Running H-stat CV (all folds, top 20 features)...")
    h_cv_summary, h_cv_raw = h_statistic_cv(
        X, y, years,
        top_n_features=20,
        n_grid=15,
        subsample=400,
        n_estimators=300,
        sample_weight=sample_weight,
        regression=regression,
        mdi_ranking=mdi_ranking,
    )
    artifacts["h_stat_cv_summary.csv"] = h_cv_summary
    artifacts["h_stat_cv_raw.csv"] = h_cv_raw

    return artifacts


def run_ale(X, y, years, sample_weight, regression, mdi_ranking):
    """Run ALE analysis."""
    artifacts = {}

    # Top 50 features by MDI (full ALE on all features is too expensive for CV)
    if mdi_ranking is not None:
        feature_subset = list(mdi_ranking.index[:50])
    else:
        feature_subset = None  # all features — single fold only

    # Single-fold full ALE (all features)
    log.info("Running ALE (single fold, all features)...")
    ale_result = ale_all_features(
        X, y, years,
        n_bins=40,
        n_estimators=300,
        sample_weight=sample_weight,
        regression=regression,
    )

    # Serialize ALE curves to a summary DataFrame
    ale_summary_records = []
    ale_curves = {}
    for fname, data in ale_result.items():
        ale_summary_records.append({
            "feature": fname,
            "ale_range": data["range"],
            "monotone": data["monotone"],
            "n_bins": len(data["centers"]),
        })
        ale_curves[fname] = {
            "centers": data["centers"],
            "ale": data["ale"],
        }
    ale_summary = pd.DataFrame(ale_summary_records).sort_values(
        "ale_range", ascending=False).reset_index(drop=True)
    artifacts["ale_summary.csv"] = ale_summary
    artifacts["ale_curves.json"] = ale_curves

    # CV stability (top 50 features only — tractable)
    log.info("Running ALE CV (all folds, top 50 features)...")
    ale_cv_df = ale_cv(
        X, y, years,
        n_bins=30,
        n_estimators=300,
        sample_weight=sample_weight,
        regression=regression,
        feature_subset=feature_subset,
    )
    artifacts["ale_cv_summary.csv"] = ale_cv_df

    return artifacts


def run_shap(X, y, years, sample_weight, regression, mdi_ranking):
    """Run TreeSHAP analysis."""
    artifacts = {}

    # Global importance (single fold)
    log.info("Running TreeSHAP (single fold, with interactions)...")
    result = shap_importance(
        X, y, years,
        n_estimators=300,
        sample_weight=sample_weight,
        regression=regression,
        compute_interactions=True,
        interaction_top_n=30,
        subsample_interactions=200,
    )
    artifacts["shap_global_importance.csv"] = result["global_importance"]
    if "interactions" in result:
        artifacts["shap_interactions.csv"] = result["interactions"]
    if "interaction_matrix" in result:
        artifacts["shap_interaction_matrix.csv"] = result["interaction_matrix"]

    # CV stability
    log.info("Running TreeSHAP CV (all folds)...")
    shap_cv_df = shap_cv(
        X, y, years,
        n_estimators=300,
        sample_weight=sample_weight,
        regression=regression,
    )
    artifacts["shap_cv_summary.csv"] = shap_cv_df

    return artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--test", required=True, choices=VALID_TESTS)
    args = parser.parse_args()

    target = args.target
    test_name = args.test
    task = "classification" if target in TARGETS_CLASSIFICATION else "regression"
    regression = (task == "regression")

    log.info(f"{'='*70}")
    log.info(f"Target: {target} | Test: {test_name} | Task: {task}")
    log.info(f"{'='*70}")

    s3 = boto3.client("s3")
    features_path = download_features(s3)

    X, y, seasons, _ = load_features(features_path, target, "2015+")
    sample_weight = compute_temporal_weights(seasons)

    nan_pct = X.isna().mean()
    valid_cols = nan_pct[nan_pct < 0.95].index.tolist()
    dropped = X.shape[1] - len(valid_cols)
    X = X[valid_cols]
    log.info(f"Features: {X.shape[1]} cols, {len(X):,} samples (dropped {dropped} >95% NaN)")

    # Load pre-computed MDI ranking for feature selection
    mdi_ranking = download_mdi_ranking(s3, target)

    # Run the test
    t0 = time.time()
    if test_name == "h_stat":
        artifacts = run_h_stat(X, y, seasons, sample_weight, regression, mdi_ranking)
    elif test_name == "ale":
        artifacts = run_ale(X, y, seasons, sample_weight, regression, mdi_ranking)
    elif test_name == "shap":
        artifacts = run_shap(X, y, seasons, sample_weight, regression, mdi_ranking)
    else:
        raise ValueError(f"Unknown test: {test_name}")

    elapsed = time.time() - t0
    log.info(f"Test {test_name} completed in {elapsed:.1f}s")

    # Upload
    upload_artifacts(s3, target, test_name, artifacts)

    summary = {
        "target": target,
        "test": test_name,
        "task": task,
        "n_features": X.shape[1],
        "n_samples": len(X),
        "elapsed_secs": round(elapsed, 1),
    }
    log.info(f"DONE: {json.dumps(summary)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
