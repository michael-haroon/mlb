"""Run ONE importance test for ONE target on EC2.

Usage:
    python3.12 scripts/run_importance_single_test_ec2.py --target home_win --test sfi
    python3.12 scripts/run_importance_single_test_ec2.py --target home_win --test sfi --cluster-key pregame/artifacts/importance/cluster_map.json

Tests: mdi_cfi_mdi, mda, cfi_mda, sfi, desub_mda, pca_mda, resid_mda

Downloads game_features.parquet from S3. If --cluster-key is provided, downloads
pre-computed clustering from S3 (avoids redundant ONC). Otherwise computes it locally.
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

from classical_learning.strategy.config import (
    TARGETS_CLASSIFICATION, SFI_CLASS_WEIGHT_OVERRIDES,
)
from classical_learning.strategy.data import compute_temporal_weights, load_features
from classical_learning.analysis.feature_importance import (
    build_rf,
    compute_shared_clustering,
    feat_imp_cfi_mda,
    feat_imp_cfi_mdi_deprado,
    feat_imp_desub_mda,
    feat_imp_mda,
    feat_imp_mdi,
    feat_imp_pca_mda,
    feat_imp_residual_mda,
    feat_imp_sfi,
    synthetic_validation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BUCKET = "mlb-265753586044-us-east-1-an"
FEATURES_KEY = "data/features/game_features.parquet"
OUTPUT_PREFIX = "classical_learning/artifacts/importance"

VALID_TESTS = ["mdi_cfi_mdi", "mda", "cfi_mda", "sfi", "desub_mda", "pca_mda", "resid_mda"]


def download_features(s3) -> Path:
    log.info(f"Downloading s3://{BUCKET}/{FEATURES_KEY} ...")
    obj = s3.get_object(Bucket=BUCKET, Key=FEATURES_KEY)
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.write(obj["Body"].read())
    tmp.close()
    log.info(f"Downloaded to {tmp.name}")
    return Path(tmp.name)


def upload_artifacts(s3, target: str, test_name: str, artifacts: dict,
                     cv_mode: str = "expanding"):
    """Upload dict of {filename: dataframe_or_dict} to S3."""
    prefix = f"{OUTPUT_PREFIX}/{cv_mode}/{target}"
    count = 0
    for filename, data in artifacts.items():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            if isinstance(data, pd.DataFrame):
                data.to_csv(f.name)
            elif isinstance(data, dict):
                with open(f.name, "w") as jf:
                    json.dump(data, jf, indent=2)
            else:
                continue
            key = f"{prefix}/{filename}"
            s3.upload_file(f.name, BUCKET, key)
            count += 1
    log.info(f"Uploaded {count} artifacts to s3://{BUCKET}/{prefix}/")


def run_test(test_name: str, X: pd.DataFrame, y: pd.Series, seasons: pd.Series,
             sample_weight: pd.Series, clusters: dict, regression: bool,
             target: str, n_repeats: int = 1, cv_mode: str = "expanding") -> dict:
    """Run a single importance test and return artifacts dict."""
    scoring = "r2" if regression else "log_loss"
    artifacts = {}
    return_repeat_sd = n_repeats > 1

    if test_name == "mdi_cfi_mdi":
        log.info("Running MDI + CFI-MDI...")
        from classical_learning.analysis.compute import get_n_jobs
        X_filled = X.fillna(X.median())
        clf = build_rf(n_estimators=1000, n_jobs=get_n_jobs(), regression=regression)
        clf.fit(X_filled, y, sample_weight=sample_weight)

        mdi, mdi_raw = feat_imp_mdi(clf, list(X.columns))
        cfi_mdi, cfi_mdi_per_feat, cfi_mdi_raw = feat_imp_cfi_mdi_deprado(
            clf, list(X.columns), clusters)

        artifacts["importance_mdi_raw.csv"] = mdi_raw
        artifacts["importance_mdi_summary.csv"] = mdi
        artifacts["importance_cfi_mdi_cluster.csv"] = cfi_mdi
        artifacts["importance_cfi_mdi_per_feature.csv"] = cfi_mdi_per_feat
        artifacts["importance_cfi_mdi_raw.csv"] = cfi_mdi_raw

    elif test_name == "mda":
        log.info(f"Running MDA (n_repeats={n_repeats}, cv_mode={cv_mode})...")
        clf = build_rf(n_estimators=300, n_jobs=1, regression=regression)
        result = feat_imp_mda(
            clf, X, y, seasons, sample_weight=sample_weight, scoring=scoring,
            n_repeats=n_repeats, return_repeat_sd=return_repeat_sd,
            cv_mode=cv_mode)
        if return_repeat_sd:
            mda, mda_raw, mda_repeat_sd = result
            artifacts["importance_mda_repeat_sd.csv"] = mda_repeat_sd
        else:
            mda, mda_raw = result
        artifacts["importance_mda_raw.csv"] = mda_raw
        artifacts["importance_mda_summary.csv"] = mda

    elif test_name == "cfi_mda":
        log.info(f"Running CFI-MDA (cluster-level permutation, cv_mode={cv_mode})...")
        from classical_learning.analysis.compute import get_n_jobs
        clf = build_rf(n_estimators=300, n_jobs=get_n_jobs(), regression=regression)
        cfi_mda, cfi_mda_raw = feat_imp_cfi_mda(
            clf, X, y, seasons, clusters, sample_weight, scoring=scoring,
            cv_mode=cv_mode)
        artifacts["importance_cfi_mda_raw.csv"] = cfi_mda_raw
        artifacts["importance_cfi_mda_summary.csv"] = cfi_mda

    elif test_name == "sfi":
        log.info(f"Running SFI (single feature importance, cv_mode={cv_mode})...")
        sfi_class_weight = SFI_CLASS_WEIGHT_OVERRIDES.get(target, "balanced")
        clf = build_rf(n_estimators=300, n_jobs=1, regression=regression)
        sfi, sfi_raw = feat_imp_sfi(
            clf, X, y, seasons, sample_weight,
            regression=regression, sfi_class_weight=sfi_class_weight,
            cv_mode=cv_mode)
        artifacts["importance_sfi_raw.csv"] = sfi_raw
        artifacts["importance_sfi_summary.csv"] = sfi

    elif test_name == "desub_mda":
        log.info(f"Running de-substituted MDA (n_repeats={n_repeats}, cv_mode={cv_mode})...")
        result = feat_imp_desub_mda(
            X, y, seasons, clusters,
            sample_weight=sample_weight, scoring=scoring,
            n_estimators=300, regression=regression,
            n_repeats=n_repeats, return_repeat_sd=return_repeat_sd,
            cv_mode=cv_mode)
        if return_repeat_sd:
            desub_mda, desub_mda_raw, desub_repeat_sd = result
            artifacts["importance_desub_mda_repeat_sd.csv"] = desub_repeat_sd
        else:
            desub_mda, desub_mda_raw = result
        artifacts["importance_desub_mda_raw.csv"] = desub_mda_raw
        artifacts["importance_desub_mda_summary.csv"] = desub_mda

    elif test_name == "pca_mda":
        log.info(f"Running PCA-MDA (orthogonal basis, cv_mode={cv_mode})...")
        pca_mda, pca_mda_raw, pca_mda_pc_summary, pca_evr = feat_imp_pca_mda(
            X, y, seasons,
            sample_weight=sample_weight, scoring=scoring,
            n_estimators=300, regression=regression,
            cv_mode=cv_mode)
        artifacts["importance_pca_mda_raw.csv"] = pca_mda_raw
        artifacts["importance_pca_mda_summary.csv"] = pca_mda
        artifacts["importance_pca_mda_pc_summary.csv"] = pca_mda_pc_summary

    elif test_name == "resid_mda":
        log.info(f"Running residualized MDA (cv_mode={cv_mode})...")
        resid_mda, resid_mda_raw = feat_imp_residual_mda(
            X, y, seasons, clusters,
            sample_weight=sample_weight, scoring=scoring,
            n_estimators=300, regression=regression,
            cv_mode=cv_mode)
        artifacts["importance_resid_mda_raw.csv"] = resid_mda_raw
        artifacts["importance_resid_mda_summary.csv"] = resid_mda

    return artifacts


def download_cluster_map(s3, cluster_key: str) -> dict:
    """Download pre-computed cluster map from S3."""
    log.info(f"Downloading pre-computed clusters from s3://{BUCKET}/{cluster_key} ...")
    obj = s3.get_object(Bucket=BUCKET, Key=cluster_key)
    raw = json.loads(obj["Body"].read())
    clusters = {int(k): v for k, v in raw.items()}
    log.info(f"Loaded {len(clusters)} clusters from S3")
    return clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--test", required=True, choices=VALID_TESTS)
    parser.add_argument("--cluster-key", default=None,
                        help="S3 key for pre-computed cluster_map.json (skips ONC)")
    parser.add_argument("--n-repeats", type=int, default=1,
                        help="Permutation repeats per fold (reduces MC noise by sqrt(n))")
    parser.add_argument("--cv-mode", default="expanding",
                        choices=["expanding", "sliding_3"],
                        help="CV fold strategy: expanding (all prior) or sliding_3 (last 3 years)")
    args = parser.parse_args()

    target = args.target
    test_name = args.test
    cluster_key = args.cluster_key
    n_repeats = args.n_repeats
    cv_mode = args.cv_mode
    task = "classification" if target in TARGETS_CLASSIFICATION else "regression"
    regression = (task == "regression")

    log.info(f"{'='*70}")
    log.info(f"Target: {target} | Test: {test_name} | Task: {task} | CV: {cv_mode}")
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

    # Clustering (target-independent)
    if cluster_key:
        clusters = download_cluster_map(s3, cluster_key)
    else:
        log.info("No --cluster-key provided, computing ONC locally...")
        X_for_clustering = X.fillna(X.median())
        clustering = compute_shared_clustering(X_for_clustering)
        clusters = clustering["clusters"]
        # Upload clustering artifacts
        cluster_artifacts = {
            "cluster_map.json": {str(k): v for k, v in clusters.items()},
            "denoising_report.json": clustering["denoising_info"],
        }
        upload_artifacts(s3, target, test_name, cluster_artifacts)

    # Run the test
    t0 = time.time()
    artifacts = run_test(test_name, X, y, seasons, sample_weight, clusters, regression,
                         target, n_repeats=n_repeats, cv_mode=cv_mode)
    elapsed = time.time() - t0
    log.info(f"Test {test_name} completed in {elapsed:.1f}s (n_repeats={n_repeats}, cv_mode={cv_mode})")

    # Upload test artifacts — cv_mode determines output subdirectory
    upload_artifacts(s3, target, test_name, artifacts, cv_mode=cv_mode)

    summary = {
        "target": target,
        "test": test_name,
        "task": task,
        "cv_mode": cv_mode,
        "n_features": X.shape[1],
        "n_samples": len(X),
        "n_clusters": len(clusters),
        "n_repeats": n_repeats,
        "elapsed_secs": round(elapsed, 1),
    }
    log.info(f"DONE: {json.dumps(summary)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
