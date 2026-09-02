"""Compute ONC clustering once and upload to S3.

Phase 1 of importance pipeline: runs ONC (target-independent) on a single
instance, uploads cluster_map.json. Phase 2 instances download this and skip
the 2+ hour clustering step.

Usage:
    python3.12 scripts/run_onc_clustering_ec2.py
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import boto3
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.strategy.data import load_features
from classical_learning.analysis.feature_importance import compute_shared_clustering

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BUCKET = "mlb-265753586044-us-east-1-an"
FEATURES_KEY = "data/features/game_features.parquet"
OUTPUT_KEY = "classical_learning/artifacts/importance/cluster_map.json"
DENOISING_KEY = "classical_learning/artifacts/importance/denoising_report.json"


def main():
    log.info("=" * 70)
    log.info("Phase 1: ONC Clustering (target-independent)")
    log.info("=" * 70)

    s3 = boto3.client("s3")

    # Download features
    log.info(f"Downloading s3://{BUCKET}/{FEATURES_KEY} ...")
    obj = s3.get_object(Bucket=BUCKET, Key=FEATURES_KEY)
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.write(obj["Body"].read())
    tmp.close()
    log.info(f"Downloaded to {tmp.name}")

    # Load any target just to get X (clustering is target-independent)
    X, _, _, _ = load_features(Path(tmp.name), "home_win", "2015+")

    nan_pct = X.isna().mean()
    valid_cols = nan_pct[nan_pct < 0.95].index.tolist()
    dropped = X.shape[1] - len(valid_cols)
    X = X[valid_cols]
    log.info(f"Features: {X.shape[1]} cols, {len(X):,} samples (dropped {dropped} >95% NaN)")

    # Compute ONC
    t0 = time.time()
    X_for_clustering = X.fillna(X.median())
    clustering = compute_shared_clustering(X_for_clustering)
    elapsed = time.time() - t0
    clusters = clustering["clusters"]
    log.info(f"ONC complete: {len(clusters)} clusters in {elapsed:.1f}s")

    # Upload cluster_map.json
    cluster_data = json.dumps({str(k): v for k, v in clusters.items()}, indent=2)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(cluster_data)
        f.flush()
        s3.upload_file(f.name, BUCKET, OUTPUT_KEY)
    log.info(f"Uploaded cluster map to s3://{BUCKET}/{OUTPUT_KEY}")

    # Upload denoising report
    denoising_data = json.dumps(clustering["denoising_info"], indent=2)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(denoising_data)
        f.flush()
        s3.upload_file(f.name, BUCKET, DENOISING_KEY)
    log.info(f"Uploaded denoising report to s3://{BUCKET}/{DENOISING_KEY}")

    # Write a sentinel file so the launcher can detect completion
    sentinel_key = "classical_learning/artifacts/importance/clustering_done.sentinel"
    s3.put_object(Bucket=BUCKET, Key=sentinel_key, Body=f"done at {time.time()}")
    log.info(f"Wrote sentinel: s3://{BUCKET}/{sentinel_key}")

    log.info(f"DONE — {len(clusters)} clusters, {elapsed:.1f}s elapsed")


if __name__ == "__main__":
    main()
