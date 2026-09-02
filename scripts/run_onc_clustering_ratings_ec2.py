"""Compute ONC clustering on RATINGS SUBSET and upload to S3.

Phase 1 of ratings importance pipeline. With 59 features this completes in
minutes (not hours), but we still run it on a dedicated instance to produce
cluster_map.json before importance instances start.

Usage:
    python3.11 scripts/run_onc_clustering_ratings_ec2.py
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
OUTPUT_KEY = "classical_learning/artifacts/importance_ratings/cluster_map.json"
DENOISING_KEY = "classical_learning/artifacts/importance_ratings/denoising_report.json"

RATING_KEYWORDS = ("massey", "colley", "elo", "wolfe", "pythag", "srs", "log5", "consensus")


def filter_rating_features(X: pd.DataFrame) -> pd.DataFrame:
    """Filter to only columns containing a rating system keyword."""
    rating_cols = []
    for c in X.columns:
        c_check = c.replace("velo", "____")
        if any(kw in c_check for kw in RATING_KEYWORDS):
            rating_cols.append(c)
    return X[rating_cols]


def main():
    log.info("=" * 70)
    log.info("Phase 1: ONC Clustering — RATINGS SUBSET")
    log.info("=" * 70)

    s3 = boto3.client("s3")

    log.info(f"Downloading s3://{BUCKET}/{FEATURES_KEY} ...")
    obj = s3.get_object(Bucket=BUCKET, Key=FEATURES_KEY)
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.write(obj["Body"].read())
    tmp.close()
    log.info(f"Downloaded to {tmp.name}")

    X, _, _, _ = load_features(Path(tmp.name), "home_win", "2015+")

    # Filter to ratings only
    X = filter_rating_features(X)
    log.info(f"Ratings subset: {X.shape[1]} features, {len(X):,} samples")

    nan_pct = X.isna().mean()
    valid_cols = nan_pct[nan_pct < 0.95].index.tolist()
    dropped = X.shape[1] - len(valid_cols)
    X = X[valid_cols]
    if dropped:
        log.info(f"Dropped {dropped} features with >95% NaN")

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

    # Write sentinel
    sentinel_key = "classical_learning/artifacts/importance_ratings/clustering_done.sentinel"
    s3.put_object(Bucket=BUCKET, Key=sentinel_key, Body=f"done at {time.time()}")
    log.info(f"Wrote sentinel: s3://{BUCKET}/{sentinel_key}")

    log.info(f"DONE — {len(clusters)} clusters, {elapsed:.1f}s elapsed")


if __name__ == "__main__":
    main()
