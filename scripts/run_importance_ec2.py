"""Run feature importance for a single target, upload results to S3.

Usage on EC2:
    python3.11 run_importance_ec2.py --target home_win

Reads game_features.parquet from S3, runs full de Prado importance pipeline,
uploads raw artifacts to s3://BUCKET/artifacts/importance/{target}/.

Routing-agnostic: produces raw measurement only.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import boto3
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.strategy.config import TARGETS_CLASSIFICATION
from classical_learning.strategy.data import compute_temporal_weights, load_features
from classical_learning.analysis.feature_importance import (
    compute_shared_clustering,
    run_all_importance,
    synthetic_validation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BUCKET = "mlb-265753586044-us-east-1-an"
FEATURES_KEY = "pregame/artifacts/features/game_features.parquet"
OUTPUT_PREFIX = "pregame/artifacts/importance"


def download_features(s3) -> Path:
    log.info(f"Downloading s3://{BUCKET}/{FEATURES_KEY} ...")
    obj = s3.get_object(Bucket=BUCKET, Key=FEATURES_KEY)
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.write(obj["Body"].read())
    tmp.close()
    log.info(f"Downloaded to {tmp.name}")
    return Path(tmp.name)


def upload_artifacts(s3, target: str, local_dir: Path):
    prefix = f"{OUTPUT_PREFIX}/{target}"
    count = 0
    for fpath in local_dir.rglob("*"):
        if fpath.is_file():
            key = f"{prefix}/{fpath.relative_to(local_dir)}"
            s3.upload_file(str(fpath), BUCKET, key)
            count += 1
    log.info(f"Uploaded {count} artifacts to s3://{BUCKET}/{prefix}/")


def run_one_target(s3, features_path: Path, target: str, output_root: Path):
    t0 = time.time()
    target_dir = output_root / target
    target_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir = target_dir / "filtered"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    task = "classification" if target in TARGETS_CLASSIFICATION else "regression"
    regression = (task == "regression")
    log.info(f"{'='*70}")
    log.info(f"Target: {target} | Task: {task}")
    log.info(f"{'='*70}")

    X, y, seasons, _game_pks = load_features(features_path, target, "2015+")
    sample_weight = compute_temporal_weights(seasons)

    nan_pct = X.isna().mean()
    valid_cols = nan_pct[nan_pct < 0.95].index.tolist()
    dropped = X.shape[1] - len(valid_cols)
    X = X[valid_cols]
    log.info(f"Features: {X.shape[1]} cols, {len(X):,} samples (dropped {dropped} >95% NaN)")

    synth = synthetic_validation()
    if not synth["mdi_pass"]:
        log.warning("SYNTHETIC VALIDATION FAILED")

    X_for_clustering = X.fillna(X.median())
    clustering = compute_shared_clustering(X_for_clustering)
    clusters = clustering["clusters"]

    with open(target_dir / "cluster_map.json", "w") as f:
        json.dump({str(k): v for k, v in clusters.items()}, f, indent=2)
    with open(target_dir / "denoising_report.json", "w") as f:
        json.dump(clustering["denoising_info"], f, indent=2)

    results = run_all_importance(
        X=X, y=y, years=seasons,
        sample_weight=sample_weight,
        run_sfi=True,
        run_desub_mda=True,
        run_pca_mda=True,
        run_residual_mda=True,
        regression=regression,
        precomputed=clustering,
    )

    results["summary"].to_csv(target_dir / "importance_summary.csv")
    results["mdi_raw"].to_csv(target_dir / "importance_mdi_raw.csv")
    if results["sfi_raw"] is not None:
        results["sfi_raw"].to_csv(target_dir / "importance_sfi_raw.csv")
    if results["desub_mda_raw"] is not None:
        results["desub_mda_raw"].to_csv(target_dir / "importance_desub_mda_raw.csv")
    if results["pca_mda_raw"] is not None:
        results["pca_mda_raw"].to_csv(target_dir / "importance_pca_mda_raw.csv")
    if results["resid_mda_raw"] is not None:
        results["resid_mda_raw"].to_csv(target_dir / "importance_resid_mda_raw.csv")
    if results["cfi_mda_raw"] is not None:
        results["cfi_mda_raw"].to_csv(target_dir / "importance_cfi_mda_raw.csv")
    if results["cfi_mdi_raw"] is not None:
        results["cfi_mdi_raw"].to_csv(target_dir / "importance_cfi_mdi_raw.csv")
    if results["pca_info"] is not None:
        results["pca_info"].to_csv(target_dir / "pca_cross_check.csv")
    if results["tau_results"] is not None:
        with open(target_dir / "kendall_tau.json", "w") as f:
            json.dump(results["tau_results"], f, indent=2)

    results["filter_report"].to_csv(filtered_dir / "feature_report.csv")

    elapsed = time.time() - t0
    tier_counts = results["filter_report"]["tier"].value_counts().to_dict()
    summary = {
        "target": target,
        "task": task,
        "n_features_input": X.shape[1],
        "n_samples": len(X),
        "n_seasons": int(seasons.nunique()),
        "n_clusters": len(clusters),
        "tier_counts": {k: int(v) for k, v in tier_counts.items()},
        "n_survivors": len(results["survivors"]),
        "synthetic_validation_pass": bool(synth["mdi_pass"]),
        "elapsed_secs": round(elapsed, 1),
    }
    with open(target_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Done: {target} in {elapsed:.1f}s | Tiers: {tier_counts}")
    upload_artifacts(s3, target, target_dir)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    s3 = boto3.client("s3")
    features_path = download_features(s3)
    output_root = Path(tempfile.mkdtemp(prefix="importance_"))
    log.info(f"Output root: {output_root}")

    summary = run_one_target(s3, features_path, args.target, output_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
