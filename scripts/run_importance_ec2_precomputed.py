"""Run feature importance with precomputed clustering (downloaded from S3).

Usage on EC2:
    python3.11 run_importance_ec2_precomputed.py --target home_win

Downloads precomputed cluster_map.json and denoising_report.json from the
uncapped cap-test run, skips the 3h clustering step, runs importance only.
Uploads results to s3://BUCKET/artifacts/importance/{target}/.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.strategy.config import TARGETS_CLASSIFICATION
from classical_learning.strategy.data import compute_temporal_weights, load_features
from classical_learning.analysis.feature_importance import (
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
FEATURES_KEY = "artifacts/features/game_features.parquet"
OUTPUT_PREFIX = "artifacts/importance"

CLUSTER_MAP_KEY = "artifacts/importance_cap_test/uncapped/home_runs/cluster_map.json"
DENOISING_KEY = "artifacts/importance_cap_test/uncapped/home_runs/denoising_report.json"


def download_features(s3) -> Path:
    log.info(f"Downloading s3://{BUCKET}/{FEATURES_KEY} ...")
    obj = s3.get_object(Bucket=BUCKET, Key=FEATURES_KEY)
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.write(obj["Body"].read())
    tmp.close()
    log.info(f"Downloaded to {tmp.name}")
    return Path(tmp.name)


def download_precomputed_clustering(s3) -> dict:
    """Download cluster_map and denoising_report from the uncapped cap-test run."""
    log.info("Downloading precomputed clustering from cap-test uncapped run...")

    obj = s3.get_object(Bucket=BUCKET, Key=CLUSTER_MAP_KEY)
    cluster_map_raw = json.loads(obj["Body"].read())
    clusters = {int(k): v for k, v in cluster_map_raw.items()}

    obj = s3.get_object(Bucket=BUCKET, Key=DENOISING_KEY)
    denoising_info = json.loads(obj["Body"].read())

    log.info(f"  Loaded {len(clusters)} clusters from precomputed run")
    return {"clusters": clusters, "denoising_info": denoising_info}


def upload_artifacts(s3, target: str, local_dir: Path):
    prefix = f"{OUTPUT_PREFIX}/{target}"
    count = 0
    for fpath in local_dir.rglob("*"):
        if fpath.is_file():
            key = f"{prefix}/{fpath.relative_to(local_dir)}"
            s3.upload_file(str(fpath), BUCKET, key)
            count += 1
    log.info(f"Uploaded {count} artifacts to s3://{BUCKET}/{prefix}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    target = args.target
    task = "classification" if target in TARGETS_CLASSIFICATION else "regression"
    regression = (task == "regression")

    log.info(f"{'='*70}")
    log.info(f"Target: {target} | Task: {task} | Clustering: precomputed")
    log.info(f"{'='*70}")

    s3 = boto3.client("s3")
    features_path = download_features(s3)
    clustering = download_precomputed_clustering(s3)

    output_root = Path(tempfile.mkdtemp(prefix=f"importance_{target}_"))
    target_dir = output_root / target
    target_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir = target_dir / "filtered"
    filtered_dir.mkdir(parents=True, exist_ok=True)

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

    clusters = clustering["clusters"]

    with open(target_dir / "cluster_map.json", "w") as f:
        json.dump({str(k): v for k, v in clusters.items()}, f, indent=2)
    with open(target_dir / "denoising_report.json", "w") as f:
        json.dump(clustering["denoising_info"], f, indent=2)

    t0 = time.time()
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
    elapsed = time.time() - t0
    log.info(f"Importance pipeline complete in {elapsed:.1f}s")

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
    if results["cfi_mdi_per_feat"] is not None:
        results["cfi_mdi_per_feat"].to_csv(target_dir / "importance_cfi_mdi_per_feature.csv")
    results["cfi_mdi"].to_csv(target_dir / "importance_cfi_mdi_cluster.csv")
    if results["pca_info"] is not None:
        results["pca_info"].to_csv(target_dir / "pca_cross_check.csv")
    if results["tau_results"] is not None:
        with open(target_dir / "kendall_tau.json", "w") as f:
            json.dump(results["tau_results"], f, indent=2)

    results["filter_report"].to_csv(filtered_dir / "feature_report.csv")

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
        "importance_time_s": round(elapsed, 1),
    }
    with open(target_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    upload_artifacts(s3, target, target_dir)
    log.info(f"DONE: {target} in {elapsed:.1f}s | Tiers: {tier_counts}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
