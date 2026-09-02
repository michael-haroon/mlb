"""Run S* sizing curve for one target on EC2, upload results to S3.

Usage on EC2:
    python3.11 run_sizing_ec2.py --target home_win
    python3.11 run_sizing_ec2.py  # runs all targets sequentially

Reads game_features.parquet and feature_report.csv from S3, runs the sizing
curve sweep for all 19 model families (including ydf_oblique_gbt), and uploads
sizing_curve_{target}.json back to s3://BUCKET/artifacts/sizing/.

Run this after any importance run that changes feature_report.csv.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.strategy.config import ALL_TARGETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BUCKET = "mlb-265753586044-us-east-1-an"
FEATURES_KEY = "artifacts/features/game_features.parquet"
IMPORTANCE_PREFIX = "artifacts/importance"
OUTPUT_PREFIX = "artifacts/sizing"


def download_features(s3) -> Path:
    log.info(f"Downloading s3://{BUCKET}/{FEATURES_KEY} ...")
    obj = s3.get_object(Bucket=BUCKET, Key=FEATURES_KEY)
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.write(obj["Body"].read())
    tmp.close()
    log.info(f"Downloaded to {tmp.name}")
    return Path(tmp.name)


def download_feature_report(s3, target: str, importance_root: Path) -> None:
    """Download feature_report.csv into importance_root/{target}/filtered/."""
    key = f"{IMPORTANCE_PREFIX}/{target}/filtered/feature_report.csv"
    log.info(f"Downloading s3://{BUCKET}/{key} ...")
    dest = importance_root / target / "filtered" / "feature_report.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    dest.write_bytes(obj["Body"].read())


def download_mda_artifacts(s3, target: str, crosscheck_root: Path) -> None:
    """Download kendall_tau.json, pca_mda_cluster/feature_importance.csv for MDA routing.

    run_sizing_curve reads these from PCA_CROSSCHECK_DIR/{target}/ to detect
    whether sizing should use MDA-validated cluster-first ordering. Without them,
    sizing falls back to evidence-group ordering and S* will differ from training.
    """
    target_dir = crosscheck_root / target
    target_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("kendall_tau.json", "pca_mda_cluster_importance.csv", "pca_mda_feature_importance.csv"):
        key = f"{IMPORTANCE_PREFIX}/{target}/{fname}"
        dest = target_dir / fname
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            dest.write_bytes(obj["Body"].read())
            log.info(f"Downloaded {key}")
        except s3.exceptions.NoSuchKey:
            log.warning(f"MDA artifact not found: {key} — sizing may fall back to evidence-group routing")


def upload_result(s3, target: str, local_path: Path):
    key = f"{OUTPUT_PREFIX}/sizing_curve_{target}.json"
    s3.upload_file(str(local_path), BUCKET, key)
    log.info(f"Uploaded to s3://{BUCKET}/{key}")

    # Also upload the sizing_summary.json if it exists alongside
    summary_path = local_path.parent / "sizing_summary.json"
    if summary_path.exists():
        s3.upload_file(str(summary_path), BUCKET, f"{OUTPUT_PREFIX}/sizing_summary.json")


def run_one_target(s3, features_path: Path, target: str,
                   output_dir: Path, importance_root: Path,
                   crosscheck_root: Path) -> dict:
    from classical_learning.strategy.feature_sizing import run_sizing_curve

    download_feature_report(s3, target, importance_root)
    download_mda_artifacts(s3, target, crosscheck_root)

    t0 = time.time()
    log.info(f"{'='*70}")
    log.info(f"Sizing: {target}")
    log.info(f"{'='*70}")

    result = run_sizing_curve(
        features_path=features_path,
        target=target,
        output_dir=output_dir,
        data_mode="2015+",
        importance_dir=importance_root,
        fine_grained=True,
    )

    elapsed = time.time() - t0
    log.info(f"Done: {target} in {elapsed:.1f}s | S*={result.get('optimal_S')} "
             f"(sizing fold {result.get('sizing_fold_season')})")

    curve_path = output_dir / f"sizing_curve_{target}.json"
    upload_result(s3, target, curve_path)
    return result


def main():
    parser = argparse.ArgumentParser(description="Sizing curve on EC2")
    parser.add_argument("--target", required=True,
                        help="Single target to size (one instance = one target)")
    parser.add_argument("--self-terminate", action="store_true",
                        help="Halt the OS after upload so EC2 terminate-on-shutdown fires")
    args = parser.parse_args()

    s3 = boto3.client("s3")
    features_path = download_features(s3)

    tmproot = Path(tempfile.mkdtemp())
    output_dir = tmproot / "sizing"
    output_dir.mkdir(parents=True)
    importance_root = tmproot / "importance"
    importance_root.mkdir(parents=True)
    # PCA crosscheck artifacts go under data/importance/ relative to repo root
    # so that PCA_CROSSCHECK_DIR (config.py) resolves correctly at runtime.
    crosscheck_root = Path("/home/ec2-user/mlb/data/importance")
    crosscheck_root.mkdir(parents=True, exist_ok=True)

    log.info(f"Running sizing for target: {args.target}")
    exit_code = 0
    try:
        result = run_one_target(s3, features_path, args.target, output_dir, importance_root, crosscheck_root)
        log.info(f"SUCCESS: {args.target} S*={result.get('optimal_S')} "
                 f"fold={result.get('sizing_fold_season')}")
    except Exception as e:
        log.error(f"FAILED {args.target}: {e}", exc_info=True)
        exit_code = 1

    Path(features_path).unlink(missing_ok=True)

    if args.self_terminate:
        log.info("Self-terminating instance via OS halt...")
        import subprocess
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
