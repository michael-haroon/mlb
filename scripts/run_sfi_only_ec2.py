"""Run SFI only (no MDI/MDA/PCA cross-check) with class_weight fix for extra_innings.

Usage on EC2:
    python3.11 scripts/run_sfi_only_ec2.py --target extra_innings

Downloads game_features.parquet from S3.  Calls feat_imp_sfi directly to avoid
re-running MDI, CFI, PCA cross-check (those artifacts are already saved from the
previous full run).  Uploads only two files:
  - importance_sfi_raw.csv   (folds × features score matrix)
  - importance_sfi_meta.json (sfi_null for regate_and_route.py)

Run regate_and_route.py locally after downloading those two files to refresh
feature_report.csv and routing_report.json.
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

from classical_learning.strategy.config import SFI_CLASS_WEIGHT_OVERRIDES, TARGETS_CLASSIFICATION
from classical_learning.strategy.data import compute_temporal_weights, load_features
from classical_learning.analysis.feature_importance import build_rf, feat_imp_sfi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BUCKET = "mlb-265753586044-us-east-1-an"
FEATURES_KEY = "artifacts/features/game_features.parquet"
OUTPUT_PREFIX = "artifacts/importance"


def download_features(s3) -> Path:
    log.info(f"Downloading s3://{BUCKET}/{FEATURES_KEY} ...")
    obj = s3.get_object(Bucket=BUCKET, Key=FEATURES_KEY)
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.write(obj["Body"].read())
    tmp.close()
    log.info(f"Downloaded to {tmp.name}")
    return Path(tmp.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    target = args.target
    task = "classification" if target in TARGETS_CLASSIFICATION else "regression"
    regression = (task == "regression")
    sfi_class_weight = SFI_CLASS_WEIGHT_OVERRIDES.get(target, "balanced")

    log.info(f"{'='*70}")
    log.info(f"Target: {target} | Task: {task} | sfi_class_weight={sfi_class_weight!r}")
    log.info(f"{'='*70}")

    s3 = boto3.client("s3")
    features_path = download_features(s3)

    X, y, seasons, _game_pks = load_features(features_path, target, "2015+")
    sample_weight = compute_temporal_weights(seasons)

    nan_pct = X.isna().mean()
    valid_cols = nan_pct[nan_pct < 0.95].index.tolist()
    dropped = X.shape[1] - len(valid_cols)
    X = X[valid_cols]
    log.info(f"Features: {X.shape[1]} cols, {len(X):,} samples (dropped {dropped} >95% NaN)")

    log.info("Running SFI (LOYO, BaggingClassifier, n_estimators=300)...")
    t0 = time.time()
    sfi, sfi_raw = feat_imp_sfi(
        build_rf(n_estimators=300, n_jobs=1, regression=regression),
        X, y, seasons, sample_weight,
        regression=regression,
        sfi_class_weight=sfi_class_weight,
    )
    elapsed = time.time() - t0
    log.info(f"SFI complete in {elapsed:.1f}s")

    null_col = "null_r2" if regression else "null_log_loss"
    sfi_null = float(sfi[null_col].iloc[0]) if null_col in sfi.columns else 0.0
    log.info(f"sfi_null={sfi_null:.6f}")

    # Write both artifacts to a temp dir, upload only those two files
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"sfi_{target}_"))
    sfi_raw_path = tmp_dir / "importance_sfi_raw.csv"
    meta_path = tmp_dir / "importance_sfi_meta.json"

    sfi_raw.to_csv(sfi_raw_path)
    with open(meta_path, "w") as f:
        json.dump({"sfi_null": sfi_null, "target": target, "sfi_class_weight": sfi_class_weight}, f)

    prefix = f"{OUTPUT_PREFIX}/{target}"
    s3.upload_file(str(sfi_raw_path), BUCKET, f"{prefix}/importance_sfi_raw.csv")
    s3.upload_file(str(meta_path), BUCKET, f"{prefix}/importance_sfi_meta.json")
    log.info(f"Uploaded 2 artifacts to s3://{BUCKET}/{prefix}/")

    top10 = sfi.head(10).index.tolist()
    log.info(f"Top-10 SFI features: {top10}")
    log.info(f"DONE: {target} | sfi_null={sfi_null:.4f} | elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
