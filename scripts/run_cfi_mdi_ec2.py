"""Run CFI-MDI only (de Prado per-tree aggregation) on EC2.

Downloads precomputed clustering and features from S3, fits RF,
computes correct cluster MDI, uploads ONLY the CFI-MDI artifacts.
Does NOT overwrite existing importance files.

Usage on EC2:
    python3.11 scripts/run_cfi_mdi_ec2.py --target home_win
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
from classical_learning.analysis.feature_importance import run_cfi_mdi_only

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


def download_cluster_map(s3, target: str) -> dict:
    """Download cluster_map.json for this target from the existing importance run."""
    key = f"{OUTPUT_PREFIX}/{target}/cluster_map.json"
    log.info(f"Downloading s3://{BUCKET}/{key} ...")
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    raw = json.loads(obj["Body"].read())
    clusters = {int(k): v for k, v in raw.items()}
    log.info(f"  Loaded {len(clusters)} clusters")
    return clusters


def upload_cfi_mdi_artifacts(s3, target: str, local_dir: Path):
    """Upload ONLY CFI-MDI files — does not touch other importance artifacts."""
    prefix = f"{OUTPUT_PREFIX}/{target}"
    files = [
        "importance_cfi_mdi_cluster.csv",
        "importance_cfi_mdi_per_feature.csv",
        "importance_cfi_mdi_raw.csv",
    ]
    for fname in files:
        fpath = local_dir / fname
        if fpath.exists():
            s3.upload_file(str(fpath), BUCKET, f"{prefix}/{fname}")
            log.info(f"  Uploaded {fname} → s3://{BUCKET}/{prefix}/{fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    target = args.target
    regression = target not in TARGETS_CLASSIFICATION

    log.info(f"{'='*70}")
    log.info(f"CFI-MDI (de Prado) | Target: {target} | "
             f"Task: {'regression' if regression else 'classification'}")
    log.info(f"{'='*70}")

    s3 = boto3.client("s3")
    features_path = download_features(s3)
    clusters = download_cluster_map(s3, target)

    output_dir = Path(tempfile.mkdtemp(prefix=f"cfi_mdi_{target}_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, seasons, _ = load_features(features_path, target, "2015+")
    sample_weight = compute_temporal_weights(seasons)

    nan_pct = X.isna().mean()
    valid_cols = nan_pct[nan_pct < 0.95].index.tolist()
    dropped = X.shape[1] - len(valid_cols)
    X = X[valid_cols]
    log.info(f"Features: {X.shape[1]} cols, {len(X):,} samples (dropped {dropped} >95% NaN)")

    t0 = time.time()
    results = run_cfi_mdi_only(
        X=X, y=y, clusters=clusters,
        sample_weight=sample_weight, regression=regression,
    )
    elapsed = time.time() - t0
    log.info(f"CFI-MDI complete in {elapsed:.1f}s")

    results["cfi_mdi_cluster"].to_csv(output_dir / "importance_cfi_mdi_cluster.csv")
    results["cfi_mdi_per_feature"].to_csv(output_dir / "importance_cfi_mdi_per_feature.csv")
    results["cfi_mdi_raw"].to_csv(output_dir / "importance_cfi_mdi_raw.csv")

    upload_cfi_mdi_artifacts(s3, target, output_dir)

    summary = {
        "target": target,
        "n_features": X.shape[1],
        "n_samples": len(X),
        "n_clusters": len(clusters),
        "elapsed_s": round(elapsed, 1),
    }
    log.info(f"DONE: {json.dumps(summary)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
