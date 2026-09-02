"""Run feature importance with capped vs uncapped ONC clustering comparison.

Usage on EC2:
    # Capped (fast): runs home_win classification
    python3.11 run_importance_ec2_cap_test.py --target home_win --cap-clusters 20

    # Uncapped (slow): runs home_runs regression
    python3.11 run_importance_ec2_cap_test.py --target home_runs

Both produce artifacts uploaded to separate S3 prefixes so results can be
compared after both complete. Clustering is target-independent so any
difference in cluster_map.json proves the cap matters on real data.
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
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.strategy.config import TARGETS_CLASSIFICATION
from classical_learning.strategy.data import compute_temporal_weights, load_features
from classical_learning.analysis.feature_importance import (
    _onc_flat,
    _global_mean_silhouette,
    _partition_quality,
    compute_shared_clustering,
    denoise_corr,
    detone_corr,
    onc_cluster,
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
OUTPUT_PREFIX = "artifacts/importance_cap_test"


def download_features(s3) -> Path:
    log.info(f"Downloading s3://{BUCKET}/{FEATURES_KEY} ...")
    obj = s3.get_object(Bucket=BUCKET, Key=FEATURES_KEY)
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.write(obj["Body"].read())
    tmp.close()
    log.info(f"Downloaded to {tmp.name}")
    return Path(tmp.name)


def onc_cluster_capped(corr, cap):
    """ONC with capped recursive subdivision. Outer call remains uncapped."""
    log.info(f"    Running ONC with recursive cap={cap}...")
    partition = _onc_flat(corr, max_clusters=None, n_init=20)
    if partition is None or len(partition) <= 1:
        return {0: list(corr.columns)}

    current_quality = _global_mean_silhouette(corr, partition)
    log.info(f"    Initial partition: {len(partition)} clusters, quality={current_quality:.4f}")

    improved = True
    iteration = 0
    while improved:
        improved = False
        iteration += 1
        qualities = _partition_quality(corr, partition)
        sorted_cids = sorted(qualities, key=lambda c: qualities[c])

        for cid in sorted_cids:
            members = partition[cid]
            if len(members) < 4:
                continue

            sub_corr = corr.loc[members, members]
            recursive_cap = min(cap, len(members) // 2)
            sub = _onc_flat(sub_corr, max_clusters=recursive_cap, n_init=20)
            if sub is None or len(sub) <= 1:
                continue

            next_id = max(partition.keys()) + 1
            candidate = {c: m for c, m in partition.items() if c != cid}
            for sub_members in sub.values():
                candidate[next_id] = sub_members
                next_id += 1

            candidate_quality = _global_mean_silhouette(corr, candidate)
            if candidate_quality > current_quality:
                log.info(f"    Iter {iteration}: split cluster {cid} "
                         f"({len(members)} members) → {len(sub)} sub-clusters, "
                         f"quality {current_quality:.4f} → {candidate_quality:.4f}")
                partition = candidate
                current_quality = candidate_quality
                improved = True
                break

    log.info(f"    Converged after {iteration} iterations: {len(partition)} clusters")
    return partition


def compute_clustering_with_cap(X, cap):
    """compute_shared_clustering but with capped ONC."""
    import pandas as pd
    from classical_learning.analysis.feature_importance import blas_full

    log.info(f"Computing target-independent clustering on {X.shape[1]} features (cap={cap})...")
    log.info("  1/2  Marcenko-Pastur denoising + detoning...")

    with blas_full():
        corr_raw = X.corr().fillna(0)
        q = X.shape[0] / X.shape[1]

        evals_raw = np.linalg.eigvalsh(corr_raw.values)
        lambda_plus = (1.0 + (1.0 / q) ** 0.5) ** 2
        n_signal = int((evals_raw > lambda_plus).sum())
        n_noise = int((evals_raw <= lambda_plus).sum())
        signal_var = float(evals_raw[evals_raw > lambda_plus].sum())
        total_var = float(evals_raw.sum())

        denoising_info = {
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "q_ratio": float(q),
            "lambda_plus": float(lambda_plus),
            "n_signal_eigenvalues": n_signal,
            "n_noise_eigenvalues": n_noise,
            "signal_variance_pct": round(100 * signal_var / total_var, 1) if total_var > 0 else 0,
        }
        corr_denoised = denoise_corr(corr_raw, q=q)
        corr_cluster = detone_corr(corr_denoised, n_remove=0)

    log.info(f"    lambda+ = {lambda_plus:.4f}, signal: {n_signal}, noise: {n_noise}")
    log.info("  2/2  ONC clustering (capped greedy divisive)...")

    t0 = time.time()
    clusters = onc_cluster_capped(corr_cluster, cap=cap)
    clustering_time = time.time() - t0

    log.info(f"    Found {len(clusters)} clusters in {clustering_time:.1f}s")
    for cid, members in sorted(clusters.items(), key=lambda x: -len(x[1]))[:10]:
        log.info(f"    Cluster {cid} ({len(members)} features): {members[:3]}...")

    return {
        "clusters": clusters,
        "denoising_info": denoising_info,
        "clustering_time_s": clustering_time,
    }


def upload_artifacts(s3, prefix: str, local_dir: Path):
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
    parser.add_argument("--cap-clusters", type=int, default=None,
                        help="Cap max_clusters in recursive ONC subdivision. "
                             "None = uncapped (original behavior).")
    args = parser.parse_args()

    cap = args.cap_clusters
    variant = f"capped_{cap}" if cap else "uncapped"
    log.info(f"{'='*70}")
    log.info(f"FEATURE IMPORTANCE: target={args.target}, clustering={variant}")
    log.info(f"{'='*70}")

    s3 = boto3.client("s3")
    features_path = download_features(s3)

    output_root = Path(tempfile.mkdtemp(prefix=f"importance_{variant}_"))
    target_dir = output_root / args.target
    target_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output root: {output_root}")

    task = "classification" if args.target in TARGETS_CLASSIFICATION else "regression"
    regression = (task == "regression")
    log.info(f"Target: {args.target} | Task: {task}")

    X, y, seasons, _game_pks = load_features(features_path, args.target, "2015+")
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

    # --- Clustering (the part under test) ---
    t_cluster_start = time.time()
    if cap:
        clustering = compute_clustering_with_cap(X_for_clustering, cap=cap)
    else:
        clustering = compute_shared_clustering(X_for_clustering)
    t_cluster_total = time.time() - t_cluster_start
    clusters = clustering["clusters"]
    log.info(f"CLUSTERING COMPLETE: {len(clusters)} clusters in {t_cluster_total:.1f}s")

    # Save clustering artifacts
    with open(target_dir / "cluster_map.json", "w") as f:
        json.dump({str(k): v for k, v in clusters.items()}, f, indent=2)
    with open(target_dir / "denoising_report.json", "w") as f:
        json.dump(clustering["denoising_info"], f, indent=2)
    meta = {
        "variant": variant,
        "cap_value": cap,
        "target": args.target,
        "task": task,
        "n_clusters": len(clusters),
        "clustering_time_s": round(t_cluster_total, 1),
        "cluster_sizes": sorted([len(m) for m in clusters.values()], reverse=True),
    }
    with open(target_dir / "clustering_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Upload clustering results immediately (in case importance takes forever)
    prefix = f"{OUTPUT_PREFIX}/{variant}/{args.target}"
    upload_artifacts(s3, prefix, target_dir)
    log.info("Clustering artifacts uploaded — starting importance pipeline...")

    # --- Full importance pipeline ---
    t_imp_start = time.time()
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
    t_imp_total = time.time() - t_imp_start
    log.info(f"IMPORTANCE COMPLETE in {t_imp_total:.1f}s")

    # Save importance artifacts
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
    if results["pca_info"] is not None:
        results["pca_info"].to_csv(target_dir / "pca_cross_check.csv")
    if results["tau_results"] is not None:
        with open(target_dir / "kendall_tau.json", "w") as f:
            json.dump(results["tau_results"], f, indent=2)
    results["filter_report"].to_csv(target_dir / "feature_report.csv")

    # Final summary
    tier_counts = results["filter_report"]["tier"].value_counts().to_dict()
    summary = {
        "variant": variant,
        "cap_value": cap,
        "target": args.target,
        "task": task,
        "n_features_input": X.shape[1],
        "n_samples": len(X),
        "n_clusters": len(clusters),
        "clustering_time_s": round(t_cluster_total, 1),
        "importance_time_s": round(t_imp_total, 1),
        "total_time_s": round(t_cluster_total + t_imp_total, 1),
        "tier_counts": {k: int(v) for k, v in tier_counts.items()},
        "n_survivors": len(results["survivors"]),
    }
    with open(target_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Final upload
    upload_artifacts(s3, prefix, target_dir)
    log.info(f"DONE: {variant}/{args.target} in {t_cluster_total + t_imp_total:.0f}s total")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
