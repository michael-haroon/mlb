"""PCA cross-check for a single target on EC2.

De Prado's procedure (AFML Ch.8):
  1. PCA the standardized feature matrix → k PCs (95% variance)
  2. Run MDI, MDA, SFI on the principal components
  3. Compare eigenvalue rank of PC_k to importance rank of PC_k (weighted tau)

Usage on EC2:
    python3.11 scripts/run_pca_crosscheck_ec2.py --target home_win

Reads game_features.parquet from S3, uploads results to
s3://BUCKET/artifacts/importance/{target}/kendall_tau.json and pca_cross_check.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from scipy.stats import weightedtau
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pregame.strategy.config import TARGETS_CLASSIFICATION
from pregame.strategy.data import compute_temporal_weights, load_features
from pregame.analysis.feature_importance import (
    ExpandingWindowYearCV,
    build_rf,
    feat_imp_mda,
    feat_imp_mdi,
)
from pregame.analysis.compute import get_n_jobs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BUCKET = "mlb-265753586044-us-east-1-an"
FEATURES_KEY = "artifacts/features/game_features.parquet"
OUTPUT_PREFIX = "artifacts/importance"
VARIANCE_THRESHOLD = 0.95


def compute_tau(eigenvalue_ranks, importance_ranks):
    """Weighted Kendall's tau with permutation p-value."""
    tau, _ = weightedtau(eigenvalue_ranks, importance_ranks)
    rng = np.random.default_rng(42)
    n_perm = 1000
    perm_taus = np.array([
        weightedtau(eigenvalue_ranks, rng.permutation(importance_ranks))[0]
        for _ in range(n_perm)
    ])
    p = float((np.abs(perm_taus) >= np.abs(tau)).mean())
    return float(tau) if not np.isnan(tau) else None, p


def sfi_on_pcs(X_pc, y, seasons, sample_weight, regression, n_jobs):
    """Single Feature Importance: fit model on each PC alone, expanding-window CV."""
    from sklearn.metrics import log_loss, mean_squared_error

    cv = ExpandingWindowYearCV(seasons)
    folds = list(cv.split(X_pc, y, groups=seasons.values))
    n_pcs = X_pc.shape[1]
    scores = np.zeros(n_pcs)

    for i in range(n_pcs):
        fold_scores = []
        for tr_idx, te_idx in folds:
            clf = build_rf(n_estimators=100, n_jobs=n_jobs, regression=regression)
            Xi_tr = X_pc.iloc[tr_idx, [i]]
            Xi_te = X_pc.iloc[te_idx, [i]]
            w_tr = sample_weight.iloc[tr_idx].values if sample_weight is not None else None
            clf.fit(Xi_tr, y.iloc[tr_idx], sample_weight=w_tr)
            if regression:
                pred = clf.predict(Xi_te)
                fold_scores.append(-mean_squared_error(y.iloc[te_idx], pred))
            else:
                pred = clf.predict_proba(Xi_te)
                fold_scores.append(-log_loss(y.iloc[te_idx], pred))
        scores[i] = np.mean(fold_scores)

    return scores


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

    t0 = time.time()
    s3 = boto3.client("s3")
    n_jobs = get_n_jobs()
    log.info(f"n_jobs={n_jobs}, target={target}")

    features_path = download_features(s3)

    X, y, seasons, _ = load_features(features_path, target, "2015+")

    nan_pct = X.isna().mean()
    valid_cols = nan_pct[nan_pct < 0.95].index.tolist()
    X = X[valid_cols]

    task = "classification" if target in TARGETS_CLASSIFICATION else "regression"
    regression = (task == "regression")
    sample_weight = compute_temporal_weights(seasons)
    scoring = "log_loss" if not regression else "r2"

    log.info(f"Target={target}, task={task}, features={X.shape[1]}, samples={len(X)}")

    # Step 1: PCA on correlation matrix (standardized)
    X_std = (X - X.mean()) / X.std().replace(0, 1)
    X_filled = X_std.fillna(0)
    pca = PCA()
    pca.fit(X_filled.values)
    var_ratios = pca.explained_variance_ratio_

    cum_var = np.cumsum(var_ratios)
    k = int(np.searchsorted(cum_var, VARIANCE_THRESHOLD)) + 1
    k = min(k, X_filled.shape[1])

    W = pca.components_[:k].T
    P_vals = X_filled.values @ W
    pc_names = [f"PC_{i}" for i in range(k)]
    X_pc = pd.DataFrame(P_vals, index=X.index, columns=pc_names)

    # Eigenvalue ranks: PC_0 = most variance = rank 1
    eigenvalue_ranks = np.arange(1, k + 1, dtype=float)

    log.info(f"PCA: {k} PCs, {cum_var[k-1]:.1%} variance explained")

    tau_results = {}

    # Step 2a: MDI on PCs
    log.info("Running MDI on PCs...")
    clf = build_rf(n_estimators=1000, n_jobs=n_jobs, regression=regression)
    clf.fit(X_pc, y, sample_weight=sample_weight.values)
    mdi_summary, _ = feat_imp_mdi(clf, pc_names)
    pc_mdi = mdi_summary.loc[pc_names, "mean"].values
    mdi_ranks = pd.Series(pc_mdi).rank(ascending=False).values
    tau, p = compute_tau(eigenvalue_ranks, mdi_ranks)
    tau_results["MDI"] = {"tau": tau, "p_value": p}
    log.info(f"  MDI: tau={tau:+.4f}, p={p:.4f}")

    # Step 2b: MDA on PCs (permutation importance via expanding-window CV)
    log.info("Running MDA on PCs...")
    clf_mda = build_rf(n_estimators=300, n_jobs=1, regression=regression)
    mda_summary, _ = feat_imp_mda(
        clf_mda, X_pc, y, seasons,
        sample_weight=sample_weight,
        scoring=scoring,
    )
    mda_vals = mda_summary.loc[pc_names, "mean"].values
    mda_ranks = pd.Series(mda_vals).rank(ascending=False).values
    tau, p = compute_tau(eigenvalue_ranks, mda_ranks)
    tau_results["MDA"] = {"tau": tau, "p_value": p}
    log.info(f"  MDA: tau={tau:+.4f}, p={p:.4f}")

    # Step 2c: SFI on PCs (single-feature importance per PC)
    log.info("Running SFI on PCs...")
    sfi_scores = sfi_on_pcs(X_pc, y, seasons, sample_weight, regression, n_jobs)
    sfi_ranks = pd.Series(sfi_scores).rank(ascending=False).values
    tau, p = compute_tau(eigenvalue_ranks, sfi_ranks)
    tau_results["SFI"] = {"tau": tau, "p_value": p}
    log.info(f"  SFI: tau={tau:+.4f}, p={p:.4f}")

    # Save per-PC detail
    pca_info = pd.DataFrame({
        "explained_variance_ratio": var_ratios[:k],
        "eigenvalue_rank": eigenvalue_ranks,
        "mdi": pc_mdi,
        "mdi_rank": mdi_ranks,
        "mda": mda_vals,
        "mda_rank": mda_ranks,
        "sfi": sfi_scores,
        "sfi_rank": sfi_ranks,
    }, index=pc_names)

    # Upload
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "pca_cross_check.csv"
        json_path = Path(tmpdir) / "kendall_tau.json"
        pca_info.to_csv(csv_path)
        with open(json_path, "w") as f:
            json.dump(tau_results, f, indent=2)

        prefix = f"{OUTPUT_PREFIX}/{target}"
        s3.upload_file(str(csv_path), BUCKET, f"{prefix}/pca_cross_check.csv")
        s3.upload_file(str(json_path), BUCKET, f"{prefix}/kendall_tau.json")
        log.info(f"Uploaded to s3://{BUCKET}/{prefix}/")

    elapsed = time.time() - t0
    log.info(f"Done: {target} in {elapsed:.1f}s")

    # Print summary
    print(json.dumps({
        "target": target,
        "task": task,
        "n_pcs": k,
        "variance_explained": float(cum_var[k-1]),
        "tau_results": tau_results,
        "elapsed_secs": round(elapsed, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
