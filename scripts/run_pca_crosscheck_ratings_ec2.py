"""PCA cross-check for a single target on EC2 — RATINGS SUBSET ONLY.

Identical to run_pca_crosscheck_ec2.py but filters to the 59 rating features.

Usage:
    python3.11 scripts/run_pca_crosscheck_ratings_ec2.py --target home_win
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
import pandas as pd
from scipy.stats import weightedtau
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.strategy.config import TARGETS_CLASSIFICATION
from classical_learning.strategy.data import compute_temporal_weights, load_features
from classical_learning.analysis.feature_importance import (
    ExpandingWindowYearCV,
    build_rf,
    feat_imp_mda,
    feat_imp_mdi,
)
from classical_learning.analysis.compute import get_n_jobs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BUCKET = "mlb-265753586044-us-east-1-an"
FEATURES_KEY = "data/features/game_features.parquet"
OUTPUT_PREFIX = "classical_learning/artifacts/importance_ratings"
VARIANCE_THRESHOLD = 0.95

RATING_KEYWORDS = ("massey", "colley", "elo", "wolfe", "pythag", "srs", "log5", "consensus")


def filter_rating_features(X: pd.DataFrame) -> pd.DataFrame:
    """Filter to only columns containing a rating system keyword."""
    rating_cols = []
    for c in X.columns:
        c_check = c.replace("velo", "____")
        if any(kw in c_check for kw in RATING_KEYWORDS):
            rating_cols.append(c)
    return X[rating_cols]


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


def marchenko_pastur_bound(n_samples, n_features, eigenvalues):
    """Find signal/noise cutoff via Marchenko-Pastur law."""
    q = n_samples / n_features
    lambda_plus = (1 + 1 / np.sqrt(q)) ** 2
    k_signal = int(np.sum(eigenvalues > lambda_plus))
    return max(k_signal, 2), lambda_plus


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
    log.info(f"RATINGS SUBSET | n_jobs={n_jobs}, target={target}")

    features_path = download_features(s3)

    X, y, seasons, _ = load_features(features_path, target, "2015+")

    # Filter to ratings only
    X = filter_rating_features(X)
    log.info(f"Ratings subset: {X.shape[1]} features")

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

    # Step 2b: MDA on PCs
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

    # Step 2c: SFI on PCs
    log.info("Running SFI on PCs...")
    sfi_scores = sfi_on_pcs(X_pc, y, seasons, sample_weight, regression, n_jobs)
    sfi_ranks = pd.Series(sfi_scores).rank(ascending=False).values
    tau, p = compute_tau(eigenvalue_ranks, sfi_ranks)
    tau_results["SFI"] = {"tau": tau, "p_value": p}
    log.info(f"  SFI: tau={tau:+.4f}, p={p:.4f}")

    # Per-PC detail
    pca_info = pd.DataFrame({
        "explained_variance_ratio": var_ratios[:k],
        "eigenvalue_rank": eigenvalue_ranks,
        "mdi": pc_mdi,
        "mdi_rank": mdi_ranks,
        "mda_importance": mda_vals,
        "mda_rank": mda_ranks,
        "sfi": sfi_scores,
        "sfi_rank": sfi_ranks,
    }, index=pc_names)

    # Denoised variant (Marchenko-Pastur)
    log.info("=" * 60)
    log.info("DENOISED VARIANT (Marchenko-Pastur signal PCs only)")
    log.info("=" * 60)

    eigenvalues = pca.explained_variance_
    k_signal, lambda_plus = marchenko_pastur_bound(
        len(X_filled), X_filled.shape[1], eigenvalues
    )
    log.info(f"MP bound: lambda_+={lambda_plus:.4f}, "
             f"signal PCs={k_signal}/{X_filled.shape[1]}, "
             f"variance explained by signal={cum_var[k_signal-1]:.1%}")

    W_sig = pca.components_[:k_signal].T
    P_sig = X_filled.values @ W_sig
    pc_sig_names = [f"PC_{i}" for i in range(k_signal)]
    X_pc_sig = pd.DataFrame(P_sig, index=X.index, columns=pc_sig_names)
    eigenvalue_ranks_sig = np.arange(1, k_signal + 1, dtype=float)

    tau_denoised = {}

    log.info("Running MDI on signal PCs...")
    clf_d = build_rf(n_estimators=1000, n_jobs=n_jobs, regression=regression)
    clf_d.fit(X_pc_sig, y, sample_weight=sample_weight.values)
    mdi_d_summary, _ = feat_imp_mdi(clf_d, pc_sig_names)
    pc_mdi_d = mdi_d_summary.loc[pc_sig_names, "mean"].values
    mdi_d_ranks = pd.Series(pc_mdi_d).rank(ascending=False).values
    tau, p = compute_tau(eigenvalue_ranks_sig, mdi_d_ranks)
    tau_denoised["MDI"] = {"tau": tau, "p_value": p}
    log.info(f"  MDI (denoised): tau={tau:+.4f}, p={p:.4f}")

    log.info("Running MDA on signal PCs...")
    clf_d_mda = build_rf(n_estimators=300, n_jobs=1, regression=regression)
    mda_d_summary, _ = feat_imp_mda(
        clf_d_mda, X_pc_sig, y, seasons,
        sample_weight=sample_weight,
        scoring=scoring,
    )
    mda_d_vals = mda_d_summary.loc[pc_sig_names, "mean"].values
    mda_d_ranks = pd.Series(mda_d_vals).rank(ascending=False).values
    tau, p = compute_tau(eigenvalue_ranks_sig, mda_d_ranks)
    tau_denoised["MDA"] = {"tau": tau, "p_value": p}
    log.info(f"  MDA (denoised): tau={tau:+.4f}, p={p:.4f}")

    log.info("Running SFI on signal PCs...")
    sfi_d_scores = sfi_on_pcs(X_pc_sig, y, seasons, sample_weight, regression, n_jobs)
    sfi_d_ranks = pd.Series(sfi_d_scores).rank(ascending=False).values
    tau, p = compute_tau(eigenvalue_ranks_sig, sfi_d_ranks)
    tau_denoised["SFI"] = {"tau": tau, "p_value": p}
    log.info(f"  SFI (denoised): tau={tau:+.4f}, p={p:.4f}")

    # Upload
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "pca_cross_check.csv"
        json_path = Path(tmpdir) / "kendall_tau.json"
        loadings_path = Path(tmpdir) / "pca_loadings.npz"
        pca_info.to_csv(csv_path)

        combined_results = {
            "raw": tau_results,
            "denoised": tau_denoised,
            "mp_info": {
                "lambda_plus": float(lambda_plus),
                "k_signal": k_signal,
                "k_raw_95pct": k,
                "signal_variance_explained": float(cum_var[k_signal - 1]),
            },
        }
        with open(json_path, "w") as f:
            json.dump(combined_results, f, indent=2)
        np.savez_compressed(
            loadings_path,
            W=W,
            feature_names=np.array(valid_cols),
            pc_names=np.array(pc_names),
            explained_variance_ratio=var_ratios[:k],
        )

        prefix = f"{OUTPUT_PREFIX}/{target}"
        s3.upload_file(str(csv_path), BUCKET, f"{prefix}/pca_cross_check.csv")
        s3.upload_file(str(json_path), BUCKET, f"{prefix}/kendall_tau.json")
        s3.upload_file(str(loadings_path), BUCKET, f"{prefix}/pca_loadings.npz")
        log.info(f"Uploaded to s3://{BUCKET}/{prefix}/")

    elapsed = time.time() - t0
    log.info(f"Done: {target} in {elapsed:.1f}s")

    print(json.dumps({
        "target": target,
        "task": task,
        "n_pcs_raw": k,
        "n_pcs_signal": k_signal,
        "lambda_plus": float(lambda_plus),
        "variance_explained_raw": float(cum_var[k - 1]),
        "variance_explained_signal": float(cum_var[k_signal - 1]),
        "raw": tau_results,
        "denoised": tau_denoised,
        "elapsed_secs": round(elapsed, 1),
        "subset": "ratings",
    }, indent=2))


if __name__ == "__main__":
    main()
