"""Compute per-feature and per-cluster MDA-projected importance for total_runs.

The PCA cross-check validated MDA as structurally grounded for total_runs
(tau=+0.3803, p=0.001). This script:
  1. Recomputes PCA (same standardization as crosscheck)
  2. Loads stored per-PC MDA importance from pca_cross_check.csv
  3. Projects back to features: importance_i = sum_j |W[i,j]| * mda_importance[j]
  4. Aggregates per cluster using cluster_map.json
  5. Saves: pca_loadings.npz, pca_mda_feature_importance.csv,
            pca_mda_cluster_importance.csv

Usage:
    conda run -n pred python scripts/compute_mda_projected_importance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.strategy.config import TARGETS_CLASSIFICATION
from classical_learning.strategy.data import load_features, compute_temporal_weights

TARGET = "total_runs"
FEATURES_PATH = Path("tmp/game_features.parquet")
CROSSCHECK_DIR = Path("data/importance/total_runs")
CLUSTER_MAP_PATH = Path("pregame/artifacts/importance/total_runs/cluster_map.json")
OUTPUT_DIR = Path("data/importance/total_runs")
VARIANCE_THRESHOLD = 0.95


def main():
    # Load features (same pipeline as crosscheck)
    X, y, seasons, _ = load_features(FEATURES_PATH, TARGET, "2015+")
    nan_pct = X.isna().mean()
    valid_cols = nan_pct[nan_pct < 0.95].index.tolist()
    X = X[valid_cols]
    print(f"Loaded: {X.shape[1]} features, {len(X)} samples")

    # PCA (same standardization as crosscheck)
    X_std = (X - X.mean()) / X.std().replace(0, 1)
    X_filled = X_std.fillna(0)
    pca = PCA()
    pca.fit(X_filled.values)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum_var, VARIANCE_THRESHOLD)) + 1
    k = min(k, X_filled.shape[1])
    print(f"PCA: {k} PCs, {cum_var[k-1]:.1%} variance explained")

    W = pca.components_[:k].T  # (n_features, k)

    # Load stored MDA importance from crosscheck
    pca_info = pd.read_csv(CROSSCHECK_DIR / "pca_cross_check.csv", index_col=0)
    pc_names = [f"PC_{i}" for i in range(k)]
    mda_importance = pca_info.loc[pc_names, "mda_importance"].values  # (k,)

    # Project: importance_i = sum_j |W[i,j]| * mda_importance[j]
    abs_loadings = np.abs(W)
    feat_imp = abs_loadings @ mda_importance  # (n_features,)
    total = feat_imp.sum()
    if total > 0:
        feat_imp_normed = feat_imp / total
    else:
        feat_imp_normed = feat_imp

    feat_imp_df = pd.DataFrame({
        "mda_projected_importance": feat_imp_normed,
        "mda_projected_importance_raw": feat_imp,
        "rank": pd.Series(feat_imp_normed, index=X.columns).rank(ascending=False),
    }, index=X.columns).sort_values("rank")

    # Aggregate per cluster
    with open(CLUSTER_MAP_PATH) as f:
        cluster_map = json.load(f)

    cluster_rows = []
    for cid, members in cluster_map.items():
        members_in_X = [m for m in members if m in X.columns]
        if not members_in_X:
            continue
        cluster_imp = feat_imp_normed[X.columns.get_indexer(members_in_X)]
        cluster_rows.append({
            "cluster_id": cid,
            "n_features": len(members_in_X),
            "mda_projected_importance": cluster_imp.sum(),
            "top_feature": feat_imp_df.loc[members_in_X, "mda_projected_importance"].idxmax(),
            "top_feature_importance": feat_imp_df.loc[members_in_X, "mda_projected_importance"].max(),
        })

    cluster_df = pd.DataFrame(cluster_rows).sort_values(
        "mda_projected_importance", ascending=False
    ).reset_index(drop=True)
    cluster_df["rank"] = range(1, len(cluster_df) + 1)
    cluster_df["cumulative_importance"] = cluster_df["mda_projected_importance"].cumsum()

    # Save artifacts
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_DIR / "pca_loadings.npz",
        W=W,
        feature_names=np.array(X.columns.tolist()),
        pc_names=np.array(pc_names),
        explained_variance_ratio=pca.explained_variance_ratio_[:k],
    )
    feat_imp_df.to_csv(OUTPUT_DIR / "pca_mda_feature_importance.csv")
    cluster_df.to_csv(OUTPUT_DIR / "pca_mda_cluster_importance.csv", index=False)

    # Print summary
    print(f"\n{'='*70}")
    print("  Top 15 clusters by MDA-projected importance")
    print(f"{'='*70}")
    print(f"  {'Cluster':<12} {'Imp':>8} {'Cum':>8} {'N':>4} {'Top feature'}")
    print(f"  {'-'*65}")
    for _, row in cluster_df.head(15).iterrows():
        print(f"  {row['cluster_id']:<12} {row['mda_projected_importance']:>8.4f} "
              f"{row['cumulative_importance']:>8.4f} {row['n_features']:>4} "
              f"{row['top_feature']}")

    print(f"\n  Top 10 features by MDA-projected importance:")
    for feat, row in feat_imp_df.head(10).iterrows():
        print(f"    {int(row['rank']):>3}. {feat}: {row['mda_projected_importance']:.6f}")

    print(f"\n  Saved to {OUTPUT_DIR}/:")
    print(f"    pca_loadings.npz ({W.shape})")
    print(f"    pca_mda_feature_importance.csv ({len(feat_imp_df)} features)")
    print(f"    pca_mda_cluster_importance.csv ({len(cluster_df)} clusters)")


if __name__ == "__main__":
    main()
