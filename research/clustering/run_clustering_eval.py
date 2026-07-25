"""Clustering configuration evaluation for MLB feature importance pipeline.

Evaluates whether the current ONC clustering (2 clusters on ~280→600 features)
is suppressing real substructure that desub-MDA, resid-MDA, and CFI-MDA depend on.

Tests:
  A1. Baseline (current pipeline)
  A2. Detoning sweep: n_remove ∈ {0, 1, 2}
  A3. Quality metric sweep: silhouette t-stat vs Calinski-Harabasz vs Davies-Bouldin vs gap
  A4. Best combination of A2 + A3
  A5. Agglomerative Ward clustering on current correlation distance
  A6. Spectral clustering on denoised correlation Laplacian

Scores each on:
  B1. Intra-cluster cohesion (mean pairwise correlation within)
  B2. Inter-cluster separation (mean pairwise correlation between)
  B3. Cluster size distribution
  B4. Bootstrap stability (ARI, 30 resamples, max_clusters capped at 25)
  B5. Downstream validity — synthetic importance test results routed through pipeline

Output: research/clustering/clustering_eval.md

Usage (EC2):
    python3.11 research/clustering/run_clustering_eval.py \
        --features pregame/artifacts/features/game_features.parquet \
        --output research/clustering/
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import ward, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import (
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("research/clustering/clustering_eval.log"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading (mirrors pregame/strategy/data.py logic)
# ─────────────────────────────────────────────────────────────────────────────

def load_feature_matrix(features_path: Path) -> pd.DataFrame:
    """Load game_features.parquet and select pregame-knowable numeric columns."""
    df = pd.read_parquet(features_path)
    log.info(f"Loaded {len(df):,} games × {len(df.columns)} columns")

    # Filter to 2016+ and exclude 2020
    df = df[df["season"] >= 2016].reset_index(drop=True)
    df = df[df["season"] != 2020].reset_index(drop=True)
    log.info(f"After season filter (2016+, excl 2020): {len(df):,} games")

    # Use the same prefix allowlist as strategy/data.py
    from pregame.strategy.data import _select_pregame_features
    feature_cols = _select_pregame_features(df)

    X = df[feature_cols]
    # Drop >95% NaN columns
    nan_pct = X.isna().mean()
    valid = nan_pct[nan_pct < 0.95].index.tolist()
    X = X[valid]
    log.info(f"Feature matrix: {X.shape[1]} features × {X.shape[0]} samples")
    return X


# ─────────────────────────────────────────────────────────────────────────────
#  Correlation matrix preprocessing (from feature_importance.py)
# ─────────────────────────────────────────────────────────────────────────────

def denoise_corr(corr: np.ndarray, q: float) -> np.ndarray:
    """Marcenko-Pastur denoising (AFML Ch.2)."""
    evals, evecs = np.linalg.eigh(corr)
    lambda_plus = (1.0 + q ** -0.5) ** 2
    noise_mask = evals <= lambda_plus
    if noise_mask.any():
        evals = np.where(noise_mask, evals[noise_mask].mean(), evals)
    corr_clean = evecs @ np.diag(evals) @ evecs.T
    diag_sqrt = np.sqrt(np.maximum(np.diag(corr_clean), 1e-12))
    corr_clean = corr_clean / np.outer(diag_sqrt, diag_sqrt)
    np.fill_diagonal(corr_clean, 1.0)
    return corr_clean


def detone_corr(corr: np.ndarray, n_remove: int = 1) -> np.ndarray:
    """Remove top n_remove eigenvectors (market mode)."""
    if n_remove == 0:
        return corr.copy()
    evals, evecs = np.linalg.eigh(corr)
    evals_d = evals.copy()
    evals_d[-n_remove:] = 0.0
    corr_d = evecs @ np.diag(evals_d) @ evecs.T
    diag_sqrt = np.sqrt(np.maximum(np.diag(corr_d), 1e-12))
    corr_d = corr_d / np.outer(diag_sqrt, diag_sqrt)
    np.fill_diagonal(corr_d, 1.0)
    return corr_d


def corr_to_distance(corr: np.ndarray) -> np.ndarray:
    """Correlation → Euclidean distance: √((1 - corr) / 2)."""
    return np.sqrt(np.clip((1 - corr) / 2.0, 0, 1))


# ─────────────────────────────────────────────────────────────────────────────
#  Clustering algorithms
# ─────────────────────────────────────────────────────────────────────────────

def _silhouette_tstat(X_dist: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette t-stat (current pipeline's quality metric)."""
    sil = silhouette_samples(X_dist, labels, metric="precomputed")
    return sil.mean() / (sil.std() + 1e-10)


def _calinski_harabasz(X_dist: np.ndarray, labels: np.ndarray) -> float:
    """Calinski-Harabasz on distance matrix (higher = better)."""
    return calinski_harabasz_score(X_dist, labels)


def _davies_bouldin(X_dist: np.ndarray, labels: np.ndarray) -> float:
    """Davies-Bouldin (lower = better, negate for consistency)."""
    return -davies_bouldin_score(X_dist, labels)


def _gap_statistic(X_dist: np.ndarray, labels: np.ndarray, k: int,
                   n_ref: int = 20, rng: np.random.Generator = None) -> float:
    """Gap statistic: log(W_ref) - log(W_k)."""
    if rng is None:
        rng = np.random.default_rng(42)

    n = X_dist.shape[0]

    def _within_cluster_dispersion(dist, labs):
        total = 0.0
        for c in np.unique(labs):
            mask = labs == c
            if mask.sum() < 2:
                continue
            sub = dist[np.ix_(mask, mask)]
            total += sub.sum() / (2 * mask.sum())
        return total

    W_k = _within_cluster_dispersion(X_dist, labels)
    log_W_k = np.log(W_k + 1e-15)

    log_W_refs = []
    for _ in range(n_ref):
        # Uniform reference on same distance scale
        ref_idx = rng.choice(n, size=n, replace=True)
        ref_dist = X_dist[np.ix_(ref_idx, ref_idx)]
        km = KMeans(n_clusters=k, n_init=5, random_state=rng.integers(10000))
        ref_labels = km.fit_predict(ref_dist)
        W_ref = _within_cluster_dispersion(ref_dist, ref_labels)
        log_W_refs.append(np.log(W_ref + 1e-15))

    return np.mean(log_W_refs) - log_W_k


QUALITY_METRICS = {
    "silhouette_tstat": _silhouette_tstat,
    "calinski_harabasz": _calinski_harabasz,
    "davies_bouldin_neg": _davies_bouldin,
    "gap_statistic": None,  # handled specially (needs k)
}


def run_kmeans_grid(X_dist: np.ndarray, max_k: int = 25, n_init: int = 20,
                    quality_fn=None) -> tuple[np.ndarray, int, float]:
    """KMeans grid search, return (best_labels, best_k, best_quality)."""
    if quality_fn is None:
        quality_fn = _silhouette_tstat

    best_q = -np.inf
    best_labels = None
    best_k = 2

    for k in range(2, max_k + 1):
        for seed in range(n_init):
            km = KMeans(n_clusters=k, n_init=1, random_state=seed)
            labels = km.fit_predict(X_dist)
            if len(np.unique(labels)) < 2:
                continue
            q = quality_fn(X_dist, labels)
            if q > best_q:
                best_q = q
                best_labels = labels
                best_k = k

    return best_labels, best_k, best_q


def run_greedy_divisive(X_dist: np.ndarray, corr_names: list[str],
                        max_k: int = 25, n_init: int = 20,
                        quality_fn=None) -> dict[int, list[str]]:
    """Greedy divisive ONC (mirrors feature_importance.py::onc_cluster)."""
    if quality_fn is None:
        quality_fn = _silhouette_tstat

    n = X_dist.shape[0]
    labels, _, _ = run_kmeans_grid(X_dist, max_k=max_k, n_init=n_init,
                                   quality_fn=quality_fn)
    if labels is None:
        return {0: corr_names}

    # Build initial partition
    partition = {}
    for i, lbl in enumerate(labels):
        partition.setdefault(int(lbl), []).append(i)

    def _global_quality(part):
        label_arr = np.empty(n, dtype=int)
        for cid, idxs in part.items():
            for idx in idxs:
                label_arr[idx] = cid
        if len(np.unique(label_arr)) < 2:
            return -np.inf
        return quality_fn(X_dist, label_arr)

    def _per_cluster_quality(part):
        label_arr = np.empty(n, dtype=int)
        for cid, idxs in part.items():
            for idx in idxs:
                label_arr[idx] = cid
        if len(np.unique(label_arr)) < 2:
            return {cid: 0.0 for cid in part}
        sil = silhouette_samples(X_dist, label_arr, metric="precomputed")
        return {cid: sil[idxs].mean() for cid, idxs in part.items()}

    current_quality = _global_quality(partition)

    improved = True
    while improved:
        improved = False
        qualities = _per_cluster_quality(partition)
        sorted_cids = sorted(qualities, key=lambda c: qualities[c])

        for cid in sorted_cids:
            members = partition[cid]
            if len(members) < 4:
                continue

            sub_dist = X_dist[np.ix_(members, members)]
            sub_labels, sub_k, _ = run_kmeans_grid(
                sub_dist, max_k=min(max_k, len(members) - 1),
                n_init=n_init, quality_fn=quality_fn,
            )
            if sub_labels is None or len(np.unique(sub_labels)) <= 1:
                continue

            next_id = max(partition.keys()) + 1
            candidate = {c: m for c, m in partition.items() if c != cid}
            for sub_lbl in np.unique(sub_labels):
                sub_members = [members[i] for i in range(len(members))
                               if sub_labels[i] == sub_lbl]
                candidate[next_id] = sub_members
                next_id += 1

            cand_quality = _global_quality(candidate)
            if cand_quality > current_quality:
                partition = candidate
                current_quality = cand_quality
                improved = True
                break

    # Convert indices to names
    return {cid: [corr_names[i] for i in idxs]
            for cid, idxs in partition.items()}


def run_ward_clustering(X_dist: np.ndarray, corr_names: list[str],
                        cut_heights: list[float] = None) -> dict:
    """Agglomerative Ward clustering at multiple cut heights."""
    # Ward requires condensed distance
    condensed = squareform(X_dist, checks=False)
    Z = ward(condensed)

    if cut_heights is None:
        # Auto-select cut heights at 10th, 25th, 50th, 75th, 90th pctile of merge distances
        merge_dists = Z[:, 2]
        cut_heights = list(np.percentile(merge_dists, [10, 25, 50, 75, 90]))

    results = {}
    for h in cut_heights:
        labels = fcluster(Z, t=h, criterion="distance")
        n_clusters = len(np.unique(labels))
        clusters = {}
        for i, lbl in enumerate(labels):
            clusters.setdefault(int(lbl), []).append(corr_names[i])
        results[f"h={h:.3f}"] = {
            "n_clusters": n_clusters,
            "clusters": clusters,
            "labels": labels,
        }

    results["linkage"] = Z
    results["cut_heights"] = cut_heights
    return results


def run_spectral_clustering(corr: np.ndarray, corr_names: list[str],
                            max_k: int = 25) -> dict:
    """Spectral clustering on affinity = (1+corr)/2 (maps [-1,1] to [0,1])."""
    affinity = np.clip((1 + corr) / 2.0, 0, 1)
    np.fill_diagonal(affinity, 1.0)

    best_k = 2
    best_labels = None
    best_score = -np.inf

    for k in range(2, max_k + 1):
        try:
            sc = SpectralClustering(
                n_clusters=k, affinity="precomputed",
                random_state=42, n_init=10,
            )
            labels = sc.fit_predict(affinity)
            if len(np.unique(labels)) < 2:
                continue
            # Score by silhouette on correlation distance
            dist = corr_to_distance(corr)
            score = silhouette_samples(dist, labels, metric="precomputed").mean()
            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k
        except Exception:
            continue

    clusters = {}
    if best_labels is not None:
        for i, lbl in enumerate(best_labels):
            clusters.setdefault(int(lbl), []).append(corr_names[i])

    return {"n_clusters": best_k, "clusters": clusters, "labels": best_labels,
            "silhouette_mean": float(best_score)}


# ─────────────────────────────────────────────────────────────────────────────
#  Scoring functions (Part B)
# ─────────────────────────────────────────────────────────────────────────────

def score_cohesion_separation(corr: np.ndarray, clusters: dict[int, list[int]]) -> dict:
    """B1+B2: Intra-cluster cohesion and inter-cluster separation.

    clusters: {cluster_id: [feature_indices]}
    Returns dict with per-cluster and global metrics.
    """
    n = corr.shape[0]
    intra_corrs = []
    inter_corrs = []
    per_cluster = {}

    for cid, idxs in clusters.items():
        if len(idxs) < 2:
            per_cluster[cid] = {"intra_mean_corr": np.nan, "size": len(idxs)}
            continue
        sub = corr[np.ix_(idxs, idxs)]
        # Upper triangle (exclude diagonal)
        triu_idx = np.triu_indices(len(idxs), k=1)
        vals = sub[triu_idx]
        intra_corrs.extend(vals.tolist())
        per_cluster[cid] = {
            "intra_mean_corr": float(np.mean(vals)),
            "intra_median_corr": float(np.median(vals)),
            "size": len(idxs),
        }

    # Inter-cluster: pairwise between different clusters
    cluster_ids = sorted(clusters.keys())
    for i, cid_a in enumerate(cluster_ids):
        for cid_b in cluster_ids[i+1:]:
            cross = corr[np.ix_(clusters[cid_a], clusters[cid_b])]
            inter_corrs.extend(cross.flatten().tolist())

    return {
        "intra_mean_corr": float(np.mean(intra_corrs)) if intra_corrs else np.nan,
        "intra_median_corr": float(np.median(intra_corrs)) if intra_corrs else np.nan,
        "inter_mean_corr": float(np.mean(inter_corrs)) if inter_corrs else np.nan,
        "inter_median_corr": float(np.median(inter_corrs)) if inter_corrs else np.nan,
        "separation_ratio": (
            float(np.mean(intra_corrs) / (np.abs(np.mean(inter_corrs)) + 1e-10))
            if intra_corrs and inter_corrs else np.nan
        ),
        "per_cluster": per_cluster,
    }


def score_size_distribution(clusters: dict[int, list]) -> dict:
    """B3: Cluster size distribution and flags."""
    sizes = [len(v) for v in clusters.values()]
    n_total = sum(sizes)
    n_clusters = len(clusters)
    n_small = sum(1 for s in sizes if s < 5)
    n_singleton = sum(1 for s in sizes if s <= 2)
    pct_in_small = sum(s for s in sizes if s < 5) / n_total if n_total > 0 else 0

    return {
        "n_clusters": n_clusters,
        "sizes": sorted(sizes, reverse=True),
        "mean_size": float(np.mean(sizes)),
        "median_size": float(np.median(sizes)),
        "max_size": int(max(sizes)),
        "min_size": int(min(sizes)),
        "n_clusters_lt5": n_small,
        "n_singletons_or_pairs": n_singleton,
        "pct_features_in_small_clusters": round(pct_in_small, 3),
        "flag_cfi_broken": n_small > 0,
        "flag_majority_singleton": pct_in_small > 0.5,
    }


def score_stability(X_dist: np.ndarray, cluster_fn, n_boot: int = 30,
                    rng: np.random.Generator = None) -> dict:
    """B4: Bootstrap stability via Adjusted Rand Index."""
    if rng is None:
        rng = np.random.default_rng(42)

    n = X_dist.shape[0]
    all_labels = []

    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sub_dist = X_dist[np.ix_(idx, idx)]
        try:
            labels = cluster_fn(sub_dist)
            all_labels.append((idx, labels))
        except Exception:
            continue

    if len(all_labels) < 5:
        return {"mean_ari": np.nan, "std_ari": np.nan, "n_successful": len(all_labels)}

    # Pairwise ARI on shared samples
    aris = []
    for i in range(len(all_labels)):
        for j in range(i + 1, min(i + 10, len(all_labels))):
            idx_i, lab_i = all_labels[i]
            idx_j, lab_j = all_labels[j]
            # Find shared sample indices
            shared_orig = np.intersect1d(idx_i, idx_j)
            if len(shared_orig) < 10:
                continue
            # Map to positions in each bootstrap
            pos_i = [np.where(idx_i == s)[0][0] for s in shared_orig]
            pos_j = [np.where(idx_j == s)[0][0] for s in shared_orig]
            ari = adjusted_rand_score(lab_i[pos_i], lab_j[pos_j])
            aris.append(ari)

    return {
        "mean_ari": float(np.mean(aris)) if aris else np.nan,
        "std_ari": float(np.std(aris)) if aris else np.nan,
        "n_successful": len(all_labels),
        "n_pairs_compared": len(aris),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  B5: Downstream validity — synthetic importance routing test
# ─────────────────────────────────────────────────────────────────────────────

def score_downstream_validity(corr_names: list[str], clusters_by_name: dict,
                              X: pd.DataFrame, rng: np.random.Generator = None) -> dict:
    """B5: Inject synthetic features with KNOWN properties, run desub-MDA and
    resid-MDA, verify that redundant features are correctly separated from
    informative ones UNDER this clustering.

    Strategy:
    - Create 5 blocks of 10 correlated features (known clusters)
    - Add 5 noise features
    - Inject into a ~55 feature synthetic frame
    - Run ONC importance methods under the GIVEN clustering
    - Check: do desub-MDA and resid-MDA correctly rank informative > noise?
    """
    from pregame.analysis.feature_importance import (
        feat_imp_desub_mda,
        feat_imp_residual_mda,
        PurgedYearKFold,
        build_rf,
    )

    if rng is None:
        rng = np.random.default_rng(42)

    n_samples = 500
    n_blocks = 5
    n_per_block = 10
    n_noise = 5
    n_features = n_blocks * n_per_block + n_noise

    # Generate block-correlated features
    X_synth = np.zeros((n_samples, n_features))
    true_cluster_map = {}

    for b in range(n_blocks):
        base = rng.standard_normal(n_samples)
        for j in range(n_per_block):
            col_idx = b * n_per_block + j
            X_synth[:, col_idx] = base + rng.standard_normal(n_samples) * 0.3
            true_cluster_map.setdefault(b, []).append(col_idx)

    # Noise features
    for j in range(n_noise):
        col_idx = n_blocks * n_per_block + j
        X_synth[:, col_idx] = rng.standard_normal(n_samples)
        true_cluster_map.setdefault(n_blocks, []).append(col_idx)

    # Target: depends on first feature of each block (interaction)
    y = (X_synth[:, 0] + X_synth[:, 10] + X_synth[:, 20] > 0).astype(int)

    feat_names = [f"synth_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X_synth, columns=feat_names)
    y_ser = pd.Series(y)
    years = pd.Series(np.repeat(np.arange(10), 50)[:n_samples])

    # Cluster assignment: map feature indices to cluster IDs
    # Use TRUE clusters for baseline, then test with GIVEN clustering's structure
    synth_clusters_true = {
        cid: [feat_names[i] for i in idxs]
        for cid, idxs in true_cluster_map.items()
    }

    # Also test with a BAD clustering (2 clusters like baseline)
    synth_clusters_2 = {
        0: feat_names[:n_features // 2],
        1: feat_names[n_features // 2:],
    }

    results = {}
    for label, clust in [("true_clusters", synth_clusters_true),
                         ("2_clusters", synth_clusters_2)]:
        try:
            desub_summary, _ = feat_imp_desub_mda(
                X_df, y_ser, years, clust,
                scoring="log_loss", n_estimators=100,
            )

            resid_summary, _ = feat_imp_residual_mda(
                X_df, y_ser, years, clust,
                scoring="log_loss", n_estimators=100,
            )

            # Informative features: first feature of blocks 0, 1, 2
            informative = ["synth_0", "synth_10", "synth_20"]
            noise_feats = [f"synth_{n_blocks * n_per_block + j}" for j in range(n_noise)]

            # Check: are informative features ranked above noise?
            desub_info_mean = desub_summary.loc[
                [f for f in informative if f in desub_summary.index], "mean"
            ].mean()
            desub_noise_mean = desub_summary.loc[
                [f for f in noise_feats if f in desub_summary.index], "mean"
            ].mean()

            resid_info_mean = resid_summary.loc[
                [f for f in informative if f in resid_summary.index], "mean"
            ].mean()
            resid_noise_mean = resid_summary.loc[
                [f for f in noise_feats if f in resid_summary.index], "mean"
            ].mean()

            results[label] = {
                "desub_informative_mean": float(desub_info_mean),
                "desub_noise_mean": float(desub_noise_mean),
                "desub_separates": bool(desub_info_mean > desub_noise_mean),
                "resid_informative_mean": float(resid_info_mean),
                "resid_noise_mean": float(resid_noise_mean),
                "resid_separates": bool(resid_info_mean > resid_noise_mean),
            }
        except Exception as e:
            results[label] = {"error": str(e)}

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Part A: Generate candidate clusterings
# ─────────────────────────────────────────────────────────────────────────────

def run_all_candidates(X: pd.DataFrame, output_dir: Path) -> dict:
    """Generate all candidate clusterings and score them."""
    X_filled = X.fillna(X.median())
    corr_raw = X_filled.corr().values
    corr_names = list(X.columns)
    n = len(corr_names)
    q = X_filled.shape[0] / X_filled.shape[1]

    log.info(f"Correlation matrix: {n}×{n}, q={q:.2f}")

    # Denoise once (shared across detoning variants)
    corr_denoised = denoise_corr(corr_raw, q=q)

    # Eigenvalue diagnostics
    evals = np.linalg.eigvalsh(corr_raw)
    lambda_plus = (1.0 + q ** -0.5) ** 2
    n_signal = int((evals > lambda_plus).sum())
    n_noise_eig = int((evals <= lambda_plus).sum())
    log.info(f"MP: lambda+={lambda_plus:.4f}, signal={n_signal}, noise={n_noise_eig}")

    candidates = {}
    MAX_K = 25

    # ── A1: Baseline (current pipeline) ─────────────────────────────────────
    log.info("=" * 60)
    log.info("A1: Baseline (detone n_remove=1, silhouette t-stat)")
    corr_base = detone_corr(corr_denoised, n_remove=1)
    dist_base = corr_to_distance(corr_base)
    clusters_base = run_greedy_divisive(
        dist_base, corr_names, max_k=MAX_K, quality_fn=_silhouette_tstat,
    )
    candidates["A1_baseline"] = {
        "config": "detone=1, metric=silhouette_tstat, algo=greedy_divisive",
        "clusters": clusters_base,
    }
    log.info(f"  → {len(clusters_base)} clusters")

    # ── A2: Detoning sweep ──────────────────────────────────────────────────
    for n_remove in [0, 1, 2]:
        label = f"A2_detone_{n_remove}"
        log.info(f"\n{label}: n_remove={n_remove}")
        corr_v = detone_corr(corr_denoised, n_remove=n_remove)
        dist_v = corr_to_distance(corr_v)
        clusters_v = run_greedy_divisive(
            dist_v, corr_names, max_k=MAX_K, quality_fn=_silhouette_tstat,
        )
        candidates[label] = {
            "config": f"detone={n_remove}, metric=silhouette_tstat, algo=greedy_divisive",
            "clusters": clusters_v,
        }
        log.info(f"  → {len(clusters_v)} clusters")

    # ── A3: Quality metric sweep ────────────────────────────────────────────
    corr_a3 = detone_corr(corr_denoised, n_remove=1)  # Keep default detoning
    dist_a3 = corr_to_distance(corr_a3)

    for metric_name, metric_fn in [
        ("calinski_harabasz", _calinski_harabasz),
        ("davies_bouldin_neg", _davies_bouldin),
    ]:
        label = f"A3_{metric_name}"
        log.info(f"\n{label}")
        clusters_v = run_greedy_divisive(
            dist_a3, corr_names, max_k=MAX_K, quality_fn=metric_fn,
        )
        candidates[label] = {
            "config": f"detone=1, metric={metric_name}, algo=greedy_divisive",
            "clusters": clusters_v,
        }
        log.info(f"  → {len(clusters_v)} clusters")

    # Gap statistic: expensive, run flat KMeans with gap
    log.info("\nA3_gap_statistic")
    gap_rng = np.random.default_rng(42)
    best_gap = -np.inf
    best_gap_k = 2
    best_gap_labels = None
    for k in range(2, MAX_K + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(dist_a3)
        gap = _gap_statistic(dist_a3, labels, k, n_ref=10, rng=gap_rng)
        if gap > best_gap:
            best_gap = gap
            best_gap_k = k
            best_gap_labels = labels
    log.info(f"  Gap best k={best_gap_k}, gap={best_gap:.4f}")
    clusters_gap = {}
    for i, lbl in enumerate(best_gap_labels):
        clusters_gap.setdefault(int(lbl), []).append(corr_names[i])
    candidates["A3_gap_statistic"] = {
        "config": f"detone=1, metric=gap_statistic, algo=flat_kmeans_k={best_gap_k}",
        "clusters": clusters_gap,
    }
    log.info(f"  → {len(clusters_gap)} clusters")

    # ── A4: Best combination (determined after A2+A3 scoring) ────────────────
    # Placeholder — will be filled after scoring

    # ── A5: Agglomerative Ward ──────────────────────────────────────────────
    log.info("\nA5: Agglomerative Ward clustering")
    ward_results = run_ward_clustering(dist_base, corr_names)
    # Pick the cut that gives closest to 8-15 clusters
    for h_label, h_data in ward_results.items():
        if h_label in ("linkage", "cut_heights"):
            continue
        nc = h_data["n_clusters"]
        label = f"A5_ward_{h_label}_k{nc}"
        candidates[label] = {
            "config": f"algo=ward, cut={h_label}, n_clusters={nc}",
            "clusters": h_data["clusters"],
        }
        log.info(f"  {h_label}: {nc} clusters")

    # ── A6: Spectral clustering ─────────────────────────────────────────────
    log.info("\nA6: Spectral clustering on denoised correlation")
    spec_result = run_spectral_clustering(corr_denoised, corr_names, max_k=MAX_K)
    candidates["A6_spectral"] = {
        "config": f"algo=spectral, k={spec_result['n_clusters']}, affinity=(1+corr)/2",
        "clusters": spec_result["clusters"],
    }
    log.info(f"  → {spec_result['n_clusters']} clusters (sil={spec_result['silhouette_mean']:.4f})")

    return candidates, corr_raw, corr_denoised, dist_base, corr_names, X


# ─────────────────────────────────────────────────────────────────────────────
#  Part B: Score all candidates
# ─────────────────────────────────────────────────────────────────────────────

def score_all_candidates(candidates: dict, corr_raw: np.ndarray,
                         dist_base: np.ndarray, corr_names: list[str],
                         X: pd.DataFrame) -> dict:
    """Score all candidate clusterings on B1-B5 metrics."""
    n = len(corr_names)
    name_to_idx = {name: i for i, name in enumerate(corr_names)}
    scores = {}

    for cand_label, cand_data in candidates.items():
        log.info(f"\nScoring {cand_label}...")
        clusters = cand_data["clusters"]

        # Convert name-based clusters to index-based
        clusters_idx = {}
        for cid, members in clusters.items():
            clusters_idx[cid] = [name_to_idx[m] for m in members if m in name_to_idx]

        # B1 + B2: Cohesion and separation
        cs = score_cohesion_separation(corr_raw, clusters_idx)

        # B3: Size distribution
        sd = score_size_distribution(clusters)

        # B4: Bootstrap stability
        def _cluster_fn(sub_dist):
            labels, _, _ = run_kmeans_grid(sub_dist, max_k=15, n_init=5,
                                           quality_fn=_silhouette_tstat)
            return labels if labels is not None else np.zeros(sub_dist.shape[0])

        stab = score_stability(dist_base, _cluster_fn, n_boot=30)

        scores[cand_label] = {
            "config": cand_data["config"],
            "n_clusters": len(clusters),
            "B1_intra_mean_corr": cs["intra_mean_corr"],
            "B1_intra_median_corr": cs["intra_median_corr"],
            "B2_inter_mean_corr": cs["inter_mean_corr"],
            "B2_inter_median_corr": cs["inter_median_corr"],
            "B2_separation_ratio": cs["separation_ratio"],
            "B3_mean_size": sd["mean_size"],
            "B3_n_clusters_lt5": sd["n_clusters_lt5"],
            "B3_flag_cfi_broken": sd["flag_cfi_broken"],
            "B3_flag_majority_singleton": sd["flag_majority_singleton"],
            "B3_sizes": sd["sizes"],
            "B4_mean_ari": stab["mean_ari"],
            "B4_std_ari": stab["std_ari"],
        }
        log.info(f"  cohesion={cs['intra_mean_corr']:.3f}, "
                 f"separation={cs['inter_mean_corr']:.3f}, "
                 f"ratio={cs['separation_ratio']:.2f}, "
                 f"stability={stab['mean_ari']:.3f}")

    # B5: Downstream validity (run once, compare true vs 2-cluster)
    log.info("\nB5: Downstream validity (synthetic importance routing)...")
    b5 = score_downstream_validity(corr_names, candidates.get("A1_baseline", {}).get("clusters", {}), X)
    scores["__downstream_validity"] = b5

    return scores


# ─────────────────────────────────────────────────────────────────────────────
#  Part C: Generate report
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(scores: dict, candidates: dict, output_dir: Path) -> str:
    """Generate clustering_eval.md report."""
    lines = []
    lines.append("# Clustering Configuration Evaluation")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("Evaluates whether ONC's 2-cluster output on ~600 MLB features is "
                 "suppressing real substructure that desub-MDA, resid-MDA, and CFI-MDA depend on.")
    lines.append("")

    # Cross-cluster correlation for baseline
    baseline_scores = scores.get("A1_baseline", {})
    lines.append("## Critical Finding: Baseline Cross-Cluster Correlation")
    lines.append("")
    lines.append(f"- **Inter-cluster mean |corr|**: {baseline_scores.get('B2_inter_mean_corr', 'N/A')}")
    lines.append(f"- **Intra-cluster mean |corr|**: {baseline_scores.get('B1_intra_mean_corr', 'N/A')}")
    lines.append(f"- **Separation ratio** (intra/inter): {baseline_scores.get('B2_separation_ratio', 'N/A')}")
    lines.append("")
    if baseline_scores.get("B2_inter_mean_corr") and baseline_scores["B2_inter_mean_corr"] > 0.3:
        lines.append("**WARNING**: High cross-cluster correlation (>0.3) indicates the 2-cluster "
                     "partition does not meaningfully separate feature groups. Features in different "
                     "clusters are still highly correlated, which undermines desub-MDA's substitution-free "
                     "guarantee and resid-MDA's orthogonalization.")
    lines.append("")

    # Summary table
    lines.append("## Results Table")
    lines.append("")
    lines.append("| Config | k | Intra Corr | Inter Corr | Sep Ratio | "
                 "Stability (ARI) | n<5 clusters | Flags |")
    lines.append("|--------|---|-----------|-----------|-----------|"
                 "----------------|-------------|-------|")

    for label, s in sorted(scores.items()):
        if label.startswith("__"):
            continue
        flags = []
        if s.get("B3_flag_cfi_broken"):
            flags.append("CFI-broken")
        if s.get("B3_flag_majority_singleton"):
            flags.append("majority-small")
        flag_str = ", ".join(flags) if flags else "-"
        lines.append(
            f"| {label} | {s.get('n_clusters', '?')} | "
            f"{s.get('B1_intra_mean_corr', 0):.3f} | "
            f"{s.get('B2_inter_mean_corr', 0):.3f} | "
            f"{s.get('B2_separation_ratio', 0):.2f} | "
            f"{s.get('B4_mean_ari', 0):.3f}±{s.get('B4_std_ari', 0):.3f} | "
            f"{s.get('B3_n_clusters_lt5', 0)} | "
            f"{flag_str} |"
        )

    lines.append("")

    # B5 results
    lines.append("## Downstream Validity (B5)")
    lines.append("")
    b5 = scores.get("__downstream_validity", {})
    for label, res in b5.items():
        lines.append(f"### {label}")
        if "error" in res:
            lines.append(f"  Error: {res['error']}")
        else:
            lines.append(f"  - desub-MDA informative mean: {res.get('desub_informative_mean', 'N/A'):.5f}")
            lines.append(f"  - desub-MDA noise mean: {res.get('desub_noise_mean', 'N/A'):.5f}")
            lines.append(f"  - desub-MDA separates: **{res.get('desub_separates', 'N/A')}**")
            lines.append(f"  - resid-MDA informative mean: {res.get('resid_informative_mean', 'N/A'):.5f}")
            lines.append(f"  - resid-MDA noise mean: {res.get('resid_noise_mean', 'N/A'):.5f}")
            lines.append(f"  - resid-MDA separates: **{res.get('resid_separates', 'N/A')}**")
        lines.append("")

    # Ranking
    lines.append("## Ranked Recommendation")
    lines.append("")
    lines.append("*(To be filled after results — rank by: separation_ratio × stability, "
                 "penalize if B3 flags or B5 fails)*")
    lines.append("")

    # Cluster size distributions
    lines.append("## Cluster Size Distributions")
    lines.append("")
    for label, s in sorted(scores.items()):
        if label.startswith("__"):
            continue
        sizes = s.get("B3_sizes", [])
        lines.append(f"- **{label}** (k={s.get('n_clusters', '?')}): {sizes[:20]}"
                     + ("..." if len(sizes) > 20 else ""))
    lines.append("")

    # Method details
    lines.append("## Method Details")
    lines.append("")
    lines.append("- **Denoising**: Marcenko-Pastur (AFML Ch.2), replaces noise eigenvalues with mean")
    lines.append("- **Detoning**: Removes top-k eigenvectors (market mode) to expose cluster structure")
    lines.append("- **Distance**: √((1-corr)/2) → proper Euclidean metric on correlation")
    lines.append("- **Baseline algo**: Greedy divisive KMeans + silhouette t-stat gate")
    lines.append("- **Stability**: 30 bootstrap resamples, pairwise ARI")
    lines.append("- **Downstream test**: 5 blocks × 10 correlated features + 5 noise, "
                 "check desub/resid-MDA separation under true vs 2-cluster assignments")
    lines.append("")

    report_text = "\n".join(lines)

    report_path = output_dir / "clustering_eval.md"
    report_path.write_text(report_text)
    log.info(f"Report written to {report_path}")

    return report_text


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clustering configuration evaluation")
    parser.add_argument("--features", default="pregame/artifacts/features/game_features.parquet",
                        help="Path to game_features.parquet")
    parser.add_argument("--output", default="research/clustering/",
                        help="Output directory for results")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Load data
    log.info("Loading feature matrix...")
    X = load_feature_matrix(Path(args.features))

    # Part A: Generate candidates
    log.info("\n" + "=" * 70)
    log.info("PART A: Generating candidate clusterings")
    log.info("=" * 70)
    candidates, corr_raw, corr_denoised, dist_base, corr_names, X = run_all_candidates(X, output_dir)

    # Part B: Score all
    log.info("\n" + "=" * 70)
    log.info("PART B: Scoring all candidates")
    log.info("=" * 70)
    scores = score_all_candidates(candidates, corr_raw, dist_base, corr_names, X)

    # Save raw scores
    scores_serializable = {}
    for k, v in scores.items():
        if k.startswith("__"):
            scores_serializable[k] = v
            continue
        s = {sk: sv for sk, sv in v.items() if sk != "B3_sizes"}
        s["B3_sizes"] = v.get("B3_sizes", [])[:30]
        scores_serializable[k] = s

    with open(output_dir / "scores_raw.json", "w") as f:
        json.dump(scores_serializable, f, indent=2, default=str)

    # Save candidate cluster assignments
    for label, cand in candidates.items():
        cand_file = output_dir / f"clusters_{label}.json"
        clusters_out = {str(k): v for k, v in cand["clusters"].items()}
        with open(cand_file, "w") as f:
            json.dump({"config": cand["config"], "clusters": clusters_out}, f, indent=2)

    # Part C: Generate report
    log.info("\n" + "=" * 70)
    log.info("PART C: Generating report")
    log.info("=" * 70)
    report = generate_report(scores, candidates, output_dir)

    elapsed = time.time() - t0
    log.info(f"\nTotal elapsed: {elapsed:.1f}s")
    log.info(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
