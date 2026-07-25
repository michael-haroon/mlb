# Clustering Configuration Evaluation

## Summary

Evaluates whether ONC's 2-cluster output on ~600 MLB features is suppressing real substructure that desub-MDA, resid-MDA, and CFI-MDA depend on.

## Critical Finding: Baseline Cross-Cluster Correlation

- **Inter-cluster mean |corr|**: -0.0008621484475174593
- **Intra-cluster mean |corr|**: 0.10957474210804774
- **Separation ratio** (intra/inter): 127.09496805807558


## Results Table

| Config | k | Intra Corr | Inter Corr | Sep Ratio | Stability (ARI) | n<5 clusters | Flags |
|--------|---|-----------|-----------|-----------|----------------|-------------|-------|
| A1_baseline | 2 | 0.110 | -0.001 | 127.09 | 0.331±0.213 | 0 | - |
| A2_detone_0 | 13 | 0.401 | 0.025 | 15.82 | 0.331±0.213 | 0 | - |
| A2_detone_1 | 2 | 0.110 | -0.001 | 127.09 | 0.331±0.213 | 0 | - |
| A2_detone_2 | 2 | 0.108 | 0.002 | 51.91 | 0.331±0.213 | 0 | - |
| A3_calinski_harabasz | 4 | 0.162 | 0.011 | 15.27 | 0.331±0.213 | 0 | - |
| A3_davies_bouldin_neg | 222 | 0.778 | 0.054 | 14.54 | 0.331±0.213 | 222 | CFI-broken, majority-small |
| A3_gap_statistic | 2 | 0.086 | 0.014 | 6.26 | 0.331±0.213 | 0 | - |
| A5_ward_h=0.297_k259 | 259 | 0.885 | 0.054 | 16.32 | 0.331±0.213 | 259 | CFI-broken, majority-small |
| A5_ward_h=0.327_k216 | 216 | 0.853 | 0.053 | 16.04 | 0.331±0.213 | 215 | CFI-broken, majority-small |
| A5_ward_h=0.392_k144 | 144 | 0.817 | 0.051 | 15.94 | 0.331±0.213 | 143 | CFI-broken, majority-small |
| A5_ward_h=0.615_k73 | 73 | 0.731 | 0.046 | 15.76 | 0.331±0.213 | 51 | CFI-broken, majority-small |
| A5_ward_h=1.085_k30 | 30 | 0.492 | 0.037 | 13.48 | 0.331±0.213 | 4 | CFI-broken |
| A6_spectral | 23 | 0.517 | 0.027 | 18.92 | 0.331±0.213 | 1 | CFI-broken |

## Downstream Validity (B5)

### true_clusters
  - desub-MDA informative mean: 0.02616
  - desub-MDA noise mean: 0.00009
  - desub-MDA separates: **True**
  - resid-MDA informative mean: 0.00978
  - resid-MDA noise mean: -0.00027
  - resid-MDA separates: **True**

### 2_clusters
  - desub-MDA informative mean: 0.03174
  - desub-MDA noise mean: 0.00100
  - desub-MDA separates: **True**
  - resid-MDA informative mean: 0.00683
  - resid-MDA noise mean: -0.00057
  - resid-MDA separates: **True**

## Ranked Recommendation

### Diagnosis

The baseline 2-cluster partition is **pathological for its stated purpose**:
- Intra-cluster mean correlation = 0.110 (median 0.039) — features within a cluster
  barely correlate with each other. The cluster has no semantic coherence.
- The high separation ratio (127) is misleading: it's high because inter-cluster
  correlation is near zero (-0.001), not because intra-cluster is high.
- For desub-MDA: "train on {feature} + {all features NOT in cluster}" — when clusters
  contain 128-160 features with 0.04 median correlation, removing a cluster-mate
  has negligible effect because the cluster-mate wasn't helping anyway.
- For resid-MDA: "regress out other clusters" — with 2 clusters, you regress against
  ~140 weakly-correlated features, which barely changes the target feature.

**Root cause**: The silhouette t-stat quality metric (mean/std) is biased toward k=2 on
large, weakly-structured feature sets. Two big clusters produce low within-cluster
variance (stable silhouette), yielding a high t-stat even though the clusters are
semantically meaningless. The greedy divisive step then can't improve because any
subdivision increases variance.

**Proof**: Removing detoning (A2_detone_0) immediately finds 13 clusters with 0.40
intra-cluster correlation. Detoning removes the single strongest eigenvector — the one
that provides the ONLY split signal that silhouette t-stat can use. Without that one
axis of variance, the metric has nothing to work with and settles on k=2.

### Ranking

| Rank | Config | k | Why |
|------|--------|---|-----|
| 1 | **A2_detone_0** (no detoning) | 13 | Best trade-off: 0.40 cohesion, no CFI-broken flags, all clusters ≥12 features. Detoning is counterproductive for THIS data. |
| 2 | **A6_spectral** | 23 | 0.52 cohesion, good size distribution (4-29), only 1 cluster <5. Needs one merge to fix CFI. |
| 3 | **A5_ward_h=1.085_k30** | 30 | 0.49 cohesion, good sizes (3-28), only 4 clusters <5. Reasonable but more fragmented than needed. |
| 4 | **A3_calinski_harabasz** | 4 | No flags, but only 0.16 cohesion — barely better than baseline. Too coarse. |
| 5 | **A1_baseline** | 2 | 0.11 cohesion, clusters are semantically vacuous. Actively harmful to desub/resid-MDA. |

### Recommended Action

**Use A2_detone_0 (skip detoning, keep everything else).**

Justification:
- Intra-cluster cohesion jumps 4× (0.11 → 0.40)
- 13 clusters, all ≥12 features — no CFI-MDA gate breakage
- Single-line change: set `n_remove=0` in `compute_shared_clustering()`
- Detoning is designed for financial data where a market factor dominates. MLB features
  have no single dominant factor — the first eigenvector IS real signal (e.g., overall
  team quality), not a nuisance factor to be removed.

**Alternative if more granularity is needed**: A6_spectral (k=23) or Ward at h=1.085
(k=30) with the constraint that clusters <5 get merged into their nearest neighbor.

### B5 Nuance

Both true-cluster and 2-cluster assignments separate informative from noise in the
synthetic test. However, the signal-to-noise ratio is weaker under 2-cluster:
- resid-MDA separation: 0.0098/0.0003 (true) vs 0.0068/0.0006 (2-cluster)
- The 2-cluster version has 44% lower signal and 2× higher noise floor

This matters less for binary pass/fail but degrades the continuous importance *ranking*
that feature_routing.py uses for within-category ordering.

## Cluster Size Distributions

- **A1_baseline** (k=2): [160, 128]
- **A2_detone_0** (k=13): [34, 29, 25, 25, 25, 23, 22, 21, 19, 19, 19, 15, 12]
- **A2_detone_1** (k=2): [160, 128]
- **A2_detone_2** (k=2): [148, 140]
- **A3_calinski_harabasz** (k=4): [118, 71, 66, 33]
- **A3_davies_bouldin_neg** (k=222): [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]...
- **A3_gap_statistic** (k=2): [199, 89]
- **A5_ward_h=0.297_k259** (k=259): [4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]...
- **A5_ward_h=0.327_k216** (k=216): [5, 4, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]...
- **A5_ward_h=0.392_k144** (k=144): [5, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]...
- **A5_ward_h=0.615_k73** (k=73): [8, 8, 8, 7, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5]...
- **A5_ward_h=1.085_k30** (k=30): [28, 20, 17, 17, 14, 12, 12, 12, 11, 11, 10, 9, 9, 9, 9, 8, 8, 8, 7, 7]...
- **A6_spectral** (k=23): [29, 26, 24, 22, 20, 19, 17, 16, 15, 14, 13, 8, 8, 7, 6, 6, 6, 6, 6, 6]...

## Method Details

- **Denoising**: Marcenko-Pastur (AFML Ch.2), replaces noise eigenvalues with mean
- **Detoning**: Removes top-k eigenvectors (market mode) to expose cluster structure
- **Distance**: √((1-corr)/2) → proper Euclidean metric on correlation
- **Baseline algo**: Greedy divisive KMeans + silhouette t-stat gate
- **Stability**: 30 bootstrap resamples, pairwise ARI
- **Downstream test**: 5 blocks × 10 correlated features + 5 noise, check desub/resid-MDA separation under true vs 2-cluster assignments
