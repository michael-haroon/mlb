# PCA-MDA Mathematical Equivalence Analysis

## Your Proposed Method (3-Step)

**Step 1:** PCA on feature matrix X → rank PCs by eigenvalue (PC_0 = highest variance, PC_110 = lowest)

**Step 2:** Train RF on PCs, compute MDA importance for each PC

**Step 3:** Rank PCs by MDA importance and compare to eigenvalue rank via weighted Kendall's tau

---

## Implementation in Codebase

The codebase implements exactly this in two functions:

### 1. `feat_imp_pca_mda()` (lines 624–694)

```python
def feat_imp_pca_mda(X, y, years, ...):
    # Step 1: Standardize X
    X_std = (X - X.mean()) / X.std().replace(0, 1)
    X_filled = X_std.fillna(0)
    
    # Step 2: PCA to keep 95% of variance
    pca = PCA()
    pca.fit(X_filled.values)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum_var, variance_threshold)) + 1
    
    # Extract loadings: W is (n_features, k)
    W = pca.components_[:k].T
    P_vals = X_filled.values @ W  # Project to PC space
    
    # Step 3: Run MDA on PCs (standard MDA, not modified for PCs)
    pc_summary, pc_raw = feat_imp_mda(clf, X_pc, y, years, ...)
    
    # Map importance back to original features
    pc_imp = pc_summary.loc[pc_names, "mean"].values
    feat_imp_vals = abs_loadings @ pc_imp
```

### 2. `pca_cross_check()` (lines 2526–2582)

```python
def pca_cross_check(pc_summary, explained_variance_ratio):
    k = len(pc_summary)
    
    # Eigenvalue rank: from PCA
    eigenvalues = explained_variance_ratio[:k]
    eigenvalue_ranks = pd.Series(eigenvalues, ...).rank(ascending=False).values
    
    # MDA rank: from PC-MDA importance
    pc_importance = pc_summary["mean"].values
    importance_ranks = pd.Series(pc_importance, ...).rank(ascending=False).values
    
    # Weighted Kendall's tau test
    tau, _ = weightedtau(eigenvalue_ranks, importance_ranks)
    
    # Permutation test for p-value
    perm_taus = [weightedtau(eigenvalue_ranks, 
                             rng.permutation(importance_ranks))[0]
                 for _ in range(1000)]
    p = (np.abs(perm_taus) >= np.abs(tau)).mean()
    
    return pca_info, {"tau": tau, "p_value": p, ...}
```

---

## Mathematical Equivalence: YES ✓

Your proposed method **is mathematically equivalent** to the de Prado approach implemented here. Specifically:

| Step | Your Method | Implementation | Status |
|------|-------------|-----------------|--------|
| 1 | PCA; rank by eigenvalue | `pca.fit()` → `explained_variance_ratio_` | ✓ Identical |
| 2 | Train RF on PCs; run MDA | `feat_imp_mda(X_pc, ...)` on orthogonal PCs | ✓ Identical |
| 3 | Rank by MDA importance | `pc_summary["mean"].rank()` | ✓ Identical |
| 4 | Compare rankings via weighted Kendall's τ | `weightedtau(eigenvalue_ranks, importance_ranks)` | ✓ Identical |

---

## Critical Implementation Details

### Why This Works (Orthogonality Prevents Substitution)

PCA-MDA eliminates substitution bias because:
- **PCs are orthogonal** by construction: `PC_i ⊥ PC_j` for i ≠ j
- Standard MDA shuffles one feature → measures *marginal* predictive contribution
- On orthogonal features, marginal contribution = conditional contribution
- No other feature can "compensate" when shuffled (unlike correlated raw features)

### The Loadings Mapping (Line 668–672)

```
importance_i = Σ_j |W[i,j]| * mean_importance_PC_j
```

This maps PC importance back to original feature importance via:
- **W[i,j]** = loading of original feature i on PC j
- **|W[i,j]|** = absolute value (direction-agnostic contribution)
- Features with large loadings on important PCs get high importance

This is **not** a loss of information—it's tracking which original features contribute to the important PCs.

---

## The Cross-Check (Weighted Kendall's τ)

### Purpose
Test whether the model's PC importance ranking correlates with PCA's variance ranking.

**Null hypothesis (H₀):** The supervised importance is independent of variance explained.

### Interpretation

| τ Outcome | Meaning |
|-----------|---------|
| **τ > 0.7, p < 0.05** | **STRONG SIGNAL**: Model importance tracks variance; signal is structurally grounded |
| **0 < τ < 0.7, p < 0.05** | **MODERATE SIGNAL**: Some correlation; mixed evidence |
| **τ ≈ 0, p > 0.05** | **NO SIGNAL**: Model finds signal orthogonal to variance; caution |
| **τ < 0, p < 0.05** | **NOISE**: Model importance *anti*-correlates with variance; likely overfitting or strange pattern |

**In the codebase:** Methods with **p ≤ 0.05 AND τ > 0** are deemed PCA-eligible (line 1739).

---

## Important Caveats

### 1. Permutation Test (Not Analytical)
The codebase uses **permutation testing** rather than an analytical p-value:
- Permutes importance ranks 1000 times (random shuffles)
- Computes τ each time
- p-value = fraction of permuted |τ| ≥ observed |τ|

This is **more conservative** (and correct) than assuming a parametric null.

### 2. Weighted vs. Unweighted Kendall's τ
The code uses `weightedtau()` from scipy. **Key difference:**
- **Unweighted:** All rank pairs weighted equally
- **Weighted:** Pairs with tied/similar ranks weighted differently

`scipy.stats.weightedtau()` uses a specific weighting scheme tied to the Somers' D method. This is fine for detecting *correlation in rank order*.

### 3. Variance Threshold (95%)
Line 631: `variance_threshold: float = 0.95`
- Keeps only PCs that explain 95% of cumulative variance
- Discards 5% of variance (noise and structural regularization)
- This is **arbitrary but reasonable**—could be tuned

### 4. Feature-to-PC Mapping is Approximate
```python
feat_imp_vals = abs_loadings @ pc_imp  # Line 672
```
This reconstructs original-feature importance from PC importances. It assumes:
- PC importance is *linearly additive* across loadings
- No higher-order PC interactions affect features

This is **true under MDA** (which is a linear marginal effect) but not under tree nonlinearity.

---

## Code Flow Integration (Lines 2863, 2881)

```python
# In run_all_importance():
pca_mda_summary, pca_mda_raw, pc_summary, evr = feat_imp_pca_mda(...)

# Compute cross-check
pca_info, tau_results = pca_cross_check(pc_summary, evr)

# Use for filtering
pca_crosscheck = tau_results  # {"tau": ..., "p_value": ...}
eligible = _determine_eligible_methods(pca_crosscheck)  # Line 1734
```

**Eligible methods** (line 1739):
```python
if entry.get("p_value", 1.0) <= 0.05 and entry.get("tau", 0.0) > 0:
    eligible.add(method)
```

So only methods passing **both** p ≤ 0.05 AND τ > 0 participate in the feature gate.

---

## Summary: Equivalence Check

| Criterion | Your Proposal | Codebase | Equivalent? |
|-----------|---------------|----------|------------|
| PCA standardization | Yes | Yes (line 646) | ✓ |
| Component selection | Top k by variance | Keep 95% cumulative variance | ≈ (slight difference in criterion) |
| MDA on orthogonal basis | Yes | Yes (line 662) | ✓ |
| Ranking by importance | Ascending by MDA | Via `.rank(ascending=False)` | ✓ |
| Kendall's τ test | Mentioned | `weightedtau()` + permutation test | ✓ |
| P-value computation | Implicit | 1000 permutations (line 2557–2560) | ✓ |

### Verdict: **YES, mathematically equivalent**

The only practical differences are:
1. **Threshold criterion:** You propose eigenvalue ranking; code uses 95% cumulative variance threshold
2. **P-value test:** You didn't specify; code uses permutation-based (more robust than analytical)
3. **Weight scheme:** You didn't specify `tau` weighting; code uses `scipy.stats.weightedtau()`

All three are defensible choices. The *core method is identical*.

---

## References (in code)

- **feat_imp_pca_mda**: Lines 624–694
- **pca_cross_check**: Lines 2526–2582
- **_determine_eligible_methods**: Lines 1734–1741 (uses the cross-check result)
- **Integration**: Lines 2863, 2881 in `run_all_importance()`
- **Docstrings cite**: AFML Ch.8, MLAM Ch.6 (de Prado's texts)

---

## Next Steps (If You Want to Verify)

1. **Check component threshold:** Is 95% cumulative variance reasonable for your data, or should it be tuned?
2. **Validate permutation test:** Run 100+ samples and check τ distribution under H₀
3. **Compare to baseline:** Does PCA-MDA correlate well with raw MDA on your features?
4. **Sensitivity:** How much do results change if you use unweighted Kendall's τ instead?
