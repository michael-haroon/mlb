"""DEPRECATED: Old PCA cross-check implementation (wrong).

This was replaced in August 2026. Two problems:

1. It only ran ONE method (PCA-MDA) on PCs and compared eigenvalue rank to
   that single method's importance rank. De Prado's procedure requires multiple
   independent methods (MDI, MDA, SFI) each compared to eigenvalue rank.

2. It used PCA-MDA to validate itself — PCA-MDA is one of the importance methods
   being validated by the cross-check, so using its output as the cross-check
   is circular.

The correct implementation lives in pregame/analysis/feature_importance.py::pca_cross_check()
which runs MDI, MDA, and SFI independently on the PC matrix and returns a
method-keyed dict: {"MDI": {"tau": ..., "p_value": ...}, "MDA": {...}, "SFI": {...}}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import weightedtau


def pca_cross_check_old(pc_summary: pd.DataFrame,
                        explained_variance_ratio: np.ndarray) -> tuple:
    """WRONG: Single-method PCA cross-check using PCA-MDA output.

    Compares eigenvalue rank of PC_k to the PCA-MDA importance rank of PC_k.
    This is circular because PCA-MDA is one of the methods being validated.

    Parameters
    ----------
    pc_summary : DataFrame
        Per-PC importance from feat_imp_pca_mda (index=PC_0..PC_k, columns=[mean, std]).
    explained_variance_ratio : array
        PCA explained_variance_ratio_ for the same components.

    Returns:
        (pca_info DataFrame indexed by PC_k, tau_results dict)
    """
    k = len(pc_summary)
    pc_names = pc_summary.index.tolist()

    eigenvalues = explained_variance_ratio[:k]
    eigenvalue_ranks = pd.Series(eigenvalues, index=pc_names).rank(ascending=False).values

    pc_importance = pc_summary["mean"].values
    importance_ranks = pd.Series(pc_importance, index=pc_names).rank(ascending=False).values

    tau, _ = weightedtau(eigenvalue_ranks, importance_ranks)
    rng_perm = np.random.default_rng(42)
    n_perm = 1000
    perm_taus = np.array([
        weightedtau(eigenvalue_ranks, rng_perm.permutation(importance_ranks))[0]
        for _ in range(n_perm)
    ])
    p = float((np.abs(perm_taus) >= np.abs(tau)).mean())

    pca_info = pd.DataFrame({
        "explained_variance_ratio": eigenvalues,
        "eigenvalue_rank": eigenvalue_ranks,
        "mda_importance": pc_importance,
        "mda_rank": importance_ranks,
    }, index=pc_names)

    tau_results = {
        "tau": float(tau) if not np.isnan(tau) else None,
        "p_value": p,
        "n_components": k,
        "variance_explained": float(np.sum(eigenvalues)),
    }

    return pca_info, tau_results
