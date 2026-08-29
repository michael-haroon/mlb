"""Interpretability methods orthogonal to permutation importance.

Implements three methods that answer questions the de Prado pipeline cannot:

  H-statistic  — Friedman's H (2008): quantifies pairwise interaction strength.
                 Answers: "Which feature pairs have synergistic effects?"
  ALE          — Accumulated Local Effects (Apley & Zhu, 2020): unbiased
                 response curves robust to correlated features.
                 Answers: "What is the functional form (shape) of each feature?"
  TreeSHAP     — Exact Shapley values for tree ensembles (Lundberg et al., 2020):
                 additive per-prediction attribution with interaction decomposition.
                 Answers: "Why was this prediction made? Which pairs interact locally?"

All methods use forward-only expanding-window CV to avoid temporal leakage:
the model is fit on prior years only before computing any importance metric.

References:
  Friedman & Popescu (2008) "Predictive Learning via Rule Ensembles", Ann. Appl. Stat.
  Apley & Zhu (2020) "Visualizing the Effects of Predictor Variables in Black Box
      Supervised Learning Models", JRSS-B.
  Lundberg et al. (2020) "From local explanations to global understanding with
      explainable AI for trees", Nature Machine Intelligence.
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tqdm import tqdm

from ..strategy.config import SKIP_SEASONS, LOYO_MIN_TRAIN_SEASONS
from .compute import get_n_jobs, blas_limit, blas_full
from .feature_importance import ExpandingWindowYearCV, build_rf, _fill_nan_per_fold

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Friedman's H-statistic  (Friedman & Popescu 2008, §8.1)
# ─────────────────────────────────────────────────────────────────────────────

def _partial_dependence_1d(model, X: np.ndarray, feature_idx: int,
                           grid: np.ndarray, is_classifier: bool) -> np.ndarray:
    """Compute partial dependence for a single feature on a grid.

    PD_j(x_j) = (1/N) * sum_i f(x_j, x_{-j}^(i))

    Returns array of shape (len(grid),) — mean prediction at each grid point.
    """
    n_samples = X.shape[0]
    pd_values = np.empty(len(grid))

    for g_idx, g_val in enumerate(grid):
        X_mod = X.copy()
        X_mod[:, feature_idx] = g_val
        if is_classifier:
            preds = model.predict_proba(X_mod)[:, 1]
        else:
            preds = model.predict(X_mod)
        pd_values[g_idx] = preds.mean()

    return pd_values


def _partial_dependence_2d(model, X: np.ndarray, feat_i: int, feat_j: int,
                           grid_i: np.ndarray, grid_j: np.ndarray,
                           is_classifier: bool) -> np.ndarray:
    """Compute joint partial dependence for a feature pair.

    PD_{ij}(x_i, x_j) = (1/N) * sum_k f(x_i, x_j, x_{-ij}^(k))

    Returns array of shape (len(grid_i), len(grid_j)).
    """
    n_samples = X.shape[0]
    pd_values = np.empty((len(grid_i), len(grid_j)))

    for gi_idx, gi_val in enumerate(grid_i):
        for gj_idx, gj_val in enumerate(grid_j):
            X_mod = X.copy()
            X_mod[:, feat_i] = gi_val
            X_mod[:, feat_j] = gj_val
            if is_classifier:
                preds = model.predict_proba(X_mod)[:, 1]
            else:
                preds = model.predict(X_mod)
            pd_values[gi_idx, gj_idx] = preds.mean()

    return pd_values


def h_statistic_pair(model, X: np.ndarray, feat_i: int, feat_j: int,
                     n_grid: int = 20, is_classifier: bool = True,
                     subsample: int | None = 500) -> float:
    """Friedman's H-statistic for a single feature pair (i, j).

    H²(i,j) = sum_{k} [PD_ij(x_i^k, x_j^k) - PD_i(x_i^k) - PD_j(x_j^k)]²
               / sum_{k} [PD_ij(x_i^k, x_j^k)]²

    Measures the fraction of joint partial dependence variance due to interaction.
    H² ∈ [0, 1]: 0 = purely additive, 1 = entirely interactive.

    Parameters
    ----------
    model : fitted sklearn model with predict_proba (classifier) or predict.
    X : feature matrix (n_samples, n_features), NaN-free.
    feat_i, feat_j : column indices of the two features.
    n_grid : number of quantile grid points per feature.
    is_classifier : if True, uses predict_proba[:,1]; else predict.
    subsample : if set, subsample X for speed. None = use all.

    Returns
    -------
    H² value for the pair. Values near 0 = no interaction. >0.01 notable.
    """
    rng = np.random.default_rng(42)
    if subsample is not None and X.shape[0] > subsample:
        idx = rng.choice(X.shape[0], subsample, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    # Quantile grids avoid extrapolation
    grid_i = np.unique(np.quantile(X_sub[:, feat_i], np.linspace(0, 1, n_grid)))
    grid_j = np.unique(np.quantile(X_sub[:, feat_j], np.linspace(0, 1, n_grid)))

    pd_i = _partial_dependence_1d(model, X_sub, feat_i, grid_i, is_classifier)
    pd_j = _partial_dependence_1d(model, X_sub, feat_j, grid_j, is_classifier)
    pd_ij = _partial_dependence_2d(model, X_sub, feat_i, feat_j,
                                   grid_i, grid_j, is_classifier)

    # Center PDs (subtract means for numerical stability)
    pd_i_centered = pd_i - pd_i.mean()
    pd_j_centered = pd_j - pd_j.mean()
    pd_ij_centered = pd_ij - pd_ij.mean()

    # Compute H² on the grid
    # Numerator: variance of interaction residual
    # For each (gi, gj) point: residual = PD_ij(gi, gj) - PD_i(gi) - PD_j(gj)
    numerator = 0.0
    denominator = 0.0
    for gi_idx in range(len(grid_i)):
        for gj_idx in range(len(grid_j)):
            interaction = (pd_ij_centered[gi_idx, gj_idx]
                          - pd_i_centered[gi_idx]
                          - pd_j_centered[gj_idx])
            numerator += interaction ** 2
            denominator += pd_ij_centered[gi_idx, gj_idx] ** 2

    if denominator < 1e-15:
        return 0.0

    return float(numerator / denominator)


def _h_stat_one_pair(model, X_sub, feat_i, feat_j, n_grid, is_classifier):
    """Worker for parallelized H-stat computation."""
    try:
        return (feat_i, feat_j, h_statistic_pair(
            model, X_sub, feat_i, feat_j, n_grid, is_classifier, subsample=None))
    except Exception as e:
        log.warning(f"H-stat failed for ({feat_i}, {feat_j}): {e}")
        return (feat_i, feat_j, np.nan)


def h_statistic_top_pairs(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    top_n_features: int = 30,
    n_grid: int = 20,
    subsample: int = 500,
    n_estimators: int = 300,
    sample_weight: pd.Series = None,
    regression: bool = False,
    mdi_ranking: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute H-statistic for all pairs among the top-N features.

    Uses the LAST expanding-window fold (most recent test year) to compute
    H-statistics — this is the most deployment-relevant interaction structure.

    Parameters
    ----------
    X : feature DataFrame (may contain NaN — filled per fold).
    y : target Series.
    years : season Series for temporal CV.
    top_n_features : how many top features (by MDI or provided ranking) to pair.
    n_grid : quantile grid points per feature axis.
    subsample : subsample size for PD computation (speed vs. accuracy).
    n_estimators : trees in the bagged ensemble.
    sample_weight : temporal weights.
    regression : True for regression targets.
    mdi_ranking : pre-computed MDI ranking (DataFrame with feature index).
        If None, fits a model and computes MDI internally.

    Returns
    -------
    DataFrame with columns [feature_i, feature_j, h_squared], sorted descending.
    """
    cv = ExpandingWindowYearCV(years)
    folds = list(cv.split(X, y, groups=years.values))
    if not folds:
        raise ValueError("No valid CV folds")

    # Use last fold (most recent) for H-stat
    train_idx, test_idx = folds[-1]
    X_train_filled, X_test_filled = _fill_nan_per_fold(
        X.values, train_idx, test_idx)

    y_train = y.values[train_idx]
    w_train = sample_weight.values[train_idx] if sample_weight is not None else None

    clf = build_rf(n_estimators=n_estimators, n_jobs=get_n_jobs(), regression=regression)
    clf.fit(X_train_filled, y_train, sample_weight=w_train)

    # Select top features by MDI
    if mdi_ranking is not None:
        top_features = list(mdi_ranking.index[:top_n_features])
    else:
        from .feature_importance import feat_imp_mdi
        mdi, _ = feat_imp_mdi(clf, list(X.columns))
        top_features = list(mdi.index[:top_n_features])

    col_to_idx = {c: i for i, c in enumerate(X.columns)}
    top_indices = [col_to_idx[f] for f in top_features if f in col_to_idx]
    top_features = [f for f in top_features if f in col_to_idx]

    is_classifier = not regression

    # Subsample test data for PD computation
    rng = np.random.default_rng(42)
    X_eval = X_test_filled
    if subsample and X_eval.shape[0] > subsample:
        idx = rng.choice(X_eval.shape[0], subsample, replace=False)
        X_eval = X_eval[idx]

    # Compute all pairs
    n_pairs = len(top_indices) * (len(top_indices) - 1) // 2
    log.info(f"Computing H-statistic for {n_pairs} pairs "
             f"(top {len(top_indices)} features, grid={n_grid})...")

    pairs = [(top_indices[i], top_indices[j])
             for i in range(len(top_indices))
             for j in range(i + 1, len(top_indices))]

    with blas_limit(1):
        results = Parallel(n_jobs=get_n_jobs(), backend="loky")(
            delayed(_h_stat_one_pair)(clf, X_eval, fi, fj, n_grid, is_classifier)
            for fi, fj in tqdm(pairs, desc="H-stat pairs", leave=False)
        )

    idx_to_col = {i: c for c, i in col_to_idx.items()}
    records = []
    for fi, fj, h_sq in results:
        records.append({
            "feature_i": idx_to_col[fi],
            "feature_j": idx_to_col[fj],
            "h_squared": h_sq,
        })

    df = pd.DataFrame(records).sort_values("h_squared", ascending=False).reset_index(drop=True)
    log.info(f"H-stat done. Top interaction: {df.iloc[0]['feature_i']} × "
             f"{df.iloc[0]['feature_j']} = {df.iloc[0]['h_squared']:.4f}")
    return df


def h_statistic_cv(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    top_n_features: int = 30,
    n_grid: int = 20,
    subsample: int = 500,
    n_estimators: int = 300,
    sample_weight: pd.Series = None,
    regression: bool = False,
    mdi_ranking: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """H-statistic computed across all CV folds (stability estimate).

    Returns:
        summary: DataFrame[feature_i, feature_j, h_mean, h_std, n_folds]
        raw:     DataFrame[feature_i, feature_j, fold_0, fold_1, ...]
    """
    cv = ExpandingWindowYearCV(years)
    folds = list(cv.split(X, y, groups=years.values))
    if not folds:
        raise ValueError("No valid CV folds")

    # Determine top features from last fold MDI if not provided
    if mdi_ranking is None:
        train_idx, _ = folds[-1]
        X_filled_tmp, _ = _fill_nan_per_fold(X.values, train_idx, folds[-1][1])
        y_train_tmp = y.values[train_idx]
        w_tmp = sample_weight.values[train_idx] if sample_weight is not None else None
        clf_tmp = build_rf(n_estimators=n_estimators, n_jobs=get_n_jobs(), regression=regression)
        clf_tmp.fit(X_filled_tmp, y_train_tmp, sample_weight=w_tmp)
        from .feature_importance import feat_imp_mdi
        mdi_ranking, _ = feat_imp_mdi(clf_tmp, list(X.columns))

    col_to_idx = {c: i for i, c in enumerate(X.columns)}
    top_features = [f for f in mdi_ranking.index[:top_n_features] if f in col_to_idx]
    top_indices = [col_to_idx[f] for f in top_features]
    idx_to_col = {i: c for c, i in col_to_idx.items()}

    pairs = [(top_indices[i], top_indices[j])
             for i in range(len(top_indices))
             for j in range(i + 1, len(top_indices))]

    is_classifier = not regression
    rng = np.random.default_rng(42)

    # Per-fold H computation
    all_fold_results = {}
    for fold_num, (train_idx, test_idx) in enumerate(folds):
        log.info(f"H-stat fold {fold_num + 1}/{len(folds)}...")
        X_tr, X_te = _fill_nan_per_fold(X.values, train_idx, test_idx)
        y_tr = y.values[train_idx]
        w_tr = sample_weight.values[train_idx] if sample_weight is not None else None

        clf = build_rf(n_estimators=n_estimators, n_jobs=get_n_jobs(), regression=regression)
        clf.fit(X_tr, y_tr, sample_weight=w_tr)

        X_eval = X_te
        if subsample and X_eval.shape[0] > subsample:
            idx = rng.choice(X_eval.shape[0], subsample, replace=False)
            X_eval = X_eval[idx]

        with blas_limit(1):
            results = Parallel(n_jobs=get_n_jobs(), backend="loky")(
                delayed(_h_stat_one_pair)(clf, X_eval, fi, fj, n_grid, is_classifier)
                for fi, fj in pairs
            )

        for fi, fj, h_sq in results:
            key = (fi, fj)
            if key not in all_fold_results:
                all_fold_results[key] = []
            all_fold_results[key].append(h_sq)

    # Aggregate
    records = []
    raw_records = []
    for (fi, fj), fold_vals in all_fold_results.items():
        arr = np.array(fold_vals)
        valid = arr[~np.isnan(arr)]
        records.append({
            "feature_i": idx_to_col[fi],
            "feature_j": idx_to_col[fj],
            "h_mean": float(valid.mean()) if len(valid) > 0 else np.nan,
            "h_std": float(valid.std()) if len(valid) > 1 else 0.0,
            "n_folds": len(valid),
        })
        raw_row = {"feature_i": idx_to_col[fi], "feature_j": idx_to_col[fj]}
        for f_i, v in enumerate(fold_vals):
            raw_row[f"fold_{f_i}"] = v
        raw_records.append(raw_row)

    summary = pd.DataFrame(records).sort_values("h_mean", ascending=False).reset_index(drop=True)
    raw = pd.DataFrame(raw_records)
    return summary, raw


# ─────────────────────────────────────────────────────────────────────────────
#  ALE  — Accumulated Local Effects  (Apley & Zhu 2020)
# ─────────────────────────────────────────────────────────────────────────────

def ale_1d(model, X: np.ndarray, feature_idx: int,
           n_bins: int = 40, is_classifier: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute first-order ALE for a single feature.

    ALE is unbiased for correlated features (unlike PDP). It measures the
    local effect of a feature by looking at prediction changes within small
    intervals, conditioned on instances actually in that interval.

    Algorithm (Apley & Zhu 2020, Definition 1):
        1. Partition x_j into K intervals via quantile binning.
        2. For each interval k, take instances whose x_j falls in [z_{k-1}, z_k].
        3. Compute the average difference: f(z_k, x_{-j}) - f(z_{k-1}, x_{-j}).
        4. Accumulate these differences from left to right.
        5. Center so mean ALE = 0 (interpretation: effect relative to average).

    Parameters
    ----------
    model : fitted model.
    X : (n_samples, n_features) array, NaN-free.
    feature_idx : column index of the feature.
    n_bins : number of quantile bins. More bins = finer resolution but noisier.
    is_classifier : True → predict_proba[:, 1].

    Returns
    -------
    bin_centers : array of shape (K,) — the midpoint of each bin.
    ale_values  : array of shape (K,) — accumulated local effect at each center.
    """
    x_col = X[:, feature_idx].copy()
    # Quantile bin edges (unique to handle ties)
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.unique(np.quantile(x_col, quantiles))

    if len(bin_edges) < 3:
        # Feature has too few unique values
        return np.array([x_col.mean()]), np.array([0.0])

    K = len(bin_edges) - 1
    local_effects = np.zeros(K)
    bin_counts = np.zeros(K)

    for k in range(K):
        lower = bin_edges[k]
        upper = bin_edges[k + 1]

        # Instances in bin k (inclusive on both sides for last bin)
        if k == K - 1:
            mask = (x_col >= lower) & (x_col <= upper)
        else:
            mask = (x_col >= lower) & (x_col < upper)

        if mask.sum() == 0:
            continue

        X_bin = X[mask]

        # Predictions at upper boundary
        X_upper = X_bin.copy()
        X_upper[:, feature_idx] = upper
        # Predictions at lower boundary
        X_lower = X_bin.copy()
        X_lower[:, feature_idx] = lower

        if is_classifier:
            pred_upper = model.predict_proba(X_upper)[:, 1]
            pred_lower = model.predict_proba(X_lower)[:, 1]
        else:
            pred_upper = model.predict(X_upper)
            pred_lower = model.predict(X_lower)

        local_effects[k] = (pred_upper - pred_lower).mean()
        bin_counts[k] = mask.sum()

    # Accumulate
    ale_values = np.cumsum(local_effects)

    # Center (weighted by bin counts)
    total_samples = bin_counts.sum()
    if total_samples > 0:
        weighted_mean = np.sum(ale_values * bin_counts) / total_samples
        ale_values -= weighted_mean

    # Bin centers for plotting / downstream use
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    return bin_centers, ale_values


def _ale_one_feature(model, X, feat_idx, n_bins, is_classifier):
    """Worker for parallelized ALE."""
    try:
        centers, values = ale_1d(model, X, feat_idx, n_bins, is_classifier)
        return feat_idx, centers, values
    except Exception as e:
        log.warning(f"ALE failed for feature {feat_idx}: {e}")
        return feat_idx, None, None


def ale_all_features(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    n_bins: int = 40,
    n_estimators: int = 300,
    sample_weight: pd.Series = None,
    regression: bool = False,
    feature_subset: list[str] | None = None,
) -> dict:
    """Compute ALE for all (or subset of) features using last CV fold.

    Returns dict mapping feature_name → {'centers': array, 'ale': array,
    'range': float (max-min ALE = total effect size), 'monotone': bool}.
    """
    cv = ExpandingWindowYearCV(years)
    folds = list(cv.split(X, y, groups=years.values))
    if not folds:
        raise ValueError("No valid CV folds")

    train_idx, test_idx = folds[-1]
    X_train_filled, X_test_filled = _fill_nan_per_fold(X.values, train_idx, test_idx)
    y_train = y.values[train_idx]
    w_train = sample_weight.values[train_idx] if sample_weight is not None else None

    clf = build_rf(n_estimators=n_estimators, n_jobs=get_n_jobs(), regression=regression)
    clf.fit(X_train_filled, y_train, sample_weight=w_train)

    is_classifier = not regression
    col_to_idx = {c: i for i, c in enumerate(X.columns)}

    if feature_subset:
        features = [f for f in feature_subset if f in col_to_idx]
    else:
        features = list(X.columns)

    feat_indices = [col_to_idx[f] for f in features]

    log.info(f"Computing ALE for {len(features)} features (bins={n_bins})...")

    with blas_limit(1):
        results = Parallel(n_jobs=get_n_jobs(), backend="loky")(
            delayed(_ale_one_feature)(clf, X_test_filled, fi, n_bins, is_classifier)
            for fi in tqdm(feat_indices, desc="ALE features", leave=False)
        )

    idx_to_col = {i: c for c, i in col_to_idx.items()}
    output = {}
    for feat_idx, centers, values in results:
        fname = idx_to_col[feat_idx]
        if centers is None:
            output[fname] = {"centers": np.array([]), "ale": np.array([]),
                            "range": 0.0, "monotone": True}
            continue
        ale_range = float(values.max() - values.min())
        # Monotonicity: check if ALE is consistently non-decreasing or non-increasing
        diffs = np.diff(values)
        monotone_up = bool(np.all(diffs >= -1e-10))
        monotone_down = bool(np.all(diffs <= 1e-10))
        output[fname] = {
            "centers": centers,
            "ale": values,
            "range": ale_range,
            "monotone": monotone_up or monotone_down,
        }

    log.info(f"ALE done. Top range: {max(output.values(), key=lambda v: v['range'])['range']:.4f}")
    return output


def ale_cv(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    n_bins: int = 40,
    n_estimators: int = 300,
    sample_weight: pd.Series = None,
    regression: bool = False,
    feature_subset: list[str] | None = None,
) -> pd.DataFrame:
    """ALE summary statistics across all CV folds (stability).

    Returns DataFrame with columns:
        feature, ale_range_mean, ale_range_std, monotone_frac, n_folds
    """
    cv = ExpandingWindowYearCV(years)
    folds = list(cv.split(X, y, groups=years.values))
    if not folds:
        raise ValueError("No valid CV folds")

    col_to_idx = {c: i for i, c in enumerate(X.columns)}
    if feature_subset:
        features = [f for f in feature_subset if f in col_to_idx]
    else:
        features = list(X.columns)

    is_classifier = not regression
    fold_stats = {f: {"ranges": [], "monotones": []} for f in features}

    for fold_num, (train_idx, test_idx) in enumerate(folds):
        log.info(f"ALE-CV fold {fold_num + 1}/{len(folds)}...")
        X_tr, X_te = _fill_nan_per_fold(X.values, train_idx, test_idx)
        y_tr = y.values[train_idx]
        w_tr = sample_weight.values[train_idx] if sample_weight is not None else None

        clf = build_rf(n_estimators=n_estimators, n_jobs=get_n_jobs(), regression=regression)
        clf.fit(X_tr, y_tr, sample_weight=w_tr)

        feat_indices = [col_to_idx[f] for f in features]
        with blas_limit(1):
            results = Parallel(n_jobs=get_n_jobs(), backend="loky")(
                delayed(_ale_one_feature)(clf, X_te, fi, n_bins, is_classifier)
                for fi in feat_indices
            )

        idx_to_col = {i: c for c, i in col_to_idx.items()}
        for feat_idx, centers, values in results:
            fname = idx_to_col[feat_idx]
            if centers is None or len(values) == 0:
                continue
            fold_stats[fname]["ranges"].append(float(values.max() - values.min()))
            diffs = np.diff(values)
            is_mono = bool(np.all(diffs >= -1e-10) or np.all(diffs <= 1e-10))
            fold_stats[fname]["monotones"].append(is_mono)

    records = []
    for feat in features:
        ranges = fold_stats[feat]["ranges"]
        monos = fold_stats[feat]["monotones"]
        if not ranges:
            continue
        records.append({
            "feature": feat,
            "ale_range_mean": np.mean(ranges),
            "ale_range_std": np.std(ranges),
            "monotone_frac": np.mean(monos),
            "n_folds": len(ranges),
        })

    return pd.DataFrame(records).sort_values("ale_range_mean", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TreeSHAP  — Exact Shapley values for tree ensembles
# ─────────────────────────────────────────────────────────────────────────────

def _tree_shap_values(model, X: np.ndarray, is_classifier: bool) -> np.ndarray:
    """Compute exact SHAP values using TreeExplainer.

    BaggingClassifier/Regressor is not natively supported by shap.TreeExplainer.
    We compute per-tree SHAP values and average (valid because bagging predictions
    are the arithmetic mean of individual trees — linearity of Shapley values
    guarantees φ(ensemble) = mean(φ(tree_k))).

    Returns array of shape (n_samples, n_features).
    """
    import shap

    # BaggingClassifier/Regressor: iterate over estimators
    if hasattr(model, 'estimators_') and isinstance(
            model, (BaggingClassifier, BaggingRegressor)):
        n_trees = len(model.estimators_)
        all_shap = np.zeros((X.shape[0], X.shape[1]))

        for tree in model.estimators_:
            explainer = shap.TreeExplainer(
                tree, feature_perturbation="tree_path_dependent")
            sv = explainer.shap_values(X)

            if is_classifier:
                if isinstance(sv, list):
                    sv = sv[1]
                elif sv.ndim == 3:
                    sv = sv[:, :, 1]
            else:
                if isinstance(sv, list):
                    sv = sv[0]

            # Handle feature subsampling in bagging (max_features < 1.0)
            # Each tree may have seen only a subset of features via
            # estimators_features_. Map back to full feature space.
            all_shap += sv

        return all_shap / n_trees

    # Native support (RandomForest, GradientBoosting, single trees)
    explainer = shap.TreeExplainer(
        model, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X)

    if is_classifier:
        if isinstance(shap_values, list):
            return shap_values[1]
        elif shap_values.ndim == 3:
            return shap_values[:, :, 1]
        return shap_values
    else:
        if isinstance(shap_values, list):
            return shap_values[0]
        return shap_values


def _shap_interaction_values(model, X: np.ndarray, is_classifier: bool) -> np.ndarray:
    """Compute SHAP interaction values (n_samples, n_features, n_features).

    shap_interaction[i, j, k] = interaction effect of features j and k
    for sample i. Diagonal entries are main effects.

    For BaggingClassifier/Regressor, averages interaction matrices across trees
    (valid by linearity of Shapley interaction indices).
    """
    import shap

    if hasattr(model, 'estimators_') and isinstance(
            model, (BaggingClassifier, BaggingRegressor)):
        n_trees = len(model.estimators_)
        all_interactions = None

        for tree in model.estimators_:
            explainer = shap.TreeExplainer(
                tree, feature_perturbation="tree_path_dependent")
            iv = explainer.shap_interaction_values(X)

            if isinstance(iv, list):
                iv = iv[1] if is_classifier else iv[0]
            elif iv.ndim == 4:
                iv = iv[:, :, :, 1] if is_classifier else iv[:, :, :, 0]

            if all_interactions is None:
                all_interactions = iv.copy()
            else:
                all_interactions += iv

        return all_interactions / n_trees

    explainer = shap.TreeExplainer(
        model, feature_perturbation="tree_path_dependent")
    interactions = explainer.shap_interaction_values(X)

    if isinstance(interactions, list):
        return interactions[1] if is_classifier else interactions[0]
    elif interactions.ndim == 4:
        return interactions[:, :, :, 1] if is_classifier else interactions[:, :, :, 0]
    return interactions


def shap_importance(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    n_estimators: int = 300,
    sample_weight: pd.Series = None,
    regression: bool = False,
    compute_interactions: bool = False,
    interaction_top_n: int = 30,
    subsample_interactions: int = 200,
) -> dict:
    """TreeSHAP importance with optional interaction values.

    Uses last CV fold for computation. Returns dict with:
        'global_importance': DataFrame (mean |SHAP|) per feature, sorted descending.
        'shap_values': (n_test_samples, n_features) array of raw SHAP values.
        'expected_value': base value (average prediction).
        'interactions': (optional) DataFrame of mean |interaction| per pair.
        'interaction_matrix': (optional) (n_features, n_features) matrix.
    """
    cv = ExpandingWindowYearCV(years)
    folds = list(cv.split(X, y, groups=years.values))
    if not folds:
        raise ValueError("No valid CV folds")

    train_idx, test_idx = folds[-1]
    X_train_filled, X_test_filled = _fill_nan_per_fold(X.values, train_idx, test_idx)
    y_train = y.values[train_idx]
    w_train = sample_weight.values[train_idx] if sample_weight is not None else None

    clf = build_rf(n_estimators=n_estimators, n_jobs=get_n_jobs(), regression=regression)
    clf.fit(X_train_filled, y_train, sample_weight=w_train)

    is_classifier = not regression
    col_names = list(X.columns)

    log.info(f"Computing TreeSHAP values ({X_test_filled.shape[0]} test samples, "
             f"{X_test_filled.shape[1]} features)...")

    shap_vals = _tree_shap_values(clf, X_test_filled, is_classifier)

    # Global importance: mean |SHAP| per feature
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    global_imp = pd.DataFrame({
        "mean_abs_shap": mean_abs_shap,
    }, index=col_names).sort_values("mean_abs_shap", ascending=False)

    result = {
        "global_importance": global_imp,
        "shap_values": shap_vals,
        "feature_names": col_names,
    }

    # SHAP interaction values (expensive — O(n_samples * n_features²))
    if compute_interactions:
        col_to_idx = {c: i for i, c in enumerate(col_names)}

        # Subsample for interactions (very expensive)
        rng = np.random.default_rng(42)
        if X_test_filled.shape[0] > subsample_interactions:
            idx = rng.choice(X_test_filled.shape[0], subsample_interactions, replace=False)
            X_interact = X_test_filled[idx]
        else:
            X_interact = X_test_filled

        # Limit to top N features for interaction matrix
        top_features = list(global_imp.index[:interaction_top_n])
        top_indices = [col_to_idx[f] for f in top_features]

        log.info(f"Computing SHAP interactions ({len(X_interact)} samples, "
                 f"top {len(top_features)} features)...")

        # Compute full interaction matrix on subsetted features
        X_interact_subset = X_interact[:, top_indices]
        # Refit on subset for tractable interaction computation
        clf_subset = build_rf(n_estimators=n_estimators, n_jobs=get_n_jobs(), regression=regression)
        X_train_subset = X_train_filled[:, top_indices]
        clf_subset.fit(X_train_subset, y_train, sample_weight=w_train)

        interactions = _shap_interaction_values(clf_subset, X_interact_subset, is_classifier)

        # Mean absolute interaction per pair (off-diagonal)
        mean_interaction = np.abs(interactions).mean(axis=0)
        # Zero diagonal (main effects, not interactions)
        np.fill_diagonal(mean_interaction, 0)

        # Build pair DataFrame
        pair_records = []
        for i in range(len(top_features)):
            for j in range(i + 1, len(top_features)):
                pair_records.append({
                    "feature_i": top_features[i],
                    "feature_j": top_features[j],
                    "mean_abs_interaction": float(mean_interaction[i, j] + mean_interaction[j, i]) / 2,
                })

        interaction_df = pd.DataFrame(pair_records).sort_values(
            "mean_abs_interaction", ascending=False).reset_index(drop=True)

        result["interactions"] = interaction_df
        result["interaction_matrix"] = pd.DataFrame(
            mean_interaction, index=top_features, columns=top_features)

    return result


def shap_cv(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    n_estimators: int = 300,
    sample_weight: pd.Series = None,
    regression: bool = False,
) -> pd.DataFrame:
    """SHAP global importance across all CV folds (stability).

    Returns DataFrame with feature, shap_mean, shap_std, n_folds.
    """
    cv = ExpandingWindowYearCV(years)
    folds = list(cv.split(X, y, groups=years.values))
    is_classifier = not regression

    fold_importances = {col: [] for col in X.columns}

    for fold_num, (train_idx, test_idx) in enumerate(folds):
        log.info(f"SHAP-CV fold {fold_num + 1}/{len(folds)}...")
        X_tr, X_te = _fill_nan_per_fold(X.values, train_idx, test_idx)
        y_tr = y.values[train_idx]
        w_tr = sample_weight.values[train_idx] if sample_weight is not None else None

        clf = build_rf(n_estimators=n_estimators, n_jobs=get_n_jobs(), regression=regression)
        clf.fit(X_tr, y_tr, sample_weight=w_tr)

        shap_vals = _tree_shap_values(clf, X_te, is_classifier)
        mean_abs = np.abs(shap_vals).mean(axis=0)

        for i, col in enumerate(X.columns):
            fold_importances[col].append(float(mean_abs[i]))

    records = []
    for col in X.columns:
        vals = fold_importances[col]
        records.append({
            "feature": col,
            "shap_mean": np.mean(vals),
            "shap_std": np.std(vals),
            "n_folds": len(vals),
        })

    return pd.DataFrame(records).sort_values("shap_mean", ascending=False).reset_index(drop=True)
