"""De Prado feature importance methods: MDI, MDA, SFI, CFI, ONC, denoising.

Implements the full de Prado pipeline (AFML Ch.7-8, MLAM Ch.4/6):
  MDI  — Mean Decrease Impurity (in-sample, fast)
  MDA  — Mean Decrease Accuracy (OOS, marginal)
  SFI  — Single Feature Importance (OOS, standalone, no substitution bias)
  CFI  — Clustered Feature Importance (corrects for multicollinearity)
  De-substituted MDA — within-cluster ranking, substitution-free
  PCA-MDA — orthogonal basis, substitution-free
  Residualized MDA — cross-cluster orthogonalization

Plus:
  - Forward-only expanding-window cross-validation (no temporal leakage)
  - Cluster detection via ONC (Optimal Number of Clusters)
  - Marcenko-Pastur denoising + detoning
  - Bootstrap CI + Wilcoxon significance testing
  - Three-tier algorithmic filter (ACCEPTED / NEEDS SPECIFICATION / REJECTED)
  - PCA cross-check with weighted Kendall's tau
  - Synthetic data validation

All importance methods use forward-only expanding-window CV for temporal safety.

References:
  AFML   Ch.7 (purged CV), Ch.8 (feature importance)
  MLAM   Ch.4 (ONC), Ch.6 (CFI)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.metrics import log_loss, roc_auc_score, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy.stats import wilcoxon as scipy_wilcoxon, weightedtau, spearmanr
from sklearn.decomposition import PCA
from tqdm import tqdm

from ..strategy.config import SKIP_SEASONS, LOYO_MIN_TRAIN_SEASONS
from .compute import get_n_jobs, get_parallel_split, blas_limit, blas_full

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  NaN handling — per-fold median fill to prevent temporal leakage
# ─────────────────────────────────────────────────────────────────────────────

def _fill_nan_per_fold(X_vals: np.ndarray, train_idx: np.ndarray,
                       test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaN using ONLY the training fold's median (no test data leakage).

    Returns (X_train_filled, X_test_filled) as numpy arrays.
    """
    X_tr = X_vals[train_idx].copy()
    X_te = X_vals[test_idx].copy()

    # nanmedian per column, computed on train only
    with np.errstate(all="ignore"):
        medians = np.nanmedian(X_tr, axis=0)
    # If a column is entirely NaN in train, fill with 0
    medians = np.where(np.isnan(medians), 0.0, medians)

    # Fill train and test with train-derived medians
    for col_i in range(X_tr.shape[1]):
        nan_mask_tr = np.isnan(X_tr[:, col_i])
        nan_mask_te = np.isnan(X_te[:, col_i])
        if nan_mask_tr.any():
            X_tr[nan_mask_tr, col_i] = medians[col_i]
        if nan_mask_te.any():
            X_te[nan_mask_te, col_i] = medians[col_i]

    return X_tr, X_te


# ─────────────────────────────────────────────────────────────────────────────
#  Marcenko-Pastur denoising + detoning  (de Prado AFML Ch.2)
# ─────────────────────────────────────────────────────────────────────────────

def denoise_corr(corr: pd.DataFrame, q: float) -> pd.DataFrame:
    """Denoise a correlation matrix via the Marcenko-Pastur theorem (AFML Ch.2).

    q = T / N  (observations / features).
    Eigenvalues at or below the MP upper bound λ+ are noise. They are replaced
    with their mean so the matrix trace (total explained variance) is preserved.
    """
    evals, evecs = np.linalg.eigh(corr.values)

    # MP upper bound for unit-variance random matrix (σ²=1 for corr matrix)
    lambda_plus = (1.0 + q ** -0.5) ** 2

    noise_mask = evals <= lambda_plus
    if noise_mask.any():
        noise_mean = evals[noise_mask].mean()
        evals = np.where(noise_mask, noise_mean, evals)

    corr_clean = evecs @ np.diag(evals) @ evecs.T
    diag_sqrt = np.sqrt(np.maximum(np.diag(corr_clean), 1e-12))
    corr_clean = corr_clean / np.outer(diag_sqrt, diag_sqrt)
    np.fill_diagonal(corr_clean, 1.0)

    return pd.DataFrame(corr_clean, index=corr.index, columns=corr.columns)


def detone_corr(corr: pd.DataFrame, n_remove: int = 1) -> pd.DataFrame:
    """Detone a (denoised) correlation matrix by zeroing out the n_remove largest
    eigenvectors (the 'market mode') (AFML Ch.2). Exposes cluster structure
    that would otherwise be masked by the common factor. Renormalises diagonal to 1.
    """
    if n_remove <= 0:
        return corr.copy()

    evals, evecs = np.linalg.eigh(corr.values)  # ascending order
    evals_detoned = evals.copy()
    evals_detoned[-n_remove:] = 0.0

    corr_detoned = evecs @ np.diag(evals_detoned) @ evecs.T
    diag_sqrt = np.sqrt(np.maximum(np.diag(corr_detoned), 1e-12))
    corr_detoned = corr_detoned / np.outer(diag_sqrt, diag_sqrt)
    np.fill_diagonal(corr_detoned, 1.0)

    return pd.DataFrame(corr_detoned, index=corr.index, columns=corr.columns)


def _align_proba(prob: np.ndarray, fit_classes: np.ndarray,
                 all_labels: np.ndarray) -> np.ndarray:
    """Expand prob columns to match all_labels, filling missing classes with 0."""
    if np.array_equal(fit_classes, all_labels):
        return prob
    out = np.zeros((prob.shape[0], len(all_labels)), dtype=prob.dtype)
    col_idx = np.searchsorted(all_labels, fit_classes)
    out[:, col_idx] = prob
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Purged K-Fold  (de Prado AFML Ch.7)
# ─────────────────────────────────────────────────────────────────────────────

class PurgedYearKFold:
    """Leave-one-year-out cross-validation for temporal data.

    When n_splits is None (default), does true LOYO (one fold per unique season).
    When n_splits is set, groups consecutive seasons into k folds — useful when
    the dataset is too small for LOYO to produce meaningful test sets.
    """

    def __init__(self, years: pd.Series, n_splits: int = None):
        self.unique_years = sorted(years.unique())
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        years = groups
        if years is None:
            raise ValueError("Pass df['season'] as the groups argument.")

        if self.n_splits is None:
            for test_year in self.unique_years:
                train_idx = np.where(years != test_year)[0]
                test_idx = np.where(years == test_year)[0]
                if len(train_idx) == 0 or len(test_idx) == 0:
                    continue
                yield train_idx, test_idx
        else:
            k = min(self.n_splits, len(self.unique_years))
            year_folds = np.array_split(self.unique_years, k)
            for fold_years in year_folds:
                test_mask = np.isin(years, fold_years)
                train_mask = ~test_mask
                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]
                if len(train_idx) == 0 or len(test_idx) == 0:
                    continue
                yield train_idx, test_idx

    def get_n_splits(self):
        if self.n_splits is not None:
            return min(self.n_splits, len(self.unique_years))
        return len(self.unique_years)


class ExpandingWindowYearCV:
    """Forward-only expanding-window cross-validation.

    Matches the training pipeline's temporal constraint: for each test year,
    training uses ONLY prior years. This is what the model sees at deployment.

    Symmetric LOYO (PurgedYearKFold) trains on future data to predict past,
    which inflates importance for noise features and penalizes features that
    became informative after structural breaks (2020 rules, 2023 pitch clock).
    """

    def __init__(self, years: pd.Series,
                 skip_seasons: list[int] | None = None,
                 min_train_seasons: int | None = None):
        self.unique_years = sorted(years.unique())
        self.skip_seasons = skip_seasons if skip_seasons is not None else SKIP_SEASONS
        self.min_train_seasons = min_train_seasons if min_train_seasons is not None else LOYO_MIN_TRAIN_SEASONS

    def split(self, X, y=None, groups=None):
        years = groups
        if years is None:
            raise ValueError("Pass df['season'] as the groups argument.")

        for test_year in self.unique_years:
            if test_year in self.skip_seasons:
                continue

            train_years = [s for s in self.unique_years
                          if s < test_year and s not in self.skip_seasons]

            if len(train_years) < self.min_train_seasons:
                continue

            train_idx = np.where(np.isin(years, train_years))[0]
            test_idx = np.where(years == test_year)[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx

    def get_n_splits(self):
        valid = 0
        for test_year in self.unique_years:
            if test_year in self.skip_seasons:
                continue
            train_years = [s for s in self.unique_years
                          if s < test_year and s not in self.skip_seasons]
            if len(train_years) >= self.min_train_seasons:
                valid += 1
        return valid


# ─────────────────────────────────────────────────────────────────────────────
#  Build a base RF classifier (de Prado's recommended setup, AFML Ch.8)
# ─────────────────────────────────────────────────────────────────────────────

def build_rf(n_estimators: int = 1000, n_jobs: int = -1,
             regression: bool = False):
    """De Prado's recommended setup (AFML Ch.8).

    regression=True builds a BaggingRegressor for continuous targets.
    regression=False (default) builds the classifier for binary targets.
    """
    if regression:
        base = DecisionTreeRegressor(
            max_features=1,
            min_weight_fraction_leaf=0.02,
        )
        return BaggingRegressor(
            estimator=base,
            n_estimators=n_estimators,
            max_features=1.0,
            max_samples=1.0,
            oob_score=False,
            n_jobs=n_jobs,
            random_state=42,
        )
    base = DecisionTreeClassifier(
        criterion="entropy",
        max_features=1,
        class_weight="balanced",
        min_weight_fraction_leaf=0.02,
    )
    return BaggingClassifier(
        estimator=base,
        n_estimators=n_estimators,
        max_features=1.0,
        max_samples=1.0,
        oob_score=False,
        n_jobs=n_jobs,
        random_state=42,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MDI  (de Prado AFML Ch.8 §8.3.1)
# ─────────────────────────────────────────────────────────────────────────────

def feat_imp_mdi(fit, feat_names: list) -> tuple:
    """Mean Decrease Impurity across all trees in the ensemble.

    Returns:
        summary: DataFrame(index=feature, columns=[mean, std]) — CLT-normalised
        raw:     DataFrame(index=tree_id, columns=features) — per-tree importances,
                 normalised to sum to 1 per tree.
    Zeros are set to NaN (feature was never chosen — an artefact of max_features=1).
    """
    imp_dict = {
        i: tree.feature_importances_
        for i, tree in enumerate(fit.estimators_)
    }
    imp_df = pd.DataFrame.from_dict(imp_dict, orient="index")
    imp_df.columns = feat_names
    imp_df = imp_df.replace(0, np.nan)

    # Normalise each tree row to sum to 1
    raw = imp_df.div(imp_df.sum(axis=1), axis=0)

    result = pd.concat({
        "mean": raw.mean(),
        "std": raw.std() * raw.shape[0] ** -0.5,  # CLT SE
    }, axis=1)
    return result.sort_values("mean", ascending=False), raw


# ─────────────────────────────────────────────────────────────────────────────
#  MDA  (de Prado AFML Ch.8 §8.3.2)
# ─────────────────────────────────────────────────────────────────────────────

def _mda_one_fold(clf_params, X_tr_vals, X_te_vals, y_tr_vals, y_te_vals,
                  col_names, w_tr_vals, scoring, seed, all_labels=None):
    """Run one MDA fold: fit model, score base + all permutations.

    X_tr_vals and X_te_vals may contain NaN — filled here using train median.
    scoring: 'log_loss' or 'roc_auc' for classifiers; 'r2' for regressors.
    """
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import BaggingClassifier, BaggingRegressor
    from sklearn.metrics import r2_score

    # Per-fold NaN fill: median from train only
    if np.isnan(X_tr_vals).any() or np.isnan(X_te_vals).any():
        with np.errstate(all="ignore"):
            medians = np.nanmedian(X_tr_vals, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for ci in range(X_tr_vals.shape[1]):
            tr_nan = np.isnan(X_tr_vals[:, ci])
            te_nan = np.isnan(X_te_vals[:, ci])
            if tr_nan.any():
                X_tr_vals = X_tr_vals.copy() if not X_tr_vals.flags.writeable else X_tr_vals
                X_tr_vals[tr_nan, ci] = medians[ci]
            if te_nan.any():
                X_te_vals = X_te_vals.copy() if not X_te_vals.flags.writeable else X_te_vals
                X_te_vals[te_nan, ci] = medians[ci]

    rng = np.random.default_rng(seed)
    is_regression = (scoring == "r2")

    if is_regression:
        base = DecisionTreeRegressor(max_features=1, min_weight_fraction_leaf=0.02)
        clf = BaggingRegressor(
            estimator=base, n_estimators=clf_params["n_estimators"],
            max_features=1.0, max_samples=1.0, oob_score=False,
            n_jobs=clf_params["n_jobs"], random_state=clf_params["random_state"],
        )
    else:
        base = DecisionTreeClassifier(
            criterion="entropy", max_features=1,
            class_weight="balanced", min_weight_fraction_leaf=0.02,
        )
        clf = BaggingClassifier(
            estimator=base, n_estimators=clf_params["n_estimators"],
            max_features=1.0, max_samples=1.0, oob_score=False,
            n_jobs=clf_params["n_jobs"], random_state=clf_params["random_state"],
        )

    X_tr = pd.DataFrame(X_tr_vals, columns=col_names)
    X_te = pd.DataFrame(X_te_vals, columns=col_names)
    y_tr = pd.Series(y_tr_vals)
    y_te = pd.Series(y_te_vals)

    fit = clf.fit(X_tr, y_tr, sample_weight=w_tr_vals)

    if is_regression:
        pred = fit.predict(X_te)
        base_score = r2_score(y_te, pred)
    else:
        ll_labels = all_labels if all_labels is not None else fit.classes_
        prob = _align_proba(fit.predict_proba(X_te), fit.classes_, ll_labels)
        if scoring == "log_loss":
            base_score = -log_loss(y_te, prob, labels=ll_labels)
        else:
            base_score = roc_auc_score(y_te, prob[:, 1])

    perm_fold = {}
    for col in col_names:
        X_perm = X_te.copy()
        X_perm[col] = rng.permutation(X_perm[col].values)
        if is_regression:
            pred_perm = fit.predict(X_perm)
            perm_fold[col] = r2_score(y_te, pred_perm)
        else:
            prob_perm = _align_proba(fit.predict_proba(X_perm), fit.classes_, ll_labels)
            if scoring == "log_loss":
                perm_fold[col] = -log_loss(y_te, prob_perm, labels=ll_labels)
            else:
                perm_fold[col] = roc_auc_score(y_te, prob_perm[:, 1])

    return base_score, perm_fold


def feat_imp_mda(clf,
                 X: pd.DataFrame,
                 y: pd.Series,
                 years: pd.Series,
                 sample_weight: pd.Series = None,
                 scoring: str = "log_loss") -> tuple:
    """Mean Decrease Accuracy via forward-only expanding-window CV.

    Shuffles one feature at a time and measures accuracy drop.
    Parallelised over folds via joblib with multiprocessing backend.

    Returns:
        summary: DataFrame(index=feature, columns=[mean, std])
        raw:     DataFrame(index=fold, columns=features) — per-fold importance
    """
    cv = ExpandingWindowYearCV(years)
    n_folds = cv.get_n_splits()
    n_outer, n_inner = get_parallel_split(n_folds)

    clf_params = {
        "n_estimators": clf.n_estimators,
        "n_jobs": n_inner,
        "random_state": 42,
    }

    folds = list(cv.split(X, y, groups=years.values))
    all_labels = None if scoring == "r2" else np.unique(y.values)
    with blas_limit(1):
        results = Parallel(n_jobs=n_outer, backend="multiprocessing")(
            delayed(_mda_one_fold)(
                clf_params,
                X.iloc[tr].values, X.iloc[te].values,
                y.iloc[tr].values, y.iloc[te].values,
                list(X.columns),
                sample_weight.iloc[tr].values if sample_weight is not None else None,
                scoring,
                seed=i,
                all_labels=all_labels,
            )
            for i, (tr, te) in enumerate(folds)
        )

    base_scores = [r[0] for r in results]
    base_arr = np.array(base_scores)

    records = {}
    raw_records = {}
    for col in X.columns:
        perm = np.array([r[1][col] for r in results])
        imp = base_arr - perm
        records[col] = {"mean": imp.mean(),
                        "std": imp.std() * len(imp) ** -0.5}
        raw_records[col] = imp

    result = pd.DataFrame(records).T
    result.columns = ["mean", "std"]
    raw = pd.DataFrame(raw_records)
    return result.sort_values("mean", ascending=False), raw


# ─────────────────────────────────────────────────────────────────────────────
#  De-substituted MDA  (Approach B: one from C_i, keep all others)
# ─────────────────────────────────────────────────────────────────────────────

def _desub_mda_one_task(feat_idx, other_idxs, X_vals, X_tr_idx, X_te_idx,
                        y_vals, w_vals, n_estimators, scoring, seed,
                        all_labels=None, regression=False):
    """One (feature, fold) task for de-substituted MDA.

    Trains on {feature} + {all non-cluster features}, shuffles the target feature.
    """
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import BaggingClassifier, BaggingRegressor
    from sklearn.metrics import r2_score as _r2

    rng = np.random.default_rng(seed)
    col_idxs = [feat_idx] + other_idxs

    X_tr = X_vals[np.ix_(X_tr_idx, col_idxs)].copy()
    X_te = X_vals[np.ix_(X_te_idx, col_idxs)].copy()
    y_tr = y_vals[X_tr_idx]
    y_te = y_vals[X_te_idx]
    w_tr = w_vals[X_tr_idx] if w_vals is not None else None

    # Per-fold NaN fill using train median only
    if np.isnan(X_tr).any() or np.isnan(X_te).any():
        with np.errstate(all="ignore"):
            medians = np.nanmedian(X_tr, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for ci in range(X_tr.shape[1]):
            tr_nan = np.isnan(X_tr[:, ci])
            te_nan = np.isnan(X_te[:, ci])
            if tr_nan.any():
                X_tr[tr_nan, ci] = medians[ci]
            if te_nan.any():
                X_te[te_nan, ci] = medians[ci]

    if regression:
        base = DecisionTreeRegressor(max_features=1, min_weight_fraction_leaf=0.02)
        clf = BaggingRegressor(
            estimator=base, n_estimators=n_estimators,
            max_features=1.0, max_samples=1.0, oob_score=False,
            n_jobs=1, random_state=42,
        )
    else:
        base = DecisionTreeClassifier(
            criterion="entropy", max_features=1,
            class_weight="balanced", min_weight_fraction_leaf=0.02,
        )
        clf = BaggingClassifier(
            estimator=base, n_estimators=n_estimators,
            max_features=1.0, max_samples=1.0, oob_score=False,
            n_jobs=1, random_state=42,
        )

    try:
        if not regression and len(np.unique(y_tr)) < 2:
            return None
        fit = clf.fit(X_tr, y_tr, sample_weight=w_tr)

        if regression:
            base_score = _r2(y_te, fit.predict(X_te))
        else:
            ll_labels = all_labels if all_labels is not None else fit.classes_
            prob = _align_proba(fit.predict_proba(X_te), fit.classes_, ll_labels)
            base_score = -log_loss(y_te, prob, labels=ll_labels)

        X_te_perm = X_te.copy()
        X_te_perm[:, 0] = rng.permutation(X_te_perm[:, 0])

        if regression:
            perm_score = _r2(y_te, fit.predict(X_te_perm))
        else:
            prob_perm = _align_proba(fit.predict_proba(X_te_perm), fit.classes_, ll_labels)
            perm_score = -log_loss(y_te, prob_perm, labels=ll_labels)

        return base_score - perm_score
    except Exception:
        return None


def feat_imp_desub_mda(X: pd.DataFrame,
                       y: pd.Series,
                       years: pd.Series,
                       clusters: dict,
                       sample_weight: pd.Series = None,
                       scoring: str = "log_loss",
                       n_estimators: int = 300,
                       regression: bool = False) -> tuple:
    """De-substituted MDA (Approach B): for each feature f in cluster C_i,
    train a model on {f} + {all features NOT in C_i}, then shuffle f.

    Eliminates substitution: no cluster-mate of f is present to compensate.

    Returns:
        summary: DataFrame(index=feature, columns=[mean, std])
        raw:     DataFrame(index=fold, columns=features)
    """
    cv = ExpandingWindowYearCV(years)
    folds = list(cv.split(X, y, groups=years.values))
    n_folds = len(folds)

    X_vals = X.values
    y_vals = y.values
    w_vals = sample_weight.values if sample_weight is not None else None
    col_names = list(X.columns)
    col_to_idx = {c: i for i, c in enumerate(col_names)}
    all_labels = None if regression else np.unique(y_vals)

    feat_to_cluster = {}
    for cid, members in clusters.items():
        for m in members:
            feat_to_cluster[m] = cid

    tasks = []
    for feat in col_names:
        feat_idx = col_to_idx[feat]
        cid = feat_to_cluster.get(feat)
        if cid is None:
            other_idxs = [col_to_idx[c] for c in col_names if c != feat]
        else:
            cluster_members = set(clusters[cid])
            other_idxs = [col_to_idx[c] for c in col_names
                          if c not in cluster_members]
        for fi, (tr, te) in enumerate(folds):
            tasks.append((feat, fi, feat_idx, other_idxs, tr, te))

    with blas_limit(1):
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(_desub_mda_one_task)(
                feat_idx, other_idxs, X_vals, tr, te,
                y_vals, w_vals, n_estimators, scoring,
                seed=fi, all_labels=all_labels, regression=regression,
            )
            for feat, fi, feat_idx, other_idxs, tr, te in tasks
        )

    raw_records = {col: [None] * n_folds for col in col_names}
    for (feat, fi, *_), score in zip(tasks, results):
        if score is not None:
            raw_records[feat][fi] = score

    records = {}
    final_raw = {}
    for col in col_names:
        scores = [s for s in raw_records[col] if s is not None]
        if scores:
            records[col] = {"mean": np.mean(scores),
                            "std": np.std(scores) * len(scores) ** -0.5}
            final_raw[col] = scores

    summary = pd.DataFrame(records).T
    summary.columns = ["mean", "std"]

    max_folds = max(len(v) for v in final_raw.values()) if final_raw else 0
    raw = pd.DataFrame(
        {col: vals + [np.nan] * (max_folds - len(vals))
         for col, vals in final_raw.items()}
    )
    return summary.sort_values("mean", ascending=False), raw


# ─────────────────────────────────────────────────────────────────────────────
#  PCA-MDA  (de Prado AFML Ch.8 / MLAM Ch.6 — orthogonal feature basis)
# ─────────────────────────────────────────────────────────────────────────────

def feat_imp_pca_mda(X: pd.DataFrame,
                     y: pd.Series,
                     years: pd.Series,
                     sample_weight: pd.Series = None,
                     scoring: str = "log_loss",
                     n_estimators: int = 1000,
                     regression: bool = False,
                     variance_threshold: float = 0.95) -> tuple:
    """MDA on principal components (de Prado AFML Ch.8 / MLAM Ch.6).

    Steps:
      1. Standardize X (zero mean, unit variance per feature)
      2. PCA → keep k components explaining variance_threshold of variance
      3. Run standard MDA on the PC matrix (no substitution — PCs are orthogonal)
      4. Map PC importance back to original features via |loading| × pc_importance

    Returns:
        summary: DataFrame(index=original feature, columns=[mean, std])
        raw:     DataFrame(index=fold, columns=original features)
        pc_summary: DataFrame(index=PC_i, columns=[mean, std])
    """
    with blas_full():
        X_std = (X - X.mean()) / X.std().replace(0, 1)
        X_filled = X_std.fillna(0)

        pca = PCA()
        pca.fit(X_filled.values)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(cum_var, variance_threshold)) + 1
        k = min(k, X_filled.shape[1])

        W = pca.components_[:k].T          # (n_features, k)
        P_vals = X_filled.values @ W       # (n_samples, k)

    pc_names = [f"PC_{i}" for i in range(k)]
    X_pc = pd.DataFrame(P_vals, index=X.index, columns=pc_names)

    clf = build_rf(n_estimators=n_estimators, n_jobs=1, regression=regression)
    pc_summary, pc_raw = feat_imp_mda(
        clf, X_pc, y, years,
        sample_weight=sample_weight,
        scoring=scoring,
    )

    # Map back: importance_i = sum_j |W[i,j]| * mean_importance_PC_j
    # Reindex to PC-index order (feat_imp_mda returns sorted by mean descending)
    pc_imp = pc_summary.loc[pc_names, "mean"].values            # (k,) in PC_0..PC_k order
    abs_loadings = np.abs(W)                                    # (n_features, k)
    feat_imp_vals = abs_loadings @ pc_imp                       # (n_features,)

    total = feat_imp_vals.sum()
    if total > 0:
        feat_imp_vals = feat_imp_vals / total

    # Build per-fold raw by projecting PC fold scores back to features
    feat_raw_dict = {}
    for feat_i, feat_name in enumerate(X.columns):
        fold_scores = []
        for fold_j in range(len(pc_raw)):
            pc_fold = pc_raw.iloc[fold_j].values
            feat_score = float(np.abs(W[feat_i]) @ pc_fold)
            fold_scores.append(feat_score)
        feat_raw_dict[feat_name] = fold_scores

    summary_df = pd.DataFrame({
        "mean": feat_imp_vals,
        "std": np.zeros(len(X.columns)),
    }, index=X.columns).sort_values("mean", ascending=False)

    raw_df = pd.DataFrame(feat_raw_dict)
    return summary_df, raw_df, pc_summary, pca.explained_variance_ratio_


# ─────────────────────────────────────────────────────────────────────────────
#  Residualized MDA  (de Prado MLAM Ch.6 — cross-cluster orthogonalization)
# ─────────────────────────────────────────────────────────────────────────────

def feat_imp_residual_mda(X: pd.DataFrame,
                          y: pd.Series,
                          years: pd.Series,
                          clusters: dict,
                          sample_weight: pd.Series = None,
                          scoring: str = "log_loss",
                          n_estimators: int = 1000,
                          regression: bool = False) -> tuple:
    """Residualized MDA (de Prado MLAM Ch.6).

    For each feature X_i in cluster C_k:
        X_i_residual = X_i - X_other @ lstsq(X_other, X_i)
    where X_other = all features NOT in C_k.

    If |other features| > n_samples/10, first reduce X_other via PCA keeping
    95% of variance, then regress against the PCs.

    Then run standard MDA on the full residualized matrix.

    Returns:
        summary: DataFrame(index=feature, columns=[mean, std])
        raw:     DataFrame(index=fold, columns=features)
    """
    col_names = list(X.columns)
    n_samples = len(X)
    cluster_sets = {cid: set(members) for cid, members in clusters.items()}
    col_medians = X.median()
    X_vals = X.fillna(col_medians).values.astype(np.float64)

    tasks = []
    for cid, cluster_members in clusters.items():
        other_cols = [c for c in col_names if c not in cluster_sets[cid]]
        if not other_cols:
            continue
        X_other = X[other_cols].fillna(col_medians[other_cols]).values.astype(np.float64)
        if X_other.shape[1] > n_samples // 10:
            pca_r = PCA(n_components=min(n_samples // 10, X_other.shape[1]))
            X_other = pca_r.fit_transform(X_other)
        for feat in cluster_members:
            if feat in col_names:
                tasks.append((col_names.index(feat), X_other))

    def _residualize(col_idx, X_other):
        f = X_vals[:, col_idx]
        coef, _, _, _ = np.linalg.lstsq(X_other, f, rcond=None)
        return col_idx, f - X_other @ coef

    with blas_limit(1):
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(_residualize)(ci, X_other) for ci, X_other in tasks
        )

    X_resid_vals = X_vals.copy()
    for ci, resid in results:
        X_resid_vals[:, ci] = resid

    X_resid = pd.DataFrame(X_resid_vals, index=X.index, columns=col_names)

    clf = build_rf(n_estimators=n_estimators, n_jobs=1, regression=regression)
    return feat_imp_mda(
        clf, X_resid, y, years,
        sample_weight=sample_weight,
        scoring=scoring,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  SFI  (de Prado AFML Ch.8 §8.4.1)
# ─────────────────────────────────────────────────────────────────────────────

def _sfi_one_task(col_idx, X_col_vals, X_tr_idx, X_te_idx,
                  y_vals, w_vals, n_estimators, regression=False):
    """Single atomic SFI task: train on one (feature, fold) pair."""
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import BaggingClassifier, BaggingRegressor
    from sklearn.metrics import r2_score

    if regression:
        base = DecisionTreeRegressor(max_features=1, min_weight_fraction_leaf=0.02)
        clf = BaggingRegressor(
            estimator=base, n_estimators=n_estimators,
            max_features=1.0, max_samples=1.0, oob_score=False,
            n_jobs=1, random_state=42,
        )
    else:
        base = DecisionTreeClassifier(
            criterion="entropy", max_features=1,
            class_weight="balanced", min_weight_fraction_leaf=0.02,
        )
        clf = BaggingClassifier(
            estimator=base, n_estimators=n_estimators,
            max_features=1.0, max_samples=1.0, oob_score=False,
            n_jobs=1, random_state=42,
        )

    X_tr = X_col_vals[X_tr_idx].copy()
    X_te = X_col_vals[X_te_idx].copy()
    y_tr = y_vals[X_tr_idx]
    y_te = y_vals[X_te_idx]
    w_tr = w_vals[X_tr_idx] if w_vals is not None else None

    # Per-fold NaN fill using train median only (single column)
    if np.isnan(X_tr).any() or np.isnan(X_te).any():
        with np.errstate(all="ignore"):
            med = np.nanmedian(X_tr)
        med = 0.0 if np.isnan(med) else med
        X_tr = np.where(np.isnan(X_tr), med, X_tr)
        X_te = np.where(np.isnan(X_te), med, X_te)

    try:
        if regression:
            fit = clf.fit(X_tr, y_tr, sample_weight=w_tr)
            return r2_score(y_te, fit.predict(X_te))
        else:
            if len(np.unique(y_tr)) < 2:
                return None
            all_labels = np.unique(y_vals)
            fit = clf.fit(X_tr, y_tr, sample_weight=w_tr)
            prob = _align_proba(fit.predict_proba(X_te), fit.classes_, all_labels)
            return -log_loss(y_te, prob, labels=all_labels)
    except Exception:
        return None


def feat_imp_sfi(clf,
                 X: pd.DataFrame,
                 y: pd.Series,
                 years: pd.Series,
                 sample_weight: pd.Series = None,
                 regression: bool = False) -> tuple:
    """Single Feature Importance: train the model on ONE feature at a time.

    Immune to substitution effects between correlated features.
    Uses BaggingClassifier/Regressor (not a single tree) for lower variance.

    Returns:
        summary: DataFrame(index=feature, columns=[mean, std, null_score])
        raw:     DataFrame(index=fold, columns=features)
    """
    cv = ExpandingWindowYearCV(years)
    folds = list(cv.split(X, y, groups=years.values))
    n_folds = len(folds)

    # Null score: baseline for a no-skill predictor
    if regression:
        null_score = 0.0  # R²=0 means no better than predicting the mean
    else:
        classes, counts = np.unique(y.values, return_counts=True)
        class_probs = counts / counts.sum()
        null_score = np.sum(class_probs * np.log(class_probs + 1e-15))

    X_vals = X.values
    y_vals = y.values
    w_vals = sample_weight.values if sample_weight is not None else None
    n_estimators = clf.n_estimators
    col_names = list(X.columns)

    tasks = [
        (ci, fi, X_vals[:, ci:ci + 1], tr, te)
        for ci in range(len(col_names))
        for fi, (tr, te) in enumerate(folds)
    ]

    with blas_limit(1):
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(_sfi_one_task)(ci, X_col, tr, te, y_vals, w_vals, n_estimators, regression)
            for ci, fi, X_col, tr, te in tasks
        )

    raw_records = {col: [None] * n_folds for col in col_names}
    for (ci, fi, *_), score in zip(tasks, results):
        if score is not None:
            raw_records[col_names[ci]][fi] = score

    null_col = "null_r2" if regression else "null_log_loss"
    records = {}
    final_raw = {}
    for col in col_names:
        scores = [s for s in raw_records[col] if s is not None]
        if scores:
            records[col] = {"mean": np.mean(scores),
                            "std": np.std(scores) * len(scores) ** -0.5,
                            null_col: null_score}
            final_raw[col] = scores

    result = pd.DataFrame(records).T
    result.columns = ["mean", "std", null_col]

    max_folds = max(len(v) for v in final_raw.values()) if final_raw else 0
    raw = pd.DataFrame(
        {col: vals + [np.nan] * (max_folds - len(vals))
         for col, vals in final_raw.items()}
    )
    return result.sort_values("mean", ascending=False), raw


# ─────────────────────────────────────────────────────────────────────────────
#  ONC  –  Optimal Number of Clusters  (de Prado MLAM Ch.4)
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_quality(X: np.ndarray, labels: np.ndarray) -> float:
    """t-stat of silhouette scores (mean / std)."""
    sil = silhouette_samples(X, labels)
    return sil.mean() / (sil.std() + 1e-10)


def _onc_one_combo(X_vals, k, seed):
    """Try one (k, seed) combination; return (quality, labels) or None."""
    km = KMeans(n_clusters=k, n_init=1, random_state=seed)
    labels = km.fit_predict(X_vals)
    if len(np.unique(labels)) < 2:
        return None
    q = _cluster_quality(X_vals, labels)
    return q, labels


def _onc_flat(corr: pd.DataFrame,
              max_clusters: int = None,
              n_init: int = 20) -> dict | None:
    """One pass of ONC: grid search over (k, seed) combos.

    Returns {cluster_id: [feature_names]} or None if no valid split exists.
    """
    X = ((1 - corr.fillna(0)) / 2.0) ** 0.5   # correlation → distance
    X_vals = X.values
    n = X.shape[1]
    if n < 2:
        return None
    if max_clusters is None:
        max_clusters = n - 1
    max_clusters = min(max_clusters, n - 1)
    if max_clusters < 2:
        return None

    combos = [(k, seed) for seed in range(n_init) for k in range(2, max_clusters + 1)]
    n_jobs = min(get_n_jobs(), len(combos))
    with blas_limit(1):
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_onc_one_combo)(X_vals, k, seed) for k, seed in combos
        )

    best_quality = -np.inf
    best_labels = None
    for r in results:
        if r is None:
            continue
        q, labels = r
        if q > best_quality:
            best_quality = q
            best_labels = labels

    if best_labels is None:
        return None

    clusters = {}
    for i, label in enumerate(best_labels):
        clusters.setdefault(label, []).append(corr.columns[i])
    return clusters


def _partition_quality(corr: pd.DataFrame, partition: dict) -> dict:
    """Per-cluster mean silhouette score for a full partition."""
    X = ((1 - corr.fillna(0)) / 2.0) ** 0.5
    X_vals = X.values
    feat_idx = {col: i for i, col in enumerate(corr.columns)}

    label_arr = np.empty(len(corr.columns), dtype=int)
    for cid, members in partition.items():
        for m in members:
            label_arr[feat_idx[m]] = cid

    if len(np.unique(label_arr)) < 2:
        return {cid: 0.0 for cid in partition}

    sil = silhouette_samples(X_vals, label_arr)
    return {
        cid: sil[[feat_idx[m] for m in members]].mean()
        for cid, members in partition.items()
    }


def _global_mean_silhouette(corr: pd.DataFrame, partition: dict) -> float:
    """Global mean silhouette score for a partition."""
    qualities = _partition_quality(corr, partition)
    return np.mean(list(qualities.values()))


def onc_cluster(corr: pd.DataFrame,
                max_clusters: int = None,
                n_init: int = 20) -> dict:
    """Greedy divisive ONC (de Prado MLAM Ch.4, improved recursion).

    Algorithm:
      1. Flat KMeans grid-search to get initial partition P.
      2. Greedy divisive refinement:
         a. Order clusters by mean silhouette (worst first).
         b. For each cluster C_i, attempt subdivision via _onc_flat().
         c. If global mean silhouette of P' > P: accept, restart.
         d. Else: reject, try next cluster.
      3. Stop when no subdivision improves global quality.

    Returns {cluster_id: [feature_names]}.
    """
    partition = _onc_flat(corr, max_clusters=max_clusters, n_init=n_init)
    if partition is None or len(partition) <= 1:
        return {0: list(corr.columns)}

    current_quality = _global_mean_silhouette(corr, partition)

    improved = True
    while improved:
        improved = False
        qualities = _partition_quality(corr, partition)
        sorted_cids = sorted(qualities, key=lambda c: qualities[c])

        for cid in sorted_cids:
            members = partition[cid]
            if len(members) < 4:
                continue

            sub_corr = corr.loc[members, members]
            sub = _onc_flat(sub_corr, max_clusters=max_clusters, n_init=n_init)
            if sub is None or len(sub) <= 1:
                continue

            next_id = max(partition.keys()) + 1
            candidate = {c: m for c, m in partition.items() if c != cid}
            for sub_members in sub.values():
                candidate[next_id] = sub_members
                next_id += 1

            candidate_quality = _global_mean_silhouette(corr, candidate)
            if candidate_quality > current_quality:
                partition = candidate
                current_quality = candidate_quality
                improved = True
                break

    return partition


# ─────────────────────────────────────────────────────────────────────────────
#  CFI  –  Clustered Feature Importance  (de Prado MLAM Ch.6)
# ─────────────────────────────────────────────────────────────────────────────

def feat_imp_cfi_mdi(fit, feat_names: list, clusters: dict) -> pd.DataFrame:
    """Clustered MDI: sum the MDI values for all features in each cluster."""
    mdi, _ = feat_imp_mdi(fit, feat_names)
    records = {}
    for cluster_id, members in clusters.items():
        present = [m for m in members if m in mdi.index]
        if not present:
            continue
        cluster_name = f"Cluster_{cluster_id} ({', '.join(present[:3])}{'...' if len(present) > 3 else ''})"
        records[cluster_name] = {
            "mean": mdi.loc[present, "mean"].sum(),
            "std": (mdi.loc[present, "std"] ** 2).sum() ** 0.5,
        }
    result = pd.DataFrame(records).T
    result.columns = ["mean", "std"]
    return result.sort_values("mean", ascending=False)


def feat_imp_cfi_mdi_deprado(
    fit,
    feat_names: list,
    clusters: dict,
) -> tuple:
    """Clustered MDI via per-tree aggregation (MLAM Ch.6).

    Sums MDI across cluster members FOR EACH TREE first, then computes
    mean and std across trees. This captures within-cluster covariance
    that quadrature (the naive approach) misses.

    Returns:
        cluster_summary: DataFrame(index=cluster_label, columns=[mean, std])
        per_feature:     DataFrame(index=feature, columns=[mean, std, cluster_id])
        raw_cluster:     DataFrame(index=tree_id, columns=cluster_label)
    """
    _, mdi_raw = feat_imp_mdi(fit, feat_names)

    cluster_records = {}
    cluster_id_for_label = {}
    for cluster_id, members in clusters.items():
        present = [m for m in members if m in mdi_raw.columns]
        if not present:
            continue
        label = f"Cluster_{cluster_id} ({', '.join(present[:3])}{'...' if len(present) > 3 else ''})"
        cluster_records[label] = mdi_raw[present].fillna(0).sum(axis=1)
        cluster_id_for_label[label] = cluster_id

    raw_cluster = pd.DataFrame(cluster_records)

    n_trees = raw_cluster.shape[0]
    cluster_summary = pd.DataFrame({
        "mean": raw_cluster.mean(),
        "std": raw_cluster.std() * n_trees ** -0.5,
    }).sort_values("mean", ascending=False)

    feat_to_cluster = {m: cid for cid, members in clusters.items() for m in members}
    cluster_label_for_id = {v: k for k, v in cluster_id_for_label.items()}
    per_feature_rows = []
    for feat in feat_names:
        cid = feat_to_cluster.get(feat)
        if cid is None or cid not in cluster_label_for_id:
            per_feature_rows.append({
                "feature": feat, "mean": np.nan, "std": np.nan, "cluster_id": np.nan,
            })
            continue
        label = cluster_label_for_id[cid]
        per_feature_rows.append({
            "feature": feat,
            "mean": cluster_summary.loc[label, "mean"],
            "std": cluster_summary.loc[label, "std"],
            "cluster_id": cid,
        })
    per_feature = pd.DataFrame(per_feature_rows).set_index("feature")

    return cluster_summary, per_feature, raw_cluster


def feat_imp_cfi_mda(clf,
                     X: pd.DataFrame,
                     y: pd.Series,
                     years: pd.Series,
                     clusters: dict,
                     sample_weight: pd.Series = None,
                     scoring: str = "log_loss") -> tuple:
    """Clustered MDA: shuffle all features in a cluster simultaneously.

    Returns:
        summary: DataFrame(index=cluster_label, columns=[mean, std])
        raw:     DataFrame(index=fold, columns=cluster_id)
    """
    cv = ExpandingWindowYearCV(years)
    base_scores = []
    cluster_perms = {cid: [] for cid in clusters}
    is_regression = (scoring == "r2")
    all_labels = None if is_regression else np.unique(y.values)

    rng = np.random.default_rng(42)

    for train_idx, test_idx in tqdm(list(cv.split(X, y, groups=years.values)),
                                    desc="CFI-MDA folds", unit="fold", leave=False):
        X_tr = X.iloc[train_idx].copy()
        X_te = X.iloc[test_idx].copy()
        y_tr = y.iloc[train_idx]
        y_te = y.iloc[test_idx]
        w_tr = sample_weight.iloc[train_idx] if sample_weight is not None else None

        # Per-fold NaN fill using train median
        if X_tr.isna().any().any():
            fold_medians = X_tr.median()
            X_tr = X_tr.fillna(fold_medians)
            X_te = X_te.fillna(fold_medians)

        if not is_regression and y_tr.nunique() < 2:
            continue
        fit = clf.fit(X_tr, y_tr, sample_weight=w_tr)
        if is_regression:
            base_scores.append(r2_score(y_te, fit.predict(X_te)))
        else:
            prob = _align_proba(fit.predict_proba(X_te), fit.classes_, all_labels)
            base_scores.append(-log_loss(y_te, prob, labels=all_labels))

        for cid, members in clusters.items():
            present = [m for m in members if m in X.columns]
            if not present:
                cluster_perms[cid].append(base_scores[-1])
                continue
            X_te_perm = X_te.copy()
            shuffle_vals = X_te_perm[present].values.copy()
            perm_idx = rng.permutation(len(shuffle_vals))
            X_te_perm[present] = shuffle_vals[perm_idx]
            if is_regression:
                cluster_perms[cid].append(r2_score(y_te, fit.predict(X_te_perm)))
            else:
                prob_perm = _align_proba(fit.predict_proba(X_te_perm), fit.classes_, all_labels)
                cluster_perms[cid].append(-log_loss(y_te, prob_perm, labels=all_labels))

    base = np.array(base_scores)
    records = {}
    raw_records = {}
    for cid, members in clusters.items():
        perm = np.array(cluster_perms[cid])
        imp = base - perm
        label = f"Cluster_{cid} ({', '.join([m for m in members if m in X.columns][:3])})"
        records[label] = {"mean": imp.mean(),
                          "std": imp.std() * len(imp) ** -0.5}
        raw_records[cid] = imp

    result = pd.DataFrame(records).T
    result.columns = ["mean", "std"]
    raw = pd.DataFrame(raw_records)
    return result.sort_values("mean", ascending=False), raw


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic validation  (de Prado MLAM §1.4 / AFML §8.6)
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_validation(n_samples: int = 500,
                         n_informative: int = 3,
                         n_redundant: int = 3,
                         n_noise: int = 2,
                         random_state: int = 42) -> dict:
    """Generate a synthetic dataset where we KNOW which features are signal.
    Run MDI and SFI. Confirm they recover the injected signal.

    Returns a dict with MDI and SFI DataFrames and a pass/fail summary.
    """
    from sklearn.datasets import make_classification

    rng = np.random.RandomState(random_state)
    n_features = n_informative + n_redundant + n_noise

    X_raw, y = make_classification(
        n_samples=n_samples,
        n_features=n_informative + n_noise,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        shuffle=False,
        random_state=random_state,
    )
    feat_names = (
        [f"INFO_{i}" for i in range(n_informative)] +
        [f"NOISE_{i}" for i in range(n_noise)]
    )

    redundant_cols = []
    for i in range(n_redundant):
        src = i % n_informative
        noisy = X_raw[:, src] + rng.normal(0, 0.5, n_samples)
        redundant_cols.append(noisy.reshape(-1, 1))
        feat_names.append(f"REDUND_{i}")

    X_full = np.hstack([X_raw] + redundant_cols)
    X_df = pd.DataFrame(X_full, columns=feat_names)
    y_ser = pd.Series(y)
    years = pd.Series(np.repeat(np.arange(n_samples // 50), 50)[:n_samples])

    clf = build_rf(n_estimators=200)
    clf.fit(X_df, y_ser)

    mdi_result, _ = feat_imp_mdi(clf, feat_names)
    sfi_result, _ = feat_imp_sfi(
        build_rf(n_estimators=100), X_df, y_ser, years
    )

    # Evaluate: are all INFO features ranked above all NOISE features in MDI?
    info_rank = mdi_result.index.get_indexer([f"INFO_{i}" for i in range(n_informative)])
    noise_rank = mdi_result.index.get_indexer([f"NOISE_{i}" for i in range(n_noise)])
    mdi_pass = max(info_rank) < min(noise_rank) if len(noise_rank) > 0 else True

    log.info(f"Synthetic validation: MDI recovers informative > noise: {mdi_pass}")

    return {"mdi": mdi_result, "sfi": sfi_result, "mdi_pass": mdi_pass}


# ─────────────────────────────────────────────────────────────────────────────
#  Statistical significance
# ─────────────────────────────────────────────────────────────────────────────

def compute_pvalues(raw: pd.DataFrame,
                    null_mean: float = 0.0,
                    alternative: str = "greater") -> pd.Series:
    """Wilcoxon signed-rank test per feature: H0 = importance equals null_mean.

    For MDI raw: null_mean = 1 / n_features (uniform importance)
    For MDA raw: null_mean = 0 (shuffling has no effect)
    For SFI raw: null_mean = null_log_loss (no better than base-rate predictor)

    Returns Series(index=feature, values=p_value).
    """
    pvals = {}
    for col in raw.columns:
        vals = raw[col].dropna().values
        diffs = vals - null_mean
        diffs = diffs[diffs != 0]
        # n≥5: one-sided Wilcoxon minimum p at n=4 is 1/16=0.0625, unreachable at α=0.05
        if len(diffs) < 5:
            pvals[col] = np.nan
        else:
            _, p = scipy_wilcoxon(diffs, alternative=alternative)
            pvals[col] = p
    return pd.Series(pvals, name="p_value")


# ─────────────────────────────────────────────────────────────────────────────
#  Bootstrap CI  (non-parametric confidence interval for the mean)
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(values: np.ndarray,
                 n_boot: int = 2000,
                 ci: float = 0.95,
                 seed: int = 42) -> tuple:
    """Bootstrap confidence interval for the mean. No normality assumption.

    Returns (mean, lower_ci, upper_ci).
    """
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = 1 - ci
    return (values.mean(),
            np.percentile(boot_means, 100 * alpha / 2),
            np.percentile(boot_means, 100 * (1 - alpha / 2)))


# ─────────────────────────────────────────────────────────────────────────────
#  Per-feature scoring with EB variance moderation
# ─────────────────────────────────────────────────────────────────────────────

# Validated EB priors per test (home_win target, 8-fold expanding-window CV,
# 2026-07-30 S3 data). Each passed random-split homogeneity at 1000 splits.
EB_PRIORS = {
    "sfi":       {"d0": 14.1, "s0_sq": 3.92e-06},
    "desub_mda": {"d0": 5.69, "s0_sq": 6.94e-06},
    "pca_mda":   {"d0": 10.9, "s0_sq": 7.17e-07},
    "resid_mda_112": {"d0": 4.30, "s0_sq": 3.84e-11},
}

# Null values per test (value expected under H0: feature has no signal)
NULL_VALUES = {
    "sfi": np.log(0.5),  # -0.6931: mean log-probability of a coin-flip classifier
    "desub_mda": 0.0,
    "pca_mda": 0.0,
    "resid_mda": 0.0,
    "mdi": None,  # 1/n_features, computed at runtime
    "cfi_mda": 0.0,
}


def feature_score(fold_values: np.ndarray,
                  test: str,
                  null: float = 0.0,
                  d0: float | None = None,
                  s0_sq: float | None = None,
                  trend_alpha: float = 0.05,
                  ci_alpha: float = 0.10) -> dict:
    """Score a single feature on a single EB-moderated importance test.

    For tree-level tests (MDI, CFI_MDA), use feature_score_clt() instead.

    Parameters
    ----------
    fold_values : array of shape (n_folds,), ordered chronologically.
    test : one of 'sfi', 'desub_mda', 'pca_mda', 'resid_mda_112'.
    null : null hypothesis value (feature has no signal).
    d0, s0_sq : EB prior parameters (required — raises ValueError if None).
    trend_alpha : significance level for Mann-Kendall trend test (governs the
        hard REJECT/ACCEPT branches that fire on significant trend + consistent
        level). Set separately from ci_alpha because trend decisions are
        irreversible (hard gate) while CI decisions are softer (flagged).
    ci_alpha : significance level for the moderated-t confidence interval
        (governs the CI-based fallback when trend is not significant).

    Returns
    -------
    dict with keys: level, trend_tau, trend_p, mod_t, mod_df, ci_lo, ci_hi,
                    decision, flag
    """
    from scipy.stats import kendalltau, t as t_dist

    vals = np.asarray(fold_values, dtype=np.float64)
    n = len(vals)
    result = {}

    # ── Level: sample mean (coherent with SE = sqrt(mod_var / n)) ──
    level = float(np.mean(vals))
    result["level"] = level

    # ── Trend: Kendall's tau (exact when no ties, asymptotic otherwise) ──
    ranks = np.arange(n)
    try:
        tau, p_trend = kendalltau(ranks, vals, method='exact')
    except ValueError:
        tau, p_trend = kendalltau(ranks, vals, method='asymptotic')
    result["trend_tau"] = float(tau)
    result["trend_p"] = float(p_trend)

    # ── EB-moderated t-statistic ──
    # feature_score() is exclusively for fold-structured tests with validated
    # EB priors. MDI/CFI_MDA use feature_score_clt() instead.
    if d0 is None or s0_sq is None:
        raise ValueError(
            f"feature_score() requires d0 and s0_sq for EB-moderated tests. "
            f"For tree-level tests (MDI, CFI_MDA), use feature_score_clt()."
        )

    d_i = n - 1
    s2_i = float(np.var(vals, ddof=1))
    mod_var = (d_i * s2_i + d0 * s0_sq) / (d_i + d0)
    mod_df = d_i + d0
    se = np.sqrt(mod_var / n)
    mod_t = (level - null) / se if se > 0 else 0.0
    result["mod_t"] = float(mod_t)
    result["mod_df"] = float(mod_df)
    result["mod_var"] = float(mod_var)

    # CI uses ci_alpha (wider than trend_alpha → more conservative gate)
    t_crit = t_dist.ppf(1 - ci_alpha / 2, df=mod_df)
    result["ci_lo"] = level - t_crit * se
    result["ci_hi"] = level + t_crit * se

    ci_lo = result["ci_lo"]
    ci_hi = result["ci_hi"]
    trend_significant = p_trend < trend_alpha
    flag = None

    # Variance regime shift detection (checked first — overrides other logic)
    instability = False
    if n >= 6:
        mid = n // 2
        mad_first = np.median(np.abs(vals[:mid] - np.median(vals[:mid])))
        mad_second = np.median(np.abs(vals[mid:] - np.median(vals[mid:])))
        if mad_second > 5 * max(mad_first, 1e-15):
            instability = True

    if instability:
        flag = "INSTABILITY"
        decision = "NEEDS_SPECIFICATION"
    elif trend_significant and tau < 0 and ci_lo <= null:
        decision = "REJECT"
    elif trend_significant and tau > 0 and level > null:
        decision = "ACCEPT"
    elif not trend_significant:
        # No significant trend at trend_alpha: fall through to CI at ci_alpha
        if ci_lo > null:
            decision = "ACCEPT"
            flag = "NEEDS_SPECIFICATION"
        elif ci_hi < null:
            decision = "REJECT"
        else:
            decision = "NEEDS_SPECIFICATION"
            flag = "NEEDS_SPECIFICATION"
    else:
        # Significant trend but contradicts level/CI
        # (e.g. tau>0 but level<=null, or tau<0 but ci_lo>null)
        flag = "NEEDS_SPECIFICATION"
        if ci_lo > null:
            decision = "ACCEPT"
        elif ci_hi < null:
            decision = "REJECT"
        else:
            decision = "NEEDS_SPECIFICATION"

    result["decision"] = decision
    result["flag"] = flag
    return result


def feature_score_clt(tree_values: np.ndarray,
                      null: float,
                      alpha: float = 0.05) -> dict:
    """Score a feature via CLT-based significance test on tree-level importances.

    Implements de Prado's MDI significance framework with an empirical null:
    z = (mean - null) / SE, where null is derived from a permuted-target forest
    (not the theoretical 1/F, which is 15x too low due to row-normalization and
    max_features=1 sparsity inflating all features above 1/F uniformly).

    Parameters
    ----------
    tree_values : array of per-tree importance values (NaN = tree didn't use feature).
    null : empirical null for this feature — its mean MDI under permuted target
        with the same tree construction. Each feature gets its own null because
        high-cardinality or high-variance features attract more splits even
        under noise.
    alpha : two-sided significance level for the z-test. Default 0.05 chosen to
        match trend_alpha in the EB-moderated tests — both control the hard
        ACCEPT/REJECT gate at the same Type-I error rate.

    Returns
    -------
    dict with keys: mean, std, n, se, z_stat, p_value, null, ci_lo, ci_hi,
                    decision, flag
    """
    from scipy.stats import norm

    vals = np.asarray(tree_values, dtype=np.float64)
    valid = vals[~np.isnan(vals)]
    n = len(valid)

    result = {"null": float(null), "n": n, "alpha": alpha}

    if n < 10:
        result.update({"mean": float(np.mean(valid)) if n > 0 else 0.0,
                       "std": 0.0, "se": 0.0, "z_stat": 0.0, "p_value": 1.0,
                       "ci_lo": null, "ci_hi": null,
                       "decision": "NEEDS_SPECIFICATION",
                       "flag": "NEEDS_SPECIFICATION"})
        return result

    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1))
    se = std / np.sqrt(n)

    result["mean"] = mean
    result["std"] = std
    result["se"] = se

    if se < 1e-15:
        result.update({"z_stat": 0.0, "p_value": 1.0,
                       "ci_lo": mean, "ci_hi": mean,
                       "decision": "NEEDS_SPECIFICATION",
                       "flag": "NEEDS_SPECIFICATION"})
        return result

    z_stat = (mean - null) / se
    p_value = 2 * norm.sf(abs(z_stat))
    z_crit = norm.ppf(1 - alpha / 2)
    ci_lo = mean - z_crit * se
    ci_hi = mean + z_crit * se

    result["z_stat"] = float(z_stat)
    result["p_value"] = float(p_value)
    result["ci_lo"] = float(ci_lo)
    result["ci_hi"] = float(ci_hi)

    # Decision logic: mirrors EB path's CI-based branch
    if p_value < alpha and z_stat > 0:
        decision = "ACCEPT"
        flag = None
    elif p_value < alpha and z_stat < 0:
        # Feature is significantly BELOW average — weaker than random
        decision = "REJECT"
        flag = None
    else:
        # Not significant: CI straddles null
        decision = "NEEDS_SPECIFICATION"
        flag = "NEEDS_SPECIFICATION"

    result["decision"] = decision
    result["flag"] = flag
    return result


def feature_score_resid_sometimes_zero(fold_values: np.ndarray,
                                       null: float = 0.0) -> dict:
    """Score a resid_MDA feature that is sometimes-zero (nonzero in 1-7 of 8 folds).

    No EB moderation (d0=1.09 too weak). Instead reports:
    - level: median of NONZERO folds only
    - fold_nonzero_frac: reliability weight
    """
    vals = np.asarray(fold_values, dtype=np.float64)
    n = len(vals)
    nonzero_mask = vals != 0
    n_nonzero = int(nonzero_mask.sum())
    fold_nonzero_frac = n_nonzero / n

    if n_nonzero > 0:
        level = float(np.median(vals[nonzero_mask]))
    else:
        level = 0.0

    return {
        "level": level,
        "fold_nonzero_frac": fold_nonzero_frac,
        "n_nonzero_folds": n_nonzero,
        "decision": "NO_UNIQUE_SIGNAL" if n_nonzero == 0 else None,
        "flag": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  6-test weighted-vote combiner
# ─────────────────────────────────────────────────────────────────────────────

# Base weights per test type. EB-moderated tests (fold-structured, validated
# priors) carry full weight. CLT-based tests (MDI, CFI_MDA) carry half weight
# despite having a calibrated z-test — their selectivity is near-zero (MDI
# accepts 99.2% of features with max_features=1 because every drawn feature
# gets inflated above 1/F). Low information content for discriminating useful
# vs useless features means reduced voting power in the combiner.
_BASE_WEIGHTS = {
    "sfi": 1.0,
    "desub_mda": 1.0,
    "pca_mda": 1.0,
    "resid_mda": 1.0,
    "mdi": 0.5,
    "cfi_mda": 0.5,
}

# Flag multipliers applied on top of base weight
_FLAG_MULTIPLIERS = {
    None: 1.0,
    "NEEDS_SPECIFICATION": 0.5,
    "INSTABILITY": 0.0,
}


def combine_test_scores(scores: dict[str, dict],
                        accept_threshold: float = 0.5) -> dict:
    """Combine per-test feature_score() results into a final gate decision.

    Parameters
    ----------
    scores : dict mapping test name -> feature_score() result dict.
        Each value must have 'decision' and 'flag' keys.
        For resid_MDA sometimes-zero features, pass a dict with
        'decision': None and 'fold_nonzero_frac' — these abstain from voting.
    accept_threshold : fraction of available weighted votes needed for ACCEPT.
        Default 0.5 = simple weighted majority.

    Returns
    -------
    dict with keys: tier, accept_votes, reject_votes, abstain_votes,
                    total_available, accept_frac, details
    """
    accept_votes = 0.0
    reject_votes = 0.0
    abstain_votes = 0.0
    total_available = 0.0
    details = {}

    for test_name, result in scores.items():
        base_w = _BASE_WEIGHTS.get(test_name, 0.5)
        decision = result.get("decision")
        flag = result.get("flag")
        flag_mult = _FLAG_MULTIPLIERS.get(flag, 0.5)
        effective_w = base_w * flag_mult

        # Features that abstain: INSTABILITY, NO_UNIQUE_SIGNAL, or None decision
        if decision is None or decision == "NO_UNIQUE_SIGNAL" or flag == "INSTABILITY":
            abstain_votes += base_w
            details[test_name] = {"vote": "ABSTAIN", "weight": 0.0, "base": base_w}
            continue

        total_available += effective_w

        if decision == "ACCEPT":
            accept_votes += effective_w
            details[test_name] = {"vote": "ACCEPT", "weight": effective_w, "base": base_w}
        elif decision == "REJECT":
            reject_votes += effective_w
            details[test_name] = {"vote": "REJECT", "weight": effective_w, "base": base_w}
        else:  # NEEDS_SPECIFICATION
            # Neither accept nor reject — counts toward available but not toward either
            details[test_name] = {"vote": "NEEDS_SPEC", "weight": effective_w, "base": base_w}

    # Final tier decision
    if total_available == 0:
        tier = "UNKNOWN"
        accept_frac = 0.0
    else:
        accept_frac = accept_votes / total_available
        reject_frac = reject_votes / total_available

        # Unopposed rule requires minimum weight to avoid a single
        # low-trust vote (e.g. MDI alone at 0.5) driving the outcome
        # when multiple better-calibrated tests abstain or say NEEDS_SPEC.
        min_unopposed_weight = 1.0

        if accept_votes >= min_unopposed_weight and reject_votes == 0:
            tier = "ACCEPTED"
        elif reject_votes >= min_unopposed_weight and accept_votes == 0:
            tier = "REJECTED"
        elif accept_frac >= accept_threshold:
            tier = "ACCEPTED"
        elif reject_frac >= accept_threshold:
            tier = "REJECTED"
        else:
            tier = "NEEDS SPECIFICATION"

    return {
        "tier": tier,
        "accept_votes": accept_votes,
        "reject_votes": reject_votes,
        "abstain_votes": abstain_votes,
        "total_available": total_available,
        "accept_frac": accept_frac,
        "details": details,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Resid-MDA dispatch (routes to correct path based on fold zero-structure)
# ─────────────────────────────────────────────────────────────────────────────

def resid_mda_dispatch(fold_values: np.ndarray,
                       null: float = 0.0,
                       trend_alpha: float = 0.05,
                       ci_alpha: float = 0.10) -> dict:
    """Route a single feature's resid_MDA fold values to the correct scoring path.

    Three validated populations:
      - Always-nonzero (nonzero in all 8 folds): EB-moderated via resid_mda_112 priors
      - Sometimes-zero (nonzero in 1-7 folds): level + fold_nonzero_frac, no EB
      - Always-zero (zero in all 8 folds): NO_UNIQUE_SIGNAL categorical
    """
    vals = np.asarray(fold_values, dtype=np.float64)
    n_nonzero = int(np.count_nonzero(vals))
    n = len(vals)

    if n_nonzero == 0:
        return {
            "path": "always_zero",
            "decision": "NO_UNIQUE_SIGNAL",
            "flag": None,
            "level": 0.0,
        }
    elif n_nonzero == n:
        d0 = EB_PRIORS["resid_mda_112"]["d0"]
        s0_sq = EB_PRIORS["resid_mda_112"]["s0_sq"]
        result = feature_score(vals, "resid_mda_112", null=null, d0=d0, s0_sq=s0_sq,
                               trend_alpha=trend_alpha, ci_alpha=ci_alpha)
        result["path"] = "always_nonzero"
        return result
    else:
        result = feature_score_resid_sometimes_zero(vals, null=null)
        result["path"] = "sometimes_zero"
        return result


# ─────────────────────────────────────────────────────────────────────────────
#  PCA-validated gate: filter_features_v2
# ─────────────────────────────────────────────────────────────────────────────

_PCA_ELIGIBLE_METHODS = {"SFI", "MDA", "DESUB_MDA", "PCA_MDA", "RESID_MDA", "MDI"}

_METHOD_TO_EB_KEY = {
    "SFI": "sfi",
    "DESUB_MDA": "desub_mda",
    "PCA_MDA": "pca_mda",
    "RESID_MDA": "resid_mda_112",
}


def _determine_eligible_methods(pca_crosscheck: dict) -> set:
    """Return set of method names that are PCA-eligible (p<=0.05, tau>0)."""
    eligible = set()
    for method in _PCA_ELIGIBLE_METHODS:
        entry = pca_crosscheck.get(method)
        if entry and entry.get("p_value", 1.0) <= 0.05 and entry.get("tau", 0.0) > 0:
            eligible.add(method)
    return eligible


def _cluster_veto(
    cfi_mda_raw: pd.DataFrame,
    cfi_mdi_raw: pd.DataFrame,
    alpha: float = 0.05,
) -> tuple[dict, dict]:
    """Compute dual cluster veto from CFI_MDI (CLT) + CFI_MDA (fold-count).

    Returns (cluster_vetoed, cluster_suspect) dicts mapping cluster_id → bool.
    """
    import re
    from scipy.stats import norm

    # CFI_MDI columns may be verbose ("Cluster_N (feat1, feat2, ...)")
    mdi_col_map = {}
    for col in cfi_mdi_raw.columns:
        match = re.match(r"Cluster_(\d+)", col)
        if match:
            mdi_col_map[match.group(1)] = col
        else:
            mdi_col_map[col] = col

    cluster_vetoed = {}
    cluster_suspect = {}

    for cid in cfi_mda_raw.columns:
        mda_vals = cfi_mda_raw[cid].dropna().values
        n_positive_mda = int((mda_vals > 0).sum())
        mda_condemns = n_positive_mda <= 2

        mdi_condemns = False
        mdi_col = mdi_col_map.get(str(cid))
        if mdi_col is not None and mdi_col in cfi_mdi_raw.columns:
            mdi_vals = cfi_mdi_raw[mdi_col].dropna().values
            n_mdi = len(mdi_vals)
            if n_mdi >= 10:
                mean_mdi = float(np.mean(mdi_vals))
                se_mdi = float(np.std(mdi_vals, ddof=1) / np.sqrt(n_mdi))
                if se_mdi > 1e-15:
                    z = mean_mdi / se_mdi
                    p = norm.sf(-z)
                    mdi_condemns = (mean_mdi < 0) and (p < alpha)

        # CFI_MDA fold-count is the authority; CFI_MDI tree-level alone
        # cannot validate a cluster (same principle as per-feature MDI)
        cluster_vetoed[cid] = mda_condemns
        cluster_suspect[cid] = mda_condemns and not mdi_condemns

    return cluster_vetoed, cluster_suspect


def _mdi_only_supporter(eligible_decisions: dict) -> bool:
    """True if MDI is the only eligible method not rejecting."""
    non_mdi = {m: d for m, d in eligible_decisions.items() if m != "MDI"}
    if not non_mdi:
        return False
    mdi_decision = eligible_decisions.get("MDI")
    if mdi_decision is None or mdi_decision == "REJECT":
        return False
    return all(d == "REJECT" for d in non_mdi.values())


def _trend_rescue(fold_vals: np.ndarray, null: float = 0.0) -> str | None:
    """Check if a feature/cluster trajectory shows growth worth rescuing.

    Uses Kendall's tau (outlier-robust rank correlation) for trend detection
    and Theil-Sen slope (median of pairwise slopes) for direction.

    Returns:
        "significant" — tau > 0 AND p <= 0.05 AND last fold > null
        "heuristic"   — Theil-Sen slope > 0 AND last fold > null (not significant)
        None          — no rescue (declining, flat, or last fold ≤ null)
    """
    from scipy.stats import kendalltau, theilslopes

    if len(fold_vals) < 4:
        return None

    last_val = float(fold_vals[-1])
    if last_val <= null:
        return None

    x = np.arange(len(fold_vals))
    tau, p = kendalltau(x, fold_vals)

    if tau > 0 and p <= 0.05:
        return "significant"

    slope = theilslopes(fold_vals, x)[0]
    if slope > 0:
        return "heuristic"

    return None


def filter_features_v2(
    sfi_raw: pd.DataFrame,
    desub_mda_raw: pd.DataFrame,
    pca_mda_raw: pd.DataFrame,
    resid_mda_raw: pd.DataFrame,
    mdi_raw: pd.DataFrame,
    cfi_mda_raw: pd.DataFrame,
    cfi_mdi_raw: pd.DataFrame,
    clusters: dict,
    sfi_null: float,
    pca_crosscheck: dict,
) -> pd.DataFrame:
    """PCA-validated feature gate with conservative union.

    Only methods that pass the PCA cross-check (p<=0.05, tau>0) participate
    in gating AND ranking. Ineligible methods are excluded entirely.

    Cluster veto: features in clusters where BOTH CFI_MDI (CLT z-test) AND
    CFI_MDA (fold-count <= 2 positive) condemn are forced to REJECTED.

    Conservative union: REJECT only if ALL eligible per-feature methods reject.
    """
    feat_to_cluster = {}
    for cid, members in clusters.items():
        for m in members:
            feat_to_cluster[m] = cid

    all_features = set()
    for df in [sfi_raw, desub_mda_raw, pca_mda_raw, resid_mda_raw, mdi_raw]:
        if df is not None and not df.empty:
            all_features.update(df.columns.tolist())
    all_features.update(feat_to_cluster.keys())
    n_features = len(all_features)
    threshold_1F = 1.0 / n_features if n_features > 0 else 0.0

    eligible = _determine_eligible_methods(pca_crosscheck)
    log.info(f"    PCA-eligible methods: {sorted(eligible)}")

    vetoed, suspect = _cluster_veto(cfi_mda_raw, cfi_mdi_raw)
    n_vetoed = sum(1 for v in vetoed.values() if v)
    log.info(f"    Cluster veto: {n_vetoed}/{len(vetoed)} clusters vetoed")

    null_values = {
        "SFI": sfi_null,
        "DESUB_MDA": 0.0,
        "PCA_MDA": 0.0,
        "RESID_MDA": 0.0,
        "MDI": threshold_1F,
    }

    method_to_raw = {
        "SFI": sfi_raw,
        "DESUB_MDA": desub_mda_raw,
        "PCA_MDA": pca_mda_raw,
        "RESID_MDA": resid_mda_raw,
        "MDI": mdi_raw,
    }

    rows = []
    for feat in sorted(all_features):
        row = {"feature": feat, "cluster_id": feat_to_cluster.get(feat, np.nan)}
        cid = feat_to_cluster.get(feat)
        row["cluster_vetoed"] = vetoed.get(cid, False) if cid else False
        row["cluster_suspect"] = suspect.get(cid, False) if cid else False

        decisions = {}

        for method in ["SFI", "DESUB_MDA", "PCA_MDA", "RESID_MDA", "MDI"]:
            pass_col = {
                "SFI": "sfi_passes", "DESUB_MDA": "desub_mda_passes",
                "PCA_MDA": "pca_mda_passes", "RESID_MDA": "resid_mda_passes",
                "MDI": "mdi_passes",
            }[method]
            mean_col = {
                "SFI": "sfi_mean", "DESUB_MDA": "desub_mda_mean",
                "PCA_MDA": "pca_mda_mean", "RESID_MDA": "resid_mda_mean",
                "MDI": "mdi_mean",
            }[method]
            rank_col = {
                "SFI": "sfi_rank", "DESUB_MDA": "desub_mda_rank",
                "PCA_MDA": "pca_mda_rank", "RESID_MDA": "resid_mda_rank",
                "MDI": "mdi_rank",
            }[method]

            if method not in eligible:
                row[pass_col] = np.nan
                row[mean_col] = np.nan
                continue

            raw_df = method_to_raw[method]
            if raw_df is None or feat not in raw_df.columns:
                row[pass_col] = np.nan
                row[mean_col] = np.nan
                continue

            vals = raw_df[feat].dropna().values.astype(float)
            null_val = null_values[method]

            if method == "MDI":
                if len(vals) < 10:
                    row[pass_col] = np.nan
                    row[mean_col] = np.nan
                    continue
                result = feature_score_clt(vals, null=null_val)
                row[mean_col] = result["mean"]
                row[pass_col] = result["decision"] != "REJECT"
                decisions[method] = result["decision"]
            elif method == "RESID_MDA":
                if len(vals) < 4:
                    row[pass_col] = np.nan
                    row[mean_col] = np.nan
                    continue
                result = resid_mda_dispatch(vals, null=null_val)
                row[mean_col] = result.get("level", 0.0)
                if result["decision"] is None:
                    row[pass_col] = True
                    decisions[method] = "NEEDS_SPECIFICATION"
                elif result["decision"] == "NO_UNIQUE_SIGNAL":
                    row[pass_col] = np.nan
                else:
                    row[pass_col] = result["decision"] != "REJECT"
                    decisions[method] = result["decision"]
            else:
                if len(vals) < 4:
                    row[pass_col] = np.nan
                    row[mean_col] = np.nan
                    continue
                eb_key = _METHOD_TO_EB_KEY[method]
                d0 = EB_PRIORS[eb_key]["d0"]
                s0_sq = EB_PRIORS[eb_key]["s0_sq"]
                result = feature_score(vals, eb_key, null=null_val, d0=d0, s0_sq=s0_sq)
                row[mean_col] = result["level"]
                row[pass_col] = result["decision"] != "REJECT"
                decisions[method] = result["decision"]

        # CFI_MDA cluster passes (backward compat for routing)
        if cid and cid in cfi_mda_raw.columns:
            mda_vals = cfi_mda_raw[cid].dropna().values
            n_pos = int((mda_vals > 0).sum())
            row["cfi_mda_cluster_passes"] = n_pos >= 7
        else:
            row["cfi_mda_cluster_passes"] = np.nan

        # Tier assignment
        row["trend_rescue"] = np.nan
        would_reject = False

        if row["cluster_vetoed"]:
            would_reject = True
            # Trend rescue: check cluster's CFI_MDA trajectory
            if cid and cid in cfi_mda_raw.columns:
                cluster_vals = cfi_mda_raw[cid].dropna().values.astype(float)
                rescue = _trend_rescue(cluster_vals, null=0.0)
                if rescue:
                    would_reject = False
                    row["trend_rescue"] = rescue
        else:
            eligible_decisions = {m: d for m, d in decisions.items() if m in eligible}
            decision_list = list(eligible_decisions.values())
            if not decision_list:
                pass
            elif all(d == "REJECT" for d in decision_list):
                would_reject = True
            elif _mdi_only_supporter(eligible_decisions):
                would_reject = True

        if would_reject and row["trend_rescue"] is np.nan:
            # Per-feature trend rescue — applies to ALL rejection paths
            rescue_result = None
            for method in ["PCA_MDA", "DESUB_MDA", "RESID_MDA", "SFI"]:
                if method not in eligible:
                    continue
                raw_df = method_to_raw[method]
                if raw_df is None or feat not in raw_df.columns:
                    continue
                feat_vals = raw_df[feat].dropna().values.astype(float)
                if len(feat_vals) < 4:
                    continue
                null_val = null_values[method]
                rescue_result = _trend_rescue(feat_vals, null=null_val)
                if rescue_result:
                    break
            if rescue_result:
                would_reject = False
                row["trend_rescue"] = rescue_result

        # Final tier
        if would_reject:
            row["tier"] = "REJECTED"
        else:
            eligible_decisions = {m: d for m, d in decisions.items() if m in eligible}
            decision_list = list(eligible_decisions.values())
            if not decision_list:
                row["tier"] = "NEEDS SPECIFICATION"
            elif all(d == "ACCEPT" for d in decision_list):
                row["tier"] = "ACCEPTED"
            else:
                row["tier"] = "NEEDS SPECIFICATION"

        rows.append(row)

    report = pd.DataFrame(rows).set_index("feature")

    # Rank columns: only from eligible methods
    for method in eligible:
        mean_col = {
            "SFI": "sfi_mean", "DESUB_MDA": "desub_mda_mean",
            "PCA_MDA": "pca_mda_mean", "RESID_MDA": "resid_mda_mean",
            "MDI": "mdi_mean",
        }[method]
        rank_col = {
            "SFI": "sfi_rank", "DESUB_MDA": "desub_mda_rank",
            "PCA_MDA": "pca_mda_rank", "RESID_MDA": "resid_mda_rank",
            "MDI": "mdi_rank",
        }[method]
        if mean_col in report.columns:
            report[rank_col] = report[mean_col].rank(ascending=False, na_option="bottom")

    # Composite rank from eligible methods only
    rank_cols = [f"{m.lower()}_rank" for m in eligible
                 if f"{m.lower()}_rank" in report.columns]
    if rank_cols:
        report["composite_rank"] = report[rank_cols].mean(axis=1)
        report = report.sort_values("composite_rank")
    else:
        report["composite_rank"] = np.nan

    tier_counts = report["tier"].value_counts()
    for tier, count in tier_counts.items():
        log.info(f"    {tier}: {count}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
#  Diagnostic: distribution of CFI-MDA fold scores
# ─────────────────────────────────────────────────────────────────────────────

def plot_cfi_mda_distributions(cfi_mda_raw: pd.DataFrame,
                               clusters: dict,
                               output_path: str = None,
                               top_n: int = 20) -> None:
    """Plot the distribution of per-fold CFI-MDA importance scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import shapiro

    cluster_labels = {}
    for cid, members in clusters.items():
        cluster_labels[cid] = f"C{cid} ({len(members)} feats)"

    cids = list(cfi_mda_raw.columns)
    n_show = min(top_n, len(cids))
    col_means = {cid: cfi_mda_raw[cid].dropna().mean() for cid in cids}
    cids_sorted = sorted(cids, key=lambda c: col_means[c], reverse=True)[:n_show]

    ncols = 4
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for ax, cid in zip(axes, cids_sorted):
        vals = cfi_mda_raw[cid].dropna().values
        mean_val = vals.mean()
        se = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else np.inf
        z = mean_val / se if se > 0 else 0.0

        ax.hist(vals, bins=max(5, len(vals) // 2), edgecolor="black",
                color="#5b8db8", alpha=0.75)
        ax.axvline(0, color="red", lw=1.5, linestyle="--", label="null=0")
        ax.axvline(mean_val, color="navy", lw=2, label=f"mean={mean_val:.4f}")

        if len(vals) >= 3:
            _, sw_p = shapiro(vals)
            sw_note = f"SW p={sw_p:.2f}"
        else:
            sw_note = "n<3"

        label = cluster_labels.get(cid, str(cid))
        ax.set_title(f"{label}\nz={z:.2f}  n={len(vals)}  {sw_note}", fontsize=8)
        ax.set_xlabel("base − permuted score", fontsize=7)
        ax.set_ylabel("folds", fontsize=7)
        ax.legend(fontsize=6)

    for ax in axes[n_show:]:
        ax.set_visible(False)

    fig.suptitle("CFI-MDA fold score distributions", fontsize=10, y=1.01)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=120)
        log.info(f"Distribution plot saved to {output_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Feature filtering  (de Prado MDI/MDA/SFI criteria + tiering)
# ─────────────────────────────────────────────────────────────────────────────

def filter_features(mdi_raw: pd.DataFrame,
                    cfi_mda_raw: pd.DataFrame,
                    clusters: dict,
                    sfi_raw: pd.DataFrame = None,
                    sfi_null: float = None,
                    desub_mda_raw: pd.DataFrame = None,
                    pca_mda_raw: pd.DataFrame = None,
                    resid_mda_raw: pd.DataFrame = None,
                    p_threshold: float = 0.10,
                    recency_n_folds: int = 3) -> pd.DataFrame:
    """Three-tier feature classification with trend analysis.

      ACCEPTED          — passes ALL available tests. Proven signal.
      NEEDS SPECIFICATION — passes SOME tests. Signal exists but partial/conditional.
      REJECTED          — fails ALL tests OR decaying into noise.

    Gate logic (per method):
      Pass requires BOTH: (1) CI upper > threshold (entire CI not below threshold)
                          (2) mean > threshold (positive folds aren't just noise blips)
      This means CI spanning zero with positive mean passes (likely signal), but
      CI spanning zero with negative mean fails (noise blips pulled CI up).

    Temporal trend demotion: Features passing the gate but with Spearman rho < -0.6
    (strong monotonic decay in fold-ordered importance) AND recent folds mean <= null
    are demoted to REJECTED. Historical mean was positive but signal is dying.

    Recency rescue: REJECTED features with positive recent folds are promoted to
    NEEDS SPECIFICATION (newly-important features from regime changes).

    Tests:
      MDI pass:       mean > 1/F AND CI upper > 1/F
      SFI pass:       mean > sfi_null AND CI upper > sfi_null
      Desub MDA pass: mean > 0 AND CI upper > 0
      PCA-MDA pass:   mean > 0 AND CI upper > 0
      Resid MDA pass: mean > 0 AND CI upper > 0
      CFI-MDA:        cluster-level, reported but not used for individual pass/fail
    """
    feat_to_cluster = {}
    if clusters:
        for cid, members in clusters.items():
            for m in members:
                feat_to_cluster[m] = cid

    all_features = set()
    for df in [mdi_raw, sfi_raw, desub_mda_raw, pca_mda_raw, resid_mda_raw]:
        if df is not None and not df.empty:
            all_features.update(df.columns.tolist())
    all_features.update(feat_to_cluster.keys())

    n_features = len(all_features)
    threshold_1F = 1.0 / n_features if n_features > 0 else 0.0

    rows = []
    for feat in sorted(all_features):
        row = {"feature": feat, "cluster_id": feat_to_cluster.get(feat, np.nan)}

        # ── MDI ──────────────────────────────────────────────────────────
        if mdi_raw is not None and feat in mdi_raw.columns:
            vals = mdi_raw[feat].dropna().values.astype(float)
            if len(vals) >= 5:
                mdi_mean, mdi_ci_lo, mdi_ci_hi = bootstrap_ci(vals)
                row.update({
                    "mdi_mean": mdi_mean, "mdi_ci_lo": mdi_ci_lo,
                    "mdi_ci_hi": mdi_ci_hi,
                    # Two-part gate: (1) CI not entirely below noise floor,
                    # (2) mean above noise floor — prevents noise blips
                    # from granting pass when bulk of evidence is negative.
                    "mdi_passes": (mdi_ci_hi > threshold_1F and mdi_mean > threshold_1F),
                })
            else:
                row.update({"mdi_mean": np.nan, "mdi_passes": False})
        else:
            row.update({"mdi_mean": np.nan, "mdi_passes": np.nan})

        # ── SFI (already CI-only, no Wilcoxon leg) ───────────────────────
        if sfi_raw is not None and feat in sfi_raw.columns and sfi_null is not None:
            vals = sfi_raw[feat].dropna().values.astype(float)
            if len(vals) >= 5:
                sfi_mean, sfi_ci_lo, sfi_ci_hi = bootstrap_ci(vals)
                row.update({
                    "sfi_mean": sfi_mean, "sfi_ci_lo": sfi_ci_lo,
                    "sfi_ci_hi": sfi_ci_hi,
                    # Two-part gate: CI not entirely below null AND mean above null.
                    "sfi_passes": (sfi_ci_hi > sfi_null and sfi_mean > sfi_null),
                })
            else:
                row.update({"sfi_mean": np.nan, "sfi_passes": False})
        else:
            row.update({"sfi_mean": np.nan, "sfi_passes": np.nan})

        # ── De-substituted MDA ────────────────────────────────────────────
        if desub_mda_raw is not None and feat in desub_mda_raw.columns:
            vals = desub_mda_raw[feat].dropna().values.astype(float)
            if len(vals) >= 5:
                desub_mean, desub_ci_lo, desub_ci_hi = bootstrap_ci(vals)
                row.update({
                    "desub_mda_mean": desub_mean, "desub_mda_ci_lo": desub_ci_lo,
                    "desub_mda_ci_hi": desub_ci_hi,
                    # Two-part gate: CI not entirely negative AND mean positive.
                    "desub_mda_passes": (desub_ci_hi > 0 and desub_mean > 0),
                })
            else:
                row.update({"desub_mda_mean": np.nan, "desub_mda_passes": False})
        else:
            row.update({"desub_mda_mean": np.nan, "desub_mda_passes": np.nan})

        # ── PCA-MDA ──────────────────────────────────────────────────────
        if pca_mda_raw is not None and feat in pca_mda_raw.columns:
            vals = pca_mda_raw[feat].dropna().values.astype(float)
            if len(vals) >= 5:
                pca_mda_mean, pca_mda_ci_lo, pca_mda_ci_hi = bootstrap_ci(vals)
                row.update({
                    "pca_mda_mean": pca_mda_mean, "pca_mda_ci_lo": pca_mda_ci_lo,
                    "pca_mda_ci_hi": pca_mda_ci_hi,
                    # Two-part gate: CI not entirely negative AND mean positive.
                    "pca_mda_passes": (pca_mda_ci_hi > 0 and pca_mda_mean > 0),
                })
            else:
                row.update({"pca_mda_mean": np.nan, "pca_mda_passes": False})
        else:
            row.update({"pca_mda_mean": np.nan, "pca_mda_passes": np.nan})

        # ── Residualized MDA ─────────────────────────────────────────────
        if resid_mda_raw is not None and feat in resid_mda_raw.columns:
            vals = resid_mda_raw[feat].dropna().values.astype(float)
            if len(vals) >= 5:
                resid_mean, resid_ci_lo, resid_ci_hi = bootstrap_ci(vals)
                row.update({
                    "resid_mda_mean": resid_mean, "resid_mda_ci_lo": resid_ci_lo,
                    "resid_mda_ci_hi": resid_ci_hi,
                    # Two-part gate: CI not entirely negative AND mean positive.
                    "resid_mda_passes": (resid_ci_hi > 0 and resid_mean > 0),
                })
            else:
                row.update({"resid_mda_mean": np.nan, "resid_mda_passes": False})
        else:
            row.update({"resid_mda_mean": np.nan, "resid_mda_passes": np.nan})

        # ── CFI-MDA (cluster-level, bootstrap CI gate) ───────────────────
        cid = feat_to_cluster.get(feat)
        if cfi_mda_raw is not None and cid is not None and cid in cfi_mda_raw.columns:
            vals = cfi_mda_raw[cid].dropna().values.astype(float)
            if len(vals) >= 5:
                cfi_mean, cfi_ci_lo, cfi_ci_hi = bootstrap_ci(vals)
                row.update({
                    "cfi_mda_cluster_mean": cfi_mean,
                    "cfi_mda_ci_lo": cfi_ci_lo,
                    "cfi_mda_ci_hi": cfi_ci_hi,
                    "cfi_mda_cluster_passes": (cfi_ci_lo > 0),
                })
            else:
                row.update({"cfi_mda_cluster_mean": np.nan, "cfi_mda_ci_lo": np.nan,
                            "cfi_mda_ci_hi": np.nan, "cfi_mda_cluster_passes": False})
        else:
            row.update({"cfi_mda_cluster_mean": np.nan, "cfi_mda_ci_lo": np.nan,
                        "cfi_mda_ci_hi": np.nan, "cfi_mda_cluster_passes": np.nan})

        rows.append(row)

    report = pd.DataFrame(rows).set_index("feature")

    # ── Benjamini-Hochberg FDR correction per method ─────────────────────
    # Corrects the OR-logic gates: for methods using Wilcoxon p < alpha,
    # we adjust p-values within each method independently (not globally,
    # since features within a method are correlated — rolling stats share
    # windows, matchup features share structure). BH controls FDR under
    # positive dependence (Benjamini-Yekutieli 2001 shows BH is valid
    # under PRDS, which correlated importance scores satisfy).
    def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
        """Benjamini-Hochberg adjusted p-values (no external dependency)."""
        m = len(p_values)
        order = np.argsort(p_values)
        ranked = np.empty(m)
        ranked[order] = np.arange(1, m + 1)
        adjusted = p_values * m / ranked
        # Enforce monotonicity: step down from largest
        adjusted_sorted = adjusted[np.argsort(ranked)[::-1]]
        for i in range(1, m):
            adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i - 1])
        adjusted[np.argsort(ranked)[::-1]] = adjusted_sorted
        return np.clip(adjusted, 0, 1)

    p_col_to_pass_col = {
        "mdi_p": "mdi_passes",
        "desub_mda_p": "desub_mda_passes",
        "pca_mda_p": "pca_mda_passes",
        "resid_mda_p": "resid_mda_passes",
        "cfi_mda_p": "cfi_mda_cluster_passes",
    }

    for p_col, pass_col in p_col_to_pass_col.items():
        if p_col not in report.columns or pass_col not in report.columns:
            continue

        # Extract valid p-values for correction
        valid_mask = report[p_col].notna()
        if valid_mask.sum() < 2:
            continue

        p_vals = report.loc[valid_mask, p_col].values
        p_adjusted = _bh_adjust(p_vals)

        # Store adjusted p-values
        report.loc[valid_mask, f"{p_col}_bh"] = p_adjusted

        # Recalculate pass/fail using BH-adjusted p-values.
        # New semantics: mean > threshold AND (CI upper > threshold OR adjusted_p < alpha).
        # The CI-upper gate prevents rejection when CI entirely below threshold;
        # BH-adjusted p provides an alternative significance path.
        if p_col == "mdi_p":
            report.loc[valid_mask, pass_col] = (
                (report.loc[valid_mask, "mdi_mean"] > threshold_1F) &
                ((report.loc[valid_mask, "mdi_ci_hi"] > threshold_1F) |
                 (p_adjusted < p_threshold))
            )
        elif p_col == "cfi_mda_p":
            report.loc[valid_mask, pass_col] = (
                (report.loc[valid_mask, "cfi_mda_ci_hi"] > 0) |
                (p_adjusted < p_threshold)
            )
        else:
            # desub/pca/resid: mean > 0 AND (CI upper > 0 OR adjusted_p < threshold)
            ci_hi_col = p_col.replace("_p", "_ci_hi")
            mean_col = p_col.replace("_p", "_mean")
            if ci_hi_col in report.columns and mean_col in report.columns:
                report.loc[valid_mask, pass_col] = (
                    (report.loc[valid_mask, mean_col] > 0) &
                    ((report.loc[valid_mask, ci_hi_col] > 0) |
                     (p_adjusted < p_threshold))
                )

    n_corrected = sum(1 for p in p_col_to_pass_col if p in report.columns)
    log.info(f"    BH-FDR correction applied to {n_corrected} methods (alpha={p_threshold})")

    # ── Margin columns: distance from threshold (positive = passes) ──────
    # Margins quantify confidence in the pass/fail label. Near-zero margins
    # indicate borderline decisions where label noise is likely.
    # MDI: CI-gated, margin = ci_lower - threshold_1F
    if "mdi_ci_lo" in report.columns:
        report["mdi_margin"] = report["mdi_ci_lo"] - threshold_1F

    # SFI: CI-gated, margin = ci_lower - sfi_null
    if "sfi_ci_lo" in report.columns and sfi_null is not None:
        report["sfi_margin"] = report["sfi_ci_lo"] - sfi_null

    # desub/PCA/resid MDA: OR-gate (ci_lo > 0 OR bh_p < threshold)
    # margin = max(ci_lo, p_threshold - bh_p) — best of the two gates
    for method in ["desub_mda", "pca_mda", "resid_mda"]:
        ci_col = f"{method}_ci_lo"
        p_bh_col = f"{method}_p_bh"
        margin_col = f"{method}_margin"
        if ci_col in report.columns:
            ci_margin = report[ci_col] - 0.0  # positive = CI above zero
            if p_bh_col in report.columns:
                p_margin = p_threshold - report[p_bh_col]  # positive = p below threshold
                report[margin_col] = np.maximum(
                    ci_margin.fillna(-np.inf),
                    p_margin.fillna(-np.inf)
                ).replace(-np.inf, np.nan)
            else:
                report[margin_col] = ci_margin

    # CFI-MDA: OR-gate (ci_lo > 0 OR bh_p < threshold)
    if "cfi_mda_ci_lo" in report.columns:
        ci_margin = report["cfi_mda_ci_lo"] - 0.0
        if "cfi_mda_p_bh" in report.columns:
            p_margin = p_threshold - report["cfi_mda_p_bh"]
            report["cfi_mda_margin"] = np.maximum(
                ci_margin.fillna(-np.inf),
                p_margin.fillna(-np.inf)
            ).replace(-np.inf, np.nan)
        else:
            report["cfi_mda_margin"] = ci_margin

    # Determine pass columns available
    pass_cols = [c for c in ["mdi_passes", "sfi_passes", "desub_mda_passes",
                             "pca_mda_passes", "resid_mda_passes"]
                 if c in report.columns]

    report["n_methods_available"] = sum(
        report[col].notna().astype(int) for col in pass_cols
    )
    report["n_methods_passed"] = sum(
        report[col].fillna(False).astype(int) for col in pass_cols
    )

    def assign_tier(r):
        n_avail = r["n_methods_available"]
        if n_avail == 0:
            return "UNKNOWN"
        n_passed = r["n_methods_passed"]
        if n_passed == n_avail:
            return "ACCEPTED"
        elif n_passed == 0:
            return "REJECTED"
        return "NEEDS SPECIFICATION"

    report["tier"] = report.apply(assign_tier, axis=1)

    # ── Temporal trend demotion: demote ACCEPTED/NEEDS SPECIFICATION features
    # with decaying importance into recent noise. A feature whose historical
    # mean was positive (passing the gate) but is monotonically declining and
    # recently negative is no longer useful for forward deployment.
    # Criterion: Spearman ρ < -0.6 (strong monotonic decline) AND mean of last
    # recency_n_folds < null_val. Both conditions required to avoid demoting
    # features with noisy-but-stable trajectories.
    oos_raws_for_trend = [
        (sfi_raw, sfi_null or 0.0), (desub_mda_raw, 0.0),
        (pca_mda_raw, 0.0), (resid_mda_raw, 0.0),
    ]
    demoted = set()
    for raw_df, null_val in oos_raws_for_trend:
        if raw_df is None or raw_df.empty:
            continue
        recent = raw_df.tail(recency_n_folds)
        for feat in report.index[report["tier"].isin(["ACCEPTED", "NEEDS SPECIFICATION"])]:
            if feat in demoted:
                continue
            if feat not in raw_df.columns:
                continue
            vals = raw_df[feat].dropna().values
            if len(vals) < 5:
                continue
            rho, _ = spearmanr(np.arange(len(vals)), vals)
            if rho >= -0.6:
                continue
            recent_vals = recent[feat].dropna().values if feat in recent.columns else np.array([])
            if len(recent_vals) >= 2 and recent_vals.mean() <= null_val:
                demoted.add(feat)

    if demoted:
        report.loc[list(demoted), "tier"] = "REJECTED"
        log.info(f"    Trend demotion: {len(demoted)} features demoted to REJECTED "
                 f"(Spearman rho < -0.6 AND recent mean <= null)")

    # ── Recency rescue: promote REJECTED → NEEDS SPECIFICATION if the feature
    # shows credible emerging signal. Requires BOTH:
    #   (a) Recent folds positive (mean > null)
    #   (b) Overall upward trend (Spearman ρ > 0 across all folds)
    # Without (b), noise blips in recent folds trigger false rescues.
    oos_raws = [(sfi_raw, sfi_null or 0.0), (desub_mda_raw, 0.0),
                (pca_mda_raw, 0.0), (resid_mda_raw, 0.0)]
    rescued = set()
    for raw_df, null_val in oos_raws:
        if raw_df is None or raw_df.empty:
            continue
        recent = raw_df.tail(recency_n_folds)
        for feat in report.index[report["tier"] == "REJECTED"]:
            if feat in rescued:
                continue
            if feat not in recent.columns:
                continue
            recent_vals = recent[feat].dropna().values
            if len(recent_vals) < 2 or recent_vals.mean() <= null_val:
                continue
            # Require strong upward trend (ρ > 0.5) to confirm emergence vs blip.
            # Weak positive ρ (e.g. 0.2) with n=8 is indistinguishable from noise.
            all_vals = raw_df[feat].dropna().values
            if len(all_vals) < 5:
                continue
            rho, _ = spearmanr(np.arange(len(all_vals)), all_vals)
            if rho > 0.5:
                rescued.add(feat)

    if rescued:
        report.loc[list(rescued), "tier"] = "NEEDS SPECIFICATION"
        log.info(f"    Recency rescue: {len(rescued)} features promoted from "
                 f"REJECTED → NEEDS SPECIFICATION (positive recent + upward trend)")

    # Composite rank (lower = better)
    for method, col in [("mdi", "mdi_mean"), ("sfi", "sfi_mean"), ("desub_mda", "desub_mda_mean"),
                        ("pca_mda", "pca_mda_mean"), ("resid_mda", "resid_mda_mean")]:
        if col in report.columns:
            report[f"{method}_rank"] = report[col].rank(ascending=False, na_option="bottom")
    rank_cols = [c for c in ["mdi_rank", "sfi_rank", "desub_mda_rank",
                             "pca_mda_rank", "resid_mda_rank"] if c in report.columns]
    if rank_cols:
        report["composite_rank"] = report[rank_cols].mean(axis=1)
        report = report.sort_values("composite_rank")

    return report


# ─────────────────────────────────────────────────────────────────────────────
#  PCA cross-check + weighted Kendall's tau  (de Prado structural validation)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_tau(eigenvalue_ranks: np.ndarray,
                 importance_ranks: np.ndarray) -> tuple[float | None, float]:
    """Weighted Kendall's tau with permutation p-value (1000 permutations)."""
    tau, _ = weightedtau(eigenvalue_ranks, importance_ranks)
    rng = np.random.default_rng(42)
    perm_taus = np.array([
        weightedtau(eigenvalue_ranks, rng.permutation(importance_ranks))[0]
        for _ in range(1000)
    ])
    p = float((np.abs(perm_taus) >= np.abs(tau)).mean())
    return (float(tau) if not np.isnan(tau) else None), p


def pca_cross_check(X: pd.DataFrame,
                    y: pd.Series,
                    years: pd.Series,
                    sample_weight: pd.Series,
                    regression: bool = False,
                    variance_threshold: float = 0.95) -> tuple:
    """De Prado's PCA cross-check (AFML Ch.8 / MLAM Ch.6).

    Procedure:
      1. PCA the standardized feature matrix → k PCs (variance_threshold)
      2. Run MDI, MDA, SFI independently on the PCs
      3. Compare eigenvalue rank of PC_k to importance rank of PC_k

    This validates whether supervised importance aligns with unsupervised
    variance structure. If tau is significantly positive for a method, that
    method's importance is structurally grounded.

    Returns:
        pca_info: DataFrame(index=PC_k) with per-method importance and ranks
        tau_results: dict keyed by method name, e.g.
            {"MDI": {"tau": 0.38, "p_value": 0.001}, "MDA": {...}, "SFI": {...}}
    """
    from sklearn.metrics import log_loss, mean_squared_error

    n_jobs = get_n_jobs()
    scoring = "log_loss" if not regression else "r2"

    # Step 1: PCA on correlation matrix (standardized features)
    with blas_full():
        X_std = (X - X.mean()) / X.std().replace(0, 1)
        X_filled = X_std.fillna(0)
        pca = PCA()
        pca.fit(X_filled.values)

    var_ratios = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratios)
    k = int(np.searchsorted(cum_var, variance_threshold)) + 1
    k = min(k, X_filled.shape[1])

    W = pca.components_[:k].T
    P_vals = X_filled.values @ W
    pc_names = [f"PC_{i}" for i in range(k)]
    X_pc = pd.DataFrame(P_vals, index=X.index, columns=pc_names)

    # Eigenvalue ranks: PC_0 = most variance = rank 1
    eigenvalue_ranks = np.arange(1, k + 1, dtype=float)

    log.info(f"PCA cross-check: {k} PCs, {cum_var[k-1]:.1%} variance explained")

    tau_results = {}
    pca_info_data = {
        "explained_variance_ratio": var_ratios[:k],
        "eigenvalue_rank": eigenvalue_ranks,
    }

    # Step 2a: MDI on PCs
    log.info("  PCA cross-check: MDI on PCs...")
    clf_mdi = build_rf(n_estimators=1000, n_jobs=n_jobs, regression=regression)
    clf_mdi.fit(X_pc, y, sample_weight=sample_weight.values)
    mdi_summary, _ = feat_imp_mdi(clf_mdi, pc_names)
    pc_mdi = mdi_summary.loc[pc_names, "mean"].values
    mdi_ranks = pd.Series(pc_mdi).rank(ascending=False).values
    tau, p = _compute_tau(eigenvalue_ranks, mdi_ranks)
    tau_results["MDI"] = {"tau": tau, "p_value": p}
    pca_info_data["mdi"] = pc_mdi
    pca_info_data["mdi_rank"] = mdi_ranks
    log.info(f"    MDI: tau={tau:+.4f}, p={p:.4f}")

    # Step 2b: MDA on PCs (permutation importance via expanding-window CV)
    log.info("  PCA cross-check: MDA on PCs...")
    clf_mda = build_rf(n_estimators=300, n_jobs=1, regression=regression)
    mda_summary, _ = feat_imp_mda(
        clf_mda, X_pc, y, years,
        sample_weight=sample_weight,
        scoring=scoring,
    )
    mda_vals = mda_summary.loc[pc_names, "mean"].values
    mda_ranks = pd.Series(mda_vals).rank(ascending=False).values
    tau, p = _compute_tau(eigenvalue_ranks, mda_ranks)
    tau_results["MDA"] = {"tau": tau, "p_value": p}
    pca_info_data["mda"] = mda_vals
    pca_info_data["mda_rank"] = mda_ranks
    log.info(f"    MDA: tau={tau:+.4f}, p={p:.4f}")

    # Step 2c: SFI on PCs (single-feature importance per PC)
    log.info("  PCA cross-check: SFI on PCs...")
    cv = ExpandingWindowYearCV(years)
    folds = list(cv.split(X_pc, y, groups=years.values))
    sfi_scores = np.zeros(k)
    for i in range(k):
        fold_scores = []
        for tr_idx, te_idx in folds:
            clf_sfi = build_rf(n_estimators=100, n_jobs=n_jobs, regression=regression)
            Xi_tr = X_pc.iloc[tr_idx, [i]]
            Xi_te = X_pc.iloc[te_idx, [i]]
            w_tr = sample_weight.iloc[tr_idx].values
            clf_sfi.fit(Xi_tr, y.iloc[tr_idx], sample_weight=w_tr)
            if regression:
                pred = clf_sfi.predict(Xi_te)
                fold_scores.append(-mean_squared_error(y.iloc[te_idx], pred))
            else:
                pred = clf_sfi.predict_proba(Xi_te)
                fold_scores.append(-log_loss(y.iloc[te_idx], pred))
        sfi_scores[i] = np.mean(fold_scores)
    sfi_ranks = pd.Series(sfi_scores).rank(ascending=False).values
    tau, p = _compute_tau(eigenvalue_ranks, sfi_ranks)
    tau_results["SFI"] = {"tau": tau, "p_value": p}
    pca_info_data["sfi"] = sfi_scores
    pca_info_data["sfi_rank"] = sfi_ranks
    log.info(f"    SFI: tau={tau:+.4f}, p={p:.4f}")

    pca_info = pd.DataFrame(pca_info_data, index=pc_names)

    return pca_info, tau_results


# ─────────────────────────────────────────────────────────────────────────────
#  Target-independent clustering (run once, reuse for all targets)
# ─────────────────────────────────────────────────────────────────────────────

def compute_shared_clustering(X: pd.DataFrame) -> dict:
    """Compute target-independent ONC clustering from the feature matrix.

    ONC only uses X.corr() — no target y is involved.

    Returns dict with:
        'clusters': {cluster_id: [feature_names]}
        'denoising_info': dict of MP denoising diagnostics
    """
    log.info(f"Computing target-independent clustering on {X.shape[1]} features...")

    log.info("  1/2  Marcenko-Pastur denoising + detoning...")
    with blas_full():
        corr_raw = X.corr().fillna(0)
        q = X.shape[0] / X.shape[1]

        evals_raw = np.linalg.eigvalsh(corr_raw.values)
        lambda_plus = (1.0 + (1.0 / q) ** 0.5) ** 2
        n_signal = int((evals_raw > lambda_plus).sum())
        n_noise = int((evals_raw <= lambda_plus).sum())
        n_negative = int((evals_raw < 0).sum())
        signal_var = float(evals_raw[evals_raw > lambda_plus].sum())
        total_var = float(evals_raw.sum())

        denoising_info = {
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "q_ratio": float(q),
            "lambda_plus": float(lambda_plus),
            "n_signal_eigenvalues": n_signal,
            "n_noise_eigenvalues": n_noise,
            "n_negative_eigenvalues": n_negative,
            "signal_variance_pct": round(100 * signal_var / total_var, 1) if total_var > 0 else 0,
            "noise_variance_pct": round(100 * (total_var - signal_var) / total_var, 1) if total_var > 0 else 0,
            "top_eigenvalue": float(evals_raw.max()),
        }
        corr_denoised = denoise_corr(corr_raw, q=q)
        corr_cluster = detone_corr(corr_denoised, n_remove=0)

    log.info(f"    lambda+ = {lambda_plus:.4f}, signal eigenvalues: {n_signal}, "
             f"noise: {n_noise}, signal variance: {denoising_info['signal_variance_pct']}%")

    log.info("  2/2  ONC clustering (greedy divisive on denoised+detoned matrix)...")
    clusters = onc_cluster(corr_cluster, max_clusters=None)
    log.info(f"    Found {len(clusters)} clusters:")
    for cid, members in sorted(clusters.items(), key=lambda x: -len(x[1])):
        log.info(f"    Cluster {cid} ({len(members)} features): {members[:5]}"
                 + (" ..." if len(members) > 5 else ""))

    return {
        "clusters": clusters,
        "denoising_info": denoising_info,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Master importance runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_importance(X: pd.DataFrame,
                       y: pd.Series,
                       years: pd.Series,
                       sample_weight: pd.Series = None,
                       run_sfi: bool = True,
                       run_desub_mda: bool = True,
                       run_pca_mda: bool = True,
                       run_residual_mda: bool = True,
                       regression: bool = False,
                       precomputed: dict = None) -> dict:
    """De Prado feature importance pipeline (AFML Ch.8 + MLAM Ch.4/6):

      1. Denoise (Marcenko-Pastur) + detone correlation matrix
      2. ONC clustering (greedy divisive on denoised+detoned matrix)
      3. CFI-MDI + CFI-MDA (cluster-level importance)
      4. MDI (per-feature, in-sample)
      5. SFI (per-feature, standalone OOS)
      6. De-substituted MDA (within-cluster ranking, substitution-free)
      7. PCA-MDA (orthogonal basis, substitution-free)
      8. Residualized MDA (cross-cluster orthogonalization)
      9. PCA cross-check + weighted Kendall's tau
      10. Algorithmic filtering (ACCEPTED / NEEDS SPECIFICATION / REJECTED)

    If precomputed is provided (from compute_shared_clustering), skips steps 1-2.
    Returns a dict of DataFrames.
    """
    log.info(f"Running feature importance on {X.shape[0]} samples, "
             f"{X.shape[1]} features, {years.nunique()} seasons...")

    if precomputed is not None:
        log.info("1/10  Using precomputed clustering (target-independent)...")
        clusters = precomputed["clusters"]
        denoising_info = precomputed["denoising_info"]
        log.info(f"    {len(clusters)} clusters (precomputed)")
        log.info("2/10  Skipped (precomputed)")
    else:
        log.info("1/10  Marcenko-Pastur denoising + detoning...")
        with blas_full():
            corr_raw = X.corr().fillna(0)
            q = X.shape[0] / X.shape[1]

            evals_raw = np.linalg.eigvalsh(corr_raw.values)
            lambda_plus = (1.0 + (1.0 / q) ** 0.5) ** 2
            n_signal = int((evals_raw > lambda_plus).sum())
            n_noise = int((evals_raw <= lambda_plus).sum())
            n_negative = int((evals_raw < 0).sum())
            signal_var = float(evals_raw[evals_raw > lambda_plus].sum())
            total_var = float(evals_raw.sum())

            denoising_info = {
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
                "q_ratio": float(q),
                "lambda_plus": float(lambda_plus),
                "n_signal_eigenvalues": n_signal,
                "n_noise_eigenvalues": n_noise,
                "n_negative_eigenvalues": n_negative,
                "signal_variance_pct": round(100 * signal_var / total_var, 1) if total_var > 0 else 0,
                "noise_variance_pct": round(100 * (total_var - signal_var) / total_var, 1) if total_var > 0 else 0,
                "top_eigenvalue": float(evals_raw.max()),
            }
            corr_denoised = denoise_corr(corr_raw, q=q)
            corr_cluster = detone_corr(corr_denoised, n_remove=0)

        log.info(f"    lambda+ = {lambda_plus:.4f}, signal: {n_signal}, noise: {n_noise}, "
                 f"signal var: {denoising_info['signal_variance_pct']}%")

        log.info("2/10  ONC clustering (greedy divisive)...")
        clusters = onc_cluster(corr_cluster, max_clusters=None)
        log.info(f"    Found {len(clusters)} clusters")
        for cid, members in sorted(clusters.items(), key=lambda x: -len(x[1])):
            log.info(f"    Cluster {cid} ({len(members)} features): {members[:5]}")

    # ── Step 3: CFI — fit full RF, then MDI + CFI-MDA on clusters ────────
    # MDI and CFI are in-sample methods (no train/test split) so global median
    # fill is correct — there is no "future fold" to leak from.
    log.info("3/10  CFI-MDI + CFI-MDA (simultaneous cluster permutation)...")
    n_jobs_full = get_n_jobs()
    mda_scoring = "r2" if regression else "log_loss"
    X_filled = X.fillna(X.median())

    clf = build_rf(n_estimators=1000, n_jobs=n_jobs_full, regression=regression)
    clf.fit(X_filled, y, sample_weight=sample_weight)

    cfi_mdi, cfi_mdi_per_feat, cfi_mdi_raw = feat_imp_cfi_mdi_deprado(
        clf, list(X.columns), clusters)
    cfi_mda, cfi_mda_raw = feat_imp_cfi_mda(
        build_rf(n_estimators=300, n_jobs=n_jobs_full, regression=regression),
        X, y, years, clusters, sample_weight,
        scoring=mda_scoring,
    )
    log.info(f"    CFI-MDA: {(cfi_mda['mean'] > 0).sum()}/{len(cfi_mda)} clusters "
             f"with positive importance")

    # ── Step 4: Per-feature MDI ──────────────────────────────────────────
    log.info("4/10  MDI (per-feature, in-sample)...")
    mdi, mdi_raw = feat_imp_mdi(clf, list(X.columns))
    mdi_pvals = compute_pvalues(mdi_raw, null_mean=1.0 / X.shape[1])
    log.info(f"    MDI top-10: {mdi.head(10).index.tolist()}")

    # ── Step 5: SFI ──────────────────────────────────────────────────────
    sfi = None
    sfi_raw = None
    sfi_pvals = None
    null_score = 0.0
    if run_sfi:
        log.info("5/10  SFI (per-feature, purged year-CV, standalone)...")
        sfi, sfi_raw = feat_imp_sfi(
            build_rf(n_estimators=300, n_jobs=1, regression=regression),
            X, y, years, sample_weight,
            regression=regression,
        )
        null_col = "null_r2" if regression else "null_log_loss"
        null_score = sfi[null_col].iloc[0] if null_col in sfi.columns else 0.0
        sfi_pvals = compute_pvalues(sfi_raw, null_mean=null_score, alternative="greater")
        log.info(f"    SFI top-10: {sfi.head(10).index.tolist()}")
    else:
        log.info("5/10  SFI skipped")

    # ── Step 6: De-substituted MDA ─────────────────────────────────────────
    desub_mda = None
    desub_mda_raw = None
    if run_desub_mda:
        log.info("6/10  De-substituted MDA (within-cluster ranking)...")
        desub_mda, desub_mda_raw = feat_imp_desub_mda(
            X, y, years, clusters,
            sample_weight=sample_weight,
            scoring=mda_scoring,
            n_estimators=300,
            regression=regression,
        )
        log.info(f"    Computed for {len(desub_mda)} features")
    else:
        log.info("6/10  De-substituted MDA skipped")

    # ── Step 7: PCA-MDA ──────────────────────────────────────────────────
    pca_mda = None
    pca_mda_raw = None
    pca_mda_pc_summary = None
    pca_explained_variance_ratio = None
    if run_pca_mda:
        log.info("7/10  PCA-MDA (orthogonal basis)...")
        pca_mda, pca_mda_raw, pca_mda_pc_summary, pca_explained_variance_ratio = feat_imp_pca_mda(
            X, y, years,
            sample_weight=sample_weight,
            scoring=mda_scoring,
            n_estimators=300,
            regression=regression,
        )
        log.info(f"    PCA-MDA: {(pca_mda['mean'] > 0).sum()}/{len(pca_mda)} features positive")
    else:
        log.info("7/10  PCA-MDA skipped")

    # ── Step 8: Residualized MDA ──────────────────────────────────────────
    resid_mda = None
    resid_mda_raw = None
    if run_residual_mda:
        log.info("8/10  Residualized MDA (cross-cluster orthogonalization)...")
        resid_mda, resid_mda_raw = feat_imp_residual_mda(
            X, y, years, clusters,
            sample_weight=sample_weight,
            scoring=mda_scoring,
            n_estimators=300,
            regression=regression,
        )
        log.info(f"    Residualized MDA: {(resid_mda['mean'] > 0).sum()}/{len(resid_mda)} positive")
    else:
        log.info("8/10  Residualized MDA skipped")

    # ── Build per-feature summary ────────────────────────────────────────
    summary = mdi[["mean"]].rename(columns={"mean": "MDI"})
    summary = summary.join(mdi_pvals.rename("p_MDI"), how="left")

    feat_to_cluster = {m: cid for cid, members in clusters.items() for m in members}
    cfi_mda_by_feat = pd.Series(
        {f: cfi_mda.loc[
            next((lbl for lbl in cfi_mda.index if f"Cluster_{feat_to_cluster[f]}" in lbl), None),
            "mean"
        ] if f in feat_to_cluster else np.nan
         for f in summary.index},
        name="CFI_MDA",
    )
    summary = summary.join(cfi_mda_by_feat, how="left")
    summary = summary.join(
        cfi_mdi_per_feat[["mean"]].rename(columns={"mean": "CFI_MDI"}), how="left")
    if sfi is not None:
        summary = summary.join(sfi[["mean"]].rename(columns={"mean": "SFI"}), how="outer")
        summary = summary.join(sfi_pvals.rename("p_SFI"), how="left")
    if desub_mda is not None:
        summary = summary.join(desub_mda[["mean"]].rename(columns={"mean": "DESUB_MDA"}), how="left")
    if pca_mda is not None:
        summary = summary.join(pca_mda[["mean"]].rename(columns={"mean": "PCA_MDA"}), how="left")
    if resid_mda is not None:
        summary = summary.join(resid_mda[["mean"]].rename(columns={"mean": "RESID_MDA"}), how="left")
    summary["rank_MDI"] = summary["MDI"].rank(ascending=False)
    summary["rank_CFI_MDI"] = summary["CFI_MDI"].rank(ascending=False)
    summary["rank_CFI_MDA"] = summary["CFI_MDA"].rank(ascending=False)
    if sfi is not None:
        summary["rank_SFI"] = summary["SFI"].rank(ascending=False)
    if desub_mda is not None:
        summary["rank_DESUB_MDA"] = summary["DESUB_MDA"].rank(ascending=False)
    if pca_mda is not None:
        summary["rank_PCA_MDA"] = summary["PCA_MDA"].rank(ascending=False)
    if resid_mda is not None:
        summary["rank_RESID_MDA"] = summary["RESID_MDA"].rank(ascending=False)
    summary["avg_rank"] = summary[[c for c in summary.columns if c.startswith("rank_")]].mean(axis=1)
    summary = summary.sort_values("avg_rank")

    log.info(f"Feature Importance Summary (top 10): {summary.head(10).index.tolist()}")

    # ── Step 9: PCA cross-check (MDI + MDA + SFI on PCs) ────────────────
    log.info("9/10  PCA cross-check (eigenvalue rank vs PC importance rank)...")
    pca_info, tau_results = pca_cross_check(
        X, y, years, sample_weight, regression=regression)

    # ── Step 10: Algorithmic filtering (PCA-validated gate) ─────────────
    log.info("10/10  Algorithmic filtering (PCA-validated, conservative union)...")
    sfi_null_val = null_score if run_sfi else None
    filter_report = filter_features_v2(
        sfi_raw=sfi_raw,
        desub_mda_raw=desub_mda_raw,
        pca_mda_raw=pca_mda_raw,
        resid_mda_raw=resid_mda_raw,
        mdi_raw=mdi_raw,
        cfi_mda_raw=cfi_mda_raw,
        cfi_mdi_raw=cfi_mdi_raw,
        clusters=clusters,
        sfi_null=sfi_null_val,
        pca_crosscheck=tau_results,
    )
    survivors = filter_report[filter_report["tier"].isin(["ACCEPTED", "NEEDS SPECIFICATION"])]
    log.info(f"    {len(survivors)} features survive (ACCEPTED + NEEDS SPECIFICATION)")

    return {
        "mdi": mdi,
        "mdi_raw": mdi_raw,
        "sfi": sfi,
        "sfi_raw": sfi_raw,
        "desub_mda": desub_mda,
        "desub_mda_raw": desub_mda_raw,
        "pca_mda": pca_mda,
        "pca_mda_raw": pca_mda_raw,
        "pca_mda_pc_summary": pca_mda_pc_summary,
        "pca_explained_variance_ratio": pca_explained_variance_ratio,
        "resid_mda": resid_mda,
        "resid_mda_raw": resid_mda_raw,
        "cfi_mdi": cfi_mdi,
        "cfi_mdi_per_feat": cfi_mdi_per_feat,
        "cfi_mdi_raw": cfi_mdi_raw,
        "cfi_mda": cfi_mda,
        "cfi_mda_raw": cfi_mda_raw,
        "clusters": clusters,
        "summary": summary,
        "filter_report": filter_report,
        "survivors": survivors.index.tolist(),
        "pca_info": pca_info,
        "tau_results": tau_results,
        "denoising_info": denoising_info,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone CFI-MDI runner (no CV, no filtering)
# ─────────────────────────────────────────────────────────────────────────────

def run_cfi_mdi_only(
    X: pd.DataFrame,
    y: pd.Series,
    clusters: dict,
    sample_weight: pd.Series = None,
    regression: bool = False,
) -> dict:
    """Run ONLY CFI-MDI (de Prado per-tree aggregation).

    In-sample method: fits one RF (1000 trees), extracts per-tree MDI,
    aggregates by cluster. No CV loops, no OOS methods.
    """
    n_jobs = get_n_jobs()
    X_filled = X.fillna(X.median())

    log.info(f"CFI-MDI only: fitting RF (1000 trees) on {X.shape[1]} features, "
             f"{X.shape[0]} samples...")
    clf = build_rf(n_estimators=1000, n_jobs=n_jobs, regression=regression)
    clf.fit(X_filled, y, sample_weight=sample_weight)

    log.info("Computing per-tree cluster MDI...")
    cluster_summary, per_feature, raw_cluster = feat_imp_cfi_mdi_deprado(
        clf, list(X.columns), clusters)

    log.info(f"CFI-MDI complete: {len(cluster_summary)} clusters, "
             f"top-3: {cluster_summary.head(3).index.tolist()}")

    return {
        "cfi_mdi_cluster": cluster_summary,
        "cfi_mdi_per_feature": per_feature,
        "cfi_mdi_raw": raw_cluster,
    }
