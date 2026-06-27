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
  - Purged LOYO cross-validation (no temporal leakage)
  - Cluster detection via ONC (Optimal Number of Clusters)
  - Marcenko-Pastur denoising + detoning
  - Bootstrap CI + Wilcoxon significance testing
  - Three-tier algorithmic filter (ACCEPTED / NEEDS SPECIFICATION / REJECTED)
  - PCA cross-check with weighted Kendall's tau
  - Synthetic data validation

All importance methods use LOYO CV for temporal safety.

References:
  AFML   Ch.7 (purged CV), Ch.8 (feature importance)
  MLAM   Ch.4 (ONC), Ch.6 (CFI)
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.metrics import log_loss, roc_auc_score, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy.stats import wilcoxon as scipy_wilcoxon, weightedtau
from sklearn.decomposition import PCA
from tqdm import tqdm

from .compute import get_n_jobs, get_parallel_split, blas_limit, blas_full

log = logging.getLogger(__name__)


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

    scoring: 'log_loss' or 'roc_auc' for classifiers; 'r2' for regressors.
    """
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import BaggingClassifier, BaggingRegressor
    from sklearn.metrics import r2_score

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
                 scoring: str = "log_loss",
                 cv_splits: int = None) -> tuple:
    """Mean Decrease Accuracy via purged year-CV.

    Shuffles one feature at a time and measures accuracy drop.
    Parallelised over folds via joblib with multiprocessing backend.

    Returns:
        summary: DataFrame(index=feature, columns=[mean, std])
        raw:     DataFrame(index=fold, columns=features) — per-fold importance
    """
    cv = PurgedYearKFold(years, n_splits=cv_splits)
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

    X_tr = X_vals[np.ix_(X_tr_idx, col_idxs)]
    X_te = X_vals[np.ix_(X_te_idx, col_idxs)]
    y_tr = y_vals[X_tr_idx]
    y_te = y_vals[X_te_idx]
    w_tr = w_vals[X_tr_idx] if w_vals is not None else None

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
                       cv_splits: int = None,
                       n_estimators: int = 300,
                       regression: bool = False) -> tuple:
    """De-substituted MDA (Approach B): for each feature f in cluster C_i,
    train a model on {f} + {all features NOT in C_i}, then shuffle f.

    Eliminates substitution: no cluster-mate of f is present to compensate.

    Returns:
        summary: DataFrame(index=feature, columns=[mean, std])
        raw:     DataFrame(index=fold, columns=features)
    """
    cv = PurgedYearKFold(years, n_splits=cv_splits)
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
                     cv_splits: int = None,
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
        cv_splits=cv_splits,
    )

    # Map back: importance_i = sum_j |W[i,j]| * mean_importance_PC_j
    pc_imp = pc_summary["mean"].values                          # (k,)
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
    return summary_df, raw_df, pc_summary


# ─────────────────────────────────────────────────────────────────────────────
#  Residualized MDA  (de Prado MLAM Ch.6 — cross-cluster orthogonalization)
# ─────────────────────────────────────────────────────────────────────────────

def feat_imp_residual_mda(X: pd.DataFrame,
                          y: pd.Series,
                          years: pd.Series,
                          clusters: dict,
                          sample_weight: pd.Series = None,
                          scoring: str = "log_loss",
                          cv_splits: int = None,
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
    X_vals = X.fillna(0).values.astype(np.float64)

    tasks = []
    for cid, cluster_members in clusters.items():
        other_cols = [c for c in col_names if c not in cluster_sets[cid]]
        if not other_cols:
            continue
        X_other = X[other_cols].fillna(0).values.astype(np.float64)
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
        cv_splits=cv_splits,
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

    X_tr = X_col_vals[X_tr_idx]
    X_te = X_col_vals[X_te_idx]
    y_tr = y_vals[X_tr_idx]
    y_te = y_vals[X_te_idx]
    w_tr = w_vals[X_tr_idx] if w_vals is not None else None

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
                 regression: bool = False,
                 cv_splits: int = None) -> tuple:
    """Single Feature Importance: train the model on ONE feature at a time.

    Immune to substitution effects between correlated features.
    Uses BaggingClassifier/Regressor (not a single tree) for lower variance.

    Returns:
        summary: DataFrame(index=feature, columns=[mean, std, null_score])
        raw:     DataFrame(index=fold, columns=features)
    """
    cv = PurgedYearKFold(years, n_splits=cv_splits)
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


def feat_imp_cfi_mda(clf,
                     X: pd.DataFrame,
                     y: pd.Series,
                     years: pd.Series,
                     clusters: dict,
                     sample_weight: pd.Series = None,
                     scoring: str = "log_loss",
                     cv_splits: int = None) -> tuple:
    """Clustered MDA: shuffle all features in a cluster simultaneously.

    Returns:
        summary: DataFrame(index=cluster_label, columns=[mean, std])
        raw:     DataFrame(index=fold, columns=cluster_id)
    """
    cv = PurgedYearKFold(years, n_splits=cv_splits)
    base_scores = []
    cluster_perms = {cid: [] for cid in clusters}
    is_regression = (scoring == "r2")
    all_labels = None if is_regression else np.unique(y.values)

    for train_idx, test_idx in tqdm(list(cv.split(X, y, groups=years.values)),
                                    desc="CFI-MDA folds", unit="fold", leave=False):
        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_tr = y.iloc[train_idx]
        y_te = y.iloc[test_idx]
        w_tr = sample_weight.iloc[train_idx] if sample_weight is not None else None

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
            np.random.shuffle(shuffle_vals)
            X_te_perm[present] = shuffle_vals
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
        if len(diffs) < 4:
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
                    p_threshold: float = 0.10) -> pd.DataFrame:
    """Three-tier feature classification:

      ACCEPTED          — passes ALL available tests. Proven signal.
      NEEDS SPECIFICATION — passes SOME tests. Signal exists but partial/conditional.
      REJECTED          — fails ALL tests. No signal detected.

    Tests:
      MDI pass:      mean > 1/F AND (bootstrap CI lower > 1/F OR Wilcoxon p < threshold)
      SFI pass:      mean > sfi_null AND bootstrap CI lower > sfi_null
      Desub MDA pass: bootstrap CI lower > 0 OR Wilcoxon p < threshold
      PCA-MDA pass:  bootstrap CI lower > 0 OR Wilcoxon p < threshold
      Resid MDA pass: bootstrap CI lower > 0 OR Wilcoxon p < threshold
      CFI-MDA:       cluster-level, reported but not used for individual pass/fail
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
            if len(vals) >= 4:
                mdi_mean, mdi_ci_lo, mdi_ci_hi = bootstrap_ci(vals)
                diffs = vals - threshold_1F
                diffs = diffs[diffs != 0]
                p_mdi = scipy_wilcoxon(diffs, alternative='greater')[1] if len(diffs) >= 4 else np.nan
                row.update({
                    "mdi_mean": mdi_mean, "mdi_ci_lo": mdi_ci_lo,
                    "mdi_ci_hi": mdi_ci_hi, "mdi_p": p_mdi,
                    "mdi_passes": (
                        mdi_mean > threshold_1F and
                        (mdi_ci_lo > threshold_1F or (not np.isnan(p_mdi) and p_mdi < p_threshold))
                    ),
                })
            else:
                row.update({"mdi_mean": np.nan, "mdi_passes": False})
        else:
            row.update({"mdi_mean": np.nan, "mdi_passes": np.nan})

        # ── SFI ──────────────────────────────────────────────────────────
        if sfi_raw is not None and feat in sfi_raw.columns and sfi_null is not None:
            vals = sfi_raw[feat].dropna().values.astype(float)
            if len(vals) >= 4:
                sfi_mean, sfi_ci_lo, sfi_ci_hi = bootstrap_ci(vals)
                row.update({
                    "sfi_mean": sfi_mean, "sfi_ci_lo": sfi_ci_lo,
                    "sfi_ci_hi": sfi_ci_hi,
                    "sfi_passes": (sfi_mean > sfi_null and sfi_ci_lo > sfi_null),
                })
            else:
                row.update({"sfi_mean": np.nan, "sfi_passes": False})
        else:
            row.update({"sfi_mean": np.nan, "sfi_passes": np.nan})

        # ── De-substituted MDA ────────────────────────────────────────────
        if desub_mda_raw is not None and feat in desub_mda_raw.columns:
            vals = desub_mda_raw[feat].dropna().values.astype(float)
            if len(vals) >= 4:
                desub_mean, desub_ci_lo, desub_ci_hi = bootstrap_ci(vals)
                diffs = vals - 0.0
                diffs = diffs[diffs != 0]
                p_desub = scipy_wilcoxon(diffs, alternative='greater')[1] if len(diffs) >= 4 else np.nan
                row.update({
                    "desub_mda_mean": desub_mean, "desub_mda_ci_lo": desub_ci_lo,
                    "desub_mda_ci_hi": desub_ci_hi, "desub_mda_p": p_desub,
                    "desub_mda_passes": (
                        desub_ci_lo > 0 or (not np.isnan(p_desub) and p_desub < p_threshold)
                    ),
                })
            else:
                row.update({"desub_mda_mean": np.nan, "desub_mda_passes": False})
        else:
            row.update({"desub_mda_mean": np.nan, "desub_mda_passes": np.nan})

        # ── PCA-MDA ──────────────────────────────────────────────────────
        if pca_mda_raw is not None and feat in pca_mda_raw.columns:
            vals = pca_mda_raw[feat].dropna().values.astype(float)
            if len(vals) >= 4:
                pca_mda_mean, pca_mda_ci_lo, pca_mda_ci_hi = bootstrap_ci(vals)
                diffs = vals - 0.0
                diffs = diffs[diffs != 0]
                p_pca = scipy_wilcoxon(diffs, alternative='greater')[1] if len(diffs) >= 4 else np.nan
                row.update({
                    "pca_mda_mean": pca_mda_mean, "pca_mda_ci_lo": pca_mda_ci_lo,
                    "pca_mda_ci_hi": pca_mda_ci_hi, "pca_mda_p": p_pca,
                    "pca_mda_passes": (
                        pca_mda_ci_lo > 0 or (not np.isnan(p_pca) and p_pca < p_threshold)
                    ),
                })
            else:
                row.update({"pca_mda_mean": np.nan, "pca_mda_passes": False})
        else:
            row.update({"pca_mda_mean": np.nan, "pca_mda_passes": np.nan})

        # ── Residualized MDA ─────────────────────────────────────────────
        if resid_mda_raw is not None and feat in resid_mda_raw.columns:
            vals = resid_mda_raw[feat].dropna().values.astype(float)
            if len(vals) >= 4:
                resid_mean, resid_ci_lo, resid_ci_hi = bootstrap_ci(vals)
                diffs = vals - 0.0
                diffs = diffs[diffs != 0]
                p_resid = scipy_wilcoxon(diffs, alternative='greater')[1] if len(diffs) >= 4 else np.nan
                row.update({
                    "resid_mda_mean": resid_mean, "resid_mda_ci_lo": resid_ci_lo,
                    "resid_mda_ci_hi": resid_ci_hi, "resid_mda_p": p_resid,
                    "resid_mda_passes": (
                        resid_ci_lo > 0 or (not np.isnan(p_resid) and p_resid < p_threshold)
                    ),
                })
            else:
                row.update({"resid_mda_mean": np.nan, "resid_mda_passes": False})
        else:
            row.update({"resid_mda_mean": np.nan, "resid_mda_passes": np.nan})

        # ── CFI-MDA (cluster-level, for reporting) ───────────────────────
        cid = feat_to_cluster.get(feat)
        if cfi_mda_raw is not None and cid is not None and cid in cfi_mda_raw.columns:
            vals = cfi_mda_raw[cid].dropna().values.astype(float)
            if len(vals) >= 2:
                cfi_mean = vals.mean()
                row.update({
                    "cfi_mda_cluster_mean": cfi_mean,
                    "cfi_mda_cluster_passes": cfi_mean > 0,
                })
            else:
                row.update({"cfi_mda_cluster_mean": np.nan, "cfi_mda_cluster_passes": False})
        else:
            row.update({"cfi_mda_cluster_mean": np.nan, "cfi_mda_cluster_passes": np.nan})

        rows.append(row)

    report = pd.DataFrame(rows).set_index("feature")

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

def pca_cross_check(X: pd.DataFrame,
                    importance_summary: pd.DataFrame) -> tuple:
    """De Prado's PCA cross-check:
    1. Run PCA on the feature matrix
    2. Rank features by variance-weighted absolute loading
    3. Compute weighted Kendall's tau between PCA ranks and importance ranks

    A tau near 1 means supervised importance aligns with unsupervised structure.

    Returns:
        (pca_info, tau_results)
    """
    X_clean = X.dropna(axis=1, how="all")
    X_filled = X_clean.fillna(X_clean.median())
    X_filled = X_filled.dropna(axis=1)

    pca = PCA()
    pca.fit(X_filled)

    loadings = np.abs(pca.components_)
    var_ratios = pca.explained_variance_ratio_
    weighted_loadings = (loadings.T * var_ratios).sum(axis=1)

    pca_info = pd.DataFrame({
        "pca_weighted_loading": weighted_loadings,
        "pca_rank": pd.Series(weighted_loadings).rank(ascending=False).values,
    }, index=X_filled.columns)

    tau_results = {}
    for method in ["MDI", "MDA", "SFI"]:
        rank_col = f"rank_{method}"
        if rank_col not in importance_summary.columns:
            continue
        ranks = importance_summary[rank_col].dropna()
        common = pca_info.index.intersection(ranks.index)
        if len(common) < 3:
            tau_results[method] = {"tau": np.nan, "p_value": np.nan}
            continue
        pca_ranks = pca_info.loc[common, "pca_rank"].values
        imp_ranks = ranks[common].values
        tau, _ = weightedtau(pca_ranks, imp_ranks)
        rng_perm = np.random.default_rng(42)
        n_perm = 1000
        perm_taus = np.array([
            weightedtau(pca_ranks, rng_perm.permutation(imp_ranks))[0]
            for _ in range(n_perm)
        ])
        p = float((np.abs(perm_taus) >= np.abs(tau)).mean())
        tau_results[method] = {
            "tau": float(tau) if not np.isnan(tau) else None,
            "p_value": p,
        }

    n_components_90 = np.searchsorted(np.cumsum(var_ratios), 0.90) + 1
    log.info(f"PCA: {n_components_90} components explain 90%% variance "
             f"(of {len(var_ratios)} total)")
    for method, res in tau_results.items():
        tau_str = f"{res['tau']:.3f}" if res['tau'] is not None else "N/A"
        log.info(f"Kendall's tau (PCA vs {method}): {tau_str} (p={res['p_value']:.3f})")

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
        corr_cluster = detone_corr(corr_denoised, n_remove=1)

    log.info(f"    lambda+ = {lambda_plus:.4f}, signal eigenvalues: {n_signal}, "
             f"noise: {n_noise}, signal variance: {denoising_info['signal_variance_pct']}%")

    log.info("  2/2  ONC clustering (greedy divisive on denoised+detoned matrix)...")
    clusters = onc_cluster(corr_cluster, max_clusters=5)
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
                       cv_splits: int = None,
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
            corr_cluster = detone_corr(corr_denoised, n_remove=1)

        log.info(f"    lambda+ = {lambda_plus:.4f}, signal: {n_signal}, noise: {n_noise}, "
                 f"signal var: {denoising_info['signal_variance_pct']}%")

        log.info("2/10  ONC clustering (greedy divisive)...")
        clusters = onc_cluster(corr_cluster, max_clusters=5)
        log.info(f"    Found {len(clusters)} clusters")
        for cid, members in sorted(clusters.items(), key=lambda x: -len(x[1])):
            log.info(f"    Cluster {cid} ({len(members)} features): {members[:5]}")

    # ── Step 3: CFI — fit full RF, then MDI + CFI-MDA on clusters ────────
    log.info("3/10  CFI-MDI + CFI-MDA (simultaneous cluster permutation)...")
    n_jobs_full = get_n_jobs()
    mda_scoring = "r2" if regression else "log_loss"
    clf = build_rf(n_estimators=1000, n_jobs=n_jobs_full, regression=regression)
    clf.fit(X, y, sample_weight=sample_weight)

    cfi_mdi = feat_imp_cfi_mdi(clf, list(X.columns), clusters)
    cfi_mda, cfi_mda_raw = feat_imp_cfi_mda(
        build_rf(n_estimators=300, n_jobs=n_jobs_full, regression=regression),
        X, y, years, clusters, sample_weight,
        scoring=mda_scoring, cv_splits=cv_splits,
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
            regression=regression, cv_splits=cv_splits,
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
            cv_splits=cv_splits,
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
    if run_pca_mda:
        log.info("7/10  PCA-MDA (orthogonal basis)...")
        pca_mda, pca_mda_raw, pca_mda_pc_summary = feat_imp_pca_mda(
            X, y, years,
            sample_weight=sample_weight,
            scoring=mda_scoring,
            cv_splits=cv_splits,
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
            cv_splits=cv_splits,
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

    # ── Step 9: PCA cross-check ──────────────────────────────────────────
    log.info("9/10  PCA cross-check + weighted Kendall's tau...")
    with blas_full():
        pca_info, tau_results = pca_cross_check(X, summary)

    # ── Step 10: Algorithmic filtering ───────────────────────────────────
    log.info("10/10  Algorithmic filtering (ACCEPTED / NEEDS SPECIFICATION / REJECTED)...")
    sfi_null_val = null_score if run_sfi else None
    filter_report = filter_features(
        mdi_raw, cfi_mda_raw, clusters,
        sfi_raw=sfi_raw,
        sfi_null=sfi_null_val,
        desub_mda_raw=desub_mda_raw,
        pca_mda_raw=pca_mda_raw,
        resid_mda_raw=resid_mda_raw,
    )
    tier_counts = filter_report["tier"].value_counts()
    for tier, count in tier_counts.items():
        log.info(f"    {tier}: {count}")
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
        "resid_mda": resid_mda,
        "resid_mda_raw": resid_mda_raw,
        "cfi_mdi": cfi_mdi,
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
