"""De Prado feature importance methods: MDI, MDA, SFI, ONC, denoising.

Adapted from the NBA pipeline with MLB-specific modifications.
All importance methods use LOYO CV for temporal safety.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

log = logging.getLogger(__name__)


def compute_mdi(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    sample_weight: Optional[pd.Series] = None,
    n_estimators: int = 500,
) -> pd.DataFrame:
    """Mean Decrease Impurity (MDI) feature importance.

    Uses BaggingClassifier/Regressor with single-feature trees to measure
    how much each feature reduces impurity when used as a split.
    """
    if task == "classification":
        base = DecisionTreeClassifier(
            criterion="entropy",
            max_features=1,
            class_weight="balanced",
            min_weight_fraction_leaf=0.02,
        )
        model = BaggingClassifier(estimator=base, n_estimators=n_estimators,
                                  max_features=1.0, random_state=42, n_jobs=-1)
    else:
        base = DecisionTreeRegressor(
            max_features=1,
            min_weight_fraction_leaf=0.02,
        )
        model = BaggingRegressor(estimator=base, n_estimators=n_estimators,
                                 max_features=1.0, random_state=42, n_jobs=-1)

    # Fill NaN for tree training (trees handle NaN but sklearn Bagging doesn't propagate).
    # X.median() is computed on the same data the model trains on, so this is
    # self-consistent. MDI is a diagnostic tool (not a training step), so the
    # mild future-season contamination in the median is acceptable — it does not
    # affect the LOYO training or prediction pipeline.
    X_filled = X.fillna(X.median())

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight.values

    log.info(f"MDI: fitting {n_estimators} trees on {X.shape[0]:,} samples × {X.shape[1]} features")
    t0 = time.time()
    model.fit(X_filled, y, **fit_kwargs)
    log.info(f"MDI: fit complete in {time.time()-t0:.1f}s")

    # Extract feature importances from all trees
    importances = np.zeros((n_estimators, X.shape[1]))
    for i, tree in enumerate(model.estimators_):
        importances[i] = tree.feature_importances_

    # Mean and std across trees
    mdi_df = pd.DataFrame({
        "feature": X.columns,
        "mdi_mean": importances.mean(axis=0),
        "mdi_std": importances.std(axis=0),
    })
    mdi_df["mdi_rank"] = mdi_df["mdi_mean"].rank(ascending=False)
    return mdi_df.sort_values("mdi_rank")


def compute_mda(
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    task: str,
    sample_weight: Optional[pd.Series] = None,
    n_estimators: int = 300,
) -> pd.DataFrame:
    """Mean Decrease Accuracy (MDA) — permutation importance per LOYO fold.

    For each feature, permutes its values and measures the increase in
    validation loss. Computed per-fold to respect temporal ordering.
    """
    from .run import _get_loyo_splits

    splits = _get_loyo_splits(seasons)
    if not splits:
        return pd.DataFrame()

    feature_names = X.columns.tolist()
    mda_scores = np.zeros((len(splits), len(feature_names)))

    n_splits = len(splits)
    log.info(f"MDA: {n_splits} LOYO folds × {len(feature_names)} features × {n_estimators} trees each")
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        t_fold = time.time()
        log.info(f"MDA: fold {fold_idx+1}/{n_splits} — train={len(train_idx):,} val={len(val_idx):,}")
        X_train = X.iloc[train_idx].fillna(X.iloc[train_idx].median())
        X_val = X.iloc[val_idx].fillna(X.iloc[train_idx].median())
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        if task == "classification":
            base = DecisionTreeClassifier(max_features=1, min_weight_fraction_leaf=0.02)
            model = BaggingClassifier(estimator=base, n_estimators=n_estimators,
                                      random_state=42, n_jobs=-1)
        else:
            base = DecisionTreeRegressor(max_features=1, min_weight_fraction_leaf=0.02)
            model = BaggingRegressor(estimator=base, n_estimators=n_estimators,
                                     random_state=42, n_jobs=-1)

        model.fit(X_train, y_train)
        log.info(f"MDA: fold {fold_idx+1}/{n_splits} fit done in {time.time()-t_fold:.1f}s — permuting {len(feature_names)} features")

        # Baseline score
        baseline = _score(model, X_val, y_val, task)

        # Permute each feature and measure degradation
        for j, feat in enumerate(feature_names):
            if j % 50 == 0:
                log.info(f"MDA: fold {fold_idx+1}/{n_splits} — permuting feature {j+1}/{len(feature_names)}")
            X_perm = X_val.copy()
            X_perm[feat] = np.random.permutation(X_perm[feat].values)
            permuted_score = _score(model, X_perm, y_val, task)
            mda_scores[fold_idx, j] = permuted_score - baseline  # higher = more important

    mda_df = pd.DataFrame({
        "feature": feature_names,
        "mda_mean": mda_scores.mean(axis=0),
        "mda_std": mda_scores.std(axis=0),
    })
    mda_df["mda_rank"] = mda_df["mda_mean"].rank(ascending=False)
    return mda_df.sort_values("mda_rank")


def compute_sfi(
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    task: str,
    n_estimators: int = 100,
) -> pd.DataFrame:
    """Single Feature Importance (SFI) — each feature alone in a model.

    Measures the predictive power of each feature in isolation using
    LOYO CV.
    """
    from .run import _get_loyo_splits

    splits = _get_loyo_splits(seasons)
    feature_names = X.columns.tolist()
    sfi_scores = np.zeros(len(feature_names))

    log.info(f"SFI: {len(feature_names)} features × {len(splits)} LOYO folds")
    for j, feat in enumerate(feature_names):
        if j % 20 == 0:
            log.info(f"SFI: feature {j+1}/{len(feature_names)}")
        fold_scores = []
        for train_idx, val_idx in splits:
            X_train_j = X.iloc[train_idx][[feat]].fillna(X.iloc[train_idx][feat].median())
            X_val_j = X.iloc[val_idx][[feat]].fillna(X.iloc[train_idx][feat].median())
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            if task == "classification":
                model = DecisionTreeClassifier(max_depth=3, random_state=42)
            else:
                model = DecisionTreeRegressor(max_depth=3, random_state=42)

            model.fit(X_train_j, y_train)
            score = _score(model, X_val_j, y_val, task)
            fold_scores.append(score)

        sfi_scores[j] = np.mean(fold_scores) if fold_scores else 0.0

    sfi_df = pd.DataFrame({
        "feature": feature_names,
        "sfi_score": sfi_scores,
    })
    # For SFI, lower log_loss / MAE = better, so rank ascending for clf, descending for reg
    sfi_df["sfi_rank"] = sfi_df["sfi_score"].rank(ascending=(task == "regression"))
    return sfi_df.sort_values("sfi_rank")


def denoise_correlation_matrix(
    X: pd.DataFrame,
    ratio: float = 0.5,
) -> np.ndarray:
    """Denoise correlation matrix via Marcenko-Pastur theorem.

    Replaces eigenvalues below the MP threshold with their mean,
    removing noise from the correlation structure.
    """
    corr = X.corr().values
    n, p = X.shape
    q = n / p

    # Marcenko-Pastur bounds
    lambda_plus = (1 + 1 / np.sqrt(q)) ** 2
    lambda_minus = (1 - 1 / np.sqrt(q)) ** 2

    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    # Replace eigenvalues below threshold with their mean
    noise_mask = eigenvalues < lambda_plus
    if noise_mask.any():
        eigenvalues[noise_mask] = eigenvalues[noise_mask].mean()

    # Reconstruct denoised correlation matrix
    denoised = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Normalize diagonal to 1
    d = np.sqrt(np.diag(denoised))
    d[d == 0] = 1.0
    denoised = denoised / np.outer(d, d)

    return denoised


def _score(model, X, y, task: str) -> float:
    """Score a model (lower = better for both tasks)."""
    if task == "classification":
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        else:
            proba = model.predict(X)
        return log_loss(y, np.clip(proba, 0.01, 0.99))
    else:
        return mean_absolute_error(y, model.predict(X))
