"""Orthogonality-based ensemble construction with SLSQP weight optimization.

1. Filter candidates by metric quality
2. Compute pairwise correlation of OOF predictions
3. Greedy forward selection maximizing diversity
4. SLSQP weight optimization
5. Refit selected members on full training data and persist to pickle
"""
from __future__ import annotations

import gc
import inspect
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import MAX_CORRELATION, MAX_ENSEMBLE_SIZE, METRIC_TOLERANCE, NEEDS_IMPUTATION, NEEDS_SCALING, TARGETS_CLASSIFICATION

log = logging.getLogger(__name__)


def build_ensemble(
    oof_matrix: dict[str, np.ndarray],
    y_true: np.ndarray,
    metrics: dict[str, dict],
    task: str,
    max_size: int = MAX_ENSEMBLE_SIZE,
    max_correlation: float = MAX_CORRELATION,
    metric_tolerance: float = METRIC_TOLERANCE,
) -> dict:
    """Build diversity-optimized ensemble from candidate OOF predictions.

    Parameters
    ----------
    oof_matrix : dict[str, np.ndarray]
        Mapping of candidate_name → OOF predictions array.
    y_true : np.ndarray
        Ground truth for the OOF predictions.
    metrics : dict[str, dict]
        Mapping of candidate_name → metric dict (from evaluate.py).
    task : str
        "classification" or "regression".
    max_size : int
        Maximum ensemble members.
    max_correlation : float
        Orthogonality threshold (reject if max corr to any existing member > this).
    metric_tolerance : float
        Keep candidates within this factor of the best metric.

    Returns
    -------
    dict with keys: members, weights, correlation_matrix, ensemble_metrics
    """
    if not oof_matrix:
        return {"members": [], "weights": [], "error": "no candidates"}

    # --- Step 1: Filter by metric quality ---
    primary_metric = "log_loss" if task == "classification" else "mae"
    candidate_scores = {}
    for name, m in metrics.items():
        if primary_metric in m:
            candidate_scores[name] = m[primary_metric]

    if not candidate_scores:
        return {"members": [], "weights": [], "error": "no valid metrics"}

    best_score = min(candidate_scores.values())
    threshold = best_score * metric_tolerance
    viable = {name for name, score in candidate_scores.items() if score <= threshold}
    log.info(f"Filtering: {len(candidate_scores)} candidates → {len(viable)} within {metric_tolerance}x best")

    # --- Step 2: Correlation matrix of OOF predictions ---
    viable_names = sorted(viable)
    n_viable = len(viable_names)

    # Stack predictions into matrix (n_samples × n_candidates)
    valid_mask = ~np.isnan(y_true)
    for name in viable_names:
        valid_mask &= ~np.isnan(oof_matrix[name])

    pred_matrix = np.column_stack([oof_matrix[name][valid_mask] for name in viable_names])
    corr_matrix = np.corrcoef(pred_matrix.T)

    # --- Step 3: Greedy forward selection ---
    # Start with the best single model
    best_candidate = min(viable_names, key=lambda n: candidate_scores[n])
    selected = [best_candidate]
    remaining = [n for n in viable_names if n != best_candidate]

    while len(selected) < max_size and remaining:
        best_next = None
        best_max_corr = 1.0

        for candidate in remaining:
            cand_idx = viable_names.index(candidate)
            # Max correlation with any already-selected member
            max_corr_to_selected = max(
                abs(corr_matrix[cand_idx, viable_names.index(s)])
                for s in selected
            )

            if max_corr_to_selected < best_max_corr:
                best_max_corr = max_corr_to_selected
                best_next = candidate

        if best_next is None or best_max_corr > max_correlation:
            break

        selected.append(best_next)
        remaining.remove(best_next)
        log.info(f"  Added {best_next} (max_corr={best_max_corr:.3f}, ensemble size={len(selected)})")

    log.info(f"Selected {len(selected)} ensemble members")

    # --- Step 4: SLSQP weight optimization ---
    selected_preds = np.column_stack([oof_matrix[name][valid_mask] for name in selected])
    y_valid = y_true[valid_mask]
    weights = _optimize_weights(selected_preds, y_valid, task)

    # --- Compute ensemble metrics ---
    ensemble_pred = selected_preds @ weights
    from .evaluate import compute_metrics
    ensemble_metrics = compute_metrics(y_valid, ensemble_pred, task)

    return {
        "members": selected,
        "weights": weights.tolist(),
        "correlation_matrix": {
            name: {other: float(corr_matrix[viable_names.index(name), viable_names.index(other)])
                   for other in selected}
            for name in selected
        },
        "ensemble_metrics": ensemble_metrics,
        "candidate_count": n_viable,
    }


def _optimize_weights(
    pred_matrix: np.ndarray,
    y_true: np.ndarray,
    task: str,
) -> np.ndarray:
    """Optimize ensemble weights via SLSQP: minimize loss s.t. Σw=1, w≥0."""
    n_models = pred_matrix.shape[1]

    if task == "classification":
        def objective(w):
            blended = pred_matrix @ w
            blended = np.clip(blended, 0.01, 0.99)
            return -np.mean(y_true * np.log(blended) + (1 - y_true) * np.log(1 - blended))
    else:
        def objective(w):
            blended = pred_matrix @ w
            return np.mean(np.abs(y_true - blended))

    # Constraints: sum to 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    # Bounds: each weight in [0, 1]
    bounds = [(0.0, 1.0)] * n_models
    # Initial: equal weights
    w0 = np.ones(n_models) / n_models

    try:
        result = minimize(objective, w0, method="SLSQP", bounds=bounds,
                          constraints=constraints, options={"maxiter": 1000, "ftol": 1e-10})
        if result.success:
            return result.x
    except Exception as e:
        log.warning(f"Weight optimization failed: {e}")

    # Fallback: equal weights
    log.warning("Falling back to equal weights")
    return np.ones(n_models) / n_models


def predict_ensemble(
    X: np.ndarray,
    models: list,
    weights: np.ndarray,
    task: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ensemble prediction with uncertainty estimate.

    Returns
    -------
    tuple of (blended prediction, ensemble std across members)
    """
    preds = []
    for model in models:
        if task == "classification":
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X)[:, 1]
            else:
                dec = model.decision_function(X)
                p = 1.0 / (1.0 + np.exp(-dec))
        else:
            p = model.predict(X)
        preds.append(p)

    pred_matrix = np.column_stack(preds)
    blended = pred_matrix @ weights
    ensemble_std = pred_matrix.std(axis=1)

    return blended, ensemble_std


def load_ensemble_oof(
    models_dir: Path,
    target: str,
    tier: str,
) -> dict[str, np.ndarray]:
    """Load all saved OOF arrays for a target into a name → array dict.

    Returns a dict keyed by family name (e.g. "lightgbm") mapping to the
    OOF prediction array saved by train_target().
    """
    models_dir = Path(models_dir)
    oof_files = sorted(models_dir.glob(f"oof_{target}_*_{tier}.npy"))

    result = {}
    for f in oof_files:
        # Filename pattern: oof_{target}_{family}_{tier}.npy
        # Strip leading oof_{target}_ and trailing _{tier}.npy
        stem = f.stem  # e.g. oof_home_win_lightgbm_A
        prefix = f"oof_{target}_"
        suffix = f"_{tier}"
        family = stem[len(prefix):]
        if family.endswith(suffix):
            family = family[: -len(suffix)]
        result[family] = np.load(f)
        log.debug(f"Loaded OOF for {family}: shape={result[family].shape}")

    log.info(f"Loaded {len(result)} OOF arrays for target={target} tier={tier}")
    return result


def fit_and_save_ensemble(
    features_path: Path,
    models_dir: Path,
    target: str,
    tier: str,
    members: list[str],
    weights: list[float],
    data_mode: str = "2015+",
    output_path: Path | None = None,
) -> Path:
    """Refit selected ensemble members on the full training set and save to pickle.

    After build_ensemble() selects members and optimizes weights from OOF arrays,
    this function refits each selected model on the full dataset (all LOYO folds
    combined) and bundles everything into a deployable pickle.

    Parameters
    ----------
    features_path : Path
        Path to game_features.parquet.
    models_dir : Path
        Directory containing params_{target}_{family}_{tier}.json files.
    target : str
        Target column name.
    tier : str
        Data tier ("A", "B", or "C").
    members : list[str]
        Family names selected by build_ensemble().
    weights : list[float]
        Blend weights aligned with members.
    data_mode : str
        "2015+" or "all".
    output_path : Path, optional
        Destination pickle path. Defaults to models_dir/ensemble_{target}_{tier}.pkl.

    Returns
    -------
    Path
        Path to the saved ensemble pickle.
    """
    from sklearn.preprocessing import StandardScaler

    from .data import _semantic_impute, compute_temporal_weights, load_features
    from .models import build_model

    models_dir = Path(models_dir)
    output_path = output_path or models_dir / f"ensemble_{target}_{tier}.pkl"
    task = "classification" if target in TARGETS_CLASSIFICATION else "regression"

    # Drop negligible-weight members — SLSQP may return tiny floats (e.g. 1e-8) rather
    # than exact zeros; threshold at 1% so rounding artifacts don't survive into the pickle
    nonzero = [(m, w) for m, w in zip(members, weights) if w >= 0.01]
    if len(nonzero) < len(members):
        dropped = [m for m, w in zip(members, weights) if w == 0]
        log.info(f"Dropping {len(dropped)} zero-weight members: {dropped}")
        members, weights = zip(*nonzero)
        members, weights = list(members), list(weights)

    log.info(f"Refitting {len(members)} ensemble members for {target} on full training data")

    X, y, seasons = load_features(features_path, target, data_mode)
    sample_weights = compute_temporal_weights(seasons)

    # Load importance filter if present — mirrors train_target() behavior
    filter_report = None
    from .config import IMPORTANCE_DIR
    report_path = IMPORTANCE_DIR / target / "filtered" / "feature_report.csv"
    if report_path.exists():
        filter_report = pd.read_csv(report_path, index_col="feature")
        log.info(f"  Importance filter loaded: {len(filter_report)} features")

    member_bundles = []

    for family in members:
        params_file = models_dir / f"params_{target}_{family}_{tier}.json"
        if not params_file.exists():
            raise FileNotFoundError(
                f"Params file not found: {params_file}. Run train first."
            )
        with open(params_file) as f:
            best_params = json.load(f)

        # Resolve feature set (mirrors prepare_fold logic in train.py)
        importance_features = None
        if filter_report is not None:
            from ..analysis.feature_routing import get_feature_set
            importance_features = get_feature_set(family, filter_report)

        X_fit = X[importance_features] if importance_features is not None else X
        feature_columns = list(X_fit.columns)

        # Imputation then scaling — both fit only on training data (the full set here)
        if family in NEEDS_IMPUTATION:
            X_fit = _semantic_impute(X_fit)

        scaler = None
        if family in NEEDS_SCALING:
            scaler = StandardScaler()
            X_arr = scaler.fit_transform(X_fit.values)
        else:
            X_arr = X_fit.values

        model = build_model(family, task, best_params)
        fit_kwargs = {}
        if "sample_weight" in inspect.signature(model.fit).parameters:
            fit_kwargs["sample_weight"] = sample_weights.values

        model.fit(X_arr, y.values, **fit_kwargs)

        member_bundles.append({
            "family": family,
            "model": model,
            "scaler": scaler,
            "feature_columns": feature_columns,
            "needs_imputation": family in NEEDS_IMPUTATION,
        })

        log.info(f"  Fitted {family} on {X_arr.shape[0]} rows")
        del X_arr
        gc.collect()

    bundle = {
        "target": target,
        "task": task,
        "tier": tier,
        "members": members,
        "weights": np.array(weights),
        "member_bundles": member_bundles,
    }

    with open(output_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    log.info(f"Ensemble saved to {output_path}")
    return output_path
