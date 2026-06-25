"""Orthogonality-based ensemble construction with SLSQP weight optimization.

1. Filter candidates by metric quality
2. Compute pairwise correlation of OOF predictions
3. Greedy forward selection maximizing diversity
4. SLSQP weight optimization
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .config import MAX_CORRELATION, MAX_ENSEMBLE_SIZE, METRIC_TOLERANCE

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
