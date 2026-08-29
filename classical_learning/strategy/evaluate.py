"""Evaluation metrics: Brier, AUC, ECE, log-loss, Huber, MAE, RMSE.

Per-season breakdown and aggregated metrics with overfit detection.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from .config import ECE_N_BINS

log = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> dict:
    """Compute evaluation metrics for one fold.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values (probabilities for clf, continuous for reg).
    task : str
        "classification" or "regression".

    Returns
    -------
    dict of metric_name: value
    """
    # Filter valid pairs
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 10:
        return {"n_valid": int(valid.sum())}

    yt = y_true[valid]
    yp = y_pred[valid]

    if task == "classification":
        yp_clipped = np.clip(yp, 0.01, 0.99)
        metrics = {
            "log_loss": log_loss(yt, yp_clipped),
            "brier_score": brier_score_loss(yt, yp_clipped),
            "ece": expected_calibration_error(yt, yp_clipped),
            "accuracy": accuracy_score(yt, (yp >= 0.5).astype(int)),
            "n_valid": int(valid.sum()),
        }
        # AUC requires both classes present
        if len(np.unique(yt)) > 1:
            metrics["auc_roc"] = roc_auc_score(yt, yp_clipped)
        return metrics
    else:
        metrics = {
            "mae": mean_absolute_error(yt, yp),
            "rmse": np.sqrt(mean_squared_error(yt, yp)),
            "r2": r2_score(yt, yp),
            "huber_loss": huber_loss(yt, yp),
            "n_valid": int(valid.sum()),
        }
        return metrics


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = ECE_N_BINS,
) -> float:
    """Expected Calibration Error (ECE).

    ECE = Σ (n_b / N) * |avg_confidence_b - avg_accuracy_b|

    Bins predictions into n_bins equal-width bins [0, 1/n_bins), [1/n_bins, 2/n_bins), ...
    and computes weighted absolute difference between predicted and observed frequency.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += (n_bin / n_total) * abs(avg_confidence - avg_accuracy)

    return ece


def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.5) -> float:
    """Huber loss: quadratic for |error| <= delta, linear beyond."""
    residual = y_true - y_pred
    abs_r = np.abs(residual)
    loss = np.where(
        abs_r <= delta,
        0.5 * residual ** 2,
        delta * (abs_r - 0.5 * delta),
    )
    return float(loss.mean())


def compute_overfit_gap(train_metrics: dict, val_metrics: dict, task: str) -> float:
    """Compute train-val metric gap for overfit detection."""
    if task == "classification":
        key = "log_loss"
    else:
        key = "rmse"

    train_val = train_metrics.get(key, 0)
    val_val = val_metrics.get(key, 0)
    return val_val - train_val


def print_model_comparison(
    summary: dict,
    task: str,
) -> str:
    """Print ranked table of model families from a training_summary JSON dict.

    Returns the name of the best-performing family.
    """
    primary = "log_loss" if task == "classification" else "rmse"
    rows = []
    for family, v in summary.items():
        if v.get("status") != "success":
            continue
        agg = v.get("aggregate_metrics", {})
        val = agg.get(primary)
        if val is None:
            # Fall back to mean over fold_metrics
            folds = v.get("fold_metrics", [])
            vals = [f[primary] for f in folds if primary in f]
            val = sum(vals) / len(vals) if vals else None
        if val is not None:
            rows.append((val, family, agg))

    if not rows:
        print("No successful results found.")
        return ""

    rows.sort()
    best_family = rows[0][1]

    header = f"{'Family':<30}  {primary.upper():<10}"
    if task == "classification":
        header += f"  {'AUC':<8}  {'ECE':<8}  {'BRIER':<8}"
    else:
        header += f"  {'RMSE':<8}  {'R2':<8}"
    print(header)
    print("-" * len(header))

    for val, family, agg in rows:
        line = f"{family:<30}  {val:<10.4f}"
        if task == "classification":
            line += f"  {agg.get('auc_roc', float('nan')):<8.4f}"
            line += f"  {agg.get('ece', float('nan')):<8.4f}"
            line += f"  {agg.get('brier_score', float('nan')):<8.4f}"
        else:
            line += f"  {agg.get('rmse', float('nan')):<8.4f}"
            line += f"  {agg.get('r2', float('nan')):<8.4f}"
        print(line)

    return best_family


def ensemble_diagnostics(
    bundle: dict,
    y_true: np.ndarray,
    y_oof: np.ndarray,
    ensemble_std: np.ndarray | None = None,
) -> dict:
    """Compute ensemble-level evaluation metrics and calibration diagnostics.

    Parameters
    ----------
    bundle : dict
        Loaded ensemble pickle (must contain 'task').
    y_true : np.ndarray
        Ground truth labels aligned with y_oof.
    y_oof : np.ndarray
        Ensemble OOF predictions (blended, before calibration).
    ensemble_std : np.ndarray, optional
        Per-sample std across ensemble members.

    Returns
    -------
    dict of diagnostic metrics.
    """
    task = bundle.get("task", "regression")
    metrics = compute_metrics(y_true, y_oof, task)

    result = {"ensemble_metrics": metrics}

    if task == "classification":
        cal_curve = calibration_curve_data(y_true, np.clip(y_oof, 0.01, 0.99))
        result["calibration_curve"] = cal_curve

        # Reliability: expected vs observed probability by bin
        expected = cal_curve["bin_midpoints"]
        observed = cal_curve["fraction_positives"]
        if expected:
            max_deviation = max(abs(e - o) for e, o in zip(expected, observed))
            result["max_calibration_deviation"] = max_deviation

    if ensemble_std is not None:
        result["mean_ensemble_std"] = float(np.nanmean(ensemble_std))
        result["std_p25"] = float(np.nanpercentile(ensemble_std, 25))
        result["std_p75"] = float(np.nanpercentile(ensemble_std, 75))

    return result


def calibration_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute calibration curve (reliability diagram) data.

    Returns bin midpoints, fraction of positives, and bin counts
    for plotting.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_midpoints = []
    fraction_positives = []
    bin_counts = []

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        bin_midpoints.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        fraction_positives.append(y_true[mask].mean())
        bin_counts.append(int(n_bin))

    return {
        "bin_midpoints": bin_midpoints,
        "fraction_positives": fraction_positives,
        "bin_counts": bin_counts,
    }
