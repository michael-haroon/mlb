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
        key = "mae"

    train_val = train_metrics.get(key, 0)
    val_val = val_metrics.get(key, 0)
    return val_val - train_val


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
