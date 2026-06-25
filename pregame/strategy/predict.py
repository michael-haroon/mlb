"""Inference pipeline: features → calibrated probabilities.

Implements the data-availability-conditioned ensemble: checks which features
are actually populated at inference time and routes to the appropriate
model tiers (A/B/C).
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .calibration import CalibrationBundle, apply_calibration, cover_probability

log = logging.getLogger(__name__)


def predict_game(
    features: pd.DataFrame,
    ensemble_path: Path,
    target: str,
) -> dict:
    """Generate calibrated prediction for one or more games.

    Parameters
    ----------
    features : pd.DataFrame
        Feature row(s) for the game(s) to predict.
    ensemble_path : Path
        Path to the ensemble .pkl file.
    target : str
        Target name (determines task type and calibration).

    Returns
    -------
    dict with calibrated predictions, confidence, and distributional params.
    """
    # Load ensemble bundle
    with open(ensemble_path, "rb") as f:
        bundle = pickle.load(f)

    models = bundle["models"]
    weights = np.array(bundle["weights"])
    calibration: CalibrationBundle = bundle["calibration"]
    task = bundle["task"]
    tier_info = bundle.get("tier_info", {})

    # --- Data-availability-conditioned routing ---
    # Check which features are actually populated
    populated_pct = features.notna().mean(axis=0)
    n_features = len(features.columns)
    n_populated = (populated_pct > 0.5).sum()

    log.info(f"Predicting {target}: {n_populated}/{n_features} features populated (>{50}%)")

    # Generate per-model predictions
    preds_per_model = []
    valid_weights = []

    for i, (model_info, model, weight) in enumerate(
        zip(bundle.get("model_info", [{}] * len(models)), models, weights)
    ):
        tier = model_info.get("tier", "A")
        required_features = model_info.get("feature_columns", features.columns.tolist())

        # Check if this model's required features are available
        available = [f for f in required_features if f in features.columns]
        missing_pct = 1.0 - len(available) / max(len(required_features), 1)

        # Skip models with >30% missing features at inference
        if missing_pct > 0.3:
            log.debug(f"  Skipping model {i} (tier={tier}): {missing_pct:.0%} features missing")
            continue

        try:
            X_model = features[available].copy()

            # Apply model-specific preprocessing
            imputer = model_info.get("imputer")
            scaler = model_info.get("scaler")

            if imputer is not None:
                X_model = pd.DataFrame(
                    imputer.transform(X_model),
                    columns=available,
                    index=features.index,
                )
            if scaler is not None:
                X_model = pd.DataFrame(
                    scaler.transform(X_model),
                    columns=available,
                    index=features.index,
                )

            # Predict
            if task == "classification":
                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X_model)[:, 1]
                else:
                    dec = model.decision_function(X_model)
                    pred = 1.0 / (1.0 + np.exp(-dec))
            else:
                pred = model.predict(X_model)

            preds_per_model.append(pred)
            valid_weights.append(weight)

        except Exception as e:
            log.warning(f"  Model {i} (tier={tier}) prediction failed: {e}")
            continue

    if not preds_per_model:
        return {"error": "all models failed at inference", "target": target}

    # --- Blend predictions ---
    pred_matrix = np.column_stack(preds_per_model)
    w = np.array(valid_weights)
    w = w / w.sum()  # renormalize after dropping failed models

    blended = pred_matrix @ w
    ensemble_std = pred_matrix.std(axis=1)

    # --- Apply calibration ---
    calibrated = apply_calibration(blended, calibration, ensemble_std)

    # --- Build output ---
    output = {
        "target": target,
        "task": task,
        "n_models_used": len(preds_per_model),
        "n_models_total": len(models),
    }
    output.update(calibrated)

    # For regression targets, compute cover probabilities for common lines
    if task == "regression" and calibration.residual_df is not None:
        output["distribution"] = {
            "type": "student_t",
            "mu": float(blended[0]) if len(blended) == 1 else blended.tolist(),
            "df": calibration.residual_df,
            "scale": calibration.residual_scale,
        }

    return output


def predict_spread_lines(
    point_estimate: float,
    df: float,
    scale: float,
    lines: Optional[list[float]] = None,
) -> dict[float, float]:
    """Compute cover probabilities for spread/total market lines.

    Parameters
    ----------
    point_estimate : float
        Predicted spread or total (mu from ensemble).
    df : float
        Student-t degrees of freedom (from calibration).
    scale : float
        Student-t scale (from calibration).
    lines : list[float], optional
        Market lines to price. Defaults to standard MLB lines.

    Returns
    -------
    dict of line → P(actual > line)
    """
    if lines is None:
        # Standard MLB spread lines
        lines = [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]

    return {
        line: cover_probability(point_estimate, line, df, scale, direction="over")
        for line in lines
    }


def predict_total_lines(
    point_estimate: float,
    df: float,
    scale: float,
    lines: Optional[list[float]] = None,
) -> dict[float, float]:
    """Compute over probabilities for total runs market lines."""
    if lines is None:
        lines = [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]

    return {
        line: cover_probability(point_estimate, line, df, scale, direction="over")
        for line in lines
    }
