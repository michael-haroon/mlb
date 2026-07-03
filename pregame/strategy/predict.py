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

    # member_bundles is the authoritative list of fitted members; older pickles used
    # "models" (a plain list with no per-model metadata) which caused the scaler,
    # feature_columns, and isotonic_calibrator to be inaccessible at inference.
    member_bundles = bundle["member_bundles"]
    weights = np.array(bundle["weights"])
    calibration: CalibrationBundle = bundle["calibration"]
    task = bundle["task"]

    # --- Data-availability-conditioned routing ---
    # Check which features are actually populated
    populated_pct = features.notna().mean(axis=0)
    n_features = len(features.columns)
    n_populated = (populated_pct > 0.5).sum()

    log.info(f"Predicting {target}: {n_populated}/{n_features} features populated (>{50}%)")

    # Generate per-model predictions
    preds_per_model = []
    valid_weights = []

    for mb, weight in zip(member_bundles, weights):
        family = mb["family"]
        model = mb["model"]
        scaler = mb.get("scaler")
        feature_columns = mb.get("feature_columns", features.columns.tolist())
        needs_imputation = mb.get("needs_imputation", False)
        # Per-model isotonic calibrator fitted on OOF predictions during training.
        # None for regression targets or when the OOF had too few valid rows.
        isotonic_cal = mb.get("isotonic_calibrator")

        available = [f for f in feature_columns if f in features.columns]
        missing_pct = 1.0 - len(available) / max(len(feature_columns), 1)

        # Skip models with >30% missing features at inference
        if missing_pct > 0.3:
            log.debug(f"  Skipping {family}: {missing_pct:.0%} features missing")
            continue

        try:
            X_model = features[available].copy()

            if needs_imputation:
                from .data import _semantic_impute
                X_model = _semantic_impute(X_model)
            if scaler is not None:
                X_model = pd.DataFrame(
                    scaler.transform(X_model),
                    columns=available,
                    index=features.index,
                )

            # Raw prediction
            if task == "classification":
                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X_model)[:, 1]
                else:
                    dec = model.decision_function(X_model)
                    pred = 1.0 / (1.0 + np.exp(-dec))
                # Apply per-model isotonic calibration before blending.
                # This mirrors the calibration applied to OOFs during SLSQP weight
                # optimization so that each model's contribution to the blend is on
                # the same calibrated probability scale as at training time.
                if isotonic_cal is not None:
                    pred = isotonic_cal.predict(pred)
            else:
                # Regression: no per-model isotonic; leave raw prediction unchanged.
                pred = model.predict(X_model)

            preds_per_model.append(pred)
            valid_weights.append(weight)

        except Exception as e:
            log.warning(f"  {family} prediction failed: {e}")
            continue

    if not preds_per_model:
        return {"error": "all models failed at inference", "target": target}

    # --- Blend predictions ---
    pred_matrix = np.column_stack(preds_per_model)
    w = np.array(valid_weights)
    w = w / w.sum()  # renormalize after dropping failed models

    blended = pred_matrix @ w
    ensemble_std = pred_matrix.std(axis=1)

    # --- Apply post-blend calibration (CalibrationBundle layer) ---
    # This is the second calibration stage: a single isotonic fit on the blended
    # OOF signal. It runs AFTER the per-model isotonic (which already corrected
    # individual model miscalibration) and corrects any residual miscalibration
    # introduced by the blending step itself.
    calibrated = apply_calibration(blended, calibration, ensemble_std)

    # --- Build output ---
    output = {
        "target": target,
        "task": task,
        "n_models_used": len(preds_per_model),
        "n_models_total": len(member_bundles),
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
