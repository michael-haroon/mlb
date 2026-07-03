"""Calibration: isotonic regression (clf), Student-t residuals (reg), confidence tiers.

Post-hoc calibration applied to ensemble OOF predictions to produce
well-calibrated probabilities and distributional estimates.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression

log = logging.getLogger(__name__)


@dataclass
class CalibrationBundle:
    """Serializable calibration artifacts for deployment."""
    task: str
    # Classification
    isotonic: IsotonicRegression | None = None
    # Regression
    residual_df: float | None = None  # Student-t degrees of freedom
    residual_scale: float | None = None  # Student-t scale
    # Confidence tiers (tercile boundaries of ensemble std)
    std_p33: float | None = None
    std_p67: float | None = None
    # Bias correction table (delta bins → empirical cover rate)
    bias_correction: dict | None = None


def calibrate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ensemble_std: np.ndarray | None = None,
) -> CalibrationBundle:
    """Fit isotonic calibration on OOF classification predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth.
    y_pred : np.ndarray
        Raw ensemble probabilities.
    ensemble_std : np.ndarray, optional
        Standard deviation across ensemble members (for confidence tiers).
    """
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[valid]
    yp = y_pred[valid]

    # Isotonic regression: monotonic probability mapping
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(yp, yt)

    # Confidence tiers from ensemble disagreement
    std_p33, std_p67 = None, None
    if ensemble_std is not None:
        es = ensemble_std[valid]
        std_p33 = float(np.percentile(es, 33))
        std_p67 = float(np.percentile(es, 67))

    return CalibrationBundle(
        task="classification",
        isotonic=iso,
        std_p33=std_p33,
        std_p67=std_p67,
    )


def calibrate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ensemble_std: np.ndarray | None = None,
) -> CalibrationBundle:
    """Fit Student-t distribution to OOF residuals for regression.

    The fitted (df, scale) parameters are used at inference to compute
    cover probabilities: P(actual > threshold) for any spread/total line.
    """
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[valid]
    yp = y_pred[valid]
    residuals = yt - yp

    # Fit Student-t distribution (loc forced to 0 since residuals should be unbiased)
    df, loc, scale = stats.t.fit(residuals, floc=0)

    log.info(f"Residual Student-t fit: df={df:.2f}, scale={scale:.3f} (loc forced=0)")

    # Confidence tiers
    std_p33, std_p67 = None, None
    if ensemble_std is not None:
        es = ensemble_std[valid]
        std_p33 = float(np.percentile(es, 33))
        std_p67 = float(np.percentile(es, 67))

    # Bias correction table: bin by |residual| and compute empirical cover rates
    bias_correction = _build_bias_correction(residuals, yp, yt)

    return CalibrationBundle(
        task="regression",
        residual_df=float(df),
        residual_scale=float(scale),
        std_p33=std_p33,
        std_p67=std_p67,
        bias_correction=bias_correction,
    )


def apply_calibration(
    y_pred: np.ndarray,
    bundle: CalibrationBundle,
    ensemble_std: np.ndarray | None = None,
) -> dict:
    """Apply calibration to raw predictions.

    Returns
    -------
    dict with calibrated predictions and metadata.
    """
    if bundle.task == "classification":
        calibrated = bundle.isotonic.predict(np.clip(y_pred, 0.01, 0.99))
        result = {
            "calibrated_prob": calibrated,
            "raw_prob": y_pred,
        }
    else:
        result = {
            "point_estimate": y_pred,
            "residual_df": bundle.residual_df,
            "residual_scale": bundle.residual_scale,
        }

    # Confidence tier assignment
    if ensemble_std is not None and bundle.std_p33 is not None:
        tiers = np.where(
            ensemble_std <= bundle.std_p33, "HIGH",
            np.where(ensemble_std <= bundle.std_p67, "MEDIUM", "LOW")
        )
        result["confidence_tier"] = tiers
        result["ensemble_std"] = ensemble_std

    return result


def cover_probability(
    point_estimate: float,
    threshold: float,
    df: float,
    scale: float,
    direction: str = "over",
) -> float:
    """Compute P(actual > threshold) or P(actual < threshold) from fitted Student-t.

    Parameters
    ----------
    point_estimate : float
        Ensemble prediction (mu).
    threshold : float
        Market line threshold.
    df : float
        Student-t degrees of freedom.
    scale : float
        Student-t scale parameter.
    direction : str
        "over" for P(actual > threshold), "under" for P(actual < threshold).
    """
    # Standardize: how many scale units is the threshold from our estimate
    z = (threshold - point_estimate) / scale
    cdf_value = stats.t.cdf(z, df)

    if direction == "over":
        return 1.0 - cdf_value
    else:
        return cdf_value


def _build_bias_correction(
    residuals: np.ndarray,
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Build empirical bias correction table by delta bins.

    For each bin of |predicted - some reference|, compute the actual
    cover rate vs the model-predicted cover rate. The difference is the bias.
    """
    abs_residuals = np.abs(residuals)
    edges = np.percentile(abs_residuals, np.linspace(0, 100, n_bins + 1))

    correction = {}
    for i in range(n_bins):
        mask = (abs_residuals >= edges[i]) & (abs_residuals < edges[i + 1])
        if i == n_bins - 1:
            mask = (abs_residuals >= edges[i]) & (abs_residuals <= edges[i + 1])

        if mask.sum() < 10:
            continue

        bin_residuals = residuals[mask]
        # Empirical fraction that exceeded the prediction
        empirical_over_rate = (bin_residuals > 0).mean()
        bin_key = f"bin_{i}"
        correction[bin_key] = {
            "edge_low": float(edges[i]),
            "edge_high": float(edges[i + 1]),
            "n_samples": int(mask.sum()),
            "empirical_over_rate": float(empirical_over_rate),
            "bias": float(empirical_over_rate - 0.5),
        }

    return correction


def fit_platt(oof_raw: np.ndarray, y_true: np.ndarray):
    """Fit Platt scaling: logistic regression on raw scores.

    Returns (clf, clip_lo, clip_hi) where clf is a fitted LogisticRegression.
    C=1e5 makes it nearly unconstrained — we want the sigmoid shape, not regularization.
    """
    from sklearn.linear_model import LogisticRegression
    clip_lo = np.nanpercentile(oof_raw, 0.1)
    clip_hi = np.nanpercentile(oof_raw, 99.9)
    X = np.clip(oof_raw, clip_lo, clip_hi).reshape(-1, 1)
    clf = LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000)
    clf.fit(X, y_true)
    return clf, clip_lo, clip_hi

def apply_platt(oof_raw: np.ndarray, clf, clip_lo: float, clip_hi: float) -> np.ndarray:
    X = np.clip(oof_raw, clip_lo, clip_hi).reshape(-1, 1)
    return clf.predict_proba(X)[:, 1]

def fit_temperature(oof_raw: np.ndarray, y_true: np.ndarray) -> float:
    """Fit temperature scaling: single scalar T minimizing NLL.

    p_cal = sigmoid(logit(p) / T). T>1 softens, T<1 sharpens.
    """
    from scipy.optimize import minimize_scalar
    p = np.clip(oof_raw, 1e-7, 1 - 1e-7)
    logits = np.log(p / (1 - p))

    def nll(T):
        T = max(T, 1e-3)
        p_cal = 1.0 / (1.0 + np.exp(-logits / T))
        p_cal = np.clip(p_cal, 1e-7, 1 - 1e-7)
        return -np.mean(y_true * np.log(p_cal) + (1 - y_true) * np.log(1 - p_cal))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)

def apply_temperature(oof_raw: np.ndarray, T: float) -> np.ndarray:
    p = np.clip(oof_raw, 1e-7, 1 - 1e-7)
    logits = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-logits / T))

def fit_isotonic_per_model(oof_raw: np.ndarray, y_true: np.ndarray):
    """Fit isotonic regression on a single model's OOF predictions."""
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(oof_raw, y_true)
    return ir
