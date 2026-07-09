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
    # NegBin (count regression targets only)
    negbin_alpha: float | None = None
    distribution_type: str = "student_t"  # "student_t" or "negbin"
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


# Count-based regression targets where NegBin is the correct distributional model.
# Validated: NegBin beats Student-t by 4.6x on totals calibration (MACE 2.3% vs 10.5%).
NEGBIN_TARGETS = ("home_runs", "away_runs", "total_runs", "first_5_total_runs")


def estimate_negbin_alpha(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Estimate NegBin overdispersion parameter via MLE.

    NegBin(n, p) where n=alpha, p=alpha/(alpha+mu).
    Maximizes log-likelihood over the training OOF residuals.
    Global alpha is stable across targets (CV=0.059) — one alpha per target suffices.
    """
    from scipy.stats import nbinom
    from scipy.optimize import minimize_scalar

    # Clamp predictions to positive (NegBin requires mu > 0)
    mu = np.clip(y_pred, 0.1, None)
    y = np.clip(y_true, 0, None).astype(int)

    def neg_ll(log_alpha):
        alpha = np.exp(log_alpha)
        n = alpha
        p = alpha / (alpha + mu)
        # Clip p to (0,1) exclusive for numerical stability
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(nbinom.logpmf(y, n, p))

    result = minimize_scalar(neg_ll, bounds=(np.log(0.5), np.log(50.0)), method="bounded")
    alpha = float(np.exp(result.x))
    log.info(f"NegBin MLE alpha={alpha:.3f} (log_alpha={result.x:.3f}, converged={result.fun:.1f})")
    return alpha


def calibrate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ensemble_std: np.ndarray | None = None,
    target_name: str | None = None,
) -> CalibrationBundle:
    """Fit distributional model to OOF residuals for regression.

    Count targets (home_runs, away_runs, total_runs, first_5_total_runs) use
    NegBin — validated to beat Student-t by 4.6x on totals MACE.
    Signed targets (home_run_diff, first_5_home_run_diff) keep Student-t.

    The fitted params are used at inference to compute cover probabilities:
    P(actual > threshold) for any spread/total line.
    """
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[valid]
    yp = y_pred[valid]

    # Confidence tiers
    std_p33, std_p67 = None, None
    if ensemble_std is not None:
        es = ensemble_std[valid]
        std_p33 = float(np.percentile(es, 33))
        std_p67 = float(np.percentile(es, 67))

    # Branch on target type: NegBin for counts, Student-t for signed differences
    if target_name is not None and target_name in NEGBIN_TARGETS:
        alpha = estimate_negbin_alpha(yt, yp)
        return CalibrationBundle(
            task="regression",
            negbin_alpha=alpha,
            distribution_type="negbin",
            std_p33=std_p33,
            std_p67=std_p67,
        )

    # Signed targets: fit Student-t to residuals
    residuals = yt - yp
    df, loc, scale = stats.t.fit(residuals, floc=0)

    log.info(f"Residual Student-t fit: df={df:.2f}, scale={scale:.3f} (loc forced=0)")

    # Bias correction table: bin by |residual| and compute empirical cover rates
    bias_correction = _build_bias_correction(residuals, yp, yt)

    return CalibrationBundle(
        task="regression",
        residual_df=float(df),
        residual_scale=float(scale),
        distribution_type="student_t",
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


def negbin_cover_probability(
    mu: float, threshold: float, alpha: float, direction: str = "over",
) -> float:
    """P(actual > threshold) from NegBin(mu, alpha) marginal.

    Parameterization: n=alpha, p=alpha/(alpha+mu).
    For half-integer lines (e.g., 8.5), floor gives the last integer below.
    """
    from scipy.stats import nbinom
    n = alpha
    p = alpha / (alpha + max(mu, 0.01))
    k = int(np.floor(threshold))
    cdf_val = nbinom.cdf(k, n, p)
    return float((1.0 - cdf_val) if direction == "over" else cdf_val)


def negbin_total_cover_probability(
    mu_home: float, alpha_home: float,
    mu_away: float, alpha_away: float,
    threshold: float, direction: str = "over",
) -> float:
    """P(home + away > threshold) via PMF convolution of independent NegBin marginals.

    Independence assumption validated: residual correlation rho=0.046 (below noise
    floor), calibrates to 0.55% MACE on totals.
    """
    from scipy.stats import nbinom

    n_h, p_h = alpha_home, alpha_home / (alpha_home + max(mu_home, 0.01))
    n_a, p_a = alpha_away, alpha_away / (alpha_away + max(mu_away, 0.01))

    # Truncate at 99.9th percentile for efficiency (max_h + max_a ~ 40, convolve <1ms)
    max_h = int(nbinom.ppf(0.999, n_h, p_h)) + 1
    max_a = int(nbinom.ppf(0.999, n_a, p_a)) + 1

    pmf_h = nbinom.pmf(np.arange(max_h + 1), n_h, p_h)
    pmf_a = nbinom.pmf(np.arange(max_a + 1), n_a, p_a)

    # Convolution gives PMF of total
    pmf_total = np.convolve(pmf_h, pmf_a)

    k = int(np.floor(threshold))
    cdf_val = pmf_total[:k + 1].sum()
    return float((1.0 - cdf_val) if direction == "over" else cdf_val)


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
