"""Parametric and nonparametric distribution fitting with QQ plots.

Fits candidate distributions to residuals (regression) or predicted
probabilities (classification), selects best via AIC/BIC, and generates
QQ diagnostic plots.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

log = logging.getLogger(__name__)


CANDIDATE_DISTRIBUTIONS_CONTINUOUS = {
    "normal": stats.norm,
    "student_t": stats.t,
    "skewnorm": stats.skewnorm,
    "laplace": stats.laplace,
    "logistic": stats.logistic,
}

CANDIDATE_DISTRIBUTIONS_COUNT = {
    "poisson": stats.poisson,
    "nbinom": stats.nbinom,
}


def fit_best_distribution(
    data: np.ndarray,
    candidates: Optional[dict] = None,
    is_count: bool = False,
) -> dict:
    """Fit candidate distributions and select best by AIC.

    Parameters
    ----------
    data : np.ndarray
        Observed values (residuals for regression, counts for count targets).
    candidates : dict, optional
        Mapping of name → scipy.stats distribution. Auto-selected if None.
    is_count : bool
        If True, use count distributions (Poisson, NegBin).

    Returns
    -------
    dict with keys: best_name, best_params, all_fits (sorted by AIC), gof_stats
    """
    if candidates is None:
        candidates = CANDIDATE_DISTRIBUTIONS_COUNT if is_count else CANDIDATE_DISTRIBUTIONS_CONTINUOUS

    data = data[~np.isnan(data)]
    n = len(data)
    if n < 30:
        return {"best_name": None, "error": "insufficient data"}

    fits = []
    for name, dist in candidates.items():
        try:
            if is_count:
                fit_result = _fit_count_distribution(data, name, dist)
            else:
                params = dist.fit(data)
                nll = -dist.logpdf(data, *params).sum()
                k = len(params)
                aic = 2 * k + 2 * nll
                bic = k * np.log(n) + 2 * nll

                # KS test
                ks_stat, ks_pvalue = stats.kstest(data, dist.cdf, args=params)

                fit_result = {
                    "name": name,
                    "params": params,
                    "nll": float(nll),
                    "aic": float(aic),
                    "bic": float(bic),
                    "ks_stat": float(ks_stat),
                    "ks_pvalue": float(ks_pvalue),
                    "n_params": k,
                }

            fits.append(fit_result)
        except Exception as e:
            log.debug(f"Failed to fit {name}: {e}")
            continue

    if not fits:
        return {"best_name": None, "error": "all fits failed"}

    # Sort by AIC
    fits.sort(key=lambda f: f.get("aic", float("inf")))
    best = fits[0]

    return {
        "best_name": best["name"],
        "best_params": best.get("params"),
        "all_fits": fits,
        "n_samples": n,
    }


def _fit_count_distribution(data: np.ndarray, name: str, dist) -> dict:
    """Fit count distributions (Poisson, Negative Binomial)."""
    data_int = np.round(data).astype(int)
    data_int = data_int[data_int >= 0]
    n = len(data_int)

    if name == "poisson":
        mu = data_int.mean()
        params = (mu,)
        nll = -stats.poisson.logpmf(data_int, mu).sum()
        k = 1
    elif name == "nbinom":
        # Method of moments for NegBin
        mean = data_int.mean()
        var = data_int.var()
        if var <= mean:
            # Overdispersion required for NegBin
            raise ValueError("No overdispersion; Poisson preferred")
        p = mean / var
        r = mean * p / (1 - p)
        params = (r, p)
        nll = -stats.nbinom.logpmf(data_int, r, p).sum()
        k = 2
    else:
        raise ValueError(f"Unknown count distribution: {name}")

    aic = 2 * k + 2 * nll
    bic = k * np.log(n) + 2 * nll

    return {
        "name": name,
        "params": params,
        "nll": float(nll),
        "aic": float(aic),
        "bic": float(bic),
        "ks_stat": None,
        "ks_pvalue": None,
        "n_params": k,
    }


def qq_plot_data(
    data: np.ndarray,
    dist_name: str,
    params: tuple,
) -> dict:
    """Compute QQ plot coordinates for given distribution fit.

    Returns theoretical quantiles vs observed quantiles for plotting.
    """
    data = np.sort(data[~np.isnan(data)])
    n = len(data)

    # Theoretical quantiles
    pp = (np.arange(1, n + 1) - 0.5) / n

    dist = CANDIDATE_DISTRIBUTIONS_CONTINUOUS.get(dist_name)
    if dist is None:
        return {"error": f"Unknown distribution: {dist_name}"}

    theoretical = dist.ppf(pp, *params)

    return {
        "observed": data.tolist(),
        "theoretical": theoretical.tolist(),
        "dist_name": dist_name,
        "params": [float(p) for p in params],
        "n_points": n,
    }


def generate_qq_plots(
    residuals: np.ndarray,
    output_dir: Path,
    target_name: str,
    season: Optional[int] = None,
) -> list[Path]:
    """Generate QQ plot PNGs for the best-fit distribution.

    Returns list of paths to generated plot files.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping QQ plots")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fit best distribution
    fit_result = fit_best_distribution(residuals, is_count=False)
    if fit_result["best_name"] is None:
        return []

    best_name = fit_result["best_name"]
    best_params = fit_result["best_params"]

    # Compute QQ data
    qq = qq_plot_data(residuals, best_name, best_params)
    if "error" in qq:
        return []

    # Plot
    suffix = f"_{season}" if season else ""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(qq["theoretical"], qq["observed"], s=4, alpha=0.5)

    # Reference line
    lims = [
        min(min(qq["theoretical"]), min(qq["observed"])),
        max(max(qq["theoretical"]), max(qq["observed"])),
    ]
    ax.plot(lims, lims, "r--", linewidth=1)

    ax.set_xlabel(f"Theoretical ({best_name})")
    ax.set_ylabel("Observed")
    ax.set_title(f"QQ Plot: {target_name}{suffix} vs {best_name}")
    ax.grid(True, alpha=0.3)

    plot_path = output_dir / f"qq_{target_name}{suffix}.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return [plot_path]
