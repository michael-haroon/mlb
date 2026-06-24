from __future__ import annotations

import math


def gaussian_nll(target, mu, sigma):
    import torch

    sigma = sigma.clamp_min(1e-6)
    return 0.5 * (((target - mu) / sigma) ** 2 + 2.0 * torch.log(sigma) + math.log(2.0 * math.pi))


def weighted_mean(loss, sample_weight):
    denom = sample_weight.sum().clamp_min(1e-6)
    return (loss * sample_weight).sum() / denom


def suggest_distribution(values, target_name: str | None = None) -> dict:
    """Suggest an output family from observed target values.

    This is a lightweight audit helper, not a substitute for validation-set
    log-loss comparisons.
    """

    import numpy as np

    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"family": "unknown", "reason": "no finite values"}

    unique = np.unique(arr)
    if set(unique).issubset({0.0, 1.0}):
        return {"family": "bernoulli", "reason": "binary target"}

    is_integer = np.all(np.isclose(arr, np.round(arr)))
    non_negative = np.all(arr >= 0)
    mean = float(np.mean(arr))
    var = float(np.var(arr))

    if is_integer and non_negative:
        if var > mean * 1.25:
            return {
                "family": "negative_binomial",
                "reason": "non-negative count with over-dispersion",
                "mean": mean,
                "variance": var,
            }
        return {
            "family": "poisson",
            "reason": "non-negative count with variance near mean",
            "mean": mean,
            "variance": var,
        }

    if target_name and "diff" in target_name:
        return {
            "family": "skellam_or_gaussian",
            "reason": "signed run differential; use Skellam when team-run means are modeled separately",
            "mean": mean,
            "variance": var,
        }

    return {
        "family": "gaussian",
        "reason": "continuous or signed scalar target",
        "mean": mean,
        "variance": var,
    }

