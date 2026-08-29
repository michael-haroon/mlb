"""Total runs target configuration and pricing logic."""
from __future__ import annotations

import numpy as np

from .base import TargetConfig


def _negbin_half_spread(mu: float, alpha: float) -> int:
    """NegBin-derived spread width. Higher mu = wider NegBin std = wider spread.

    At mu=9 (avg): negbin_std=4.7 → 3 cents
    At mu=13 (high): negbin_std=6.2 → 4 cents
    At mu=7 (low): negbin_std=3.8 → 3 cents
    """
    negbin_std = np.sqrt(mu + mu**2 / alpha)
    raw = 1 + 0.4 * negbin_std
    return max(2, min(5, int(round(raw))))


# Default config (constants loaded from current hardcoded values)
TOTAL_RUNS_CONFIG = TargetConfig(
    name="total_runs",
    distribution_type="negbin",
    model_error_std=0.795,  # Legacy: loaded from artifact at runtime when available
    half_spread_base_cents=3,
    price_floor=0.12,
    price_ceiling=0.88,
    cluster_max_contracts=10,
    standard_lines=(5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5),
    negbin_alpha=6.732,
    compute_half_spread=_negbin_half_spread,
)


def load_total_runs_config(calibration_bundle) -> TargetConfig:
    """Construct TotalRunsConfig from a loaded CalibrationBundle.

    Falls back to hardcoded defaults for fields the bundle doesn't have.
    This allows old pickles (without model_error_std) to still work.
    """
    alpha = getattr(calibration_bundle, 'negbin_alpha', 6.732)
    model_error_std = getattr(calibration_bundle, 'model_error_std', None) or 0.795

    return TargetConfig(
        name="total_runs",
        distribution_type="negbin",
        model_error_std=model_error_std,
        half_spread_base_cents=3,
        price_floor=0.12,
        price_ceiling=0.88,
        cluster_max_contracts=10,
        standard_lines=(5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5),
        negbin_alpha=alpha,
        compute_half_spread=_negbin_half_spread,
    )
