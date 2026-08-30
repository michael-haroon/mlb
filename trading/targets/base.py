"""Base target configuration dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class TargetConfig:
    """Everything target-specific for trading. Each target implements one."""
    name: str
    distribution_type: str  # "negbin" | "student_t"
    model_error_std: float
    half_spread_base_cents: int
    price_floor: float
    price_ceiling: float
    cluster_max_contracts: int
    standard_lines: tuple[float, ...]

    # Loaded from artifact at runtime (None = use defaults)
    negbin_alpha: Optional[float] = None
    recalibrate_cover_prob: Optional[Callable[[float, float], float]] = None
    compute_half_spread: Optional[Callable[[float, float], int]] = None

    def get_half_spread(self, mu: float = 9.0, alpha: float = 6.73) -> int:
        """Return half-spread in cents. Delegates to callable if set."""
        if self.compute_half_spread is not None:
            return self.compute_half_spread(mu, alpha)
        return self.half_spread_base_cents
