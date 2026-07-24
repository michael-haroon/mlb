"""Distribution and market derivation module for live MLB prediction trading.

This module implements:
1. Negative Binomial loss functions for count-data regression (runs scored)
2. Joint distribution computation over bivariate score outcomes
3. Market probability derivation for all 21 Kalshi market families
4. Numerical integration over distribution tails for pricing

WHY Negative Binomial: MLB run distributions exhibit overdispersion (variance >> mean)
due to clustering in big innings. NegBin captures this via a dispersion parameter;
Poisson (variance = mean) systematically underestimates tail probabilities.

WHY joint distribution: All market families (win, spread, totals, YRFI, F5) derive from
a single bivariate prediction over (home_runs, away_runs). This preserves consistency
across markets and allows real-time repricing from a single model forward pass.

References:
- Karlis & Ntzoufras (2003): "Analysis of Sports Data Using Bivariate Poisson Models"
- Goddard (2005): "Regression Models for Forecasting Goals and Match Results"

Example usage:

    # Training: use negbin_nll as the loss function
    import torch
    from live.mlb_dl.distributions import negbin_nll, weighted_mean

    y_true = torch.tensor([3.0, 5.0, 2.0])
    mu = model_output["mu_home_remaining"]  # shape: (batch_size,)
    alpha = model_output["alpha_home_remaining"]

    loss = negbin_nll(y_true, mu, alpha)
    weighted_loss = weighted_mean(loss, sample_weights)
    weighted_loss.backward()

    # Inference: derive market probabilities
    from live.mlb_dl.distributions import MarketDeriver

    deriver = MarketDeriver(max_runs=25)
    markets = deriver.derive_all_markets(
        h_observed=3,
        a_observed=2,
        mu_h_remaining=2.8,
        alpha_h_remaining=1.6,
        mu_a_remaining=3.1,
        alpha_a_remaining=1.7,
        innings_completed=5.5,
    )

    # markets["home_win"] = 0.4823
    # markets["total_runs_over_8.5"] = 0.5234
    # markets["first_5_home_win"] = N/A (past 5th inning)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# ============================================================================
# Legacy Loss Functions (backward compatibility)
# ============================================================================


def gaussian_nll(target, mu, sigma):
    """Gaussian negative log-likelihood. Legacy; kept for backward compatibility."""
    import torch

    sigma = sigma.clamp_min(1e-6)
    return 0.5 * (((target - mu) / sigma) ** 2 + 2.0 * torch.log(sigma) + math.log(2.0 * math.pi))


def weighted_mean(loss, sample_weight):
    """Sample-weight-aware mean. Stable denominator prevents division by zero."""
    import torch

    denom = sample_weight.sum().clamp_min(1e-6)
    return (loss * sample_weight).sum() / denom


def suggest_distribution(values, target_name: str | None = None) -> dict:
    """Suggest an output family from observed target values.

    This is a lightweight audit helper, not a substitute for validation-set
    log-loss comparisons.
    """

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


# ============================================================================
# Negative Binomial Loss Functions
# ============================================================================


def negbin_nll(y_true, mu, alpha) -> Any:
    """Negative Binomial negative log-likelihood.

    Parameterization:
        mean = mu
        variance = mu + mu²/alpha (alpha is dispersion parameter)

    When alpha→∞, variance→mu and the distribution converges to Poisson.
    When alpha is small, overdispersion is high (big-inning clustering).

    WHY this is the right loss: NLL is a proper scoring rule — the only loss that
    incentivizes the model to output the true data-generating distribution. MSE on
    counts incentivizes predicting the conditional mean, not the full distribution.

    Args:
        y_true: Observed counts (non-negative integers)
        mu: Predicted mean (must be > 0)
        alpha: Dispersion parameter (must be > 0)

    Returns:
        Negative log-likelihood per sample (scalar per row)
    """
    import torch
    from torch.nn.functional import softplus

    # Model head already applies softplus; only clamp here for numerical safety
    mu = mu.clamp_min(1e-6)
    alpha = alpha.clamp_min(1e-3)

    # NegBin NLL: -log P(Y=y | mu, alpha)
    # P(Y=y) = Γ(y+α)/[Γ(α)Γ(y+1)] * (α/(α+μ))^α * (μ/(α+μ))^y
    # => NLL = -lgamma(y+α) + lgamma(α) + lgamma(y+1)
    #          - α*log(α) + α*log(α+μ) - y*log(μ) + y*log(α+μ)
    nll = (
        -torch.lgamma(y_true + alpha)
        + torch.lgamma(alpha)
        + torch.lgamma(y_true + 1.0)
        - alpha * torch.log(alpha)
        + alpha * torch.log(alpha + mu)
        - y_true * torch.log(mu)
        + y_true * torch.log(alpha + mu)
    )

    return nll


# ============================================================================
# Negative Binomial PMF / CDF (NumPy for inference)
# ============================================================================


def negbin_pmf(k: np.ndarray, mu: float, alpha: float) -> np.ndarray:
    """P(X=k) for X ~ NegBin(mu, alpha). Vectorized over k.

    Uses scipy's gamma functions for numerical stability.
    """
    from scipy.special import gammaln

    # Clamp parameters
    mu = max(mu, 1e-6)
    alpha = max(alpha, 1e-3)

    # Convert to scipy parameterization: n=alpha, p=alpha/(alpha+mu)
    p = alpha / (alpha + mu)

    # log P(X=k) = log Γ(k + α) - log Γ(α) - log Γ(k+1) + α log(p) + k log(1-p)
    log_pmf = gammaln(k + alpha) - gammaln(alpha) - gammaln(k + 1.0) + alpha * np.log(p) + k * np.log(1.0 - p)

    return np.exp(log_pmf)


def negbin_cdf(k: np.ndarray, mu: float, alpha: float) -> np.ndarray:
    """P(X <= k) for X ~ NegBin(mu, alpha). Vectorized over k."""
    from scipy.stats import nbinom

    mu = max(mu, 1e-6)
    alpha = max(alpha, 1e-3)

    # scipy parameterization: n=alpha, p=alpha/(alpha+mu)
    n = alpha
    p = alpha / (alpha + mu)

    return nbinom.cdf(k, n, p)


def negbin_log_pmf(k: np.ndarray, mu: float, alpha: float) -> np.ndarray:
    """Log P(X=k) — for numerical stability in tail computations."""
    from scipy.special import gammaln

    mu = max(mu, 1e-6)
    alpha = max(alpha, 1e-3)

    p = alpha / (alpha + mu)
    log_pmf = gammaln(k + alpha) - gammaln(alpha) - gammaln(k + 1.0) + alpha * np.log(p) + k * np.log(1.0 - p)

    return log_pmf


# ============================================================================
# Joint Distribution (Bivariate Scoring)
# ============================================================================


def joint_pmf_independent(
    h: np.ndarray,
    a: np.ndarray,
    mu_h: float,
    alpha_h: float,
    mu_a: float,
    alpha_a: float,
) -> np.ndarray:
    """P(Home=h, Away=a) under independence assumption.

    WHY independence: Starting with independent marginals because the shared
    environmental covariance (lambda_3 in Karlis & Ntzoufras 2003) requires
    a copula or shared latent variable that adds training complexity.
    Independence is the conservative baseline — copula is Phase 2.

    Returns:
        Joint probability for each (h, a) pair
    """
    p_h = negbin_pmf(h, mu_h, alpha_h)
    p_a = negbin_pmf(a, mu_a, alpha_a)
    return p_h * p_a


def joint_pmf_grid(
    mu_h: float,
    alpha_h: float,
    mu_a: float,
    alpha_a: float,
    max_runs: int = 25,
) -> np.ndarray:
    """Compute the full joint PMF grid P(H=h, A=a) for h,a in [0, max_runs].

    WHY max_runs=25: P(team scores >25 runs) < 1e-8 for mu < 10, so truncation
    error is negligible. Grids larger than 26x26 slow down without improving accuracy.

    Returns:
        (max_runs+1, max_runs+1) array where [h, a] = P(Home=h, Away=a)
    """
    h_vals = np.arange(max_runs + 1)
    a_vals = np.arange(max_runs + 1)

    # Compute marginal PMFs
    p_h = negbin_pmf(h_vals, mu_h, alpha_h)
    p_a = negbin_pmf(a_vals, mu_a, alpha_a)

    # Outer product for joint under independence
    joint_grid = np.outer(p_h, p_a)

    # Normalize to ensure sum=1.0 (accounts for tail truncation)
    joint_grid /= joint_grid.sum()

    return joint_grid


# ============================================================================
# Market Derivation
# ============================================================================


class MarketDeriver:
    """Converts NegBin distribution parameters into market probabilities.

    Given observed scores (h_obs, a_obs) and predicted remaining-run distributions,
    derives probabilities for all market families.

    WHY this approach: The model predicts REMAINING runs (conditioned on game state),
    not total runs. Final score = observed + remaining. This structural decomposition
    means the model only needs to predict the unknown future, not re-derive what's
    already happened.
    """

    def __init__(self, max_runs: int = 35):
        """
        Args:
            max_runs: Caps the PMF grid. 35 covers early-game states (mu=10-12)
                without significant truncation bias (<0.1% mass lost).
        """
        self.max_runs = max_runs

    def derive_all_markets(
        self,
        h_observed: int,
        a_observed: int,
        mu_h_remaining: float,
        alpha_h_remaining: float,
        mu_a_remaining: float,
        alpha_a_remaining: float,
        innings_completed: float,
    ) -> dict[str, float]:
        """Derive probabilities for all market families.

        Args:
            h_observed: Runs scored by home team so far
            a_observed: Runs scored by away team so far
            mu_h_remaining: Predicted mean remaining runs for home
            alpha_h_remaining: Dispersion parameter for home remaining runs
            mu_a_remaining: Predicted mean remaining runs for away
            alpha_a_remaining: Dispersion parameter for away remaining runs
            innings_completed: Game progress (e.g., 5.0 after top of 6th)

        Returns:
            Dictionary with market keys and probabilities:
                "home_win": P(H_final > A_final)
                "away_win": P(A_final > H_final)
                "extra_innings": P(H_final == A_final after 9 innings)
                "total_runs_over_X.5": P(H_final + A_final > X.5)
                "home_runs_over_X.5": P(H_final > X.5)
                "away_runs_over_X.5": P(A_final > X.5)
                "home_spread_minus_X.5": P(H_final - A_final > X.5)
                "first_5_home_win": P(H_5 > A_5) — only valid before inning 6
                "first_5_total_over_X.5": P(H_5 + A_5 > X.5)
        """
        # Build joint PMF grid over remaining runs
        joint_grid_remaining = joint_pmf_grid(
            mu_h_remaining,
            alpha_h_remaining,
            mu_a_remaining,
            alpha_a_remaining,
            max_runs=self.max_runs,
        )

        # Shift grid to reflect final scores (observed + remaining)
        # joint_grid_final[h_final, a_final] = P(Home=h_final, Away=a_final)
        joint_grid_final = self._shift_joint_grid(joint_grid_remaining, h_observed, a_observed)

        markets = {}

        # Win probabilities
        markets["home_win"] = self.p_home_win(joint_grid_final)
        markets["away_win"] = 1.0 - markets["home_win"] - self.p_tie(joint_grid_final)
        markets["extra_innings"] = self.p_tie(joint_grid_final)

        # Standard lines
        standard_lines = self.derive_standard_lines(joint_grid_final)
        markets.update(standard_lines)

        # First-5 markets (only valid if innings_completed < 5)
        if innings_completed < 5.0:
            f5_markets = self.derive_first_5_markets(
                h_observed,
                a_observed,
                mu_h_remaining,
                alpha_h_remaining,
                mu_a_remaining,
                alpha_a_remaining,
                innings_completed,
            )
            markets.update(f5_markets)

        return markets

    def _shift_joint_grid(self, joint_grid: np.ndarray, h_obs: int, a_obs: int) -> np.ndarray:
        """Shift PMF grid from remaining runs to final scores.

        If grid is [0..max_runs] and observed = (h_obs, a_obs),
        then final[h_obs + h_rem, a_obs + a_rem] = grid[h_rem, a_rem].

        Returns:
            Shifted grid of shape (max_runs + h_obs + 1, max_runs + a_obs + 1)
        """
        n_h, n_a = joint_grid.shape
        final_max_h = n_h + h_obs - 1
        final_max_a = n_a + a_obs - 1

        shifted = np.zeros((final_max_h + 1, final_max_a + 1))

        for h_rem in range(n_h):
            for a_rem in range(n_a):
                h_final = h_obs + h_rem
                a_final = a_obs + a_rem
                shifted[h_final, a_final] = joint_grid[h_rem, a_rem]

        return shifted

    def p_home_win(self, joint_grid: np.ndarray) -> float:
        """P(home wins) = sum over all (h, a) where h > a."""
        n_h, n_a = joint_grid.shape
        prob = 0.0
        for h in range(n_h):
            for a in range(min(h, n_a)):  # a < h
                prob += joint_grid[h, a]
        return float(prob)

    def p_tie(self, joint_grid: np.ndarray) -> float:
        """P(tie) = sum over diagonal where h == a."""
        n_h, n_a = joint_grid.shape
        prob = 0.0
        for h in range(min(n_h, n_a)):
            prob += joint_grid[h, h]
        return float(prob)

    def p_total_over(self, joint_grid: np.ndarray, line: float) -> float:
        """P(total > line). Line can be X.5 (standard) or integer."""
        n_h, n_a = joint_grid.shape
        prob = 0.0
        for h in range(n_h):
            for a in range(n_a):
                if h + a > line:
                    prob += joint_grid[h, a]
        return float(prob)

    def p_spread_cover(self, joint_grid: np.ndarray, spread: float) -> float:
        """P(home - away > spread). Spread is from home perspective.

        Example: spread = -1.5 means home is 1.5-run underdog.
        We compute P(H - A > -1.5) = P(H >= A - 1) = P(H wins or loses by 1).
        """
        n_h, n_a = joint_grid.shape
        prob = 0.0
        for h in range(n_h):
            for a in range(n_a):
                if h - a > spread:
                    prob += joint_grid[h, a]
        return float(prob)

    def p_team_total_over(
        self,
        mu_remaining: float,
        alpha_remaining: float,
        observed: int,
        line: float,
    ) -> float:
        """P(team_final > line) where team_final = observed + remaining.

        WHY marginal computation: Faster than joint grid for single-team markets.
        """
        # P(observed + remaining > line) = P(remaining > line - observed)
        threshold = line - observed

        if threshold < 0:
            return 1.0  # Already over the line

        # Compute marginal CDF
        k_max = int(np.ceil(threshold)) + 20  # tail buffer
        k_vals = np.arange(k_max + 1)
        cdf = negbin_cdf(k_vals, mu_remaining, alpha_remaining)

        # P(X > threshold) = 1 - P(X <= floor(threshold)) if line is X.5
        if threshold == int(threshold) + 0.5:
            k_threshold = int(threshold)
            return 1.0 - cdf[k_threshold]
        else:
            # Integer line: P(X > k) = 1 - P(X <= k)
            k_threshold = int(threshold)
            return 1.0 - cdf[k_threshold]

    def derive_standard_lines(self, joint_grid: np.ndarray) -> dict[str, float]:
        """Compute probabilities for all standard Kalshi lines.

        Standard total lines: 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5
        Standard spreads: -2.5, -1.5, -0.5, 0.5, 1.5, 2.5

        # TODO: validate — lines from Kalshi API 2026-07-06; may add/remove lines
        """
        markets = {}

        # Total lines
        for line in [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]:
            markets[f"total_runs_over_{line}"] = self.p_total_over(joint_grid, line)

        # Spread lines (home perspective)
        for spread in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
            key = f"home_spread_{'+' if spread >= 0 else ''}{spread}"
            key = key.replace(".", "_")  # Normalize to "home_spread_minus_2_5"
            key = key.replace("+-", "minus_")
            key = key.replace("+", "plus_")
            key = key.replace("-", "minus_")
            markets[key] = self.p_spread_cover(joint_grid, spread)

        return markets

    def derive_first_5_markets(
        self,
        h_observed: int,
        a_observed: int,
        mu_h_remaining_full: float,
        alpha_h_remaining_full: float,
        mu_a_remaining_full: float,
        alpha_a_remaining_full: float,
        innings_completed: float,
    ) -> dict[str, float]:
        """Derive first-5-innings markets from full-game predictions.

        WHY scaling: The model predicts remaining runs for the full game. For F5 markets,
        we need P(runs in innings [current+1 ... 5]). Linear scaling of mu by fraction
        of innings remaining in F5 window; alpha unchanged (overdispersion is a property
        of the process, not time).

        # TODO: validate — linear mu scaling vs. learned F5-specific head
        """
        # How many innings left in F5 window?
        innings_left_in_f5 = max(0.0, 5.0 - innings_completed)
        innings_left_full = 9.0 - innings_completed

        if innings_left_full <= 0 or innings_left_in_f5 <= 0:
            return {}  # No F5 markets after inning 5

        # Scale mu by innings fraction
        scale_factor = innings_left_in_f5 / innings_left_full
        mu_h_f5 = mu_h_remaining_full * scale_factor
        mu_a_f5 = mu_a_remaining_full * scale_factor

        # Build joint grid for F5 remaining runs
        joint_grid_f5_remaining = joint_pmf_grid(
            mu_h_f5,
            alpha_h_remaining_full,  # alpha unchanged
            mu_a_f5,
            alpha_a_remaining_full,
            max_runs=self.max_runs,
        )

        joint_grid_f5_final = self._shift_joint_grid(joint_grid_f5_remaining, h_observed, a_observed)

        markets = {}
        markets["first_5_home_win"] = self.p_home_win(joint_grid_f5_final)

        # F5 total lines
        for line in [3.5, 4.5, 5.5, 6.5]:
            markets[f"first_5_total_over_{line}"] = self.p_total_over(joint_grid_f5_final, line)

        return markets


# ============================================================================
# YRFI / NRFI Derivation
# ============================================================================


def p_yrfi_from_negbin(
    mu_h_first_inning: float,
    alpha_h_first_inning: float,
    mu_a_first_inning: float,
    alpha_a_first_inning: float,
) -> float:
    """P(YRFI) = 1 - P(Home=0 in 1st) * P(Away=0 in 1st).

    WHY NegBin over Poisson for first inning: Even single-inning run distributions
    are overdispersed due to multi-run scoring events (HR with runners on, errors
    leading to big innings). NegBin captures this; Poisson underestimates P(0).

    WHY independence: First-inning scoring is weakly correlated (shared environment
    affects both teams, but batting/pitching matchups dominate). Independence is
    the conservative baseline.
    """
    p_h_zero = negbin_pmf(np.array([0]), mu_h_first_inning, alpha_h_first_inning)[0]
    p_a_zero = negbin_pmf(np.array([0]), mu_a_first_inning, alpha_a_first_inning)[0]

    p_nrfi = p_h_zero * p_a_zero
    return 1.0 - p_nrfi


# ============================================================================
# Utility: Scale full-game predictions to first-5
# ============================================================================


def scale_to_first_5(
    mu_full: float,
    alpha_full: float,
    innings_completed: float,
) -> tuple[float, float]:
    """Scale full-game NegBin parameters to first-5-innings horizon.

    WHY: If the model predicts remaining runs for the full game, and we're
    in inning 3, we need P(runs in innings 4-5 only) for F5 markets.

    Approach: Linear scaling of mu by fraction of innings remaining in F5 window.
    alpha stays unchanged (overdispersion is a property of the process, not time).

    # TODO: validate — linear mu scaling vs. learned F5-specific head

    Args:
        mu_full: Predicted mean remaining runs for full game
        alpha_full: Dispersion parameter for full game
        innings_completed: Current inning progress (e.g., 3.5 after bottom of 4th)

    Returns:
        (mu_f5, alpha_f5): Scaled parameters for first-5-innings window
    """
    innings_left_in_f5 = max(0.0, 5.0 - innings_completed)
    innings_left_full = 9.0 - innings_completed

    if innings_left_full <= 0 or innings_left_in_f5 <= 0:
        return (0.0, alpha_full)

    scale_factor = innings_left_in_f5 / innings_left_full
    mu_f5 = mu_full * scale_factor
    # WHY scale alpha: NegBin variance = mu + mu²/alpha. To preserve the
    # variance-to-mean ratio under time-subsetting, alpha must scale with mu.
    # Keeping alpha fixed produces variance that is too low (overconfident).
    alpha_f5 = alpha_full * scale_factor

    return (mu_f5, alpha_f5)
