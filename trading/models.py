"""
pregame/trading/models.py
-------------------------
Loads trained ensemble bundles and wraps the inference pipeline for trading.

Provides a unified interface that:
1. Loads all ensemble .pkl files (one per target)
2. Runs predict_game() for classification targets
3. Runs predict_spread_lines() / predict_total_lines() for regression targets
4. Returns probabilities for specific Kalshi market lines
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from classical_learning.strategy.predict import predict_game, predict_spread_lines, predict_total_lines
from classical_learning.strategy.calibration import cover_probability, negbin_cover_probability, negbin_total_cover_probability
from .config import ARTIFACTS_DIR
from .market_map import MODEL_TO_SERIES

logger = logging.getLogger(__name__)

MODELS_DIR = ARTIFACTS_DIR / "models"


class EnsembleStore:
    """Manages loaded ensemble bundles for all targets."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self._models_dir = models_dir
        self._bundles: dict[str, Path] = {}  # target → ensemble .pkl path
        self._loaded: dict[str, dict] = {}   # target → loaded bundle dict (cached)
        # Cross-scan inference cache: (away, home, target, features_hash) → result dict.
        # Cleared on reload_all() — i.e. only after a settlement-triggered feature rebuild.
        # Without this, 608 markets × full ensemble runs every 60s all night on unchanged data.
        self.inference_cache: dict[tuple, dict] = {}
        self._inference_cache_features_hash: str | None = None

    def discover(self) -> list[str]:
        """Find all trained ensemble .pkl files and register them.

        Naming convention on disk: ensemble_{target}_A.pkl (flat in models/).
        """
        self._bundles.clear()
        if not self._models_dir.exists():
            logger.warning(f"Models directory not found: {self._models_dir}")
            return []

        for pkl in sorted(self._models_dir.glob("ensemble_*_A.pkl")):
            # Strip "ensemble_" prefix and "_A" suffix to recover target name
            target = pkl.stem[len("ensemble_"):-len("_A")]
            self._bundles[target] = pkl
            logger.debug(f"Found ensemble for target: {target}")

        tradeable = [t for t in self._bundles if MODEL_TO_SERIES.get(t) is not None]
        logger.info(f"Discovered {len(self._bundles)} ensembles, "
                    f"{len(tradeable)} have Kalshi markets: {tradeable}")
        return tradeable

    def get_bundle(self, target: str) -> Optional[dict]:
        """Load and cache an ensemble bundle."""
        if target in self._loaded:
            return self._loaded[target]
        pkl_path = self._bundles.get(target)
        if not pkl_path or not pkl_path.exists():
            return None
        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)
        self._loaded[target] = bundle
        logger.info(f"Loaded ensemble for {target}: {len(bundle.get('member_bundles', []))} members")
        return bundle

    def reload_all(self) -> None:
        """Clear caches and re-discover. Call after feature refresh."""
        self._loaded.clear()
        self.inference_cache.clear()
        self._inference_cache_features_hash = None
        self.discover()

    def invalidate_inference_cache(self) -> None:
        """Clear the cross-scan inference cache when features change."""
        self.inference_cache.clear()
        self._inference_cache_features_hash = None

    @property
    def tradeable_targets(self) -> list[str]:
        return [t for t in self._bundles if MODEL_TO_SERIES.get(t) is not None]

    def validate_features(self, features: pd.DataFrame, max_missing_pct: float = 0.05) -> None:
        """Validate that feature parquet contains what all loaded models expect.

        Raises RuntimeError if any model has >max_missing_pct features absent
        from the parquet. Call at startup before entering the trading loop.
        """
        parquet_cols = set(features.columns)
        failures = []

        for target in self.tradeable_targets:
            bundle = self.get_bundle(target)
            if bundle is None:
                continue
            for i, mb in enumerate(bundle.get("member_bundles", [])):
                model_cols = mb.get("feature_columns", [])
                if not model_cols:
                    continue
                missing = [c for c in model_cols if c not in parquet_cols]
                missing_pct = len(missing) / len(model_cols)
                if missing_pct > max_missing_pct:
                    failures.append(
                        f"{target}[{i}]: {len(missing)}/{len(model_cols)} "
                        f"({missing_pct:.1%}) features missing — "
                        f"first 5: {missing[:5]}"
                    )

        if failures:
            msg = (
                "FATAL: Feature/model mismatch — refusing to trade.\n"
                + "\n".join(f"  {f}" for f in failures)
                + "\nRebuild the feature parquet with current engineering code."
            )
            logger.error(msg)
            raise RuntimeError(msg)


_TIER_RANK = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}


def predict_derived_team_total(
    which: str,
    features: pd.DataFrame,
    total_path: Path,
    diff_path: Path,
    line: float,
    direction: str = "over",
    label: str = "",
) -> dict:
    """Price a team-total market by deriving from total_runs and home_run_diff.

    home_runs = (total_runs + home_run_diff) / 2
    away_runs = (total_runs - home_run_diff) / 2

    When total_runs uses NegBin distribution, team totals are priced as
    independent NegBin marginals with mu = total_mu / 2 adjusted by diff.
    For Student-t fallback, scale propagation assumes independence:
      scale_derived = sqrt(scale_total² + scale_diff²) / 2
    """
    result_total = predict_game(features, total_path, "total_runs", label=label)
    if "error" in result_total:
        return result_total

    result_diff = predict_game(features, diff_path, "home_run_diff", label=label)
    if "error" in result_diff:
        return result_diff

    mu_total = float(result_total["point_estimate"][0]) if hasattr(result_total["point_estimate"], "__len__") else float(result_total["point_estimate"])
    mu_diff  = float(result_diff["point_estimate"][0])  if hasattr(result_diff["point_estimate"],  "__len__") else float(result_diff["point_estimate"])

    if which == "home_runs":
        mu = (mu_total + mu_diff) / 2.0
    else:
        mu = (mu_total - mu_diff) / 2.0

    std_total = float(result_total.get("ensemble_std", np.array([0.05]))[0])
    std_diff  = float(result_diff.get("ensemble_std",  np.array([0.05]))[0])
    ensemble_std = np.sqrt(std_total ** 2 + std_diff ** 2) / 2.0

    tier_total = str(result_total.get("confidence_tier", np.array(["MEDIUM"]))[0])
    tier_diff  = str(result_diff.get("confidence_tier",  np.array(["MEDIUM"]))[0])
    confidence_tier = tier_total if _TIER_RANK.get(tier_total, 1) <= _TIER_RANK.get(tier_diff, 1) else tier_diff

    dist_total = result_total.get("distribution", {})
    dist_type = dist_total.get("type", "student_t")

    if dist_type == "negbin":
        alpha = dist_total["alpha"]
        # NegBin marginal for team total: use the derived mu with same alpha.
        # Alpha is a global overdispersion param stable across targets (CV=0.059).
        if line is None:
            return {
                "ensemble_std": ensemble_std,
                "confidence_tier": confidence_tier,
                "task": "regression",
                "n_models_used": result_total["n_models_used"] + result_diff["n_models_used"],
                "point_estimate": mu,
                "distribution": {"type": "negbin", "mu": mu, "alpha": alpha},
            }
        prob = negbin_cover_probability(mu, line, alpha, direction=direction)
        return {
            "prob": prob,
            "ensemble_std": ensemble_std,
            "confidence_tier": confidence_tier,
            "task": "regression",
            "n_models_used": result_total["n_models_used"] + result_diff["n_models_used"],
            "point_estimate": mu,
        }

    # Student-t fallback for signed targets
    sc_total = dist_total.get("scale", 1.0)
    sc_diff  = result_diff.get("distribution", {}).get("scale", 1.0)
    scale = np.sqrt(sc_total ** 2 + sc_diff ** 2) / 2.0

    df = min(dist_total.get("df", 7), result_diff.get("distribution", {}).get("df", 7))

    # line=None means the scanner wants the raw distribution for caching;
    # _apply_line will re-integrate at each market's specific line cheaply.
    if line is None:
        return {
            "ensemble_std": ensemble_std,
            "confidence_tier": confidence_tier,
            "task": "regression",
            "n_models_used": result_total["n_models_used"] + result_diff["n_models_used"],
            "point_estimate": mu,
            "residual_df": df,
            "residual_scale": scale,
            "distribution": {"type": "student_t", "mu": mu, "df": df, "scale": scale},
        }

    prob = cover_probability(mu, line, df, scale, direction=direction)

    return {
        "prob": prob,
        "ensemble_std": ensemble_std,
        "confidence_tier": confidence_tier,
        "task": "regression",
        "n_models_used": result_total["n_models_used"] + result_diff["n_models_used"],
        "point_estimate": mu,
        "residual_df": df,
        "residual_scale": scale,
    }


def predict_market_prob(
    target: str,
    features: pd.DataFrame,
    ensemble_path: Path,
    line: Optional[float] = None,
    direction: str = "over",
    label: str = "",
) -> dict:
    """Generate a calibrated probability for a specific market line.

    For classification targets (home_win, yrfi, etc.):
      Returns calibrated_prob directly — no line needed.

    For regression targets (total_runs, home_run_diff, etc.):
      Uses the Student-t distribution to compute P(actual > line).

    Returns:
        dict with keys: prob, ensemble_std, confidence_tier, task, n_models_used
    """
    result = predict_game(features, ensemble_path, target, label=label)

    if "error" in result:
        return result

    task = result["task"]

    if task == "classification":
        prob = float(result["calibrated_prob"][0])
        std = float(result.get("ensemble_std", np.array([0.05]))[0])
        tier = result.get("confidence_tier", np.array(["MEDIUM"]))[0]
        return {
            "prob": prob,
            "ensemble_std": std,
            "confidence_tier": str(tier),
            "task": task,
            "n_models_used": result["n_models_used"],
        }

    # Regression: extract distribution params
    mu = float(result["point_estimate"][0]) if hasattr(result["point_estimate"], "__len__") else float(result["point_estimate"])
    std = float(result.get("ensemble_std", np.array([0.05]))[0])
    tier = result.get("confidence_tier", np.array(["MEDIUM"]))[0]
    dist = result.get("distribution", {})
    dist_type = dist.get("type", "student_t")

    # line=None: scanner wants the raw distribution for caching;
    # _apply_line will re-integrate at each market's specific line cheaply.
    if line is None:
        return {
            "ensemble_std": std,
            "confidence_tier": str(tier),
            "task": task,
            "n_models_used": result["n_models_used"],
            "point_estimate": mu,
            "distribution": dist,
        }

    # Price at specific line using the appropriate distributional model
    if dist_type == "negbin":
        alpha = dist["alpha"]
        prob = negbin_cover_probability(mu, line, alpha, direction=direction)
        return {
            "prob": prob,
            "ensemble_std": std,
            "confidence_tier": str(tier),
            "task": task,
            "n_models_used": result["n_models_used"],
            "point_estimate": mu,
        }

    # Student-t fallback
    df = dist.get("df", 7)
    scale = dist.get("scale", 1.0)
    prob = cover_probability(mu, line, df, scale, direction=direction)

    return {
        "prob": prob,
        "ensemble_std": std,
        "confidence_tier": str(tier),
        "task": task,
        "n_models_used": result["n_models_used"],
        "point_estimate": mu,
        "residual_df": df,
        "residual_scale": scale,
    }
