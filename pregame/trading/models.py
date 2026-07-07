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

from ..strategy.predict import predict_game, predict_spread_lines, predict_total_lines
from ..strategy.calibration import cover_probability
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
        """Clear cache and re-discover. Call after feature refresh."""
        self._loaded.clear()
        self.discover()

    @property
    def tradeable_targets(self) -> list[str]:
        return [t for t in self._bundles if MODEL_TO_SERIES.get(t) is not None]


_TIER_RANK = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}


def predict_derived_team_total(
    which: str,
    features: pd.DataFrame,
    total_path: Path,
    diff_path: Path,
    line: float,
    direction: str = "over",
) -> dict:
    """Price a team-total market by deriving from total_runs and home_run_diff.

    home_runs = (total_runs + home_run_diff) / 2
    away_runs = (total_runs - home_run_diff) / 2

    Scale propagation assumes independence (conservative upper bound on uncertainty):
      scale_derived = sqrt(scale_total² + scale_diff²) / 2
    df is taken as min(df_total, df_diff) for heavier tails (more conservative).
    """
    result_total = predict_game(features, total_path, "total_runs")
    if "error" in result_total:
        return result_total

    result_diff = predict_game(features, diff_path, "home_run_diff")
    if "error" in result_diff:
        return result_diff

    mu_total = float(result_total["point_estimate"][0]) if hasattr(result_total["point_estimate"], "__len__") else float(result_total["point_estimate"])
    mu_diff  = float(result_diff["point_estimate"][0])  if hasattr(result_diff["point_estimate"],  "__len__") else float(result_diff["point_estimate"])

    if which == "home_runs":
        mu = (mu_total + mu_diff) / 2.0
    else:
        mu = (mu_total - mu_diff) / 2.0

    sc_total = result_total["distribution"]["scale"]
    sc_diff  = result_diff["distribution"]["scale"]
    scale = np.sqrt(sc_total ** 2 + sc_diff ** 2) / 2.0

    df = min(result_total["distribution"]["df"], result_diff["distribution"]["df"])

    std_total = float(result_total.get("ensemble_std", np.array([0.05]))[0])
    std_diff  = float(result_diff.get("ensemble_std",  np.array([0.05]))[0])
    ensemble_std = np.sqrt(std_total ** 2 + std_diff ** 2) / 2.0

    tier_total = str(result_total.get("confidence_tier", np.array(["MEDIUM"]))[0])
    tier_diff  = str(result_diff.get("confidence_tier",  np.array(["MEDIUM"]))[0])
    confidence_tier = tier_total if _TIER_RANK.get(tier_total, 1) <= _TIER_RANK.get(tier_diff, 1) else tier_diff

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
) -> dict:
    """Generate a calibrated probability for a specific market line.

    For classification targets (home_win, yrfi, etc.):
      Returns calibrated_prob directly — no line needed.

    For regression targets (total_runs, home_run_diff, etc.):
      Uses the Student-t distribution to compute P(actual > line).

    Returns:
        dict with keys: prob, ensemble_std, confidence_tier, task, n_models_used
    """
    result = predict_game(features, ensemble_path, target)

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

    # Regression: need a specific line to price
    if line is None:
        return {"error": f"Regression target {target} requires a line to price"}

    mu = float(result["point_estimate"][0]) if hasattr(result["point_estimate"], "__len__") else float(result["point_estimate"])
    df = result["distribution"]["df"]
    scale = result["distribution"]["scale"]

    prob = cover_probability(mu, line, df, scale, direction=direction)
    std = float(result.get("ensemble_std", np.array([0.05]))[0])
    tier = result.get("confidence_tier", np.array(["MEDIUM"]))[0]

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
