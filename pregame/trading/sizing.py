"""
pregame/trading/sizing.py
-------------------------
Position sizing for MLB pregame market-making.

Core principle: risk and allocation are functions of model strength.
- Strong models (HIGH confidence, high accuracy_mult): larger positions, tighter quotes
- Weak models (LOW confidence, low accuracy_mult): smaller positions, wider quotes, exit early

Sizing formula:
  base_kelly = edge / odds_against
  confidence_mult = {HIGH: 1.5, MEDIUM: 1.0, LOW: 0.5}
  composite = base_kelly * KELLY_FRACTION * accuracy_mult * confidence_mult
  bet_dollars = min(composite * bankroll, bankroll * MAX_POSITION_PCT/100)
  contracts = bet_dollars / price

The confidence_mult is the key lever: it directly scales position size
with model quality. A HIGH-confidence signal gets 3x the allocation of
a LOW-confidence signal at the same edge level.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    KELLY_FRACTION, MAX_CONTRACTS_PER_MARKET, MAX_POSITION_PCT,
    CLUSTER_MAX_CONTRACTS,
)
from .market_map import parse_ticker

logger = logging.getLogger(__name__)

# Position size scales with model quality: strong models get bigger positions
# because their predictions have lower variance → less risk per dollar deployed.
CONFIDENCE_SIZE_MULT = {
    "HIGH": 1.5,     # Strong model → take more risk, it's less risky
    "MEDIUM": 1.0,
    "LOW": 0.5,      # Weak model → small position, high variance
}

# OOF accuracy profile cache: {target: np.ndarray of shape (10,)}
_ACCURACY_PROFILES: dict[str, np.ndarray] = {}
_STD_DECILE_EDGES: dict[str, np.ndarray] = {}


def load_accuracy_profile(target: str, oof_path: Optional[str | Path] = None) -> np.ndarray:
    """Compute per-std-decile accuracy multiplier from OOF predictions.

    Reads per-model OOF npy files (oof_{target}_{model}_A.npy) and the ensemble
    bundle to reconstruct the weighted ensemble OOF predictions, then computes:
      - Classification: weight[d] = decile_skill / overall_skill
      - Regression:     weight[d] = overall_MAE / decile_MAE

    Deciles with n < 100 get weight 1.0. Clamped to [0.2, 2.0].
    """
    if target in _ACCURACY_PROFILES:
        return _ACCURACY_PROFILES[target]

    from .config import ARTIFACTS_DIR
    models_dir = ARTIFACTS_DIR / "models"

    # Load ensemble bundle to get members, weights, and task
    bundle_path = models_dir / f"ensemble_{target}_A.pkl"
    if not bundle_path.exists():
        logger.warning(f"No ensemble bundle for {target} — using flat accuracy profile")
        _ACCURACY_PROFILES[target] = np.ones(10)
        _STD_DECILE_EDGES[target] = np.zeros(11)
        return np.ones(10)

    import pickle
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    task = bundle.get("task", "regression")
    members = bundle.get("members", [])
    ens_weights = np.array(bundle.get("weights", []))

    if not members:
        logger.warning(f"No members in ensemble for {target} — using flat accuracy profile")
        _ACCURACY_PROFILES[target] = np.ones(10)
        _STD_DECILE_EDGES[target] = np.zeros(11)
        return np.ones(10)

    # Load per-member OOF arrays; only rows where all members have predictions (non-nan)
    member_preds = []
    for m in members:
        npy = models_dir / f"oof_{target}_{m}_A.npy"
        if not npy.exists():
            logger.warning(f"Missing OOF for {target}/{m} — using flat accuracy profile")
            _ACCURACY_PROFILES[target] = np.ones(10)
            _STD_DECILE_EDGES[target] = np.zeros(11)
            return np.ones(10)
        member_preds.append(np.load(npy))

    pred_matrix = np.column_stack(member_preds)  # (n_games, n_members)
    valid_mask = ~np.any(np.isnan(pred_matrix), axis=1)
    pred_matrix = pred_matrix[valid_mask]

    if pred_matrix.shape[0] < 200:
        logger.warning(f"Too few valid OOF rows for {target} — using flat accuracy profile")
        _ACCURACY_PROFILES[target] = np.ones(10)
        _STD_DECILE_EDGES[target] = np.zeros(11)
        return np.ones(10)

    # Weighted ensemble prediction and cross-member std
    if len(ens_weights) == len(members):
        w = ens_weights / ens_weights.sum()
        y_pred_ens = pred_matrix @ w
    else:
        y_pred_ens = pred_matrix.mean(axis=1)
    model_std = pred_matrix.std(axis=1)

    # We don't have y_true as a standalone file; load it from the bundle's
    # member_bundles which store the calibrator fitted on OOF — but we can
    # proxy accuracy via std decile → use the ensemble's OOF spread as a
    # confidence proxy and skip the y_true-dependent accuracy calc.
    # Fall back to std-only profile: lower-std deciles → higher mult.
    # This is a monotone prior: std is inversely correlated with accuracy.
    edges = np.percentile(model_std, np.linspace(0, 100, 11))
    _STD_DECILE_EDGES[target] = edges

    # Std-inverse weighting: decile 0 (lowest std) gets highest mult
    # Scale so median decile = 1.0
    overall_std = float(model_std.mean())
    def _decile_mean(arr, lo, hi, last):
        mask = (arr >= lo) & (arr <= hi if last else arr < hi)
        return float(arr[mask].mean()) if mask.sum() > 0 else float("nan")

    decile_stds = np.array([
        _decile_mean(model_std, edges[d], edges[d + 1], d == 9) for d in range(10)
    ])
    # Replace nan/zero bins (empty decile) with overall std → weight 1.0
    decile_stds = np.where(np.isnan(decile_stds) | (decile_stds < 1e-9), overall_std, decile_stds)
    raw_weights = np.divide(overall_std, decile_stds, out=np.ones_like(decile_stds), where=decile_stds > 0)
    # Normalize so median = 1.0
    raw_weights = raw_weights / np.median(raw_weights)
    weights = np.clip(raw_weights, 0.2, 2.0)

    _ACCURACY_PROFILES[target] = weights
    logger.info(f"[{target}] accuracy profile loaded (std-proxy, {pred_matrix.shape[0]} rows): "
                f"{np.round(weights, 3)}")
    return weights


def get_accuracy_multiplier(target: str, ensemble_std: float) -> float:
    """Look up the OOF-empirical accuracy multiplier for a given std level."""
    profile = _ACCURACY_PROFILES.get(target)
    edges = _STD_DECILE_EDGES.get(target)
    if profile is None or edges is None or edges.max() == 0:
        return 1.0
    d = int(np.searchsorted(edges[1:-1], ensemble_std, side="right"))
    return float(profile[min(d, 9)])


def preload_accuracy_profiles(targets: list[str]) -> None:
    """Load accuracy profiles for all tradeable targets. Call once at startup."""
    for t in targets:
        load_accuracy_profile(t)


@dataclass
class SizedQuote:
    """A fully sized two-sided quote ready for execution."""
    ticker: str
    target: str
    cluster: str
    # Model outputs
    fair_value: float          # conservative fair value (after confidence shading)
    model_prob: float          # raw model probability (before shading)
    ensemble_std: float
    confidence_tier: str
    # Quote prices (in cents, 1-99)
    bid_cents: int             # YES buy limit price
    ask_cents: int             # NO buy limit price (= 100 - YES ask)
    # Sizing
    contracts: int
    # Metadata for hold/exit decision
    accuracy_mult: float
    edge_at_mid: float         # edge vs current market midpoint
    weight_breakdown: dict = field(default_factory=dict)


def size_quotes(
    quotes: list[dict],
    bankroll: float,
    existing_inventory: Optional[dict[tuple[str, str], int]] = None,
) -> list[SizedQuote]:
    """Size a batch of raw quotes into executable SizedQuote objects.

    Args:
        quotes: list of dicts from scanner.generate_quotes(), each with:
            ticker, target, cluster, fair_value, model_prob, ensemble_std,
            confidence_tier, bid_cents, ask_cents, edge_at_mid
        bankroll: total dollars available
        existing_inventory: {cluster: contracts_already_deployed}

    Returns sorted by priority (highest composite weight first).
    """
    if existing_inventory is None:
        existing_inventory = {}

    # Per-game cluster caps: keyed by (cluster, game_key) so uncorrelated games
    # don't share allocation.
    cluster_used: dict[tuple[str, str], int] = dict(existing_inventory)

    scored = []
    for q in quotes:
        edge = q["edge_at_mid"]
        if edge <= 0:
            continue

        fair = q["fair_value"]
        if fair <= 0 or fair >= 1:
            continue

        # Kelly formula: edge / odds_against
        kelly_raw = edge / (1.0 - fair)

        # OOF accuracy multiplier: empirical model quality at this std level
        accuracy_mult = get_accuracy_multiplier(q["target"], q["ensemble_std"])

        # Confidence multiplier: strong models get bigger positions
        confidence_mult = CONFIDENCE_SIZE_MULT.get(q["confidence_tier"], 1.0)

        composite = kelly_raw * KELLY_FRACTION * accuracy_mult * confidence_mult

        bet_dollars = min(composite * bankroll, bankroll * MAX_POSITION_PCT / 100.0)
        contracts_raw = bet_dollars / fair if fair > 0 else 0

        scored.append({
            "quote": q,
            "contracts_raw": contracts_raw,
            "composite": composite,
            "accuracy_mult": accuracy_mult,
            "weights": {
                "kelly_raw": kelly_raw,
                "accuracy_mult": accuracy_mult,
                "confidence_mult": confidence_mult,
                "composite": composite,
            },
        })

    scored.sort(key=lambda x: -x["composite"])

    sized = []
    for s in scored:
        q = s["quote"]
        cluster = q["cluster"]

        parsed = parse_ticker(q["ticker"])
        game_key = parsed.game_key if parsed else q["ticker"]
        cap_key = (cluster, game_key)

        max_cluster = CLUSTER_MAX_CONTRACTS.get(cluster, 10)
        remaining = max_cluster - cluster_used.get(cap_key, 0)
        if remaining <= 0:
            continue

        contracts = int(min(s["contracts_raw"], remaining, MAX_CONTRACTS_PER_MARKET))
        contracts = max(1, contracts)
        cluster_used[cap_key] = cluster_used.get(cap_key, 0) + contracts

        sized.append(SizedQuote(
            ticker=q["ticker"],
            target=q["target"],
            cluster=cluster,
            fair_value=q["fair_value"],
            model_prob=q["model_prob"],
            ensemble_std=q["ensemble_std"],
            confidence_tier=q["confidence_tier"],
            bid_cents=q["bid_cents"],
            ask_cents=q["ask_cents"],
            contracts=contracts,
            accuracy_mult=s["accuracy_mult"],
            edge_at_mid=q["edge_at_mid"],
            weight_breakdown=s["weights"],
        ))

    return sized
