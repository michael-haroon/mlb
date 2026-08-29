"""
pregame/trading/sizing.py
-------------------------
Position sizing for MLB pregame market-making.

EBR-proportional allocation:
  1. Equal capital per game: bankroll / n_active_games
  2. Within each game, allocate to lines proportional to EBR
  3. EBR = |mu_hat - line| / MODEL_RMSE

EBR (Error-Budget Ratio) measures how far the model's point estimate sits
from the market threshold, normalized by model error. High EBR = model is
confident the total lands clearly on one side of the line. Backtested on
22,316 OOF games: LOW tercile 55.8% WR → HIGH tercile 72.7% WR.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .config import (
    MAX_CONTRACTS_PER_MARKET,
    CLUSTER_MAX_CONTRACTS, MODEL_ERROR_STD, BANKROLL,
)
from .market_map import parse_ticker

logger = logging.getLogger(__name__)

def compute_ebr(point_estimate: float, line: float, model_error_std: float = MODEL_ERROR_STD) -> float:
    """EBR = |mu_hat - line| / model_error_std.

    Measures how many model-error standard deviations the point estimate
    sits from the line. High EBR = model is confident about line crossing.
    Uses model-error std (not full RMSE) to strip irreducible NegBin noise.
    """
    return abs(point_estimate - line) / model_error_std


def preload_accuracy_profiles(targets: list[str]) -> None:
    """Legacy stub — kept for backwards compatibility with runner.start()."""
    pass


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
    # Metadata
    accuracy_mult: float
    edge_at_mid: float         # edge vs current market midpoint
    weight_breakdown: dict = field(default_factory=dict)


def size_quotes(
    quotes: list[dict],
    bankroll: float,
    existing_inventory: Optional[dict[tuple[str, str], int]] = None,
    n_active_games: int = 15,
    model_error_std: float = MODEL_ERROR_STD,
) -> list[SizedQuote]:
    """Size a batch of raw quotes using EBR-proportional allocation.

    Strategy:
      1. Equal capital per game: bankroll / n_active_games
      2. Within each game, allocate proportional to EBR
      3. EBR = |mu_hat - line| / model_error_std

    Args:
        quotes: list of dicts from scanner.generate_quotes()
        bankroll: total dollars available
        existing_inventory: {(cluster, game_key): contracts_already_deployed}
        n_active_games: number of distinct games with open markets today
        model_error_std: model prediction error std for EBR normalization

    Returns sorted by EBR (highest confidence first).
    """
    if existing_inventory is None:
        existing_inventory = {}

    cluster_used: dict[tuple[str, str], int] = dict(existing_inventory)
    per_game_alloc = bankroll / max(n_active_games, 1)

    # Group quotes by game_key and compute EBR
    game_quotes: dict[str, list[dict]] = defaultdict(list)
    for q in quotes:
        edge = q["edge_at_mid"]
        if edge <= 0:
            continue

        fair = q["fair_value"]
        if fair <= 0 or fair >= 1:
            continue

        parsed = parse_ticker(q["ticker"])
        game_key = parsed.game_key if parsed else q["ticker"]

        point_estimate = q.get("point_estimate")
        line = q.get("line")

        if point_estimate is not None and line is not None:
            ebr = compute_ebr(point_estimate, line, model_error_std)
        else:
            ebr = 0.0

        game_quotes[game_key].append({
            "quote": q,
            "game_key": game_key,
            "ebr": ebr,
        })

    # Compute within-game EBR weights and dollar allocations
    scored = []
    for game_key, gq_list in game_quotes.items():
        sum_ebr = sum(item["ebr"] for item in gq_list)
        for item in gq_list:
            if sum_ebr > 0:
                within_game_weight = item["ebr"] / sum_ebr
            else:
                within_game_weight = 1.0 / len(gq_list)

            dollar_alloc = per_game_alloc * within_game_weight

            fair = item["quote"]["fair_value"]
            contracts_raw = dollar_alloc / fair if fair > 0 else 0

            item["contracts_raw"] = contracts_raw
            item["dollar_alloc"] = dollar_alloc
            item["within_game_weight"] = within_game_weight
            scored.append(item)

    scored.sort(key=lambda x: -x["ebr"])

    # Pre-compute cluster caps per game: distribute proportionally by EBR weight
    # so all lines within a game share the cap fairly.
    game_cluster_totals: dict[tuple[str, str], float] = defaultdict(float)
    for s in scored:
        cap_key = (s["quote"]["cluster"], s["game_key"])
        game_cluster_totals[cap_key] += s["contracts_raw"]

    sized = []
    for s in scored:
        q = s["quote"]
        cluster = q["cluster"]
        game_key = s["game_key"]
        cap_key = (cluster, game_key)

        max_cluster = CLUSTER_MAX_CONTRACTS.get(cluster, 10)
        already_used = cluster_used.get(cap_key, 0)
        remaining = max_cluster - already_used
        if remaining <= 0:
            continue

        # Proportional cap share: this quote's fraction of the game's total raw contracts
        total_raw = game_cluster_totals[cap_key]
        if total_raw > 0:
            share_of_cap = (s["contracts_raw"] / total_raw) * max_cluster
        else:
            share_of_cap = max_cluster

        contracts = int(min(s["contracts_raw"], remaining, share_of_cap, MAX_CONTRACTS_PER_MARKET))
        contracts = max(1, contracts)
        cluster_used[cap_key] = already_used + contracts

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
            accuracy_mult=1.0,
            edge_at_mid=q["edge_at_mid"],
            weight_breakdown={
                "ebr": s["ebr"],
                "per_game_alloc": per_game_alloc,
                "within_game_weight": s["within_game_weight"],
                "dollar_alloc": s["dollar_alloc"],
            },
        ))

    return sized
