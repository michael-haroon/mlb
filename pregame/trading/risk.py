"""
pregame/trading/risk.py
-----------------------
Risk management gates. Every order must pass check_limits() before execution.

Collateral return (netting_enabled) reduces effective exposure for hedged
positions within mutually exclusive or directional market groups. We compute
exposure as max-possible-loss rather than sum-of-all-costs.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from .config import (
    MAX_POSITION_PCT, MAX_DAILY_EXPOSURE_PCT,
    MAX_CONCURRENT_POSITIONS, DAILY_LOSS_LIMIT_PCT,
    MAX_CONTRACTS_PER_MARKET, MIN_HOURS_TO_FIRST_PITCH,
    PRICE_FLOOR, PRICE_CEILING,
)

logger = logging.getLogger(__name__)


def check_limits(
    ticker: str,
    price: float,
    contracts: int,
    hours_to_first_pitch: float,
    bankroll: float,
    portfolio_state: dict,
    max_exposure_override: Optional[float] = None,
) -> tuple[bool, str]:
    """Gate check for a proposed order.

    Args:
        portfolio_state: dict with keys:
            positions: list of position dicts
            open_orders: list of order dicts
            daily_pnl: float (realized P&L today)
            position_tickers: set of tickers with existing positions/orders

    Returns:
        (allowed: bool, reason: str)
    """
    # Circuit breaker: daily loss limit
    day_pnl = portfolio_state.get("daily_pnl", 0.0)
    loss_limit = bankroll * DAILY_LOSS_LIMIT_PCT / 100.0
    if day_pnl < -loss_limit:
        return False, f"Circuit breaker: daily P&L ${day_pnl:.2f} exceeds -${loss_limit:.2f}"

    # No duplicate positions on same ticker
    if ticker in portfolio_state.get("position_tickers", set()):
        return False, f"Already positioned in {ticker}"

    # Max concurrent positions
    n_positions = len(portfolio_state.get("positions", []))
    n_orders = len(portfolio_state.get("open_orders", []))
    if n_positions + n_orders >= MAX_CONCURRENT_POSITIONS:
        return False, f"Max concurrent positions ({MAX_CONCURRENT_POSITIONS}) reached"

    # Max contracts per market
    if contracts > MAX_CONTRACTS_PER_MARKET:
        return False, f"Contracts ({contracts}) > max ({MAX_CONTRACTS_PER_MARKET})"

    # Single position size as % of bankroll
    position_value = price * contracts
    max_single = bankroll * MAX_POSITION_PCT / 100.0
    if position_value > max_single:
        return False, f"Position ${position_value:.2f} > max ${max_single:.2f} ({MAX_POSITION_PCT}%)"

    # Total exposure (collateral-aware)
    current_exposure = compute_net_exposure(
        portfolio_state.get("positions", []),
        portfolio_state.get("open_orders", []),
    )
    max_exp = max_exposure_override or (bankroll * MAX_DAILY_EXPOSURE_PCT / 100.0)
    if current_exposure + position_value > max_exp:
        return False, (
            f"Exposure ${current_exposure + position_value:.2f} "
            f"> max ${max_exp:.2f} ({MAX_DAILY_EXPOSURE_PCT}%)"
        )

    # Timing gate: only block markets too close to first pitch.
    # No upper-bound cap — we want to trade as early as possible pregame.
    if hours_to_first_pitch < MIN_HOURS_TO_FIRST_PITCH:
        return False, f"Too close to first pitch ({hours_to_first_pitch:.1f}h)"

    # Price range
    if price < PRICE_FLOOR:
        return False, f"Price {price:.2f} below floor {PRICE_FLOOR}"
    if price > PRICE_CEILING:
        return False, f"Price {price:.2f} above ceiling {PRICE_CEILING}"

    return True, "OK"


def compute_net_exposure(
    positions: list[dict],
    open_orders: list[dict],
) -> float:
    """Compute collateral-aware net exposure.

    With netting_enabled, hedged positions within the same event have
    reduced collateral requirements. The actual capital at risk per event
    is: sum_of_costs - guaranteed_minimum_payout.

    For directional groups (spread ladders, total ladders):
      YES on "over 7" + NO on "over 9" → at least one settles YES.
      Max loss = total_cost - $1.

    For mutually exclusive groups (game winner with 2 teams):
      NO on Team A + NO on Team B → at least one settles YES.
      Max loss = total_cost - $1.

    Positions in different events have independent risk (sum normally).
    """
    # Group by event (game_key extracted from ticker)
    by_event: dict[str, list[dict]] = defaultdict(list)

    for pos in positions:
        event_key = _extract_event_key(pos.get("ticker", ""))
        by_event[event_key].append({
            "cost": pos.get("entry_price", 0) * pos.get("contracts", 0),
            "side": pos.get("side", ""),
            "ticker": pos.get("ticker", ""),
        })

    for order in open_orders:
        event_key = _extract_event_key(order.get("ticker", ""))
        by_event[event_key].append({
            "cost": order.get("price_cents", 0) / 100.0 * order.get("contracts", 0),
            "side": order.get("side", ""),
            "ticker": order.get("ticker", ""),
        })

    total_exposure = 0.0
    for event_key, items in by_event.items():
        event_cost = sum(item["cost"] for item in items)

        if len(items) < 2:
            total_exposure += event_cost
            continue

        # Collateral return logic:
        # 1. Mutually exclusive markets (e.g., game winner: NO on both teams)
        #    → at least one NO must settle YES. Guaranteed payout = $1 per pair.
        # 2. Directional markets (e.g., YES "over 7" + NO "over 9")
        #    → at least one must be correct. Guaranteed payout = $1 per pair.
        # 3. Mixed YES+NO on same event also qualifies.
        #
        # Conservative: guaranteed settlements = min(n_distinct_tickers, n_items) - 1
        # because in a group of N mutually exclusive positions, at most 1 can lose all.
        distinct_tickers = len(set(i["ticker"] for i in items))

        if distinct_tickers >= 2:
            # With N positions on distinct strikes in the same event,
            # at least (N-1) cannot all be wrong simultaneously in mutually exclusive markets.
            # Conservative: assume 1 guaranteed payout per pair of positions.
            n_guaranteed = min(distinct_tickers - 1, len(items) - 1)
            guaranteed_payout = n_guaranteed * 1.0
            total_exposure += max(0, event_cost - guaranteed_payout)
        else:
            total_exposure += event_cost

    return total_exposure


def _extract_event_key(ticker: str) -> str:
    """Extract the event identifier (series + game_key) from a ticker.

    "MLBTOTAL-26JUL03NYMLAD-9" → "MLBTOTAL-26JUL03NYMLAD"
    This groups all strikes within the same series+game together.
    """
    parts = ticker.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return ticker
