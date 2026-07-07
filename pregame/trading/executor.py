"""
pregame/trading/executor.py
---------------------------
Order execution layer for Kalshi MLB markets.

Supports:
- Two-sided quoting (bid YES + ask via NO buy)
- Taker aggression (immediate fill when edge is extreme)
- Cancel and reprice for top-of-book fighting
- Dry-run mode (logs decisions without API calls)

All orders logged to JSONL regardless of mode.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import LOGS_DIR
from .kalshi_client import KalshiClient, RateLimitError

logger = logging.getLogger(__name__)

LOGS_DIR.mkdir(exist_ok=True)


def _log_order(order: dict, mode: str) -> None:
    """Append order to daily JSONL log file."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    suffix = "_dry" if mode == "dry_run" else ""
    log_file = LOGS_DIR / f"{date_str}_orders{suffix}.jsonl"
    order["log_mode"] = mode
    order["log_time"] = datetime.now(timezone.utc).isoformat()
    with open(log_file, "a") as f:
        f.write(json.dumps(order, default=str) + "\n")


def post_two_sided(
    client: KalshiClient,
    ticker: str,
    bid_cents: int,
    ask_cents: int,
    contracts: int,
    dry_run: bool = True,
    metadata: Optional[dict] = None,
) -> dict:
    """Post a two-sided quote: bid (YES buy) + ask (NO buy at 100 - ask).

    The ask side is implemented as buying NO at (100 - ask_cents)c,
    which creates the equivalent of a YES ask at ask_cents.
    Maker fees are $0 — spread captured is pure profit.

    Returns dict with bid/ask order results.
    """
    no_buy_cents = 100 - ask_cents

    order_info = {
        "type": "two_sided",
        "ticker": ticker,
        "bid_cents": bid_cents,
        "ask_cents": ask_cents,
        "no_buy_cents": no_buy_cents,
        "contracts": contracts,
        "metadata": metadata or {},
    }

    if dry_run:
        order_info["status"] = "DRY_RUN"
        logger.info(
            f"[DRY] QUOTE {ticker}: "
            f"BID {contracts}x YES@{bid_cents}c / "
            f"ASK {contracts}x NO@{no_buy_cents}c (=YES ask @{ask_cents}c)"
        )
        _log_order(order_info, "dry_run")
        return order_info

    results = {}

    # Post bid: buy YES at bid_cents
    bid_id = f"bid_{uuid.uuid4().hex[:12]}"
    results["bid"] = _place_with_retry(
        client, ticker, side="yes", price=bid_cents,
        contracts=contracts, client_order_id=bid_id,
    )

    # Post ask: buy NO at (100 - ask_cents) — equivalent to selling YES at ask_cents
    ask_id = f"ask_{uuid.uuid4().hex[:12]}"
    results["ask"] = _place_with_retry(
        client, ticker, side="no", price=no_buy_cents,
        contracts=contracts, client_order_id=ask_id,
    )

    order_info["results"] = results
    order_info["status"] = "SUBMITTED" if any(
        r.get("status") == "SUBMITTED" for r in results.values()
    ) else "ERROR"

    logger.info(
        f"[LIVE] QUOTE {ticker}: "
        f"bid={results['bid'].get('status')} / ask={results['ask'].get('status')}"
    )
    _log_order(order_info, "live")
    return order_info


def execute_taker(
    client: KalshiClient,
    ticker: str,
    side: str,
    contracts: int,
    price_cents: int,
    dry_run: bool = True,
    reason: str = "",
) -> dict:
    """Aggress the book: buy immediately at the current best ask.

    Used only when edge exceeds TAKER_EDGE_THRESHOLD (pays the 7% fee).
    """
    order_info = {
        "type": "taker",
        "ticker": ticker,
        "side": side,
        "price_cents": price_cents,
        "contracts": contracts,
        "reason": reason,
        "client_order_id": f"taker_{uuid.uuid4().hex[:12]}",
    }

    if dry_run:
        order_info["status"] = "DRY_RUN"
        logger.info(f"[DRY] TAKE {contracts}x {side.upper()} @ {price_cents}c on {ticker}")
        _log_order(order_info, "dry_run")
        return order_info

    result = _place_with_retry(
        client, ticker, side=side, price=price_cents,
        contracts=contracts, client_order_id=order_info["client_order_id"],
    )
    order_info.update(result)
    _log_order(order_info, "live" if result.get("status") == "SUBMITTED" else "live_error")
    return order_info


def cancel_order(
    client: KalshiClient,
    order_id: str,
    dry_run: bool = True,
) -> bool:
    """Cancel a resting order."""
    if dry_run:
        logger.info(f"[DRY] CANCEL {order_id}")
        return True
    try:
        client.cancel_order(order_id)
        logger.info(f"[LIVE] CANCELLED {order_id}")
        return True
    except Exception as e:
        logger.error(f"[LIVE] Cancel FAILED {order_id}: {e}")
        return False


def _place_with_retry(
    client: KalshiClient,
    ticker: str,
    side: str,
    price: int,
    contracts: int,
    client_order_id: str,
    max_retries: int = 4,
) -> dict:
    """Place order with exponential backoff on 429 rate limits."""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(0.5 * (2 ** attempt))
            result = client.create_order(
                ticker=ticker,
                side=side,
                action="buy",
                count=contracts,
                price=price,
                order_type="limit",
                client_order_id=client_order_id,
            )
            logger.info(f"[LIVE] {side.upper()} {contracts}x @{price}c on {ticker}")
            return {"status": "SUBMITTED", "response": result, "order_id": client_order_id}
        except RateLimitError:
            if attempt < max_retries - 1:
                logger.warning(f"429 on {ticker}, retry {attempt + 1}/{max_retries}...")
                continue
            return {"status": "RATE_LIMITED", "error": "429 after max retries"}
        except Exception as e:
            logger.error(f"Order failed on {ticker}: {e}")
            return {"status": "ERROR", "error": str(e)}
    return {"status": "ERROR", "error": "exhausted retries"}
