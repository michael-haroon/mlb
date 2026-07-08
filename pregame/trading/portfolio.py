"""
pregame/trading/portfolio.py
----------------------------
Position ledger with hold/exit state machine for MLB pregame trading.

Tracks:
- Open positions (filled)
- Resting orders (unfilled quotes)
- Position lifecycle: RESTING → FILLED → HOLDING/EXITING → SETTLED/EXITED

The hold/exit decision is a function of model strength:
- HOLD (strong): ride to settlement — model's edge survives game variance
- EXIT (weak): take profits in-game — model's noise may erase pregame edge
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from .config import LOGS_DIR
from .kalshi_client import KalshiClient

logger = logging.getLogger(__name__)

LOGS_DIR.mkdir(exist_ok=True)


class PositionState(str, Enum):
    RESTING = "resting"      # Quote posted, not yet filled
    FILLED = "filled"        # One or both sides filled pre-game
    HOLDING = "holding"      # Game started, riding to settlement
    EXITING = "exiting"      # Game started, looking to exit for profit
    SETTLED = "settled"      # Game over, P&L realized
    EXITED = "exited"        # Sold before settlement


_STATE_FILE = LOGS_DIR / "portfolio_state.json"


class Portfolio:
    """Thread-safe position and order ledger."""

    def __init__(self, client: Optional[KalshiClient] = None, dry_run: bool = True):
        self._client = client
        self._dry_run = dry_run
        self._lock = threading.Lock()

        # Positions indexed by ticker
        self._positions: dict[str, dict] = {}
        # Resting orders indexed by order_id
        self._orders: dict[str, dict] = {}
        # Daily realized P&L
        self._daily_pnl: float = 0.0

        if dry_run:
            self._load_state()

    def _load_state(self) -> None:
        """Restore dry-run positions and P&L from disk so paper trades survive restarts."""
        if not _STATE_FILE.exists():
            return
        try:
            state = json.loads(_STATE_FILE.read_text())
            self._positions = state.get("positions", {})
            self._daily_pnl = state.get("daily_pnl", 0.0)
            # Resting orders are not restored — they are ephemeral (no longer on Kalshi).
            logger.info(
                f"[POS] Restored dry-run state: {len(self._positions)} positions, "
                f"P&L ${self._daily_pnl:+.2f}"
            )
        except Exception as e:
            logger.warning(f"[POS] Could not restore state: {e}")

    def _save_state(self) -> None:
        """Persist dry-run positions and P&L to disk."""
        try:
            with self._lock:
                state = {
                    "positions": self._positions,
                    "daily_pnl": self._daily_pnl,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                }
            _STATE_FILE.write_text(json.dumps(state, default=str))
        except Exception as e:
            logger.warning(f"[POS] Could not save state: {e}")

    def refresh(self) -> None:
        """Sync state from Kalshi API. No-op in dry-run mode."""
        if self._dry_run:
            return
        self._refresh_positions_from_api()
        self._refresh_orders_from_api()

    def _refresh_positions_from_api(self) -> None:
        try:
            resp = self._client.get_positions()
            api_positions = resp.get("market_positions", [])
            with self._lock:
                refreshed = {}
                for p in api_positions:
                    ticker = p.get("ticker", p.get("market_ticker", ""))
                    position_count = float(p.get("position_fp", p.get("position", 0)))
                    if not ticker or position_count == 0:
                        continue
                    side = "yes" if position_count > 0 else "no"
                    contracts = abs(int(position_count))
                    exposure = float(p.get("market_exposure_dollars", 0))
                    entry_price = exposure / contracts if contracts else 0.0

                    # Preserve existing metadata (model info, state) if we already track this
                    existing = self._positions.get(ticker, {})
                    refreshed[ticker] = {
                        "ticker": ticker,
                        "side": side,
                        "contracts": contracts,
                        "entry_price": entry_price,
                        "state": existing.get("state", PositionState.FILLED),
                        "target": existing.get("target", ""),
                        "confidence_tier": existing.get("confidence_tier", "MEDIUM"),
                        "accuracy_mult": existing.get("accuracy_mult", 1.0),
                        "entry_edge": existing.get("entry_edge", 0.0),
                    }
                self._positions = refreshed
            logger.info(f"[POS] Refreshed: {len(self._positions)} positions")
        except Exception as e:
            logger.warning(f"[POS] API refresh failed: {e}")

    def _refresh_orders_from_api(self) -> None:
        try:
            resp = self._client.get_orders(status="resting")
            api_orders = resp.get("orders", [])
            with self._lock:
                refreshed = {}
                for o in api_orders:
                    ticker = o.get("ticker", "")
                    if not any(ticker.startswith(s) for s in ("KXMLB",)):
                        continue
                    order_id = o.get("order_id", "")
                    side = o.get("side", "")
                    remaining = int(float(o.get("remaining_count_fp", 0)))
                    if side == "yes":
                        price_cents = round(float(o.get("yes_price_dollars", "0")) * 100)
                    else:
                        price_cents = round(float(o.get("no_price_dollars", "0")) * 100)

                    refreshed[order_id] = {
                        "order_id": order_id,
                        "ticker": ticker,
                        "side": side,
                        "price_cents": price_cents,
                        "contracts": remaining,
                    }
                self._orders = refreshed
            logger.info(f"[POS] Refreshed: {len(self._orders)} resting orders")
        except Exception as e:
            logger.warning(f"[POS] Orders refresh failed: {e}")

    # ── Position management ──────────────────────────────────────────────────

    def add_position(
        self,
        ticker: str,
        side: str,
        entry_price: float,
        contracts: int,
        target: str = "",
        confidence_tier: str = "MEDIUM",
        accuracy_mult: float = 1.0,
        entry_edge: float = 0.0,
    ) -> None:
        """Record a filled position with model metadata for hold/exit decision."""
        record = {
            "ticker": ticker,
            "side": side,
            "entry_price": entry_price,
            "contracts": contracts,
            "state": PositionState.FILLED,
            "target": target,
            "confidence_tier": confidence_tier,
            "accuracy_mult": accuracy_mult,
            "entry_edge": entry_edge,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._positions[ticker] = record
        self._log_event("position_add", record)
        if self._dry_run:
            self._save_state()
        logger.info(
            f"{'[DRY] ' if self._dry_run else '[LIVE] '}"
            f"Position: {contracts}x {side} @{entry_price:.0f}c on {ticker} "
            f"(tier={confidence_tier}, acc={accuracy_mult:.2f})"
        )

    def add_order(
        self,
        order_id: str,
        ticker: str,
        side: str,
        price_cents: int,
        contracts: int,
        model_prob: Optional[float] = None,
    ) -> None:
        """Track a resting order."""
        record = {
            "order_id": order_id,
            "ticker": ticker,
            "side": side,
            "price_cents": price_cents,
            "contracts": contracts,
            "model_prob": model_prob,
            "placed_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._orders[order_id] = record

    def remove_order(self, order_id: str) -> None:
        with self._lock:
            self._orders.pop(order_id, None)

    def has_position(self, ticker: str) -> bool:
        with self._lock:
            return ticker in self._positions

    def has_order_on(self, ticker: str) -> bool:
        with self._lock:
            return any(o["ticker"] == ticker for o in self._orders.values())

    # ── Hold/Exit decision ───────────────────────────────────────────────────

    def classify_hold_or_exit(self, ticker: str) -> PositionState:
        """Determine whether a position should be held through settlement or exited.

        HOLD conditions (strong model → low variance → ride it):
          - confidence_tier == HIGH AND accuracy_mult >= 1.0
          - The model's pregame edge is reliable enough to survive in-game noise

        EXIT conditions (weak model → high variance → take profits):
          - confidence_tier == LOW OR accuracy_mult < 0.7
          - The model's edge is marginal; in-game price movement may offer
            better exit than waiting for settlement

        Default: HOLD for MEDIUM confidence (trust the model's calibration).
        """
        with self._lock:
            pos = self._positions.get(ticker)
            if not pos:
                return PositionState.HOLDING

        tier = pos.get("confidence_tier", "MEDIUM")
        acc = pos.get("accuracy_mult", 1.0)
        edge = pos.get("entry_edge", 0.0)

        if tier == "HIGH" and acc >= 1.0:
            new_state = PositionState.HOLDING
        elif tier == "LOW" or acc < 0.7:
            new_state = PositionState.EXITING
        elif edge < 0.02:
            # Marginal edge: better to take profit than risk settlement
            new_state = PositionState.EXITING
        else:
            new_state = PositionState.HOLDING

        with self._lock:
            pos["state"] = new_state
        return new_state

    def record_settlement(self, ticker: str, yes_won: bool) -> Optional[float]:
        """Compute and record realized P&L for a settled position.

        Returns the P&L dollars, or None if no position tracked for this ticker.
        """
        with self._lock:
            pos = self._positions.get(ticker)
            if not pos:
                return None

        side = pos["side"]
        entry_price = pos["entry_price"]
        contracts = pos["contracts"]

        # YES win: YES holder gains (1 - entry), NO holder loses entry
        if yes_won:
            pnl = (1.0 - entry_price) * contracts if side == "yes" else -entry_price * contracts
        else:
            pnl = -entry_price * contracts if side == "yes" else (1.0 - entry_price) * contracts

        with self._lock:
            pos["state"] = PositionState.SETTLED
            pos["settled_pnl"] = pnl
            self._daily_pnl += pnl

        self._log_event("settlement", {
            "ticker": ticker,
            "side": side,
            "contracts": contracts,
            "entry_price": entry_price,
            "yes_won": yes_won,
            "pnl": pnl,
            "daily_pnl": self._daily_pnl,
        })
        if self._dry_run:
            self._save_state()
        logger.info(
            f"{'[DRY] ' if self._dry_run else '[LIVE] '}"
            f"Settled {ticker}: {'YES' if yes_won else 'NO'} won, "
            f"{side} @{entry_price:.2f} × {contracts} → "
            f"P&L ${pnl:+.2f} (day ${self._daily_pnl:+.2f})"
        )
        return pnl

    def record_exit(self, ticker: str, exit_price: float) -> Optional[float]:
        """Record P&L when a position is exited before settlement."""
        with self._lock:
            pos = self._positions.get(ticker)
            if not pos:
                return None

        side = pos["side"]
        entry_price = pos["entry_price"]
        contracts = pos["contracts"]

        # Exit by selling: gain = (exit - entry) for YES, reversed for NO
        if side == "yes":
            pnl = (exit_price - entry_price) * contracts
        else:
            pnl = (entry_price - exit_price) * contracts

        with self._lock:
            pos["state"] = PositionState.EXITED
            pos["exited_pnl"] = pnl
            self._daily_pnl += pnl

        self._log_event("exit", {
            "ticker": ticker,
            "side": side,
            "contracts": contracts,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "daily_pnl": self._daily_pnl,
        })
        if self._dry_run:
            self._save_state()
        logger.info(
            f"{'[DRY] ' if self._dry_run else '[LIVE] '}"
            f"Exited {ticker}: {side} entry={entry_price:.2f} exit={exit_price:.2f} × {contracts} → "
            f"P&L ${pnl:+.2f} (day ${self._daily_pnl:+.2f})"
        )
        return pnl

    def on_game_start(self, game_key: str) -> dict[str, PositionState]:
        """Transition all positions for a game from FILLED → HOLDING/EXITING.

        Returns {ticker: new_state} for all affected positions.
        """
        affected = {}
        with self._lock:
            tickers = [
                t for t, p in self._positions.items()
                if game_key in t and p.get("state") == PositionState.FILLED
            ]
        for ticker in tickers:
            state = self.classify_hold_or_exit(ticker)
            affected[ticker] = state
            self._log_event("game_start_transition", {"ticker": ticker, "new_state": state})
        return affected

    # ── Exposure and P&L ─────────────────────────────────────────────────────

    def total_exposure(self) -> float:
        """Total dollars at risk: filled positions + resting orders."""
        with self._lock:
            pos_exp = sum(
                p["entry_price"] * p["contracts"] for p in self._positions.values()
            )
            ord_exp = sum(
                o["price_cents"] / 100.0 * o["contracts"] for o in self._orders.values()
            )
        return pos_exp + ord_exp

    def position_count(self) -> int:
        with self._lock:
            return len(self._positions)

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    def get_portfolio_state(self) -> dict:
        """Snapshot for risk.check_limits()."""
        with self._lock:
            return {
                "positions": list(self._positions.values()),
                "open_orders": list(self._orders.values()),
                "daily_pnl": self._daily_pnl,
                "position_tickers": set(self._positions.keys()) | {
                    o["ticker"] for o in self._orders.values()
                },
            }

    def get_open_orders(self) -> list[dict]:
        with self._lock:
            return list(self._orders.values())

    def get_positions(self) -> list[dict]:
        with self._lock:
            return list(self._positions.values())

    def summary(self) -> str:
        n = self.position_count()
        exp = self.total_exposure()
        with self._lock:
            n_orders = len(self._orders)
        return f"Positions: {n} | Orders: {n_orders} | Exposure: ${exp:.2f} | Day P&L: ${self._daily_pnl:.2f}"

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log_event(self, event_type: str, data: dict) -> None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        suffix = "_dry" if self._dry_run else ""
        log_file = LOGS_DIR / f"{date_str}_portfolio{suffix}.jsonl"
        entry = {"event": event_type, "ts": datetime.now(timezone.utc).isoformat(), **data}
        with open(log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def stop(self) -> None:
        """Cleanup on shutdown."""
        pass
