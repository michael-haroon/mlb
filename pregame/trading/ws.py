"""
pregame/trading/ws.py
---------------------
Kalshi WebSocket client for MLB pregame trading.

Maintains:
- Real-time orderbook via orderbook_delta channel
- Trade tape via trade channel
- Market lifecycle events (game start, settlement) via market_lifecycle_v2

The lifecycle events drive the hold/exit transition:
- "active" → game started, cancel unfilled quotes, enter position management
- "settled" → trigger feature refresh + P&L logging
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import websocket
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

from .config import KALSHI_WS_URL, KALSHI_DEMO_WS_URL, LOGS_DIR
from .kalshi_client import _load_private_key

logger = logging.getLogger(__name__)

LOGS_DIR.mkdir(exist_ok=True)


class LocalBook:
    """Thread-safe local orderbook maintained from WS snapshots + deltas."""

    def __init__(self):
        self._lock = threading.Lock()
        self._books: dict[str, dict[str, dict[str, float]]] = {}
        self._seqs: dict[int, int] = {}  # sid → last_seq

    def apply_snapshot(self, ticker: str, yes_levels: list, no_levels: list, seq: int, sid: int):
        with self._lock:
            self._books[ticker] = {
                "yes": {p: float(s) for p, s in (yes_levels or [])},
                "no": {p: float(s) for p, s in (no_levels or [])},
            }
            self._seqs[sid] = seq

    def apply_delta(self, ticker: str, side: str, price: str, delta: float, seq: int, sid: int) -> bool:
        """Apply incremental book update. Returns False on sequence gap (need snapshot)."""
        with self._lock:
            last_seq = self._seqs.get(sid, 0)
            if seq <= last_seq:
                return True  # duplicate
            if seq != last_seq + 1:
                logger.warning(f"Seq gap on {ticker} (sid={sid}): expected {last_seq+1}, got {seq}")
                # Advance past the gap so subsequent messages don't all trigger warnings.
                # The book is stale; snapshot will re-sync it.
                self._seqs[sid] = seq
                return False

            self._seqs[sid] = seq
            book = self._books.get(ticker)
            if not book:
                return False

            levels = book[side]
            current = levels.get(price, 0.0)
            new_val = current + delta
            if new_val <= 0:
                levels.pop(price, None)
            else:
                levels[price] = new_val
            return True

    def get_top(self, ticker: str) -> tuple[Optional[int], Optional[int]]:
        """Returns (best_yes_bid_cents, best_yes_ask_cents).

        Best YES bid = highest price someone will buy YES at.
        Best YES ask = 100 - highest NO bid price (equivalence via mutual-No).
        """
        with self._lock:
            book = self._books.get(ticker)
            if not book:
                return None, None

            yes_prices = [float(p) for p in book["yes"].keys() if book["yes"][p] > 0]
            no_prices = [float(p) for p in book["no"].keys() if book["no"][p] > 0]

            best_bid = round(max(yes_prices) * 100) if yes_prices else None
            best_ask = 100 - round(max(no_prices) * 100) if no_prices else None

            # Sanity: bid should be < ask
            if best_bid is not None and best_ask is not None and best_bid >= best_ask:
                return None, None
            return best_bid, best_ask

    def has_ticker(self, ticker: str) -> bool:
        with self._lock:
            return ticker in self._books


class KalshiWS:
    """Persistent WebSocket connection to Kalshi for MLB markets.

    Subscribes to orderbook_delta, trade, and market_lifecycle_v2.
    Calls on_game_start and on_settle callbacks when lifecycle events fire.
    """

    def __init__(
        self,
        api_key: str,
        rsa_key_path: str | Path,
        env: str = "prod",
        on_game_start: Optional[Callable[[str], None]] = None,
        on_settle: Optional[Callable[[str], None]] = None,
    ):
        self._api_key = api_key
        self._private_key = _load_private_key(rsa_key_path)
        self._url = KALSHI_WS_URL if env == "prod" else KALSHI_DEMO_WS_URL
        self._on_game_start = on_game_start
        self._on_settle = on_settle

        self.book = LocalBook()
        self._trades: list[dict] = []
        self._trades_lock = threading.Lock()

        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._msg_id = 0
        self._subscribed_tickers: set[str] = set()
        # Tickers with a snapshot request already in-flight; prevents request storms on seq gaps.
        self._snapshot_pending: set[str] = set()

        self._tape_file = LOGS_DIR / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}_ws_trades.jsonl"

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _auth_headers(self) -> dict:
        ts_ms = int(time.time() * 1000)
        msg = f"{ts_ms}GET/trade-api/ws/v2".encode("utf-8")
        sig = self._private_key.sign(
            msg,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self._api_key,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("utf-8"),
            "KALSHI-ACCESS-TIMESTAMP": str(ts_ms),
        }

    # ── Subscription management ──────────────────────────────────────────────

    def subscribe_market(self, ticker: str):
        """Subscribe to orderbook_delta and trade for a market."""
        self._subscribed_tickers.add(ticker)
        if self._ws and self._running:
            self._send_market_subscribe(ticker)

    def unsubscribe_market(self, ticker: str):
        self._subscribed_tickers.discard(ticker)
        if self._ws and self._running:
            self._ws.send(json.dumps({
                "id": self._next_id(),
                "cmd": "unsubscribe",
                "params": {"channels": ["orderbook_delta", "trade"], "market_ticker": ticker},
            }))

    def _send_market_subscribe(self, ticker: str):
        self._ws.send(json.dumps({
            "id": self._next_id(),
            "cmd": "subscribe",
            "params": {"channels": ["orderbook_delta"], "market_ticker": ticker},
        }))
        self._ws.send(json.dumps({
            "id": self._next_id(),
            "cmd": "subscribe",
            "params": {"channels": ["trade"], "market_ticker": ticker},
        }))

    def _send_lifecycle_subscribe(self):
        self._ws.send(json.dumps({
            "id": self._next_id(),
            "cmd": "subscribe",
            "params": {"channels": ["market_lifecycle_v2"]},
        }))

    # ── WebSocket callbacks ──────────────────────────────────────────────────

    def _on_open(self, ws):
        logger.info("WebSocket connected")
        # Clear stale seq counters — new session means new sequence numbering.
        # Keeping old values would cause every first delta to fire a false seq gap.
        self.book._seqs.clear()
        self._snapshot_pending.clear()
        self._send_lifecycle_subscribe()
        for ticker in self._subscribed_tickers:
            self._send_market_subscribe(ticker)

    def _on_message(self, ws, raw):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type", "")

        if msg_type == "orderbook_snapshot":
            self._handle_snapshot(msg)
        elif msg_type == "orderbook_delta":
            self._handle_delta(msg)
        elif msg_type == "trade":
            self._handle_trade(msg)
        elif msg_type == "market_lifecycle_v2":
            self._handle_lifecycle(msg)
        elif msg_type == "error":
            err = msg.get("msg", {})
            logger.error(f"WS error: code={err.get('code')} msg={err.get('msg')}")

    def _on_error(self, ws, error):
        logger.error(f"WS error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"WS closed: {close_status_code} {close_msg}")
        if self._running:
            logger.info("Reconnecting in 5s...")
            # Spawn reconnect on a new thread — calling _connect() directly here
            # would invoke run_forever() from inside the websocket callback stack,
            # causing unbounded recursion on rapid disconnect/reconnect cycles.
            threading.Thread(target=self._reconnect_loop, daemon=True).start()

    def _reconnect_loop(self):
        time.sleep(5)
        self._connect()

    # ── Message handlers ─────────────────────────────────────────────────────

    def _handle_snapshot(self, msg):
        data = msg.get("msg", {})
        ticker = data.get("market_ticker", "")
        seq = msg.get("seq", 0)
        sid = msg.get("sid", 0)
        self.book.apply_snapshot(
            ticker,
            data.get("yes_dollars_fp", []),
            data.get("no_dollars_fp", []),
            seq, sid,
        )
        self._snapshot_pending.discard(ticker)
        bb, ba = self.book.get_top(ticker)
        logger.debug(f"[BOOK] Snapshot {ticker}: bid={bb} ask={ba}")

    def _handle_delta(self, msg):
        data = msg.get("msg", {})
        ticker = data.get("market_ticker", "")
        seq = msg.get("seq", 0)
        sid = msg.get("sid", 0)
        ok = self.book.apply_delta(
            ticker,
            data.get("side", ""),
            data.get("price_dollars", "0"),
            float(data.get("delta_fp", "0")),
            seq, sid,
        )
        if not ok and ticker not in self._snapshot_pending:
            # Sequence gap: request fresh snapshot (rate-limit: one in-flight per ticker)
            self._snapshot_pending.add(ticker)
            self._ws.send(json.dumps({
                "id": self._next_id(),
                "cmd": "subscribe",
                "params": {
                    "channels": ["orderbook_delta"],
                    "market_ticker": ticker,
                    "update_subscription": {"action": "get_snapshot"},
                },
            }))

    def _handle_trade(self, msg):
        data = msg.get("msg", {})
        trade = {
            "ticker": data.get("market_ticker", ""),
            "yes_price": data.get("yes_price_dollars", ""),
            "no_price": data.get("no_price_dollars", ""),
            "count": data.get("count_fp", ""),
            "taker_side": data.get("taker_outcome_side", ""),
            "ts_ms": data.get("ts_ms", 0),
        }
        with self._trades_lock:
            self._trades.append(trade)
        with open(self._tape_file, "a") as f:
            f.write(json.dumps(trade) + "\n")

    def _handle_lifecycle(self, msg):
        data = msg.get("msg", {})
        event_type = data.get("event_type", "")
        ticker = data.get("market_ticker", "")

        # Only process MLB markets
        if not any(ticker.startswith(s) for s in ("KXMLB",)):
            return

        logger.info(f"[LIFECYCLE] {event_type} → {ticker}")

        if event_type == "active" and self._on_game_start:
            threading.Thread(
                target=self._on_game_start, args=(ticker,), daemon=True
            ).start()
        elif event_type == "settled" and self._on_settle:
            threading.Thread(
                target=self._on_settle, args=(ticker,), daemon=True
            ).start()

    # ── Connection management ────────────────────────────────────────────────

    def _connect(self):
        headers = self._auth_headers()
        header_list = [f"{k}: {v}" for k, v in headers.items()]
        self._ws = websocket.WebSocketApp(
            self._url,
            header=header_list,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        # ping_interval=20 keeps the connection alive; ping_timeout=10 gives Kalshi
        # enough slack under high load (500+ subscriptions) without triggering false
        # disconnects that were observed at ping_timeout=5 during busy game periods.
        self._ws.run_forever(ping_interval=20, ping_timeout=10)

    def start(self):
        """Start WS in background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._connect, daemon=True)
        self._thread.start()
        time.sleep(2)

    def stop(self):
        """Gracefully close."""
        self._running = False
        if self._ws:
            self._ws.close()
        if self._thread:
            self._thread.join(timeout=5)

    def get_book(self, ticker: str) -> tuple[Optional[int], Optional[int]]:
        """Convenience: get current best bid/ask in cents."""
        return self.book.get_top(ticker)

    def get_all_book_tops(self) -> dict[str, tuple[Optional[int], Optional[int]]]:
        """Get all tracked books' top-of-book."""
        return {t: self.book.get_top(t) for t in self._subscribed_tickers}
