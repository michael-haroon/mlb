"""
pregame/trading/ws.py
---------------------
Kalshi WebSocket client for MLB pregame trading.

Maintains:
- Real-time orderbook via orderbook_delta channel
- Trade tape via trade channel
- Market lifecycle events (creation, game start, settlement) via market_lifecycle_v2
- Incremental market discovery via lifecycle `created` events

The lifecycle events drive:
- Discovery: `created` with MLB series prefix → add to tradeable market set
- Game start: `deactivated` on game markets → cancel unfilled quotes
- Delays: `close_date_updated` → invalidate GUMBO schedule cache
- Settlement: `settled` / `determined` → trigger feature refresh + P&L logging
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

from .config import KALSHI_WS_URL, KALSHI_DEMO_WS_URL, LOGS_DIR, TRADEABLE_SERIES
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
                logger.debug(f"Seq gap on {ticker} (sid={sid}): expected {last_seq+1}, got {seq}")
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

    def remove_ticker(self, ticker: str):
        with self._lock:
            self._books.pop(ticker, None)


class KalshiWS:
    """Persistent WebSocket connection to Kalshi for MLB markets.

    Subscribes to orderbook_delta, trade, market_lifecycle_v2, fill,
    user_orders, and market_positions channels.

    Lifecycle events drive game-start, settlement, and discovery callbacks.
    Fill/order/position events drive real-time portfolio tracking without polling.
    """

    def __init__(
        self,
        api_key: str,
        rsa_key_path: str | Path,
        env: str = "prod",
        on_game_start: Optional[Callable[[str], None]] = None,
        on_settle: Optional[Callable[[str, str], None]] = None,
        on_market_created: Optional[Callable[[str, dict], None]] = None,
        on_close_date_updated: Optional[Callable[[str, int], None]] = None,
        on_fill: Optional[Callable[[dict], None]] = None,
        on_order_update: Optional[Callable[[dict], None]] = None,
        on_position_update: Optional[Callable[[dict], None]] = None,
        on_orderbook_delta: Optional[Callable[[str], None]] = None,
    ):
        self._api_key = api_key
        self._private_key = _load_private_key(rsa_key_path)
        self._url = KALSHI_WS_URL if env == "prod" else KALSHI_DEMO_WS_URL
        self._on_game_start = on_game_start
        self._on_settle = on_settle
        self._on_market_created = on_market_created
        self._on_close_date_updated = on_close_date_updated
        self._on_fill = on_fill
        self._on_order_update = on_order_update
        self._on_position_update = on_position_update
        self._on_orderbook_delta = on_orderbook_delta

        self.book = LocalBook()
        self._trades: list[dict] = []
        self._trades_lock = threading.Lock()

        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._msg_id = 0
        self._subscribed_tickers: set[str] = set()
        self._snapshot_pending: set[str] = set()

        # SID tracking for update_subscription calls
        self._orderbook_sid: Optional[int] = None
        self._trade_sid: Optional[int] = None
        self._lifecycle_sid: Optional[int] = None
        self._fill_sid: Optional[int] = None
        self._user_orders_sid: Optional[int] = None
        self._positions_sid: Optional[int] = None

        # Callback for runner to know when subscriptions are ready after reconnect
        self._on_reconnect: Optional[Callable[[], None]] = None

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

    def subscribe_markets_batch(self, tickers: list[str]):
        """Subscribe to orderbook_delta and trade for a batch of markets.

        Uses the market_tickers array parameter for a single WS message per channel
        instead of one message per ticker.
        """
        if not tickers:
            return

        new_tickers = [t for t in tickers if t not in self._subscribed_tickers]
        if not new_tickers:
            return

        self._subscribed_tickers.update(new_tickers)

        if not (self._ws and self._running):
            return

        # If we already have subscription SIDs, use update_subscription to add
        if self._orderbook_sid is not None:
            self._ws.send(json.dumps({
                "id": self._next_id(),
                "cmd": "update_subscription",
                "params": {
                    "sid": self._orderbook_sid,
                    "market_tickers": new_tickers,
                    "action": "add_markets",
                },
            }))
        else:
            self._ws.send(json.dumps({
                "id": self._next_id(),
                "cmd": "subscribe",
                "params": {
                    "channels": ["orderbook_delta"],
                    "market_tickers": new_tickers,
                },
            }))

        if self._trade_sid is not None:
            self._ws.send(json.dumps({
                "id": self._next_id(),
                "cmd": "update_subscription",
                "params": {
                    "sid": self._trade_sid,
                    "market_tickers": new_tickers,
                    "action": "add_markets",
                },
            }))
        else:
            self._ws.send(json.dumps({
                "id": self._next_id(),
                "cmd": "subscribe",
                "params": {
                    "channels": ["trade"],
                    "market_tickers": new_tickers,
                },
            }))

        if len(new_tickers) > 10:
            logger.info(f"Subscribed to {len(new_tickers)} new markets (total: {len(self._subscribed_tickers)})")
        else:
            logger.debug(f"Subscribed to {len(new_tickers)} new markets (total: {len(self._subscribed_tickers)})")

    def unsubscribe_markets_batch(self, tickers: list[str]):
        """Unsubscribe from orderbook_delta and trade for a batch of markets."""
        to_remove = [t for t in tickers if t in self._subscribed_tickers]
        if not to_remove:
            return

        self._subscribed_tickers -= set(to_remove)

        if not (self._ws and self._running):
            return

        if self._orderbook_sid is not None:
            self._ws.send(json.dumps({
                "id": self._next_id(),
                "cmd": "update_subscription",
                "params": {
                    "sid": self._orderbook_sid,
                    "market_tickers": to_remove,
                    "action": "delete_markets",
                },
            }))

        if self._trade_sid is not None:
            self._ws.send(json.dumps({
                "id": self._next_id(),
                "cmd": "update_subscription",
                "params": {
                    "sid": self._trade_sid,
                    "market_tickers": to_remove,
                    "action": "delete_markets",
                },
            }))

        for t in to_remove:
            self.book.remove_ticker(t)

    def subscribe_market(self, ticker: str):
        """Subscribe to a single market (convenience wrapper)."""
        self.subscribe_markets_batch([ticker])

    def unsubscribe_market(self, ticker: str):
        """Unsubscribe from a single market (convenience wrapper)."""
        self.unsubscribe_markets_batch([ticker])

    # ── WebSocket callbacks ──────────────────────────────────────────────────

    def _on_open(self, ws):
        logger.info("WebSocket connected")
        self.book._seqs.clear()
        self._snapshot_pending.clear()
        self._orderbook_sid = None
        self._trade_sid = None
        self._lifecycle_sid = None
        self._fill_sid = None
        self._user_orders_sid = None
        self._positions_sid = None

        # Subscribe to lifecycle first (global, no ticker filter needed)
        self._ws.send(json.dumps({
            "id": self._next_id(),
            "cmd": "subscribe",
            "params": {"channels": ["market_lifecycle_v2"]},
        }))

        # Subscribe to user-specific channels (fills, orders, positions)
        # These are global (no ticker filter) — receive all trading activity
        self._ws.send(json.dumps({
            "id": self._next_id(),
            "cmd": "subscribe",
            "params": {"channels": ["fill"]},
        }))
        self._ws.send(json.dumps({
            "id": self._next_id(),
            "cmd": "subscribe",
            "params": {"channels": ["user_orders"]},
        }))
        self._ws.send(json.dumps({
            "id": self._next_id(),
            "cmd": "subscribe",
            "params": {"channels": ["market_positions"]},
        }))

        # Re-subscribe to all tracked tickers in one batch per channel
        if self._subscribed_tickers:
            tickers_list = list(self._subscribed_tickers)
            self._ws.send(json.dumps({
                "id": self._next_id(),
                "cmd": "subscribe",
                "params": {
                    "channels": ["orderbook_delta"],
                    "market_tickers": tickers_list,
                },
            }))
            self._ws.send(json.dumps({
                "id": self._next_id(),
                "cmd": "subscribe",
                "params": {
                    "channels": ["trade"],
                    "market_tickers": tickers_list,
                },
            }))
            logger.info(f"Re-subscribed {len(tickers_list)} markets after reconnect")

        if self._on_reconnect:
            threading.Thread(target=self._on_reconnect, daemon=True).start()

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
        elif msg_type == "fill":
            self._handle_fill(msg)
        elif msg_type == "user_order":
            self._handle_user_order(msg)
        elif msg_type == "market_position":
            self._handle_market_position(msg)
        elif msg_type == "subscribed":
            self._handle_subscribed(msg)
        elif msg_type == "error":
            err = msg.get("msg", {})
            logger.error(f"WS error: code={err.get('code')} msg={err.get('msg')}")

    def _on_error(self, ws, error):
        logger.error(f"WS error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"WS closed: {close_status_code} {close_msg}")
        if self._running:
            logger.info("Reconnecting in 5s...")
            threading.Thread(target=self._reconnect_loop, daemon=True).start()

    def _reconnect_loop(self):
        time.sleep(5)
        self._connect()

    # ── Message handlers ─────────────────────────────────────────────────────

    def _handle_subscribed(self, msg):
        """Track SIDs returned from subscribe commands for later update_subscription."""
        inner = msg.get("msg", {})
        channel = inner.get("channel", "")
        sid = inner.get("sid")
        if sid is None:
            return
        if channel == "orderbook_delta":
            self._orderbook_sid = sid
        elif channel == "trade":
            self._trade_sid = sid
        elif channel == "market_lifecycle_v2":
            self._lifecycle_sid = sid
        elif channel == "fill":
            self._fill_sid = sid
        elif channel == "user_orders":
            self._user_orders_sid = sid
        elif channel == "market_positions":
            self._positions_sid = sid
        logger.debug(f"Subscribed to {channel}, sid={sid}")

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
            self._snapshot_pending.add(ticker)
            if self._orderbook_sid is not None:
                self._ws.send(json.dumps({
                    "id": self._next_id(),
                    "cmd": "update_subscription",
                    "params": {
                        "sid": self._orderbook_sid,
                        "market_tickers": [ticker],
                        "action": "get_snapshot",
                    },
                }))

        if ok and self._on_orderbook_delta:
            self._on_orderbook_delta(ticker)

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
        if not ticker.startswith("KXMLB"):
            return

        series = ticker.split("-")[0]
        in_tradeable = series in TRADEABLE_SERIES

        # Log at INFO for series we trade, DEBUG for others
        if in_tradeable:
            logger.info(f"[LIFECYCLE] {event_type} → {ticker}")
        else:
            logger.debug(f"[LIFECYCLE] {event_type} → {ticker} (non-tradeable series)")

        if event_type == "created":
            # New market created — check if it's a series we trade
            if in_tradeable and self._on_market_created:
                close_ts = data.get("close_ts")
                metadata = data.get("additional_metadata", {})
                threading.Thread(
                    target=self._on_market_created,
                    args=(ticker, {"close_ts": close_ts, "metadata": metadata}),
                    daemon=True,
                ).start()

        elif event_type == "deactivated":
            # Game markets get `deactivated` when game starts (trading paused)
            if self._on_game_start:
                threading.Thread(
                    target=self._on_game_start, args=(ticker,), daemon=True
                ).start()

        elif event_type == "close_date_updated":
            # Schedule change (delay, postponement) — new close_ts provided
            close_ts = data.get("close_ts")
            if close_ts and self._on_close_date_updated:
                threading.Thread(
                    target=self._on_close_date_updated,
                    args=(ticker, close_ts),
                    daemon=True,
                ).start()

        elif event_type in ("settled", "determined"):
            self._subscribed_tickers.discard(ticker)
            if self._on_settle:
                threading.Thread(
                    target=self._on_settle, args=(ticker, event_type), daemon=True
                ).start()

    # ── User trading activity handlers ─────────────────────────────────────────

    def _handle_fill(self, msg):
        """Process fill notification — our order was (partially) filled."""
        data = msg.get("msg", {})
        ticker = data.get("market_ticker", "")
        if not ticker.startswith("KXMLB"):
            return

        fill = {
            "trade_id": data.get("trade_id", ""),
            "order_id": data.get("order_id", ""),
            "market_ticker": ticker,
            "side": data.get("side", ""),
            "action": data.get("action", ""),
            "is_taker": data.get("is_taker", False),
            "yes_price": float(data.get("yes_price_dollars", "0")),
            "count": float(data.get("count_fp", "0")),
            "fee_cost": float(data.get("fee_cost", "0")),
            "post_position": float(data.get("post_position_fp", "0")),
            "purchased_side": data.get("purchased_side", ""),
            "ts_ms": data.get("ts_ms", 0),
            "client_order_id": data.get("client_order_id", ""),
        }

        logger.info(
            f"[FILL] {fill['action']} {fill['count']:.0f}x {fill['side']} "
            f"@{fill['yes_price']:.3f} on {ticker} "
            f"({'taker' if fill['is_taker'] else 'maker'}, fee=${fill['fee_cost']:.4f})"
        )

        if self._on_fill:
            threading.Thread(target=self._on_fill, args=(fill,), daemon=True).start()

    def _handle_user_order(self, msg):
        """Process order state change — resting, canceled, or executed."""
        data = msg.get("msg", {})
        ticker = data.get("ticker", "")
        if not ticker.startswith("KXMLB"):
            return

        order = {
            "order_id": data.get("order_id", ""),
            "ticker": ticker,
            "status": data.get("status", ""),
            "side": data.get("side", ""),
            "outcome_side": data.get("outcome_side", ""),
            "yes_price": float(data.get("yes_price_dollars", "0")),
            "fill_count": float(data.get("fill_count_fp", "0")),
            "remaining_count": float(data.get("remaining_count_fp", "0")),
            "initial_count": float(data.get("initial_count_fp", "0")),
            "taker_fill_cost": float(data.get("taker_fill_cost_dollars", "0")),
            "maker_fill_cost": float(data.get("maker_fill_cost_dollars", "0")),
            "taker_fees": float(data.get("taker_fees_dollars", "0")),
            "maker_fees": float(data.get("maker_fees_dollars", "0")),
            "client_order_id": data.get("client_order_id", ""),
            "created_ts_ms": data.get("created_ts_ms", 0),
            "last_updated_ts_ms": data.get("last_updated_ts_ms", 0),
        }

        logger.info(
            f"[ORDER] {order['status']} {order['order_id'][:8]}… "
            f"{order['side']} @{order['yes_price']:.4f} on {ticker} "
            f"(filled={order['fill_count']:.0f}, remaining={order['remaining_count']:.0f})"
        )

        if self._on_order_update:
            threading.Thread(target=self._on_order_update, args=(order,), daemon=True).start()

    def _handle_market_position(self, msg):
        """Process position update — net position changed due to fill or settlement."""
        data = msg.get("msg", {})
        ticker = data.get("market_ticker", "")
        if not ticker.startswith("KXMLB"):
            return

        position = {
            "market_ticker": ticker,
            "position": float(data.get("position_fp", "0")),
            "position_cost": float(data.get("position_cost_dollars", "0")),
            "realized_pnl": float(data.get("realized_pnl_dollars", "0")),
            "fees_paid": float(data.get("fees_paid_dollars", "0")),
            "volume": float(data.get("volume_fp", "0")),
        }

        logger.info(
            f"[POSITION] {ticker}: pos={position['position']:.0f} "
            f"cost=${position['position_cost']:.4f} "
            f"realized_pnl=${position['realized_pnl']:.4f}"
        )

        if self._on_position_update:
            threading.Thread(target=self._on_position_update, args=(position,), daemon=True).start()

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

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._running
