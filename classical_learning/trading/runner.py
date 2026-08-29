"""
pregame/trading/runner.py
-------------------------
Main trading loop for MLB pregame market-making.

Two-phase operation:
  Phase A (pre-game): Generate two-sided quotes, fight for top-of-book, size by Kelly
  Phase B (in-game): Hold strong positions to settlement, exit weak ones for profit

Usage:
    # Dry-run: log decisions without executing
    conda run -n pred python -m pregame.trading.runner --dry-run --once

    # Live with safety-net Kelly override
    conda run -n pred python -m pregame.trading.runner --live --kelly-override 0.015

    # Continuous loop
    conda run -n pred python -m pregame.trading.runner --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from classical_learning.trading.config import (
    SCAN_INTERVAL_SEC, EXIT_BUFFER_MINUTES, CANCEL_BEFORE_FIRST_PITCH_MIN,
    TAKER_EDGE_THRESHOLD, REPRICE_MIN_TICK_MOVE, MIN_REPRICE_INTERVAL_SEC,
    MAX_REPRICES_PER_ORDER, TRADEABLE_SERIES, LOGS_DIR, DRY_RUN,
    DISCOVERY_INTERVAL_SEC, MODEL_ERROR_STD, BANKROLL,
)
from classical_learning.trading.kalshi_client import make_client, make_write_client
from classical_learning.trading.models import EnsembleStore
from classical_learning.trading.scanner import generate_quotes, min_edge_for_profit
from classical_learning.trading.sizing import size_quotes, preload_accuracy_profiles
from classical_learning.trading.risk import check_limits
from classical_learning.trading.executor import post_two_sided, execute_taker, cancel_order, _log_decision
from classical_learning.trading.portfolio import Portfolio, PositionState
from classical_learning.trading.ws import KalshiWS
from classical_learning.trading.features import FeatureManager
from classical_learning.trading.market_map import parse_ticker
from classical_learning.trading import schedule as gumbo_schedule

LOGS_DIR.mkdir(exist_ok=True)

# ── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "runner.log"),
    ],
)
# File handler at DEBUG for granular diagnostics
debug_handler = logging.FileHandler(LOGS_DIR / "runner_debug.log")
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
logging.getLogger("pregame.trading").addHandler(debug_handler)

logger = logging.getLogger(__name__)


class TradingRunner:
    """Orchestrates the MLB pregame market-making loop."""

    def __init__(self, dry_run: bool = True, env: str = "prod", bankroll: float = 1000.0):
        self._dry_run = dry_run
        self._env = env
        self._bankroll = bankroll
        self._running = False

        # Core components (initialized in start())
        self._client = None
        self._ws: KalshiWS | None = None
        self._portfolio: Portfolio | None = None
        self._features: FeatureManager | None = None
        self._ensemble_store: EnsembleStore | None = None
        self._tradeable_series: list[str] = []  # Populated in start() from available models

        # Reprice tracking: {order_id: {"count": int, "last_reprice": float}}
        self._reprice_state: dict[str, dict] = {}

        # Market set: ticker → market dict. Maintained via WS lifecycle events
        # with hourly REST reconciliation as fallback.
        self._market_set: dict[str, dict] = {}
        self._market_set_lock = threading.Lock()
        self._last_full_discovery: float = 0

    def start(self) -> None:
        """Initialize all components and start the trading loop."""
        logger.info(f"Starting MLB pregame trader (mode={'DRY' if self._dry_run else 'LIVE'}, "
                    f"bankroll=${self._bankroll:.0f}, env={self._env})")

        # 1. Connect to Kalshi
        if self._dry_run:
            self._client = make_client(self._env)
        else:
            self._client = make_write_client(self._env)
            self._verify_netting()

        # Upgrade rate limit to Advanced (300r/300w) — idempotent
        self._upgrade_rate_limit()

        # 2. Load features
        self._features = FeatureManager()
        self._features.load()

        # 3. Load model ensembles and determine tradeable series dynamically
        self._ensemble_store = EnsembleStore()
        targets = self._ensemble_store.discover()
        preload_accuracy_profiles(targets)

        # Validate that feature parquet contains what models expect
        self._ensemble_store.validate_features(self._features.get_features())

        # Build tradeable series list dynamically from available models
        self._tradeable_series = self._get_tradeable_series_from_models()
        if not self._tradeable_series:
            logger.warning("No tradeable series found — no models have Kalshi markets!")
        else:
            logger.info(f"Tradeable series based on available models: {self._tradeable_series}")

        # 4. Initialize portfolio
        self._portfolio = Portfolio(client=self._client, dry_run=self._dry_run)
        self._portfolio.refresh()

        # 5. Start WebSocket with lifecycle-driven discovery callbacks + trading activity
        api_key = os.environ.get("KALSHI_READ_KEY", "")
        rsa_path = os.environ.get("KALSHI_READ_RSA_PATH", "")
        if api_key and rsa_path:
            self._ws = KalshiWS(
                api_key=api_key,
                rsa_key_path=rsa_path,
                env=self._env,
                on_game_start=self._handle_game_start,
                on_settle=self._handle_settlement,
                on_market_created=self._handle_market_created,
                on_close_date_updated=self._handle_close_date_updated,
                on_fill=self._handle_ws_fill,
                on_order_update=self._handle_ws_order,
                on_position_update=self._handle_ws_position,
                on_orderbook_delta=self._handle_orderbook_delta,
            )
            self._ws._on_reconnect = self._on_ws_reconnect
            self._ws.start()
            logger.info("WebSocket connected (orderbook + fill/order/position channels)")
        else:
            logger.warning("No WS credentials — running without real-time book updates")

        # 6. Initial full REST discovery to populate market set
        self._full_discovery()

        # 7. Subscribe to all discovered markets in one batch
        if self._ws:
            with self._market_set_lock:
                tickers = list(self._market_set.keys())
            if tickers:
                logger.info(f"Subscribing to {len(tickers)} markets across {len(self._tradeable_series)} series: {self._tradeable_series}")
                self._ws.subscribe_markets_batch(tickers)
                time.sleep(2)  # allow snapshots to arrive

        self._running = True
        logger.info(f"Initialized. {self._portfolio.summary()}")

    def run_once(self) -> None:
        """Execute a single scan cycle."""
        # Portfolio state is maintained in real-time via WS fill/order/position channels.
        # REST refresh only when WS is disconnected (startup + reconnect handle the rest).
        if not self._ws or not self._ws.is_connected:
            self._portfolio.refresh()
        self._sweep_stale_positions()
        self._features.check_and_refresh()

        # Periodic REST reconciliation (hourly) to catch markets missed during WS gaps
        if time.time() - self._last_full_discovery >= DISCOVERY_INTERVAL_SEC:
            self._full_discovery()

        # Use the maintained market set (populated by initial discovery + WS lifecycle)
        with self._market_set_lock:
            markets = list(self._market_set.values())

        if not markets:
            logger.info("No tradeable markets found")
            return

        logger.debug(f"Active market set: {len(markets)} markets")

        # Filter out markets whose teams have a pending unprocessed settled game.
        ready_markets, blocked_games = [], set()
        for m in markets:
            parsed = parse_ticker(m["ticker"])
            if parsed and self._features.is_stale_for_game(parsed.home_team, parsed.away_team):
                blocked_games.add(parsed.game_key)
            else:
                ready_markets.append(m)
        if blocked_games:
            logger.info(
                f"Holding quotes for {len(blocked_games)} game(s) pending feature rebuild: "
                + ", ".join(sorted(blocked_games))
            )
        markets = ready_markets

        if not markets:
            logger.info("All markets blocked pending feature rebuild")
            return

        # Get current book state from WS (already subscribed)
        book_tops = {}
        if self._ws:
            book_tops = self._ws.get_all_book_tops()
        else:
            for m in markets:
                try:
                    ob = self._client.get_orderbook(m["ticker"], depth=3)
                    book_tops[m["ticker"]] = self._parse_rest_book(ob)
                except Exception:
                    pass

        # Generate quotes (synthetic row construction uses feature_manager for GUMBO context)
        features = self._features.get_features()
        quotes = generate_quotes(
            markets, features, self._ensemble_store, book_tops,
            feature_manager=self._features,
        )

        # Size quotes: EBR-proportional allocation
        inventory = self._compute_cluster_inventory()
        with self._market_set_lock:
            n_active_games = max(1, len({
                parse_ticker(t).game_key
                for t in self._market_set
                if parse_ticker(t) is not None
            }))
        sized = size_quotes(
            quotes, self._bankroll,
            existing_inventory=inventory,
            n_active_games=n_active_games,
            model_error_std=MODEL_ERROR_STD,
        )

        # Execute
        executed = 0
        for sq in sized:
            bb, ba = book_tops.get(sq.ticker, (None, None))
            decision = {
                "ticker": sq.ticker,
                "target": sq.target,
                "fair_value": sq.fair_value,
                "model_prob": sq.model_prob,
                "ensemble_std": sq.ensemble_std,
                "confidence_tier": sq.confidence_tier,
                "bid_cents": sq.bid_cents,
                "ask_cents": sq.ask_cents,
                "book_bid": bb,
                "book_ask": ba,
                "edge_at_mid": sq.edge_at_mid,
                "kelly_raw": sq.weight_breakdown.get("kelly_raw"),
                "accuracy_mult": sq.accuracy_mult,
                "contracts": sq.contracts,
            }

            # Time gate
            hours_to_fp = self._hours_to_first_pitch(sq.ticker)
            if hours_to_fp is None:
                decision["action"] = "SKIP_HOURS"
                decision["action_reason"] = "first_pitch_unknown"
                _log_decision(decision, self._dry_run)
                continue
            if hours_to_fp < EXIT_BUFFER_MINUTES / 60.0:
                decision["action"] = "SKIP_HOURS"
                decision["action_reason"] = (
                    f"hours_to_fp={hours_to_fp:.1f} < {EXIT_BUFFER_MINUTES / 60.0:.2f}"
                )
                _log_decision(decision, self._dry_run)
                continue

            # Risk gate
            state = self._portfolio.get_portfolio_state()
            allowed, risk_reason = check_limits(
                ticker=sq.ticker,
                price=sq.fair_value,
                contracts=sq.contracts,
                hours_to_first_pitch=hours_to_fp,
                bankroll=self._bankroll,
                portfolio_state=state,
            )
            if not allowed:
                logger.debug(f"Risk blocked {sq.ticker}: {risk_reason}")
                decision["action"] = "SKIP_RISK"
                decision["action_reason"] = risk_reason
                _log_decision(decision, self._dry_run)
                continue

            # Check for taker opportunity (book crossed far past our fair)
            taker_edge = self._check_taker_opportunity(sq, bb, ba)
            if taker_edge:
                decision["action"] = f"TAKE_{taker_edge['side'].upper()}"
                decision["action_reason"] = f"edge={taker_edge['edge']:.3f}"

                execute_taker(
                    self._client, sq.ticker,
                    side=taker_edge["side"],
                    contracts=sq.contracts,
                    price_cents=taker_edge["price_cents"],
                    dry_run=self._dry_run,
                    reason=f"edge={taker_edge['edge']:.3f}",
                )
                self._portfolio.add_position(
                    ticker=sq.ticker, side=taker_edge["side"],
                    entry_price=taker_edge["price_cents"] / 100.0,
                    contracts=sq.contracts, target=sq.target,
                    confidence_tier=sq.confidence_tier,
                    accuracy_mult=sq.accuracy_mult,
                    entry_edge=sq.edge_at_mid,
                )
                # Post resting maker on the opposite side
                post_two_sided(
                    self._client, sq.ticker,
                    bid_cents=sq.bid_cents,
                    ask_cents=sq.ask_cents,
                    contracts=sq.contracts,
                    dry_run=self._dry_run,
                    metadata={"target": sq.target, "fair": sq.fair_value,
                              "reason": "opposite_side_after_take"},
                    skip_bid=(taker_edge["side"] == "yes"),
                    skip_ask=(taker_edge["side"] == "no"),
                )
                executed += 1
            else:
                # Post two-sided maker quote
                result = post_two_sided(
                    self._client, sq.ticker,
                    bid_cents=sq.bid_cents,
                    ask_cents=sq.ask_cents,
                    contracts=sq.contracts,
                    dry_run=self._dry_run,
                    metadata={"target": sq.target, "fair": sq.fair_value},
                )
                if result.get("status") in ("DRY_RUN", "SUBMITTED"):
                    decision["action"] = "MAKE"
                    decision["action_reason"] = (
                        f"two_sided bid={sq.bid_cents}c/ask={sq.ask_cents}c"
                    )
                    executed += 1
                else:
                    decision["action"] = "SKIP_EDGE"
                    decision["action_reason"] = (
                        f"post_two_sided status={result.get('status')}"
                    )

            _log_decision(decision, self._dry_run)

        # Reprice existing resting orders to fight for top-of-book
        self._reprice_resting_orders(book_tops)

        # Cancel orders too close to first pitch
        self._cancel_stale_orders()

        logger.info(f"Scan complete: {len(quotes)} quotes, {executed} posted. "
                    f"{self._portfolio.summary()}")

    def run_loop(self) -> None:
        """Continuous trading loop."""
        while self._running:
            try:
                self.run_once()
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Scan error: {e}", exc_info=True)
            time.sleep(SCAN_INTERVAL_SEC)

    def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        if self._ws:
            self._ws.stop()
        if self._portfolio:
            self._portfolio.stop()
        logger.info("Shutdown complete")

    # ── Market discovery ─────────────────────────────────────────────────────

    def _get_tradeable_series_from_models(self) -> list[str]:
        """Map discovered model targets to their Kalshi series.

        Returns only series for which we have trained ensemble models.
        Dynamically expands as new models are added.
        """
        from .market_map import MODEL_TO_SERIES

        series_set = set()
        for target in self._ensemble_store.tradeable_targets:
            series = MODEL_TO_SERIES.get(target)
            if series:
                series_set.add(series)

        # Derived targets (home_runs, away_runs) map to KXMLBTEAMTOTAL
        # and require both total_runs and home_run_diff ensembles
        if "total_runs" in self._ensemble_store.tradeable_targets and \
           "home_run_diff" in self._ensemble_store.tradeable_targets:
            team_total_series = MODEL_TO_SERIES.get("home_runs")
            if team_total_series:
                series_set.add(team_total_series)

        return sorted(series_set)

    def _full_discovery(self) -> None:
        """Full REST discovery: populate/reconcile the market set.

        Called once at startup and then hourly as a fallback to catch markets
        that may have been missed during WS disconnection windows.

        Only discovers markets for which we have trained models.
        """
        if not self._tradeable_series:
            logger.warning("No tradeable series — skipping market discovery")
            return

        discovered = {}
        skipped_g2 = 0
        skipped_no_model = 0
        for series in self._tradeable_series:
            try:
                resp = self._client.get_markets(series_ticker=series, status="open", limit=200)
                markets = resp.get("markets", [])
                for m in markets:
                    parsed = parse_ticker(m["ticker"])
                    if parsed is None:
                        continue
                    game_key = parsed.game_key
                    date_str = _game_key_to_date(game_key)
                    if date_str is None:
                        discovered[m["ticker"]] = m
                        continue
                    # Skip doubleheader game 2
                    game_num = gumbo_schedule.get_game_number(
                        parsed.away_team, parsed.home_team, date_str, parsed.ticker_time
                    )
                    if game_num is not None and game_num > 1:
                        skipped_g2 += 1
                        continue
                    if not gumbo_schedule.game_has_started(parsed.away_team, parsed.home_team, date_str):
                        discovered[m["ticker"]] = m
            except Exception as e:
                logger.warning(f"Market discovery failed for {series}: {e}")

        # Also check for markets we can't trade (no model) and count them at DEBUG level
        for series in TRADEABLE_SERIES:
            if series not in self._tradeable_series:
                try:
                    resp = self._client.get_markets(series_ticker=series, status="open", limit=200)
                    skipped_no_model += len(resp.get("markets", []))
                except Exception:
                    pass

        if skipped_g2:
            logger.info(f"Skipped {skipped_g2} doubleheader game-2 markets")
        if skipped_no_model:
            logger.debug(f"Skipped {skipped_no_model} markets with no trained model")

        # Diff against current set: subscribe new, unsubscribe removed
        with self._market_set_lock:
            current_tickers = set(self._market_set.keys())
            new_tickers = set(discovered.keys())
            added = new_tickers - current_tickers
            removed = current_tickers - new_tickers
            self._market_set = discovered

        if self._ws and added:
            self._ws.subscribe_markets_batch(list(added))
        if self._ws and removed:
            self._ws.unsubscribe_markets_batch(list(removed))

        self._last_full_discovery = time.time()
        logger.info(
            f"Discovery reconciliation: {len(discovered)} markets "
            f"(+{len(added)} new, -{len(removed)} removed)"
        )

    def _handle_market_created(self, ticker: str, info: dict) -> None:
        """WS lifecycle: new market created. Add to market set and immediately quote.

        Only processes markets for which we have trained models.
        """
        parsed = parse_ticker(ticker)
        if parsed is None:
            return

        # Filter by tradeable series (those with models)
        if parsed.series not in self._tradeable_series:
            logger.debug(f"Ignoring market {ticker}: no model for series {parsed.series}")
            return

        date_str = _game_key_to_date(parsed.game_key)
        if date_str:
            # Skip doubleheader game 2
            game_num = gumbo_schedule.get_game_number(
                parsed.away_team, parsed.home_team, date_str, parsed.ticker_time
            )
            if game_num is not None and game_num > 1:
                logger.debug(f"Skipping doubleheader game {game_num}: {ticker}")
                return
            if gumbo_schedule.game_has_started(parsed.away_team, parsed.home_team, date_str):
                return  # game already in progress

        market_dict = {"ticker": ticker, "status": "open"}
        with self._market_set_lock:
            self._market_set[ticker] = market_dict

        if self._ws:
            self._ws.subscribe_markets_batch([ticker])

        logger.info(f"Market created via WS: {ticker}")

        # Immediately quote if inference is available (enter at market open)
        if self._ensemble_store and self._features and self._running:
            threading.Thread(
                target=self._quote_single_market,
                args=(ticker,),
                daemon=True,
            ).start()

    def _quote_single_market(self, ticker: str) -> None:
        """Generate and submit a quote for a single market immediately.

        Called from WS market_created callback for instant entry at market open.
        """
        try:
            parsed = parse_ticker(ticker)
            if parsed is None:
                return

            # Check if features are stale for this game
            if self._features.is_stale_for_game(parsed.home_team, parsed.away_team):
                logger.debug(f"Skipping immediate quote for {ticker}: features stale")
                return

            features = self._features.get_features()
            market_dict = {"ticker": ticker, "status": "open"}

            # Get book state (may be empty if snapshot hasn't arrived yet)
            book_tops = {}
            if self._ws:
                top = self._ws.book.get_top(ticker)
                if top != (None, None):
                    book_tops[ticker] = top

            quotes = generate_quotes([market_dict], features, self._ensemble_store, book_tops)
            if not quotes:
                return

            # Size with current game count
            inventory = self._compute_cluster_inventory()
            with self._market_set_lock:
                n_active_games = max(1, len({
                    parse_ticker(t).game_key
                    for t in self._market_set
                    if parse_ticker(t) is not None
                }))
            sized = size_quotes(
                quotes, self._bankroll,
                existing_inventory=inventory,
                n_active_games=n_active_games,
                model_error_std=MODEL_ERROR_STD,
            )

            for sq in sized:
                # Time gate
                hours_to_fp = self._hours_to_first_pitch(sq.ticker)
                if hours_to_fp is None or hours_to_fp < EXIT_BUFFER_MINUTES / 60.0:
                    continue

                # Risk gate
                state = self._portfolio.get_portfolio_state()
                allowed, risk_reason = check_limits(
                    ticker=sq.ticker,
                    price=sq.fair_value,
                    contracts=sq.contracts,
                    hours_to_first_pitch=hours_to_fp,
                    bankroll=self._bankroll,
                    portfolio_state=state,
                )
                if not allowed:
                    logger.debug(f"Risk blocked immediate quote {sq.ticker}: {risk_reason}")
                    continue

                post_two_sided(
                    self._client, sq.ticker,
                    bid_cents=sq.bid_cents,
                    ask_cents=sq.ask_cents,
                    contracts=sq.contracts,
                    dry_run=self._dry_run,
                    metadata={"target": sq.target, "fair": sq.fair_value,
                              "reason": "market_open_entry"},
                )
                logger.info(f"Immediate quote posted on market open: {sq.ticker} "
                            f"({sq.contracts}x, EBR={sq.weight_breakdown.get('ebr', 0):.3f})")

        except Exception as e:
            logger.warning(f"Immediate quote failed for {ticker}: {e}")

    def _handle_close_date_updated(self, ticker: str, close_ts: int) -> None:
        """WS lifecycle: close date changed (delay/postponement).

        Invalidate GUMBO schedule cache for the affected date so that
        hours_to_first_pitch re-fetches the updated time.
        """
        parsed = parse_ticker(ticker)
        if parsed is None:
            return

        date_str = _game_key_to_date(parsed.game_key)
        if date_str:
            gumbo_schedule.invalidate(date_str)
            logger.info(f"Schedule invalidated for {date_str} due to close_date_updated on {ticker}")

    def _on_ws_reconnect(self) -> None:
        """Called after WS reconnects. Market subscriptions are already re-sent
        in the WS _on_open handler via the batch mechanism.

        REST reconciliation catches anything missed during the disconnect window.
        """
        logger.info("WS reconnected — subscriptions restored, running REST reconciliation")
        self._portfolio.refresh()

    # ── WS trading activity handlers ────────────────────────────────────────

    def _handle_ws_fill(self, fill: dict) -> None:
        """WS fill event: our order was matched."""
        self._portfolio.on_fill(fill)

    def _handle_ws_order(self, order: dict) -> None:
        """WS order state change: resting, canceled, or executed."""
        self._portfolio.on_order_update(order)

    def _handle_ws_position(self, position: dict) -> None:
        """WS position update: authoritative net position from exchange."""
        self._portfolio.on_position_update(position)

    # ── Repricing ────────────────────────────────────────────────────────────

    def _handle_orderbook_delta(self, ticker: str) -> None:
        """Called inline from WS when orderbook changes for a subscribed ticker.

        Triggers immediate reprice check for orders on this ticker.
        """
        if not self._portfolio:
            return
        for order in self._portfolio.get_open_orders():
            if order["ticker"] != ticker:
                continue
            if self._ws and ticker in self._ws._snapshot_pending:
                return
            top = self._ws.book.get_top(ticker) if self._ws else (None, None)
            if top == (None, None):
                return
            self._reprice_single_order(order, {ticker: top})

    def _reprice_single_order(self, order: dict, book_tops: dict) -> None:
        """Reprice a single resting order to maintain top-of-book position."""
        ticker = order["ticker"]
        bb, ba = book_tops.get(ticker, (None, None))
        if bb is None or ba is None:
            return

        oid = order.get("order_id", "")
        state = self._reprice_state.get(oid, {"count": 0, "last_reprice": 0})
        if state["count"] >= MAX_REPRICES_PER_ORDER:
            return
        if time.time() - state["last_reprice"] < MIN_REPRICE_INTERVAL_SEC:
            return

        side = order["side"]
        current_price = order["price_cents"]

        if side == "yes":
            target = min(bb + 1, ba - 1)
        else:
            target = current_price + 1

        if abs(target - current_price) < REPRICE_MIN_TICK_MOVE:
            return

        if cancel_order(self._client, oid, self._dry_run):
            self._portfolio.remove_order(oid)
            from .executor import _place_with_retry
            import uuid
            new_id = f"reprice_{uuid.uuid4().hex[:8]}"
            if not self._dry_run:
                _place_with_retry(
                    self._client, ticker, side=side,
                    price=target, contracts=order["contracts"],
                    client_order_id=new_id,
                )
            self._portfolio.add_order(new_id, ticker, side, target, order["contracts"])
            state["count"] += 1
            state["last_reprice"] = time.time()
            self._reprice_state[oid] = state

    def _reprice_resting_orders(self, book_tops: dict) -> None:
        """Sweep all resting orders for reprice opportunities (safety net)."""
        for order in self._portfolio.get_open_orders():
            ticker = order["ticker"]
            if self._ws and ticker in self._ws._snapshot_pending:
                continue
            if ticker not in book_tops:
                continue
            self._reprice_single_order(order, book_tops)

    def _cancel_stale_orders(self) -> None:
        """Cancel resting orders for games approaching first pitch."""
        for order in self._portfolio.get_open_orders():
            hours = self._hours_to_first_pitch(order["ticker"])
            if hours is not None and hours < CANCEL_BEFORE_FIRST_PITCH_MIN / 60.0:
                cancel_order(self._client, order.get("order_id", ""), self._dry_run)
                self._portfolio.remove_order(order.get("order_id", ""))
                logger.info(f"Cancelled stale order on {order['ticker']} ({hours:.1f}h to first pitch)")

    # ── Lifecycle handlers ───────────────────────────────────────────────────

    def _handle_game_start(self, ticker: str) -> None:
        """Called via WS when a market is deactivated (game started, trading paused)."""
        parsed = parse_ticker(ticker)
        if not parsed:
            return

        game_key = parsed.game_key
        logger.info(f"Game started (deactivated): {game_key} — {ticker}")

        # Remove all markets for this game from the active set
        with self._market_set_lock:
            to_remove = [t for t in self._market_set if game_key in t]
            for t in to_remove:
                del self._market_set[t]

        if self._ws and to_remove:
            self._ws.unsubscribe_markets_batch(to_remove)

        # Cancel all unfilled orders for this game
        for order in self._portfolio.get_open_orders():
            if game_key in order.get("ticker", ""):
                cancel_order(self._client, order.get("order_id", ""), self._dry_run)
                self._portfolio.remove_order(order.get("order_id", ""))

        # Classify positions as HOLD or EXIT
        transitions = self._portfolio.on_game_start(game_key)
        for t, state in transitions.items():
            logger.info(f"  {t} → {state.value}")

    # Series that represent a completed game outcome — trigger feature refresh.
    # Player prop series (KXMLBHR, KXMLBRBI, KXMLBSB, etc.) settle in bulk at
    # game-end but don't produce new game-level features, so we skip them.
    _GAME_SERIES = frozenset({
        "KXMLBGAME", "KXMLBSPREAD", "KXMLBTOTAL", "KXMLBTEAMTOTAL",
        "KXMLBRFI", "KXMLBEXTRAS",
    })

    def _handle_settlement(self, ticker: str, event_type: str = "settled") -> None:
        """Called via WS when a market settles or is determined."""
        logger.info(f"Settled: {ticker}")

        # Remove from active market set
        with self._market_set_lock:
            self._market_set.pop(ticker, None)

        # Only rebuild features when a game-level market settles. Player props
        # (KXMLBHR, KXMLBRBI, KXMLBSB, KXMLBF5*) settle in batches of 50+
        # at game-end but add no new game-level data worth rebuilding for.
        series = ticker.split("-")[0]
        trigger_refresh = series in self._GAME_SERIES

        # Resolve outcome via REST (read-only — safe in dry-run)
        try:
            market_resp = self._client.get_market(ticker)
            market = market_resp.get("market", {})
            result = market.get("result", "")  # "yes" or "no"
            if result in ("yes", "no"):
                yes_won = result == "yes"
                self._portfolio.record_settlement(ticker, yes_won=yes_won)
            else:
                # 'determined' fires before the REST result field propagates; 'settled'
                # always follows with the actual result, so this is not a true error.
                log_fn = logger.debug if event_type == "determined" else logger.warning
                log_fn(f"Unknown settlement result for {ticker}: {result!r}")
        except Exception as e:
            logger.warning(f"Could not fetch settlement result for {ticker}: {e}")

        if trigger_refresh:
            parsed = parse_ticker(ticker)
            if parsed:
                # Mark both teams stale before the async rebuild starts so that
                # is_stale_for_game() blocks quoting immediately, not just once
                # the rebuild lock is acquired.
                self._features.mark_teams_pending(parsed.home_team, parsed.away_team)
            self._features.refresh_async(callback=self._on_features_refreshed)

    # ── Stale position recovery ────────────────────────────────────────────────

    _STALE_POSITION_HOURS = 6  # positions older than this get polled via REST

    def _sweep_stale_positions(self) -> None:
        """Settle positions whose markets are determined/settled but we missed the WS event.

        WS reconnections can cause missed 'settled' lifecycle events, leaving
        positions in 'filled' state indefinitely. This polls REST as a fallback.
        Runs once per scan but only queries positions old enough to have settled.
        """
        now = datetime.now(timezone.utc)
        stale = []
        for pos in self._portfolio.get_positions():
            if pos.get("state") not in (PositionState.FILLED, "filled"):
                continue
            opened_at = pos.get("opened_at")
            if not opened_at:
                stale.append(pos)
                continue
            try:
                opened = datetime.fromisoformat(opened_at)
                age_hours = (now - opened).total_seconds() / 3600.0
                if age_hours >= self._STALE_POSITION_HOURS:
                    stale.append(pos)
            except (ValueError, TypeError):
                stale.append(pos)

        if not stale:
            return

        settled_count = 0
        for pos in stale:
            ticker = pos["ticker"]
            try:
                market_resp = self._client.get_market(ticker)
                market = market_resp.get("market", {})
                result = market.get("result", "")
                status = market.get("status", "")
                if result in ("yes", "no"):
                    yes_won = result == "yes"
                    pnl = self._portfolio.record_settlement(ticker, yes_won=yes_won)
                    logger.info(f"Sweep-settled stale position {ticker}: "
                                f"{'YES' if yes_won else 'NO'} won, P&L ${pnl:+.2f}")
                    settled_count += 1
                elif status in ("settled", "finalized"):
                    logger.warning(f"Market {ticker} is {status} but result={result!r} — cannot settle")
            except Exception as e:
                logger.debug(f"Could not poll {ticker} for sweep: {e}")

        if settled_count:
            logger.info(f"Stale sweep settled {settled_count}/{len(stale)} positions")
            if self._dry_run:
                self._portfolio._save_state()

    def _on_features_refreshed(self, changed: bool) -> None:
        """Callback after async feature refresh."""
        if changed:
            logger.info("Features updated — reloading models")
            self._ensemble_store.reload_all()
            targets = self._ensemble_store.tradeable_targets
            preload_accuracy_profiles(targets)

            # Re-validate feature/model alignment after rebuild
            try:
                self._ensemble_store.validate_features(self._features.get_features())
            except RuntimeError as e:
                logger.error(f"Post-rebuild validation failed — halting: {e}")
                self._running = False
                return

            # Rebuild tradeable series list (may have changed if models were added/removed)
            old_series = set(self._tradeable_series)
            self._tradeable_series = self._get_tradeable_series_from_models()
            new_series = set(self._tradeable_series)

            if new_series != old_series:
                added = new_series - old_series
                removed = old_series - new_series
                logger.info(f"Tradeable series updated: +{added} -{removed}")
                # Trigger a full discovery to subscribe to new series
                self._full_discovery()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _verify_netting(self) -> None:
        """Hard invariant: refuse to trade if netting (collateral return) is not confirmed."""
        try:
            account = self._client.get_account()
            # The exact field name may vary; check common variants
            netting = account.get("netting_enabled", account.get("collateral_return_enabled"))
            if netting is False:
                raise RuntimeError(
                    "CRITICAL: netting_enabled is False. Enable collateral return in account "
                    "settings before trading. Capital efficiency requires it."
                )
            logger.info(f"Netting/collateral return confirmed: {netting}")
        except KeyError:
            logger.warning("Could not verify netting status — proceed with caution")

    def _upgrade_rate_limit(self) -> None:
        """Upgrade API rate limit to Advanced tier (300r/300w). Idempotent."""
        try:
            self._client.upgrade_rate_limit()
            logger.info("API rate limit upgraded to Advanced (300r/300w)")
        except Exception as e:
            logger.warning(f"Rate limit upgrade failed (may already be at Advanced+): {e}")

    def _hours_to_first_pitch(self, ticker: str) -> float | None:
        """Return hours until first pitch using GUMBO gameDate (UTC).

        No REST call per ticker — uses the shared GUMBO schedule cache.
        Returns negative values for games already in progress (caller decides action).
        """
        parsed = parse_ticker(ticker)
        if parsed is None:
            return None
        date_str = _game_key_to_date(parsed.game_key)
        if date_str is None:
            return None
        return gumbo_schedule.hours_to_first_pitch(parsed.away_team, parsed.home_team, date_str)

    def _check_taker_opportunity(
        self, sq, best_bid: int | None, best_ask: int | None,
    ) -> dict | None:
        """Check if the book is so mispriced that taker aggression is warranted.

        Only aggress if edge exceeds TAKER_EDGE_THRESHOLD × taker breakeven.
        """
        if best_ask is None or best_bid is None:
            return None

        fair_cents = int(round(sq.fair_value * 100))

        # YES side: if best ask is far below our fair → buy YES immediately
        if best_ask < fair_cents:
            edge = (fair_cents - best_ask) / 100.0
            taker_breakeven = min_edge_for_profit(best_ask / 100.0, maker=False)
            if edge >= taker_breakeven * TAKER_EDGE_THRESHOLD:
                return {"side": "yes", "price_cents": best_ask, "edge": edge}

        # NO side: if best bid is far above our fair → buy NO (equiv. sell YES)
        if best_bid > fair_cents:
            edge = (best_bid - fair_cents) / 100.0
            no_price = 100 - best_bid
            taker_breakeven = min_edge_for_profit(no_price / 100.0, maker=False)
            if edge >= taker_breakeven * TAKER_EDGE_THRESHOLD:
                return {"side": "no", "price_cents": no_price, "edge": edge}

        return None

    def _compute_cluster_inventory(self) -> dict[tuple[str, str], int]:
        """Sum current positions by (cluster, game_key) for per-game sizing caps."""
        from .market_map import classify_cluster
        inventory: dict[tuple[str, str], int] = {}
        for pos in self._portfolio.get_positions():
            parsed = parse_ticker(pos.get("ticker", ""))
            if parsed:
                cluster = classify_cluster(parsed.series)
                key = (cluster, parsed.game_key)
                inventory[key] = inventory.get(key, 0) + pos.get("contracts", 0)
        return inventory

    def _parse_rest_book(self, ob: dict) -> tuple[int | None, int | None]:
        """Parse REST orderbook response into (best_bid_cents, best_ask_cents)."""
        try:
            yes_bids = ob.get("orderbook", {}).get("yes", [])
            no_bids = ob.get("orderbook", {}).get("no", [])
            best_bid = round(float(yes_bids[0][0]) * 100) if yes_bids else None
            best_ask = 100 - round(float(no_bids[0][0]) * 100) if no_bids else None
            return best_bid, best_ask
        except (IndexError, ValueError, TypeError):
            return None, None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _game_key_to_date(game_key: str) -> str | None:
    """Convert a Kalshi game_key (e.g. '26JUL101840PHIDET') to 'YYYY-MM-DD'.

    Game keys start with YYMMMDD (7 chars): '26JUL10' → 2026-07-10.
    The date is the local calendar date of the game, which matches GUMBO's
    officialDate field.  West Coast night games may have a UTC gameDate that
    falls on the following calendar day — GUMBO handles this correctly when
    queried by officialDate, so we pass the encoded date as-is.
    """
    try:
        from datetime import datetime as _dt
        prefix = game_key[:7]  # e.g. "26JUL10"
        dt = _dt.strptime(prefix, "%y%b%d")
        return dt.strftime("%Y-%m-%d")
    except (ValueError, IndexError):
        return None


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLB pregame market-making trader")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Log decisions without executing (default)")
    parser.add_argument("--live", action="store_true",
                        help="Execute real orders on Kalshi")
    parser.add_argument("--once", action="store_true",
                        help="Run a single scan then exit")
    parser.add_argument("--bankroll", type=float, default=BANKROLL,
                        help="Total trading bankroll in dollars")
    parser.add_argument("--kelly-override", type=float, default=None,
                        help="Override KELLY_FRACTION (for ramp-up)")
    parser.add_argument("--env", choices=["prod", "demo"], default="prod")
    args = parser.parse_args()

    dry_run = not args.live
    if args.kelly_override is not None:
        from classical_learning.trading import config
        config.KELLY_FRACTION = args.kelly_override
        # Also update sizing module's import
        from classical_learning.trading import sizing
        sizing.KELLY_FRACTION = args.kelly_override
        logger.info(f"Kelly fraction overridden to {args.kelly_override}")

    runner = TradingRunner(dry_run=dry_run, env=args.env, bankroll=args.bankroll)

    # Graceful shutdown on SIGTERM/SIGINT
    def _shutdown(signum, frame):
        logger.info(f"Signal {signum} received, shutting down...")
        runner.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    runner.start()

    if args.once:
        runner.run_once()
    else:
        runner.run_loop()

    runner.stop()


if __name__ == "__main__":
    main()
