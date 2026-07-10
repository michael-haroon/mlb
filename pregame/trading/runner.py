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
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pregame.trading.config import (
    SCAN_INTERVAL_SEC, EXIT_BUFFER_MINUTES, CANCEL_BEFORE_FIRST_PITCH_MIN,
    TAKER_EDGE_THRESHOLD, REPRICE_MIN_TICK_MOVE, MIN_REPRICE_INTERVAL_SEC,
    MAX_REPRICES_PER_ORDER, TRADEABLE_SERIES, LOGS_DIR, DRY_RUN,
)
from pregame.trading.kalshi_client import make_client, make_write_client
from pregame.trading.models import EnsembleStore
from pregame.trading.scanner import generate_quotes, min_edge_for_profit
from pregame.trading.sizing import size_quotes, preload_accuracy_profiles
from pregame.trading.risk import check_limits
from pregame.trading.executor import post_two_sided, execute_taker, cancel_order, _log_decision
from pregame.trading.portfolio import Portfolio, PositionState
from pregame.trading.ws import KalshiWS
from pregame.trading.features import FeatureManager
from pregame.trading.market_map import parse_ticker
from pregame.trading import schedule as gumbo_schedule

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

        # Reprice tracking: {order_id: {"count": int, "last_reprice": float}}
        self._reprice_state: dict[str, dict] = {}

    def start(self) -> None:
        """Initialize all components and start the trading loop."""
        logger.info(f"Starting MLB pregame trader (mode={'DRY' if self._dry_run else 'LIVE'}, "
                    f"bankroll=${self._bankroll:.0f}, env={self._env})")

        # 1. Connect to Kalshi
        if self._dry_run:
            self._client = make_client(self._env)
        else:
            self._client = make_write_client(self._env)
            # Verify netting is enabled — hard invariant
            self._verify_netting()

        # 2. Load features
        self._features = FeatureManager()
        self._features.load()

        # 3. Load model ensembles
        self._ensemble_store = EnsembleStore()
        targets = self._ensemble_store.discover()
        preload_accuracy_profiles(targets)

        # 4. Initialize portfolio
        self._portfolio = Portfolio(client=self._client, dry_run=self._dry_run)
        self._portfolio.refresh()

        # 5. Start WebSocket
        api_key = os.environ.get("KALSHI_READ_KEY", "")
        rsa_path = os.environ.get("KALSHI_READ_RSA_PATH", "")
        if api_key and rsa_path:
            self._ws = KalshiWS(
                api_key=api_key,
                rsa_key_path=rsa_path,
                env=self._env,
                on_game_start=self._handle_game_start,
                on_settle=self._handle_settlement,
            )
            self._ws.start()
            logger.info("WebSocket connected")
        else:
            logger.warning("No WS credentials — running without real-time book updates")

        self._running = True
        logger.info(f"Initialized. {self._portfolio.summary()}")

    def run_once(self) -> None:
        """Execute a single scan cycle."""
        self._portfolio.refresh()
        self._features.check_and_refresh()

        # Discover tradeable games
        markets = self._discover_tradeable_markets()
        if not markets:
            logger.info("No tradeable markets found")
            return

        # Filter out markets whose teams have a pending unprocessed settled game.
        # This prevents pricing a doubleheader game 2 on stale pre-game-1 features.
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

        # Subscribe to discovered markets for real-time book
        if self._ws:
            for m in markets:
                self._ws.subscribe_market(m["ticker"])
            time.sleep(1)  # brief pause for snapshots to arrive

        # Get current book state
        book_tops = {}
        if self._ws:
            book_tops = self._ws.get_all_book_tops()
        else:
            # REST fallback for book data
            for m in markets:
                try:
                    ob = self._client.get_orderbook(m["ticker"], depth=3)
                    book_tops[m["ticker"]] = self._parse_rest_book(ob)
                except Exception:
                    pass

        # Generate quotes
        features = self._features.get_features()
        quotes = generate_quotes(markets, features, self._ensemble_store, book_tops)

        # Size quotes
        inventory = self._compute_cluster_inventory()
        sized = size_quotes(quotes, self._bankroll, existing_inventory=inventory)

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

    def _discover_tradeable_markets(self) -> list[dict]:
        """Find all open MLB markets whose game has not yet started.

        Uses GUMBO gameDate (UTC) for first-pitch time — accurate to the minute.
        The old approach of exp - 3h was a guess and caused markets to be wrongly
        included (games in progress) or excluded (markets with unusual expiry offsets).
        """
        all_markets = []
        for series in TRADEABLE_SERIES:
            try:
                resp = self._client.get_markets(series_ticker=series, status="open", limit=200)
                markets = resp.get("markets", [])
                for m in markets:
                    parsed = parse_ticker(m["ticker"])
                    if parsed is None:
                        continue
                    # Extract the calendar date from the Kalshi game_key (first 7 chars: YYMMMDD)
                    game_key = parsed.game_key
                    date_str = _game_key_to_date(game_key)
                    if date_str is None:
                        all_markets.append(m)  # can't determine date → include conservatively
                        continue
                    if not gumbo_schedule.game_has_started(parsed.away_team, parsed.home_team, date_str):
                        all_markets.append(m)
            except Exception as e:
                logger.warning(f"Market discovery failed for {series}: {e}")

        logger.info(f"Discovered {len(all_markets)} open pre-game markets")
        return all_markets

    # ── Repricing ────────────────────────────────────────────────────────────

    def _reprice_resting_orders(self, book_tops: dict) -> None:
        """Update resting orders to maintain top-of-book position.

        Never crosses the conservative fair value boundary.
        """
        for order in self._portfolio.get_open_orders():
            ticker = order["ticker"]
            # Skip reprice if a snapshot is pending (orderbook state is uncertain)
            if ticker in self._ws._snapshot_pending:
                continue
            bb, ba = book_tops.get(ticker, (None, None))
            if bb is None or ba is None:
                continue

            # Check reprice constraints
            oid = order.get("order_id", "")
            state = self._reprice_state.get(oid, {"count": 0, "last_reprice": 0})
            if state["count"] >= MAX_REPRICES_PER_ORDER:
                continue
            if time.time() - state["last_reprice"] < MIN_REPRICE_INTERVAL_SEC:
                continue

            side = order["side"]
            current_price = order["price_cents"]

            if side == "yes":
                # Our YES bid should be at or near best_bid + 1
                target = min(bb + 1, ba - 1)  # never cross the ask
            else:
                # Our NO bid — best NO bid + 1
                target = current_price + 1  # simplified; full logic needs NO book

            if abs(target - current_price) < REPRICE_MIN_TICK_MOVE:
                continue

            # Execute cancel + repost
            if cancel_order(self._client, oid, self._dry_run):
                self._portfolio.remove_order(oid)
                # Repost at new price (single-sided for reprice)
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
        """Called via WS when a game transitions to active (first pitch)."""
        parsed = parse_ticker(ticker)
        if not parsed:
            return

        game_key = parsed.game_key
        logger.info(f"Game started: {game_key}")

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

    def _handle_settlement(self, ticker: str) -> None:
        """Called via WS when a market settles."""
        logger.info(f"Settled: {ticker}")

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
                logger.warning(f"Unknown settlement result for {ticker}: {result!r}")
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

    def _on_features_refreshed(self, changed: bool) -> None:
        """Callback after async feature refresh."""
        if changed:
            logger.info("Features updated — reloading models")
            self._ensemble_store.reload_all()
            targets = self._ensemble_store.tradeable_targets
            preload_accuracy_profiles(targets)

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
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Total trading bankroll in dollars")
    parser.add_argument("--kelly-override", type=float, default=None,
                        help="Override KELLY_FRACTION (for ramp-up)")
    parser.add_argument("--env", choices=["prod", "demo"], default="prod")
    args = parser.parse_args()

    dry_run = not args.live
    if args.kelly_override is not None:
        from pregame.trading import config
        config.KELLY_FRACTION = args.kelly_override
        # Also update sizing module's import
        from pregame.trading import sizing
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
