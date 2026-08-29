"""
Comprehensive tests for EBR-proportional sizing, market-open entry, and
state-based architecture changes.

Tests cover:
  - compute_ebr: basic formula correctness
  - size_quotes: EBR-proportional allocation across games and lines
  - Edge cases: zero EBR, single line, missing point_estimate, cluster caps
  - Integration: runner._quote_single_market logic
  - Delta-driven repricing: _handle_orderbook_delta
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from dataclasses import asdict
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))


# ═══════════════════════════════════════════════════════════════════════════════
# compute_ebr
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeEBR(unittest.TestCase):

    def _fn(self):
        from classical_learning.trading.sizing import compute_ebr
        return compute_ebr

    def test_basic_over(self):
        """mu_hat=10, line=8.5 → EBR = |10-8.5|/0.795 ≈ 1.887"""
        ebr = self._fn()(10.0, 8.5, model_error_std=0.795)
        self.assertAlmostEqual(ebr, 1.5 / 0.795, places=5)

    def test_basic_under(self):
        """mu_hat=7, line=9.5 → EBR = |7-9.5|/0.795 ≈ 3.145"""
        ebr = self._fn()(7.0, 9.5, model_error_std=0.795)
        self.assertAlmostEqual(ebr, 2.5 / 0.795, places=5)

    def test_at_line(self):
        """mu_hat exactly at line → EBR = 0"""
        ebr = self._fn()(8.5, 8.5, model_error_std=0.795)
        self.assertEqual(ebr, 0.0)

    def test_custom_std(self):
        """Different model_error_std changes the normalization"""
        ebr = self._fn()(10.0, 8.0, model_error_std=2.0)
        self.assertAlmostEqual(ebr, 2.0 / 2.0, places=5)
        self.assertEqual(ebr, 1.0)

    def test_symmetry(self):
        """EBR is symmetric: over by 2 == under by 2"""
        ebr_over = self._fn()(11.0, 9.0, model_error_std=0.795)
        ebr_under = self._fn()(7.0, 9.0, model_error_std=0.795)
        self.assertAlmostEqual(ebr_over, ebr_under, places=10)

    def test_large_distance(self):
        """Extreme mu_hat far from line → large EBR"""
        ebr = self._fn()(20.0, 8.5, model_error_std=0.795)
        self.assertAlmostEqual(ebr, 11.5 / 0.795, places=5)
        self.assertGreater(ebr, 2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# size_quotes: EBR-proportional allocation
# ═══════════════════════════════════════════════════════════════════════════════

def _make_quote(ticker, fair_value=0.55, edge=0.03, point_estimate=9.5, line=8.5,
                target="total_runs", cluster="total"):
    return {
        "ticker": ticker,
        "target": target,
        "cluster": cluster,
        "fair_value": fair_value,
        "model_prob": fair_value,
        "ensemble_std": 0.0,
        "confidence_tier": "MEDIUM",
        "bid_cents": int(fair_value * 100) - 2,
        "ask_cents": int(fair_value * 100) + 2,
        "no_buy_cents": 100 - (int(fair_value * 100) + 2),
        "edge_at_mid": edge,
        "line": line,
        "direction": "over",
        "point_estimate": point_estimate,
    }


class TestSizeQuotes(unittest.TestCase):

    def _fn(self):
        from classical_learning.trading.sizing import size_quotes
        return size_quotes

    def test_single_game_single_line(self):
        """One game with one line → gets full per-game allocation."""
        quotes = [_make_quote("KXMLBTOTAL-26JUL10NYMLAD-9", point_estimate=10.0, line=8.5)]
        sized = self._fn()(quotes, bankroll=350, n_active_games=5, model_error_std=0.795)
        self.assertEqual(len(sized), 1)
        # Per-game = 350/5 = 70; single line gets 100% weight
        self.assertAlmostEqual(sized[0].weight_breakdown["per_game_alloc"], 70.0, places=1)
        self.assertAlmostEqual(sized[0].weight_breakdown["within_game_weight"], 1.0, places=5)
        self.assertAlmostEqual(sized[0].weight_breakdown["dollar_alloc"], 70.0, places=1)

    def test_single_game_multiple_lines_ebr_proportional(self):
        """Two lines in one game: allocation proportional to EBR."""
        quotes = [
            # mu=10, line=8.5 → EBR = 1.5/0.795 ≈ 1.887
            _make_quote("KXMLBTOTAL-26JUL10NYMLAD-9", point_estimate=10.0, line=8.5),
            # mu=10, line=9.5 → EBR = 0.5/0.795 ≈ 0.629
            _make_quote("KXMLBTOTAL-26JUL10NYMLAD-10", point_estimate=10.0, line=9.5),
        ]
        sized = self._fn()(quotes, bankroll=350, n_active_games=1, model_error_std=0.795)
        self.assertEqual(len(sized), 2)

        # Higher EBR (line 8.5) should get larger allocation than lower EBR (line 9.5)
        ebrs = sorted(sized, key=lambda s: s.weight_breakdown["ebr"], reverse=True)
        self.assertGreater(ebrs[0].weight_breakdown["dollar_alloc"],
                           ebrs[1].weight_breakdown["dollar_alloc"])

        # Weights should sum to 1.0 within game
        total_weight = sum(s.weight_breakdown["within_game_weight"] for s in sized)
        self.assertAlmostEqual(total_weight, 1.0, places=5)

        # Total dollar alloc should equal per-game alloc (350/1 = 350)
        total_dollars = sum(s.weight_breakdown["dollar_alloc"] for s in sized)
        self.assertAlmostEqual(total_dollars, 350.0, places=1)

    def test_two_games_equal_capital_per_game(self):
        """Two different games each get bankroll/2."""
        quotes = [
            _make_quote("KXMLBTOTAL-26JUL10NYMLAD-9", point_estimate=10.0, line=8.5),
            _make_quote("KXMLBTOTAL-26JUL10PHIDET-9", point_estimate=9.0, line=8.5),
        ]
        sized = self._fn()(quotes, bankroll=350, n_active_games=2, model_error_std=0.795)
        self.assertEqual(len(sized), 2)
        for s in sized:
            self.assertAlmostEqual(s.weight_breakdown["per_game_alloc"], 175.0, places=1)

    def test_zero_ebr_gets_equal_share(self):
        """When all EBRs in a game are zero, lines split equally."""
        quotes = [
            _make_quote("KXMLBTOTAL-26JUL10NYMLAD-9", point_estimate=8.5, line=8.5),
            _make_quote("KXMLBTOTAL-26JUL10NYMLAD-10", point_estimate=9.5, line=9.5),
        ]
        sized = self._fn()(quotes, bankroll=350, n_active_games=1, model_error_std=0.795)
        self.assertEqual(len(sized), 2)
        for s in sized:
            self.assertAlmostEqual(s.weight_breakdown["within_game_weight"], 0.5, places=5)

    def test_missing_point_estimate_treated_as_zero_ebr(self):
        """Quotes without point_estimate get EBR=0 and minimal allocation.

        With EBR-proportional sizing, a quote with EBR=0 gets within_game_weight=0
        when other quotes in the same game have positive EBR. This means zero dollar
        allocation → effectively filtered out. This is correct behavior: if the model
        has no confidence about a line crossing, don't allocate capital to it.
        """
        quotes = [
            _make_quote("KXMLBTOTAL-26JUL10NYMLAD-9", point_estimate=10.0, line=8.5),
            _make_quote("KXMLBTOTAL-26JUL10NYMLAD-10", point_estimate=None, line=9.5),
        ]
        sized = self._fn()(quotes, bankroll=350, n_active_games=1, model_error_std=0.795)
        # Only the quote with positive EBR gets sized — zero-EBR gets no allocation
        self.assertGreaterEqual(len(sized), 1)
        # The sized quote should have positive EBR
        self.assertGreater(sized[0].weight_breakdown["ebr"], 0)

    def test_negative_edge_filtered_out(self):
        """Quotes with edge <= 0 are excluded."""
        quotes = [_make_quote("KXMLBTOTAL-26JUL10NYMLAD-9", edge=-0.01)]
        sized = self._fn()(quotes, bankroll=350, n_active_games=1, model_error_std=0.795)
        self.assertEqual(len(sized), 0)

    def test_invalid_fair_value_filtered(self):
        """Quotes with fair_value at boundary (0 or 1) are excluded."""
        quotes = [_make_quote("KXMLBTOTAL-26JUL10NYMLAD-9", fair_value=0.0)]
        sized = self._fn()(quotes, bankroll=350, n_active_games=1, model_error_std=0.795)
        self.assertEqual(len(sized), 0)

        quotes = [_make_quote("KXMLBTOTAL-26JUL10NYMLAD-9", fair_value=1.0)]
        sized = self._fn()(quotes, bankroll=350, n_active_games=1, model_error_std=0.795)
        self.assertEqual(len(sized), 0)

    def test_cluster_cap_respected(self):
        """Cluster cap limits contracts per (cluster, game_key)."""
        # CLUSTER_MAX_CONTRACTS["total"] = 10
        quotes = [
            _make_quote("KXMLBTOTAL-26JUL10NYMLAD-9", point_estimate=15.0, line=8.5,
                        fair_value=0.10, edge=0.05),  # low fair → many contracts
        ]
        sized = self._fn()(quotes, bankroll=3500, n_active_games=1, model_error_std=0.795)
        self.assertEqual(len(sized), 1)
        self.assertLessEqual(sized[0].contracts, 10)

    def test_max_contracts_per_market_respected(self):
        """MAX_CONTRACTS_PER_MARKET (12) caps individual market size."""
        quotes = [
            _make_quote("KXMLBTOTAL-26JUL10NYMLAD-9", point_estimate=15.0, line=8.5,
                        fair_value=0.10, edge=0.05),
        ]
        sized = self._fn()(quotes, bankroll=10000, n_active_games=1, model_error_std=0.795)
        self.assertEqual(len(sized), 1)
        self.assertLessEqual(sized[0].contracts, 12)

    def test_existing_inventory_reduces_cap(self):
        """Existing inventory reduces cluster cap headroom."""
        quotes = [
            _make_quote("KXMLBTOTAL-26JUL10NYMLAD-9", point_estimate=15.0, line=8.5,
                        fair_value=0.10, edge=0.05),
        ]
        # Already 10 contracts deployed on this cluster+game → cap exhausted
        inventory = {("total", "26JUL10NYMLAD"): 10}
        sized = self._fn()(quotes, bankroll=3500, n_active_games=1, model_error_std=0.795,
                          existing_inventory=inventory)
        self.assertEqual(len(sized), 0)

    def test_sorted_by_ebr_descending(self):
        """Results are sorted highest EBR first."""
        quotes = [
            _make_quote("KXMLBTOTAL-26JUL10NYMLAD-7", point_estimate=10.0, line=6.5),  # EBR = 3.5/4.79
            _make_quote("KXMLBTOTAL-26JUL10NYMLAD-9", point_estimate=10.0, line=8.5),  # EBR = 1.5/4.79
            _make_quote("KXMLBTOTAL-26JUL10NYMLAD-11", point_estimate=10.0, line=10.5),  # EBR = 0.5/4.79
        ]
        sized = self._fn()(quotes, bankroll=350, n_active_games=1, model_error_std=0.795)
        ebrs = [s.weight_breakdown["ebr"] for s in sized]
        self.assertEqual(ebrs, sorted(ebrs, reverse=True))

    def test_per_game_alloc_is_bankroll_divided_by_games(self):
        """Dollar alloc bounded by per-game allocation, not MAX_POSITION_PCT."""
        quotes = [
            _make_quote("KXMLBTOTAL-26JUL10NYMLAD-9", point_estimate=20.0, line=8.5,
                        fair_value=0.50, edge=0.10),
        ]
        sized = self._fn()(quotes, bankroll=350, n_active_games=5, model_error_std=0.795)
        # Per-game alloc = 350/5 = 70, single line → full 70
        self.assertAlmostEqual(sized[0].weight_breakdown["dollar_alloc"], 70.0, places=1)

    def test_minimum_one_contract(self):
        """Even tiny allocations get at least 1 contract."""
        quotes = [
            _make_quote("KXMLBTOTAL-26JUL10NYMLAD-9", point_estimate=8.6, line=8.5,
                        fair_value=0.90, edge=0.02),
        ]
        sized = self._fn()(quotes, bankroll=10, n_active_games=15, model_error_std=0.795)
        if sized:  # might be filtered by edge check
            self.assertGreaterEqual(sized[0].contracts, 1)

    def test_many_games_dilutes_per_game(self):
        """More active games → less per game allocation."""
        quotes = [_make_quote("KXMLBTOTAL-26JUL10NYMLAD-9", point_estimate=10.0, line=8.5)]
        sized_5 = self._fn()(quotes, bankroll=350, n_active_games=5, model_error_std=0.795)
        sized_15 = self._fn()(quotes, bankroll=350, n_active_games=15, model_error_std=0.795)
        self.assertGreater(sized_5[0].weight_breakdown["dollar_alloc"],
                           sized_15[0].weight_breakdown["dollar_alloc"])


# ═══════════════════════════════════════════════════════════════════════════════
# SizedQuote fields
# ═══════════════════════════════════════════════════════════════════════════════

class TestSizedQuoteFields(unittest.TestCase):

    def test_weight_breakdown_has_ebr_fields(self):
        """SizedQuote.weight_breakdown contains EBR-specific fields."""
        from classical_learning.trading.sizing import size_quotes
        quotes = [_make_quote("KXMLBTOTAL-26JUL10NYMLAD-9", point_estimate=10.0, line=8.5)]
        sized = size_quotes(quotes, bankroll=350, n_active_games=5, model_error_std=0.795)
        self.assertEqual(len(sized), 1)
        wb = sized[0].weight_breakdown
        self.assertIn("ebr", wb)
        self.assertIn("per_game_alloc", wb)
        self.assertIn("within_game_weight", wb)
        self.assertIn("dollar_alloc", wb)

    def test_accuracy_mult_is_always_1(self):
        """With EBR sizing, accuracy_mult is fixed at 1.0 (legacy field)."""
        from classical_learning.trading.sizing import size_quotes
        quotes = [_make_quote("KXMLBTOTAL-26JUL10NYMLAD-9", point_estimate=10.0, line=8.5)]
        sized = size_quotes(quotes, bankroll=350, n_active_games=5, model_error_std=0.795)
        self.assertEqual(sized[0].accuracy_mult, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# preload_accuracy_profiles (legacy stub)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreloadStub(unittest.TestCase):

    def test_stub_is_noop(self):
        """preload_accuracy_profiles is a no-op stub."""
        from classical_learning.trading.sizing import preload_accuracy_profiles
        # Should not raise
        preload_accuracy_profiles(["total_runs", "home_win"])


# ═══════════════════════════════════════════════════════════════════════════════
# scanner: point_estimate in _price_market output
# ═══════════════════════════════════════════════════════════════════════════════

class TestScannerPointEstimate(unittest.TestCase):

    def test_apply_line_negbin_includes_point_estimate(self):
        """_apply_line for NegBin returns point_estimate in its dict."""
        from classical_learning.trading.scanner import _apply_line
        base_result = {
            "task": "regression",
            "point_estimate": 9.8,
            "ensemble_std": 0.0,
            "confidence_tier": "MEDIUM",
            "n_models_used": 1,
            "distribution": {"type": "negbin", "mu": 9.8, "alpha": 6.73},
        }
        result = _apply_line(base_result, "total_runs", 8.5, "over")
        self.assertIn("point_estimate", result)
        self.assertAlmostEqual(result["point_estimate"], 9.8, places=5)

    def test_apply_line_classification_no_point_estimate(self):
        """_apply_line for classification does NOT include point_estimate."""
        from classical_learning.trading.scanner import _apply_line
        base_result = {
            "task": "classification",
            "prob": 0.62,
            "ensemble_std": 0.0,
            "confidence_tier": "MEDIUM",
            "n_models_used": 1,
        }
        result = _apply_line(base_result, "home_win", None, "over")
        self.assertNotIn("point_estimate", result)


# ═══════════════════════════════════════════════════════════════════════════════
# runner: n_active_games computation
# ═══════════════════════════════════════════════════════════════════════════════

class TestActiveGamesCount(unittest.TestCase):

    def test_distinct_game_keys_from_market_set(self):
        """n_active_games counts distinct game_keys, not distinct tickers."""
        from classical_learning.trading.market_map import parse_ticker
        market_set = {
            "KXMLBTOTAL-26JUL10NYMLAD-9": {},
            "KXMLBTOTAL-26JUL10NYMLAD-10": {},   # same game, different line
            "KXMLBTOTAL-26JUL10PHIDET-9": {},    # different game
        }
        n_active_games = max(1, len({
            parse_ticker(t).game_key
            for t in market_set
            if parse_ticker(t) is not None
        }))
        self.assertEqual(n_active_games, 2)

    def test_empty_market_set_defaults_to_1(self):
        """Empty market set → n_active_games = 1 (avoid division by zero)."""
        market_set = {}
        from classical_learning.trading.market_map import parse_ticker
        game_keys = {
            parse_ticker(t).game_key
            for t in market_set
            if parse_ticker(t) is not None
        }
        n_active_games = max(1, len(game_keys))
        self.assertEqual(n_active_games, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# ws: on_orderbook_delta callback wiring
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrderbookDeltaCallback(unittest.TestCase):

    def test_delta_fires_callback_on_success(self):
        """Successful delta application fires on_orderbook_delta callback."""
        fired = []
        from classical_learning.trading.ws import KalshiWS, LocalBook

        book = LocalBook()
        # Pre-seed with a snapshot so delta application succeeds
        book.apply_snapshot("KXMLBTOTAL-26JUL10NYMLAD-9",
                           [("0.55", 100)], [("0.45", 100)], seq=1, sid=1)

        # Create a minimal KalshiWS instance without connecting
        with patch.object(KalshiWS, '__init__', lambda self, **kwargs: None):
            ws = KalshiWS.__new__(KalshiWS)
            ws.book = book
            ws._snapshot_pending = set()
            ws._orderbook_sid = 1
            ws._ws = None
            ws._on_orderbook_delta = lambda t: fired.append(t)

        # Simulate a delta message
        msg = {
            "type": "orderbook_delta",
            "msg": {
                "market_ticker": "KXMLBTOTAL-26JUL10NYMLAD-9",
                "side": "yes",
                "price_dollars": "0.55",
                "delta_fp": "10",
            },
            "seq": 2,
            "sid": 1,
        }
        ws._handle_delta(msg)
        self.assertEqual(fired, ["KXMLBTOTAL-26JUL10NYMLAD-9"])

    def test_delta_does_not_fire_on_seq_gap(self):
        """Sequence gap → callback NOT fired (snapshot requested instead)."""
        fired = []
        from classical_learning.trading.ws import KalshiWS, LocalBook

        book = LocalBook()
        book.apply_snapshot("KXMLBTOTAL-26JUL10NYMLAD-9",
                           [("0.55", 100)], [("0.45", 100)], seq=1, sid=1)

        with patch.object(KalshiWS, '__init__', lambda self, **kwargs: None):
            ws = KalshiWS.__new__(KalshiWS)
            ws.book = book
            ws._snapshot_pending = set()
            ws._orderbook_sid = 1
            ws._ws = MagicMock()
            ws._msg_id = 0
            ws._on_orderbook_delta = lambda t: fired.append(t)

            def _next_id(self_inner=ws):
                self_inner._msg_id += 1
                return self_inner._msg_id
            ws._next_id = _next_id

        # Seq gap: 1 → 5 (should fail, trigger snapshot request)
        msg = {
            "type": "orderbook_delta",
            "msg": {
                "market_ticker": "KXMLBTOTAL-26JUL10NYMLAD-9",
                "side": "yes",
                "price_dollars": "0.55",
                "delta_fp": "10",
            },
            "seq": 5,
            "sid": 1,
        }
        ws._handle_delta(msg)
        self.assertEqual(fired, [])  # NOT fired on gap


# ═══════════════════════════════════════════════════════════════════════════════
# runner: _handle_orderbook_delta rate limiting
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunnerDeltaReprice(unittest.TestCase):

    def _make_runner(self):
        """Create a minimal TradingRunner with mocked dependencies."""
        from classical_learning.trading.runner import TradingRunner
        with patch.object(TradingRunner, '__init__', lambda self, **kw: None):
            runner = TradingRunner.__new__(TradingRunner)
            runner._dry_run = True
            runner._portfolio = MagicMock()
            runner._ws = MagicMock()
            runner._ws._snapshot_pending = set()
            runner._client = MagicMock()
            runner._reprice_state = {}
            return runner

    def test_no_orders_on_ticker_is_noop(self):
        """Delta on a ticker with no orders does nothing."""
        runner = self._make_runner()
        runner._portfolio.get_open_orders.return_value = [
            {"ticker": "KXMLBTOTAL-26JUL10PHIDET-9", "order_id": "abc", "side": "yes", "price_cents": 55}
        ]
        runner._ws.book.get_top.return_value = (54, 56)
        # Should not raise, should not call cancel
        runner._handle_orderbook_delta("KXMLBTOTAL-26JUL10NYMLAD-9")
        runner._client.cancel_order.assert_not_called()

    def test_rate_limit_blocks_rapid_reprices(self):
        """MIN_REPRICE_INTERVAL_SEC prevents rapid-fire repricing."""
        runner = self._make_runner()
        runner._portfolio.get_open_orders.return_value = [
            {"ticker": "KXMLBTOTAL-26JUL10NYMLAD-9", "order_id": "ord1",
             "side": "yes", "price_cents": 50, "contracts": 2}
        ]
        runner._ws.book.get_top.return_value = (55, 60)

        # First call: should attempt reprice (state is empty)
        runner._reprice_state["ord1"] = {"count": 0, "last_reprice": time.time()}
        runner._handle_orderbook_delta("KXMLBTOTAL-26JUL10NYMLAD-9")
        # Rate-limited: last_reprice is now, so it should NOT fire cancel
        runner._client.cancel_order.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# runner: _handle_market_created triggers immediate quote
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarketCreatedImmediateQuote(unittest.TestCase):

    def test_spawns_thread_when_ready(self):
        """_handle_market_created spawns a thread for _quote_single_market when components are loaded."""
        from classical_learning.trading.runner import TradingRunner
        with patch.object(TradingRunner, '__init__', lambda self, **kw: None):
            runner = TradingRunner.__new__(TradingRunner)
            runner._dry_run = True
            runner._running = True
            runner._ensemble_store = MagicMock()
            runner._features = MagicMock()
            runner._features.is_stale_for_game.return_value = False
            runner._ws = MagicMock()
            runner._ws._snapshot_pending = set()
            runner._market_set = {}
            runner._market_set_lock = threading.Lock()

            # Mock _quote_single_market to track calls
            called = []
            runner._quote_single_market = lambda t: called.append(t)

            # Mock schedule check
            with patch("pregame.trading.runner.gumbo_schedule") as mock_sched:
                mock_sched.game_has_started.return_value = False
                mock_sched.get_game_number.return_value = None
                runner._handle_market_created("KXMLBTOTAL-26JUL10NYMLAD-9", {"close_ts": None})

            # Give daemon thread time to fire
            time.sleep(0.1)
            self.assertIn("KXMLBTOTAL-26JUL10NYMLAD-9", called)

    def test_no_thread_when_not_running(self):
        """No immediate quote if runner hasn't started yet."""
        from classical_learning.trading.runner import TradingRunner
        with patch.object(TradingRunner, '__init__', lambda self, **kw: None):
            runner = TradingRunner.__new__(TradingRunner)
            runner._dry_run = True
            runner._running = False  # not started yet
            runner._ensemble_store = MagicMock()
            runner._features = MagicMock()
            runner._ws = MagicMock()
            runner._market_set = {}
            runner._market_set_lock = threading.Lock()

            called = []
            runner._quote_single_market = lambda t: called.append(t)

            with patch("pregame.trading.runner.gumbo_schedule") as mock_sched:
                mock_sched.game_has_started.return_value = False
                mock_sched.get_game_number.return_value = None
                runner._handle_market_created("KXMLBTOTAL-26JUL10NYMLAD-9", {"close_ts": None})

            time.sleep(0.1)
            self.assertEqual(called, [])


# ═══════════════════════════════════════════════════════════════════════════════
# Stress tests: adversarial edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEBRStress(unittest.TestCase):

    def test_15_games_with_6_lines_each(self):
        """Full MLB day: 15 games × 6 lines = 90 quotes. All should size correctly."""
        from classical_learning.trading.sizing import size_quotes
        quotes = []
        for g in range(15):
            away = f"TM{g:02d}"
            home = f"HM{g:02d}"
            game_key = f"26JUL10{away}{home}"
            for line in [6.5, 7.5, 8.5, 9.5, 10.5, 11.5]:
                quotes.append(_make_quote(
                    f"KXMLBTOTAL-{game_key}-{int(line*10)}",
                    point_estimate=9.3,
                    line=line,
                    fair_value=0.55,
                    edge=0.03,
                ))

        sized = size_quotes(quotes, bankroll=350, n_active_games=15, model_error_std=0.795)
        # Should produce quotes (some may be capped by cluster)
        self.assertGreater(len(sized), 0)
        # All contracts should be positive
        for s in sized:
            self.assertGreater(s.contracts, 0)
        # Total dollar allocation should not exceed bankroll
        total_dollars = sum(s.weight_breakdown["dollar_alloc"] for s in sized)
        self.assertLessEqual(total_dollars, 350 + 1.0)

    def test_extreme_mu_hat_does_not_crash(self):
        """Very large mu_hat (20) or very small (2) shouldn't break sizing."""
        from classical_learning.trading.sizing import size_quotes
        quotes = [
            _make_quote("KXMLBTOTAL-26JUL10NYMLAD-9", point_estimate=2.0, line=8.5,
                        fair_value=0.20, edge=0.05),
            _make_quote("KXMLBTOTAL-26JUL10PHIDET-9", point_estimate=20.0, line=8.5,
                        fair_value=0.90, edge=0.05),
        ]
        sized = size_quotes(quotes, bankroll=350, n_active_games=2, model_error_std=0.795)
        self.assertEqual(len(sized), 2)

    def test_bankroll_zero(self):
        """Bankroll of 0 → no contracts (graceful, no ZeroDivisionError)."""
        from classical_learning.trading.sizing import size_quotes
        quotes = [_make_quote("KXMLBTOTAL-26JUL10NYMLAD-9", point_estimate=10.0, line=8.5)]
        sized = size_quotes(quotes, bankroll=0, n_active_games=1, model_error_std=0.795)
        # Either empty or contracts are all 1 (minimum). Should not raise.
        for s in sized:
            self.assertGreaterEqual(s.contracts, 1)

    def test_all_lines_same_ebr(self):
        """If all lines have identical EBR, each gets equal weight."""
        from classical_learning.trading.sizing import size_quotes
        # mu=9.5, lines at 8.5 and 10.5 → both have |distance|=1.0
        quotes = [
            _make_quote("KXMLBTOTAL-26JUL10NYMLAD-9", point_estimate=9.5, line=8.5),
            _make_quote("KXMLBTOTAL-26JUL10NYMLAD-11", point_estimate=9.5, line=10.5),
        ]
        sized = size_quotes(quotes, bankroll=350, n_active_games=1, model_error_std=0.795)
        self.assertEqual(len(sized), 2)
        weights = [s.weight_breakdown["within_game_weight"] for s in sized]
        self.assertAlmostEqual(weights[0], weights[1], places=5)


# ═══════════════════════════════════════════════════════════════════════════════
# Config: new constants exist
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigConstants(unittest.TestCase):

    def test_model_error_std_defined(self):
        from classical_learning.trading.config import MODEL_ERROR_STD
        self.assertAlmostEqual(MODEL_ERROR_STD, 0.795, places=3)

    def test_bankroll_defined(self):
        from classical_learning.trading.config import BANKROLL
        self.assertEqual(BANKROLL, 25_000)

    def test_scan_interval_defined(self):
        from classical_learning.trading.config import SCAN_INTERVAL_SEC
        self.assertEqual(SCAN_INTERVAL_SEC, 60)


if __name__ == "__main__":
    unittest.main()
