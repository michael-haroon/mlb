"""
Tests for WS-driven portfolio tracking:
  - ws.py: _handle_fill, _handle_user_order, _handle_market_position message parsing
  - portfolio.py: on_fill, on_order_update, on_position_update state transitions
  - kalshi_client.py: upgrade_rate_limit endpoint
"""
from __future__ import annotations

import json
import threading
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


# ── WS message parsing ──────────────────────────────────────────────────────

class TestWSFillHandler(unittest.TestCase):
    """Test that _handle_fill correctly parses fill messages and invokes callback."""

    def _make_ws(self, on_fill=None):
        from pregame.trading.ws import KalshiWS
        with patch("pregame.trading.ws._load_private_key", return_value=MagicMock()):
            ws = KalshiWS(
                api_key="test",
                rsa_key_path="/dev/null",
                on_fill=on_fill,
            )
        return ws

    def test_parses_fill_fields(self):
        received = []
        ws = self._make_ws(on_fill=lambda f: received.append(f))

        msg = {
            "type": "fill",
            "sid": 13,
            "msg": {
                "trade_id": "d91bc706-ee49-470d-82d8-11418bda6fed",
                "order_id": "ee587a1c-8b87-4dcf-b721-9f6f790619fa",
                "market_ticker": "KXMLBGAME-26JUL211907TBTOR-TB",
                "is_taker": True,
                "side": "yes",
                "action": "buy",
                "yes_price_dollars": "0.750",
                "count_fp": "5.00",
                "fee_cost": "0.0100",
                "post_position_fp": "10.00",
                "purchased_side": "yes",
                "ts_ms": 1671899397000,
                "client_order_id": "my-order-1",
            },
        }

        ws._handle_fill(msg)
        # Callback runs in a thread — give it a moment
        import time
        time.sleep(0.1)

        self.assertEqual(len(received), 1)
        fill = received[0]
        self.assertEqual(fill["market_ticker"], "KXMLBGAME-26JUL211907TBTOR-TB")
        self.assertEqual(fill["order_id"], "ee587a1c-8b87-4dcf-b721-9f6f790619fa")
        self.assertEqual(fill["side"], "yes")
        self.assertEqual(fill["action"], "buy")
        self.assertTrue(fill["is_taker"])
        self.assertAlmostEqual(fill["yes_price"], 0.75)
        self.assertAlmostEqual(fill["count"], 5.0)
        self.assertAlmostEqual(fill["fee_cost"], 0.01)
        self.assertEqual(fill["purchased_side"], "yes")

    def test_ignores_non_mlb_tickers(self):
        received = []
        ws = self._make_ws(on_fill=lambda f: received.append(f))

        msg = {
            "type": "fill",
            "sid": 13,
            "msg": {
                "market_ticker": "HIGHNY-22DEC23-B53.5",
                "order_id": "abc",
                "side": "yes",
                "action": "buy",
                "yes_price_dollars": "0.500",
                "count_fp": "1.00",
                "fee_cost": "0",
                "post_position_fp": "1.00",
                "purchased_side": "yes",
                "ts_ms": 0,
            },
        }
        ws._handle_fill(msg)
        import time
        time.sleep(0.1)
        self.assertEqual(len(received), 0)


class TestWSUserOrderHandler(unittest.TestCase):
    """Test _handle_user_order parsing."""

    def _make_ws(self, on_order_update=None):
        from pregame.trading.ws import KalshiWS
        with patch("pregame.trading.ws._load_private_key", return_value=MagicMock()):
            ws = KalshiWS(
                api_key="test",
                rsa_key_path="/dev/null",
                on_order_update=on_order_update,
            )
        return ws

    def test_parses_resting_order(self):
        received = []
        ws = self._make_ws(on_order_update=lambda o: received.append(o))

        msg = {
            "type": "user_order",
            "sid": 22,
            "msg": {
                "order_id": "ee587a1c-8b87-4dcf-b721-9f6f790619fa",
                "user_id": "user123",
                "ticker": "KXMLBTOTAL-26JUL211907TBTOR-O8",
                "status": "resting",
                "side": "yes",
                "outcome_side": "yes",
                "yes_price_dollars": "0.4200",
                "fill_count_fp": "0.00",
                "remaining_count_fp": "10.00",
                "initial_count_fp": "10.00",
                "taker_fill_cost_dollars": "0.0000",
                "maker_fill_cost_dollars": "0.0000",
                "taker_fees_dollars": "0.0000",
                "maker_fees_dollars": "0.0000",
                "client_order_id": "scan_abc123",
                "created_ts_ms": 1733047200000,
            },
        }

        ws._handle_user_order(msg)
        import time
        time.sleep(0.1)

        self.assertEqual(len(received), 1)
        order = received[0]
        self.assertEqual(order["status"], "resting")
        self.assertEqual(order["ticker"], "KXMLBTOTAL-26JUL211907TBTOR-O8")
        self.assertAlmostEqual(order["yes_price"], 0.42)
        self.assertAlmostEqual(order["remaining_count"], 10.0)
        self.assertAlmostEqual(order["fill_count"], 0.0)

    def test_parses_canceled_order(self):
        received = []
        ws = self._make_ws(on_order_update=lambda o: received.append(o))

        msg = {
            "type": "user_order",
            "sid": 22,
            "msg": {
                "order_id": "cancel-me-123",
                "ticker": "KXMLBGAME-26JUL211907TBTOR-TB",
                "status": "canceled",
                "side": "no",
                "outcome_side": "no",
                "yes_price_dollars": "0.6000",
                "fill_count_fp": "0.00",
                "remaining_count_fp": "0.00",
                "initial_count_fp": "5.00",
                "taker_fill_cost_dollars": "0.0000",
                "maker_fill_cost_dollars": "0.0000",
                "taker_fees_dollars": "0.0000",
                "maker_fees_dollars": "0.0000",
                "client_order_id": "",
                "created_ts_ms": 1733047200000,
            },
        }

        ws._handle_user_order(msg)
        import time
        time.sleep(0.1)

        self.assertEqual(received[0]["status"], "canceled")

    def test_ignores_non_mlb(self):
        received = []
        ws = self._make_ws(on_order_update=lambda o: received.append(o))

        msg = {
            "type": "user_order",
            "sid": 22,
            "msg": {
                "order_id": "xyz",
                "ticker": "FED-23DEC-T3.00",
                "status": "resting",
                "side": "yes",
                "yes_price_dollars": "0.35",
                "fill_count_fp": "0.00",
                "remaining_count_fp": "10.00",
                "initial_count_fp": "10.00",
                "taker_fill_cost_dollars": "0",
                "maker_fill_cost_dollars": "0",
                "taker_fees_dollars": "0",
                "maker_fees_dollars": "0",
                "client_order_id": "",
                "created_ts_ms": 0,
            },
        }
        ws._handle_user_order(msg)
        import time
        time.sleep(0.1)
        self.assertEqual(len(received), 0)


class TestWSMarketPositionHandler(unittest.TestCase):
    """Test _handle_market_position parsing."""

    def _make_ws(self, on_position_update=None):
        from pregame.trading.ws import KalshiWS
        with patch("pregame.trading.ws._load_private_key", return_value=MagicMock()):
            ws = KalshiWS(
                api_key="test",
                rsa_key_path="/dev/null",
                on_position_update=on_position_update,
            )
        return ws

    def test_parses_position_update(self):
        received = []
        ws = self._make_ws(on_position_update=lambda p: received.append(p))

        msg = {
            "type": "market_position",
            "sid": 14,
            "msg": {
                "user_id": "user123",
                "market_ticker": "KXMLBSPREAD-26JUL211907TBTOR-TB5",
                "position_fp": "8.00",
                "position_cost_dollars": "3.2000",
                "realized_pnl_dollars": "0.0000",
                "fees_paid_dollars": "0.0000",
                "position_fee_cost_dollars": "0.0000",
                "volume_fp": "8.00",
            },
        }

        ws._handle_market_position(msg)
        import time
        time.sleep(0.1)

        self.assertEqual(len(received), 1)
        pos = received[0]
        self.assertEqual(pos["market_ticker"], "KXMLBSPREAD-26JUL211907TBTOR-TB5")
        self.assertAlmostEqual(pos["position"], 8.0)
        self.assertAlmostEqual(pos["position_cost"], 3.2)
        self.assertAlmostEqual(pos["realized_pnl"], 0.0)

    def test_zero_position_means_closed(self):
        received = []
        ws = self._make_ws(on_position_update=lambda p: received.append(p))

        msg = {
            "type": "market_position",
            "sid": 14,
            "msg": {
                "user_id": "user123",
                "market_ticker": "KXMLBGAME-26JUL211907TBTOR-TB",
                "position_fp": "0.00",
                "position_cost_dollars": "0.0000",
                "realized_pnl_dollars": "2.5000",
                "fees_paid_dollars": "0.1000",
                "volume_fp": "10.00",
            },
        }

        ws._handle_market_position(msg)
        import time
        time.sleep(0.1)

        self.assertEqual(received[0]["position"], 0.0)
        self.assertAlmostEqual(received[0]["realized_pnl"], 2.5)


# ── Portfolio WS-driven updates ──────────────────────────────────────────────

class TestPortfolioOnFill(unittest.TestCase):
    """Test Portfolio.on_fill() state transitions."""

    def _make_portfolio(self):
        from pregame.trading.portfolio import Portfolio
        p = Portfolio(client=None, dry_run=True)
        # Reset state loaded from disk so tests are isolated
        p._positions = {}
        p._orders = {}
        p._daily_pnl = 0.0
        p._save_state = MagicMock()
        p._log_event = MagicMock()
        return p

    def test_fill_creates_new_position(self):
        p = self._make_portfolio()

        fill = {
            "order_id": "order-1",
            "market_ticker": "KXMLBGAME-26JUL211907TBTOR-TB",
            "side": "yes",
            "purchased_side": "yes",
            "count": 5.0,
            "yes_price": 0.42,
        }
        p.on_fill(fill)

        self.assertTrue(p.has_position("KXMLBGAME-26JUL211907TBTOR-TB"))
        positions = p.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["side"], "yes")
        self.assertEqual(positions[0]["contracts"], 5)
        self.assertAlmostEqual(positions[0]["entry_price"], 0.42)

    def test_fill_accumulates_existing_position(self):
        from pregame.trading.portfolio import PositionState
        p = self._make_portfolio()

        # Seed an existing position
        p._positions["KXMLBGAME-26JUL211907TBTOR-TB"] = {
            "ticker": "KXMLBGAME-26JUL211907TBTOR-TB",
            "side": "yes",
            "entry_price": 0.40,
            "contracts": 5,
            "state": PositionState.FILLED,
            "target": "home_win",
            "confidence_tier": "HIGH",
            "accuracy_mult": 1.2,
            "entry_edge": 0.03,
        }

        fill = {
            "order_id": "order-2",
            "market_ticker": "KXMLBGAME-26JUL211907TBTOR-TB",
            "side": "yes",
            "purchased_side": "yes",
            "count": 5.0,
            "yes_price": 0.50,
        }
        p.on_fill(fill)

        pos = p._positions["KXMLBGAME-26JUL211907TBTOR-TB"]
        self.assertEqual(pos["contracts"], 10)
        # Weighted average: (0.40*5 + 0.50*5) / 10 = 0.45
        self.assertAlmostEqual(pos["entry_price"], 0.45)

    def test_fill_removes_fully_filled_order(self):
        p = self._make_portfolio()
        p._orders["order-1"] = {
            "order_id": "order-1",
            "ticker": "KXMLBGAME-26JUL211907TBTOR-TB",
            "side": "yes",
            "price_cents": 42,
            "contracts": 5,
        }

        fill = {
            "order_id": "order-1",
            "market_ticker": "KXMLBGAME-26JUL211907TBTOR-TB",
            "purchased_side": "yes",
            "count": 5.0,
            "yes_price": 0.42,
        }
        p.on_fill(fill)

        self.assertNotIn("order-1", p._orders)

    def test_partial_fill_decrements_order(self):
        p = self._make_portfolio()
        p._orders["order-1"] = {
            "order_id": "order-1",
            "ticker": "KXMLBGAME-26JUL211907TBTOR-TB",
            "side": "yes",
            "price_cents": 42,
            "contracts": 10,
        }

        fill = {
            "order_id": "order-1",
            "market_ticker": "KXMLBGAME-26JUL211907TBTOR-TB",
            "purchased_side": "yes",
            "count": 3.0,
            "yes_price": 0.42,
        }
        p.on_fill(fill)

        self.assertIn("order-1", p._orders)
        self.assertEqual(p._orders["order-1"]["contracts"], 7)

    def test_no_side_fill_uses_correct_entry_price(self):
        """When purchased_side is 'no', entry_price = 1 - yes_price."""
        p = self._make_portfolio()

        fill = {
            "order_id": "order-1",
            "market_ticker": "KXMLBGAME-26JUL211907TBTOR-TB",
            "side": "no",
            "purchased_side": "no",
            "count": 5.0,
            "yes_price": 0.60,
        }
        p.on_fill(fill)

        pos = p._positions["KXMLBGAME-26JUL211907TBTOR-TB"]
        self.assertEqual(pos["side"], "no")
        # Entry price for NO = 1 - 0.60 = 0.40
        self.assertAlmostEqual(pos["entry_price"], 0.40)


class TestPortfolioOnOrderUpdate(unittest.TestCase):
    """Test Portfolio.on_order_update() state transitions."""

    def _make_portfolio(self):
        from pregame.trading.portfolio import Portfolio
        p = Portfolio(client=None, dry_run=True)
        p._positions = {}
        p._orders = {}
        p._daily_pnl = 0.0
        p._save_state = MagicMock()
        p._log_event = MagicMock()
        return p

    def test_resting_order_added(self):
        p = self._make_portfolio()

        order = {
            "order_id": "new-order-1",
            "ticker": "KXMLBTOTAL-26JUL211907TBTOR-O8",
            "status": "resting",
            "side": "yes",
            "yes_price": 0.42,
            "remaining_count": 10.0,
        }
        p.on_order_update(order)

        self.assertIn("new-order-1", p._orders)
        self.assertEqual(p._orders["new-order-1"]["ticker"], "KXMLBTOTAL-26JUL211907TBTOR-O8")
        self.assertEqual(p._orders["new-order-1"]["price_cents"], 42)
        self.assertEqual(p._orders["new-order-1"]["contracts"], 10)

    def test_canceled_order_removed(self):
        p = self._make_portfolio()
        p._orders["existing-order"] = {
            "order_id": "existing-order",
            "ticker": "KXMLBGAME-26JUL211907TBTOR-TB",
            "side": "yes",
            "price_cents": 55,
            "contracts": 5,
        }

        order = {
            "order_id": "existing-order",
            "ticker": "KXMLBGAME-26JUL211907TBTOR-TB",
            "status": "canceled",
            "side": "yes",
            "yes_price": 0.55,
            "remaining_count": 0.0,
        }
        p.on_order_update(order)

        self.assertNotIn("existing-order", p._orders)

    def test_executed_order_removed(self):
        p = self._make_portfolio()
        p._orders["exec-order"] = {
            "order_id": "exec-order",
            "ticker": "KXMLBRFI-26JUL211907TBTOR",
            "side": "yes",
            "price_cents": 30,
            "contracts": 3,
        }

        order = {
            "order_id": "exec-order",
            "ticker": "KXMLBRFI-26JUL211907TBTOR",
            "status": "executed",
            "side": "yes",
            "yes_price": 0.30,
            "remaining_count": 0.0,
        }
        p.on_order_update(order)

        self.assertNotIn("exec-order", p._orders)

    def test_no_side_price_conversion(self):
        """NO side orders store price as 100 - yes_price_cents."""
        p = self._make_portfolio()

        order = {
            "order_id": "no-order-1",
            "ticker": "KXMLBGAME-26JUL211907TBTOR-TB",
            "status": "resting",
            "side": "no",
            "yes_price": 0.60,
            "remaining_count": 5.0,
        }
        p.on_order_update(order)

        # NO at yes_price 0.60 → NO price = 100 - 60 = 40 cents
        self.assertEqual(p._orders["no-order-1"]["price_cents"], 40)


class TestPortfolioOnPositionUpdate(unittest.TestCase):
    """Test Portfolio.on_position_update() — authoritative exchange state."""

    def _make_portfolio(self):
        from pregame.trading.portfolio import Portfolio
        p = Portfolio(client=None, dry_run=True)
        p._positions = {}
        p._orders = {}
        p._daily_pnl = 0.0
        p._save_state = MagicMock()
        p._log_event = MagicMock()
        return p

    def test_new_position_created(self):
        p = self._make_portfolio()

        position = {
            "market_ticker": "KXMLBGAME-26JUL211907TBTOR-TB",
            "position": 8.0,
            "position_cost": 3.2,
            "realized_pnl": 0.0,
            "fees_paid": 0.0,
            "volume": 8.0,
        }
        p.on_position_update(position)

        pos = p._positions["KXMLBGAME-26JUL211907TBTOR-TB"]
        self.assertEqual(pos["side"], "yes")
        self.assertEqual(pos["contracts"], 8)
        # entry_price = cost / contracts = 3.2 / 8 = 0.4
        self.assertAlmostEqual(pos["entry_price"], 0.4)

    def test_negative_position_is_no_side(self):
        p = self._make_portfolio()

        position = {
            "market_ticker": "KXMLBGAME-26JUL211907TBTOR-TB",
            "position": -5.0,
            "position_cost": -2.0,
            "realized_pnl": 0.0,
            "fees_paid": 0.0,
            "volume": 5.0,
        }
        p.on_position_update(position)

        pos = p._positions["KXMLBGAME-26JUL211907TBTOR-TB"]
        self.assertEqual(pos["side"], "no")
        self.assertEqual(pos["contracts"], 5)
        self.assertAlmostEqual(pos["entry_price"], 0.4)

    def test_zero_position_removes_and_records_pnl(self):
        from pregame.trading.portfolio import PositionState
        p = self._make_portfolio()

        # Seed existing position
        p._positions["KXMLBGAME-26JUL211907TBTOR-TB"] = {
            "ticker": "KXMLBGAME-26JUL211907TBTOR-TB",
            "side": "yes",
            "entry_price": 0.40,
            "contracts": 5,
            "state": PositionState.FILLED,
        }

        position = {
            "market_ticker": "KXMLBGAME-26JUL211907TBTOR-TB",
            "position": 0.0,
            "position_cost": 0.0,
            "realized_pnl": 3.0,
            "fees_paid": 0.1,
            "volume": 5.0,
        }
        p.on_position_update(position)

        self.assertNotIn("KXMLBGAME-26JUL211907TBTOR-TB", p._positions)
        self.assertAlmostEqual(p._daily_pnl, 3.0)

    def test_preserves_existing_metadata(self):
        """When position_update adjusts an existing position, model metadata is kept."""
        from pregame.trading.portfolio import PositionState
        p = self._make_portfolio()

        p._positions["KXMLBTOTAL-26JUL211907TBTOR-O8"] = {
            "ticker": "KXMLBTOTAL-26JUL211907TBTOR-O8",
            "side": "yes",
            "entry_price": 0.35,
            "contracts": 5,
            "state": PositionState.HOLDING,
            "target": "total_runs",
            "confidence_tier": "HIGH",
            "accuracy_mult": 1.3,
            "entry_edge": 0.04,
            "opened_at": "2026-07-21T18:00:00+00:00",
        }

        position = {
            "market_ticker": "KXMLBTOTAL-26JUL211907TBTOR-O8",
            "position": 8.0,
            "position_cost": 3.2,
            "realized_pnl": 0.0,
            "fees_paid": 0.0,
            "volume": 8.0,
        }
        p.on_position_update(position)

        pos = p._positions["KXMLBTOTAL-26JUL211907TBTOR-O8"]
        self.assertEqual(pos["contracts"], 8)
        # Metadata preserved from existing
        self.assertEqual(pos["state"], PositionState.HOLDING)
        self.assertEqual(pos["target"], "total_runs")
        self.assertEqual(pos["confidence_tier"], "HIGH")
        self.assertAlmostEqual(pos["accuracy_mult"], 1.3)


# ── Rate limit upgrade ──────────────────────────────────────────────────────

class TestUpgradeRateLimit(unittest.TestCase):
    """Test kalshi_client.upgrade_rate_limit() calls correct endpoint."""

    def test_calls_post_endpoint(self):
        from pregame.trading.kalshi_client import KalshiClient
        client = KalshiClient.__new__(KalshiClient)
        client._request = MagicMock(return_value={})

        client.upgrade_rate_limit()

        client._request.assert_called_once_with(
            "POST", "/account/api_usage_level/upgrade"
        )


# ── WS _on_message routing ──────────────────────────────────────────────────

class TestWSMessageRouting(unittest.TestCase):
    """Test that _on_message routes new message types correctly."""

    def _make_ws(self):
        from pregame.trading.ws import KalshiWS
        with patch("pregame.trading.ws._load_private_key", return_value=MagicMock()):
            ws = KalshiWS(api_key="test", rsa_key_path="/dev/null")
        ws._handle_fill = MagicMock()
        ws._handle_user_order = MagicMock()
        ws._handle_market_position = MagicMock()
        return ws

    def test_routes_fill(self):
        ws = self._make_ws()
        ws._on_message(None, json.dumps({"type": "fill", "msg": {}}))
        ws._handle_fill.assert_called_once()

    def test_routes_user_order(self):
        ws = self._make_ws()
        ws._on_message(None, json.dumps({"type": "user_order", "msg": {}}))
        ws._handle_user_order.assert_called_once()

    def test_routes_market_position(self):
        ws = self._make_ws()
        ws._on_message(None, json.dumps({"type": "market_position", "msg": {}}))
        ws._handle_market_position.assert_called_once()

    def test_still_routes_existing_types(self):
        ws = self._make_ws()
        ws._handle_snapshot = MagicMock()
        ws._handle_delta = MagicMock()
        ws._handle_trade = MagicMock()
        ws._handle_lifecycle = MagicMock()

        ws._on_message(None, json.dumps({"type": "orderbook_snapshot", "msg": {}}))
        ws._handle_snapshot.assert_called_once()

        ws._on_message(None, json.dumps({"type": "orderbook_delta", "msg": {}}))
        ws._handle_delta.assert_called_once()

        ws._on_message(None, json.dumps({"type": "trade", "msg": {}}))
        ws._handle_trade.assert_called_once()

        ws._on_message(None, json.dumps({"type": "market_lifecycle_v2", "msg": {}}))
        ws._handle_lifecycle.assert_called_once()


if __name__ == "__main__":
    unittest.main()
