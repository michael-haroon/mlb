"""Tests for stale position sweep and per-scan sharpness check."""

import json
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import numpy as np
import pytest


# ── Fix 2: Sharpness is per-scan, not cumulative ────────────────────────────


class TestSharpnessPerScan:
    """Verify that sharpness collapse does not persist across scans."""

    def test_sharpness_halts_only_current_scan(self):
        """If std is low THIS scan, quotes are removed; next scan starts fresh."""
        from trading.scanner import _check_batch_sharpness

        # Simulate a batch with collapsed home_win predictions (all near 0.5)
        quotes_collapsed = [
            {"target": "home_win", "model_prob": 0.50},
            {"target": "home_win", "model_prob": 0.51},
            {"target": "home_win", "model_prob": 0.49},
            {"target": "home_win", "model_prob": 0.50},
            {"target": "total_runs", "model_prob": 0.30},
        ]
        halted = _check_batch_sharpness(quotes_collapsed)
        assert "home_win" in halted
        assert "total_runs" not in halted  # only 1 quote, below min 3

    def test_sharpness_does_not_persist(self):
        """Calling _check_batch_sharpness again with good data returns empty."""
        from trading.scanner import _check_batch_sharpness

        # First scan: collapsed
        quotes_bad = [
            {"target": "home_win", "model_prob": 0.50 + i * 0.001}
            for i in range(5)
        ]
        halted_bad = _check_batch_sharpness(quotes_bad)
        assert "home_win" in halted_bad

        # Second scan: healthy spread
        quotes_good = [
            {"target": "home_win", "model_prob": 0.3},
            {"target": "home_win", "model_prob": 0.5},
            {"target": "home_win", "model_prob": 0.7},
            {"target": "home_win", "model_prob": 0.6},
        ]
        halted_good = _check_batch_sharpness(quotes_good)
        assert "home_win" not in halted_good

    def test_sharpness_returns_set(self):
        """_check_batch_sharpness returns a set (not None)."""
        from trading.scanner import _check_batch_sharpness

        halted = _check_batch_sharpness([])
        assert isinstance(halted, set)
        assert len(halted) == 0

    def test_sharpness_requires_min_3_quotes(self):
        """Targets with fewer than 3 quotes are never halted."""
        from trading.scanner import _check_batch_sharpness

        quotes = [
            {"target": "home_win", "model_prob": 0.50},
            {"target": "home_win", "model_prob": 0.50},
        ]
        halted = _check_batch_sharpness(quotes)
        assert "home_win" not in halted


# ── Fix 1: Stale position sweep ─────────────────────────────────────────────


class TestSweepStalePositions:
    """Verify that the sweep correctly settles old positions via REST."""

    def _make_runner(self):
        """Create a minimal TradingRunner with mocked components."""
        from trading.runner import TradingRunner
        runner = TradingRunner(dry_run=True, env="prod", bankroll=350.0)

        # Mock the client
        runner._client = MagicMock()

        # Mock the portfolio
        runner._portfolio = MagicMock()

        return runner

    def test_sweep_settles_old_positions(self):
        """Positions older than 6h whose market has a result get settled."""
        runner = self._make_runner()
        old_time = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()

        runner._portfolio.get_positions.return_value = [
            {
                "ticker": "KXMLBGAME-26JUL082210COLLAD-COL",
                "state": "filled",
                "side": "yes",
                "entry_price": 0.31,
                "contracts": 10,
                "opened_at": old_time,
            }
        ]
        runner._client.get_market.return_value = {
            "market": {"status": "finalized", "result": "no"}
        }
        runner._portfolio.record_settlement.return_value = -3.10

        runner._sweep_stale_positions()

        runner._portfolio.record_settlement.assert_called_once_with(
            "KXMLBGAME-26JUL082210COLLAD-COL", yes_won=False
        )

    def test_sweep_skips_recent_positions(self):
        """Positions younger than 6h are not polled."""
        runner = self._make_runner()
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()

        runner._portfolio.get_positions.return_value = [
            {
                "ticker": "KXMLBGAME-26JUL182210COLLAD-COL",
                "state": "filled",
                "side": "yes",
                "entry_price": 0.50,
                "contracts": 10,
                "opened_at": recent_time,
            }
        ]

        runner._sweep_stale_positions()

        runner._client.get_market.assert_not_called()
        runner._portfolio.record_settlement.assert_not_called()

    def test_sweep_skips_already_settled(self):
        """Positions in 'settled' state are not re-processed."""
        runner = self._make_runner()
        old_time = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

        runner._portfolio.get_positions.return_value = [
            {
                "ticker": "KXMLBGAME-26JUL082210COLLAD-COL",
                "state": "settled",
                "side": "yes",
                "entry_price": 0.31,
                "contracts": 10,
                "opened_at": old_time,
            }
        ]

        runner._sweep_stale_positions()

        runner._client.get_market.assert_not_called()

    def test_sweep_handles_undetermined_market(self):
        """If market has no result yet, position stays filled."""
        runner = self._make_runner()
        old_time = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()

        runner._portfolio.get_positions.return_value = [
            {
                "ticker": "KXMLBGAME-26JUL182210COLLAD-COL",
                "state": "filled",
                "side": "yes",
                "entry_price": 0.50,
                "contracts": 10,
                "opened_at": old_time,
            }
        ]
        runner._client.get_market.return_value = {
            "market": {"status": "open", "result": ""}
        }

        runner._sweep_stale_positions()

        runner._portfolio.record_settlement.assert_not_called()

    def test_sweep_handles_api_error(self):
        """API errors for one position don't crash the sweep."""
        runner = self._make_runner()
        old_time = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()

        runner._portfolio.get_positions.return_value = [
            {
                "ticker": "KXMLBGAME-26JUL082210COLLAD-COL",
                "state": "filled",
                "side": "yes",
                "entry_price": 0.31,
                "contracts": 10,
                "opened_at": old_time,
            },
            {
                "ticker": "KXMLBGAME-26JUL091910PHICIN-PHI",
                "state": "filled",
                "side": "no",
                "entry_price": 0.41,
                "contracts": 10,
                "opened_at": old_time,
            },
        ]
        # First call errors, second succeeds
        runner._client.get_market.side_effect = [
            Exception("connection timeout"),
            {"market": {"status": "finalized", "result": "yes"}},
        ]
        runner._portfolio.record_settlement.return_value = 5.90

        runner._sweep_stale_positions()

        # Only the second should be settled
        runner._portfolio.record_settlement.assert_called_once_with(
            "KXMLBGAME-26JUL091910PHICIN-PHI", yes_won=True
        )


# ── Fix 3: Exposure excludes settled/exited positions ──────────────────────


class TestExposureExcludesSettled:
    """Verify that total_exposure() only counts open positions."""

    def _make_portfolio(self):
        from trading.portfolio import Portfolio
        p = Portfolio(client=None, dry_run=True)
        return p

    def test_settled_positions_excluded_from_exposure(self):
        """Positions in 'settled' state do not count toward exposure."""
        from trading.portfolio import PositionState
        p = self._make_portfolio()
        p._positions = {
            "TICK-A": {"entry_price": 0.40, "contracts": 10, "state": PositionState.SETTLED},
            "TICK-B": {"entry_price": 0.30, "contracts": 10, "state": PositionState.FILLED},
        }
        # Only TICK-B should count: 0.30 * 10 = 3.0
        assert abs(p.total_exposure() - 3.0) < 0.001

    def test_exited_positions_excluded_from_exposure(self):
        """Positions in 'exited' state do not count toward exposure."""
        from trading.portfolio import PositionState
        p = self._make_portfolio()
        p._positions = {
            "TICK-A": {"entry_price": 0.50, "contracts": 10, "state": PositionState.EXITED},
            "TICK-B": {"entry_price": 0.60, "contracts": 5, "state": PositionState.HOLDING},
        }
        # Only TICK-B: 0.60 * 5 = 3.0
        assert abs(p.total_exposure() - 3.0) < 0.001

    def test_string_state_also_excluded(self):
        """Handles string 'settled'/'exited' (from JSON deserialization)."""
        p = self._make_portfolio()
        p._positions = {
            "TICK-A": {"entry_price": 0.40, "contracts": 10, "state": "settled"},
            "TICK-B": {"entry_price": 0.40, "contracts": 10, "state": "exited"},
            "TICK-C": {"entry_price": 0.25, "contracts": 10, "state": "filled"},
        }
        # Only TICK-C: 0.25 * 10 = 2.5
        assert abs(p.total_exposure() - 2.5) < 0.001

    def test_all_settled_means_zero_exposure(self):
        """If every position is settled, exposure is $0."""
        p = self._make_portfolio()
        p._positions = {
            "TICK-A": {"entry_price": 0.40, "contracts": 10, "state": "settled"},
            "TICK-B": {"entry_price": 0.60, "contracts": 20, "state": "settled"},
            "TICK-C": {"entry_price": 0.35, "contracts": 15, "state": "settled"},
        }
        assert p.total_exposure() == 0.0

    def test_resting_orders_still_counted(self):
        """Resting orders contribute to exposure even when positions are settled."""
        p = self._make_portfolio()
        p._positions = {
            "TICK-A": {"entry_price": 0.40, "contracts": 10, "state": "settled"},
        }
        p._orders = {
            "order-1": {"price_cents": 35, "contracts": 10},
        }
        # Position excluded, order counted: 35/100 * 10 = 3.5
        assert abs(p.total_exposure() - 3.5) < 0.001


# ── Fix 3b: Settlement removes position from live dict ─────────────────────


class TestSettlementRemovesPosition:
    """Verify that record_settlement/record_exit purge positions from the dict."""

    def _make_portfolio(self):
        from trading.portfolio import Portfolio, PositionState
        p = Portfolio(client=None, dry_run=True)
        p._positions = {
            "TICK-A": {
                "ticker": "TICK-A",
                "side": "yes",
                "entry_price": 0.40,
                "contracts": 10,
                "state": PositionState.FILLED,
            },
            "TICK-B": {
                "ticker": "TICK-B",
                "side": "no",
                "entry_price": 0.35,
                "contracts": 5,
                "state": PositionState.FILLED,
            },
        }
        return p

    def test_record_settlement_removes_position(self):
        """After settlement, position is gone from the dict."""
        p = self._make_portfolio()
        assert "TICK-A" in p._positions
        pnl = p.record_settlement("TICK-A", yes_won=True)
        assert pnl is not None
        assert "TICK-A" not in p._positions
        # Other position unaffected
        assert "TICK-B" in p._positions

    def test_record_settlement_updates_pnl(self):
        """P&L is accumulated even though position is removed."""
        p = self._make_portfolio()
        p._daily_pnl = 0.0
        pnl = p.record_settlement("TICK-A", yes_won=True)
        # YES won, we held YES at 0.40, gain = (1 - 0.40) * 10 = 6.0
        assert abs(pnl - 6.0) < 0.001
        assert abs(p.daily_pnl - 6.0) < 0.001

    def test_record_exit_removes_position(self):
        """After exit, position is gone from the dict."""
        p = self._make_portfolio()
        pnl = p.record_exit("TICK-B", exit_price=0.55)
        assert pnl is not None
        assert "TICK-B" not in p._positions
        assert "TICK-A" in p._positions

    def test_position_count_decreases_after_settlement(self):
        """position_count() reflects removal."""
        p = self._make_portfolio()
        assert p.position_count() == 2
        p.record_settlement("TICK-A", yes_won=False)
        assert p.position_count() == 1

    def test_exposure_drops_after_settlement(self):
        """total_exposure() drops when a position is settled."""
        p = self._make_portfolio()
        before = p.total_exposure()
        p.record_settlement("TICK-A", yes_won=True)
        after = p.total_exposure()
        # TICK-A was 0.40 * 10 = 4.0 of exposure
        assert abs(before - after - 4.0) < 0.001


# ── Fix 4: Feature rebuild loop prevention ─────────────────────────────────


class TestFeatureRebuildLoop:
    """Verify that features don't rebuild every scan when no new games exist."""

    def _make_manager(self, tmp_path):
        from trading.features import FeatureManager
        features_path = tmp_path / "game_features.parquet"
        # Create a dummy parquet file
        import pandas as pd
        df = pd.DataFrame({"game_date": ["2026-07-18"], "x": [1]})
        df.to_parquet(features_path)
        fm = FeatureManager(
            s3_uri="s3://fake-bucket/data",
            features_path=features_path,
            local_cache=tmp_path / "raw_cache",
        )
        return fm

    def test_stale_after_first_run_uses_last_rebuild(self, tmp_path):
        """After a successful rebuild, is_stale checks _last_rebuild, not mtime."""
        from datetime import datetime, timezone, timedelta
        fm = self._make_manager(tmp_path)

        # File mtime is now (just created) — not stale
        assert not fm.is_stale()

        # Simulate old file: _last_rebuild is None, file was made 20h ago
        import os
        old_time = time.time() - 20 * 3600
        os.utime(fm._features_path, (old_time, old_time))
        assert fm.is_stale()  # mtime fallback triggers

        # Now simulate a rebuild completed 2h ago
        fm._last_rebuild = datetime.now(timezone.utc) - timedelta(hours=2)
        assert not fm.is_stale()  # uses _last_rebuild, not mtime

    def test_stale_triggers_again_after_max_age(self, tmp_path):
        """_last_rebuild older than FEATURES_MAX_AGE_HOURS triggers rebuild."""
        from datetime import datetime, timezone, timedelta
        from trading.config import FEATURES_MAX_AGE_HOURS
        fm = self._make_manager(tmp_path)

        # Last rebuild was beyond max age
        fm._last_rebuild = datetime.now(timezone.utc) - timedelta(hours=FEATURES_MAX_AGE_HOURS + 1)
        assert fm.is_stale()

    def test_not_stale_within_max_age(self, tmp_path):
        """_last_rebuild within FEATURES_MAX_AGE_HOURS does not trigger."""
        from datetime import datetime, timezone, timedelta
        from trading.config import FEATURES_MAX_AGE_HOURS
        fm = self._make_manager(tmp_path)

        fm._last_rebuild = datetime.now(timezone.utc) - timedelta(hours=FEATURES_MAX_AGE_HOURS - 1)
        assert not fm.is_stale()


# ── Fix 5: WebSocket subscription cleanup on reconnect ─────────────────────


class TestWSSubscriptionCleanup:
    """Verify that WS clears subscriptions on reconnect to prevent flood."""

    def test_on_open_clears_subscribed_tickers(self):
        """_on_open clears _subscribed_tickers so reconnect doesn't replay stale subs."""
        from unittest.mock import MagicMock, patch
        from trading.ws import KalshiWS

        with patch("trading.ws._load_private_key", return_value=MagicMock()):
            ws = KalshiWS(
                api_key="fake",
                rsa_key_path="/fake/path",
                env="prod",
            )

        # Simulate accumulated subscriptions from scan loop
        ws._subscribed_tickers = {"TICK-1", "TICK-2", "TICK-3", "TICK-700"}
        ws._ws = MagicMock()

        ws._on_open(ws._ws)

        # After reconnect, subscribed set should be empty
        assert len(ws._subscribed_tickers) == 0

    def test_settled_lifecycle_removes_from_subscriptions(self):
        """Settlement lifecycle event discards ticker from _subscribed_tickers."""
        from unittest.mock import MagicMock, patch
        from trading.ws import KalshiWS

        settle_calls = []

        with patch("trading.ws._load_private_key", return_value=MagicMock()):
            ws = KalshiWS(
                api_key="fake",
                rsa_key_path="/fake/path",
                env="prod",
                on_settle=lambda t: settle_calls.append(t),
            )

        ws._subscribed_tickers = {
            "KXMLBGAME-26JUL181605NYMPHI-PHI",
            "KXMLBTOTAL-26JUL181605NYMPHI-7",
        }
        ws._ws = MagicMock()

        # Simulate settled lifecycle event
        msg = {
            "type": "market_lifecycle_v2",
            "msg": {
                "event_type": "settled",
                "market_ticker": "KXMLBGAME-26JUL181605NYMPHI-PHI",
            },
        }
        ws._handle_lifecycle(msg)

        assert "KXMLBGAME-26JUL181605NYMPHI-PHI" not in ws._subscribed_tickers
        assert "KXMLBTOTAL-26JUL181605NYMPHI-7" in ws._subscribed_tickers

    def test_subscribe_market_adds_to_set_and_sends(self):
        """subscribe_market adds ticker and sends to live connection."""
        from unittest.mock import MagicMock, patch
        from trading.ws import KalshiWS

        with patch("trading.ws._load_private_key", return_value=MagicMock()):
            ws = KalshiWS(
                api_key="fake",
                rsa_key_path="/fake/path",
                env="prod",
            )

        ws._ws = MagicMock()
        ws._running = True

        ws.subscribe_market("KXMLBGAME-NEW-TICKER")

        assert "KXMLBGAME-NEW-TICKER" in ws._subscribed_tickers
        # Should have sent 2 messages (orderbook_delta + trade)
        assert ws._ws.send.call_count == 2
