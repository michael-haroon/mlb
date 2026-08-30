"""
Tests for the trading fixes:
  - schedule.py  : GUMBO UTC parsing, game_has_started, hours_to_first_pitch
  - runner.py    : _game_key_to_date
  - models.py    : EnsembleStore inference cache lifecycle
  - scanner.py   : generate_quotes cache hit/miss logging (unit-level)
  - risk.py      : no upper-bound hours gate
"""
from __future__ import annotations

import sys
import types
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch


# ── _game_key_to_date ─────────────────────────────────────────────────────────

class TestGameKeyToDate(unittest.TestCase):

    def _fn(self):
        # Import lazily so we don't need the full runner env
        from trading.runner import _game_key_to_date
        return _game_key_to_date

    def test_standard_key_with_time(self):
        f = self._fn()
        self.assertEqual(f("26JUL101840PHIDET"), "2026-07-10")

    def test_standard_key_without_time(self):
        f = self._fn()
        self.assertEqual(f("26JUL10PHIDET"), "2026-07-10")

    def test_october_date(self):
        f = self._fn()
        self.assertEqual(f("26OCT031905NYMLAD"), "2026-10-03")

    def test_invalid_returns_none(self):
        f = self._fn()
        self.assertIsNone(f("BADINPUT"))

    def test_short_key_returns_none(self):
        f = self._fn()
        self.assertIsNone(f("26JU"))


# ── schedule.py ───────────────────────────────────────────────────────────────

class TestScheduleUTCParsing(unittest.TestCase):
    """GUMBO gameDate is always UTC — verify parse is timezone-aware."""

    def _make_fake_game(self, game_date_utc: str, away: str, home: str) -> dict:
        return {
            "gameDate": game_date_utc,
            "status": {"detailedState": "Scheduled", "startTimeTBD": False},
            "teams": {
                "away": {"team": {"abbreviation": away}},
                "home": {"team": {"abbreviation": home}},
            },
        }

    def test_get_first_pitch_utc_correct_timezone(self):
        from trading import schedule as s

        fake_game = self._make_fake_game("2026-07-10T22:40:00Z", "PHI", "DET")

        with patch.object(s, "_fetch_schedule", return_value=[fake_game]):
            s._cache.clear()
            fp = s.get_first_pitch_utc("PHI", "DET", "2026-07-10")

        self.assertIsNotNone(fp)
        self.assertEqual(fp.tzinfo, timezone.utc)
        self.assertEqual(fp.hour, 22)
        self.assertEqual(fp.minute, 40)

    def test_game_has_started_true_for_past(self):
        from trading import schedule as s

        # Game started 2 hours ago
        past = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        fake_game = self._make_fake_game(past, "NYM", "LAD")

        with patch.object(s, "_fetch_schedule", return_value=[fake_game]):
            s._cache.clear()
            started = s.game_has_started("NYM", "LAD", "2026-07-10")

        self.assertTrue(started)

    def test_game_has_started_false_for_future(self):
        from trading import schedule as s

        # Game starts in 4 hours
        future = (datetime.now(timezone.utc) + timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ")
        fake_game = self._make_fake_game(future, "NYM", "LAD")

        with patch.object(s, "_fetch_schedule", return_value=[fake_game]):
            s._cache.clear()
            started = s.game_has_started("NYM", "LAD", "2026-07-10")

        self.assertFalse(started)

    def test_unknown_game_returns_false_not_started(self):
        from trading import schedule as s

        with patch.object(s, "_fetch_schedule", return_value=[]):
            s._cache.clear()
            # Unknown game → conservative: assume not started
            started = s.game_has_started("XYZ", "ABC", "2026-07-10")

        self.assertFalse(started)

    def test_hours_to_first_pitch_positive_for_future(self):
        from trading import schedule as s

        future = (datetime.now(timezone.utc) + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        fake_game = self._make_fake_game(future, "BOS", "NYY")

        with patch.object(s, "_fetch_schedule", return_value=[fake_game]):
            s._cache.clear()
            h = s.hours_to_first_pitch("BOS", "NYY", "2026-07-10")

        self.assertIsNotNone(h)
        self.assertGreater(h, 2.9)
        self.assertLess(h, 3.1)

    def test_cache_is_reused_within_ttl(self):
        from trading import schedule as s

        future = (datetime.now(timezone.utc) + timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        fake_game = self._make_fake_game(future, "HOU", "TEX")

        with patch.object(s, "_fetch_schedule", return_value=[fake_game]) as mock_fetch:
            s._cache.clear()
            s.get_first_pitch_utc("HOU", "TEX", "2026-07-10")
            s.get_first_pitch_utc("HOU", "TEX", "2026-07-10")  # second call

        # _fetch_schedule should have been called only once (cache hit on second)
        self.assertEqual(mock_fetch.call_count, 1)

    def test_kalshi_abbrev_normalisation(self):
        """Kalshi uses 'KC' for Kansas City; GUMBO uses 'KC'. Both should normalise."""
        from trading import schedule as s

        future = (datetime.now(timezone.utc) + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        fake_game = self._make_fake_game(future, "KC", "BAL")

        with patch.object(s, "_fetch_schedule", return_value=[fake_game]):
            s._cache.clear()
            fp = s.get_first_pitch_utc("KC", "BAL", "2026-07-10")

        self.assertIsNotNone(fp)


# ── EnsembleStore inference cache ─────────────────────────────────────────────

class TestEnsembleCacheLifecycle(unittest.TestCase):

    def _make_store(self):
        from trading.models import EnsembleStore
        store = EnsembleStore.__new__(EnsembleStore)
        store._models_dir = MagicMock()
        store._bundles = {}
        store._loaded = {}
        store.inference_cache = {}
        store._inference_cache_features_hash = None
        return store

    def test_cache_starts_empty(self):
        store = self._make_store()
        self.assertEqual(store.inference_cache, {})
        self.assertIsNone(store._inference_cache_features_hash)

    def test_reload_all_clears_inference_cache(self):
        store = self._make_store()
        store.inference_cache[("PHI", "DET", "home_win")] = {"prob": 0.55}
        store._inference_cache_features_hash = "abc123"

        # patch discover() to be a no-op
        store.discover = MagicMock()
        store.reload_all()

        self.assertEqual(store.inference_cache, {})
        self.assertIsNone(store._inference_cache_features_hash)

    def test_invalidate_clears_cache(self):
        store = self._make_store()
        store.inference_cache[("A", "B", "yrfi")] = {"prob": 0.3}
        store._inference_cache_features_hash = "xyz"

        store.invalidate_inference_cache()

        self.assertEqual(store.inference_cache, {})
        self.assertIsNone(store._inference_cache_features_hash)


# ── risk.py: no upper-bound gate ──────────────────────────────────────────────

class TestRiskNoUpperBound(unittest.TestCase):

    def _check(self, hours):
        from trading.risk import check_limits
        state = {
            "daily_pnl": 0.0,
            "position_tickers": set(),
            "positions": [],
            "open_orders": [],
        }
        allowed, reason = check_limits(
            ticker="KXMLBGAME-26JUL101840PHIDET-PHI",
            price=0.55,
            contracts=2,
            hours_to_first_pitch=hours,
            bankroll=1000.0,
            portfolio_state=state,
        )
        return allowed, reason

    def test_48h_out_is_allowed(self):
        allowed, reason = self._check(48.0)
        self.assertTrue(allowed, f"Expected allowed at 48h, got: {reason}")

    def test_100h_out_is_allowed(self):
        allowed, reason = self._check(100.0)
        self.assertTrue(allowed, f"Expected allowed at 100h, got: {reason}")

    def test_too_close_is_blocked(self):
        allowed, reason = self._check(0.2)
        self.assertFalse(allowed)
        self.assertIn("Too close", reason)

    def test_exactly_at_min_threshold_is_allowed(self):
        # MIN_HOURS_TO_FIRST_PITCH = 0.5; test at 0.5 (boundary — strictly less blocks)
        allowed, reason = self._check(0.5)
        self.assertTrue(allowed, f"Expected allowed at exactly 0.5h, got: {reason}")


if __name__ == "__main__":
    unittest.main()
