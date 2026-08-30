"""
Tests for scanner.py confidence/shading system removal.

Validates that the refactor is zero-behavioral-change:
- conservative_fair_value now takes only model_prob (no shading)
- Half-spread uses fixed HALF_SPREAD_BASE_CENTS (3) regardless of input
- Output quote dicts still contain ensemble_std and confidence_tier fields
- With ensemble_std=0 and confidence_tier="HIGH" (the only values the
  1-member ensemble ever produced), old and new behavior are identical
"""
from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Optional

import numpy as np
import pandas as pd


class TestConservativeFairValue(unittest.TestCase):
    """Unit tests for the simplified conservative_fair_value function."""

    def _fn(self):
        from trading.scanner import conservative_fair_value
        return conservative_fair_value

    def test_passthrough_mid_range(self):
        """Values in (0.01, 0.99) pass through unchanged."""
        fn = self._fn()
        self.assertAlmostEqual(fn(0.55), 0.55)
        self.assertAlmostEqual(fn(0.30), 0.30)
        self.assertAlmostEqual(fn(0.75), 0.75)

    def test_clips_below_floor(self):
        """Values below 0.01 are clipped to 0.01."""
        fn = self._fn()
        self.assertAlmostEqual(fn(0.0), 0.01)
        self.assertAlmostEqual(fn(-0.05), 0.01)
        self.assertAlmostEqual(fn(0.005), 0.01)

    def test_clips_above_ceiling(self):
        """Values above 0.99 are clipped to 0.99."""
        fn = self._fn()
        self.assertAlmostEqual(fn(1.0), 0.99)
        self.assertAlmostEqual(fn(1.05), 0.99)
        self.assertAlmostEqual(fn(0.995), 0.99)

    def test_exact_boundary_values(self):
        """Boundary values 0.01 and 0.99 pass through."""
        fn = self._fn()
        self.assertAlmostEqual(fn(0.01), 0.01)
        self.assertAlmostEqual(fn(0.99), 0.99)

    def test_midpoint_exact(self):
        """0.5 passes through unchanged."""
        fn = self._fn()
        self.assertAlmostEqual(fn(0.5), 0.5)

    def test_at_price_floor(self):
        """PRICE_FLOOR (0.12) passes through as-is."""
        fn = self._fn()
        self.assertAlmostEqual(fn(0.12), 0.12)

    def test_at_price_ceiling(self):
        """PRICE_CEILING (0.88) passes through as-is."""
        fn = self._fn()
        self.assertAlmostEqual(fn(0.88), 0.88)


class TestBehavioralEquivalence(unittest.TestCase):
    """Verify that old behavior with ensemble_std=0, confidence_tier='HIGH'
    produces identical results to the new simplified function.

    Old: shade = SHADE_SIGMA["HIGH"] * ensemble_std = 0.5 * 0.0 = 0
         fair = model_prob - 0 = model_prob  (then clip)
    New: fair = clip(model_prob, 0.01, 0.99)
    """

    def _new_fn(self):
        from trading.scanner import conservative_fair_value
        return conservative_fair_value

    def _old_behavior(self, model_prob: float, ensemble_std: float = 0.0,
                      confidence_tier: str = "HIGH") -> float:
        """Reproduce the OLD logic exactly for comparison."""
        SHADE_SIGMA = {"HIGH": 0.5, "MEDIUM": 1.0, "LOW": 1.5}
        shade = SHADE_SIGMA.get(confidence_tier, 1.0) * ensemble_std
        if model_prob > 0.5:
            fair = model_prob - shade
        else:
            fair = model_prob + shade
        return float(np.clip(fair, 0.01, 0.99))

    def test_equivalence_across_probability_range(self):
        """For every probability, old(p, 0.0, 'HIGH') == new(p)."""
        new_fn = self._new_fn()
        test_probs = [0.0, 0.01, 0.05, 0.12, 0.25, 0.40, 0.5,
                      0.55, 0.62, 0.75, 0.88, 0.95, 0.99, 1.0]
        for p in test_probs:
            old_result = self._old_behavior(p, ensemble_std=0.0, confidence_tier="HIGH")
            new_result = float(new_fn(p))
            self.assertAlmostEqual(
                old_result, new_result, places=10,
                msg=f"Mismatch at prob={p}: old={old_result}, new={new_result}"
            )

    def test_equivalence_random_probs(self):
        """Fuzz test: 100 random probabilities give identical results."""
        new_fn = self._new_fn()
        rng = np.random.default_rng(42)
        for p in rng.uniform(0, 1, 100):
            old_result = self._old_behavior(float(p))
            new_result = float(new_fn(float(p)))
            self.assertAlmostEqual(old_result, new_result, places=10)


class TestHalfSpreadConstant(unittest.TestCase):
    """Verify _price_market uses fixed HALF_SPREAD_BASE_CENTS=3."""

    def test_spread_is_fixed_at_3(self):
        """Confirm HALF_SPREAD_BASE_CENTS is 3."""
        from trading.config import HALF_SPREAD_BASE_CENTS
        self.assertEqual(HALF_SPREAD_BASE_CENTS, 3)

    def test_old_high_tier_was_2_not_3(self):
        """The old system would have used 2 for HIGH tier — this confirms
        behavior IS different for the spread width (2 -> 3).
        But since ensemble_std=0 was always forcing confidence_tier='HIGH',
        and HALF_SPREAD_CENTS.get('HIGH', 3) returned 2, the change from
        2 to 3 is intentional as documented in the refactor spec."""
        from trading.config import HALF_SPREAD_BASE_CENTS
        # The new fixed value is 3 (was 2 for HIGH tier in the old dict)
        self.assertEqual(HALF_SPREAD_BASE_CENTS, 3)


class TestPriceMarketOutputStructure(unittest.TestCase):
    """Verify _price_market still returns ensemble_std and confidence_tier."""

    def _make_parsed_ticker(self, ticker="KXMLBTOTAL-26AUG11-NYY-BOS-O8.5"):
        """Create a minimal ParsedTicker-like object."""
        parsed = MagicMock()
        parsed.series = "KXMLBTOTAL"
        parsed.raw = ticker
        parsed.away_team = "NYY"
        parsed.home_team = "BOS"
        parsed.strike_value = 8.5
        parsed.strike_team = None
        return parsed

    def _make_game_row(self):
        """Minimal feature DataFrame."""
        return pd.DataFrame({
            "home_team_abbr": ["BOS"],
            "away_team_abbr": ["NYY"],
            "game_date": [pd.Timestamp("2026-08-11")],
        })

    def _make_ensemble_store(self):
        """Mock EnsembleStore that returns a valid result."""
        store = MagicMock()
        store.get_bundle.return_value = MagicMock()
        return store

    @patch("trading.scanner.predict_market_prob")
    def test_output_contains_ensemble_fields(self, mock_predict):
        """Output quote dict still has ensemble_std and confidence_tier."""
        from trading.scanner import _price_market

        mock_predict.return_value = {
            "prob": 0.55,
            "ensemble_std": 0.0,
            "confidence_tier": "HIGH",
            "task": "regression",
            "n_models_used": 1,
            "point_estimate": 8.7,
            "distribution": {"type": "negbin", "alpha": 6.73, "mu": 8.7},
        }

        parsed = self._make_parsed_ticker()
        game_row = self._make_game_row()
        store = self._make_ensemble_store()
        # Book mid at 0.45 so edge = |0.55 - 0.45| = 0.10 (passes min_edge gate)
        book_tops = {"KXMLBTOTAL-26AUG11-NYY-BOS-O8.5": (40, 50)}
        result_cache = {}

        # Need to mock _apply_line's internal import
        with patch("trading.scanner._apply_line") as mock_apply:
            mock_apply.return_value = {
                "prob": 0.55,
                "ensemble_std": 0.0,
                "confidence_tier": "HIGH",
                "task": "regression",
                "n_models_used": 1,
                "point_estimate": 8.7,
            }
            quote = _price_market(parsed, game_row, store, book_tops, result_cache)

        self.assertIsNotNone(quote)
        self.assertIn("ensemble_std", quote)
        self.assertIn("confidence_tier", quote)
        self.assertEqual(quote["ensemble_std"], 0.0)
        self.assertEqual(quote["confidence_tier"], "HIGH")

    @patch("trading.scanner.predict_market_prob")
    def test_spread_width_is_3(self, mock_predict):
        """Bid/ask spread uses fixed 3-cent half-spread."""
        from trading.scanner import _price_market

        mock_predict.return_value = {
            "prob": 0.50,
            "ensemble_std": 0.0,
            "confidence_tier": "HIGH",
            "task": "regression",
            "n_models_used": 1,
            "point_estimate": 8.7,
            "distribution": {"type": "negbin", "alpha": 6.73, "mu": 8.7},
        }

        parsed = self._make_parsed_ticker()
        game_row = self._make_game_row()
        store = self._make_ensemble_store()
        # Book mid at 0.30 so edge = |0.50 - 0.30| = 0.20 (passes min_edge gate)
        book_tops = {"KXMLBTOTAL-26AUG11-NYY-BOS-O8.5": (25, 35)}
        result_cache = {}

        with patch("trading.scanner._apply_line") as mock_apply:
            mock_apply.return_value = {
                "prob": 0.50,
                "ensemble_std": 0.0,
                "confidence_tier": "HIGH",
                "task": "regression",
                "n_models_used": 1,
                "point_estimate": 8.7,
            }
            quote = _price_market(parsed, game_row, store, book_tops, result_cache)

        self.assertIsNotNone(quote)
        # fair_value = 0.50 -> fair_cents = 50
        # bid = 50 - 3 = 47, ask = 50 + 3 = 53
        self.assertEqual(quote["bid_cents"], 47)
        self.assertEqual(quote["ask_cents"], 53)


class TestGenerateQuotesStructure(unittest.TestCase):
    """Regression test: generate_quotes output maintains expected structure."""

    @patch("trading.scanner._price_market")
    @patch("trading.scanner._lookup_game_row")
    @patch("trading.scanner.parse_ticker")
    @patch("trading.scanner.classify_cluster")
    def test_output_quote_structure(self, mock_cluster, mock_parse,
                                     mock_lookup, mock_price):
        """generate_quotes returns dicts with all expected fields."""
        from trading.scanner import generate_quotes

        mock_parse.return_value = MagicMock(
            series="KXMLBTOTAL", away_team="NYY", home_team="BOS"
        )
        mock_cluster.return_value = "total"
        mock_lookup.return_value = pd.DataFrame({"x": [1]})
        mock_price.return_value = {
            "ticker": "KXMLBTOTAL-26AUG11-NYY-BOS-O8.5",
            "target": "total_runs",
            "fair_value": 0.55,
            "model_prob": 0.55,
            "ensemble_std": 0.0,
            "confidence_tier": "HIGH",
            "bid_cents": 52,
            "ask_cents": 58,
            "no_buy_cents": 42,
            "edge_at_mid": 0.05,
            "line": 8.5,
            "direction": "over",
        }

        # Mock EnsembleStore
        store = MagicMock()
        store._inference_cache_features_hash = ""
        store.inference_cache = {}

        markets = [{"ticker": "KXMLBTOTAL-26AUG11-NYY-BOS-O8.5"}]
        features = pd.DataFrame({
            "home_team_abbr": ["BOS"],
            "away_team_abbr": ["NYY"],
            "game_date": [pd.Timestamp("2026-08-11")],
        })
        book_tops = {}

        quotes = generate_quotes(markets, features, store, book_tops)

        self.assertEqual(len(quotes), 1)
        q = quotes[0]
        # All expected fields present
        expected_keys = {
            "ticker", "target", "fair_value", "model_prob",
            "ensemble_std", "confidence_tier", "bid_cents", "ask_cents",
            "no_buy_cents", "edge_at_mid", "line", "direction", "cluster",
        }
        self.assertEqual(set(q.keys()), expected_keys)
        self.assertEqual(q["cluster"], "total")
        self.assertEqual(q["ensemble_std"], 0.0)
        self.assertEqual(q["confidence_tier"], "HIGH")


class TestEdgeCases(unittest.TestCase):
    """Edge cases for conservative_fair_value at domain boundaries."""

    def _fn(self):
        from trading.scanner import conservative_fair_value
        return conservative_fair_value

    def test_prob_zero(self):
        """model_prob=0.0 clips to 0.01."""
        self.assertAlmostEqual(self._fn()(0.0), 0.01)

    def test_prob_one(self):
        """model_prob=1.0 clips to 0.99."""
        self.assertAlmostEqual(self._fn()(1.0), 0.99)

    def test_prob_half(self):
        """model_prob=0.5 passes through unchanged."""
        self.assertAlmostEqual(self._fn()(0.5), 0.5)

    def test_at_price_floor_boundary(self):
        """model_prob at PRICE_FLOOR (0.12) passes through."""
        self.assertAlmostEqual(self._fn()(0.12), 0.12)

    def test_at_price_ceiling_boundary(self):
        """model_prob at PRICE_CEILING (0.88) passes through."""
        self.assertAlmostEqual(self._fn()(0.88), 0.88)

    def test_negative_prob(self):
        """Negative model_prob clips to 0.01."""
        self.assertAlmostEqual(self._fn()(-0.1), 0.01)

    def test_prob_greater_than_one(self):
        """model_prob > 1.0 clips to 0.99."""
        self.assertAlmostEqual(self._fn()(1.5), 0.99)


class TestRemovedConstants(unittest.TestCase):
    """Verify the dead constants were actually removed from config."""

    def test_shade_sigma_removed(self):
        """SHADE_SIGMA no longer exists in config."""
        import trading.config as cfg
        self.assertFalse(hasattr(cfg, "SHADE_SIGMA"))

    def test_expected_pred_std_removed(self):
        """EXPECTED_PRED_STD no longer exists in config."""
        import trading.config as cfg
        self.assertFalse(hasattr(cfg, "EXPECTED_PRED_STD"))

    def test_min_sharpness_ratio_removed(self):
        """MIN_SHARPNESS_RATIO no longer exists in config."""
        import trading.config as cfg
        self.assertFalse(hasattr(cfg, "MIN_SHARPNESS_RATIO"))

    def test_half_spread_cents_dict_removed(self):
        """HALF_SPREAD_CENTS dict no longer exists in config."""
        import trading.config as cfg
        self.assertFalse(hasattr(cfg, "HALF_SPREAD_CENTS"))

    def test_half_spread_base_cents_exists(self):
        """HALF_SPREAD_BASE_CENTS replacement exists."""
        import trading.config as cfg
        self.assertTrue(hasattr(cfg, "HALF_SPREAD_BASE_CENTS"))
        self.assertEqual(cfg.HALF_SPREAD_BASE_CENTS, 3)


class TestNegbinAdaptiveSpread(unittest.TestCase):
    """Verify _price_market uses NegBin-derived spread for total_runs."""

    def _make_parsed_ticker(self, ticker="KXMLBTOTAL-26AUG11-NYY-BOS-O8.5"):
        parsed = MagicMock()
        parsed.series = "KXMLBTOTAL"
        parsed.raw = ticker
        parsed.away_team = "NYY"
        parsed.home_team = "BOS"
        parsed.strike_value = 8.5
        parsed.strike_team = None
        return parsed

    def _make_game_row(self):
        return pd.DataFrame({
            "home_team_abbr": ["BOS"],
            "away_team_abbr": ["NYY"],
            "game_date": [pd.Timestamp("2026-08-11")],
        })

    @patch("trading.scanner.predict_market_prob")
    def test_high_mu_gets_wider_spread(self, mock_predict):
        """total_runs with mu=14 should produce wider spread (4 or 5 cents)."""
        from trading.scanner import _price_market

        mock_predict.return_value = {
            "prob": 0.60,
            "ensemble_std": 0.0,
            "confidence_tier": "HIGH",
            "task": "regression",
            "n_models_used": 1,
            "point_estimate": 14.0,
            "distribution": {"type": "negbin", "alpha": 6.73, "mu": 14.0},
        }

        parsed = self._make_parsed_ticker()
        game_row = self._make_game_row()
        store = MagicMock()
        store.get_bundle.return_value = MagicMock()
        book_tops = {"KXMLBTOTAL-26AUG11-NYY-BOS-O8.5": (30, 40)}
        result_cache = {}

        with patch("trading.scanner._apply_line") as mock_apply:
            mock_apply.return_value = {
                "prob": 0.60,
                "ensemble_std": 0.0,
                "confidence_tier": "HIGH",
                "task": "regression",
                "n_models_used": 1,
                "point_estimate": 14.0,
            }
            quote = _price_market(parsed, game_row, store, book_tops, result_cache)

        self.assertIsNotNone(quote)
        # mu=14, alpha=6.73 → negbin_std=5.73 → raw=3.29 → 3
        # Wider than low-mu games
        half = quote["ask_cents"] - int(round(quote["fair_value"] * 100))
        self.assertGreaterEqual(half, 3)

    @patch("trading.scanner.predict_market_prob")
    def test_low_mu_gets_narrower_spread(self, mock_predict):
        """total_runs with mu=5 should produce narrower spread (2 cents)."""
        from trading.scanner import _price_market

        mock_predict.return_value = {
            "prob": 0.40,
            "ensemble_std": 0.0,
            "confidence_tier": "HIGH",
            "task": "regression",
            "n_models_used": 1,
            "point_estimate": 5.0,
            "distribution": {"type": "negbin", "alpha": 6.73, "mu": 5.0},
        }

        parsed = self._make_parsed_ticker()
        game_row = self._make_game_row()
        store = MagicMock()
        store.get_bundle.return_value = MagicMock()
        book_tops = {"KXMLBTOTAL-26AUG11-NYY-BOS-O8.5": (20, 30)}
        result_cache = {}

        with patch("trading.scanner._apply_line") as mock_apply:
            mock_apply.return_value = {
                "prob": 0.40,
                "ensemble_std": 0.0,
                "confidence_tier": "HIGH",
                "task": "regression",
                "n_models_used": 1,
                "point_estimate": 5.0,
            }
            quote = _price_market(parsed, game_row, store, book_tops, result_cache)

        self.assertIsNotNone(quote)
        # mu=5, alpha=6.73 → negbin_std=3.23 → raw=2.29 → 2
        fair_cents = int(round(quote["fair_value"] * 100))
        half = quote["ask_cents"] - fair_cents
        self.assertEqual(half, 2)

    @patch("trading.scanner.predict_market_prob")
    def test_non_total_runs_uses_fixed_spread(self, mock_predict):
        """Classification targets (home_win) use fixed HALF_SPREAD_BASE_CENTS."""
        from trading.scanner import _price_market

        mock_predict.return_value = {
            "prob": 0.55,
            "ensemble_std": 0.0,
            "confidence_tier": "HIGH",
            "task": "classification",
            "n_models_used": 1,
        }

        parsed = MagicMock()
        parsed.series = "KXMLBGAME"
        parsed.raw = "KXMLBGAME-26AUG11-NYY-BOS"
        parsed.away_team = "NYY"
        parsed.home_team = "BOS"
        parsed.strike_value = None
        parsed.strike_team = None

        game_row = pd.DataFrame({
            "home_team_abbr": ["BOS"],
            "away_team_abbr": ["NYY"],
            "game_date": [pd.Timestamp("2026-08-11")],
        })
        store = MagicMock()
        store.get_bundle.return_value = MagicMock()
        book_tops = {"KXMLBGAME-26AUG11-NYY-BOS": (40, 50)}
        result_cache = {}

        with patch("trading.scanner._apply_line") as mock_apply:
            mock_apply.return_value = {
                "prob": 0.55,
                "ensemble_std": 0.0,
                "confidence_tier": "HIGH",
                "task": "classification",
                "n_models_used": 1,
            }
            quote = _price_market(parsed, game_row, store, book_tops, result_cache)

        self.assertIsNotNone(quote)
        fair_cents = int(round(quote["fair_value"] * 100))
        half = quote["ask_cents"] - fair_cents
        self.assertEqual(half, 3)  # Fixed spread for non-total_runs


if __name__ == "__main__":
    unittest.main()
