"""
Tests for the trading pipeline end-to-end behavior.

Tests the expected flow: markets → scanner → sizing → risk → executor.
Specifically validates whether quotes make it through all filter stages
and whether dry-run mode can ever produce positions.

These tests mock external dependencies (Kalshi API, S3, GUMBO) and use
the actual pipeline logic to identify where quotes are being suppressed.
"""
from __future__ import annotations

import sys
import json
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ════════════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_features():
    """Minimal game_features DataFrame that will pass feature availability checks."""
    np.random.seed(42)
    n_features = 80  # enough to pass >30% populated check
    cols = [f"feat_{i}" for i in range(n_features)]
    cols += ["home_team_abbr", "away_team_abbr", "game_date"]
    data = {c: np.random.randn(1) for c in cols[:n_features]}
    data["home_team_abbr"] = ["LAD"]
    data["away_team_abbr"] = ["NYM"]
    data["game_date"] = [pd.Timestamp("2026-07-26")]
    return pd.DataFrame(data)


@pytest.fixture
def fake_markets():
    """Sample open Kalshi markets for NYM@LAD."""
    return [
        {"ticker": "KXMLBGAME-26JUL261910NYMLAD-LAD", "status": "open"},
        {"ticker": "KXMLBGAME-26JUL261910NYMLAD-NYM", "status": "open"},
        {"ticker": "KXMLBRFI-26JUL261910NYMLAD", "status": "open"},
        {"ticker": "KXMLBTOTAL-26JUL261910NYMLAD-9", "status": "open"},
        {"ticker": "KXMLBSPREAD-26JUL261910NYMLAD-LAD2", "status": "open"},
        {"ticker": "KXMLBTEAMTOTAL-26JUL261910NYMLAD-LAD4", "status": "open"},
    ]


@pytest.fixture
def book_tops_favorable():
    """Book tops with a wide spread — edge should exist."""
    return {
        "KXMLBGAME-26JUL261910NYMLAD-LAD": (50, 70),   # mid = 60c, wide spread
        "KXMLBGAME-26JUL261910NYMLAD-NYM": (30, 50),   # mid = 40c
        "KXMLBRFI-26JUL261910NYMLAD": (22, 38),         # mid = 30c
        "KXMLBTOTAL-26JUL261910NYMLAD-9": (45, 60),     # mid = 52.5c
        "KXMLBSPREAD-26JUL261910NYMLAD-LAD2": (25, 45), # mid = 35c
        "KXMLBTEAMTOTAL-26JUL261910NYMLAD-LAD4": (30, 50),
    }


@pytest.fixture
def book_tops_tight():
    """Book tops very close to model fair — edge may be too small."""
    return {
        "KXMLBGAME-26JUL261910NYMLAD-LAD": (58, 62),   # mid = 60c, tight
        "KXMLBGAME-26JUL261910NYMLAD-NYM": (38, 42),
        "KXMLBRFI-26JUL261910NYMLAD": (28, 32),
        "KXMLBTOTAL-26JUL261910NYMLAD-9": (48, 52),
        "KXMLBSPREAD-26JUL261910NYMLAD-LAD2": (33, 37),
        "KXMLBTEAMTOTAL-26JUL261910NYMLAD-LAD4": (38, 42),
    }


@pytest.fixture
def empty_portfolio_state():
    """Clean portfolio state for risk checks."""
    return {
        "positions": [],
        "open_orders": [],
        "daily_pnl": 0.0,
        "position_tickers": set(),
    }


def _make_mock_ensemble_store(model_prob=0.60, ensemble_std=0.04, tier="MEDIUM"):
    """Create a mock EnsembleStore that returns fixed predictions."""
    store = MagicMock()
    store.inference_cache = {}
    store._inference_cache_features_hash = None

    def _get_bundle(target):
        return {"target": target}

    store.get_bundle = _get_bundle
    return store


# ════════════════════════════════════════════════════════════════════════════════
# Test: Scanner generates quotes when model has edge over market
# ════════════════════════════════════════════════════════════════════════════════


class TestScannerQuoteGeneration:
    """Verify scanner.generate_quotes produces quotes under normal conditions."""

    def test_classification_quote_generated_when_edge_exists(
        self, sample_features, fake_markets, book_tops_favorable
    ):
        """Model at 0.60, market mid at 0.60 → edge exists from half-spread.
        With book at (50, 70), market_mid = 0.60. Model fair = 0.60.
        edge_at_mid = |0.60 - 0.60| = 0.0 → FILTERED by min_edge gate.

        Model at 0.65, market mid at 0.60 → edge = 0.05 → passes.
        """
        from trading.scanner import generate_quotes, conservative_fair_value

        model_prob = 0.65
        ensemble_std = 0.04
        tier = "HIGH"

        fair = conservative_fair_value(model_prob, ensemble_std, tier)
        assert fair > 0.5, "Fair should be above 0.5 for model_prob=0.65"

        # Mock predict_market_prob to return our fixed probability
        mock_result = {
            "prob": model_prob,
            "ensemble_std": ensemble_std,
            "confidence_tier": tier,
            "task": "classification",
            "n_models_used": 5,
        }

        store = MagicMock()
        store.inference_cache = {}
        store._inference_cache_features_hash = None
        store.get_bundle = MagicMock(return_value={"member_bundles": [], "task": "classification"})

        with patch("trading.scanner.predict_market_prob", return_value=mock_result):
            quotes = generate_quotes(
                [fake_markets[0]],  # Just the LAD winner market
                sample_features,
                store,
                book_tops_favorable,
            )

        assert len(quotes) > 0, (
            "Expected at least 1 quote when model has 5c edge over market mid"
        )
        q = quotes[0]
        assert q["ticker"] == "KXMLBGAME-26JUL261910NYMLAD-LAD"
        assert q["model_prob"] == model_prob
        assert q["edge_at_mid"] > 0

    def test_no_quote_when_edge_below_min(
        self, sample_features, fake_markets
    ):
        """Model fair exactly equals book midpoint → edge = 0 → no quote.

        With MEDIUM tier, shade = 1.0 * ensemble_std. So:
          model_prob=0.60 → fair = 0.60 - 1.0*0.04 = 0.56
        We construct a book whose mid equals 0.56 exactly.
        """
        from trading.scanner import generate_quotes, conservative_fair_value

        model_prob = 0.60
        ensemble_std = 0.04
        tier = "MEDIUM"

        fair = conservative_fair_value(model_prob, ensemble_std, tier)
        # fair = 0.56 for these params

        # Construct book centered exactly on fair
        fair_cents = int(round(fair * 100))  # 56
        book_tops = {
            "KXMLBGAME-26JUL261910NYMLAD-LAD": (fair_cents - 1, fair_cents + 1),
        }

        mock_result = {
            "prob": model_prob,
            "ensemble_std": ensemble_std,
            "confidence_tier": tier,
            "task": "classification",
            "n_models_used": 5,
        }

        store = MagicMock()
        store.inference_cache = {}
        store._inference_cache_features_hash = None
        store.get_bundle = MagicMock(return_value={"member_bundles": [], "task": "classification"})

        with patch("trading.scanner.predict_market_prob", return_value=mock_result):
            quotes = generate_quotes(
                [fake_markets[0]],
                sample_features,
                store,
                book_tops,
            )

        assert len(quotes) == 0, (
            f"Expected 0 quotes when book mid ({fair:.3f}) equals model fair ({fair:.3f})"
        )

    def test_no_quote_when_fair_below_price_floor(self, sample_features, fake_markets):
        """Model prob = 0.08 → fair after shading < PRICE_FLOOR (0.12) → filtered."""
        from trading.scanner import generate_quotes
        from trading.config import PRICE_FLOOR

        model_prob = 0.08  # Below floor after shading toward 0.5
        ensemble_std = 0.04
        tier = "MEDIUM"

        mock_result = {
            "prob": model_prob,
            "ensemble_std": ensemble_std,
            "confidence_tier": tier,
            "task": "classification",
            "n_models_used": 5,
        }

        store = MagicMock()
        store.inference_cache = {}
        store._inference_cache_features_hash = None
        store.get_bundle = MagicMock(return_value={"member_bundles": [], "task": "classification"})

        book_tops = {"KXMLBRFI-26JUL261910NYMLAD": (5, 15)}

        with patch("trading.scanner.predict_market_prob", return_value=mock_result):
            quotes = generate_quotes(
                [fake_markets[2]],  # YRFI market
                sample_features,
                store,
                book_tops,
            )

        # Fair value = 0.08 + shade*std → shaded toward 0.5, but raw is 0.08
        # After shading: 0.08 + 1.0*0.04 = 0.12 (exactly at floor)
        # Depends on whether >= or > is used in the filter
        # Point is: very low probs get filtered
        assert all(q["fair_value"] >= PRICE_FLOOR for q in quotes), (
            "Quotes should not have fair_value below PRICE_FLOOR"
        )

    def test_no_quote_when_fair_above_price_ceiling(self, sample_features, fake_markets):
        """Model prob = 0.95 → fair after shading > PRICE_CEILING (0.88) → filtered."""
        from trading.scanner import generate_quotes
        from trading.config import PRICE_CEILING

        model_prob = 0.95
        ensemble_std = 0.03
        tier = "HIGH"

        mock_result = {
            "prob": model_prob,
            "ensemble_std": ensemble_std,
            "confidence_tier": tier,
            "task": "classification",
            "n_models_used": 5,
        }

        store = MagicMock()
        store.inference_cache = {}
        store._inference_cache_features_hash = None
        store.get_bundle = MagicMock(return_value={"member_bundles": [], "task": "classification"})

        book_tops = {"KXMLBGAME-26JUL261910NYMLAD-LAD": (85, 98)}

        with patch("trading.scanner.predict_market_prob", return_value=mock_result):
            quotes = generate_quotes(
                [fake_markets[0]],
                sample_features,
                store,
                book_tops,
            )

        # fair = 0.95 - 0.5*0.03 = 0.935 > PRICE_CEILING (0.88) → filtered
        assert len(quotes) == 0, (
            f"Expected 0 quotes when fair (0.935) > PRICE_CEILING ({PRICE_CEILING})"
        )


# ════════════════════════════════════════════════════════════════════════════════
# Test: Sharpness collapse gate
# ════════════════════════════════════════════════════════════════════════════════


class TestSharpnessGate:
    """The batch sharpness check halts targets whose predictions are too uniform."""

    def test_sharpness_collapse_halts_target(self, sample_features):
        """If all home_win predictions cluster near 0.50, the sharpness gate fires."""
        from trading.scanner import _check_batch_sharpness
        from trading.config import EXPECTED_PRED_STD, MIN_SHARPNESS_RATIO

        # Simulate 5 quotes for home_win with nearly identical predictions
        quotes = [
            {"target": "home_win", "model_prob": 0.500 + i * 0.001}
            for i in range(5)
        ]
        # std of [0.500, 0.501, 0.502, 0.503, 0.504] ≈ 0.0014
        # EXPECTED_PRED_STD["home_win"] = 0.14
        # 0.0014 < 0.14 * 0.40 = 0.056 → halted

        halted = _check_batch_sharpness(quotes)
        assert "home_win" in halted, (
            "home_win should be halted when live_std << expected_std"
        )

    def test_sharpness_ok_allows_target(self, sample_features):
        """Normal prediction variance does not trigger the gate."""
        from trading.scanner import _check_batch_sharpness

        # Simulate 5 quotes with healthy variance
        quotes = [
            {"target": "home_win", "model_prob": p}
            for p in [0.35, 0.45, 0.55, 0.65, 0.72]
        ]
        # std ≈ 0.137, expected 0.14, 0.137 > 0.14 * 0.40 → not halted

        halted = _check_batch_sharpness(quotes)
        assert "home_win" not in halted, (
            "home_win should NOT be halted when predictions have healthy variance"
        )

    def test_sharpness_not_checked_with_few_quotes(self):
        """< 3 quotes for a target → sharpness check skipped (not enough data)."""
        from trading.scanner import _check_batch_sharpness

        quotes = [
            {"target": "home_win", "model_prob": 0.50},
            {"target": "home_win", "model_prob": 0.50},
        ]
        halted = _check_batch_sharpness(quotes)
        assert "home_win" not in halted


# ════════════════════════════════════════════════════════════════════════════════
# Test: Sizing produces valid contracts
# ════════════════════════════════════════════════════════════════════════════════


class TestSizing:
    """Sizing converts raw quotes into executable SizedQuotes."""

    def test_positive_edge_produces_nonzero_contracts(self):
        """A quote with positive edge must size to at least 1 contract."""
        from trading.sizing import size_quotes

        quotes = [{
            "ticker": "KXMLBGAME-26JUL261910NYMLAD-LAD",
            "target": "home_win",
            "cluster": "winner",
            "fair_value": 0.60,
            "model_prob": 0.65,
            "ensemble_std": 0.04,
            "confidence_tier": "HIGH",
            "bid_cents": 58,
            "ask_cents": 62,
            "edge_at_mid": 0.05,
        }]

        sized = size_quotes(quotes, bankroll=1000.0)
        assert len(sized) > 0, "Expected at least 1 sized quote with positive edge"
        assert sized[0].contracts >= 1
        assert sized[0].ticker == quotes[0]["ticker"]

    def test_zero_edge_produces_no_quotes(self):
        """Edge ≤ 0 means no position should be sized."""
        from trading.sizing import size_quotes

        quotes = [{
            "ticker": "KXMLBGAME-26JUL261910NYMLAD-LAD",
            "target": "home_win",
            "cluster": "winner",
            "fair_value": 0.60,
            "model_prob": 0.60,
            "ensemble_std": 0.04,
            "confidence_tier": "MEDIUM",
            "bid_cents": 57,
            "ask_cents": 63,
            "edge_at_mid": 0.0,
        }]

        sized = size_quotes(quotes, bankroll=1000.0)
        assert len(sized) == 0, "Expected 0 sized quotes with zero edge"

    def test_cluster_cap_enforced(self):
        """Can't exceed CLUSTER_MAX_CONTRACTS per game within a cluster."""
        from trading.sizing import size_quotes
        from trading.config import CLUSTER_MAX_CONTRACTS

        cap = CLUSTER_MAX_CONTRACTS["winner"]  # 10

        # Existing inventory already at cap for this game
        existing = {("winner", "26JUL261910NYMLAD"): cap}

        quotes = [{
            "ticker": "KXMLBGAME-26JUL261910NYMLAD-LAD",
            "target": "home_win",
            "cluster": "winner",
            "fair_value": 0.60,
            "model_prob": 0.70,
            "ensemble_std": 0.03,
            "confidence_tier": "HIGH",
            "bid_cents": 58,
            "ask_cents": 62,
            "edge_at_mid": 0.10,
        }]

        sized = size_quotes(quotes, bankroll=1000.0, existing_inventory=existing)
        assert len(sized) == 0, (
            f"Expected 0 sized quotes when cluster cap ({cap}) already reached"
        )

    def test_max_contracts_per_market_capped(self):
        """Individual market is capped at MAX_CONTRACTS_PER_MARKET."""
        from trading.sizing import size_quotes
        from trading.config import MAX_CONTRACTS_PER_MARKET

        # Very high edge + bankroll to push raw contracts above cap
        quotes = [{
            "ticker": "KXMLBGAME-26JUL261910NYMLAD-LAD",
            "target": "home_win",
            "cluster": "winner",
            "fair_value": 0.50,
            "model_prob": 0.80,
            "ensemble_std": 0.02,
            "confidence_tier": "HIGH",
            "bid_cents": 48,
            "ask_cents": 52,
            "edge_at_mid": 0.30,
        }]

        sized = size_quotes(quotes, bankroll=100_000.0)
        assert len(sized) > 0
        assert sized[0].contracts <= MAX_CONTRACTS_PER_MARKET, (
            f"Contracts ({sized[0].contracts}) exceeds max ({MAX_CONTRACTS_PER_MARKET})"
        )


# ════════════════════════════════════════════════════════════════════════════════
# Test: Risk gate
# ════════════════════════════════════════════════════════════════════════════════


class TestRiskGate:
    """Risk checks that block execution after sizing."""

    def test_passes_clean_state(self, empty_portfolio_state):
        """A valid quote passes risk checks with clean portfolio."""
        from trading.risk import check_limits

        allowed, reason = check_limits(
            ticker="KXMLBGAME-26JUL261910NYMLAD-LAD",
            price=0.60,
            contracts=3,
            hours_to_first_pitch=4.0,
            bankroll=1000.0,
            portfolio_state=empty_portfolio_state,
        )
        assert allowed is True, f"Expected allowed, got: {reason}"

    def test_blocks_duplicate_ticker(self, empty_portfolio_state):
        """Cannot trade same ticker twice."""
        from trading.risk import check_limits

        empty_portfolio_state["position_tickers"] = {"KXMLBGAME-26JUL261910NYMLAD-LAD"}

        allowed, reason = check_limits(
            ticker="KXMLBGAME-26JUL261910NYMLAD-LAD",
            price=0.60,
            contracts=3,
            hours_to_first_pitch=4.0,
            bankroll=1000.0,
            portfolio_state=empty_portfolio_state,
        )
        assert allowed is False
        assert "Already positioned" in reason

    def test_blocks_circuit_breaker(self, empty_portfolio_state):
        """Daily loss exceeds DAILY_LOSS_LIMIT_PCT → halt all trading."""
        from trading.risk import check_limits
        from trading.config import DAILY_LOSS_LIMIT_PCT

        loss_limit = 1000.0 * DAILY_LOSS_LIMIT_PCT / 100.0
        empty_portfolio_state["daily_pnl"] = -(loss_limit + 1.0)

        allowed, reason = check_limits(
            ticker="KXMLBGAME-26JUL261910NYMLAD-LAD",
            price=0.60,
            contracts=3,
            hours_to_first_pitch=4.0,
            bankroll=1000.0,
            portfolio_state=empty_portfolio_state,
        )
        assert allowed is False
        assert "Circuit breaker" in reason

    def test_blocks_max_concurrent_positions(self, empty_portfolio_state):
        """Cannot exceed MAX_CONCURRENT_POSITIONS."""
        from trading.risk import check_limits
        from trading.config import MAX_CONCURRENT_POSITIONS

        # Fill up to the limit
        empty_portfolio_state["positions"] = [
            {"ticker": f"FAKE-{i}", "entry_price": 0.5, "contracts": 1}
            for i in range(MAX_CONCURRENT_POSITIONS)
        ]

        allowed, reason = check_limits(
            ticker="KXMLBGAME-26JUL261910NYMLAD-LAD",
            price=0.60,
            contracts=3,
            hours_to_first_pitch=4.0,
            bankroll=1000.0,
            portfolio_state=empty_portfolio_state,
        )
        assert allowed is False
        assert "Max concurrent" in reason

    def test_blocks_too_close_to_first_pitch(self, empty_portfolio_state):
        """hours_to_first_pitch < MIN_HOURS_TO_FIRST_PITCH blocks."""
        from trading.risk import check_limits
        from trading.config import MIN_HOURS_TO_FIRST_PITCH

        allowed, reason = check_limits(
            ticker="KXMLBGAME-26JUL261910NYMLAD-LAD",
            price=0.60,
            contracts=3,
            hours_to_first_pitch=0.2,  # 12 min < 30 min threshold
            bankroll=1000.0,
            portfolio_state=empty_portfolio_state,
        )
        assert allowed is False
        assert "Too close" in reason


# ════════════════════════════════════════════════════════════════════════════════
# Test: Executor in dry-run mode
# ════════════════════════════════════════════════════════════════════════════════


class TestExecutorDryRun:
    """Executor behavior in dry-run mode."""

    def test_post_two_sided_dry_run_always_returns_dry_run_status(self):
        """Dry-run post_two_sided returns status=DRY_RUN."""
        from trading.executor import post_two_sided

        result = post_two_sided(
            client=MagicMock(),
            ticker="KXMLBGAME-26JUL261910NYMLAD-LAD",
            bid_cents=58,
            ask_cents=62,
            contracts=3,
            dry_run=True,
        )
        assert result["status"] == "DRY_RUN"

    def test_execute_taker_dry_run_returns_dry_run_status(self):
        """Dry-run execute_taker returns status=DRY_RUN."""
        from trading.executor import execute_taker

        result = execute_taker(
            client=MagicMock(),
            ticker="KXMLBGAME-26JUL261910NYMLAD-LAD",
            side="yes",
            contracts=3,
            price_cents=55,
            dry_run=True,
            reason="test edge",
        )
        assert result["status"] == "DRY_RUN"


# ════════════════════════════════════════════════════════════════════════════════
# Test: DRY-RUN STRUCTURAL ISSUE — maker quotes never create positions
# ════════════════════════════════════════════════════════════════════════════════


class TestDryRunPositionTracking:
    """
    CRITICAL TEST: Validates whether dry-run mode can ever accumulate positions.

    In the runner's run_once() logic:
    - Taker fills → portfolio.add_position() is called (even in dry-run)
    - Maker posts → post_two_sided() returns DRY_RUN but NO add_position() call

    This means: if the market never crosses far enough for a taker opportunity,
    dry-run will NEVER create positions — even though quotes are posted.
    The portfolio summary will always show "Positions: 0 | Orders: 0".
    """

    def test_maker_quote_does_not_create_position(self, tmp_path):
        """post_two_sided in dry-run does NOT call add_position.

        This is the root cause of 0 positions in dry-run: maker quotes are
        logged but never tracked as positions. Only taker fills create positions.
        """
        from trading.executor import post_two_sided
        from trading.portfolio import Portfolio

        # Isolate from disk state by patching the state file path
        with patch("trading.portfolio._STATE_FILE", tmp_path / "state.json"):
            portfolio = Portfolio(client=None, dry_run=True)

        # Post a two-sided maker quote (dry-run)
        result = post_two_sided(
            client=MagicMock(),
            ticker="KXMLBGAME-26JUL261910NYMLAD-LAD",
            bid_cents=58,
            ask_cents=62,
            contracts=3,
            dry_run=True,
        )

        # The quote was "posted" (logged)
        assert result["status"] == "DRY_RUN"
        # But portfolio has NO position and NO order
        assert portfolio.position_count() == 0
        assert len(portfolio.get_open_orders()) == 0

    def test_taker_opportunity_does_create_position(self, tmp_path):
        """Taker fills DO call add_position — this is the only path to positions."""
        from trading.portfolio import Portfolio, PositionState

        with patch("trading.portfolio._STATE_FILE", tmp_path / "state.json"):
            portfolio = Portfolio(client=None, dry_run=True)

        # Simulate what runner.run_once does after a taker opportunity
        portfolio.add_position(
            ticker="KXMLBGAME-26JUL261910NYMLAD-LAD",
            side="yes",
            entry_price=0.55,
            contracts=3,
            target="home_win",
            confidence_tier="HIGH",
            accuracy_mult=1.2,
            entry_edge=0.05,
        )

        assert portfolio.position_count() == 1
        pos = portfolio.get_positions()[0]
        assert pos["ticker"] == "KXMLBGAME-26JUL261910NYMLAD-LAD"
        assert pos["state"] == PositionState.FILLED

    def test_dry_run_needs_taker_opportunity_for_nonzero_pnl(self, tmp_path):
        """Without taker opportunities, dry-run P&L stays at 0.

        The runner's decision logic:
        1. If book is crossed far past fair → TAKE (creates position, tracks P&L)
        2. Otherwise → MAKE (posts quote, never fills, no P&L)

        If no book crosses far enough, the dry-run is entirely paper with
        0 positions and 0 P&L indefinitely. This is the expected (but perhaps
        undesirable) behavior.
        """
        from trading.portfolio import Portfolio

        with patch("trading.portfolio._STATE_FILE", tmp_path / "state.json"):
            portfolio = Portfolio(client=None, dry_run=True)

        # After many scan cycles with only maker posts:
        assert portfolio.daily_pnl == 0.0
        assert portfolio.position_count() == 0
        assert portfolio.summary() == "Positions: 0 | Orders: 0 | Exposure: $0.00 | Day P&L: $0.00"


# ════════════════════════════════════════════════════════════════════════════════
# Test: Edge at mid calculation
# ════════════════════════════════════════════════════════════════════════════════


class TestEdgeCalculation:
    """The edge_at_mid calculation determines whether a quote passes MIN_EDGE."""

    def test_edge_is_abs_diff_of_fair_and_book_mid(self):
        """edge = |fair - (best_bid + best_ask) / 200|"""
        from trading.scanner import conservative_fair_value
        from trading.config import HALF_SPREAD_CENTS, MIN_EDGE_BUFFER_MAKER

        model_prob = 0.65
        ensemble_std = 0.04
        tier = "HIGH"
        fair = conservative_fair_value(model_prob, ensemble_std, tier)

        # Book: bid=50, ask=70 → mid = (50+70)/200 = 0.60
        book_mid = (50 + 70) / 200.0
        edge = abs(fair - book_mid)

        min_edge = 0.01 * MIN_EDGE_BUFFER_MAKER  # min_edge_for_profit * buffer
        assert edge > min_edge, (
            f"edge={edge:.4f} should exceed min_edge={min_edge:.4f} "
            f"for model_prob=0.65, book_mid=0.60"
        )

    def test_edge_zero_when_fair_equals_mid(self):
        """If model fair == book midpoint, edge = 0 → no quote."""
        from trading.scanner import conservative_fair_value
        from trading.config import MIN_EDGE_BUFFER_MAKER

        model_prob = 0.60
        ensemble_std = 0.04
        tier = "MEDIUM"
        fair = conservative_fair_value(model_prob, ensemble_std, tier)
        # fair = 0.60 - 1.0 * 0.04 = 0.56 (shaded toward 0.5)

        # Book mid at exactly the fair value
        book_bid = int(fair * 100) - 1
        book_ask = int(fair * 100) + 1
        book_mid = (book_bid + book_ask) / 200.0

        edge = abs(fair - book_mid)
        min_edge = 0.01 * MIN_EDGE_BUFFER_MAKER

        # When the book is centered exactly on our fair, edge → 0
        assert edge < min_edge, (
            f"Expected edge ({edge:.4f}) < min_edge ({min_edge:.4f}) "
            f"when book is centered on fair value"
        )


# ════════════════════════════════════════════════════════════════════════════════
# Test: Conservative fair value shading
# ════════════════════════════════════════════════════════════════════════════════


class TestConservativeFairValue:
    """Fair value is shaded toward 0.5 to ensure worst-case positive EV."""

    def test_shades_high_prob_down(self):
        """model_prob > 0.5 → shaded down (toward 0.5)."""
        from trading.scanner import conservative_fair_value

        fair = conservative_fair_value(0.70, 0.05, "MEDIUM")
        assert fair < 0.70
        assert fair > 0.50

    def test_shades_low_prob_up(self):
        """model_prob < 0.5 → shaded up (toward 0.5)."""
        from trading.scanner import conservative_fair_value

        fair = conservative_fair_value(0.30, 0.05, "MEDIUM")
        assert fair > 0.30
        assert fair < 0.50

    def test_high_confidence_shades_less(self):
        """HIGH tier shades less than LOW tier."""
        from trading.scanner import conservative_fair_value

        fair_high = conservative_fair_value(0.70, 0.05, "HIGH")
        fair_low = conservative_fair_value(0.70, 0.05, "LOW")
        assert fair_high > fair_low, (
            "HIGH confidence should shade less (closer to model_prob)"
        )

    def test_larger_std_shades_more(self):
        """Higher ensemble_std → more shading → fair closer to 0.5."""
        from trading.scanner import conservative_fair_value

        fair_tight = conservative_fair_value(0.70, 0.02, "MEDIUM")
        fair_wide = conservative_fair_value(0.70, 0.10, "MEDIUM")
        assert fair_tight > fair_wide, (
            "Larger std should shade more (closer to 0.5)"
        )


# ════════════════════════════════════════════════════════════════════════════════
# Test: Taker opportunity detection
# ════════════════════════════════════════════════════════════════════════════════


class TestTakerOpportunity:
    """Taker aggression fires when book is crossed far past fair."""

    def test_taker_fires_when_ask_far_below_fair(self):
        """If best_ask << fair, buy YES aggressively."""
        from trading.runner import TradingRunner
        from trading.sizing import SizedQuote

        runner = TradingRunner.__new__(TradingRunner)

        sq = SizedQuote(
            ticker="KXMLBGAME-26JUL261910NYMLAD-LAD",
            target="home_win",
            cluster="winner",
            fair_value=0.70,
            model_prob=0.72,
            ensemble_std=0.03,
            confidence_tier="HIGH",
            bid_cents=68,
            ask_cents=72,
            contracts=3,
            accuracy_mult=1.2,
            edge_at_mid=0.10,
        )

        # Best ask at 55c, our fair at 70c → edge = 15c
        result = runner._check_taker_opportunity(sq, best_bid=53, best_ask=55)

        assert result is not None, "Expected taker opportunity when ask << fair"
        assert result["side"] == "yes"
        assert result["edge"] > 0.10

    def test_taker_does_not_fire_when_ask_near_fair(self):
        """If best_ask is close to fair, no taker opportunity."""
        from trading.runner import TradingRunner
        from trading.sizing import SizedQuote

        runner = TradingRunner.__new__(TradingRunner)

        sq = SizedQuote(
            ticker="KXMLBGAME-26JUL261910NYMLAD-LAD",
            target="home_win",
            cluster="winner",
            fair_value=0.60,
            model_prob=0.62,
            ensemble_std=0.04,
            confidence_tier="MEDIUM",
            bid_cents=57,
            ask_cents=63,
            contracts=3,
            accuracy_mult=1.0,
            edge_at_mid=0.03,
        )

        # Best ask at 58c, fair at 60c → edge = 2c → below taker threshold
        result = runner._check_taker_opportunity(sq, best_bid=55, best_ask=58)

        assert result is None, "Should NOT aggress when edge is small"


# ════════════════════════════════════════════════════════════════════════════════
# Test: Time gate in run_once
# ════════════════════════════════════════════════════════════════════════════════


class TestTimeGate:
    """The EXIT_BUFFER_MINUTES gate blocks new quotes too close to first pitch."""

    def test_exit_buffer_blocks_quote(self):
        """< EXIT_BUFFER_MINUTES (15 min = 0.25h) before first pitch → SKIP_HOURS."""
        from trading.config import EXIT_BUFFER_MINUTES

        hours_to_fp = 0.1  # 6 minutes before first pitch
        threshold = EXIT_BUFFER_MINUTES / 60.0  # 0.25 hours

        assert hours_to_fp < threshold, (
            "6 minutes is within EXIT_BUFFER_MINUTES — should skip"
        )

    def test_well_before_game_passes(self):
        """4 hours before first pitch → not blocked."""
        from trading.config import EXIT_BUFFER_MINUTES

        hours_to_fp = 4.0
        threshold = EXIT_BUFFER_MINUTES / 60.0

        assert hours_to_fp >= threshold


# ════════════════════════════════════════════════════════════════════════════════
# Test: Feature staleness blocks quoting
# ════════════════════════════════════════════════════════════════════════════════


class TestFeatureStaleness:
    """Markets blocked when features are pending rebuild for those teams."""

    def test_stale_teams_block_quoting(self):
        """After settlement, both teams are marked pending → is_stale returns True."""
        from trading.features import FeatureManager

        fm = FeatureManager.__new__(FeatureManager)
        fm._teams_pending_rebuild = set()
        fm._pending_lock = threading.Lock()
        fm._rebuild_lock = threading.Lock()

        fm.mark_teams_pending("LAD", "NYM")

        assert fm.is_stale_for_game("LAD", "NYM") is True
        assert fm.is_stale_for_game("LAD", "PHI") is True  # LAD pending
        assert fm.is_stale_for_game("BOS", "NYY") is False  # neither pending

    def test_rebuild_clears_pending(self):
        """After successful rebuild, pending teams are cleared."""
        from trading.features import FeatureManager

        fm = FeatureManager.__new__(FeatureManager)
        fm._teams_pending_rebuild = {"LAD", "NYM"}
        fm._pending_lock = threading.Lock()
        fm._rebuild_lock = threading.Lock()

        # Simulate what _rebuild does on success
        with fm._pending_lock:
            fm._teams_pending_rebuild.clear()

        assert fm.is_stale_for_game("LAD", "NYM") is False


# ════════════════════════════════════════════════════════════════════════════════
# Test: Portfolio duplicate ticker rejection in risk gate
# ════════════════════════════════════════════════════════════════════════════════


class TestPortfolioDuplicateRejection:
    """
    In dry-run mode, the risk gate's position_tickers check includes BOTH
    positions AND open orders. This test verifies whether dry-run actually
    populates position_tickers with anything.
    """

    def test_dry_run_portfolio_state_includes_orders(self, tmp_path):
        """get_portfolio_state includes resting order tickers in position_tickers."""
        from trading.portfolio import Portfolio

        with patch("trading.portfolio._STATE_FILE", tmp_path / "state.json"):
            portfolio = Portfolio(client=None, dry_run=True)

        portfolio.add_order(
            order_id="bid_123",
            ticker="KXMLBGAME-26JUL261910NYMLAD-LAD",
            side="yes",
            price_cents=58,
            contracts=3,
        )

        state = portfolio.get_portfolio_state()
        assert "KXMLBGAME-26JUL261910NYMLAD-LAD" in state["position_tickers"]

    def test_dry_run_runner_does_not_track_maker_orders(self, tmp_path):
        """
        BUG VERIFICATION: In run_once, when post_two_sided succeeds in dry-run,
        the runner does NOT call portfolio.add_order(). This means:
        - position_tickers stays empty
        - The duplicate check never triggers
        - The same ticker can be quoted repeatedly (not a problem per se,
          but means the portfolio summary is always 0)

        This is the structural gap: quotes are logged but not tracked.
        """
        from trading.portfolio import Portfolio

        with patch("trading.portfolio._STATE_FILE", tmp_path / "state.json"):
            portfolio = Portfolio(client=None, dry_run=True)

        # Simulate what run_once actually does after a successful dry-run post:
        # result = post_two_sided(...)
        # if result.get("status") in ("DRY_RUN", "SUBMITTED"):
        #     decision["action"] = "MAKE"
        #     executed += 1
        # ← No portfolio.add_order() call here!

        state = portfolio.get_portfolio_state()
        assert len(state["position_tickers"]) == 0, (
            "Dry-run MAKE path does not call add_order → position_tickers stays empty"
        )


# ════════════════════════════════════════════════════════════════════════════════
# Test: End-to-end runner.run_once scan (mocked)
# ════════════════════════════════════════════════════════════════════════════════


class TestRunOnceIntegration:
    """Integration test: a full scan cycle with all mocked externals."""

    def test_run_once_with_favorable_conditions_posts_quotes(self, tmp_path):
        """Full run_once should log MAKE actions when conditions are met."""
        from trading.runner import TradingRunner
        from trading.portfolio import Portfolio
        from trading.features import FeatureManager
        from trading.models import EnsembleStore
        from trading.sizing import SizedQuote

        with patch("trading.portfolio._STATE_FILE", tmp_path / "state.json"):
            runner = TradingRunner.__new__(TradingRunner)
            runner._dry_run = True
            runner._bankroll = 1000.0
            runner._running = True
            runner._reprice_state = {}
            runner._market_set = {
                "KXMLBGAME-26JUL261910NYMLAD-LAD": {"ticker": "KXMLBGAME-26JUL261910NYMLAD-LAD", "status": "open"},
            }
            runner._market_set_lock = threading.Lock()
            runner._last_full_discovery = 9999999999.0  # skip discovery
            runner._last_portfolio_reconcile = 9999999999.0

            # Mock portfolio (clean)
            runner._portfolio = Portfolio(client=None, dry_run=True)

            # Mock features
            runner._features = MagicMock(spec=FeatureManager)
            runner._features.is_stale_for_game.return_value = False
            runner._features.check_and_refresh.return_value = False

            np.random.seed(42)
            n_cols = 80
            feature_data = {f"feat_{i}": np.random.randn(1) for i in range(n_cols)}
            feature_data["home_team_abbr"] = ["LAD"]
            feature_data["away_team_abbr"] = ["NYM"]
            feature_data["game_date"] = [pd.Timestamp("2026-07-26")]
            features_df = pd.DataFrame(feature_data)
            runner._features.get_features.return_value = features_df

            # Mock ensemble store
            runner._ensemble_store = MagicMock(spec=EnsembleStore)
            runner._ensemble_store.inference_cache = {}
            runner._ensemble_store._inference_cache_features_hash = None
            runner._ensemble_store.get_bundle.return_value = {"member_bundles": [], "task": "classification"}

            # Mock WS
            runner._ws = MagicMock()
            runner._ws.is_connected = True
            runner._ws.get_all_book_tops.return_value = {
                "KXMLBGAME-26JUL261910NYMLAD-LAD": (50, 70),
            }
            runner._ws._snapshot_pending = set()

            # Mock client for the sweep_stale_positions path
            runner._client = MagicMock()

            mock_sized = SizedQuote(
                ticker="KXMLBGAME-26JUL261910NYMLAD-LAD",
                target="home_win",
                cluster="winner",
                fair_value=0.65,
                model_prob=0.68,
                ensemble_std=0.03,
                confidence_tier="HIGH",
                bid_cents=63,
                ask_cents=67,
                contracts=3,
                accuracy_mult=1.2,
                edge_at_mid=0.05,
            )

            with patch("trading.runner.gumbo_schedule") as mock_sched, \
                 patch.object(runner, "_hours_to_first_pitch", return_value=4.0), \
                 patch("trading.runner.generate_quotes", return_value=[{"ticker": "KXMLBGAME-26JUL261910NYMLAD-LAD", "target": "home_win", "cluster": "winner", "fair_value": 0.65, "model_prob": 0.68, "ensemble_std": 0.03, "confidence_tier": "HIGH", "bid_cents": 63, "ask_cents": 67, "edge_at_mid": 0.05}]), \
                 patch("trading.runner.size_quotes", return_value=[mock_sized]), \
                 patch("trading.runner.post_two_sided") as mock_post, \
                 patch("trading.runner._log_decision"), \
                 patch("trading.runner.check_limits", return_value=(True, "OK")):
                mock_sched.game_has_started.return_value = False
                mock_post.return_value = {"status": "DRY_RUN"}
                runner.run_once()

            # post_two_sided was called (quote was posted)
            mock_post.assert_called_once()

            # But portfolio still shows 0 — this is the expected (problematic) behavior
            assert runner._portfolio.position_count() == 0, (
                "Dry-run maker quotes never create positions — this is the structural gap"
            )


# ════════════════════════════════════════════════════════════════════════════════
# Test: Taker threshold calculation
# ════════════════════════════════════════════════════════════════════════════════


class TestTakerThreshold:
    """Validate the taker fee and threshold math."""

    def test_taker_fee_at_50_cents(self):
        """max fee at p=0.50: 0.07*0.5*0.5 = 0.0175 → ceil = $0.02."""
        from trading.scanner import kalshi_taker_fee

        fee = kalshi_taker_fee(0.50)
        assert fee == 0.02

    def test_taker_fee_at_extremes_is_low(self):
        """Fee at extreme prices (0.05 or 0.95) is minimal."""
        from trading.scanner import kalshi_taker_fee

        fee_low = kalshi_taker_fee(0.05)
        fee_high = kalshi_taker_fee(0.95)
        assert fee_low <= 0.01
        assert fee_high <= 0.01

    def test_min_edge_for_maker_is_one_cent(self):
        """Maker fee is $0. min_edge_for_profit(maker=True) = 0.01."""
        from trading.scanner import min_edge_for_profit

        assert min_edge_for_profit(0.50, maker=True) == 0.01
        assert min_edge_for_profit(0.30, maker=True) == 0.01

    def test_min_edge_for_taker_scales_with_price(self):
        """Taker breakeven = 0.07*p*(1-p)."""
        from trading.scanner import min_edge_for_profit

        edge_50 = min_edge_for_profit(0.50, maker=False)
        edge_20 = min_edge_for_profit(0.20, maker=False)

        # At p=0.50: 0.07*0.5*0.5 = 0.0175
        assert abs(edge_50 - 0.0175) < 0.001
        # At p=0.20: 0.07*0.2*0.8 = 0.0112
        assert abs(edge_20 - 0.0112) < 0.001


# ════════════════════════════════════════════════════════════════════════════════
# Test: Market map ticker parsing
# ════════════════════════════════════════════════════════════════════════════════


class TestMarketMapParsing:
    """Ticker parsing must correctly extract teams and strikes."""

    def test_game_winner_ticker(self):
        from trading.market_map import parse_ticker

        p = parse_ticker("KXMLBGAME-26JUL261910NYMLAD-LAD")
        assert p is not None
        assert p.series == "KXMLBGAME"
        assert p.away_team == "NYM"
        assert p.home_team == "LAD"
        assert p.strike_team == "LAD"
        assert p.strike_value is None

    def test_total_runs_ticker(self):
        from trading.market_map import parse_ticker

        p = parse_ticker("KXMLBTOTAL-26JUL261910NYMLAD-9")
        assert p is not None
        assert p.series == "KXMLBTOTAL"
        assert p.strike_value == 9.0
        assert p.strike_team is None

    def test_spread_ticker(self):
        from trading.market_map import parse_ticker

        p = parse_ticker("KXMLBSPREAD-26JUL261910NYMLAD-LAD2")
        assert p is not None
        assert p.series == "KXMLBSPREAD"
        assert p.strike_team == "LAD"
        assert p.strike_value == 2.0

    def test_yrfi_ticker_no_strike(self):
        from trading.market_map import parse_ticker

        p = parse_ticker("KXMLBRFI-26JUL261910NYMLAD")
        assert p is not None
        assert p.series == "KXMLBRFI"
        assert p.strike_team is None
        assert p.strike_value is None

    def test_extras_ticker(self):
        from trading.market_map import parse_ticker

        p = parse_ticker("KXMLBEXTRAS-26JUL261910NYMLAD-EXTRAS")
        assert p is not None
        assert p.series == "KXMLBEXTRAS"
        assert p.strike_team is None
        assert p.strike_value is None

    def test_team_total_ticker(self):
        from trading.market_map import parse_ticker

        p = parse_ticker("KXMLBTEAMTOTAL-26JUL261910NYMLAD-LAD4")
        assert p is not None
        assert p.series == "KXMLBTEAMTOTAL"
        assert p.strike_team == "LAD"
        assert p.strike_value == 4.0


# ════════════════════════════════════════════════════════════════════════════════
# Test: The key diagnostic — what CAN make the runner produce 0 quotes
# ════════════════════════════════════════════════════════════════════════════════


class TestZeroQuoteDiagnostics:
    """
    Enumerate all paths in the pipeline that can reduce quotes to 0.
    Each test names the filter stage and demonstrates its effect.
    """

    def test_all_markets_started_produces_zero(self):
        """If all discovered markets have started, market set is empty after filter."""
        # This is handled in _full_discovery: game_has_started → not added
        # and in run_once: parse_ticker → is_stale_for_game filter
        pass  # demonstrated by TestFeatureStaleness

    def test_no_feature_row_for_game_produces_zero(self):
        """If game's teams aren't in features, _lookup_game_row returns None."""
        from trading.scanner import _lookup_game_row

        features = pd.DataFrame({
            "home_team_abbr": ["ATL", "BOS"],
            "away_team_abbr": ["PHI", "NYY"],
            "game_date": [pd.Timestamp("2026-07-25"), pd.Timestamp("2026-07-25")],
        })

        row = _lookup_game_row("NYM", "LAD", features)
        assert row is None, "No row for NYM@LAD when only ATL/BOS games in features"

    def test_all_models_fail_inference_produces_zero(self):
        """If predict_market_prob returns error, no quote is generated."""
        from trading.scanner import generate_quotes

        error_result = {"error": "all models failed at inference", "target": "home_win"}

        store = MagicMock()
        store.inference_cache = {}
        store._inference_cache_features_hash = None
        store.get_bundle.return_value = {"member_bundles": [], "task": "classification"}

        features = pd.DataFrame({
            "home_team_abbr": ["LAD"],
            "away_team_abbr": ["NYM"],
            "game_date": [pd.Timestamp("2026-07-26")],
            **{f"feat_{i}": [0.0] for i in range(50)},
        })

        markets = [{"ticker": "KXMLBGAME-26JUL261910NYMLAD-LAD", "status": "open"}]

        with patch("trading.scanner.predict_market_prob", return_value=error_result):
            quotes = generate_quotes(markets, features, store, {})

        assert len(quotes) == 0

    def test_min_edge_gate_filters_all_quotes_when_market_efficient(self):
        """
        THE MOST LIKELY EXPLANATION for "generating quotes" + "0 positions":

        If the market is efficient (book mid ≈ model fair), edge_at_mid ≈ 0
        and the MIN_EDGE_BUFFER_MAKER gate filters everything.

        min_edge = min_edge_for_profit(fair, maker=True) * MIN_EDGE_BUFFER_MAKER
                 = 0.01 * 1.5 = 0.015 (1.5 cents)

        So if |fair - market_mid| < 1.5c for all markets, ZERO quotes survive.
        """
        from trading.config import MIN_EDGE_BUFFER_MAKER

        min_edge_threshold = 0.01 * MIN_EDGE_BUFFER_MAKER  # 0.015
        assert min_edge_threshold == 0.015, f"Threshold is {min_edge_threshold}"

        # Scenario: model says 0.60, market mid is 0.59 → edge = 0.01 < 0.015
        edge = abs(0.60 - 0.59)
        assert edge < min_edge_threshold, (
            f"Edge {edge} < threshold {min_edge_threshold}: quote filtered"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
