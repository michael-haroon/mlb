"""
pregame/trading/scanner.py
--------------------------
Multi-market signal generation for MLB pregame trading.

For each open Kalshi market, computes:
1. Model probability (from ensemble inference)
2. Conservative fair value (shaded by confidence)
3. Two-sided quote (bid + ask around fair)
4. Edge vs current market midpoint

The scanner does NOT decide whether to trade — it produces quotes that
sizing.py and risk.py will filter and approve.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    HALF_SPREAD_CENTS, SHADE_SIGMA, PRICE_FLOOR, PRICE_CEILING,
    MIN_EDGE_BUFFER_MAKER, EXPECTED_PRED_STD, MIN_SHARPNESS_RATIO,
)
from .market_map import (
    MODEL_TO_SERIES, ParsedTicker, parse_ticker, classify_cluster,
)
from .models import predict_market_prob, predict_derived_team_total, EnsembleStore

logger = logging.getLogger(__name__)

# Tracks which targets have been flagged as sharpness-collapsed this session
_sharpness_halted: set[str] = set()


def kalshi_taker_fee(price: float) -> float:
    """Taker fee per contract in dollars: ceil(0.07 * P * (1-P))."""
    raw = 0.07 * price * (1 - price)
    return np.ceil(raw * 100) / 100


def min_edge_for_profit(price: float, maker: bool = True) -> float:
    """Minimum edge needed to breakeven after fees.

    Maker fee is $0 on Kalshi for event contracts — edge just needs to
    be positive. But we still require a buffer (MIN_EDGE_BUFFER_MAKER)
    to avoid noise trades.

    Taker fee = 0.07 * P * (1-P), breakeven edge = fee / (1-P) = 0.07*P.
    """
    if maker:
        # Maker is free, but we want real signal, not noise.
        # Require at least 1 cent of edge (= 1% probability point).
        return 0.01
    return 0.07 * price * (1 - price)


def conservative_fair_value(
    model_prob: float,
    ensemble_std: float,
    confidence_tier: str,
) -> float:
    """Compute fair value shaded conservatively by model uncertainty.

    Shades TOWARD 0.5 (less extreme): if model says 0.62, shade to 0.60
    for HIGH confidence or 0.56 for LOW confidence.

    This ensures that even at the worst case of our confidence interval,
    our quote still has positive expected value.
    """
    shade = SHADE_SIGMA.get(confidence_tier, 1.0) * ensemble_std

    # Shade toward 0.5: reduce the distance from 0.5
    if model_prob > 0.5:
        fair = model_prob - shade
    else:
        fair = model_prob + shade

    return np.clip(fair, 0.01, 0.99)


def _lookup_game_row(
    away_team: str,
    home_team: str,
    features: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Return the single feature row for this matchup.

    Matches on home_team_abbr / away_team_abbr using the most recent row
    for this pair (features are stale but cover today's games via the last
    known ratings snapshot — the exact date need not match today).
    Returns None if no row found.
    """
    # Features store abbreviated team codes; Kalshi standard codes match these directly.
    mask = (
        (features["home_team_abbr"] == home_team) &
        (features["away_team_abbr"] == away_team)
    )
    rows = features[mask]
    if rows.empty:
        return None
    # Use the most recent row for this matchup (closest prior game = freshest ratings)
    return rows.sort_values("game_date").iloc[[-1]]


def _check_batch_sharpness(quotes: list[dict]) -> None:
    """Flag targets whose prediction std across the batch is dangerously low.

    A well-functioning model should produce varied probabilities across games.
    If all predictions for a target are within a narrow band around 0.5, the
    model has lost discriminative power (likely feature distribution shift).
    """
    from collections import defaultdict
    target_probs: dict[str, list[float]] = defaultdict(list)
    for q in quotes:
        target_probs[q["target"]].append(q["model_prob"])

    for target, probs in target_probs.items():
        if target not in EXPECTED_PRED_STD or len(probs) < 3:
            continue
        live_std = np.std(probs)
        expected = EXPECTED_PRED_STD[target]
        if live_std < expected * MIN_SHARPNESS_RATIO:
            _sharpness_halted.add(target)
            logger.warning(
                f"SHARPNESS COLLAPSE: {target} live_std={live_std:.4f} "
                f"< {MIN_SHARPNESS_RATIO:.0%} of expected {expected:.4f}. "
                f"Halting {len(probs)} quotes for this target."
            )


def generate_quotes(
    markets: list[dict],
    features: pd.DataFrame,
    ensemble_store: EnsembleStore,
    book_tops: dict[str, tuple[Optional[int], Optional[int]]],
) -> list[dict]:
    """Generate two-sided quotes for all open markets.

    Runs inference once per (game, target) pair — not once per market — to
    avoid re-running the full ensemble on every total/spread line for the
    same game.

    Args:
        markets: list of Kalshi market dicts (from API or discovery)
        features: game_features.parquet (full history; filtered per game here)
        ensemble_store: loaded EnsembleStore with all target ensembles
        book_tops: {ticker: (best_bid_cents, best_ask_cents)} from WS/REST

    Returns:
        list of quote dicts ready for sizing.size_quotes()
    """
    # Pre-compute inference results keyed by (away_team, home_team, target).
    # This ensures we call predict_game once per game×target, not once per market.
    _result_cache: dict[tuple, dict] = {}

    quotes = []

    for market in markets:
        ticker = market["ticker"]
        parsed = parse_ticker(ticker)
        if parsed is None:
            continue

        series = parsed.series
        cluster = classify_cluster(series)

        game_row = _lookup_game_row(parsed.away_team, parsed.home_team, features)
        if game_row is None:
            logger.debug(f"No feature row for {parsed.away_team}@{parsed.home_team}, skipping {ticker}")
            continue

        quote = _price_market(parsed, game_row, ensemble_store, book_tops, _result_cache)
        if quote is None:
            continue

        quote["cluster"] = cluster
        quotes.append(quote)

    # Batch sharpness check: verify prediction std across games per target.
    # If predictions are all clustered near 0.5, the model has lost signal.
    _check_batch_sharpness(quotes)
    quotes = [q for q in quotes if q["target"] not in _sharpness_halted]

    logger.info(f"Generated {len(quotes)} quotes from {len(markets)} markets")
    return quotes


def _apply_line(base_result: dict, target: str, line: Optional[float], direction: str) -> dict:
    """Apply a specific line/direction to a cached inference result.

    For classification targets the probability is already final (no line needed).
    For regression targets we re-integrate the cached distribution at
    the requested line — much cheaper than re-running the full ensemble.
    Dispatches on distribution type: NegBin for counts, Student-t for signed.
    """
    from ..strategy.calibration import cover_probability, negbin_cover_probability

    task = base_result.get("task", "classification")

    if task == "classification":
        return {
            "prob": base_result["prob"],
            "ensemble_std": base_result["ensemble_std"],
            "confidence_tier": base_result["confidence_tier"],
            "task": task,
            "n_models_used": base_result["n_models_used"],
        }

    # Regression: re-integrate the cached distribution at the new line
    if line is None:
        return {"error": f"Regression target {target} requires a line"}

    dist = base_result.get("distribution", {})
    mu = float(base_result.get("point_estimate", dist.get("mu", 0)))
    dist_type = dist.get("type", "student_t")

    if dist_type == "negbin":
        alpha = dist["alpha"]
        prob = negbin_cover_probability(mu, line, alpha, direction=direction)
        return {
            "prob": prob,
            "ensemble_std": base_result["ensemble_std"],
            "confidence_tier": base_result["confidence_tier"],
            "task": task,
            "n_models_used": base_result["n_models_used"],
            "point_estimate": mu,
        }

    # Student-t fallback
    df    = dist.get("df", base_result.get("residual_df", 7))
    scale = dist.get("scale", base_result.get("residual_scale", 1.0))

    prob = cover_probability(mu, line, df, scale, direction=direction)
    return {
        "prob": prob,
        "ensemble_std": base_result["ensemble_std"],
        "confidence_tier": base_result["confidence_tier"],
        "task": task,
        "n_models_used": base_result["n_models_used"],
        "point_estimate": mu,
        "residual_df": df,
        "residual_scale": scale,
    }


def _price_market(
    parsed: ParsedTicker,
    game_row: pd.DataFrame,
    ensemble_store: EnsembleStore,
    book_tops: dict[str, tuple[Optional[int], Optional[int]]],
    result_cache: dict,
) -> Optional[dict]:
    """Price a single market using the appropriate model target.

    result_cache is keyed by (away_team, home_team, target) and avoids
    re-running inference for multiple markets on the same game×target.
    """
    series = parsed.series
    ticker = parsed.raw

    # Map series to model target and determine pricing params
    target, line, direction = _resolve_target(parsed)
    if target is None:
        return None

    # Run inference once per (game, target); subsequent markets reuse cached result
    cache_key = (parsed.away_team, parsed.home_team, target)
    if cache_key not in result_cache:
        if target in ("home_runs", "away_runs"):
            total_bundle = ensemble_store.get_bundle("total_runs")
            diff_bundle  = ensemble_store.get_bundle("home_run_diff")
            if total_bundle is None or diff_bundle is None:
                logger.debug(f"Missing total_runs or home_run_diff ensemble, skipping {ticker}")
                return None
            result_cache[cache_key] = predict_derived_team_total(
                target, game_row, total_bundle, diff_bundle, line=None, direction="over"
            )
        else:
            bundle = ensemble_store.get_bundle(target)
            if bundle is None:
                logger.debug(f"No ensemble for target {target}, skipping {ticker}")
                return None
            result_cache[cache_key] = predict_market_prob(
                target, game_row, bundle, line=None, direction="over"
            )

    base_result = result_cache[cache_key]
    if "error" in base_result:
        return None

    # Re-compute prob for this market's specific line/direction from cached distribution
    result = _apply_line(base_result, target, line, direction)
    if "error" in result:
        logger.debug(f"Prediction failed for {ticker}: {result['error']}")
        return None

    model_prob = result["prob"]
    ensemble_std = result["ensemble_std"]
    confidence_tier = result["confidence_tier"]

    if target in _sharpness_halted:
        return None

    # Compute conservative fair value
    fair = conservative_fair_value(model_prob, ensemble_std, confidence_tier)

    # Price filter: skip extreme prices where we have no edge
    if fair < PRICE_FLOOR or fair > PRICE_CEILING:
        return None

    # Compute bid/ask around fair value
    # Spread width inversely proportional to model strength:
    # strong models → tight spread, weak models → wide spread
    half_spread = HALF_SPREAD_CENTS.get(confidence_tier, 3)

    fair_cents = int(round(fair * 100))
    bid_cents = max(1, fair_cents - half_spread)
    ask_cents = min(99, fair_cents + half_spread)

    # The ask on Kalshi is posted as a NO buy at (100 - ask_cents)
    no_buy_cents = 100 - ask_cents

    # Compute edge vs current market midpoint
    bb, ba = book_tops.get(ticker, (None, None))
    if bb is not None and ba is not None:
        market_mid = (bb + ba) / 200.0  # convert cents to probability
        edge_at_mid = abs(fair - market_mid)
    else:
        # No book data yet — use our fair as reference, edge = half-spread
        edge_at_mid = half_spread / 100.0

    # Minimum edge gate: even with zero maker fee, require meaningful signal
    min_edge = min_edge_for_profit(fair, maker=True) * MIN_EDGE_BUFFER_MAKER
    if edge_at_mid < min_edge:
        return None

    return {
        "ticker": ticker,
        "target": target,
        "fair_value": fair,
        "model_prob": model_prob,
        "ensemble_std": ensemble_std,
        "confidence_tier": confidence_tier,
        "bid_cents": bid_cents,
        "ask_cents": ask_cents,
        "no_buy_cents": no_buy_cents,  # what we actually post as ask
        "edge_at_mid": edge_at_mid,
        "line": line,
        "direction": direction,
    }


def _resolve_target(parsed: ParsedTicker) -> tuple[Optional[str], Optional[float], str]:
    """Map a parsed ticker to (model_target, line, direction).

    Returns (None, None, "") if the market can't be priced by our models.
    """
    series = parsed.series

    if series == "KXMLBGAME":
        # Winner market: classification target, no line needed
        # If strike_team == home_team → P(home_win) directly
        # If strike_team == away_team → P(away_win) = 1 - P(home_win)
        return "home_win", None, "over"

    elif series == "KXMLBRFI":
        return "yrfi", None, "over"

    elif series == "KXMLBEXTRAS":
        return "extra_innings", None, "over"

    elif series == "KXMLBSPREAD":
        # Spread: "TeamN" means "team wins by N or more"
        if parsed.strike_value is None:
            return None, None, ""
        if parsed.strike_team == parsed.home_team:
            # Home wins by N+ → P(home_run_diff >= N)
            return "home_run_diff", parsed.strike_value, "over"
        else:
            # Away wins by N+ → P(home_run_diff <= -N)
            return "home_run_diff", -parsed.strike_value, "under"

    elif series == "KXMLBTOTAL":
        # Total runs over/under: P(total_runs >= N)
        if parsed.strike_value is None:
            return None, None, ""
        return "total_runs", parsed.strike_value, "over"

    elif series == "KXMLBTEAMTOTAL":
        # Team total: "TeamN" means "team scores N or more"
        if parsed.strike_value is None:
            return None, None, ""
        if parsed.strike_team == parsed.home_team:
            return "home_runs", parsed.strike_value, "over"
        else:
            return "away_runs", parsed.strike_value, "over"

    return None, None, ""
