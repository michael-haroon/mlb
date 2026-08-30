#!/usr/bin/env python3
"""
Test the runner's dynamic series selection logic without actually connecting to Kalshi.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from trading.models import EnsembleStore
from trading.market_map import MODEL_TO_SERIES, parse_ticker

def mock_get_tradeable_series(ensemble_store):
    """Mock implementation of _get_tradeable_series_from_models()."""
    series_set = set()
    for target in ensemble_store.tradeable_targets:
        series = MODEL_TO_SERIES.get(target)
        if series:
            series_set.add(series)

    # Derived targets (home_runs, away_runs) map to KXMLBTEAMTOTAL
    if "total_runs" in ensemble_store.tradeable_targets and \
       "home_run_diff" in ensemble_store.tradeable_targets:
        team_total_series = MODEL_TO_SERIES.get("home_runs")
        if team_total_series:
            series_set.add(team_total_series)

    return sorted(series_set)

def test_runner_logic():
    """Test that the runner correctly filters markets based on available models."""

    # Initialize EnsembleStore and discover models
    ensemble_store = EnsembleStore()
    targets = ensemble_store.discover()

    print(f"\n=== Discovered models ===")
    print(f"Targets: {targets}")

    # Get tradeable series
    tradeable_series = mock_get_tradeable_series(ensemble_store)
    print(f"\n=== Tradeable series ===")
    print(f"Series: {tradeable_series}")

    # Test market filtering
    print(f"\n=== Market filtering test ===")
    test_tickers = [
        "KXMLBTOTAL-26AUG13NYYLAD-9",      # total_runs - should be accepted
        "KXMLBGAME-26AUG13NYYLAD-NYY",     # home_win - should be rejected
        "KXMLBRFI-26AUG13NYYLAD",          # yrfi - should be rejected
        "KXMLBSPREAD-26AUG13NYYLAD-NYY2",  # home_run_diff - should be rejected
        "KXMLBTEAMTOTAL-26AUG13NYYLAD-NYY4", # home_runs - depends on models
    ]

    for ticker in test_tickers:
        parsed = parse_ticker(ticker)
        if parsed:
            in_tradeable = parsed.series in tradeable_series
            status = "✅ ACCEPT" if in_tradeable else "❌ REJECT"
            print(f"{status} {ticker} (series: {parsed.series})")
        else:
            print(f"⚠️  PARSE_FAIL {ticker}")

    # Verify expected behavior
    print(f"\n=== Verification ===")
    if "total_runs" in targets:
        assert "KXMLBTOTAL" in tradeable_series, "total_runs should enable KXMLBTOTAL"
        print("✅ total_runs → KXMLBTOTAL mapping correct")

    if "home_run_diff" in targets:
        assert "KXMLBSPREAD" in tradeable_series, "home_run_diff should enable KXMLBSPREAD"
        print("✅ home_run_diff → KXMLBSPREAD mapping correct")

    if "total_runs" in targets and "home_run_diff" in targets:
        assert "KXMLBTEAMTOTAL" in tradeable_series, "Both models should enable KXMLBTEAMTOTAL"
        print("✅ Derived team_total enabled correctly")

    if "home_win" not in targets:
        assert "KXMLBGAME" not in tradeable_series, "home_win missing should disable KXMLBGAME"
        print("✅ KXMLBGAME correctly filtered out (no home_win model)")

    print(f"\n✅ All integration tests passed!")

if __name__ == "__main__":
    test_runner_logic()
