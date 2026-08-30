#!/usr/bin/env python3
"""
Quick test to verify dynamic series selection works correctly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from trading.models import EnsembleStore
from trading.market_map import MODEL_TO_SERIES

def test_series_mapping():
    """Verify that we correctly map model targets to Kalshi series."""

    # Test 1: Only total_runs model
    print("\n=== Test 1: total_runs only ===")
    mock_targets = ["total_runs"]
    series_set = set()
    for target in mock_targets:
        series = MODEL_TO_SERIES.get(target)
        if series:
            series_set.add(series)
    print(f"Models: {mock_targets}")
    print(f"Expected series: ['KXMLBTOTAL']")
    print(f"Actual series: {sorted(series_set)}")
    assert sorted(series_set) == ["KXMLBTOTAL"], "total_runs should map to KXMLBTOTAL"

    # Test 2: total_runs + home_run_diff (enables team totals)
    print("\n=== Test 2: total_runs + home_run_diff ===")
    mock_targets = ["total_runs", "home_run_diff"]
    series_set = set()
    for target in mock_targets:
        series = MODEL_TO_SERIES.get(target)
        if series:
            series_set.add(series)
    # Check derived target
    if "total_runs" in mock_targets and "home_run_diff" in mock_targets:
        team_total_series = MODEL_TO_SERIES.get("home_runs")
        if team_total_series:
            series_set.add(team_total_series)
    print(f"Models: {mock_targets}")
    print(f"Expected series: ['KXMLBSPREAD', 'KXMLBTEAMTOTAL', 'KXMLBTOTAL']")
    print(f"Actual series: {sorted(series_set)}")
    assert "KXMLBTOTAL" in series_set, "Should have KXMLBTOTAL"
    assert "KXMLBSPREAD" in series_set, "Should have KXMLBSPREAD"
    assert "KXMLBTEAMTOTAL" in series_set, "Should have KXMLBTEAMTOTAL (derived)"

    # Test 3: All classification targets
    print("\n=== Test 3: All classification targets ===")
    mock_targets = ["home_win", "yrfi", "extra_innings"]
    series_set = set()
    for target in mock_targets:
        series = MODEL_TO_SERIES.get(target)
        if series:
            series_set.add(series)
    print(f"Models: {mock_targets}")
    print(f"Expected series: ['KXMLBEXTRAS', 'KXMLBGAME', 'KXMLBRFI']")
    print(f"Actual series: {sorted(series_set)}")
    assert sorted(series_set) == ["KXMLBEXTRAS", "KXMLBGAME", "KXMLBRFI"]

    # Test 4: Check MODEL_TO_SERIES mapping is complete
    print("\n=== Test 4: MODEL_TO_SERIES completeness ===")
    print(f"MODEL_TO_SERIES = {MODEL_TO_SERIES}")
    assert MODEL_TO_SERIES.get("total_runs") == "KXMLBTOTAL"
    assert MODEL_TO_SERIES.get("home_win") == "KXMLBGAME"
    assert MODEL_TO_SERIES.get("yrfi") == "KXMLBRFI"
    assert MODEL_TO_SERIES.get("extra_innings") == "KXMLBEXTRAS"
    assert MODEL_TO_SERIES.get("home_run_diff") == "KXMLBSPREAD"
    assert MODEL_TO_SERIES.get("home_runs") == "KXMLBTEAMTOTAL"
    assert MODEL_TO_SERIES.get("away_runs") == "KXMLBTEAMTOTAL"

    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_series_mapping()
