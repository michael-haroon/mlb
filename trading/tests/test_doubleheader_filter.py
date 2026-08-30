"""Tests for doubleheader game-2 filtering.

Verifies:
1. parse_ticker extracts ticker_time from current-format tickers
2. parse_ticker returns ticker_time=None for legacy-format tickers
3. get_game_number returns 1 for single-game days
4. get_game_number disambiguates doubleheaders by time
5. get_game_number returns None when it can't disambiguate

Run: conda run -n pred python -m pytest pregame/trading/tests/test_doubleheader_filter.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timezone

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from trading.market_map import parse_ticker
from trading import schedule as gumbo_schedule


# ─── parse_ticker: ticker_time extraction ────────────────────────────────────

class TestParseTickerTime:
    def test_current_format_extracts_time(self):
        """Current format with HHMM should populate ticker_time."""
        result = parse_ticker("KXMLBTOTAL-26JUL211907TBTOR-9")
        assert result is not None
        assert result.ticker_time == "1907"
        assert result.away_team == "TBR"
        assert result.home_team == "TOR"

    def test_current_format_1300(self):
        result = parse_ticker("KXMLBGAME-26JUL051300PITWSH-PIT")
        assert result is not None
        assert result.ticker_time == "1300"
        assert result.away_team == "PIT"
        assert result.home_team == "WSH"

    def test_legacy_format_no_time(self):
        """Legacy format without HHMM should have ticker_time=None."""
        result = parse_ticker("KXMLBGAME-26JUL03NYMLAD-NYM")
        assert result is not None
        assert result.ticker_time is None
        assert result.away_team == "NYM"
        assert result.home_team == "LAD"

    def test_doubleheader_g2_suffix(self):
        """G2 suffix stripped from teams, time still extracted."""
        result = parse_ticker("KXMLBTOTAL-26JUL071945MILSTLG2-8")
        assert result is not None
        assert result.ticker_time == "1945"
        assert result.away_team == "MIL"
        assert result.home_team == "STL"

    def test_doubleheader_g1_suffix(self):
        result = parse_ticker("KXMLBGAME-26JUL071310MILSTLG1-MIL")
        assert result is not None
        assert result.ticker_time == "1310"
        assert result.away_team == "MIL"
        assert result.home_team == "STL"


# ─── get_game_number: single game ───────────────────────────────────────────

def _make_gumbo_game(away_abbr, home_abbr, game_date_utc, game_number=1):
    """Build a minimal GUMBO game dict."""
    return {
        "teams": {
            "away": {"team": {"abbreviation": away_abbr}},
            "home": {"team": {"abbreviation": home_abbr}},
        },
        "gameDate": game_date_utc,
        "gameNumber": game_number,
    }


class TestGetGameNumberSingle:
    @patch.object(gumbo_schedule, "_get_cached_schedule")
    def test_single_game_returns_1(self, mock_sched):
        """Single game for a matchup on a date → game_number = 1."""
        mock_sched.return_value = [
            _make_gumbo_game("TB", "TOR", "2026-07-21T23:07:00Z", game_number=1),
            _make_gumbo_game("NYM", "ATL", "2026-07-21T23:10:00Z", game_number=1),
        ]
        result = gumbo_schedule.get_game_number("TBR", "TOR", "2026-07-21", "1907")
        assert result == 1

    @patch.object(gumbo_schedule, "_get_cached_schedule")
    def test_no_matching_game_returns_none(self, mock_sched):
        """No games matching these teams → None."""
        mock_sched.return_value = [
            _make_gumbo_game("NYM", "ATL", "2026-07-21T23:10:00Z"),
        ]
        result = gumbo_schedule.get_game_number("TBR", "TOR", "2026-07-21", "1907")
        assert result is None


# ─── get_game_number: doubleheader disambiguation ────────────────────────────

class TestGetGameNumberDoubleheader:
    @patch.object(gumbo_schedule, "_get_cached_schedule")
    def test_game1_identified_by_time(self, mock_sched):
        """Ticker time matching game 1 (earlier) returns 1."""
        # Game 1: 1:10 PM ET = 17:10 UTC
        # Game 2: 7:45 PM ET = 23:45 UTC
        mock_sched.return_value = [
            _make_gumbo_game("MIL", "STL", "2026-07-07T17:10:00Z", game_number=1),
            _make_gumbo_game("MIL", "STL", "2026-07-07T23:45:00Z", game_number=2),
        ]
        # Ticker with 1310 ET → 1710 UTC → matches game 1
        result = gumbo_schedule.get_game_number("MIL", "STL", "2026-07-07", "1310")
        assert result == 1

    @patch.object(gumbo_schedule, "_get_cached_schedule")
    def test_game2_identified_by_time(self, mock_sched):
        """Ticker time matching game 2 (later) returns 2."""
        mock_sched.return_value = [
            _make_gumbo_game("MIL", "STL", "2026-07-07T17:10:00Z", game_number=1),
            _make_gumbo_game("MIL", "STL", "2026-07-07T23:45:00Z", game_number=2),
        ]
        # Ticker with 1945 ET → 2345 UTC → matches game 2
        result = gumbo_schedule.get_game_number("MIL", "STL", "2026-07-07", "1945")
        assert result == 2

    @patch.object(gumbo_schedule, "_get_cached_schedule")
    def test_no_time_on_doubleheader_returns_none(self, mock_sched):
        """Legacy ticker (no time) on doubleheader day → None (can't disambiguate)."""
        mock_sched.return_value = [
            _make_gumbo_game("MIL", "STL", "2026-07-07T17:10:00Z", game_number=1),
            _make_gumbo_game("MIL", "STL", "2026-07-07T23:45:00Z", game_number=2),
        ]
        result = gumbo_schedule.get_game_number("MIL", "STL", "2026-07-07", None)
        assert result is None

    @patch.object(gumbo_schedule, "_get_cached_schedule")
    def test_time_far_from_both_games_returns_none(self, mock_sched):
        """Ticker time >90 min from any GUMBO game → None (safety)."""
        mock_sched.return_value = [
            _make_gumbo_game("MIL", "STL", "2026-07-07T17:10:00Z", game_number=1),
            _make_gumbo_game("MIL", "STL", "2026-07-07T23:45:00Z", game_number=2),
        ]
        # 0800 ET → 1200 UTC — far from both 17:10 and 23:45
        result = gumbo_schedule.get_game_number("MIL", "STL", "2026-07-07", "0800")
        assert result is None
