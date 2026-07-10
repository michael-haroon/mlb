"""
pregame/trading/schedule.py
---------------------------
MLB game schedule from the GUMBO Stats API.

Provides authoritative first-pitch times in UTC for use in market filtering
and hours-to-first-pitch calculations.

GUMBO gameDate is always UTC (ISO 8601, ends in Z).  EC2 runs UTC.
No timezone conversion needed.

Cache is intentional: schedule data for a day doesn't change often
(only lineup confirms, rare postponements), so we re-fetch once per
SCHEDULE_TTL_MINUTES and reuse in between.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional

import urllib.request
import json

logger = logging.getLogger(__name__)

GUMBO_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
SCHEDULE_TTL_MINUTES = 30  # re-fetch if cached data is older than this

# Internal cache: date_str → {"fetched_at": datetime, "games": list[dict]}
_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()


def _fetch_schedule(date_str: str) -> list[dict]:
    """Fetch raw game list from GUMBO for a single calendar date (YYYY-MM-DD).

    Returns list of game dicts, each containing:
      gamePk, gameDate (UTC ISO), status.detailedState,
      teams.away.team.abbreviation, teams.home.team.abbreviation
    """
    url = (
        f"{GUMBO_SCHEDULE_URL}?sportId=1&date={date_str}"
        f"&hydrate=team&gameType=R,F,D,L,W"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "mlb-trading/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        logger.warning(f"GUMBO schedule fetch failed for {date_str}: {e}")
        return []

    games = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            games.append(g)
    return games


def get_first_pitch_utc(away_abbr: str, home_abbr: str, date_str: str) -> Optional[datetime]:
    """Return the UTC first-pitch datetime for a game, or None if not found.

    Args:
        away_abbr: Kalshi-style abbreviation (e.g. "PHI", "NYM")
        home_abbr: Kalshi-style abbreviation (e.g. "DET", "LAD")
        date_str:  YYYY-MM-DD local calendar date (as encoded in Kalshi game_key)

    GUMBO uses full team names in `teams.away.team.name` and short codes in
    `teams.away.team.abbreviation`.  We normalise via KALSHI_TO_STANDARD so
    both sides use the same 3-char codes before comparing.
    """
    from .market_map import KALSHI_TO_STANDARD

    away_std = KALSHI_TO_STANDARD.get(away_abbr, away_abbr)
    home_std = KALSHI_TO_STANDARD.get(home_abbr, home_abbr)

    games = _get_cached_schedule(date_str)

    for g in games:
        g_away = g.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", "")
        g_home = g.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "")
        g_away_std = KALSHI_TO_STANDARD.get(g_away, g_away)
        g_home_std = KALSHI_TO_STANDARD.get(g_home, g_home)

        if g_away_std == away_std and g_home_std == home_std:
            game_date_str = g.get("gameDate", "")
            if not game_date_str:
                return None
            # gameDate is always UTC; parse directly
            return datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))

    logger.debug(f"No GUMBO game found for {away_std}@{home_std} on {date_str}")
    return None


def hours_to_first_pitch(away_abbr: str, home_abbr: str, date_str: str) -> Optional[float]:
    """Return hours until first pitch (negative if game already started), or None."""
    fp = get_first_pitch_utc(away_abbr, home_abbr, date_str)
    if fp is None:
        return None
    delta = (fp - datetime.now(timezone.utc)).total_seconds() / 3600.0
    return delta


def game_has_started(away_abbr: str, home_abbr: str, date_str: str) -> bool:
    """Return True if first pitch is in the past (game started or scheduled time passed)."""
    h = hours_to_first_pitch(away_abbr, home_abbr, date_str)
    if h is None:
        return False  # unknown → assume not started; lifecycle event will catch it
    return h <= 0


def _get_cached_schedule(date_str: str) -> list[dict]:
    """Return cached game list, refreshing if TTL expired."""
    with _cache_lock:
        entry = _cache.get(date_str)
        now = datetime.now(timezone.utc)
        if entry and (now - entry["fetched_at"]).total_seconds() < SCHEDULE_TTL_MINUTES * 60:
            return entry["games"]

    # Fetch outside lock to avoid holding it during network I/O
    games = _fetch_schedule(date_str)
    with _cache_lock:
        _cache[date_str] = {"fetched_at": datetime.now(timezone.utc), "games": games}
    logger.debug(f"GUMBO schedule refreshed for {date_str}: {len(games)} games")
    return games


def invalidate(date_str: str) -> None:
    """Force a cache invalidation for a date (e.g. after a postponement notice)."""
    with _cache_lock:
        _cache.pop(date_str, None)
