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


def get_game_number(
    away_abbr: str, home_abbr: str, date_str: str, ticker_time_et: Optional[str] = None
) -> Optional[int]:
    """Identify which GUMBO game (1 or 2) a ticker refers to.

    Cross-references the ticker's encoded ET time against GUMBO gameDate (UTC)
    to disambiguate doubleheaders.

    Args:
        away_abbr: Kalshi-style away team abbreviation
        home_abbr: Kalshi-style home team abbreviation
        date_str:  YYYY-MM-DD calendar date
        ticker_time_et: HHMM string in Eastern Time (None for legacy tickers)

    Returns:
        gameNumber (1 or 2) if identified, None if no matching game found.
    """
    from .market_map import KALSHI_TO_STANDARD

    away_std = KALSHI_TO_STANDARD.get(away_abbr, away_abbr)
    home_std = KALSHI_TO_STANDARD.get(home_abbr, home_abbr)

    games = _get_cached_schedule(date_str)

    matching = []
    for g in games:
        g_away = g.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", "")
        g_home = g.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "")
        g_away_std = KALSHI_TO_STANDARD.get(g_away, g_away)
        g_home_std = KALSHI_TO_STANDARD.get(g_home, g_home)

        if g_away_std == away_std and g_home_std == home_std:
            matching.append(g)

    if len(matching) == 0:
        return None
    if len(matching) == 1:
        return matching[0].get("gameNumber", 1)

    # Doubleheader: 2+ games with same teams on same date
    if ticker_time_et is None:
        logger.warning(
            f"Doubleheader {away_std}@{home_std} on {date_str} but no ticker time — "
            f"cannot disambiguate"
        )
        return None

    # Convert ticker HHMM (ET) to UTC minutes-since-midnight
    # EDT = UTC-4 during MLB season (Apr-Oct)
    try:
        ticker_hour = int(ticker_time_et[:2])
        ticker_min = int(ticker_time_et[2:])
        ticker_utc_minutes = (ticker_hour + 4) * 60 + ticker_min  # EDT offset
    except (ValueError, IndexError):
        return None

    # Find closest GUMBO game by time
    best_game = None
    best_diff = float("inf")
    for g in matching:
        game_date_str = g.get("gameDate", "")
        if not game_date_str:
            continue
        gumbo_dt = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
        gumbo_minutes = gumbo_dt.hour * 60 + gumbo_dt.minute
        diff = abs(ticker_utc_minutes % 1440 - gumbo_minutes)
        if diff < best_diff:
            best_diff = diff
            best_game = g

    if best_game is None:
        return None

    # Sanity: if closest match is >90 min off, something is wrong
    if best_diff > 90:
        logger.warning(
            f"Ticker time {ticker_time_et} ET for {away_std}@{home_std} on {date_str} "
            f"doesn't closely match any GUMBO game (closest diff: {best_diff} min)"
        )
        return None

    return best_game.get("gameNumber", 1)


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


def get_game_states(date_str: str) -> list[dict]:
    """Return game states for a date: [{game_pk, abstract_state, home_abbr, away_abbr}].

    Used by FeatureManager to detect newly-finalized games without polling S3.
    """
    from .market_map import KALSHI_TO_STANDARD

    games = _get_cached_schedule(date_str)
    results = []
    for g in games:
        status = g.get("status", {})
        teams = g.get("teams", {})
        h_abbr = teams.get("home", {}).get("team", {}).get("abbreviation", "")
        a_abbr = teams.get("away", {}).get("team", {}).get("abbreviation", "")

        results.append({
            "game_pk": g.get("gamePk"),
            "abstract_state": status.get("abstractGameState", "Preview"),
            "home_abbr": KALSHI_TO_STANDARD.get(h_abbr, h_abbr),
            "away_abbr": KALSHI_TO_STANDARD.get(a_abbr, a_abbr),
        })
    return results


def get_games_with_context(date_str: str) -> list[dict]:
    """Return enriched game info for synthetic row construction.

    Hydrates venue, probable pitchers, league/division from GUMBO schedule.
    """
    from .market_map import KALSHI_TO_STANDARD

    games = _get_cached_schedule(date_str)
    results = []
    for g in games:
        status = g.get("status", {})
        teams = g.get("teams", {})
        venue = g.get("venue", {})

        home_team = teams.get("home", {}).get("team", {})
        away_team = teams.get("away", {}).get("team", {})

        h_abbr = home_team.get("abbreviation", "")
        a_abbr = away_team.get("abbreviation", "")

        # Probable pitchers (hydrated when available)
        home_pp = teams.get("home", {}).get("probablePitcher", {})
        away_pp = teams.get("away", {}).get("probablePitcher", {})

        game_date_str = g.get("gameDate", "")
        # Determine day/night from scheduled time (games after 5pm local are night)
        day_night = g.get("dayNight", "night")

        results.append({
            "game_pk": g.get("gamePk"),
            "abstract_state": status.get("abstractGameState", "Preview"),
            "home_abbr": KALSHI_TO_STANDARD.get(h_abbr, h_abbr),
            "away_abbr": KALSHI_TO_STANDARD.get(a_abbr, a_abbr),
            "venue_id": venue.get("id"),
            "venue_name": venue.get("name"),
            "game_datetime_utc": game_date_str,
            "probable_pitcher_home_id": home_pp.get("id"),
            "probable_pitcher_away_id": away_pp.get("id"),
            "game_number": g.get("gameNumber", 1),
            "day_night": day_night,
            "home_league_id": home_team.get("league", {}).get("id"),
            "away_league_id": away_team.get("league", {}).get("id"),
            "home_division_id": home_team.get("division", {}).get("id"),
            "away_division_id": away_team.get("division", {}).get("id"),
        })
    return results
