"""
pregame/trading/market_map.py
-----------------------------
Kalshi MLB ticker parsing and team code normalization.

Kalshi ticker format for MLB:
  {SERIES}-{GAME_KEY}-{STRIKE}

Examples:
  KXMLBGAME-26JUL03NYMLAD-NYM       (Mets win)
  KXMLBSPREAD-26JUL03NYMLAD-NYM2    (Mets win by 2+)
  KXMLBTOTAL-26JUL03NYMLAD-9        (total runs over 9)
  KXMLBTEAMTOTAL-26JUL03NYMLAD-NYM4 (Mets score 4+)
  KXMLBRFI-26JUL03NYMLAD            (YRFI — no strike suffix)
  KXMLBEXTRAS-26JUL03NYMLAD-EXTRAS  (goes to extras)

Game key encodes: {YY}{MON}{DD}{AWAY}{HOME}
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Kalshi uses variable-length team abbreviations (2-3 chars).
# Map from model targets to the Kalshi series they price.
MODEL_TO_SERIES = {
    "home_win": "KXMLBGAME",
    "yrfi": "KXMLBRFI",
    "first_5_home_win": None,        # No Kalshi F5 winner series yet
    "extra_innings": "KXMLBEXTRAS",
    "home_run_diff": "KXMLBSPREAD",
    "total_runs": "KXMLBTOTAL",
    "home_runs": "KXMLBTEAMTOTAL",
    "away_runs": "KXMLBTEAMTOTAL",
    "first_5_home_run_diff": None,   # No Kalshi F5 spread yet
    "first_5_total_runs": None,      # No Kalshi F5 total yet
}

# MLB team abbreviations as Kalshi uses them.
# Kalshi may use 2 or 3 char codes — normalize to standard 3-char.
KALSHI_TO_STANDARD = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CWS": "CWS", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KC": "KCR",
    "KCR": "KCR", "LAA": "LAA", "LAD": "LAD", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY",
    "OAK": "OAK", "PHI": "PHI", "PIT": "PIT", "SD": "SDP",
    "SDP": "SDP", "SF": "SFG", "SFG": "SFG", "SEA": "SEA",
    "STL": "STL", "TB": "TBR", "TBR": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WSH": "WSH", "WAS": "WSH",
}

STANDARD_TO_KALSHI = {v: k for k, v in KALSHI_TO_STANDARD.items()}


@dataclass
class ParsedTicker:
    """Decomposed Kalshi MLB ticker."""
    series: str          # e.g. "MLBGAME"
    game_key: str        # e.g. "26JUL03NYMLAD"
    away_team: str       # e.g. "NYM" (standardized)
    home_team: str       # e.g. "LAD" (standardized)
    strike_team: Optional[str]   # team in the strike (None for totals/binary)
    strike_value: Optional[float]  # numeric threshold (None for winner/yrfi)
    raw: str             # original ticker string
    ticker_time: Optional[str] = None  # HHMM in ET (None for legacy format)


def parse_ticker(ticker: str) -> Optional[ParsedTicker]:
    """Parse a Kalshi MLB ticker into its components.

    Returns None if the ticker doesn't match expected MLB format.
    """
    parts = ticker.split("-")
    if len(parts) < 2:
        return None

    series = parts[0]
    game_key = parts[1]

    # Extract teams from game_key.
    # Kalshi uses two formats:
    #   Legacy:  {YY}{MON}{DD}{AWAY}{HOME}             e.g. 26JUL03NYMLAD
    #   Current: {YY}{MON}{DD}{HHMM}{AWAY}{HOME}[Gn]  e.g. 26JUL051300PITWSH
    #                                                       26JUL071945MILSTLG2
    # Date = 7 chars (YYMMMDD). Optional 4-digit time follows. Optional Gn
    # doubleheader suffix (G1/G2) trails the home team code.
    import re as _re
    after_date = game_key[7:]
    # Strip leading 4-digit time block (HHMM)
    m_time = _re.match(r'^(\d{4})(.+)$', after_date)
    ticker_time = m_time.group(1) if m_time else None
    team_str = m_time.group(2) if m_time else after_date
    # Strip trailing doubleheader suffix G1/G2/G3
    team_str = _re.sub(r'G\d$', '', team_str)
    away_code, home_code = _split_team_codes(team_str)
    if not away_code or not home_code:
        return None

    away_team = KALSHI_TO_STANDARD.get(away_code, away_code)
    home_team = KALSHI_TO_STANDARD.get(home_code, home_code)

    # Parse strike (3rd part, if present)
    strike_team = None
    strike_value = None
    if len(parts) >= 3:
        strike_part = parts[2]
        if strike_part == "EXTRAS":
            # KXMLBEXTRAS-26JUL072210COLLAD-EXTRAS: binary, no numeric strike
            pass
        else:
            m = re.match(r"^([A-Z]{2,3})(\d+\.?\d*)$", strike_part)
            if m:
                # Team + numeric threshold: e.g. "NYM2" (Mets win by 2+)
                strike_team = KALSHI_TO_STANDARD.get(m.group(1), m.group(1))
                strike_value = float(m.group(2))
            elif re.match(r"^[A-Z]{2,3}$", strike_part):
                # Pure team code (winner markets): e.g. "NYM"
                strike_team = KALSHI_TO_STANDARD.get(strike_part, strike_part)
            else:
                # Pure numeric strike (totals): e.g. "9" or "8.5"
                try:
                    strike_value = float(strike_part)
                except ValueError:
                    pass

    return ParsedTicker(
        series=series,
        game_key=game_key,
        away_team=away_team,
        home_team=home_team,
        strike_team=strike_team,
        strike_value=strike_value,
        raw=ticker,
        ticker_time=ticker_time,
    )


def _split_team_codes(team_str: str) -> tuple[str, str]:
    """Split concatenated team codes like 'NYMLAD' → ('NYM', 'LAD').

    Tries 3+3, 3+2, 2+3, 2+2 splits. Returns best match against known teams.
    """
    for away_len in (3, 2):
        for home_len in (3, 2):
            if away_len + home_len != len(team_str):
                continue
            away = team_str[:away_len]
            home = team_str[away_len:]
            if away in KALSHI_TO_STANDARD and home in KALSHI_TO_STANDARD:
                return away, home
    # Fallback: try 3+remaining
    if len(team_str) >= 5:
        return team_str[:3], team_str[3:]
    if len(team_str) >= 4:
        return team_str[:2], team_str[2:]
    return "", ""


def game_key_from_matchup(date_str: str, away_team: str, home_team: str) -> str:
    """Build a Kalshi game key from date and teams.

    Args:
        date_str: ISO date "2026-07-03"
        away_team: standard 3-char code
        home_team: standard 3-char code
    """
    from datetime import datetime
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    yy = dt.strftime("%y")
    mon = dt.strftime("%b").upper()
    dd = dt.strftime("%d")
    # Use Kalshi's abbreviated codes (shortest known mapping)
    away_k = STANDARD_TO_KALSHI.get(away_team, away_team)
    home_k = STANDARD_TO_KALSHI.get(home_team, home_team)
    return f"{yy}{mon}{dd}{away_k}{home_k}"


def ticker_for_market(
    series: str,
    game_key: str,
    team: Optional[str] = None,
    threshold: Optional[float] = None,
) -> str:
    """Construct a Kalshi ticker from components."""
    if team and threshold is not None:
        team_k = STANDARD_TO_KALSHI.get(team, team)
        # Format threshold: strip trailing .0 for integers
        t_str = str(int(threshold)) if threshold == int(threshold) else str(threshold)
        return f"{series}-{game_key}-{team_k}{t_str}"
    elif threshold is not None:
        t_str = str(int(threshold)) if threshold == int(threshold) else str(threshold)
        return f"{series}-{game_key}-{t_str}"
    elif team:
        team_k = STANDARD_TO_KALSHI.get(team, team)
        return f"{series}-{game_key}-{team_k}"
    else:
        return f"{series}-{game_key}"


def classify_cluster(series: str) -> str:
    """Map a Kalshi series to its correlation cluster for position caps."""
    mapping = {
        "KXMLBGAME": "winner",
        "KXMLBSPREAD": "spread",
        "KXMLBTOTAL": "total",
        "KXMLBTEAMTOTAL": "team_total",
        "KXMLBRFI": "first_inning",
        "KXMLBEXTRAS": "extra_innings",
    }
    return mapping.get(series, "winner")
