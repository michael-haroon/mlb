"""
Daily Gumbo API enrichment: standings, rosters, player stats, platoon splits, venue info.

All tables are written to S3 under the same bucket/prefix as download_history.py.
Designed to be called once per day from live_daemon._schedule_loop at 08:00 UTC,
and also importable as a standalone CLI with --dry-run support.

New S3 paths:
  data/season={Y}/standings_{YYYY-MM-DD}.parquet
  data/season={Y}/rosters_{YYYY-MM-DD}.parquet
  data/season={Y}/pitcher_stats_{YYYY-MM-DD}.parquet
  data/season={Y}/hitter_stats_{YYYY-MM-DD}.parquet
  data/season={Y}/pitcher_splits_{YYYY-MM-DD}.parquet   (vl + vr combined)
  data/season={Y}/hitter_splits_{YYYY-MM-DD}.parquet    (vl + vr combined)
  data/venue_info.parquet                               (static, overwritten)
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# CONFIG — mirrors download_history.py constants
# ---------------------------------------------------------------------------
S3_BUCKET  = "mlb-265753586044-us-east-1-an"
S3_PREFIX  = "data"
S3_REGION  = "us-east-1"
USE_S3     = True   # overridden to False by --local flag

MLB_BASE   = "https://statsapi.mlb.com"
MAX_RETRIES = 5
BASE_BACKOFF = 2.0
MAX_BACKOFF  = 60.0

DATA_DIR = "data"
LOG_DIR  = os.path.join(DATA_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# All 30 MLB team IDs (stable across seasons)
MLB_TEAM_IDS = [
    108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
    118, 119, 120, 121, 133, 134, 135, 136, 137, 138,
    139, 140, 141, 142, 143, 144, 145, 146, 147, 158,
]

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logger = logging.getLogger("DAILY_ENRICHMENT")
logger.setLevel(logging.DEBUG)

_fh = logging.FileHandler(os.path.join(LOG_DIR, "daily_enrichment.log"))
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_fh)

_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("[ENRICHMENT] %(asctime)s - %(message)s", "%H:%M:%S"))
logger.addHandler(_ch)

# ---------------------------------------------------------------------------
# SCHEMAS
# ---------------------------------------------------------------------------
STANDINGS_SCHEMA: Dict[str, str] = {
    "date": "object", "season": "int64", "team_id": "int64", "team_abbr": "object",
    "team_name": "object",
    "wins": "int64", "losses": "int64", "win_pct": "float64",
    "runs_scored": "int64", "runs_allowed": "int64", "run_differential": "int64",
    "streak_type": "object", "streak_number": "int64",
    "games_back": "float64", "wild_card_games_back": "float64",
    "division_rank": "int64", "league_rank": "int64",
    "league_id": "int64", "division_id": "int64",
}

ROSTER_SCHEMA: Dict[str, str] = {
    "date": "object", "season": "int64", "team_id": "int64",
    "player_id": "int64", "full_name": "object", "jersey_number": "object",
    "position_code": "object", "position_type": "object",
    "status_code": "object",
}

PITCHER_STATS_SCHEMA: Dict[str, str] = {
    "date": "object", "season": "int64", "player_id": "int64", "full_name": "object",
    "team_id": "int64", "games_played": "int64", "games_started": "int64",
    "innings_pitched": "float64", "era": "float64", "whip": "float64",
    "strikeouts": "int64", "walks": "int64", "home_runs": "int64",
    "k_per_9": "float64", "bb_per_9": "float64", "hr_per_9": "float64",
    "k_bb_ratio": "float64", "hits": "int64", "earned_runs": "int64",
    "wins": "int64", "losses": "int64", "saves": "int64",
    "batters_faced": "int64",
}

HITTER_STATS_SCHEMA: Dict[str, str] = {
    "date": "object", "season": "int64", "player_id": "int64", "full_name": "object",
    "team_id": "int64", "games_played": "int64", "plate_appearances": "int64",
    "at_bats": "int64", "hits": "int64", "doubles": "int64", "triples": "int64",
    "home_runs": "int64", "rbi": "int64", "stolen_bases": "int64",
    "avg": "float64", "obp": "float64", "slg": "float64", "ops": "float64",
    "strikeouts": "int64", "walks": "int64",
}

PITCHER_SPLITS_SCHEMA: Dict[str, str] = {
    **PITCHER_STATS_SCHEMA,
    "split": "object",  # "vl" (vs LHH) or "vr" (vs RHH)
}

HITTER_SPLITS_SCHEMA: Dict[str, str] = {
    **HITTER_STATS_SCHEMA,
    "split": "object",  # "vl" (vs LHP) or "vr" (vs RHP)
}

VENUE_INFO_SCHEMA: Dict[str, str] = {
    "venue_id": "int64", "venue_name": "object",
    "lf_line": "float64", "cf_center": "float64", "rf_line": "float64",
    "lf_wall_height": "float64", "cf_wall_height": "float64", "rf_wall_height": "float64",
    "capacity": "int64", "roof_type": "object", "surface": "object",
    "latitude": "float64", "longitude": "float64",
}

# ---------------------------------------------------------------------------
# HTTP CLIENT
# ---------------------------------------------------------------------------
_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=5, pool_maxsize=10)
        _session.mount("https://", adapter)
        _session.headers.update({"User-Agent": "mlb-enrichment/1.0", "Accept": "application/json"})
    return _session


def _safe_float(value, default: float = 0.0) -> float:
    """Convert MLB API stat strings to float; returns default for dashes/None."""
    if value is None:
        return default
    s = str(value).strip()
    if not s or set(s) <= {'-', '.'}:
        return default
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def _get(path: str, params: Optional[Dict] = None) -> Optional[Dict]:
    """GET from Gumbo API with retry/backoff."""
    url = f"{MLB_BASE}{path}"
    for attempt in range(MAX_RETRIES):
        try:
            resp = _get_session().get(url, params=params, timeout=20)
            if resp.status_code == 429:
                wait = min(BASE_BACKOFF ** (attempt + 1), MAX_BACKOFF)
                logger.warning(f"Rate-limited (429) on {url}. Backing off {wait:.1f}s")
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                wait = min(BASE_BACKOFF ** attempt, MAX_BACKOFF)
                logger.warning(f"Server error {resp.status_code} on {url}. Retry in {wait:.1f}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            wait = min(BASE_BACKOFF ** attempt, MAX_BACKOFF)
            logger.warning(f"Timeout attempt {attempt+1}/{MAX_RETRIES} on {url}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
        except requests.exceptions.RequestException as exc:
            logger.error(f"Request failed on {url}: {exc}")
            return None
    logger.error(f"Exhausted retries for {url}")
    return None


# ---------------------------------------------------------------------------
# S3 / LOCAL STORAGE
# ---------------------------------------------------------------------------
_s3_client = None


def _get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=S3_REGION)
    return _s3_client


def _apply_schema(df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
    for col, dtype in schema.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError):
                if dtype == "object":
                    df[col] = df[col].astype(str)
                else:
                    df[col] = pd.NA
        else:
            df[col] = "None" if dtype == "object" else pd.NA
    return df[list(schema.keys())]


def _save_parquet(records: List[Dict[str, Any]], schema: Dict[str, str], rel_path: str, dry_run: bool = False):
    if not records:
        logger.info(f"[save] {rel_path}: 0 rows, skipping write")
        return
    df = _apply_schema(pd.DataFrame(records), schema)
    logger.info(f"[save] {rel_path}: {len(df)} rows")
    if dry_run:
        logger.info(f"[dry-run] Would write {len(df)} rows to {rel_path}")
        return
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
    buf.seek(0)
    if USE_S3:
        key = f"{S3_PREFIX}/{rel_path}"
        _get_s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
        logger.info(f"[save] Written s3://{S3_BUCKET}/{key}")
    else:
        full = os.path.join(DATA_DIR, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as f:
            f.write(buf.getvalue())
        logger.info(f"[save] Written {full}")


# ---------------------------------------------------------------------------
# FETCH: STANDINGS
# ---------------------------------------------------------------------------
def _fetch_standings(date_str: str, season: int) -> List[Dict[str, Any]]:
    """Fetch standings for both leagues as of date_str."""
    data = _get("/api/v1/standings", params={
        "leagueId": "103,104",
        "season": season,
        "date": date_str,
        "hydrate": "team",
    })
    if not data:
        return []

    records = []
    for rec in data.get("records") or []:
        league_id  = (rec.get("league") or {}).get("id", -1)
        division_id = (rec.get("division") or {}).get("id", -1)
        for rank, tr in enumerate(rec.get("teamRecords") or [], start=1):
            team  = tr.get("team") or {}
            streak = tr.get("streak") or {}
            records.append({
                "date": date_str,
                "season": season,
                "team_id": team.get("id", -1),
                "team_abbr": team.get("abbreviation", ""),
                "team_name": team.get("name", ""),
                "wins": tr.get("wins", 0),
                "losses": tr.get("losses", 0),
                "win_pct": _safe_float(tr.get("winningPercentage")),
                "runs_scored": int(tr.get("runsScored") or tr.get("runs") or 0),
                "runs_allowed": int(tr.get("runsAllowed") or 0),
                "run_differential": int(tr.get("runDifferential") or 0),
                "streak_type": streak.get("streakType", ""),
                "streak_number": int(streak.get("streakNumber") or 0),
                "games_back": _safe_float(tr.get("gamesBack")),
                "wild_card_games_back": _safe_float(tr.get("wildCardGamesBack")),
                "division_rank": rank,
                "league_rank": tr.get("leagueRank", rank),
                "league_id": league_id,
                "division_id": division_id,
            })
    return records


# ---------------------------------------------------------------------------
# FETCH: ROSTERS
# ---------------------------------------------------------------------------
def _fetch_rosters(date_str: str, season: int, team_ids: List[int]) -> List[Dict[str, Any]]:
    records = []
    for team_id in team_ids:
        data = _get(f"/api/v1/teams/{team_id}/roster", params={
            "rosterType": "active",
            "season": season,
            "date": date_str,
        })
        if not data:
            logger.warning(f"[rosters] No data for team_id={team_id}")
            continue
        for entry in data.get("roster") or []:
            person = entry.get("person") or {}
            pos    = entry.get("position") or {}
            status = entry.get("status") or {}
            records.append({
                "date": date_str,
                "season": season,
                "team_id": team_id,
                "player_id": person.get("id", -1),
                "full_name": person.get("fullName", ""),
                "jersey_number": entry.get("jerseyNumber", ""),
                "position_code": pos.get("code", ""),
                "position_type": pos.get("type", ""),
                "status_code": status.get("code", "A"),
            })
    return records


# ---------------------------------------------------------------------------
# FETCH: BULK PLAYER STATS
# ---------------------------------------------------------------------------
def _parse_stat_row(split: Dict, date_str: str, season: int, group: str) -> Optional[Dict]:
    """Parse one split row from the bulk /api/v1/stats response."""
    player = split.get("player") or {}
    team   = split.get("team") or {}
    stat   = split.get("stat") or {}
    pid    = player.get("id", -1)
    if pid == -1:
        return None

    base = {
        "date": date_str,
        "season": season,
        "player_id": pid,
        "full_name": player.get("fullName", ""),
        "team_id": team.get("id", -1),
    }

    if group == "pitching":
        ip_str = stat.get("inningsPitched", "0")
        try:
            ip = float(ip_str)
        except (TypeError, ValueError):
            ip = 0.0
        k9 = stat.get("strikeoutsPer9Inn", stat.get("k9", None))
        bb9 = stat.get("walksPer9Inn", stat.get("bb9", None))
        hr9 = stat.get("homeRunsPer9", stat.get("hr9", None))
        base.update({
            "games_played": int(stat.get("gamesPlayed") or 0),
            "games_started": int(stat.get("gamesStarted") or 0),
            "innings_pitched": ip,
            "era": _safe_float(stat.get("era")),
            "whip": _safe_float(stat.get("whip")),
            "strikeouts": int(stat.get("strikeOuts") or 0),
            "walks": int(stat.get("baseOnBalls") or 0),
            "home_runs": int(stat.get("homeRuns") or 0),
            "k_per_9": _safe_float(k9, float("nan")),
            "bb_per_9": _safe_float(bb9, float("nan")),
            "hr_per_9": _safe_float(hr9, float("nan")),
            "k_bb_ratio": _safe_float(stat.get("strikeoutWalkRatio"), float("nan")),
            "hits": int(stat.get("hits") or 0),
            "earned_runs": int(stat.get("earnedRuns") or 0),
            "wins": int(stat.get("wins") or 0),
            "losses": int(stat.get("losses") or 0),
            "saves": int(stat.get("saves") or 0),
            "batters_faced": int(stat.get("battersFaced") or 0),
        })
    else:  # hitting
        base.update({
            "games_played": int(stat.get("gamesPlayed") or 0),
            "plate_appearances": int(stat.get("plateAppearances") or 0),
            "at_bats": int(stat.get("atBats") or 0),
            "hits": int(stat.get("hits") or 0),
            "doubles": int(stat.get("doubles") or 0),
            "triples": int(stat.get("triples") or 0),
            "home_runs": int(stat.get("homeRuns") or 0),
            "rbi": int(stat.get("rbi") or 0),
            "stolen_bases": int(stat.get("stolenBases") or 0),
            "avg": _safe_float(stat.get("avg")),
            "obp": _safe_float(stat.get("obp")),
            "slg": _safe_float(stat.get("slg")),
            "ops": _safe_float(stat.get("ops")),
            "strikeouts": int(stat.get("strikeOuts") or 0),
            "walks": int(stat.get("baseOnBalls") or 0),
        })
    return base


def _fetch_bulk_stats(
    date_str: str,
    season: int,
    group: str,            # "pitching" or "hitting"
    stats_type: str = "season",    # "season" or "statSplits"
    sit_code: Optional[str] = None,  # "vl" or "vr" for splits
) -> List[Dict[str, Any]]:
    """Fetch all players' stats in one API call (bulk endpoint)."""
    params: Dict[str, Any] = {
        "stats": stats_type,
        "group": group,
        "season": season,
        "playerPool": "All",
        "gameType": "R",
        "limit": 5000,
    }
    if sit_code:
        params["sitCodes"] = sit_code

    data = _get("/api/v1/stats", params=params)
    if not data:
        return []

    records = []
    for stat_block in data.get("stats") or []:
        for split in stat_block.get("splits") or []:
            row = _parse_stat_row(split, date_str, season, group)
            if row and sit_code:
                row["split"] = sit_code
            if row:
                records.append(row)
    return records


# ---------------------------------------------------------------------------
# FETCH: VENUE INFO (static)
# ---------------------------------------------------------------------------
# All current MLB venue IDs (including recently opened venues)
MLB_VENUE_IDS = [
    2, 3, 4, 5, 7, 8, 14, 15, 17, 19,
    22, 26, 31, 32, 47, 680, 1971, 2394, 2395, 2602,
    2700, 2889, 3289, 3312, 4169, 4705, 4914, 5099, 5325, 5950,
]


def fetch_venue_info(venue_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Fetch static field dimensions for MLB venues."""
    if venue_ids is None:
        venue_ids = MLB_VENUE_IDS
    records = []
    for vid in venue_ids:
        data = _get(f"/api/v1/venues/{vid}", params={"hydrate": "fieldInfo,location"})
        if not data:
            continue
        venues = data.get("venues") or []
        if not venues:
            continue
        v = venues[0]
        field = v.get("fieldInfo") or {}
        loc   = v.get("location") or {}
        coord = loc.get("defaultCoordinates") or {}
        records.append({
            "venue_id": v.get("id", vid),
            "venue_name": v.get("name", ""),
            "lf_line": float(field.get("leftLine") or 0) or float("nan"),
            "cf_center": float(field.get("center") or 0) or float("nan"),
            "rf_line": float(field.get("rightLine") or 0) or float("nan"),
            "lf_wall_height": float(field.get("leftCenter") or 0) or float("nan"),
            "cf_wall_height": float(field.get("center") or 0) or float("nan"),
            "rf_wall_height": float(field.get("rightCenter") or 0) or float("nan"),
            "capacity": v.get("capacity", 0) or 0,
            "roof_type": (v.get("roofType") or field.get("roofType") or ""),
            "surface": (v.get("surface") or field.get("surface") or ""),
            "latitude": float(coord.get("latitude") or 0) or float("nan"),
            "longitude": float(coord.get("longitude") or 0) or float("nan"),
        })
    return records


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------
def run_daily_enrichment(date_str: str, season: int, dry_run: bool = False):
    """Fetch and store all 6 enrichment tables for one date.

    Designed to be called from live_daemon._schedule_loop at 08:00 UTC.
    A failure in any individual table is logged and skipped — never propagates.
    """
    logger.info(f"Daily enrichment start: date={date_str} season={season} dry_run={dry_run}")
    t0 = time.time()

    # --- Standings ---
    try:
        records = _fetch_standings(date_str, season)
        _save_parquet(records, STANDINGS_SCHEMA, f"season={season}/standings_{date_str}.parquet", dry_run)
    except Exception:
        logger.error("standings fetch/save failed", exc_info=True)

    # --- Rosters (30 calls) ---
    try:
        records = _fetch_rosters(date_str, season, MLB_TEAM_IDS)
        _save_parquet(records, ROSTER_SCHEMA, f"season={season}/rosters_{date_str}.parquet", dry_run)
    except Exception:
        logger.error("rosters fetch/save failed", exc_info=True)

    # --- Pitcher season stats (bulk, 1 call) ---
    try:
        records = _fetch_bulk_stats(date_str, season, group="pitching", stats_type="season")
        _save_parquet(records, PITCHER_STATS_SCHEMA, f"season={season}/pitcher_stats_{date_str}.parquet", dry_run)
    except Exception:
        logger.error("pitcher_stats fetch/save failed", exc_info=True)

    # --- Hitter season stats (bulk, 1 call) ---
    try:
        records = _fetch_bulk_stats(date_str, season, group="hitting", stats_type="season")
        _save_parquet(records, HITTER_STATS_SCHEMA, f"season={season}/hitter_stats_{date_str}.parquet", dry_run)
    except Exception:
        logger.error("hitter_stats fetch/save failed", exc_info=True)

    # --- Pitcher platoon splits vs LHH / RHH (2 calls) ---
    try:
        vl = _fetch_bulk_stats(date_str, season, group="pitching", stats_type="statSplits", sit_code="vl")
        vr = _fetch_bulk_stats(date_str, season, group="pitching", stats_type="statSplits", sit_code="vr")
        _save_parquet(vl + vr, PITCHER_SPLITS_SCHEMA, f"season={season}/pitcher_splits_{date_str}.parquet", dry_run)
    except Exception:
        logger.error("pitcher_splits fetch/save failed", exc_info=True)

    # --- Hitter platoon splits vs LHP / RHP (2 calls) ---
    try:
        vl = _fetch_bulk_stats(date_str, season, group="hitting", stats_type="statSplits", sit_code="vl")
        vr = _fetch_bulk_stats(date_str, season, group="hitting", stats_type="statSplits", sit_code="vr")
        _save_parquet(vl + vr, HITTER_SPLITS_SCHEMA, f"season={season}/hitter_splits_{date_str}.parquet", dry_run)
    except Exception:
        logger.error("hitter_splits fetch/save failed", exc_info=True)

    elapsed = time.time() - t0
    logger.info(f"Daily enrichment complete in {elapsed:.1f}s")


def run_venue_info(dry_run: bool = False):
    """One-time fetch and store of venue field dimensions."""
    try:
        records = fetch_venue_info()
        _save_parquet(records, VENUE_INFO_SCHEMA, "venue_info.parquet", dry_run)
    except Exception:
        logger.error("venue_info fetch/save failed", exc_info=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily Gumbo enrichment fetch")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: today UTC)")
    parser.add_argument("--season", type=int, default=None, help="MLB season year (default: date year)")
    parser.add_argument("--venues-only", action="store_true", help="Only fetch venue info")
    parser.add_argument("--dry-run", action="store_true", help="Print row counts, no S3 writes")
    parser.add_argument("--local", action="store_true", help="Write to local disk instead of S3")
    args = parser.parse_args()

    if args.local:
        USE_S3 = False

    from datetime import datetime, timezone
    date_str = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    season   = args.season or int(date_str[:4])

    if args.venues_only:
        run_venue_info(dry_run=args.dry_run)
    else:
        run_daily_enrichment(date_str, season, dry_run=args.dry_run)
        run_venue_info(dry_run=args.dry_run)
