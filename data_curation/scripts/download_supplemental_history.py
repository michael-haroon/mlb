"""
Historical backfill for supplemental Gumbo tables.

Fetches standings and rosters by date (API supports ?date= param),
and season-level stats/splits for each MLB season (bulk endpoint, no date param).
Venue info is one-time, fetched once and stored as a static table.

Run on a t4g.medium instance (4 GB RAM); uses MAX_WORKERS=20.
Launch via scripts/launch_supplemental_backfill_ec2.sh — self-terminating.

Usage:
  python3.11 download_supplemental_history.py               # full run
  python3.11 download_supplemental_history.py --dry-run     # validate, no S3 writes
  python3.11 download_supplemental_history.py --table standings
  python3.11 download_supplemental_history.py --start-year 2020 --end-year 2025
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import List, Optional

# daily_enrichment provides all fetch + schema + save logic
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import daily_enrichment as de

DATA_DIR = "data"
LOG_DIR  = os.path.join(DATA_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

START_YEAR   = 2015
END_YEAR     = 2026
# Conservative for 4 GB RAM; standings/rosters calls are lightweight
# Per-table defaults (different APIs handle concurrency differently):
#   standings: 1 lightweight call/date → high concurrency safe
#   rosters:   each task = 30 sequential team calls → lower to avoid burst
#   stats/splits: bulk (1 call/season) → sequential, workers unused
STANDINGS_WORKERS = 30
ROSTERS_WORKERS   = 10
RATE_SLEEP        = 0.05  # ~20 req/sec per worker; tuned down from 0.1 for burst safety

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logger = logging.getLogger("SUPPLEMENTAL_BACKFILL")
logger.setLevel(logging.DEBUG)

_fh = logging.FileHandler(os.path.join(LOG_DIR, "supplemental_backfill.log"))
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_fh)

_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("[BACKFILL] %(asctime)s - %(message)s", "%H:%M:%S"))
logger.addHandler(_ch)

# ---------------------------------------------------------------------------
# DATE HELPERS
# ---------------------------------------------------------------------------
def _regular_season_dates(year: int) -> List[str]:
    """Return Mondays + the opening/closing dates for a season as YYYY-MM-DD strings.

    Rosters are fetched once per week (Monday) per team. Roster changes
    are infrequent mid-week, so weekly snapshots approximate daily state well.
    Standings are fetched every game-day.
    """
    # Approximate regular-season window (late March – early October)
    season_start = date(year, 3, 20)
    season_end   = date(year, 10, 15)
    dates = []
    d = season_start
    while d <= season_end:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=7)  # Mondays only for rosters
    return dates


def _all_game_dates_in_season(year: int) -> List[str]:
    """All calendar dates in the regular season for standings fetches."""
    season_start = date(year, 3, 20)
    season_end   = date(year, 10, 15)
    dates = []
    d = season_start
    while d <= season_end:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return dates


# ---------------------------------------------------------------------------
# STANDINGS BACKFILL
# ---------------------------------------------------------------------------
def backfill_standings(start_year: int, end_year: int, dry_run: bool = False, workers: int = STANDINGS_WORKERS):
    """Fetch standings for every game date 2015–present.

    The API supports /api/v1/standings?date=YYYY-MM-DD, giving standings
    as of that date. ~210 dates/season × 11 seasons = ~2,310 API calls.
    """
    all_tasks = []
    for year in range(start_year, end_year + 1):
        for date_str in _all_game_dates_in_season(year):
            all_tasks.append((date_str, year))

    logger.info(f"standings backfill: {len(all_tasks)} date×season pairs  workers={workers}")

    def _fetch_one(args):
        date_str, season = args
        try:
            records = de._fetch_standings(date_str, season)
            de._save_parquet(records, de.STANDINGS_SCHEMA,
                             f"season={season}/standings_{date_str}.parquet", dry_run)
            time.sleep(RATE_SLEEP)
            return date_str, len(records)
        except Exception as exc:
            logger.error(f"standings {date_str}: {exc}")
            return date_str, -1

    success = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, task): task for task in all_tasks}
        for fut in as_completed(futures):
            d_str, n = fut.result()
            if n >= 0:
                success += 1
            if success % 100 == 0:
                logger.info(f"standings: {success}/{len(all_tasks)} dates complete")

    logger.info(f"standings backfill done: {success}/{len(all_tasks)} succeeded")


# ---------------------------------------------------------------------------
# ROSTERS BACKFILL
# ---------------------------------------------------------------------------
def backfill_rosters(start_year: int, end_year: int, dry_run: bool = False, workers: int = ROSTERS_WORKERS):
    """Fetch active rosters once per week per season.

    30 teams × ~30 Monday dates/season × 11 seasons ≈ 9,900 API calls.
    Each task makes 30 sequential team calls — keep workers lower than standings
    to avoid bursting 30*workers simultaneous connections.
    """
    all_tasks = []
    for year in range(start_year, end_year + 1):
        for date_str in _regular_season_dates(year):
            all_tasks.append((date_str, year))

    logger.info(f"rosters backfill: {len(all_tasks)} date×season pairs  workers={workers}")

    def _fetch_one(args):
        date_str, season = args
        try:
            records = de._fetch_rosters(date_str, season, de.MLB_TEAM_IDS)
            de._save_parquet(records, de.ROSTER_SCHEMA,
                             f"season={season}/rosters_{date_str}.parquet", dry_run)
            time.sleep(RATE_SLEEP)
            return date_str, len(records)
        except Exception as exc:
            logger.error(f"rosters {date_str}: {exc}")
            return date_str, -1

    success = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, task): task for task in all_tasks}
        for fut in as_completed(futures):
            d_str, n = fut.result()
            if n >= 0:
                success += 1
            if success % 100 == 0:
                logger.info(f"rosters: {success}/{len(all_tasks)} dates complete")

    logger.info(f"rosters backfill done: {success}/{len(all_tasks)} succeeded")


# ---------------------------------------------------------------------------
# STATS / SPLITS BACKFILL (season-level snapshots, no ?date= param)
# ---------------------------------------------------------------------------
def backfill_player_stats(start_year: int, end_year: int, dry_run: bool = False):
    """Fetch end-of-season stats for each year.

    The bulk endpoint has no ?date= param, so these are end-of-season totals.
    For training, rolling stats are computed from boxscore_batting/pitching S3
    tables. These snapshots are for inference caching only.
    Uses the last day of October as the reference date label.
    """
    seasons = list(range(start_year, end_year + 1))
    # Use season end as date label (end-of-season snapshot)
    date_label = lambda yr: f"{yr}-10-01"

    for season in seasons:
        d = date_label(season)
        logger.info(f"player_stats backfill: season={season}")
        try:
            records = de._fetch_bulk_stats(d, season, group="pitching", stats_type="season")
            de._save_parquet(records, de.PITCHER_STATS_SCHEMA,
                             f"season={season}/pitcher_stats_{d}.parquet", dry_run)
        except Exception as exc:
            logger.error(f"pitcher_stats season={season}: {exc}")
        try:
            records = de._fetch_bulk_stats(d, season, group="hitting", stats_type="season")
            de._save_parquet(records, de.HITTER_STATS_SCHEMA,
                             f"season={season}/hitter_stats_{d}.parquet", dry_run)
        except Exception as exc:
            logger.error(f"hitter_stats season={season}: {exc}")
        time.sleep(0.5)  # short pause between seasons

    logger.info("player_stats backfill done")


def backfill_splits(start_year: int, end_year: int, dry_run: bool = False):
    """Fetch end-of-season platoon splits for each year."""
    seasons = list(range(start_year, end_year + 1))
    date_label = lambda yr: f"{yr}-10-01"

    for season in seasons:
        d = date_label(season)
        logger.info(f"splits backfill: season={season}")
        try:
            vl = de._fetch_bulk_stats(d, season, group="pitching", stats_type="statSplits", sit_code="vl")
            vr = de._fetch_bulk_stats(d, season, group="pitching", stats_type="statSplits", sit_code="vr")
            de._save_parquet(vl + vr, de.PITCHER_SPLITS_SCHEMA,
                             f"season={season}/pitcher_splits_{d}.parquet", dry_run)
        except Exception as exc:
            logger.error(f"pitcher_splits season={season}: {exc}")
        try:
            vl = de._fetch_bulk_stats(d, season, group="hitting", stats_type="statSplits", sit_code="vl")
            vr = de._fetch_bulk_stats(d, season, group="hitting", stats_type="statSplits", sit_code="vr")
            de._save_parquet(vl + vr, de.HITTER_SPLITS_SCHEMA,
                             f"season={season}/hitter_splits_{d}.parquet", dry_run)
        except Exception as exc:
            logger.error(f"hitter_splits season={season}: {exc}")
        time.sleep(0.5)

    logger.info("splits backfill done")


# ---------------------------------------------------------------------------
# VENUE INFO (one-time, static)
# ---------------------------------------------------------------------------
def backfill_venue_info(dry_run: bool = False):
    """Fetch venue field info once."""
    logger.info("venue_info backfill: fetching all MLB venues")
    de.run_venue_info(dry_run=dry_run)
    logger.info("venue_info backfill done")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Supplemental Gumbo historical backfill")
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year",   type=int, default=END_YEAR)
    parser.add_argument("--table", choices=["standings", "rosters", "stats", "splits", "venues", "all"],
                        default="all", help="Which table(s) to backfill")
    parser.add_argument("--standings-workers", type=int, default=STANDINGS_WORKERS,
                        help=f"Threads for standings (default {STANDINGS_WORKERS}; "
                             "1 lightweight call/date, safe to parallelize heavily)")
    parser.add_argument("--rosters-workers", type=int, default=ROSTERS_WORKERS,
                        help=f"Threads for rosters (default {ROSTERS_WORKERS}; "
                             "each task = 30 sequential team calls, keep lower)")
    parser.add_argument("--dry-run", action="store_true", help="Print row counts, no S3 writes")
    parser.add_argument("--local",   action="store_true", help="Write to local disk instead of S3")
    args = parser.parse_args()

    if args.local:
        de.USE_S3 = False

    t0 = time.time()
    logger.info(
        f"Supplemental backfill starting: years={args.start_year}–{args.end_year} "
        f"table={args.table} standings_workers={args.standings_workers} "
        f"rosters_workers={args.rosters_workers} dry_run={args.dry_run}"
    )

    run_all = (args.table == "all")

    if run_all or args.table == "venues":
        backfill_venue_info(dry_run=args.dry_run)
    if run_all or args.table == "standings":
        backfill_standings(args.start_year, args.end_year,
                           dry_run=args.dry_run, workers=args.standings_workers)
    if run_all or args.table == "rosters":
        backfill_rosters(args.start_year, args.end_year,
                         dry_run=args.dry_run, workers=args.rosters_workers)
    if run_all or args.table == "stats":
        backfill_player_stats(args.start_year, args.end_year, dry_run=args.dry_run)
    if run_all or args.table == "splits":
        backfill_splits(args.start_year, args.end_year, dry_run=args.dry_run)

    elapsed = time.time() - t0
    logger.info(f"Supplemental backfill complete in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
