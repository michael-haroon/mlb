"""Load raw MLB parquet data from S3 or local via ParquetCatalog.

Reuses the proven ParquetCatalog from the deep learning module to avoid
duplicating S3/local path resolution and thread-safe Arrow reading logic.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "live"))
from mlb_dl.data_sources import ParquetCatalog, season_range

LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

_fh = logging.FileHandler(LOG_DIR / "pregame_engineering.log")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
log.addHandler(_fh)

_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.INFO)
_sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
log.addHandler(_sh)


REQUIRED_TABLES = [
    "boxscore_batting",
    "boxscore_pitching",
    "linescore",
    "pitches",
    "players",
]


def load_all(
    source: str,
    season_start: int,
    season_end: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """Load all required tables from raw parquet data.

    Parameters
    ----------
    source : str
        S3 URI or local path to the raw data root.
    season_start : int
        First season to load (inclusive).
    season_end : int, optional
        Last season to load (inclusive). Defaults to current year.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by table name.
    """
    seasons = season_range(season_start, season_end)
    catalog = ParquetCatalog(source)

    data: dict[str, pd.DataFrame] = {}
    for table in REQUIRED_TABLES:
        log.info(f"Loading {table} for seasons {seasons[0]}–{seasons[-1]}...")
        # Players table is season-agnostic
        table_seasons = None if table == "players" else seasons
        df = catalog.read_table(table, seasons=table_seasons)
        log.info(f"  {table}: {len(df):,} rows, {len(df.columns)} columns")
        data[table] = df

    return data


def load_pitches_metadata(
    source: str,
    season_start: int,
    season_end: Optional[int] = None,
) -> pd.DataFrame:
    """Load only game-level metadata columns from the pitches table.

    This avoids loading the full 170-column pitches table when we only need
    game metadata (venue, weather, teams, probable pitchers).
    """
    from .constants import PITCH_META_COLUMNS

    seasons = season_range(season_start, season_end)
    catalog = ParquetCatalog(source)

    df = catalog.read_table("pitches", columns=PITCH_META_COLUMNS, seasons=seasons)
    # Deduplicate to one row per game (pitches has one row per pitch)
    game_meta = df.drop_duplicates("game_pk").reset_index(drop=True)
    log.info(f"Game metadata: {len(game_meta):,} games")
    return game_meta
