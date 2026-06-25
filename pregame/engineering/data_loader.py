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
from .constants import (
    BOXSCORE_BATTING_COLUMNS,
    BOXSCORE_PITCHING_COLUMNS,
    LINESCORE_COLUMNS,
    PITCH_META_COLUMNS,
    PLAYERS_COLUMNS,
)

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


_TABLE_CONFIG: dict[str, dict] = {
    "boxscore_batting": {"columns": BOXSCORE_BATTING_COLUMNS, "season_agnostic": False},
    "boxscore_pitching": {"columns": BOXSCORE_PITCHING_COLUMNS, "season_agnostic": False},
    "linescore": {"columns": LINESCORE_COLUMNS, "season_agnostic": False},
    "pitches": {"columns": PITCH_META_COLUMNS, "season_agnostic": False},
    "players": {"columns": PLAYERS_COLUMNS, "season_agnostic": True},
}


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
    for table, cfg in _TABLE_CONFIG.items():
        log.info(f"Loading {table} for seasons {seasons[0]}–{seasons[-1]}...")
        table_seasons = None if cfg["season_agnostic"] else seasons
        df = catalog.read_table(table, columns=cfg["columns"], seasons=table_seasons)
        log.info(f"  {table}: {len(df):,} rows, {len(df.columns)} columns")
        data[table] = df

    return data
