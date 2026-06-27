"""Orchestrator: raw S3/local data → engineered game_features.parquet.

This is the entry point for the feature engineering pipeline. It loads raw
data, builds the game frame, computes ratings (with Optuna-tuned parameters),
engineers features, and writes the final artifact.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from .constants import MLB_FRANCHISE_IDS, VALID_GAME_TYPE_CODES
from .data_loader import load_all
from .feature_engineering import engineer_features
from .ratings import attach_all_ratings
from .ratings_tuning import tune_all_ratings

log = logging.getLogger(__name__)


def build_features(
    source: str,
    output: Path,
    season_start: int,
    season_end: int | None = None,
    tune_ratings: bool = True,
    n_trials: int = 100,
    ratings_params: dict | None = None,
) -> Path:
    """Build the full feature set from raw data.

    Parameters
    ----------
    source : str
        S3 URI or local path to raw data.
    output : Path
        Directory to write game_features.parquet and metadata.
    season_start : int
        First season (inclusive).
    season_end : int, optional
        Last season (inclusive).
    tune_ratings : bool
        If True, run Optuna HPO on rating parameters before computing them.
    n_trials : int
        Optuna trials per rating system (ignored if tune_ratings=False).
    ratings_params : dict, optional
        Pre-tuned rating parameters. If provided, skips tuning.

    Returns
    -------
    Path
        Path to the written game_features.parquet.
    """
    from .game_builder import build_game_frame

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Checkpoint paths — each step writes its result so a re-run resumes from
    # the last completed step rather than starting over.
    ckpt_games_raw   = output / "_ckpt_games_raw.parquet"
    ckpt_games_rated = output / "_ckpt_games_rated.parquet"
    params_path      = output / "ratings_params.json"

    # --- Step 1: Load raw data ---
    if ckpt_games_raw.exists():
        log.info(f"Resuming: loading game frame from checkpoint {ckpt_games_raw}")
        games = pd.read_parquet(ckpt_games_raw)
    else:
        log.info(f"Loading raw data from {source} (seasons {season_start}–{season_end})...")
        raw = load_all(source, season_start, season_end)

        # --- Step 2: Build game frame (1 row per game) ---
        log.info("Building game frame...")
        games = build_game_frame(raw)

        # --- Step 2b: Filter to competitive MLB games ---
        n_before = len(games)
        games = _filter_to_mlb(games)
        n_dropped = n_before - len(games)
        if n_dropped > 0:
            log.info(f"Dropped {n_dropped} non-MLB rows (exhibitions, all-star, non-franchise teams)")

        games.to_parquet(ckpt_games_raw, index=False, engine="pyarrow")
        log.info(f"Checkpoint written: {ckpt_games_raw} ({len(games):,} rows)")

    # --- Step 3: Tune rating parameters (or use provided) ---
    if ratings_params is not None:
        params = ratings_params
        log.info("Using provided rating parameters (skipping Optuna tuning)")
    elif params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
        log.info(f"Resuming: loaded tuned params from {params_path}")
    elif tune_ratings:
        log.info(f"Tuning rating system parameters ({n_trials} trials)...")
        params = tune_all_ratings(games, n_trials=n_trials)
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)
        log.info(f"Saved tuned params to {params_path}")
    else:
        from .ratings import DEFAULT_PARAMS
        params = DEFAULT_PARAMS
        log.info("Using default (literature) rating parameters (tuning disabled)")

    # --- Step 3b: Pre-compute pitcher ERA/WHIP from prior starts ---
    # attach_all_ratings runs before engineer_features, but compute_elo's
    # pitcher adjustment reads sp_*_season_era from each row. Those columns
    # are computed in engineer_features._starting_pitcher_features (too late).
    # We pre-compute them here — same expanding-sum-shift(1) logic — so Elo
    # has valid pre-game ERA/WHIP when it iterates the game frame.
    from .feature_engineering import _compute_pregame_pitcher_era
    games = _compute_pregame_pitcher_era(games)

    # --- Step 4: Compute ratings ---
    if ckpt_games_rated.exists():
        log.info(f"Resuming: loading rated game frame from checkpoint {ckpt_games_rated}")
        games = pd.read_parquet(ckpt_games_rated)
    else:
        log.info("Computing ratings...")
        games = attach_all_ratings(games, params=params)
        games.to_parquet(ckpt_games_rated, index=False, engine="pyarrow")
        log.info(f"Checkpoint written: {ckpt_games_rated}")

    # --- Step 5: Feature engineering ---
    log.info("Engineering features...")
    games = engineer_features(games)

    # --- Step 6: Clean up temporary columns ---
    temp_cols = [c for c in games.columns if c.startswith("_")]
    if temp_cols:
        games.drop(columns=temp_cols, inplace=True)

    # --- Step 7: Write artifact ---
    out_path = output / "game_features.parquet"
    games.to_parquet(out_path, index=False, engine="pyarrow")

    # Clean up checkpoints on success — they're large and no longer needed
    for ckpt in (ckpt_games_raw, ckpt_games_rated):
        if ckpt.exists():
            ckpt.unlink()
            log.debug(f"Removed checkpoint {ckpt}")

    elapsed = time.time() - t0
    log.info(f"Written {out_path}: {len(games):,} games × {len(games.columns)} features ({elapsed:.1f}s)")

    # Write manifest
    manifest = {
        "source": source,
        "season_start": season_start,
        "season_end": season_end,
        "n_games": len(games),
        "n_features": len(games.columns),
        "ratings_tuned": tune_ratings or ratings_params is not None,
        "elapsed_secs": round(elapsed, 1),
    }
    with open(output / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return out_path


def _filter_to_mlb(games: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that aren't competitive MLB games between two franchises.

    Removes exhibitions (vs. college/minor-league/foreign teams) and
    All-Star games (pseudo-team IDs). Keeps: Regular Season, Spring Training
    (MLB vs MLB), Division Series, League Championship, World Series, Wild Card.
    """
    mask = pd.Series(True, index=games.index)

    if "game_type_code" in games.columns:
        mask &= games["game_type_code"].isin(VALID_GAME_TYPE_CODES)

    if "home_team_id" in games.columns:
        mask &= games["home_team_id"].isin(MLB_FRANCHISE_IDS)

    if "away_team_id" in games.columns:
        mask &= games["away_team_id"].isin(MLB_FRANCHISE_IDS)

    return games[mask].reset_index(drop=True)
