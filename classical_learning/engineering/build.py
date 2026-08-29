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
from .data_loader import load_all, load_pitches_raw
from .feature_engineering import engineer_features
from .ratings import attach_all_ratings
from .weather import attach_weather_features
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

    from ..strategy.config import SKIP_SEASONS

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

        # --- Step 2c: Pitch-level features ---
        # Free raw tables before loading pitches_raw. Both pitches (game-meta,
        # loaded inside raw) and pitches_raw (per-pitch features) read from the
        # same parquet files, so they are never in _TABLE_CONFIG together.
        # Freeing raw first keeps peak RSS within 16 GB on an r8g.large.
        del raw

        from .pitch_level_features import compute_pitch_level_features
        log.info("Computing pitch-level features (loading pitches_raw in isolation)...")
        pitches_raw_df = load_pitches_raw(source, season_start, season_end)
        games = compute_pitch_level_features(pitches_raw_df, games)
        del pitches_raw_df

        games.to_parquet(ckpt_games_raw, index=False, engine="pyarrow")
        log.info(f"Checkpoint written: {ckpt_games_raw} ({len(games):,} rows)")

    # --- Step 2c: Exclude structural outlier seasons ---
    if SKIP_SEASONS and "season" in games.columns:
        pre_len = len(games)
        games = games[~games["season"].isin(SKIP_SEASONS)].reset_index(drop=True)
        if len(games) < pre_len:
            log.info(f"Excluded seasons {SKIP_SEASONS}: {pre_len:,} → {len(games):,} rows")

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

    # --- Step 4b: Weather features (ERA5 join + physics derivation) ---
    log.info("Attaching weather features from ERA5...")
    games = attach_weather_features(games, source, output)

    # --- Step 5: Feature engineering ---
    log.info("Engineering features...")
    games = engineer_features(games)

    # --- Step 6: Clean up temporary and leaky columns ---
    temp_cols = [c for c in games.columns if c.startswith("_")]
    if temp_cols:
        games.drop(columns=temp_cols, inplace=True)

    # Drop raw game-A columns that are only available AFTER the game.
    # These are intermediate data used by rolling computations (shift(1) protected)
    # but must NOT persist into the parquet where importance/models could treat
    # them as features to predict game A — that would be leakage.
    _POST_GAME_COLUMNS = {
        "attendance", "game_duration_minutes",
        # Raw game-A linescore/runners (rolling features use shift(1))
        "home_total_errors", "away_total_errors",
        "home_total_lob", "away_total_lob",
        "home_sb_attempts", "home_sb_successes",
        "home_extra_base_taken", "home_extra_base_opps",
        "home_first_to_third", "home_score_from_second",
        "away_sb_attempts", "away_sb_successes",
        "away_extra_base_taken", "away_extra_base_opps",
        "away_first_to_third", "away_score_from_second",
        # Post-game standings (pennant features use shift(1) per team)
        "home_division_games_back", "home_wild_card_games_back",
        "away_division_games_back", "away_wild_card_games_back",
        # Post-game win/loss record (rolling winpct uses shift(1))
        "home_wins", "home_losses", "home_win_pct", "home_games_played",
        "away_wins", "away_losses", "away_win_pct", "away_games_played",
    }
    leaky_cols = [c for c in games.columns if c in _POST_GAME_COLUMNS]
    if leaky_cols:
        games.drop(columns=leaky_cols, inplace=True)
        log.info(f"Dropped {len(leaky_cols)} post-game-only columns to prevent leakage")

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


def build_features_incremental(
    source: str,
    output: Path,
    tune_ratings: bool = False,
    ratings_params: dict | None = None,
) -> Path:
    """Incremental feature build: only load raw data for new games.

    Instead of loading all raw data across all seasons (OOMs on 8GB instances),
    this function:
    1. Reads a persisted game-frame checkpoint (pre-ratings, ~16k rows × ~80 cols)
    2. Loads raw data ONLY for the current season to discover new games
    3. Builds game frame rows for games not yet in the checkpoint
    4. Appends new rows and re-runs ratings + feature engineering in-process

    The game frame is ~15 MB; ratings + features on ~32k rows takes seconds.
    The expensive step (loading millions of pitch-level rows) is limited to one
    season at a time, well within the 8 GB memory envelope.

    Parameters
    ----------
    source : str
        Local path to raw_cache directory.
    output : Path
        Directory to write game_features.parquet.
    tune_ratings : bool
        If True, tune rating parameters (should be False during live trading).
    ratings_params : dict, optional
        Pre-tuned rating parameters. If provided, uses these directly.
    """
    from .game_builder import build_game_frame

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ckpt_path = output / "_game_frame.parquet"
    params_path = output / "ratings_params.json"

    from ..strategy.config import SKIP_SEASONS

    # --- Step 1: Load or bootstrap game frame checkpoint ---
    # Columns that must exist in the checkpoint for feature engineering to work.
    # If any are missing, the checkpoint predates a schema change and must be rebuilt.
    _REQUIRED_CHECKPOINT_COLS = {"umpire_hp", "umpire_2b", "home_sb_attempts"}

    if ckpt_path.exists():
        game_frame = pd.read_parquet(ckpt_path)
        missing = _REQUIRED_CHECKPOINT_COLS - set(game_frame.columns)
        if missing:
            log.info(f"Checkpoint missing required columns {missing} — re-bootstrapping")
            game_frame = _bootstrap_game_frame(source, output)
            game_frame.to_parquet(ckpt_path, index=False, engine="pyarrow")
        else:
            log.info(f"Loaded game frame checkpoint: {len(game_frame):,} rows, "
                     f"latest game_date={game_frame['game_date'].max()}")
    else:
        log.info("No game frame checkpoint found — bootstrapping from full build")
        game_frame = _bootstrap_game_frame(source, output)
        game_frame.to_parquet(ckpt_path, index=False, engine="pyarrow")
        log.info(f"Bootstrapped game frame: {len(game_frame):,} rows")

    # Filter stale SKIP_SEASONS from checkpoint (may predate this exclusion)
    if SKIP_SEASONS and "season" in game_frame.columns:
        pre_len = len(game_frame)
        game_frame = game_frame[~game_frame["season"].isin(SKIP_SEASONS)].reset_index(drop=True)
        if len(game_frame) < pre_len:
            log.info(f"Purged {pre_len - len(game_frame)} SKIP_SEASONS rows from checkpoint")

    # --- Step 2: Find new games in the current season ---
    existing_pks = set(game_frame["game_pk"].values)
    current_season = int(game_frame["game_date"].max()[:4])

    log.info(f"Loading raw data for season {current_season} to find new games...")
    raw_new = load_all(source, season_start=current_season)

    new_game_frame = build_game_frame(raw_new)
    new_game_frame = _filter_to_mlb(new_game_frame)
    del raw_new  # free memory before loading pitches_raw

    # Load pitch-level data separately (same pattern as full build) and compute
    # rolling split features. load_pitches_raw reads from the same "pitches" table
    # but selects only the per-pitch columns needed for feature engineering.
    from .pitch_level_features import compute_pitch_level_features
    log.info("Loading pitches_raw for pitch-level features...")
    pitches_raw_df = load_pitches_raw(source, current_season)
    new_game_frame = compute_pitch_level_features(pitches_raw_df, new_game_frame)
    del pitches_raw_df

    new_pks = set(new_game_frame["game_pk"].values)
    truly_new = new_pks - existing_pks
    log.info(f"Found {len(truly_new):,} new games in season {current_season} "
             f"({len(new_game_frame):,} total this season, "
             f"{len(existing_pks):,} already in checkpoint)")

    # --- Step 3: Replace current-season rows in checkpoint with pitch-enriched version ---
    out_path = output / "game_features.parquet"
    if not truly_new and out_path.exists():
        # Check if pitch columns need backfilling on existing rows
        pitch_cols_in_ckpt = [c for c in game_frame.columns if "kpct_vs" in c or "woba_vs" in c]
        if pitch_cols_in_ckpt:
            elapsed = time.time() - t0
            log.info(f"No new games and pitch features present — nothing to do ({elapsed:.1f}s)")
            del new_game_frame
            return out_path

    # Replace all current-season rows with the pitch-enriched version.
    # This both appends new games AND backfills pitch features for existing rows.
    game_frame = game_frame[~game_frame["game_pk"].isin(new_pks)]
    game_frame = pd.concat([game_frame, new_game_frame], ignore_index=True)
    game_frame = game_frame.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    game_frame.to_parquet(ckpt_path, index=False, engine="pyarrow")
    log.info(f"Updated game frame checkpoint: {len(game_frame):,} rows "
             f"({len(truly_new)} new + {len(new_pks) - len(truly_new)} refreshed)")
    del new_game_frame

    # --- Step 4: Resolve rating parameters ---
    if ratings_params is not None:
        params = ratings_params
    elif params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
        log.info(f"Using saved rating params from {params_path}")
    elif tune_ratings:
        log.info("Tuning rating parameters...")
        params = tune_all_ratings(game_frame, n_trials=100)
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)
    else:
        from .ratings import DEFAULT_PARAMS
        params = DEFAULT_PARAMS
        log.info("Using default rating parameters (tuning disabled)")

    # --- Step 5: Compute ratings + features on full game frame ---
    from .feature_engineering import _compute_pregame_pitcher_era

    games = game_frame.copy()
    games = _compute_pregame_pitcher_era(games)
    games = attach_all_ratings(games, params=params)
    games = engineer_features(games)

    # --- Step 5b: Weather features (ERA5 join + physics derivation) ---
    # ERA5 lives on S3, not local cache — use S3 URI directly.
    from ..trading.config import S3_DATA_URI
    weather_source = S3_DATA_URI  # s3://bucket/data
    log.info("Attaching weather features from ERA5...")
    games = attach_weather_features(games, weather_source, output)

    # Clean up temporary columns
    temp_cols = [c for c in games.columns if c.startswith("_")]
    if temp_cols:
        games.drop(columns=temp_cols, inplace=True)

    # Drop post-game-only columns (same blocklist as full build)
    _POST_GAME_COLUMNS = {
        "attendance", "game_duration_minutes",
        "home_total_errors", "away_total_errors",
        "home_total_lob", "away_total_lob",
        "home_sb_attempts", "home_sb_successes",
        "home_extra_base_taken", "home_extra_base_opps",
        "home_first_to_third", "home_score_from_second",
        "away_sb_attempts", "away_sb_successes",
        "away_extra_base_taken", "away_extra_base_opps",
        "away_first_to_third", "away_score_from_second",
        "home_division_games_back", "home_wild_card_games_back",
        "away_division_games_back", "away_wild_card_games_back",
        "home_wins", "home_losses", "home_win_pct", "home_games_played",
        "away_wins", "away_losses", "away_win_pct", "away_games_played",
    }
    leaky_cols = [c for c in games.columns if c in _POST_GAME_COLUMNS]
    if leaky_cols:
        games.drop(columns=leaky_cols, inplace=True)
        log.info(f"Dropped {len(leaky_cols)} post-game-only columns to prevent leakage")

    # --- Step 6: Write final artifact ---
    games.to_parquet(out_path, index=False, engine="pyarrow")

    elapsed = time.time() - t0
    log.info(f"Written {out_path}: {len(games):,} games × {len(games.columns)} features ({elapsed:.1f}s)")

    manifest = {
        "source": source,
        "mode": "incremental",
        "n_games": len(games),
        "n_features": len(games.columns),
        "n_new_games": len(truly_new),
        "ratings_tuned": tune_ratings or ratings_params is not None,
        "elapsed_secs": round(elapsed, 1),
    }
    with open(output / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return out_path


def _bootstrap_game_frame(source: str, output: Path) -> pd.DataFrame:
    """Build initial game frame checkpoint from raw data, one season at a time.

    Loads each season independently to stay within memory limits, then
    concatenates the resulting game frames (not the raw data).
    """
    from .game_builder import build_game_frame
    from datetime import datetime

    current_year = datetime.now().year
    start_year = 2015
    all_frames = []

    from ..strategy.config import SKIP_SEASONS

    for year in range(start_year, current_year + 1):
        if year in SKIP_SEASONS:
            log.info(f"Bootstrap: skipping season {year} (in SKIP_SEASONS)")
            continue
        log.info(f"Bootstrap: loading season {year}...")
        try:
            raw = load_all(source, season_start=year, season_end=year)
            frame = build_game_frame(raw)
            frame = _filter_to_mlb(frame)
            del raw

            # Compute pitch-level features (statcast era only)
            try:
                pitches_df = load_pitches_raw(source, year, year)
                if len(pitches_df) > 0:
                    from .pitch_level_features import compute_pitch_level_features
                    frame = compute_pitch_level_features(pitches_df, frame)
                del pitches_df
            except Exception as e:
                log.debug(f"  Season {year}: no pitch-level features ({e})")

            all_frames.append(frame)
            log.info(f"  Season {year}: {len(frame):,} games")
        except Exception as e:
            log.warning(f"  Season {year} failed: {e}")

    if not all_frames:
        raise RuntimeError("Bootstrap failed — no seasons loaded successfully")

    result = pd.concat(all_frames, ignore_index=True)
    result = result.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    return result


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
