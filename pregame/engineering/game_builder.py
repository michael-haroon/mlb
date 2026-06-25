"""Build one-row-per-game DataFrame from raw tables.

Each row contains: game identifiers, home/away team box score aggregates,
linescore-derived targets, starting pitcher info, and game metadata.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "live"))
from mlb_dl.targets import build_game_targets

from .constants import (
    BATTING_SUM_COLUMNS,
    MLB_REGIME_CHANGES,
    PITCHING_SUM_COLUMNS,
)

log = logging.getLogger(__name__)


def build_game_frame(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Construct one-row-per-game DataFrame with raw stats and targets.

    Parameters
    ----------
    raw : dict[str, pd.DataFrame]
        Output of data_loader.load_all().

    Returns
    -------
    pd.DataFrame
        One row per game_pk with home/away batting/pitching aggregates,
        game metadata, targets, and starting pitcher info.
    """
    batting = raw["boxscore_batting"]
    pitching = raw["boxscore_pitching"]
    linescore = raw["linescore"]
    pitches = raw["pitches"]
    players = raw["players"]

    # --- Game metadata from pitches (deduplicated to one per game) ---
    game_meta = _extract_game_metadata(pitches)

    # --- Targets from linescore ---
    targets = build_game_targets(linescore, game_meta)
    log.info(f"Built targets for {len(targets):,} games")

    # --- Aggregate batting box scores per side ---
    home_bat = _aggregate_box(batting, "home", BATTING_SUM_COLUMNS, prefix="bat")
    away_bat = _aggregate_box(batting, "away", BATTING_SUM_COLUMNS, prefix="bat")

    # --- Aggregate pitching box scores per side ---
    home_pit = _aggregate_box(pitching, "home", PITCHING_SUM_COLUMNS, prefix="pit")
    away_pit = _aggregate_box(pitching, "away", PITCHING_SUM_COLUMNS, prefix="pit")

    # --- Starting pitcher info ---
    sp_home = _extract_starting_pitchers(pitching, "home", players)
    sp_away = _extract_starting_pitchers(pitching, "away", players)

    # --- Assemble game frame ---
    games = targets[["game_pk", "season", "game_date"]].copy()

    # Merge all components
    for df in [game_meta, home_bat, away_bat, home_pit, away_pit, sp_home, sp_away]:
        if df is not None and not df.empty:
            games = games.merge(df, on="game_pk", how="left")

    # Merge targets (all columns except those already in games)
    target_cols = [c for c in targets.columns if c not in games.columns or c == "game_pk"]
    games = games.merge(targets[target_cols], on="game_pk", how="left")

    # --- Regime change flags ---
    if "game_date" in games.columns:
        gd = pd.to_datetime(games["game_date"], errors="coerce")
        for flag_name, cutoff_str in MLB_REGIME_CHANGES.items():
            cutoff = pd.Timestamp(cutoff_str)
            games[flag_name] = (gd >= cutoff).astype("float32")

    # --- Derived batting stats for BaseRuns inputs ---
    games = _compute_derived_batting(games)

    games = games.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    log.info(f"Game frame: {len(games):,} rows × {len(games.columns)} columns")
    return games


def _extract_game_metadata(pitches: pd.DataFrame) -> pd.DataFrame:
    """Extract one-row-per-game metadata from pitches table."""
    meta_cols = [
        "game_pk", "venue_id", "venue_name", "venue_latitude", "venue_longitude",
        "venue_capacity", "venue_roof_type", "weather_condition", "weather_temp",
        "weather_wind", "day_night", "attendance", "home_team_id", "home_team_name",
        "home_team_abbr", "away_team_id", "away_team_name", "away_team_abbr",
        "home_wins", "home_losses", "home_win_pct", "away_wins", "away_losses",
        "away_win_pct", "probable_pitcher_home_id", "probable_pitcher_away_id",
        "umpire_hp", "game_type_code", "double_header", "game_number",
    ]
    available = [c for c in meta_cols if c in pitches.columns]
    meta = pitches[available].drop_duplicates("game_pk").reset_index(drop=True)
    return meta


def _aggregate_box(
    box_df: pd.DataFrame,
    side: str,
    sum_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Aggregate box score stats for one side (home/away) to game level."""
    side_df = box_df[box_df["side"] == side].copy()
    available = [c for c in sum_columns if c in side_df.columns]

    for col in available:
        side_df[col] = pd.to_numeric(side_df[col], errors="coerce")

    agg = side_df.groupby("game_pk", sort=False)[available].sum().reset_index()

    # Count players used
    agg[f"{side}_{prefix}_players_used"] = (
        side_df.groupby("game_pk", sort=False).size().values
        if len(agg) == side_df.groupby("game_pk").ngroups
        else np.nan
    )

    # Rename with side prefix
    rename_map = {col: f"{side}_{prefix}_{col}" for col in available}
    agg = agg.rename(columns=rename_map)
    return agg


def _extract_starting_pitchers(
    pitching: pd.DataFrame,
    side: str,
    players: pd.DataFrame,
) -> pd.DataFrame:
    """Extract starting pitcher game stats and biographical info."""
    side_pit = pitching[pitching["side"] == side].copy()

    # Starters are typically the first pitcher (lowest index per game)
    if "is_starter" in side_pit.columns:
        starters = side_pit[side_pit["is_starter"].fillna(False).astype(bool)]
    else:
        starters = side_pit.sort_values("game_pk").groupby("game_pk").first().reset_index()

    if starters.empty:
        return pd.DataFrame(columns=["game_pk"])

    stat_cols = [
        "game_innings_pitched", "game_hits", "game_runs", "game_earned_runs",
        "game_bb", "game_so", "game_hr", "game_pitches_thrown",
        "game_strikes_thrown", "season_era", "season_whip",
    ]
    available_stats = [c for c in stat_cols if c in starters.columns]
    keep = ["game_pk", "player_id"] + available_stats
    keep = [c for c in keep if c in starters.columns]

    sp = starters[keep].drop_duplicates("game_pk").reset_index(drop=True)

    # Merge pitcher handedness from players table
    if "player_id" in sp.columns and "pitch_hand_code" in players.columns:
        hand = players[["player_id", "pitch_hand_code"]].drop_duplicates("player_id")
        sp = sp.merge(hand, on="player_id", how="left")
        sp = sp.rename(columns={"pitch_hand_code": f"sp_{side}_hand"})

    # Rename stats with side prefix
    rename_map = {
        col: f"sp_{side}_{col}" for col in available_stats
    }
    if "player_id" in sp.columns:
        rename_map["player_id"] = f"sp_{side}_id"
    sp = sp.rename(columns=rename_map)

    return sp


def _compute_derived_batting(games: pd.DataFrame) -> pd.DataFrame:
    """Compute derived batting stats needed for BaseRuns calculation.

    Derives: PA, TB, H (hits), BB, HBP, HR, IBB, SB, CS, GDP, SH, SF
    for both home and away sides from the raw box score aggregates.
    """
    for side in ("home", "away"):
        p = f"{side}_bat_"

        # Plate appearances estimate: AB + BB + HBP + SAC + SF
        ab = games.get(f"{p}game_ab", 0)
        bb = games.get(f"{p}game_bb", 0)
        hbp = games.get(f"{p}game_hbp", 0)
        sac = games.get(f"{p}game_sac", 0)
        sf = games.get(f"{p}game_sf", 0)
        games[f"{side}_PA"] = ab + bb + hbp + sac + sf

        # Total bases: singles + 2*doubles + 3*triples + 4*HR
        h = games.get(f"{p}game_hits", 0)
        d = games.get(f"{p}game_doubles", 0)
        t = games.get(f"{p}game_triples", 0)
        hr = games.get(f"{p}game_hr", 0)
        singles = h - d - t - hr
        games[f"{side}_TB"] = singles + 2 * d + 3 * t + 4 * hr

        # Alias the core stats with cleaner names for ratings.py
        games[f"{side}_H"] = h
        games[f"{side}_BB"] = bb
        games[f"{side}_HBP"] = hbp
        games[f"{side}_HR"] = hr
        games[f"{side}_IBB"] = games.get(f"{p}game_ibb", 0)
        games[f"{side}_SB"] = games.get(f"{p}game_sb", 0)
        games[f"{side}_CS"] = games.get(f"{p}game_cs", 0)
        games[f"{side}_GDP"] = games.get(f"{p}game_gidp", 0)
        games[f"{side}_SH"] = sac
        games[f"{side}_SF"] = sf

    return games
