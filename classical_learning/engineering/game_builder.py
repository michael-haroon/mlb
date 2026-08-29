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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deep_learning"))
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
    runners = raw.get("runners")

    # --- Game metadata from pitches (deduplicated to one per game) ---
    game_meta = _extract_game_metadata(pitches)

    # --- Targets from linescore ---
    targets = build_game_targets(linescore, game_meta)
    log.info(f"Built targets for {len(targets):,} games")

    # --- Extended linescore (errors, LOB) ---
    linescore_ext = _aggregate_linescore_extended(linescore)

    # --- Runners aggregation (baserunning stats per side) ---
    runners_agg = _aggregate_runners(runners, pitches) if runners is not None else None

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
    # game_date lives in game_meta (from pitches), not in targets (from linescore)
    games = targets[["game_pk", "season"]].copy()

    # Merge all components
    components = [game_meta, home_bat, away_bat, home_pit, away_pit, sp_home, sp_away,
                  linescore_ext, runners_agg]
    for df in components:
        if df is not None and not df.empty:
            games = games.merge(df, on="game_pk", how="left")

    # Merge targets (all columns except those already in games)
    target_cols = [c for c in targets.columns if c not in games.columns or c == "game_pk"]
    games = games.merge(targets[target_cols], on="game_pk", how="left")

    # --- Regime change flags ---
    if "game_date" in games.columns:
        gd = pd.to_datetime(games["game_date"], errors="coerce")
        flag_cols = {
            flag_name: (gd >= pd.Timestamp(cutoff_str)).astype("float32")
            for flag_name, cutoff_str in MLB_REGIME_CHANGES.items()
        }
        games = pd.concat([games, pd.DataFrame(flag_cols, index=games.index)], axis=1)

    # --- Derived batting stats for BaseRuns inputs ---
    games = _compute_derived_batting(games)

    # Sort by actual start time (game_datetime_utc) to resolve within-day ordering.
    # game_pk is NOT monotonic with start time — MLB assigns it at schedule creation,
    # not at first pitch. Falls back to (game_date, game_pk) if datetime is absent.
    if "game_datetime_utc" in games.columns:
        games["_sort_dt"] = pd.to_datetime(games["game_datetime_utc"], errors="coerce", utc=True)
        games = games.sort_values(["_sort_dt", "game_pk"]).reset_index(drop=True)
        games = games.drop(columns=["_sort_dt"])
    else:
        games = games.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    log.info(f"Game frame: {len(games):,} rows × {len(games.columns)} columns")
    return games


def _extract_game_metadata(pitches: pd.DataFrame) -> pd.DataFrame:
    """Extract one-row-per-game metadata from pitches table."""
    meta_cols = [
        "game_pk", "game_date", "game_datetime_utc",
        "venue_id", "venue_name", "venue_latitude", "venue_longitude",
        "venue_capacity", "venue_roof_type", "venue_surface",
        "weather_condition", "weather_temp",
        "weather_wind", "day_night", "attendance", "home_team_id", "home_team_name",
        "home_team_abbr", "home_league_id", "home_division_id",
        "away_team_id", "away_team_name", "away_team_abbr",
        "away_league_id", "away_division_id",
        # home_wins/win_pct are intentionally excluded: the MLB boxscore API
        # returns the post-game standing. Pre-game win% is computed in
        # feature_engineering.py via rolling win% with shift(1).
        "probable_pitcher_home_id", "probable_pitcher_away_id",
        "umpire_hp", "umpire_2b", "game_type_code", "double_header", "game_number",
        # Pennant race (available at Scheduled state; "-" → NaN for leaders)
        "home_division_games_back", "home_wild_card_games_back",
        "away_division_games_back", "away_wild_card_games_back",
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

    # season_era and season_whip are intentionally excluded: the MLB boxscore API
    # returns seasonStats as of game completion, so these values always include
    # the current outing. Pre-game ERA/WHIP is computed in feature_engineering.py
    # via expanding cumulative sums with shift(1) over prior starts.
    stat_cols = [
        "game_innings_pitched", "game_hits", "game_runs", "game_earned_runs",
        "game_bb", "game_so", "game_hr", "game_pitches_thrown",
        "game_strikes_thrown",
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
    new_cols: dict[str, pd.Series] = {}
    for side in ("home", "away"):
        p = f"{side}_bat_"

        ab = games.get(f"{p}game_ab", 0)
        bb = games.get(f"{p}game_bb", 0)
        hbp = games.get(f"{p}game_hbp", 0)
        sac = games.get(f"{p}game_sac", 0)
        sf = games.get(f"{p}game_sf", 0)
        new_cols[f"{side}_PA"] = ab + bb + hbp + sac + sf

        h = games.get(f"{p}game_hits", 0)
        d = games.get(f"{p}game_doubles", 0)
        t = games.get(f"{p}game_triples", 0)
        hr = games.get(f"{p}game_hr", 0)
        singles = h - d - t - hr
        new_cols[f"{side}_TB"] = singles + 2 * d + 3 * t + 4 * hr

        new_cols[f"{side}_H"] = h
        new_cols[f"{side}_BB"] = bb
        new_cols[f"{side}_HBP"] = hbp
        new_cols[f"{side}_HR"] = hr
        new_cols[f"{side}_IBB"] = games.get(f"{p}game_ibb", 0)
        new_cols[f"{side}_SB"] = games.get(f"{p}game_sb", 0)
        new_cols[f"{side}_CS"] = games.get(f"{p}game_cs", 0)
        new_cols[f"{side}_GDP"] = games.get(f"{p}game_gidp", 0)
        new_cols[f"{side}_SH"] = sac
        new_cols[f"{side}_SF"] = sf

    return pd.concat([games, pd.DataFrame(new_cols, index=games.index)], axis=1)


def _aggregate_linescore_extended(linescore: pd.DataFrame) -> pd.DataFrame:
    """Sum errors and LOB across innings to game totals per side."""
    agg_cols = {}
    for side in ("home", "away"):
        err_col = f"{side}_errors"
        lob_col = f"{side}_left_on_base"
        if err_col in linescore.columns:
            linescore[err_col] = pd.to_numeric(linescore[err_col], errors="coerce")
            agg_cols[f"{side}_total_errors"] = linescore.groupby("game_pk")[err_col].sum()
        if lob_col in linescore.columns:
            linescore[lob_col] = pd.to_numeric(linescore[lob_col], errors="coerce")
            agg_cols[f"{side}_total_lob"] = linescore.groupby("game_pk")[lob_col].sum()

    if not agg_cols:
        return pd.DataFrame(columns=["game_pk"])

    result = pd.DataFrame(agg_cols).reset_index()
    result = result.rename(columns={"index": "game_pk"}) if "index" in result.columns else result
    for c in result.columns:
        if c != "game_pk":
            result[c] = result[c].astype("float32")
    log.debug(f"Linescore extended: {len(result):,} games")
    return result


def _aggregate_runners(
    runners: pd.DataFrame, pitches: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate baserunning events to game-level per side.

    Side assignment uses half_inning from pitches joined on (game_pk, play_index):
    "top" = away batting, "bottom" = home batting.
    """
    if runners.empty:
        return pd.DataFrame(columns=["game_pk"])

    # Build side mapping from pitches (one row per play_index per game)
    if "play_index" not in pitches.columns or "half_inning" not in pitches.columns:
        log.warning("pitches missing play_index/half_inning — skipping runners aggregation")
        return pd.DataFrame(columns=["game_pk"])

    side_map = (
        pitches[["game_pk", "play_index", "half_inning"]]
        .drop_duplicates(["game_pk", "play_index"])
    )

    r = runners.merge(side_map, on=["game_pk", "play_index"], how="left")
    r = r.dropna(subset=["half_inning"])

    # Map half_inning to batting side
    r["batting_side"] = r["half_inning"].map({"bottom": "home", "top": "away"})
    r = r.dropna(subset=["batting_side"])

    results = []
    for side in ("home", "away"):
        s = r[r["batting_side"] == side]

        # Stolen base attempts and successes
        sb_mask = s["event_type"].str.contains("stolen_base", case=False, na=False)
        cs_mask = s["event_type"].str.contains("caught_stealing", case=False, na=False)
        sb_success = sb_mask & ~cs_mask

        sb_attempts = s[sb_mask | cs_mask].groupby("game_pk").size()
        sb_successes = s[sb_success].groupby("game_pk").size()

        # Extra bases taken (non-forced advancement)
        adv_mask = s["movement_reason"].str.contains("r_adv_play", case=False, na=False)
        extra_base_taken = s[adv_mask].groupby("game_pk").size()
        extra_base_opps = s[(s["movement_reason"].notna()) & (s["movement_reason"] != "None")].groupby("game_pk").size()

        # First to third on single
        f2t_mask = (
            (s["movement_start"] == "1B") &
            (s["movement_end"] == "3B") &
            s["event_type"].str.contains("single", case=False, na=False)
        )
        first_to_third = s[f2t_mask].groupby("game_pk").size()

        # Score from second on single
        s2h_mask = (
            (s["movement_start"] == "2B") &
            (s["movement_end"].str.contains("score", case=False, na=False)) &
            s["event_type"].str.contains("single", case=False, na=False)
        )
        score_from_second = s[s2h_mask].groupby("game_pk").size()

        game_agg = pd.DataFrame({
            f"{side}_sb_attempts": sb_attempts,
            f"{side}_sb_successes": sb_successes,
            f"{side}_extra_base_taken": extra_base_taken,
            f"{side}_extra_base_opps": extra_base_opps,
            f"{side}_first_to_third": first_to_third,
            f"{side}_score_from_second": score_from_second,
        }).fillna(0).astype("float32")
        results.append(game_agg)

    out = pd.concat(results, axis=1).reset_index()
    out = out.rename(columns={"index": "game_pk"}) if "index" in out.columns else out
    log.debug(f"Runners aggregation: {len(out):,} games")
    return out
