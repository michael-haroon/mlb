"""Append new feature columns to existing game_features.parquet.

Computes ONLY the new feature families (batted ball, spin, command, spray,
platoon composition, baserunning, defense/stranding, pennant race) and joins
them as new columns to the existing parquet. Does NOT rebuild the full feature
set — existing columns are preserved unchanged.

Usage:
    conda run -n pred python -m classical_learning.engineering.append_new_features

Reads raw data from S3 (pitches, linescore, runners) to compute features,
then writes the expanded parquet in-place.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deep_learning"))

from classical_learning.engineering.data_loader import load_pitches_raw
from classical_learning.engineering.constants import LINESCORE_COLUMNS, RUNNERS_COLUMNS
from classical_learning.engineering.pitch_level_features import (
    _compute_batted_ball_features,
    _compute_spin_movement_features,
    _compute_command_features,
    _compute_spray_features,
    _compute_platoon_composition,
    _compute_bat_strength_features,
    _compute_bullpen_workload_features,
    _compute_manager_tendency_features,
)
from classical_learning.engineering.feature_engineering import (
    _baserunning_features,
    _defense_stranding_features,
    _pennant_race_features,
    _postseason_flag,
    _head_to_head_features,
    _travel_features,
)
from classical_learning.engineering.game_builder import (
    _aggregate_linescore_extended,
    _aggregate_runners,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# --- Config ---
FEATURES_PATH = Path("data/features/game_features.parquet")
S3_SOURCE = "s3://mlb-265753586044-us-east-1-an/data"
SEASON_START = 2015
SEASON_END = 2026


def main():
    t0 = time.time()

    # --- Load existing parquet ---
    log.info(f"Loading existing features from {FEATURES_PATH}...")
    existing = pd.read_parquet(FEATURES_PATH)
    n_existing_cols = len(existing.columns)
    log.info(f"  {len(existing):,} rows × {n_existing_cols} cols")

    # Check which new feature families are already computed
    has_batted_ball = any("barrel" in c for c in existing.columns)
    has_spin = any("avg_spin" in c and "roll" in c for c in existing.columns)
    has_command = any("whiff_rate" in c and "roll" in c for c in existing.columns)
    has_spray = any("pull_pct" in c for c in existing.columns)
    has_platoon_comp = any("platoon_advantage" in c for c in existing.columns)
    has_baserunning = any("sb_success_rate" in c for c in existing.columns)
    has_defense = any("strand_rate" in c for c in existing.columns)
    has_pennant = any("div_games_back" in c for c in existing.columns)
    # New feature groups (Groups A, B, C, D, E, F)
    has_bat_strength = any("bat_avg_hit_distance" in c or "bat_tb_per_hit" in c
                           for c in existing.columns)
    has_bullpen = any("bullpen_pitches_last3d" in c for c in existing.columns)
    has_mgr_tendency = any("mgr_pitchers_used" in c for c in existing.columns)
    has_h2h_new = any("h2h_home_winpct_last10" in c for c in existing.columns)
    has_travel = any("travel_km_since_last_game" in c for c in existing.columns)
    has_postseason = "is_postseason" in existing.columns

    # Force recompute ALL new features (multiple bugs fixed in engineering)
    if "--force-all" in sys.argv:
        log.info("Forcing full recompute — dropping ALL new feature columns")
        new_feature_markers = (
            "barrel", "hard_hit", "avg_ev", "sweet_spot", "_gb_rate", "_fb_rate", "_ld_rate",
            "avg_spin", "spin_rate", "spin_trend", "pfx_", "extension", "velo_retention",
            "whiff", "csw", "chase", "contact_rate", "first_strike", "first_pitch_strike", "zone_pct",
            "pull_pct", "center_pct", "oppo_pct",
            "platoon_advantage", "lhh_pct", "pitchmix_matchup",
            "sb_success_rate", "sb_attempts_pg", "extra_base", "first_to_third", "score_from_second",
            "strand_rate", "errors_pg", "lob_pg",
            "div_games_back", "wc_games_back", "in_contention", "season_progress", "diff_div",
            # Groups A / B / C / D / E / F
            "bat_avg_hit_distance", "bat_tb_per_hit",
            "bullpen_pitches_last", "bullpen_appearances_last",
            "h2h_home_winpct_last", "h2h_avg_total_runs", "h2h_avg_rd",
            "travel_km_since_last_game", "timezone_delta", "travel_fatigue_flag",
            "is_postseason",
            "mgr_pitchers_used", "mgr_bunt_rate",
        )
        drop_cols = [c for c in existing.columns if any(k in c for k in new_feature_markers)]
        if drop_cols:
            existing = existing.drop(columns=drop_cols)
            log.info(f"  Dropped {len(drop_cols)} columns for recompute")
        has_batted_ball = has_spin = has_command = has_spray = False
        has_platoon_comp = has_baserunning = has_defense = has_pennant = False
        has_bat_strength = has_bullpen = has_mgr_tendency = False
        has_h2h_new = has_travel = has_postseason = False

    # Force recompute command features only
    elif has_command and "--force-command" in sys.argv:
        log.info("Forcing command feature recompute — dropping old broken columns")
        cmd_cols = [c for c in existing.columns if any(k in c for k in
                    ("whiff", "csw", "chase", "contact_rate", "first_strike", "first_pitch_strike", "zone_pct"))]
        existing = existing.drop(columns=cmd_cols)
        log.info(f"  Dropped {len(cmd_cols)} old command columns")
        has_command = False

    need_pitch_level = not (has_batted_ball and has_spin and has_command and has_spray
                            and has_platoon_comp and has_bat_strength and has_bullpen
                            and has_mgr_tendency)
    need_game_level = not (has_baserunning and has_defense and has_pennant
                           and has_h2h_new and has_travel and has_postseason)

    if not need_pitch_level and not need_game_level:
        log.info("All new features already present — nothing to do")
        return

    new_feature_blocks: list[pd.DataFrame] = []

    # --- Pitch-level features (need raw pitches) ---
    if need_pitch_level:
        log.info("Loading pitches_raw from S3 for pitch-level features...")
        from mlb_dl.data_sources import ParquetCatalog, season_range
        pitches_raw = load_pitches_raw(S3_SOURCE, SEASON_START, SEASON_END)

        # Filter to regular season, exclude 2020
        pitches = pitches_raw[
            (pitches_raw["game_type_code"] == "R") & (pitches_raw["season"] != 2020)
        ].copy()
        pitches = pitches.sort_values(
            ["game_pk", "at_bat_index", "pitch_number"], na_position="last"
        ).reset_index(drop=True)
        del pitches_raw
        log.info(f"  {len(pitches):,} regular-season pitch rows")

        # game_frame subset needed by pitch-level functions
        gf_cols = ["game_pk", "game_date", "home_team_id", "away_team_id",
                   "probable_pitcher_home_id", "probable_pitcher_away_id"]
        gf_cols = [c for c in gf_cols if c in existing.columns]
        game_frame = existing[gf_cols].copy()

        # Derive SP hand from pitches (first pitch per half_inning per game)
        if "sp_home_hand" not in game_frame.columns:
            first_pitch = pitches.sort_values(["game_pk", "at_bat_index", "pitch_number"])
            # Home SP pitches in top of 1st
            top1 = first_pitch[first_pitch["half_inning"] == "top"].drop_duplicates("game_pk", keep="first")
            game_frame = game_frame.merge(
                top1[["game_pk", "pitch_hand_code"]].rename(columns={"pitch_hand_code": "sp_home_hand"}),
                on="game_pk", how="left"
            )
            # Away SP pitches in bottom of 1st
            bot1 = first_pitch[first_pitch["half_inning"] == "bottom"].drop_duplicates("game_pk", keep="first")
            game_frame = game_frame.merge(
                bot1[["game_pk", "pitch_hand_code"]].rename(columns={"pitch_hand_code": "sp_away_hand"}),
                on="game_pk", how="left"
            )
            log.info(f"  Derived SP hand from pitches: home={game_frame['sp_home_hand'].notna().sum()}, away={game_frame['sp_away_hand'].notna().sum()}")

        if not has_batted_ball:
            log.info("Computing batted ball features...")
            bb = _compute_batted_ball_features(pitches, game_frame)
            if len(bb.columns) > 1:
                new_feature_blocks.append(bb.drop(columns=["game_pk"]))
                log.info(f"  +{len(bb.columns) - 1} batted ball columns")

        if not has_spin:
            log.info("Computing spin & movement features...")
            sp = _compute_spin_movement_features(pitches, game_frame)
            if len(sp.columns) > 1:
                new_feature_blocks.append(sp.drop(columns=["game_pk"]))
                log.info(f"  +{len(sp.columns) - 1} spin/movement columns")

        if not has_command:
            log.info("Computing command features...")
            cmd = _compute_command_features(pitches, game_frame)
            if len(cmd.columns) > 1:
                new_feature_blocks.append(cmd.drop(columns=["game_pk"]))
                log.info(f"  +{len(cmd.columns) - 1} command columns")

        if not has_spray:
            log.info("Computing spray features...")
            spr = _compute_spray_features(pitches, game_frame)
            if len(spr.columns) > 1:
                new_feature_blocks.append(spr.drop(columns=["game_pk"]))
                log.info(f"  +{len(spr.columns) - 1} spray columns")

        if not has_platoon_comp:
            log.info("Computing platoon composition features...")
            plt = _compute_platoon_composition(pitches, game_frame)
            if len(plt.columns) > 1:
                new_feature_blocks.append(plt.drop(columns=["game_pk"]))
                log.info(f"  +{len(plt.columns) - 1} platoon composition columns")

        if not has_bat_strength:
            log.info("Computing bat strength features (hit distance, TB/hit)...")
            bstr = _compute_bat_strength_features(pitches, game_frame)
            if len(bstr.columns) > 1:
                new_feature_blocks.append(bstr.drop(columns=["game_pk"]))
                log.info(f"  +{len(bstr.columns) - 1} bat strength columns")

        if not has_bullpen:
            log.info("Computing bullpen workload features...")
            bwl = _compute_bullpen_workload_features(pitches, game_frame)
            if len(bwl.columns) > 1:
                new_feature_blocks.append(bwl.drop(columns=["game_pk"]))
                log.info(f"  +{len(bwl.columns) - 1} bullpen workload columns")

        if not has_mgr_tendency:
            log.info("Computing manager tendency features...")
            mgr = _compute_manager_tendency_features(pitches, game_frame)
            if len(mgr.columns) > 1:
                new_feature_blocks.append(mgr.drop(columns=["game_pk"]))
                log.info(f"  +{len(mgr.columns) - 1} manager tendency columns")

        del pitches

    # --- Game-level features (need linescore + runners raw data) ---
    if need_game_level:
        from mlb_dl.data_sources import ParquetCatalog, season_range

        catalog = ParquetCatalog(S3_SOURCE)
        seasons = season_range(SEASON_START, SEASON_END)

        if not has_baserunning or not has_defense:
            # Load linescore for errors/LOB
            log.info("Loading linescore from S3...")
            linescore = catalog.read_table("linescore", columns=LINESCORE_COLUMNS, seasons=seasons)
            log.info(f"  linescore: {len(linescore):,} rows")

            # Aggregate to game level
            linescore_ext = _aggregate_linescore_extended(linescore)
            del linescore

            # Load runners for baserunning
            log.info("Loading runners from S3...")
            runners = catalog.read_table("runners", columns=RUNNERS_COLUMNS, seasons=seasons)
            log.info(f"  runners: {len(runners):,} rows")

            # Only need 3 columns for runners side assignment (not full PITCH_META_COLUMNS)
            _SIDE_ASSIGN_COLS = ["game_pk", "play_index", "half_inning"]
            log.info("Loading pitches (3 cols) for runners side assignment...")
            pitches_meta = catalog.read_table("pitches", columns=_SIDE_ASSIGN_COLS, seasons=seasons)

            runners_agg = _aggregate_runners(runners, pitches_meta)
            del runners, pitches_meta

            # Merge game-level aggregates into existing for feature computation
            games_extended = existing.copy()
            if linescore_ext is not None and not linescore_ext.empty:
                for col in linescore_ext.columns:
                    if col != "game_pk" and col not in games_extended.columns:
                        merged = games_extended[["game_pk"]].merge(
                            linescore_ext[["game_pk", col]], on="game_pk", how="left"
                        )
                        games_extended[col] = merged[col].values

            if runners_agg is not None and not runners_agg.empty:
                for col in runners_agg.columns:
                    if col != "game_pk" and col not in games_extended.columns:
                        merged = games_extended[["game_pk"]].merge(
                            runners_agg[["game_pk", col]], on="game_pk", how="left"
                        )
                        games_extended[col] = merged[col].values

            del linescore_ext, runners_agg

            # Compute baserunning features
            if not has_baserunning:
                log.info("Computing baserunning features...")
                games_with_br = _baserunning_features(games_extended)
                br_cols = [c for c in games_with_br.columns if c not in games_extended.columns]
                if br_cols:
                    new_feature_blocks.append(games_with_br[br_cols].copy())
                    log.info(f"  +{len(br_cols)} baserunning columns")

            # Compute defense/stranding features
            if not has_defense:
                log.info("Computing defense & stranding features...")
                games_with_def = _defense_stranding_features(games_extended)
                def_cols = [c for c in games_with_def.columns if c not in games_extended.columns]
                if def_cols:
                    new_feature_blocks.append(games_with_def[def_cols].copy())
                    log.info(f"  +{len(def_cols)} defense/stranding columns")

            del games_extended

        # Pennant race features (standings snapshots + games_played from pitches)
        if not has_pennant:
            log.info("Loading standings snapshots from S3...")
            catalog = ParquetCatalog(S3_SOURCE)
            standings = catalog.read_table(
                "standings",
                columns=["date", "team_id", "games_back", "wild_card_games_back"],
                seasons=seasons,
            )
            log.info(f"  standings: {len(standings):,} rows")

            # Load games_played from pitches (season_progress uses it)
            if "home_games_played" not in existing.columns:
                _GP_COLS = ["game_pk", "home_games_played", "away_games_played"]
                pitches_gp = catalog.read_table("pitches", columns=_GP_COLS, seasons=seasons)
                gp_data = pitches_gp.drop_duplicates("game_pk")
                del pitches_gp
                games_for_pennant = existing[["game_pk", "home_team_id", "away_team_id", "game_date"]].copy()
                games_for_pennant = games_for_pennant.merge(gp_data, on="game_pk", how="left")
            else:
                games_for_pennant = existing[["game_pk", "home_team_id", "away_team_id", "game_date",
                                              "home_games_played", "away_games_played"]].copy()

            log.info("Computing pennant race features...")
            games_with_pennant = _pennant_race_features(games_for_pennant, standings=standings)
            _PENNANT_INTERMEDIATES = {"home_games_played", "away_games_played"}
            pennant_cols = [c for c in games_with_pennant.columns if c not in existing.columns
                           and c not in ["game_pk", "home_team_id", "away_team_id", "game_date"]
                           and c not in _PENNANT_INTERMEDIATES]
            if pennant_cols:
                new_feature_blocks.append(
                    games_with_pennant[pennant_cols].reset_index(drop=True)
                )
                log.info(f"  +{len(pennant_cols)} pennant race columns")
            del games_for_pennant, standings

        # H2H, travel, and postseason features operate on existing game_features only
        if not has_postseason:
            log.info("Computing postseason flag...")
            gf_post = _postseason_flag(existing[["game_pk", "game_type_code", "game_date"]
                                                  if "game_type_code" in existing.columns
                                                  else ["game_pk", "game_date"]].copy())
            post_cols = [c for c in gf_post.columns if c not in existing.columns
                         and c != "game_pk"]
            if post_cols:
                new_feature_blocks.append(
                    gf_post[post_cols].reset_index(drop=True)
                )
                log.info(f"  +{len(post_cols)} postseason flag columns")

        if not has_h2h_new:
            log.info("Computing H2H history features...")
            h2h_needed = ["game_pk", "home_team_id", "away_team_id",
                          "home_win", "home_runs", "away_runs", "game_date"]
            h2h_cols_avail = [c for c in h2h_needed if c in existing.columns]
            gf_h2h = _head_to_head_features(existing[h2h_cols_avail].copy())
            h2h_new_cols = [c for c in gf_h2h.columns if c not in existing.columns]
            if h2h_new_cols:
                new_feature_blocks.append(
                    gf_h2h[h2h_new_cols].reset_index(drop=True)
                )
                log.info(f"  +{len(h2h_new_cols)} H2H columns")

        if not has_travel:
            log.info("Computing travel and timezone features...")
            travel_needed = ["game_pk", "home_team_id", "away_team_id",
                             "venue_latitude", "venue_longitude",
                             "game_date", "is_night_game"]
            travel_cols_avail = [c for c in travel_needed if c in existing.columns]
            gf_travel = _travel_features(existing[travel_cols_avail].copy())
            travel_new_cols = [c for c in gf_travel.columns if c not in existing.columns]
            if travel_new_cols:
                new_feature_blocks.append(
                    gf_travel[travel_new_cols].reset_index(drop=True)
                )
                log.info(f"  +{len(travel_new_cols)} travel/timezone columns")

    # --- Join all new columns to existing ---
    if not new_feature_blocks:
        log.info("No new columns computed — nothing to append")
        return

    log.info(f"Appending {sum(len(b.columns) for b in new_feature_blocks)} new columns...")
    result = pd.concat([existing] + new_feature_blocks, axis=1)

    # Drop any leaky post-game columns that may have leaked in
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
    leaky = [c for c in result.columns if c in _POST_GAME_COLUMNS]
    if leaky:
        result.drop(columns=leaky, inplace=True)
        log.info(f"  Dropped {len(leaky)} post-game-only columns")

    # Write back
    result.to_parquet(FEATURES_PATH, index=False, engine="pyarrow")
    elapsed = time.time() - t0
    n_new = len(result.columns) - n_existing_cols + len(leaky)
    log.info(
        f"Done: {FEATURES_PATH} now {len(result):,} rows × {len(result.columns)} cols "
        f"(+{n_new} new features, {elapsed:.1f}s)"
    )


if __name__ == "__main__":
    main()
