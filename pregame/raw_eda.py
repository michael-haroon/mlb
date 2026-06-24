"""
Workflow 1: EDA on ALL 7 raw tables from S3 (PITCHES + BOXSCORE_BATTING + BOXSCORE_PITCHING + RUNNERS + LINESCORE + HITS + PLAYERS).

Two-phase approach:
Phase 1 (streaming): Process one season at a time
  - Per-season stats, distributions, temporal patterns
  - Save to CSVs, clear memory, move to next season
Phase 2 (aggregate): Combine per-season results
  - Global stats across all seasons
  - Seasonal drift (KS distance matrices)
  - Correlation structure
  - Relationship validation (cross-table integrity checks)
  - Generate HTML report with per-season + aggregate summaries

Tables analyzed:
1. PITCHES (170 columns, grain: pitch/play event)
2. BOXSCORE_BATTING (33 columns, grain: batter per game)
3. BOXSCORE_PITCHING (28 columns, grain: pitcher per game)
4. RUNNERS (20 columns, grain: runner per play)
5. LINESCORE (11 columns, grain: inning per game)
6. HITS (10 columns, grain: hit per game/inning)
7. PLAYERS (25 columns, grain: unique player, global)
"""

from __future__ import annotations

import gc
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


# ========== PITCHES Column Groups (170 columns) ==========
PITCHES_GAME_SEASON_COLS = [
    "game_pk", "season", "game_date", "game_datetime_utc", "game_number",
    "game_type_code", "double_header", "tiebreaker",
]

PITCHES_SERIES_COLS = [
    "series_description", "series_game_number", "games_in_series",
]

PITCHES_GAME_STATUS_COLS = [
    "game_status_detail", "game_status_code", "start_time_tbd",
]

PITCHES_VENUE_COLS = [
    "venue_id", "venue_name", "venue_city", "venue_state", "venue_latitude",
    "venue_longitude", "venue_timezone", "venue_tz_offset", "venue_capacity",
    "venue_surface", "venue_roof_type",
]

PITCHES_WEATHER_COLS = [
    "weather_condition", "weather_temp", "weather_wind",
]

PITCHES_HOME_TEAM_COLS = [
    "home_team_id", "home_team_name", "home_team_abbr", "home_league_id",
    "home_league_name", "home_division_id", "home_division_name", "home_wins",
    "home_losses", "home_win_pct", "home_division_games_back",
    "home_wild_card_games_back", "home_games_played",
]

PITCHES_AWAY_TEAM_COLS = [
    "away_team_id", "away_team_name", "away_team_abbr", "away_league_id",
    "away_league_name", "away_division_id", "away_division_name", "away_wins",
    "away_losses", "away_win_pct", "away_division_games_back",
    "away_wild_card_games_back", "away_games_played",
]

PITCHES_GAME_META_COLS = [
    "start_time", "day_night", "attendance", "game_duration_minutes",
    "umpire_hp", "umpire_1b", "umpire_2b", "umpire_3b",
]

PITCHES_PITCHING_DECISIONS_COLS = [
    "winner_pitcher_id", "winner_pitcher_name", "loser_pitcher_id",
    "loser_pitcher_name", "save_pitcher_id", "save_pitcher_name",
]

PITCHES_REVIEW_COLS = [
    "review_home_challenges_used", "review_home_challenges_remaining",
    "review_away_challenges_used", "review_away_challenges_remaining",
]

PITCHES_GAME_FLAGS_COLS = [
    "flag_no_hitter", "flag_perfect_game", "flag_away_team_no_hitter",
    "flag_home_team_no_hitter",
]

PITCHES_PROBABLE_PITCHERS_COLS = [
    "probable_pitcher_home_id", "probable_pitcher_away_id",
]

PITCHES_GAME_LEADERS_COLS = [
    "leader_hit_distance", "leader_hit_distance_player_id", "leader_hit_speed",
    "leader_hit_speed_player_id", "leader_pitch_speed", "leader_pitch_speed_player_id",
]

PITCHES_GAME_ALERTS_COLS = [
    "game_alerts_json",
]

PITCHES_AT_BAT_CONTEXT_COLS = [
    "play_index", "at_bat_index", "inning", "half_inning", "is_top_inning",
    "captivating_index", "at_bat_start_time", "at_bat_end_time",
    "at_bat_has_review", "at_bat_is_complete",
]

PITCHES_MATCHUP_COLS = [
    "batter_id", "batter_name", "bat_side_code", "pitcher_id", "pitcher_name",
    "pitch_hand_code",
]

PITCHES_SPLITS_COLS = [
    "split_batter", "split_pitcher", "men_on_base",
]

PITCHES_BASE_STATE_PRE_COLS = [
    "pre_on_first_id", "pre_on_second_id", "pre_on_third_id",
]

PITCHES_BASE_STATE_POST_COLS = [
    "post_on_first_id", "post_on_second_id", "post_on_third_id",
]

PITCHES_AT_BAT_OUTCOME_COLS = [
    "at_bat_event", "event_type", "is_scoring_play", "rbi_count",
    "score_home", "score_away", "play_description",
]

PITCHES_COUNT_COLS = [
    "cum_balls", "cum_strikes", "cum_outs",
]

PITCHES_PITCH_LEVEL_COLS = [
    "pitch_sequence_index", "play_id", "pitch_event_type", "is_pitch",
    "pitch_number", "pitch_start_time", "pitch_end_time", "pitch_count_balls",
    "pitch_count_strikes", "pitch_count_outs",
]

PITCHES_PITCH_TYPE_CALL_COLS = [
    "pitch_type", "pitch_call", "pitch_event_flags_json",
]

PITCHES_PITCH_CLASSIFICATION_COLS = [
    "is_in_play", "is_strike", "is_ball", "has_review",
]

PITCHES_PITCH_PHYSICS_COLS = [
    "release_speed", "end_speed", "strike_zone_top", "strike_zone_bottom",
    "type_confidence", "plate_time", "extension",
]

PITCHES_PITCH_LOCATION_COLS = [
    "coord_px", "coord_pz", "coord_x0", "coord_y0", "coord_z0", "coord_vx0",
    "coord_vy0", "coord_vz0", "coord_ax", "coord_ay", "coord_az",
]

PITCHES_PITCH_BREAK_SPIN_COLS = [
    "pfx_x", "pfx_z", "break_angle", "break_length", "break_y", "spin_rate",
    "spin_direction", "zone_location",
]

PITCHES_HIT_DATA_COLS = [
    "hit_launch_speed", "hit_launch_angle", "hit_total_distance", "hit_trajectory",
    "hit_hardness", "hit_coord_x", "hit_coord_y",
]

PITCHES_COLUMN_GROUP_MAP = {
    "game_season": PITCHES_GAME_SEASON_COLS,
    "series": PITCHES_SERIES_COLS,
    "game_status": PITCHES_GAME_STATUS_COLS,
    "venue": PITCHES_VENUE_COLS,
    "weather": PITCHES_WEATHER_COLS,
    "home_team": PITCHES_HOME_TEAM_COLS,
    "away_team": PITCHES_AWAY_TEAM_COLS,
    "game_meta": PITCHES_GAME_META_COLS,
    "pitching_decisions": PITCHES_PITCHING_DECISIONS_COLS,
    "review": PITCHES_REVIEW_COLS,
    "game_flags": PITCHES_GAME_FLAGS_COLS,
    "probable_pitchers": PITCHES_PROBABLE_PITCHERS_COLS,
    "game_leaders": PITCHES_GAME_LEADERS_COLS,
    "game_alerts": PITCHES_GAME_ALERTS_COLS,
    "at_bat_context": PITCHES_AT_BAT_CONTEXT_COLS,
    "matchup": PITCHES_MATCHUP_COLS,
    "splits": PITCHES_SPLITS_COLS,
    "base_state_pre": PITCHES_BASE_STATE_PRE_COLS,
    "base_state_post": PITCHES_BASE_STATE_POST_COLS,
    "at_bat_outcome": PITCHES_AT_BAT_OUTCOME_COLS,
    "count": PITCHES_COUNT_COLS,
    "pitch_level": PITCHES_PITCH_LEVEL_COLS,
    "pitch_type_call": PITCHES_PITCH_TYPE_CALL_COLS,
    "pitch_classification": PITCHES_PITCH_CLASSIFICATION_COLS,
    "pitch_physics": PITCHES_PITCH_PHYSICS_COLS,
    "pitch_location": PITCHES_PITCH_LOCATION_COLS,
    "pitch_break_spin": PITCHES_PITCH_BREAK_SPIN_COLS,
    "hit_data": PITCHES_HIT_DATA_COLS,
}


# ========== BOXSCORE_BATTING Column Groups (33 columns) ==========
BATTING_IDENTITY_COLS = [
    "game_pk", "season", "player_id", "player_name", "side", "batting_order",
    "all_positions_json", "is_substitute",
]

BATTING_GAME_STATS_COLS = [
    "game_ab", "game_runs", "game_hits", "game_doubles", "game_triples",
    "game_hr", "game_rbi", "game_bb", "game_ibb", "game_so", "game_sb",
    "game_cs", "game_hbp", "game_sac", "game_sf", "game_gidp", "game_lob",
]

BATTING_SEASON_STATS_COLS = [
    "season_avg", "season_obp", "season_slg", "season_ops", "season_hr",
    "season_rbi", "season_sb", "season_games_played",
]

BATTING_COLUMN_GROUP_MAP = {
    "identity": BATTING_IDENTITY_COLS,
    "game_stats": BATTING_GAME_STATS_COLS,
    "season_stats": BATTING_SEASON_STATS_COLS,
}


# ========== BOXSCORE_PITCHING Column Groups (28 columns) ==========
PITCHING_IDENTITY_COLS = [
    "game_pk", "season", "player_id", "player_name", "side", "is_starter",
]

PITCHING_GAME_STATS_COLS = [
    "game_innings_pitched", "game_hits", "game_runs", "game_earned_runs",
    "game_bb", "game_so", "game_hr", "game_hbp", "game_pitches_thrown",
    "game_strikes_thrown", "game_balls_thrown", "game_strikes_looking",
    "game_strikes_swinging",
]

PITCHING_SEASON_STATS_COLS = [
    "season_era", "season_whip", "season_wins", "season_losses",
    "season_saves", "season_innings_pitched", "season_so", "season_bb",
    "season_games_played",
]

PITCHING_COLUMN_GROUP_MAP = {
    "identity": PITCHING_IDENTITY_COLS,
    "game_stats": PITCHING_GAME_STATS_COLS,
    "season_stats": PITCHING_SEASON_STATS_COLS,
}


# ========== RUNNERS Column Groups (20 columns) ==========
RUNNERS_IDENTITY_COLS = [
    "game_pk", "season", "play_index", "play_event_index", "runner_id",
    "runner_name", "responsible_pitcher_id",
]

RUNNERS_MOVEMENT_COLS = [
    "movement_start", "movement_end",
]

RUNNERS_OUT_DETAIL_COLS = [
    "is_out", "out_base", "out_number",
]

RUNNERS_SCORING_COLS = [
    "is_scoring_event", "rbi", "earned", "team_unearned",
]

RUNNERS_PLAY_CONTEXT_COLS = [
    "event", "event_type", "movement_reason", "credits_json",
]

RUNNERS_COLUMN_GROUP_MAP = {
    "identity": RUNNERS_IDENTITY_COLS,
    "movement": RUNNERS_MOVEMENT_COLS,
    "out_detail": RUNNERS_OUT_DETAIL_COLS,
    "scoring": RUNNERS_SCORING_COLS,
    "play_context": RUNNERS_PLAY_CONTEXT_COLS,
}


# ========== LINESCORE Column Groups (11 columns) ==========
LINESCORE_IDENTITY_COLS = [
    "game_pk", "season", "inning",
]

LINESCORE_RUNS_HITS_ERRORS_COLS = [
    "home_runs", "away_runs", "home_hits", "away_hits",
    "home_errors", "away_errors",
]

LINESCORE_LOB_COLS = [
    "home_left_on_base", "away_left_on_base",
]

LINESCORE_COLUMN_GROUP_MAP = {
    "identity": LINESCORE_IDENTITY_COLS,
    "runs_hits_errors": LINESCORE_RUNS_HITS_ERRORS_COLS,
    "lob": LINESCORE_LOB_COLS,
}


# ========== HITS Column Groups (10 columns) ==========
HITS_IDENTITY_COLS = [
    "game_pk", "season", "inning", "side", "batter_id", "pitcher_id", "team_id",
]

HITS_TYPE_COLS = [
    "hit_type",
]

HITS_COORDINATES_COLS = [
    "hit_x", "hit_y",
]

HITS_COLUMN_GROUP_MAP = {
    "identity": HITS_IDENTITY_COLS,
    "type": HITS_TYPE_COLS,
    "coordinates": HITS_COORDINATES_COLS,
}


# ========== PLAYERS Column Groups (25 columns) ==========
PLAYERS_NAME_COLS = [
    "player_id", "full_name", "use_name", "boxscore_name", "first_name", "last_name",
]

PLAYERS_PHYSICAL_COLS = [
    "primary_number", "birth_date", "birth_city", "birth_state", "birth_country",
    "height", "weight", "current_age",
]

PLAYERS_POSITION_COLS = [
    "position_code", "position_name", "position_type", "position_abbreviation",
]

PLAYERS_HANDEDNESS_COLS = [
    "bat_side", "pitch_hand", "strike_zone_top", "strike_zone_bottom",
]

PLAYERS_CAREER_COLS = [
    "mlb_debut_date", "draft_year", "is_active",
]

PLAYERS_COLUMN_GROUP_MAP = {
    "name": PLAYERS_NAME_COLS,
    "physical": PLAYERS_PHYSICAL_COLS,
    "position": PLAYERS_POSITION_COLS,
    "handedness": PLAYERS_HANDEDNESS_COLS,
    "career": PLAYERS_CAREER_COLS,
}


def _setup_logging() -> None:
    if log.handlers:
        return
    log.setLevel(logging.DEBUG)

    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_dir / "raw_eda.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    log.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(sh)


def run_raw_eda(source_uri: str, output_dir: str, seasons: list[int] | None = None) -> dict:
    """Orchestrate raw EDA for ALL 7 tables with per-season + aggregate analysis."""
    import pandas as pd
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "live"))
    from mlb_dl.data_sources import ParquetCatalog, season_range

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _setup_logging()
    t0 = time.time()

    log.info("Starting raw EDA (ALL 7 tables) — source=%s, output=%s", source_uri, output_dir)

    for subdir in ("per_season", "aggregate", "players", "relationships", "plots"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    catalog = ParquetCatalog(source_uri)
    if seasons is None:
        # Default to reasonable recent years if not specified
        all_seasons = season_range(2015, None)
    else:
        all_seasons = seasons if isinstance(seasons, list) else season_range(min(seasons), max(seasons))

    log.info("Found %d seasons: %s", len(all_seasons), all_seasons)

    # --- PHASE 1: Per-season streaming analysis ---
    log.info("PHASE 1: Per-season streaming analysis")

    # Don't accumulate — save directly and discard per-season results immediately
    for season in all_seasons:
        log.info("Processing season %d ...", season)
        t_season = time.time()

        # --- PITCHES (streaming, same as before) ---
        log.info("  Table: PITCHES")
        pitches = catalog.read_table("pitches", seasons=[season])
        log.debug("    Loaded %d rows", len(pitches))

        num_cols = pitches.select_dtypes(include=["number"]).columns
        pitches[num_cols] = pitches[num_cols].astype("float64")

        grouped_cols = {}
        for group_name, cols in PITCHES_COLUMN_GROUP_MAP.items():
            grouped_cols[group_name] = [c for c in cols if c in pitches.columns]

        n_pitches = len(pitches)
        n_games = pitches["game_pk"].nunique()
        n_actual_pitches = pitches["is_pitch"].sum() if "is_pitch" in pitches.columns else 0
        game_counts = pitches.groupby("game_pk").size()
        anomalies = (game_counts < 10).sum()

        game_row = {
            "season": season,
            "total_rows": n_pitches,
            "actual_pitches": n_actual_pitches,
            "num_games": n_games,
            "rows_per_game_mean": round(n_pitches / max(n_games, 1), 1),
            "anomalous_games": anomalies,
        }

        numeric_stats = analyze_all_numeric_columns(pitches, grouped_cols, season)
        categorical_stats = analyze_all_categorical_columns(pitches, grouped_cols, season)
        hit_stats = analyze_hit_data(pitches, grouped_cols["hit_data"], season)
        temporal_stats = analyze_temporal_patterns(pitches, season)

        (output_dir / "per_season" / f"pitches_game_stats_{season}.csv").write_text(
            "\n".join(f"{k},{v}" for k, v in game_row.items())
        )
        if len(numeric_stats) > 0:
            numeric_stats.to_csv(output_dir / "per_season" / f"pitches_numeric_{season}.csv", index=False)
        if len(categorical_stats) > 0:
            categorical_stats.to_csv(output_dir / "per_season" / f"pitches_categorical_{season}.csv", index=False)
        if len(hit_stats) > 0:
            hit_stats.to_csv(output_dir / "per_season" / f"pitches_hit_data_{season}.csv", index=False)
        if len(temporal_stats) > 0:
            temporal_stats.to_csv(output_dir / "per_season" / f"pitches_temporal_{season}.csv", index=False)

        del pitches, game_counts
        gc.collect()

        # --- BOXSCORE_BATTING ---
        log.info("  Table: BOXSCORE_BATTING")
        batting = catalog.read_table("boxscore_batting", seasons=[season])
        log.debug("    Loaded %d rows", len(batting))

        batting_stats = analyze_table_per_season(batting, BATTING_COLUMN_GROUP_MAP, "batting", season)
        if len(batting_stats) > 0:
            batting_stats.to_csv(output_dir / "per_season" / f"batting_{season}.csv", index=False)

        del batting
        gc.collect()

        # --- BOXSCORE_PITCHING ---
        log.info("  Table: BOXSCORE_PITCHING")
        pitching = catalog.read_table("boxscore_pitching", seasons=[season])
        log.debug("    Loaded %d rows", len(pitching))

        pitching_stats = analyze_table_per_season(pitching, PITCHING_COLUMN_GROUP_MAP, "pitching", season)
        if len(pitching_stats) > 0:
            pitching_stats.to_csv(output_dir / "per_season" / f"pitching_{season}.csv", index=False)

        del pitching
        gc.collect()

        # --- RUNNERS ---
        log.info("  Table: RUNNERS")
        runners = catalog.read_table("runners", seasons=[season])
        log.debug("    Loaded %d rows", len(runners))

        runners_stats = analyze_table_per_season(runners, RUNNERS_COLUMN_GROUP_MAP, "runners", season)
        if len(runners_stats) > 0:
            runners_stats.to_csv(output_dir / "per_season" / f"runners_{season}.csv", index=False)

        del runners
        gc.collect()

        # --- LINESCORE ---
        log.info("  Table: LINESCORE")
        linescore = catalog.read_table("linescore", seasons=[season])
        log.debug("    Loaded %d rows", len(linescore))

        linescore_stats = analyze_table_per_season(linescore, LINESCORE_COLUMN_GROUP_MAP, "linescore", season)
        if len(linescore_stats) > 0:
            linescore_stats.to_csv(output_dir / "per_season" / f"linescore_{season}.csv", index=False)

        del linescore
        gc.collect()

        # --- HITS ---
        log.info("  Table: HITS")
        hits = catalog.read_table("hits", seasons=[season])
        log.debug("    Loaded %d rows", len(hits))

        hits_stats = analyze_table_per_season(hits, HITS_COLUMN_GROUP_MAP, "hits", season)
        if len(hits_stats) > 0:
            hits_stats.to_csv(output_dir / "per_season" / f"hits_{season}.csv", index=False)

        del hits
        gc.collect()

        log.info("  Season %d complete (%.1fs)", season, time.time() - t_season)

    # --- PLAYERS (global, not per-season) ---
    log.info("PHASE 1b: Global PLAYERS analysis")
    players = catalog.read_table("players", seasons=None)
    log.debug("  Loaded %d players", len(players))

    players_stats = analyze_players_global(players)
    if len(players_stats) > 0:
        players_stats.to_csv(output_dir / "players" / "global_analysis.csv", index=False)

    del players, players_stats
    gc.collect()

    # Per-season CSVs already saved during Phase 1. Skip aggregate concatenation to avoid memory explosion.
    # For large date ranges (e.g., 1950–2026), concat would require ~50+ GB.
    log.info("PHASE 2: Skipping aggregate analysis for large date ranges (>20 years)")

    # --- PHASE 2: Aggregate analyses + relationships (SKIP for large ranges) ---
    if len(all_seasons) > 20:
        log.info("PHASE 2: Skipped (too many seasons: %d). Use per_season CSVs directly.", len(all_seasons))
        # Still generate HTML report
        log.info("Building HTML report ...")
        html_path = build_raw_eda_html_report(output_dir, all_seasons)
        elapsed = time.time() - t0
        log.info("Raw EDA complete in %.1fs — report: %s", elapsed, html_path)
        return {
            "index_html": str(html_path),
            "output_dir": str(output_dir),
            "seasons_processed": len(all_seasons),
            "elapsed_secs": round(elapsed, 1),
        }

    # Reload full datasets for cross-table validation (only for <=20 years)
    pitches_full = catalog.read_table("pitches", seasons=all_seasons)
    num_cols_full = pitches_full.select_dtypes(include=["number"]).columns
    pitches_full[num_cols_full] = pitches_full[num_cols_full].astype("float64")

    batting_full = catalog.read_table("boxscore_batting", seasons=all_seasons)
    pitching_full = catalog.read_table("boxscore_pitching", seasons=all_seasons)
    runners_full = catalog.read_table("runners", seasons=all_seasons)
    linescore_full = catalog.read_table("linescore", seasons=all_seasons)
    hits_full = catalog.read_table("hits", seasons=all_seasons)

    # Seasonal drift (physics columns from PITCHES)
    log.info("Computing seasonal drift (KS distances) ...")
    grouped_cols = {}
    for group_name, cols in PITCHES_COLUMN_GROUP_MAP.items():
        grouped_cols[group_name] = [c for c in cols if c in pitches_full.columns]
    physics_cols = [c for c in grouped_cols["pitch_physics"] + grouped_cols["pitch_break_spin"] if c in pitches_full.columns]

    seasonal_drift_df = analyze_seasonal_drift(pitches_full, physics_cols)
    if len(seasonal_drift_df) > 0:
        seasonal_drift_df.to_csv(output_dir / "aggregate" / "pitches_seasonal_drift_ks.csv", index=False)

    # Correlation structure (physics columns from PITCHES)
    log.info("Computing correlation structure ...")
    correlations_df = analyze_correlations(pitches_full, physics_cols)
    if len(correlations_df) > 0:
        correlations_df.to_csv(output_dir / "aggregate" / "pitches_correlations.csv", index=False)

    # Timestamp quality (PITCHES)
    log.info("Analyzing timestamp quality ...")
    timestamp_quality_df = analyze_timestamp_quality(pitches_full)
    if len(timestamp_quality_df) > 0:
        timestamp_quality_df.to_csv(output_dir / "aggregate" / "pitches_timestamp_quality.csv", index=False)

    # Relationship validation
    log.info("Validating cross-table relationships ...")
    relationships_df = analyze_relationships(pitches_full, batting_full, pitching_full, runners_full, linescore_full, hits_full, players)
    if len(relationships_df) > 0:
        relationships_df.to_csv(output_dir / "relationships" / "cross_table_validation.csv", index=False)

    del pitches_full, batting_full, pitching_full, runners_full, linescore_full, hits_full, players
    gc.collect()

    # --- Generate plots ---
    log.info("Generating plots ...")
    try:
        import matplotlib
        matplotlib.use("Agg")

        pitches_plot = catalog.read_table("pitches", seasons=all_seasons)
        num_cols_plot = pitches_plot.select_dtypes(include=["number"]).columns
        pitches_plot[num_cols_plot] = pitches_plot[num_cols_plot].astype("float64")

        grouped_cols_plot = {}
        for group_name, cols in PITCHES_COLUMN_GROUP_MAP.items():
            grouped_cols_plot[group_name] = [c for c in cols if c in pitches_plot.columns]
        physics_cols_plot = [c for c in grouped_cols_plot["pitch_physics"] + grouped_cols_plot["pitch_break_spin"] if c in pitches_plot.columns]

        plot_physics_distributions(pitches_plot, physics_cols_plot, output_dir / "plots")
        plot_seasonal_distributions(pitches_plot, physics_cols_plot, output_dir / "plots")

        # Plot hit coordinates heatmap
        hits_plot = catalog.read_table("hits", seasons=all_seasons)
        plot_hit_coordinates(hits_plot, output_dir / "plots")

        # Plot LINESCORE runs by inning
        linescore_plot = catalog.read_table("linescore", seasons=all_seasons)
        plot_linescore_distributions(linescore_plot, output_dir / "plots")

        del pitches_plot, hits_plot, linescore_plot
        gc.collect()
    except Exception as e:
        log.warning("Plot generation failed: %s", e)

    # --- Build HTML report ---
    log.info("Building HTML report ...")
    html_path = build_raw_eda_html_report(output_dir, all_seasons)

    elapsed = time.time() - t0
    log.info("Raw EDA complete in %.1fs — report: %s", elapsed, html_path)

    return {
        "index_html": str(html_path),
        "output_dir": str(output_dir),
        "seasons_processed": len(all_seasons),
        "elapsed_secs": round(elapsed, 1),
    }


def analyze_all_numeric_columns(df, grouped_cols, season):
    """Compute stats for all numeric columns, grouped by category."""
    import numpy as np
    import pandas as pd
    from scipy import stats as spstats

    rows = []
    for group_name, cols in grouped_cols.items():
        for col in cols:
            if col not in df.columns:
                continue
            if df[col].dtype not in ["int64", "float64"]:
                continue

            values = df[col].dropna().to_numpy(dtype="float64")
            values = values[np.isfinite(values)]

            n_total = len(df)
            n_valid = len(values)
            n_nan = n_total - n_valid
            pct_nan = round(100.0 * n_nan / max(n_total, 1), 2)

            if n_valid < 3:
                rows.append({
                    "season": season,
                    "group": group_name,
                    "col_name": col,
                    "n_valid": n_valid,
                    "pct_nan": pct_nan,
                    "pct_zero": None,
                    "mean": None,
                    "std": None,
                    "p5": None,
                    "p25": None,
                    "p50": None,
                    "p75": None,
                    "p95": None,
                    "skewness": None,
                    "kurtosis": None,
                })
                continue

            pct_zero = round(100.0 * float(np.sum(values == 0.0)) / n_valid, 2)
            mean = float(np.mean(values))
            std = float(np.std(values))
            p5, p25, p50, p75, p95 = [float(x) for x in np.percentile(values, [5, 25, 50, 75, 95])]
            skewness = float(spstats.skew(values))
            kurtosis = float(spstats.kurtosis(values))

            rows.append({
                "season": season,
                "group": group_name,
                "col_name": col,
                "n_valid": n_valid,
                "pct_nan": pct_nan,
                "pct_zero": pct_zero,
                "mean": round(mean, 4),
                "std": round(std, 4),
                "p5": round(p5, 4),
                "p25": round(p25, 4),
                "p50": round(p50, 4),
                "p75": round(p75, 4),
                "p95": round(p95, 4),
                "skewness": round(skewness, 4),
                "kurtosis": round(kurtosis, 4),
            })
            log.debug("    Numeric: %s [group=%s] n=%d, mean=%.2f", col, group_name, n_valid, mean)

    return pd.DataFrame(rows)


def analyze_all_categorical_columns(df, grouped_cols, season):
    """Compute cardinality and top-5 for all categorical columns, grouped by category."""
    import pandas as pd

    rows = []
    for group_name, cols in grouped_cols.items():
        for col in cols:
            if col not in df.columns:
                continue
            if df[col].dtype not in ["object", "bool", "category"]:
                continue

            n_total = len(df)
            n_valid = df[col].notna().sum()
            pct_nan = round(100.0 * (n_total - n_valid) / max(n_total, 1), 2)

            if df[col].dtype == "bool":
                n_true = df[col].sum()
                pct_true = round(100.0 * n_true / max(n_valid, 1), 2)
                rows.append({
                    "season": season,
                    "group": group_name,
                    "col_name": col,
                    "type": "boolean",
                    "n_valid": n_valid,
                    "pct_nan": pct_nan,
                    "cardinality": 2,
                    "pct_true": pct_true,
                    "top_value": None,
                    "top_count": None,
                })
            else:
                cardinality = df[col].nunique()
                value_counts = df[col].value_counts().head(5)
                top_value = value_counts.index[0] if len(value_counts) > 0 else None
                top_count = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0

                rows.append({
                    "season": season,
                    "group": group_name,
                    "col_name": col,
                    "type": "categorical",
                    "n_valid": n_valid,
                    "pct_nan": pct_nan,
                    "cardinality": cardinality,
                    "pct_true": None,
                    "top_value": str(top_value) if top_value is not None else None,
                    "top_count": top_count,
                })
            log.debug("    Categorical: %s [group=%s] cardinality=%d", col, group_name, cardinality if df[col].dtype != "bool" else 2)

    return pd.DataFrame(rows)


def analyze_hit_data(df, hit_cols, season):
    """Analyze hit data columns (only for is_in_play=True rows)."""
    import numpy as np
    import pandas as pd

    if "is_in_play" not in df.columns:
        log.debug("    Hit data: is_in_play column not found")
        return pd.DataFrame()

    hit_subset = df[df["is_in_play"] == True]
    n_hits = len(hit_subset)
    log.debug("    Hit data: %d is_in_play rows", n_hits)

    if n_hits < 3:
        return pd.DataFrame()

    rows = []
    for col in hit_cols:
        if col not in hit_subset.columns:
            continue

        if hit_subset[col].dtype in ["int64", "float64"]:
            values = hit_subset[col].dropna().to_numpy(dtype="float64")
            values = values[np.isfinite(values)]

            if len(values) < 3:
                continue

            rows.append({
                "season": season,
                "col_name": col,
                "n_valid": len(values),
                "pct_nan": round(100.0 * (n_hits - len(values)) / max(n_hits, 1), 2),
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
                "p5": round(float(np.percentile(values, 5)), 4),
                "p50": round(float(np.percentile(values, 50)), 4),
                "p95": round(float(np.percentile(values, 95)), 4),
            })
        elif hit_subset[col].dtype in ["object", "category"]:
            cardinality = hit_subset[col].nunique()
            value_counts = hit_subset[col].value_counts().head(3)
            rows.append({
                "season": season,
                "col_name": col,
                "n_valid": hit_subset[col].notna().sum(),
                "cardinality": cardinality,
                "top_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "top_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            })

    return pd.DataFrame(rows)


def analyze_temporal_patterns(df, season):
    """Velocity by inning, count distributions, etc."""
    import numpy as np
    import pandas as pd

    data = []

    # Velocity by inning (fatigue validation)
    if "release_speed" in df.columns and "inning" in df.columns:
        df_actual = df[df.get("is_pitch", True) == True]
        for inning in sorted(df_actual["inning"].dropna().unique()):
            inning_data = df_actual[df_actual["inning"] == inning]
            velocities = inning_data["release_speed"].dropna().to_numpy(dtype="float64")
            velocities = velocities[np.isfinite(velocities)]

            if len(velocities) > 0:
                data.append({
                    "season": season,
                    "pattern": f"velocity_inning_{int(inning)}",
                    "mean": round(float(np.mean(velocities)), 2),
                    "std": round(float(np.std(velocities)), 2),
                    "count": len(velocities),
                })

    # Count distributions
    if "cum_balls" in df.columns and "cum_strikes" in df.columns:
        for b in range(4):
            for s in range(3):
                count_subset = df[(df["cum_balls"] == b) & (df["cum_strikes"] == s)]
                if len(count_subset) > 0:
                    data.append({
                        "season": season,
                        "pattern": f"count_{b}_{s}",
                        "count": len(count_subset),
                        "freq_pct": round(100.0 * len(count_subset) / len(df), 2),
                    })

    return pd.DataFrame(data)


def analyze_table_per_season(df, column_group_map, table_name, season):
    """Generic per-season analysis for any table."""
    import pandas as pd

    if len(df) == 0:
        return pd.DataFrame()

    grouped_cols = {}
    for group_name, cols in column_group_map.items():
        grouped_cols[group_name] = [c for c in cols if c in df.columns]

    numeric_stats = analyze_all_numeric_columns(df, grouped_cols, season)
    categorical_stats = analyze_all_categorical_columns(df, grouped_cols, season)

    # Combine
    combined = pd.concat([numeric_stats, categorical_stats], ignore_index=True) if len(numeric_stats) > 0 or len(categorical_stats) > 0 else pd.DataFrame()
    if len(combined) > 0:
        combined.insert(0, "table", table_name)

    return combined


def analyze_players_global(players):
    """Analyze PLAYERS table (global, not per-season)."""
    import numpy as np
    import pandas as pd

    if len(players) == 0:
        return pd.DataFrame()

    rows = []

    # Cardinality of position, handedness
    for col in ["position_code", "position_name", "bat_side", "pitch_hand", "is_active"]:
        if col not in players.columns:
            continue
        cardinality = players[col].nunique()
        value_counts = players[col].value_counts().head(10)
        for idx, val in enumerate(value_counts.index):
            rows.append({
                "metric": f"{col}_top{idx+1}",
                "value": str(val),
                "count": int(value_counts.iloc[idx]),
                "pct": round(100.0 * value_counts.iloc[idx] / len(players), 2),
            })

    # Age distribution
    if "current_age" in players.columns:
        ages = players["current_age"].dropna().to_numpy(dtype="float64")
        ages = ages[np.isfinite(ages)]
        if len(ages) > 0:
            rows.append({
                "metric": "age_mean",
                "value": round(float(np.mean(ages)), 1),
                "count": len(ages),
                "pct": None,
            })
            rows.append({
                "metric": "age_p50",
                "value": round(float(np.percentile(ages, 50)), 1),
                "count": None,
                "pct": None,
            })

    # Weight distribution
    if "weight" in players.columns:
        weights = players["weight"].dropna().to_numpy(dtype="float64")
        weights = weights[np.isfinite(weights)]
        if len(weights) > 0:
            rows.append({
                "metric": "weight_mean",
                "value": round(float(np.mean(weights)), 1),
                "count": len(weights),
                "pct": None,
            })

    # Draft year distribution
    if "draft_year" in players.columns:
        draft_years = players[players["draft_year"] > 1900]["draft_year"].value_counts().sort_index()
        if len(draft_years) > 0:
            rows.append({
                "metric": "draft_year_earliest",
                "value": int(draft_years.index[0]),
                "count": None,
                "pct": None,
            })
            rows.append({
                "metric": "draft_year_latest",
                "value": int(draft_years.index[-1]),
                "count": None,
                "pct": None,
            })

    log.debug("  Players: %d metrics computed", len(rows))
    return pd.DataFrame(rows)


def analyze_seasonal_drift(df, physics_cols):
    """KS distance matrices between seasons for physics columns."""
    import numpy as np
    import pandas as pd
    from scipy import stats as spstats

    seasons = sorted(df["season"].dropna().unique().astype(int))
    data = []

    for col in physics_cols:
        if col not in df.columns:
            continue
        if len(seasons) < 2:
            continue

        for s1_idx, s1 in enumerate(seasons):
            v1 = df[df["season"] == s1][col].dropna().to_numpy(dtype="float64")
            v1 = v1[np.isfinite(v1)]

            for s2 in seasons[s1_idx + 1:]:
                v2 = df[df["season"] == s2][col].dropna().to_numpy(dtype="float64")
                v2 = v2[np.isfinite(v2)]

                if len(v1) >= 2 and len(v2) >= 2:
                    ks_stat, ks_pval = spstats.ks_2samp(v1, v2)
                    data.append({
                        "col_name": col,
                        "season_1": s1,
                        "season_2": s2,
                        "ks_stat": round(float(ks_stat), 4),
                        "ks_pvalue": round(float(ks_pval), 4),
                    })

    log.debug("  Seasonal drift: %d KS pairs computed", len(data))
    return pd.DataFrame(data)


def analyze_correlations(df, physics_cols):
    """Spearman correlation among physics features."""
    import numpy as np
    import pandas as pd
    import warnings

    physics_cols = [c for c in physics_cols if c in df.columns]
    if len(physics_cols) < 2:
        return pd.DataFrame()

    # Filter to actual pitches only
    if "is_pitch" in df.columns:
        df = df[df["is_pitch"] == True]

    subset = df[physics_cols].dropna()
    if len(subset) < 10:
        return pd.DataFrame()

    # Subsample for speed
    if len(subset) > 50_000:
        subset = subset.sample(n=50_000, random_state=42)

    data = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, c1 in enumerate(physics_cols):
            for c2 in physics_cols[i + 1:]:
                corr = subset[[c1, c2]].corr(method="spearman").iloc[0, 1]
                data.append({
                    "var_1": c1,
                    "var_2": c2,
                    "spearman_rho": round(float(corr), 4),
                })

    log.debug("  Correlations: %d pairs computed", len(data))
    return pd.DataFrame(data)


def analyze_timestamp_quality(df):
    """Timestamp validity, inter-pitch Δt distribution."""
    import pandas as pd

    data = []

    if "pitch_start_time" in df.columns:
        valid = df["pitch_start_time"].notna().sum()
        total = len(df)
        data.append({
            "metric": "pitch_start_time_valid_pct",
            "value": round(100.0 * valid / max(total, 1), 2),
        })

    if "at_bat_start_time" in df.columns:
        valid = df["at_bat_start_time"].notna().sum()
        total = len(df)
        data.append({
            "metric": "at_bat_start_time_valid_pct",
            "value": round(100.0 * valid / max(total, 1), 2),
        })

    log.debug("  Timestamp quality: %d metrics computed", len(data))
    return pd.DataFrame(data)


def analyze_relationships(pitches, batting, pitching, runners, linescore, hits, players):
    """Validate cross-table relationships (foreign keys, score consistency, etc.)."""
    import pandas as pd

    data = []

    # BOXSCORE_BATTING.game_pk ⊆ PITCHES.game_pk
    pitches_games = set(pitches["game_pk"].unique())
    batting_games = set(batting["game_pk"].unique())
    batting_games_in_pitches = batting_games.intersection(pitches_games)
    data.append({
        "relationship": "BATTING.game_pk ⊆ PITCHES.game_pk",
        "left_count": len(batting_games),
        "right_count": len(pitches_games),
        "intersection_count": len(batting_games_in_pitches),
        "pct_match": round(100.0 * len(batting_games_in_pitches) / max(len(batting_games), 1), 2),
    })

    # BOXSCORE_BATTING.player_id ⊆ PLAYERS.player_id
    players_ids = set(players["player_id"].unique())
    batting_player_ids = set(batting["player_id"].unique())
    batting_players_in_players = batting_player_ids.intersection(players_ids)
    data.append({
        "relationship": "BATTING.player_id ⊆ PLAYERS.player_id",
        "left_count": len(batting_player_ids),
        "right_count": len(players_ids),
        "intersection_count": len(batting_players_in_players),
        "pct_match": round(100.0 * len(batting_players_in_players) / max(len(batting_player_ids), 1), 2),
    })

    # BOXSCORE_PITCHING.game_pk ⊆ PITCHES.game_pk
    pitching_games = set(pitching["game_pk"].unique())
    pitching_games_in_pitches = pitching_games.intersection(pitches_games)
    data.append({
        "relationship": "PITCHING.game_pk ⊆ PITCHES.game_pk",
        "left_count": len(pitching_games),
        "right_count": len(pitches_games),
        "intersection_count": len(pitching_games_in_pitches),
        "pct_match": round(100.0 * len(pitching_games_in_pitches) / max(len(pitching_games), 1), 2),
    })

    # BOXSCORE_PITCHING.player_id ⊆ PLAYERS.player_id
    pitching_player_ids = set(pitching["player_id"].unique())
    pitching_players_in_players = pitching_player_ids.intersection(players_ids)
    data.append({
        "relationship": "PITCHING.player_id ⊆ PLAYERS.player_id",
        "left_count": len(pitching_player_ids),
        "right_count": len(players_ids),
        "intersection_count": len(pitching_players_in_players),
        "pct_match": round(100.0 * len(pitching_players_in_players) / max(len(pitching_player_ids), 1), 2),
    })

    # RUNNERS.game_pk ⊆ PITCHES.game_pk
    runners_games = set(runners["game_pk"].unique())
    runners_games_in_pitches = runners_games.intersection(pitches_games)
    data.append({
        "relationship": "RUNNERS.game_pk ⊆ PITCHES.game_pk",
        "left_count": len(runners_games),
        "right_count": len(pitches_games),
        "intersection_count": len(runners_games_in_pitches),
        "pct_match": round(100.0 * len(runners_games_in_pitches) / max(len(runners_games), 1), 2),
    })

    # RUNNERS.runner_id ⊆ PLAYERS.player_id
    runners_player_ids = set(runners["runner_id"].unique())
    runners_players_in_players = runners_player_ids.intersection(players_ids)
    data.append({
        "relationship": "RUNNERS.runner_id ⊆ PLAYERS.player_id",
        "left_count": len(runners_player_ids),
        "right_count": len(players_ids),
        "intersection_count": len(runners_players_in_players),
        "pct_match": round(100.0 * len(runners_players_in_players) / max(len(runners_player_ids), 1), 2),
    })

    # LINESCORE.game_pk ⊆ PITCHES.game_pk
    linescore_games = set(linescore["game_pk"].unique())
    linescore_games_in_pitches = linescore_games.intersection(pitches_games)
    data.append({
        "relationship": "LINESCORE.game_pk ⊆ PITCHES.game_pk",
        "left_count": len(linescore_games),
        "right_count": len(pitches_games),
        "intersection_count": len(linescore_games_in_pitches),
        "pct_match": round(100.0 * len(linescore_games_in_pitches) / max(len(linescore_games), 1), 2),
    })

    # HITS.game_pk ⊆ PITCHES.game_pk
    hits_games = set(hits["game_pk"].unique())
    hits_games_in_pitches = hits_games.intersection(pitches_games)
    data.append({
        "relationship": "HITS.game_pk ⊆ PITCHES.game_pk",
        "left_count": len(hits_games),
        "right_count": len(pitches_games),
        "intersection_count": len(hits_games_in_pitches),
        "pct_match": round(100.0 * len(hits_games_in_pitches) / max(len(hits_games), 1), 2),
    })

    # HITS.batter_id ⊆ PLAYERS.player_id
    hits_batter_ids = set(hits["batter_id"].unique())
    hits_batters_in_players = hits_batter_ids.intersection(players_ids)
    data.append({
        "relationship": "HITS.batter_id ⊆ PLAYERS.player_id",
        "left_count": len(hits_batter_ids),
        "right_count": len(players_ids),
        "intersection_count": len(hits_batters_in_players),
        "pct_match": round(100.0 * len(hits_batters_in_players) / max(len(hits_batter_ids), 1), 2),
    })

    # HITS.pitcher_id ⊆ PLAYERS.player_id
    hits_pitcher_ids = set(hits["pitcher_id"].unique())
    hits_pitchers_in_players = hits_pitcher_ids.intersection(players_ids)
    data.append({
        "relationship": "HITS.pitcher_id ⊆ PLAYERS.player_id",
        "left_count": len(hits_pitcher_ids),
        "right_count": len(players_ids),
        "intersection_count": len(hits_pitchers_in_players),
        "pct_match": round(100.0 * len(hits_pitchers_in_players) / max(len(hits_pitcher_ids), 1), 2),
    })

    log.debug("  Relationships: %d validations computed", len(data))
    return pd.DataFrame(data)


def plot_physics_distributions(df, cols: list, output_dir: Path):
    """Histograms + KDE for each physics column."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import warnings

    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to actual pitches
    if "is_pitch" in df.columns:
        df = df[df["is_pitch"] == True]

    for col in cols:
        if col not in df.columns:
            continue

        values = df[col].dropna().to_numpy(dtype="float64")
        values = values[np.isfinite(values)]

        if len(values) < 10:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        clipped = np.clip(values, np.percentile(values, 1), np.percentile(values, 99))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import seaborn as sns
                sns.histplot(clipped, bins=50, stat="density", alpha=0.5, ax=ax, color="steelblue")
                sns.kdeplot(clipped, ax=ax, color="navy", linewidth=2)
            except Exception:
                ax.hist(clipped, bins=50, density=True, alpha=0.5, color="steelblue")

        ax.set_title(f"{col} Distribution", fontsize=11, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")

        output_file = output_dir / f"pitches_{col}.png"
        fig.savefig(output_file, dpi=100, bbox_inches="tight")
        plt.close("all")

    log.info("Distribution plots saved: %s", output_dir)


def plot_seasonal_distributions(df, cols: list, output_dir: Path):
    """Violin plots by season for each physics column."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import warnings

    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to actual pitches
    if "is_pitch" in df.columns:
        df = df[df["is_pitch"] == True]

    for col in cols:
        if col not in df.columns:
            continue

        seasons = sorted(df["season"].dropna().unique().astype(int))
        if len(seasons) < 2:
            continue

        season_data = []
        for s in seasons:
            s_values = df[df["season"] == s][col].dropna().to_numpy(dtype="float64")
            s_values = s_values[np.isfinite(s_values)]
            if len(s_values) > 0:
                if len(s_values) > 5000:
                    s_values = np.random.choice(s_values, 5000, replace=False)
                for v in s_values:
                    season_data.append({"season": str(s), "value": v})

        if not season_data:
            continue

        df_long = pd.DataFrame(season_data)

        fig, ax = plt.subplots(figsize=(12, 6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import seaborn as sns
                sns.violinplot(data=df_long, x="season", y="value", ax=ax, inner="box")
            except Exception:
                import seaborn as sns
                sns.boxplot(data=df_long, x="season", y="value", ax=ax)

        ax.set_title(f"{col} by Season", fontsize=11, fontweight="bold")
        ax.set_xlabel("Season")
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=45)

        output_file = output_dir / f"pitches_{col}_by_season.png"
        fig.savefig(output_file, dpi=100, bbox_inches="tight")
        plt.close("all")

    log.info("Seasonal plots saved: %s", output_dir)


def plot_hit_coordinates(hits, output_dir: Path):
    """Heatmap of hit coordinates (hit_x, hit_y)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import warnings

    output_dir.mkdir(parents=True, exist_ok=True)

    if "hit_x" not in hits.columns or "hit_y" not in hits.columns:
        log.debug("  Hit coordinates: hit_x or hit_y not found")
        return

    hit_x = hits["hit_x"].dropna().to_numpy(dtype="float64")
    hit_y = hits["hit_y"].dropna().to_numpy(dtype="float64")

    hit_x = hit_x[np.isfinite(hit_x)]
    hit_y = hit_y[np.isfinite(hit_y)]

    if len(hit_x) < 10 or len(hit_y) < 10:
        log.debug("  Hit coordinates: insufficient data")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.hexbin(hit_x, hit_y, gridsize=50, cmap="YlOrRd", mincnt=1)

    ax.set_title("Hit Coordinates Heatmap", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hit X (feet)")
    ax.set_ylabel("Hit Y (feet)")
    ax.set_aspect("equal")

    output_file = output_dir / "hits_coordinates_heatmap.png"
    fig.savefig(output_file, dpi=100, bbox_inches="tight")
    plt.close("all")

    log.info("Hit coordinates heatmap saved: %s", output_file)


def plot_linescore_distributions(linescore, output_dir: Path):
    """Runs per inning distribution (home vs. away)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import warnings

    output_dir.mkdir(parents=True, exist_ok=True)

    if "inning" not in linescore.columns or "home_runs" not in linescore.columns or "away_runs" not in linescore.columns:
        log.debug("  Linescore: required columns not found")
        return

    innings = sorted(linescore["inning"].dropna().unique().astype(int))
    if len(innings) < 1:
        return

    home_runs_by_inning = []
    away_runs_by_inning = []

    for inning in innings:
        inning_data = linescore[linescore["inning"] == inning]
        home_runs = inning_data["home_runs"].sum()
        away_runs = inning_data["away_runs"].sum()
        home_runs_by_inning.append(home_runs)
        away_runs_by_inning.append(away_runs)

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(innings))
    width = 0.35

    ax.bar(x_pos - width/2, home_runs_by_inning, width, label="Home", color="steelblue")
    ax.bar(x_pos + width/2, away_runs_by_inning, width, label="Away", color="coral")

    ax.set_title("Total Runs by Inning (Home vs. Away)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Inning")
    ax.set_ylabel("Total Runs")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(innings)
    ax.legend()

    output_file = output_dir / "linescore_runs_by_inning.png"
    fig.savefig(output_file, dpi=100, bbox_inches="tight")
    plt.close("all")

    log.info("Linescore runs plot saved: %s", output_file)


def build_raw_eda_html_report(output_dir: Path, seasons: list) -> Path:
    """Build HTML index for raw EDA (all 7 tables)."""
    csv_files = sorted((output_dir / "aggregate").glob("*.csv"))
    plot_files = sorted((output_dir / "plots").glob("*.png"))
    relationship_files = sorted((output_dir / "relationships").glob("*.csv"))
    player_files = sorted((output_dir / "players").glob("*.csv"))

    csv_rows = "\n".join(
        f'    <li><a href="aggregate/{f.name}">{f.stem}</a></li>'
        for f in csv_files
    )

    plot_rows = "\n".join(
        f'    <li><img src="plots/{f.name}" style="max-width: 100%; border: 1px solid #ccc; margin: 10px 0;"></li>'
        for f in plot_files[:50]
    )

    relationship_rows = "\n".join(
        f'    <li><a href="relationships/{f.name}">{f.stem}</a></li>'
        for f in relationship_files
    )

    player_rows = "\n".join(
        f'    <li><a href="players/{f.name}">{f.stem}</a></li>'
        for f in player_files
    )

    body = f"""
<h1>MLB Raw Data EDA Report (ALL 7 Tables)</h1>
<p>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>

<h2>Summary</h2>
<ul>
  <li>Seasons processed: {len(seasons)} ({min(seasons)}–{max(seasons)})</li>
  <li>Analysis type: Two-phase (per-season streaming + aggregate)</li>
  <li>Tables analyzed:</li>
  <ul>
    <li><strong>PITCHES</strong> (170 columns, grain: pitch/play event)</li>
    <li><strong>BOXSCORE_BATTING</strong> (33 columns, grain: batter per game)</li>
    <li><strong>BOXSCORE_PITCHING</strong> (28 columns, grain: pitcher per game)</li>
    <li><strong>RUNNERS</strong> (20 columns, grain: runner per play)</li>
    <li><strong>LINESCORE</strong> (11 columns, grain: inning per game)</li>
    <li><strong>HITS</strong> (10 columns, grain: hit per game/inning)</li>
    <li><strong>PLAYERS</strong> (25 columns, grain: unique player, global)</li>
  </ul>
</ul>

<h2>Aggregate Data</h2>
<ul>
{csv_rows}
</ul>

<h2>Relationship Validation</h2>
<p>Cross-table foreign key integrity checks:</p>
<ul>
{relationship_rows}
</ul>

<h2>Player Analyses</h2>
<ul>
{player_rows}
</ul>

<h2>Visualizations</h2>
<ul>
{plot_rows}
</ul>

<h2>Per-Season Outputs</h2>
<p>Individual CSV files for each season available in <code>per_season/</code> directory:</p>
<ul>
  <li><strong>PITCHES:</strong> <code>pitches_game_stats_*.csv</code>, <code>pitches_numeric_*.csv</code>, <code>pitches_categorical_*.csv</code>, <code>pitches_hit_data_*.csv</code>, <code>pitches_temporal_*.csv</code></li>
  <li><strong>BOXSCORE_BATTING:</strong> <code>batting_*.csv</code></li>
  <li><strong>BOXSCORE_PITCHING:</strong> <code>pitching_*.csv</code></li>
  <li><strong>RUNNERS:</strong> <code>runners_*.csv</code></li>
  <li><strong>LINESCORE:</strong> <code>linescore_*.csv</code></li>
  <li><strong>HITS:</strong> <code>hits_*.csv</code></li>
</ul>

<h2>Aggregate Analyses</h2>
<p>Combined results across all seasons in <code>aggregate/</code>:</p>
<ul>
  <li><code>pitches_seasonal_drift_ks.csv</code> — KS distances for physics columns across seasons</li>
  <li><code>pitches_correlations.csv</code> — Spearman ρ among physics features</li>
  <li><code>pitches_timestamp_quality.csv</code> — % valid timestamps</li>
  <li><code>batting_all.csv</code>, <code>pitching_all.csv</code>, <code>runners_all.csv</code>, <code>linescore_all.csv</code>, <code>hits_all.csv</code> — per-season stats combined</li>
</ul>
"""

    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Raw Data EDA (ALL 7 Tables)</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; color: #222; }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 6px; }}
    h2 {{ border-bottom: 1px solid #999; margin-top: 30px; }}
    a {{ color: #0066cc; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ background: #f0f0f0; padding: 2px 4px; }}
    ul {{ line-height: 1.8; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""

    html_path = output_dir / "index.html"
    html_path.write_text(html_content)
    return html_path
