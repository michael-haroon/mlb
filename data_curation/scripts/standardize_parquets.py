"""
Standardize all Parquet files in S3 to uniform schemas.

Problem: download_history.py wrote some batches with dictionary-encoded columns
(pyarrow optimization) and others with plain int64/object columns. This makes
them impossible to merge without schema unification.

Strategy (safe, no data loss):
  1. Read each Parquet file individually via ParquetFile (bypasses dataset API)
  2. Decode any dictionary-encoded columns to their value types
  3. Cast columns to the canonical schema from download_history.py
  4. Write to a temp key (same prefix, .standardized suffix)
  5. Verify the new file is readable and row-count matches
  6. Copy standardized file over original key
  7. Delete temp file

If interrupted at any point: originals are untouched until step 6 succeeds.
Re-running is safe — already-standardized files pass through unchanged.

Usage:
    python standardize_parquets.py [--dry-run] [--table pitches] [--season 2024]
"""

import argparse
import io
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pyarrow as pa
import pyarrow.parquet as pq

S3_BUCKET = "mlb-265753586044-us-east-1-an"
S3_PREFIX = "data"
S3_REGION = "us-east-1"
MAX_WORKERS = 8

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("standardize")

# Canonical schemas — every column must be this type after standardization.
# Sourced directly from download_history.py SCHEMA_TYPE_MAP etc.
ARROW_TYPE_MAP = {
    "int64": pa.int64(),
    "float64": pa.float64(),
    "object": pa.string(),
    "bool": pa.bool_(),
}

PITCHES_SCHEMA = {
    "game_pk": "int64", "season": "int64", "game_date": "object",
    "game_datetime_utc": "object", "game_number": "int64", "game_type_code": "object",
    "double_header": "object", "tiebreaker": "object", "series_description": "object",
    "series_game_number": "int64", "games_in_series": "int64",
    "game_status_detail": "object", "game_status_code": "object", "start_time_tbd": "bool",
    "venue_id": "int64", "venue_name": "object", "venue_city": "object",
    "venue_state": "object", "venue_latitude": "float64", "venue_longitude": "float64",
    "venue_timezone": "object", "venue_tz_offset": "float64",
    "venue_capacity": "int64", "venue_surface": "object", "venue_roof_type": "object",
    "home_team_id": "int64", "home_team_name": "object", "home_team_abbr": "object",
    "home_league_id": "int64", "home_league_name": "object",
    "home_division_id": "int64", "home_division_name": "object",
    "home_wins": "int64", "home_losses": "int64", "home_win_pct": "float64",
    "home_division_games_back": "float64", "home_wild_card_games_back": "float64",
    "home_games_played": "int64",
    "away_team_id": "int64", "away_team_name": "object", "away_team_abbr": "object",
    "away_league_id": "int64", "away_league_name": "object",
    "away_division_id": "int64", "away_division_name": "object",
    "away_wins": "int64", "away_losses": "int64", "away_win_pct": "float64",
    "away_division_games_back": "float64", "away_wild_card_games_back": "float64",
    "away_games_played": "int64",
    "start_time": "object", "day_night": "object",
    "weather_condition": "object", "weather_temp": "float64", "weather_wind": "object",
    "attendance": "float64", "game_duration_minutes": "float64",
    "umpire_hp": "object", "umpire_1b": "object", "umpire_2b": "object", "umpire_3b": "object",
    "winner_pitcher_id": "int64", "winner_pitcher_name": "object",
    "loser_pitcher_id": "int64", "loser_pitcher_name": "object",
    "save_pitcher_id": "int64", "save_pitcher_name": "object",
    "review_home_challenges_used": "int64", "review_home_challenges_remaining": "int64",
    "review_away_challenges_used": "int64", "review_away_challenges_remaining": "int64",
    "flag_no_hitter": "bool", "flag_perfect_game": "bool",
    "flag_away_team_no_hitter": "bool", "flag_home_team_no_hitter": "bool",
    "probable_pitcher_home_id": "int64", "probable_pitcher_away_id": "int64",
    "leader_hit_distance": "float64", "leader_hit_distance_player_id": "int64",
    "leader_hit_speed": "float64", "leader_hit_speed_player_id": "int64",
    "leader_pitch_speed": "float64", "leader_pitch_speed_player_id": "int64",
    "game_alerts_json": "object",
    "play_index": "int64", "at_bat_index": "int64", "inning": "int64",
    "half_inning": "object", "is_top_inning": "bool", "captivating_index": "int64",
    "at_bat_start_time": "object", "at_bat_end_time": "object",
    "at_bat_has_review": "bool", "at_bat_is_complete": "bool",
    "batter_id": "int64", "batter_name": "object", "bat_side_code": "object",
    "pitcher_id": "int64", "pitcher_name": "object", "pitch_hand_code": "object",
    "split_batter": "object", "split_pitcher": "object", "men_on_base": "object",
    "pre_on_first_id": "int64", "pre_on_second_id": "int64", "pre_on_third_id": "int64",
    "post_on_first_id": "int64", "post_on_second_id": "int64", "post_on_third_id": "int64",
    "at_bat_event": "object", "event_type": "object",
    "is_scoring_play": "bool", "rbi_count": "int64",
    "score_home": "int64", "score_away": "int64", "play_description": "object",
    "cum_balls": "int64", "cum_strikes": "int64", "cum_outs": "int64",
    "pitch_sequence_index": "int64", "play_id": "object", "pitch_event_type": "object",
    "is_pitch": "bool", "pitch_number": "int64",
    "pitch_start_time": "object", "pitch_end_time": "object",
    "pitch_count_balls": "int64", "pitch_count_strikes": "int64", "pitch_count_outs": "int64",
    "pitch_type": "object", "pitch_call": "object", "pitch_event_flags_json": "object",
    "is_in_play": "bool", "is_strike": "bool", "is_ball": "bool", "has_review": "bool",
    "release_speed": "float64", "end_speed": "float64",
    "strike_zone_top": "float64", "strike_zone_bottom": "float64",
    "type_confidence": "float64", "plate_time": "float64", "extension": "float64",
    "coord_px": "float64", "coord_pz": "float64",
    "coord_x0": "float64", "coord_y0": "float64", "coord_z0": "float64",
    "coord_vx0": "float64", "coord_vy0": "float64", "coord_vz0": "float64",
    "coord_ax": "float64", "coord_ay": "float64", "coord_az": "float64",
    "pfx_x": "float64", "pfx_z": "float64",
    "break_angle": "float64", "break_length": "float64", "break_y": "float64",
    "spin_rate": "float64", "spin_direction": "float64", "zone_location": "int64",
    "hit_launch_speed": "float64", "hit_launch_angle": "float64",
    "hit_total_distance": "float64", "hit_trajectory": "object", "hit_hardness": "object",
    "hit_coord_x": "float64", "hit_coord_y": "float64",
}

LINESCORE_SCHEMA = {
    "game_pk": "int64", "season": "int64", "inning": "int64",
    "home_runs": "int64", "away_runs": "int64",
    "home_hits": "int64", "away_hits": "int64",
    "home_errors": "int64", "away_errors": "int64",
    "home_left_on_base": "int64", "away_left_on_base": "int64",
}

RUNNER_SCHEMA = {
    "game_pk": "int64", "season": "int64", "play_index": "int64",
    "play_event_index": "int64", "runner_id": "int64", "runner_name": "object",
    "responsible_pitcher_id": "int64", "movement_start": "object", "movement_end": "object",
    "is_out": "bool", "out_base": "object", "out_number": "int64",
    "is_scoring_event": "bool", "rbi": "bool", "earned": "bool", "team_unearned": "bool",
    "event": "object", "event_type": "object", "movement_reason": "object",
    "credits_json": "object",
}

BOXSCORE_BATTING_SCHEMA = {
    "game_pk": "int64", "season": "int64", "player_id": "int64", "player_name": "object",
    "side": "object", "batting_order": "int64", "all_positions_json": "object",
    "is_substitute": "bool",
    "game_ab": "int64", "game_runs": "int64", "game_hits": "int64",
    "game_doubles": "int64", "game_triples": "int64", "game_hr": "int64",
    "game_rbi": "int64", "game_bb": "int64", "game_ibb": "int64",
    "game_so": "int64", "game_sb": "int64", "game_cs": "int64",
    "game_hbp": "int64", "game_sac": "int64", "game_sf": "int64",
    "game_gidp": "int64", "game_lob": "int64",
    "season_avg": "float64", "season_obp": "float64", "season_slg": "float64",
    "season_ops": "float64", "season_hr": "int64", "season_rbi": "int64",
    "season_sb": "int64", "season_games_played": "int64",
}

BOXSCORE_PITCHING_SCHEMA = {
    "game_pk": "int64", "season": "int64", "player_id": "int64", "player_name": "object",
    "side": "object", "is_starter": "bool",
    "game_innings_pitched": "float64", "game_hits": "int64", "game_runs": "int64",
    "game_earned_runs": "int64", "game_bb": "int64", "game_so": "int64",
    "game_hr": "int64", "game_hbp": "int64", "game_pitches_thrown": "int64",
    "game_strikes_thrown": "int64", "game_balls_thrown": "int64",
    "game_strikes_looking": "int64", "game_strikes_swinging": "int64",
    "season_era": "float64", "season_whip": "float64",
    "season_wins": "int64", "season_losses": "int64", "season_saves": "int64",
    "season_innings_pitched": "float64", "season_so": "int64",
    "season_bb": "int64", "season_games_played": "int64",
}

HITS_SCHEMA = {
    "game_pk": "int64", "season": "int64", "inning": "int64", "side": "object",
    "batter_id": "int64", "pitcher_id": "int64",
    "hit_x": "float64", "hit_y": "float64", "hit_type": "object", "team_id": "int64",
}

PLAYER_SCHEMA = {
    "player_id": "int64", "full_name": "object", "use_name": "object",
    "boxscore_name": "object", "first_name": "object", "last_name": "object",
    "primary_number": "object", "birth_date": "object", "birth_city": "object",
    "birth_state": "object", "birth_country": "object",
    "height": "object", "weight": "float64", "current_age": "int64",
    "strike_zone_top": "float64", "strike_zone_bottom": "float64",
    "position_code": "object", "position_name": "object",
    "position_type": "object", "position_abbreviation": "object",
    "bat_side": "object", "pitch_hand": "object",
    "mlb_debut_date": "object", "draft_year": "int64", "is_active": "bool",
}

TABLE_SCHEMAS = {
    "pitches": PITCHES_SCHEMA,
    "linescore": LINESCORE_SCHEMA,
    "runners": RUNNER_SCHEMA,
    "boxscore_batting": BOXSCORE_BATTING_SCHEMA,
    "boxscore_pitching": BOXSCORE_PITCHING_SCHEMA,
    "hits": HITS_SCHEMA,
    "players": PLAYER_SCHEMA,
}

TABLE_PATTERNS = {
    "pitches": "season=*/pitches_batch_*.parquet",
    "linescore": "season=*/linescore_batch_*.parquet",
    "runners": "season=*/runners_batch_*.parquet",
    "boxscore_batting": "season=*/boxscore_batting_batch_*.parquet",
    "boxscore_pitching": "season=*/boxscore_pitching_batch_*.parquet",
    "hits": "season=*/hits_batch_*.parquet",
    "players": "players/players_batch_*.parquet",
}


def build_arrow_schema(schema_map: dict[str, str]) -> pa.Schema:
    fields = []
    for col, dtype in schema_map.items():
        arrow_type = ARROW_TYPE_MAP[dtype]
        if dtype in ("int64",):
            arrow_type = pa.int64()
        elif dtype == "float64":
            arrow_type = pa.float64()
        elif dtype == "bool":
            arrow_type = pa.bool_()
        else:
            arrow_type = pa.string()
        fields.append(pa.field(col, arrow_type))
    return pa.schema(fields)


def standardize_file(s3_client, key: str, target_schema: pa.Schema, dry_run: bool) -> dict:
    """Standardize a single Parquet file. Returns stats dict."""
    result = {"key": key, "status": "unchanged", "rows": 0}

    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        raw_bytes = obj["Body"].read()
    except Exception as e:
        result["status"] = f"read_error: {e}"
        return result

    try:
        buf = io.BytesIO(raw_bytes)
        pf = pq.ParquetFile(buf)
        table = pf.read()
        original_rows = table.num_rows
        result["rows"] = original_rows
    except Exception as e:
        result["status"] = f"parse_error: {e}"
        return result

    if original_rows == 0:
        result["status"] = "empty"
        return result

    # Check if already standardized
    needs_fix = False
    for field in table.schema:
        if pa.types.is_dictionary(field.type):
            needs_fix = True
            break
        target_field = target_schema.field(field.name) if field.name in [f.name for f in target_schema] else None
        if target_field and field.type != target_field.type:
            needs_fix = True
            break

    if not needs_fix:
        result["status"] = "already_ok"
        return result

    if dry_run:
        result["status"] = "would_fix"
        return result

    # Decode dictionary columns
    new_columns = []
    for i, field in enumerate(table.schema):
        col = table.column(i)
        if pa.types.is_dictionary(field.type):
            col = col.dictionary_decode()
        new_columns.append(col)
    table = pa.table(
        {table.schema.field(i).name: new_columns[i] for i in range(len(new_columns))}
    )

    # Cast to target schema (only columns present in the file)
    cast_columns = {}
    for i in range(table.num_columns):
        col_name = table.schema.field(i).name
        col = table.column(i)
        if col_name in [f.name for f in target_schema]:
            target_type = target_schema.field(col_name).type
            if col.type != target_type:
                try:
                    col = col.cast(target_type, safe=False)
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                    # For string columns that can't cast, leave as-is
                    pass
        cast_columns[col_name] = col
    table = pa.table(cast_columns)

    if table.num_rows != original_rows:
        result["status"] = "row_count_mismatch_aborted"
        return result

    # Write to temp key
    tmp_key = key + ".standardized"
    out_buf = io.BytesIO()
    pq.write_table(table, out_buf, compression="snappy")
    out_buf.seek(0)
    standardized_bytes = out_buf.getvalue()

    s3_client.put_object(Bucket=S3_BUCKET, Key=tmp_key, Body=standardized_bytes)

    # Verify temp file
    try:
        verify_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=tmp_key)
        verify_buf = io.BytesIO(verify_obj["Body"].read())
        verify_table = pq.read_table(verify_buf)
        if verify_table.num_rows != original_rows:
            result["status"] = "verification_failed"
            s3_client.delete_object(Bucket=S3_BUCKET, Key=tmp_key)
            return result
    except Exception as e:
        result["status"] = f"verify_error: {e}"
        s3_client.delete_object(Bucket=S3_BUCKET, Key=tmp_key)
        return result

    # Overwrite original with standardized version
    s3_client.copy_object(
        Bucket=S3_BUCKET,
        Key=key,
        CopySource={"Bucket": S3_BUCKET, "Key": tmp_key},
    )
    s3_client.delete_object(Bucket=S3_BUCKET, Key=tmp_key)

    result["status"] = "fixed"
    return result


def list_keys(s3_client, prefix: str, pattern_suffix: str = ".parquet") -> list[str]:
    keys = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith(pattern_suffix) and not k.endswith(".standardized"):
                keys.append(k)
    return sorted(keys)


def main():
    parser = argparse.ArgumentParser(description="Standardize MLB Parquet schemas in S3")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing")
    parser.add_argument("--table", type=str, default=None, help="Only process one table (e.g. pitches)")
    parser.add_argument("--season", type=int, default=None, help="Only process one season")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=S3_REGION)

    tables_to_process = [args.table] if args.table else list(TABLE_SCHEMAS.keys())

    total_fixed = 0
    total_ok = 0
    total_errors = 0

    for table_name in tables_to_process:
        schema_map = TABLE_SCHEMAS[table_name]
        target_schema = build_arrow_schema(schema_map)

        if args.season and table_name != "players":
            prefix = f"{S3_PREFIX}/season={args.season}/"
            # Filter to only this table's files
            all_keys = list_keys(s3, prefix)
            keys = [k for k in all_keys if f"{table_name}_batch_" in k]
        elif table_name == "players":
            keys = list_keys(s3, f"{S3_PREFIX}/players/")
        else:
            keys = list_keys(s3, f"{S3_PREFIX}/")
            keys = [k for k in keys if f"{table_name}_batch_" in k]

        logger.info(f"[{table_name}] Found {len(keys)} files to check")

        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(standardize_file, s3, key, target_schema, args.dry_run): key
                for key in keys
            }
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                if res["status"] == "fixed":
                    total_fixed += 1
                    logger.info(f"  FIXED {res['key']} ({res['rows']} rows)")
                elif res["status"] == "would_fix":
                    total_fixed += 1
                    logger.info(f"  WOULD FIX {res['key']} ({res['rows']} rows)")
                elif res["status"] == "already_ok":
                    total_ok += 1
                elif "error" in res["status"]:
                    total_errors += 1
                    logger.error(f"  ERROR {res['key']}: {res['status']}")

        logger.info(f"[{table_name}] Done: {sum(1 for r in results if r['status'] in ('fixed', 'would_fix'))} fixed, "
                    f"{sum(1 for r in results if r['status'] == 'already_ok')} ok, "
                    f"{sum(1 for r in results if 'error' in r['status'])} errors")

    logger.info(f"\nTOTAL: {total_fixed} fixed | {total_ok} already ok | {total_errors} errors")
    if args.dry_run:
        logger.info("(DRY RUN — no files were modified)")


if __name__ == "__main__":
    main()
