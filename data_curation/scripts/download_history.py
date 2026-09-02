import argparse
import io
import json
import logging
import os
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Set

import boto3
from botocore.exceptions import ClientError
import pandas as pd
import requests
from tqdm import tqdm

# --- INGESTION TARGET CONSTANTS ---
START_YEAR = 1950
END_YEAR = datetime.now().year
MAX_WORKERS = 80
RATE_LIMIT_DELAY = 0.1         # 10 req/sec target; increase if API returns 429s
SUBMIT_BATCH_SIZE = 150         # futures in flight; ~2× workers to keep queue full
CHUNK_FLUSH_THRESHOLD = 25000
LINESCORE_FLUSH_THRESHOLD = 2500
RUNNER_FLUSH_THRESHOLD = 5000
PLAYER_FLUSH_THRESHOLD = 5000
BOXSCORE_FLUSH_THRESHOLD = 5000
HITS_FLUSH_THRESHOLD = 5000

# Storage config — S3 is default; --local overrides to local disk
S3_BUCKET = "mlb-265753586044-us-east-1-an"
S3_PREFIX = "data"              # keys will be  data/season=2024/pitches_batch_*.parquet
S3_REGION = "us-east-1"
USE_S3 = True                   # overridden to False by --local flag
_s3_client = None               # lazy-initialised on first write

DATA_DIR = "data"               # used only in --local mode
LOG_DIR = os.path.join(DATA_DIR, "logs")    # always local — logs stay on the instance

os.makedirs(LOG_DIR, exist_ok=True)

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("GUMBO_ENGINE")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "gumbo_ingest.log"))
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s")
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Keeps terminal clean for tqdm
console_formatter = logging.Formatter("[GUMBO ENGINE] %(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ---------------------------------------------------------------------------
# SCHEMAS
# ---------------------------------------------------------------------------

SCHEMA_TYPE_MAP = {
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

RUNNER_SCHEMA_TYPE_MAP = {
    "game_pk": "int64", "season": "int64", "play_index": "int64",
    "play_event_index": "int64", "runner_id": "int64", "runner_name": "object",
    "responsible_pitcher_id": "int64", "movement_start": "object", "movement_end": "object",
    "is_out": "bool", "out_base": "object", "out_number": "int64",
    "is_scoring_event": "bool", "rbi": "bool", "earned": "bool", "team_unearned": "bool",
    "event": "object", "event_type": "object", "movement_reason": "object",
    "credits_json": "object",
}

BOXSCORE_BATTING_SCHEMA_TYPE_MAP = {
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

BOXSCORE_PITCHING_SCHEMA_TYPE_MAP = {
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

HITS_SCHEMA_TYPE_MAP = {
    "game_pk": "int64", "season": "int64", "inning": "int64", "side": "object",
    "batter_id": "int64", "pitcher_id": "int64",
    "hit_x": "float64", "hit_y": "float64", "hit_type": "object", "team_id": "int64",
}

LINESCORE_SCHEMA_TYPE_MAP = {
    "game_pk": "int64", "season": "int64", "inning": "int64",
    "home_runs": "int64", "away_runs": "int64",
    "home_hits": "int64", "away_hits": "int64",
    "home_errors": "int64", "away_errors": "int64",
    "home_left_on_base": "int64", "away_left_on_base": "int64",
}

PLAYER_SCHEMA_TYPE_MAP = {
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


# ---------------------------------------------------------------------------
# STORAGE LAYER — unified S3 / local abstraction
# ---------------------------------------------------------------------------
def _get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=S3_REGION)
    return _s3_client


def _s3_key(rel_path: str) -> str:
    """Converts a relative data path to a full S3 key."""
    return f"{S3_PREFIX}/{rel_path}"


def _read_json_store(rel_path: str):
    """Read a JSON file from S3 or local disk. Returns None if not found."""
    if USE_S3:
        try:
            obj = _get_s3().get_object(Bucket=S3_BUCKET, Key=_s3_key(rel_path))
            return json.loads(obj["Body"].read().decode("utf-8"))
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise
    else:
        local = os.path.join(DATA_DIR, rel_path)
        if not os.path.exists(local):
            return None
        with open(local) as f:
            return json.load(f)


def _write_json_store(rel_path: str, data: Any):
    """Write a JSON file atomically to S3 or local disk."""
    if USE_S3:
        _get_s3().put_object(
            Bucket=S3_BUCKET, Key=_s3_key(rel_path),
            Body=json.dumps(data, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        logger.debug(f"[store] Written s3://{S3_BUCKET}/{_s3_key(rel_path)}")
    else:
        local = os.path.join(DATA_DIR, rel_path)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        tmp = local + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, local)
        logger.debug(f"[store] Written {local}")


def _apply_schema(df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
    for col, dtype in schema.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
        else:
            df[col] = "None" if dtype == "object" else (False if dtype == "bool" else pd.NA)
    return df


def _save(records: List[Dict[str, Any]], schema: Dict[str, str], rel_path: str):
    """Write a Parquet batch to S3 or local disk."""
    df = _apply_schema(pd.DataFrame(records), schema)
    if USE_S3:
        key = _s3_key(rel_path)
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
        buf.seek(0)
        _get_s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
        logger.debug(f"[save] {len(records)} rows -> s3://{S3_BUCKET}/{key}")
    else:
        full = os.path.join(DATA_DIR, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        df.to_parquet(full, engine="pyarrow", compression="snappy", index=False)
        logger.debug(f"[save] {len(records)} rows -> {full}")


# ---------------------------------------------------------------------------
# TYPED SAVE HELPERS
# ---------------------------------------------------------------------------
def save_pitches(records: List[Dict[str, Any]], season: int):
    logger.debug(f"[flush:pitches] Writing {len(records)} rows for season={season}...")
    _save(records, SCHEMA_TYPE_MAP, f"season={season}/pitches_batch_{int(time.time() * 1000)}.parquet")

def save_linescores(records: List[Dict[str, Any]], season: int):
    if not records: return
    logger.debug(f"[flush:linescores] Writing {len(records)} rows for season={season}...")
    _save(records, LINESCORE_SCHEMA_TYPE_MAP, f"season={season}/linescore_batch_{int(time.time() * 1000)}.parquet")

def save_runners(records: List[Dict[str, Any]], season: int):
    if not records: return
    logger.debug(f"[flush:runners] Writing {len(records)} rows for season={season}...")
    _save(records, RUNNER_SCHEMA_TYPE_MAP, f"season={season}/runners_batch_{int(time.time() * 1000)}.parquet")

def save_players(records: List[Dict[str, Any]]):
    if not records: return
    logger.debug(f"[flush:players] Writing {len(records)} new player bio records...")
    _save(records, PLAYER_SCHEMA_TYPE_MAP, f"players/players_batch_{int(time.time() * 1000)}.parquet")

def save_boxscore_batting(records: List[Dict[str, Any]], season: int):
    if not records: return
    logger.debug(f"[flush:boxscore_batting] Writing {len(records)} rows for season={season}...")
    _save(records, BOXSCORE_BATTING_SCHEMA_TYPE_MAP, f"season={season}/boxscore_batting_batch_{int(time.time() * 1000)}.parquet")

def save_boxscore_pitching(records: List[Dict[str, Any]], season: int):
    if not records: return
    logger.debug(f"[flush:boxscore_pitching] Writing {len(records)} rows for season={season}...")
    _save(records, BOXSCORE_PITCHING_SCHEMA_TYPE_MAP, f"season={season}/boxscore_pitching_batch_{int(time.time() * 1000)}.parquet")

def save_hits(records: List[Dict[str, Any]], season: int):
    if not records: return
    logger.debug(f"[flush:hits] Writing {len(records)} spray chart rows for season={season}...")
    _save(records, HITS_SCHEMA_TYPE_MAP, f"season={season}/hits_batch_{int(time.time() * 1000)}.parquet")

def _flush_by_season(buffer: List[Dict[str, Any]], save_fn) -> None:
    if not buffer: return
    df_temp = pd.DataFrame(buffer)
    for season_id, group in df_temp.groupby("season"):
        save_fn(group.to_dict(orient="records"), int(season_id))


# ---------------------------------------------------------------------------
# UTIL
# ---------------------------------------------------------------------------
def _safe_int(value, default: int = -1) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default

def _safe_float(value) -> float:
    try:
        return float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")

def _str(value, default: str = "None") -> str:
    return str(value) if value is not None else default

def _gn(node, *keys):
    """Traverse a nested dict, treating explicit null values as absent."""
    for key in keys:
        if not isinstance(node, dict):
            return {}
        node = node.get(key)
        if node is None:
            return {}
    return node if isinstance(node, dict) else {}

def _parse_boxscore_info(info_list: list) -> Dict[str, str]:
    return {e.get("label", ""): e.get("value", "") for e in info_list if e.get("label")}

def _parse_team_record(team_node: Dict[str, Any]) -> Dict[str, Any]:
    record = team_node.get("record") or {}
    lr = record.get("leagueRecord") or {}
    return {
        "id": _safe_int(team_node.get("id"), -1),
        "name": _str(team_node.get("name"), "Unknown"),
        "abbr": _str(team_node.get("abbreviation"), "Unknown"),
        "league_id": _safe_int(_gn(team_node, "league").get("id"), -1),
        "league_name": _str(_gn(team_node, "league").get("name"), "Unknown"),
        "division_id": _safe_int(_gn(team_node, "division").get("id"), -1),
        "division_name": _str(_gn(team_node, "division").get("name"), "Unknown"),
        "wins": _safe_int(lr.get("wins", record.get("wins")), -1),
        "losses": _safe_int(lr.get("losses", record.get("losses")), -1),
        "win_pct": _safe_float(lr.get("pct", record.get("winningPercentage"))),
        "division_games_back": _safe_float(record.get("divisionGamesBack")),
        "wild_card_games_back": _safe_float(record.get("wildCardGamesBack")),
        "games_played": _safe_int(record.get("gamesPlayed"), -1),
    }


# ---------------------------------------------------------------------------
# CHECKPOINT MANAGER
# Persists to S3 or local depending on USE_S3.
# Writes are atomic: S3 put_object is atomic by nature; local uses .tmp + os.replace.
# ---------------------------------------------------------------------------
class CheckpointManager:
    CHECKPOINT_REL = "checkpoint.json"
    RETRY_REL = "retry_queue.json"

    def __init__(self):
        self._lock = threading.Lock()
        self.completed: Set[int] = set()
        self.retry_queue: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        data = _read_json_store(self.CHECKPOINT_REL)
        if data:
            self.completed = set(data.get("completed", []))
            logger.info(f"[checkpoint] Loaded {len(self.completed)} completed games.")
        else:
            logger.info("[checkpoint] No existing checkpoint found — starting fresh.")

        retry_data = _read_json_store(self.RETRY_REL)
        if retry_data:
            self.retry_queue = retry_data
            logger.info(f"[checkpoint] Loaded {len(self.retry_queue)} games in retry queue.")

    def is_completed(self, game_pk: int) -> bool:
        return game_pk in self.completed

    def mark_completed(self, game_pk: int):
        with self._lock:
            self.completed.add(game_pk)
            self._flush_checkpoint()
            logger.debug(f"[checkpoint] Marked gamePk={game_pk} as completed. Total={len(self.completed)}")

    def mark_failed(self, game_pk: int, season: int, reason: str, error: str):
        with self._lock:
            now = datetime.utcnow().isoformat()
            existing = {e["game_pk"]: e for e in self.retry_queue}
            if game_pk in existing:
                e = existing[game_pk]
                e["attempts"] += 1; e["last_error"] = error
                e["reason"] = reason; e["last_failed"] = now
            else:
                self.retry_queue.append({
                    "game_pk": game_pk, "season": season, "reason": reason,
                    "attempts": 1, "last_error": error,
                    "first_failed": now, "last_failed": now,
                })
            self._flush_retry()
            logger.debug(f"[checkpoint] gamePk={game_pk} queued for retry. reason={reason}")

    def clear_retry_entry(self, game_pk: int):
        with self._lock:
            self.retry_queue = [e for e in self.retry_queue if e["game_pk"] != game_pk]
            self.completed.add(game_pk)
            self._flush_checkpoint()
            self._flush_retry()
            logger.debug(f"[checkpoint] gamePk={game_pk} cleared from retry queue and marked completed.")

    def get_retry_games(self) -> List[Dict[str, Any]]:
        return [{"game_pk": e["game_pk"], "season": e["season"]} for e in self.retry_queue]

    def _flush_checkpoint(self):
        _write_json_store(self.CHECKPOINT_REL, {
            "completed": sorted(self.completed),
            "last_updated": datetime.utcnow().isoformat(),
        })

    def _flush_retry(self):
        _write_json_store(self.RETRY_REL, self.retry_queue)


# ---------------------------------------------------------------------------
# INGESTION ENGINE
# ---------------------------------------------------------------------------
class GumboIngestionEngine:
    def __init__(self):
        self.session = requests.Session()
        
        # Expand the connection pool to safely exceed MAX_WORKERS
        adapter = requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Accept": "application/json",
        })
        self.last_request_time = 0.0
        self._rate_lock = threading.Lock()
        self._seen_player_ids: Set[int] = set()
        self._player_lock = threading.Lock()

    def _enforce_rate_limit(self):
        # Stamp inside the lock (so workers get distinct slots), sleep outside
        # (so the lock isn't held while waiting, letting other workers proceed).
        with self._rate_lock:
            now = time.time()
            wait = max(0.0, self.last_request_time + RATE_LIMIT_DELAY - now)
            self.last_request_time = now + wait  # reserve next slot
        if wait:
            time.sleep(wait)

    def discover_schedule(self) -> List[Dict[str, Any]]:
        discovered_games = []
        logger.info(f"Mapping schedule profiles across target years: {START_YEAR} -> {END_YEAR}")

        for year in range(START_YEAR, END_YEAR + 1):
            url = "https://statsapi.mlb.com/api/v1/schedule"
            params = {"sportId": 1, "startDate": f"{year}-01-01", "endDate": f"{year}-12-31"}
            logger.debug(f"[schedule] GET {url} | params={params}")
            try:
                self._enforce_rate_limit()
                t0 = time.time()
                res = self.session.get(url, params=params, timeout=15)
                elapsed_ms = (time.time() - t0) * 1000
                logger.debug(f"[schedule] Response {year}: HTTP {res.status_code} | {len(res.content)} bytes | {elapsed_ms:.0f}ms")
                res.raise_for_status()
                schedule_data = res.json()

                date_nodes = schedule_data.get("dates", [])
                logger.debug(f"[schedule] {year}: {len(date_nodes)} date nodes in payload")

                season_count = 0
                for date_node in date_nodes:
                    for game in date_node.get("games", []):
                        state = game.get("status", {}).get("abstractGameState")
                        gk = game.get("gamePk")
                        if state == "Final":
                            discovered_games.append({"game_pk": gk, "season": year})
                            season_count += 1
                        else:
                            logger.debug(f"[schedule] Skipping gamePk={gk} state={state}")
                logger.info(f"  -> Season {year}: Found {season_count} final matches.")
            except Exception as e:
                logger.error(f"Failed to pull schedule for {year}: {e}", exc_info=True)

        return discovered_games

    def extract_and_flatten_game(
        self,
        game_metadata: Dict[str, Any],
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Flatten one GUMBO feed into the storage tables (+ the game context row).

        `payload` lets a caller that already holds the feed skip the fetch —
        live_daemon polls the same URL every 10s, so re-requesting it would both
        double the API load and open a window where the flattened rows describe a
        different instant than the state the daemon just acted on. Reusing this
        function (rather than reimplementing extraction in a live path) is what
        makes live rows schema-identical to the historical artifact by
        construction, the same parity argument as weather_asof's shared assembler.

        A mid-game payload is a legitimate input: allPlays is simply shorter.
        Callers must not assume the game is Final.
        """
        game_pk = game_metadata["game_pk"]
        season = game_metadata["season"]
        url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        empty = {"pitches": [], "linescores": [], "runners": [], "players": [],
                 "boxscore_batting": [], "boxscore_pitching": [], "hits": [],
                 "game_context": {}}

        max_retries = 5
        backoff_factor = 2.0
        gumbo_root = payload

        for attempt in range(max_retries):
            if gumbo_root is not None:
                break
            try:
                logger.debug(f"[game={game_pk}] GET {url} | attempt={attempt + 1}/{max_retries}")
                self._enforce_rate_limit()
                t0 = time.time()
                response = self.session.get(url, timeout=20)
                elapsed_ms = (time.time() - t0) * 1000
                logger.debug(f"[game={game_pk}] HTTP {response.status_code} | {len(response.content)} bytes | {elapsed_ms:.0f}ms")

                if response.status_code == 429:
                    sleep_time = backoff_factor ** attempt
                    logger.warning(f"[game={game_pk}] Rate limited (429). Backing off {sleep_time}s...")
                    time.sleep(sleep_time)
                    continue

                if response.status_code != 200:
                    raise requests.HTTPError(f"HTTP {response.status_code}")

                gumbo_root = response.json()
                logger.debug(f"[game={game_pk}] Payload parsed. Top-level keys: {list(gumbo_root.keys())}")
                break

            except (requests.exceptions.RequestException, requests.HTTPError) as e:
                if attempt == max_retries - 1:
                    logger.error(f"[game={game_pk}] Fatal retry exhaustion after {max_retries} attempts: {e}", exc_info=True)
                    raise e
                sleep_time = backoff_factor ** attempt
                logger.warning(f"[game={game_pk}] Network error: {e}. Retry in {sleep_time}s...")
                time.sleep(sleep_time)

        if not gumbo_root:
            logger.error(f"[game={game_pk}] Empty payload after retry loop — skipping game.")
            return empty

        game_data_node = gumbo_root.get("gameData") or {}
        live_data_node = gumbo_root.get("liveData") or {}

        if not game_data_node:
            logger.warning(f"[game={game_pk}] gameData node is empty or missing.")
        if not live_data_node:
            logger.warning(f"[game={game_pk}] liveData node is empty or missing.")

        # --- GAME CONTEXT ---
        game_node = game_data_node.get("game") or {}
        game_type_code = _str(game_node.get("type"), "Unknown")
        game_number = _safe_int(game_node.get("gameNumber"), 1)
        double_header = _str(game_node.get("doubleHeader"), "N")
        tiebreaker = _str(game_node.get("tiebreaker"), "N")
        series_description = _str(game_node.get("seriesDescription"), "Unknown")
        series_game_number = _safe_int(game_node.get("seriesGameNumber"), -1)
        games_in_series = _safe_int(game_node.get("gamesInSeries"), -1)

        # --- STATUS ---
        status_node = game_data_node.get("status") or {}
        game_status_detail = _str(status_node.get("detailedState"), "Unknown")
        game_status_code = _str(status_node.get("statusCode"), "Unknown")
        start_time_tbd = bool(status_node.get("startTimeTBD", False))

        # --- DATETIME ---
        datetime_node = game_data_node.get("datetime") or {}
        game_date_str = datetime_node.get("originalDate", "1970-01-01")
        game_datetime_utc = _str(datetime_node.get("dateTime"), "Unknown")
        start_time = _str(datetime_node.get("time"), "Unknown")
        day_night = _str(datetime_node.get("dayNight"), "Unknown")

        # --- VENUE ---
        venue_node = game_data_node.get("venue") or {}
        venue_id = _safe_int(venue_node.get("id"), -1)
        venue_name = _str(venue_node.get("name"), "Unknown")
        venue_location = venue_node.get("location") or {}
        venue_city = _str(venue_location.get("city"), "Unknown")
        venue_state = _str(venue_location.get("stateAbbrev", venue_location.get("state")), "Unknown")
        venue_coords = venue_location.get("defaultCoordinates") or {}
        venue_latitude = _safe_float(venue_coords.get("latitude"))
        venue_longitude = _safe_float(venue_coords.get("longitude"))
        venue_tz_node = venue_location.get("timeZone") or {}
        venue_timezone = _str(venue_tz_node.get("id"), "Unknown")
        venue_tz_offset = _safe_float(venue_tz_node.get("offset"))
        venue_field_info = venue_node.get("fieldInfo") or {}
        venue_capacity = _safe_int(venue_field_info.get("capacity"), -1)
        venue_surface = _str(venue_field_info.get("turfType"), "Unknown")
        venue_roof_type = _str(venue_field_info.get("roofType"), "Unknown")

        # --- WEATHER ---
        weather_node = game_data_node.get("weather") or {}
        weather_condition = _str(weather_node.get("condition"), "Unknown")
        weather_temp = _safe_float(weather_node.get("temp"))
        weather_wind = _str(weather_node.get("wind"), "Unknown")

        # --- TEAMS ---
        teams_node = game_data_node.get("teams") or {}
        home_rec = _parse_team_record(teams_node.get("home") or {})
        away_rec = _parse_team_record(teams_node.get("away") or {})

        # --- GAME INFO — canonical source is boxscore.info[] per GUMBO spec ---
        boxscore_node = live_data_node.get("boxscore") or {}
        boxscore_info = _parse_boxscore_info(boxscore_node.get("info") or [])
        logger.debug(f"[game={game_pk}] boxscore.info keys: {list(boxscore_info.keys())}")

        game_info_node = game_data_node.get("gameInfo") or {}
        raw_att = boxscore_info.get("Att") or boxscore_info.get("Attendance")
        if raw_att:
            try:
                attendance = float(str(raw_att).replace(",", ""))
            except ValueError:
                attendance = _safe_float(game_info_node.get("attendance"))
        else:
            attendance = _safe_float(game_info_node.get("attendance"))

        raw_duration = boxscore_info.get("T") or boxscore_info.get("Time")
        if raw_duration and ":" in str(raw_duration):
            try:
                parts = str(raw_duration).split(":")
                game_duration_minutes = float(int(parts[0]) * 60 + int(parts[1]))
            except (ValueError, IndexError):
                game_duration_minutes = _safe_float(game_info_node.get("gameDurationMinutes"))
        else:
            game_duration_minutes = _safe_float(game_info_node.get("gameDurationMinutes"))

        # --- UMPIRES ---
        officials = boxscore_node.get("officials", [])
        ump_hp, ump_1b, ump_2b, ump_3b = "None", "None", "None", "None"
        for o in officials:
            o_type = o.get("officialType", "")
            o_name = _gn(o, "official").get("fullName", "None")
            if o_type == "Home Plate":    ump_hp = o_name
            elif o_type == "First Base":  ump_1b = o_name
            elif o_type == "Second Base": ump_2b = o_name
            elif o_type == "Third Base":  ump_3b = o_name

        # --- DECISIONS ---
        decisions_node = live_data_node.get("decisions", {})
        winner_pitcher_id = _safe_int(_gn(decisions_node, "winner").get("id"), -1)
        winner_pitcher_name = _str(_gn(decisions_node, "winner").get("fullName"), "None")
        loser_pitcher_id = _safe_int(_gn(decisions_node, "loser").get("id"), -1)
        loser_pitcher_name = _str(_gn(decisions_node, "loser").get("fullName"), "None")
        save_pitcher_id = _safe_int(_gn(decisions_node, "save").get("id"), -1)
        save_pitcher_name = _str(_gn(decisions_node, "save").get("fullName"), "None")

        # --- MANAGER CHALLENGES ---
        review_node = game_data_node.get("review") or {}
        review_home_used = _safe_int(_gn(review_node, "home").get("used"), 0)
        review_home_remaining = _safe_int(_gn(review_node, "home").get("remaining"), 0)
        review_away_used = _safe_int(_gn(review_node, "away").get("used"), 0)
        review_away_remaining = _safe_int(_gn(review_node, "away").get("remaining"), 0)

        # --- FLAGS ---
        flags_node = game_data_node.get("flags") or {}
        flag_no_hitter = bool(flags_node.get("noHitter", False))
        flag_perfect_game = bool(flags_node.get("perfectGame", False))
        flag_away_no_hitter = bool(flags_node.get("awayTeamNoHitter", False))
        flag_home_no_hitter = bool(flags_node.get("homeTeamNoHitter", False))

        # --- PROBABLE PITCHERS ---
        probable_node = game_data_node.get("probablePitchers") or {}
        probable_home_id = _safe_int(_gn(probable_node, "home").get("id"), -1)
        probable_away_id = _safe_int(_gn(probable_node, "away").get("id"), -1)

        # --- GAME LEADERS ---
        leaders_node = live_data_node.get("leaders") or {}
        leader_hit_distance = _safe_float(_gn(leaders_node, "hitDistance").get("value"))
        leader_hit_distance_pid = _safe_int(_gn(leaders_node, "hitDistance", "player").get("id"), -1)
        leader_hit_speed = _safe_float(_gn(leaders_node, "hitSpeed").get("value"))
        leader_hit_speed_pid = _safe_int(_gn(leaders_node, "hitSpeed", "player").get("id"), -1)
        leader_pitch_speed = _safe_float(_gn(leaders_node, "pitchSpeed").get("value"))
        leader_pitch_speed_pid = _safe_int(_gn(leaders_node, "pitchSpeed", "player").get("id"), -1)

        # --- ALERTS ---
        alerts_raw = game_data_node.get("alerts", [])
        game_alerts_json = json.dumps(alerts_raw) if alerts_raw else "[]"
        if alerts_raw:
            logger.debug(f"[game={game_pk}] {len(alerts_raw)} game alert(s) found.")

        logger.debug(
            f"[game={game_pk}] Metadata parsed | venue={venue_name} ({venue_id}) | "
            f"date={game_date_str} | {day_night} | type={game_type_code} DH={double_header} | "
            f"weather={weather_condition} {weather_temp}F | "
            f"home={home_rec['abbr']}({home_rec['id']}) {home_rec['wins']}-{home_rec['losses']} | "
            f"away={away_rec['abbr']}({away_rec['id']}) {away_rec['wins']}-{away_rec['losses']} | "
            f"umpires HP={ump_hp} | winner={winner_pitcher_name} loser={loser_pitcher_name} | "
            f"flags: no_hitter={flag_no_hitter} perfect={flag_perfect_game}"
        )

        # --- LINESCORE ---
        linescore_innings_raw = live_data_node.get("linescore", {}).get("innings", [])
        logger.debug(f"[game={game_pk}] linescore: {len(linescore_innings_raw)} innings found.")
        linescore_records: List[Dict[str, Any]] = []
        for inning in linescore_innings_raw:
            ih, ia = inning.get("home", {}), inning.get("away", {})
            linescore_records.append({
                "game_pk": int(game_pk), "season": int(season),
                "inning": _safe_int(inning.get("num"), 0),
                "home_runs": _safe_int(ih.get("runs"), 0), "away_runs": _safe_int(ia.get("runs"), 0),
                "home_hits": _safe_int(ih.get("hits"), 0), "away_hits": _safe_int(ia.get("hits"), 0),
                "home_errors": _safe_int(ih.get("errors"), 0), "away_errors": _safe_int(ia.get("errors"), 0),
                "home_left_on_base": _safe_int(ih.get("leftOnBase"), 0),
                "away_left_on_base": _safe_int(ia.get("leftOnBase"), 0),
            })

        # --- PLAYER BIOS ---
        players_raw = game_data_node.get("players", {})
        logger.debug(f"[game={game_pk}] gameData.players: {len(players_raw)} entries in roster dict.")
        new_player_records: List[Dict[str, Any]] = []
        skipped_players = 0
        for player_data in players_raw.values():
            pid = _safe_int(player_data.get("id"), -1)
            if pid == -1:
                logger.warning(f"[game={game_pk}] Player entry missing id: {player_data.get('fullName', 'unknown')}")
                continue
            with self._player_lock:
                if pid in self._seen_player_ids:
                    skipped_players += 1
                    continue
                self._seen_player_ids.add(pid)
            pos = player_data.get("primaryPosition", {})
            new_player_records.append({
                "player_id": pid,
                "full_name": _str(player_data.get("fullName"), "Unknown"),
                "use_name": _str(player_data.get("useName"), "Unknown"),
                "boxscore_name": _str(player_data.get("boxscoreName"), "Unknown"),
                "first_name": _str(player_data.get("firstName"), "Unknown"),
                "last_name": _str(player_data.get("lastName"), "Unknown"),
                "primary_number": _str(player_data.get("primaryNumber"), "None"),
                "birth_date": _str(player_data.get("birthDate"), "Unknown"),
                "birth_city": _str(player_data.get("birthCity"), "Unknown"),
                "birth_state": _str(player_data.get("birthStateProvince"), "Unknown"),
                "birth_country": _str(player_data.get("birthCountry"), "Unknown"),
                "height": _str(player_data.get("height"), "Unknown"),
                "weight": _safe_float(player_data.get("weight")),
                "current_age": _safe_int(player_data.get("currentAge"), -1),
                "strike_zone_top": _safe_float(player_data.get("strikeZoneTop")),
                "strike_zone_bottom": _safe_float(player_data.get("strikeZoneBottom")),
                "position_code": _str(pos.get("code"), "Unknown"),
                "position_name": _str(pos.get("name"), "Unknown"),
                "position_type": _str(pos.get("type"), "Unknown"),
                "position_abbreviation": _str(pos.get("abbreviation"), "Unknown"),
                "bat_side": _str(_gn(player_data, "batSide").get("code"), "Unknown"),
                "pitch_hand": _str(_gn(player_data, "pitchHand").get("code"), "Unknown"),
                "mlb_debut_date": _str(player_data.get("mlbDebutDate"), "Unknown"),
                "draft_year": _safe_int(player_data.get("draftYear"), -1),
                "is_active": bool(player_data.get("active", False)),
            })
        logger.debug(f"[game={game_pk}] Players: {len(new_player_records)} new, {skipped_players} already seen.")

        # --- BOXSCORE BATTING / PITCHING ---
        boxscore_batting_records: List[Dict[str, Any]] = []
        boxscore_pitching_records: List[Dict[str, Any]] = []
        bs_teams = boxscore_node.get("teams", {})

        for side in ("home", "away"):
            bs_team = bs_teams.get(side, {})
            bs_players = bs_team.get("players", {})
            pitchers_list = bs_team.get("pitchers", [])
            starter_pid = pitchers_list[0] if pitchers_list else -1
            logger.debug(f"[game={game_pk}] boxscore.teams.{side}: {len(bs_players)} players, {len(pitchers_list)} pitchers")

            for player_node in bs_players.values():
                pid = _safe_int(_gn(player_node, "person").get("id"), -1)
                pname = _str(_gn(player_node, "person").get("fullName"), "Unknown")
                stats = player_node.get("stats") or {}
                season_stats = player_node.get("seasonStats") or {}
                is_sub = bool(_gn(player_node, "gameStatus").get("isSubstitute", False))
                bo_raw = player_node.get("battingOrder")
                all_pos = player_node.get("allPositions", [])

                bat = stats.get("batting", {})
                sba = season_stats.get("batting", {})
                pit = stats.get("pitching", {})
                spi = season_stats.get("pitching", {})

                if bat:
                    boxscore_batting_records.append({
                        "game_pk": int(game_pk), "season": int(season),
                        "player_id": pid, "player_name": pname, "side": side,
                        "batting_order": _safe_int(str(bo_raw).rstrip("0") if bo_raw else None, -1),
                        "all_positions_json": json.dumps([p.get("abbreviation", p.get("code")) for p in all_pos]),
                        "is_substitute": is_sub,
                        "game_ab": _safe_int(bat.get("atBats"), 0),
                        "game_runs": _safe_int(bat.get("runs"), 0),
                        "game_hits": _safe_int(bat.get("hits"), 0),
                        "game_doubles": _safe_int(bat.get("doubles"), 0),
                        "game_triples": _safe_int(bat.get("triples"), 0),
                        "game_hr": _safe_int(bat.get("homeRuns"), 0),
                        "game_rbi": _safe_int(bat.get("rbi"), 0),
                        "game_bb": _safe_int(bat.get("baseOnBalls"), 0),
                        "game_ibb": _safe_int(bat.get("intentionalWalks"), 0),
                        "game_so": _safe_int(bat.get("strikeOuts"), 0),
                        "game_sb": _safe_int(bat.get("stolenBases"), 0),
                        "game_cs": _safe_int(bat.get("caughtStealing"), 0),
                        "game_hbp": _safe_int(bat.get("hitByPitch"), 0),
                        "game_sac": _safe_int(bat.get("sacBunts"), 0),
                        "game_sf": _safe_int(bat.get("sacFlies"), 0),
                        "game_gidp": _safe_int(bat.get("groundIntoDoublePlay"), 0),
                        "game_lob": _safe_int(bat.get("leftOnBase"), 0),
                        "season_avg": _safe_float(sba.get("avg")),
                        "season_obp": _safe_float(sba.get("obp")),
                        "season_slg": _safe_float(sba.get("slg")),
                        "season_ops": _safe_float(sba.get("ops")),
                        "season_hr": _safe_int(sba.get("homeRuns"), 0),
                        "season_rbi": _safe_int(sba.get("rbi"), 0),
                        "season_sb": _safe_int(sba.get("stolenBases"), 0),
                        "season_games_played": _safe_int(sba.get("gamesPlayed"), 0),
                    })

                if pit:
                    boxscore_pitching_records.append({
                        "game_pk": int(game_pk), "season": int(season),
                        "player_id": pid, "player_name": pname, "side": side,
                        "is_starter": (pid == starter_pid),
                        "game_innings_pitched": _safe_float(pit.get("inningsPitched")),
                        "game_hits": _safe_int(pit.get("hits"), 0),
                        "game_runs": _safe_int(pit.get("runs"), 0),
                        "game_earned_runs": _safe_int(pit.get("earnedRuns"), 0),
                        "game_bb": _safe_int(pit.get("baseOnBalls"), 0),
                        "game_so": _safe_int(pit.get("strikeOuts"), 0),
                        "game_hr": _safe_int(pit.get("homeRuns"), 0),
                        "game_hbp": _safe_int(pit.get("hitBatsmen"), 0),
                        "game_pitches_thrown": _safe_int(pit.get("numberOfPitches"), 0),
                        "game_strikes_thrown": _safe_int(pit.get("strikes"), 0),
                        "game_balls_thrown": _safe_int(pit.get("balls"), 0),
                        "game_strikes_looking": _safe_int(pit.get("strikesLooking") or pit.get("strikeouts"), 0),
                        "game_strikes_swinging": _safe_int(pit.get("strikeSwinging") or pit.get("strikeoutSwings"), 0),
                        "season_era": _safe_float(spi.get("era")),
                        "season_whip": _safe_float(spi.get("whip")),
                        "season_wins": _safe_int(spi.get("wins"), 0),
                        "season_losses": _safe_int(spi.get("losses"), 0),
                        "season_saves": _safe_int(spi.get("saves"), 0),
                        "season_innings_pitched": _safe_float(spi.get("inningsPitched")),
                        "season_so": _safe_int(spi.get("strikeOuts"), 0),
                        "season_bb": _safe_int(spi.get("baseOnBalls"), 0),
                        "season_games_played": _safe_int(spi.get("gamesPlayed"), 0),
                    })

        logger.debug(f"[game={game_pk}] Boxscore: {len(boxscore_batting_records)} batting lines, {len(boxscore_pitching_records)} pitching lines")

        # --- SPRAY CHART ---
        hit_records: List[Dict[str, Any]] = []
        for inning_node in _gn(live_data_node, "plays").get("playsByInning", []):
            inning_num = _safe_int(inning_node.get("startIndex"), 0)
            for side in ("home", "away"):
                for hit in (_gn(inning_node, "hits").get(side) or []):
                    hit_records.append({
                        "game_pk": int(game_pk), "season": int(season), "inning": inning_num,
                        "side": side,
                        "batter_id": _safe_int(_gn(hit, "batter").get("id"), -1),
                        "pitcher_id": _safe_int(_gn(hit, "pitcher").get("id"), -1),
                        "hit_x": _safe_float(_gn(hit, "coordinates").get("x")),
                        "hit_y": _safe_float(_gn(hit, "coordinates").get("y")),
                        "hit_type": _str(hit.get("type"), "Unknown"),
                        "team_id": _safe_int(_gn(hit, "team").get("id"), -1),
                    })
        logger.debug(f"[game={game_pk}] Spray chart: {len(hit_records)} batted balls.")

        # --- PITCH / PLAY TABLE ---
        all_plays = (live_data_node.get("plays") or {}).get("allPlays") or []
        logger.debug(f"[game={game_pk}] liveData.plays.allPlays: {len(all_plays)} at-bats to flatten.")
        flattened_records: List[Dict[str, Any]] = []
        runner_records: List[Dict[str, Any]] = []

        game_ctx = {
            "game_pk": int(game_pk), "season": int(season), "game_date": str(game_date_str),
            "game_datetime_utc": game_datetime_utc, "game_number": game_number,
            "game_type_code": game_type_code, "double_header": double_header,
            "tiebreaker": tiebreaker, "series_description": series_description,
            "series_game_number": series_game_number, "games_in_series": games_in_series,
            "game_status_detail": game_status_detail, "game_status_code": game_status_code,
            "start_time_tbd": start_time_tbd,
            "venue_id": venue_id, "venue_name": venue_name, "venue_city": venue_city,
            "venue_state": venue_state, "venue_latitude": venue_latitude, "venue_longitude": venue_longitude,
            "venue_timezone": venue_timezone, "venue_tz_offset": venue_tz_offset,
            "venue_capacity": venue_capacity, "venue_surface": venue_surface, "venue_roof_type": venue_roof_type,
            "home_team_id": home_rec["id"], "home_team_name": home_rec["name"], "home_team_abbr": home_rec["abbr"],
            "home_league_id": home_rec["league_id"], "home_league_name": home_rec["league_name"],
            "home_division_id": home_rec["division_id"], "home_division_name": home_rec["division_name"],
            "home_wins": home_rec["wins"], "home_losses": home_rec["losses"], "home_win_pct": home_rec["win_pct"],
            "home_division_games_back": home_rec["division_games_back"],
            "home_wild_card_games_back": home_rec["wild_card_games_back"],
            "home_games_played": home_rec["games_played"],
            "away_team_id": away_rec["id"], "away_team_name": away_rec["name"], "away_team_abbr": away_rec["abbr"],
            "away_league_id": away_rec["league_id"], "away_league_name": away_rec["league_name"],
            "away_division_id": away_rec["division_id"], "away_division_name": away_rec["division_name"],
            "away_wins": away_rec["wins"], "away_losses": away_rec["losses"], "away_win_pct": away_rec["win_pct"],
            "away_division_games_back": away_rec["division_games_back"],
            "away_wild_card_games_back": away_rec["wild_card_games_back"],
            "away_games_played": away_rec["games_played"],
            "start_time": start_time, "day_night": day_night,
            "weather_condition": weather_condition, "weather_temp": weather_temp, "weather_wind": weather_wind,
            "attendance": attendance, "game_duration_minutes": game_duration_minutes,
            "umpire_hp": ump_hp, "umpire_1b": ump_1b, "umpire_2b": ump_2b, "umpire_3b": ump_3b,
            "winner_pitcher_id": winner_pitcher_id, "winner_pitcher_name": winner_pitcher_name,
            "loser_pitcher_id": loser_pitcher_id, "loser_pitcher_name": loser_pitcher_name,
            "save_pitcher_id": save_pitcher_id, "save_pitcher_name": save_pitcher_name,
            "review_home_challenges_used": review_home_used, "review_home_challenges_remaining": review_home_remaining,
            "review_away_challenges_used": review_away_used, "review_away_challenges_remaining": review_away_remaining,
            "flag_no_hitter": flag_no_hitter, "flag_perfect_game": flag_perfect_game,
            "flag_away_team_no_hitter": flag_away_no_hitter, "flag_home_team_no_hitter": flag_home_no_hitter,
            "probable_pitcher_home_id": probable_home_id, "probable_pitcher_away_id": probable_away_id,
            "leader_hit_distance": leader_hit_distance, "leader_hit_distance_player_id": leader_hit_distance_pid,
            "leader_hit_speed": leader_hit_speed, "leader_hit_speed_player_id": leader_hit_speed_pid,
            "leader_pitch_speed": leader_pitch_speed, "leader_pitch_speed_player_id": leader_pitch_speed_pid,
            "game_alerts_json": game_alerts_json,
        }

        _nan_pitch = {
            "pitch_sequence_index": 0, "play_id": "None", "pitch_event_type": "None",
            "is_pitch": False, "pitch_number": -1, "pitch_start_time": "None", "pitch_end_time": "None",
            "pitch_count_balls": -1, "pitch_count_strikes": -1, "pitch_count_outs": -1,
            "pitch_type": "None", "pitch_call": "None", "pitch_event_flags_json": "[]",
            "is_in_play": False, "is_strike": False, "is_ball": False, "has_review": False,
            "release_speed": float("nan"), "end_speed": float("nan"),
            "strike_zone_top": float("nan"), "strike_zone_bottom": float("nan"),
            "type_confidence": float("nan"), "plate_time": float("nan"), "extension": float("nan"),
            "coord_px": float("nan"), "coord_pz": float("nan"),
            "coord_x0": float("nan"), "coord_y0": float("nan"), "coord_z0": float("nan"),
            "coord_vx0": float("nan"), "coord_vy0": float("nan"), "coord_vz0": float("nan"),
            "coord_ax": float("nan"), "coord_ay": float("nan"), "coord_az": float("nan"),
            "pfx_x": float("nan"), "pfx_z": float("nan"),
            "break_angle": float("nan"), "break_length": float("nan"), "break_y": float("nan"),
            "spin_rate": float("nan"), "spin_direction": float("nan"), "zone_location": 0,
            "hit_launch_speed": float("nan"), "hit_launch_angle": float("nan"),
            "hit_total_distance": float("nan"), "hit_trajectory": "None", "hit_hardness": "None",
            "hit_coord_x": float("nan"), "hit_coord_y": float("nan"),
        }

        for idx, play in enumerate(all_plays):
            result_node = play.get("result") or {}
            about_node = play.get("about") or {}
            count_node = play.get("count") or {}
            matchup_node = play.get("matchup") or {}
            splits_node = matchup_node.get("splits") or {}

            # Runner table rows
            for runner in play.get("runners", []):
                mov = runner.get("movement") or {}
                det = runner.get("details") or {}
                credits_list = [
                    {"player_id": _safe_int(_gn(c, "player").get("id"), -1),
                     "credit": _str(c.get("credit")),
                     "position_code": _str(_gn(c, "position").get("code"))}
                    for c in runner.get("credits", [])
                ]
                runner_records.append({
                    "game_pk": int(game_pk), "season": int(season), "play_index": int(idx),
                    "play_event_index": _safe_int(det.get("playIndex"), -1),
                    "runner_id": _safe_int(_gn(det, "runner").get("id"), -1),
                    "runner_name": _str(_gn(det, "runner").get("fullName")),
                    "responsible_pitcher_id": _safe_int(_gn(det, "responsiblePitcher").get("id"), -1),
                    "movement_start": _str(mov.get("start")),
                    "movement_end": _str(mov.get("end")),
                    "is_out": bool(mov.get("isOut", False)),
                    "out_base": _str(mov.get("outBase")),
                    "out_number": _safe_int(mov.get("outNumber"), -1),
                    "is_scoring_event": bool(det.get("isScoringEvent", False)),
                    "rbi": bool(det.get("rbi", False)),
                    "earned": bool(det.get("earned", False)),
                    "team_unearned": bool(det.get("teamUnearned", False)),
                    "event": _str(det.get("event")),
                    "event_type": _str(det.get("eventType")),
                    "movement_reason": _str(det.get("movementReason")),
                    "credits_json": json.dumps(credits_list),
                })

            # Base state before play (runners with a non-null start position)
            pre_on_first_id = pre_on_second_id = pre_on_third_id = -1
            for runner in play.get("runners", []):
                start = _gn(runner, "movement").get("start")
                rid = _safe_int(_gn(runner, "details", "runner").get("id"), -1)
                if start == "1B":   pre_on_first_id = rid
                elif start == "2B": pre_on_second_id = rid
                elif start == "3B": pre_on_third_id = rid

            # Base state after play (runners who ended on base, not out and not scored)
            post_on_first_id = post_on_second_id = post_on_third_id = -1
            for runner in play.get("runners", []):
                mov = _gn(runner, "movement")
                end = mov.get("end")
                if mov.get("isOut") or end == "score":
                    continue
                rid = _safe_int(_gn(runner, "details", "runner").get("id"), -1)
                if end == "1B":   post_on_first_id = rid
                elif end == "2B": post_on_second_id = rid
                elif end == "3B": post_on_third_id = rid

            base_record = {
                **game_ctx,
                "play_index": int(idx),
                "at_bat_index": _safe_int(about_node.get("atBatIndex"), idx),
                "inning": _safe_int(about_node.get("inning"), 0),
                "half_inning": _str(about_node.get("halfInning"), "unknown"),
                "is_top_inning": about_node.get("halfInning") == "top",
                "captivating_index": _safe_int(about_node.get("captivatingIndex"), 0),
                "at_bat_start_time": _str(about_node.get("startTime")),
                "at_bat_end_time": _str(about_node.get("endTime")),
                "at_bat_has_review": bool(about_node.get("hasReview", False)),
                "at_bat_is_complete": bool(about_node.get("isComplete", True)),
                "batter_id": _safe_int(_gn(matchup_node, "batter").get("id"), -1),
                "batter_name": _str(_gn(matchup_node, "batter").get("fullName"), "Unknown"),
                "bat_side_code": _str(_gn(matchup_node, "batSide").get("code")),
                "pitcher_id": _safe_int(_gn(matchup_node, "pitcher").get("id"), -1),
                "pitcher_name": _str(_gn(matchup_node, "pitcher").get("fullName"), "Unknown"),
                "pitch_hand_code": _str(_gn(matchup_node, "pitchHand").get("code")),
                "split_batter": _str(splits_node.get("batter")),
                "split_pitcher": _str(splits_node.get("pitcher")),
                "men_on_base": _str(splits_node.get("menOnBase")),
                "pre_on_first_id": pre_on_first_id,
                "pre_on_second_id": pre_on_second_id,
                "pre_on_third_id": pre_on_third_id,
                "post_on_first_id": post_on_first_id,
                "post_on_second_id": post_on_second_id,
                "post_on_third_id": post_on_third_id,
                "at_bat_event": _str(result_node.get("event"), "unknown"),
                "event_type": _str(result_node.get("eventType"), "unknown"),
                # isScoringPlay lives in about{} per GUMBO spec; fall back to result{} for older game data
                "is_scoring_play": bool(about_node.get("isScoringPlay", result_node.get("isScoringPlay", False))),
                "rbi_count": _safe_int(result_node.get("rbi"), 0),
                "score_home": _safe_int(result_node.get("homeScore"), 0),
                "score_away": _safe_int(result_node.get("awayScore"), 0),
                "play_description": _str(result_node.get("description")),
                "cum_balls": _safe_int(count_node.get("balls"), 0),
                "cum_strikes": _safe_int(count_node.get("strikes"), 0),
                "cum_outs": _safe_int(count_node.get("outs"), 0),
            }

            play_events = play.get("playEvents", [])
            if not play_events:
                rec = base_record.copy()
                rec.update(_nan_pitch)
                flattened_records.append(rec)
            else:
                for p_idx, p_event in enumerate(play_events):
                    det = p_event.get("details") or {}
                    pd_ = p_event.get("pitchData") or {}
                    pc = pd_.get("coordinates") or {}
                    pb = pd_.get("breaks") or {}
                    hd = p_event.get("hitData") or {}
                    hc = hd.get("coordinates") or {}
                    cnt = p_event.get("count", {})
                    flags_raw = p_event.get("flags", [])

                    pitch_record = base_record.copy()
                    pitch_record.update({
                        "pitch_sequence_index": int(p_idx),
                        "play_id": _str(p_event.get("playId")),
                        "pitch_event_type": _str(p_event.get("type")),
                        "is_pitch": bool(p_event.get("isPitch", False)),
                        "pitch_number": _safe_int(p_event.get("pitchNumber"), -1),
                        "pitch_start_time": _str(p_event.get("startTime")),
                        "pitch_end_time": _str(p_event.get("endTime")),
                        "pitch_count_balls": _safe_int(cnt.get("balls"), -1),
                        "pitch_count_strikes": _safe_int(cnt.get("strikes"), -1),
                        "pitch_count_outs": _safe_int(cnt.get("outs"), -1),
                        "pitch_type": _str(_gn(det, "type").get("code")),
                        "pitch_call": _str(det.get("description")),
                        "pitch_event_flags_json": json.dumps(flags_raw) if flags_raw else "[]",
                        "is_in_play": bool(det.get("isInPlay", False)),
                        "is_strike": bool(det.get("isStrike", False)),
                        "is_ball": bool(det.get("isBall", False)),
                        "has_review": bool(det.get("hasReview", False)),
                        "release_speed": _safe_float(pd_.get("startSpeed")),
                        "end_speed": _safe_float(pd_.get("endSpeed")),
                        "strike_zone_top": _safe_float(pd_.get("strikeZoneTop")),
                        "strike_zone_bottom": _safe_float(pd_.get("strikeZoneBottom")),
                        "type_confidence": _safe_float(pd_.get("typeConfidence")),
                        "plate_time": _safe_float(pd_.get("plateTime")),
                        "extension": _safe_float(pd_.get("extension")),
                        "coord_px": _safe_float(pc.get("pX")), "coord_pz": _safe_float(pc.get("pZ")),
                        "coord_x0": _safe_float(pc.get("x0")), "coord_y0": _safe_float(pc.get("y0")),
                        "coord_z0": _safe_float(pc.get("z0")),
                        "coord_vx0": _safe_float(pc.get("vX0")), "coord_vy0": _safe_float(pc.get("vY0")),
                        "coord_vz0": _safe_float(pc.get("vZ0")),
                        "coord_ax": _safe_float(pc.get("aX")), "coord_ay": _safe_float(pc.get("aY")),
                        "coord_az": _safe_float(pc.get("aZ")),
                        "pfx_x": _safe_float(pc.get("pfxX")), "pfx_z": _safe_float(pc.get("pfxZ")),
                        "break_angle": _safe_float(pb.get("breakAngle")),
                        "break_length": _safe_float(pb.get("breakLength")),
                        "break_y": _safe_float(pb.get("breakY")),
                        "spin_rate": _safe_float(pb.get("spinRate")),
                        "spin_direction": _safe_float(pb.get("spinDirection")),
                        "zone_location": _safe_int(pd_.get("zone"), 0),
                        "hit_launch_speed": _safe_float(hd.get("launchSpeed")),
                        "hit_launch_angle": _safe_float(hd.get("launchAngle")),
                        "hit_total_distance": _safe_float(hd.get("totalDistance")),
                        "hit_trajectory": _str(hd.get("trajectory")),
                        "hit_hardness": _str(hd.get("hardness")),
                        "hit_coord_x": _safe_float(hc.get("coordX")),
                        "hit_coord_y": _safe_float(hc.get("coordY")),
                    })
                    flattened_records.append(pitch_record)

        logger.debug(
            f"[game={game_pk}] Flatten complete | pitch_rows={len(flattened_records)} "
            f"runner_rows={len(runner_records)} linescore_rows={len(linescore_records)} "
            f"new_players={len(new_player_records)} batting={len(boxscore_batting_records)} "
            f"pitching={len(boxscore_pitching_records)} hits={len(hit_records)}"
        )
        return {
            "pitches": flattened_records, "linescores": linescore_records,
            "runners": runner_records, "players": new_player_records,
            "boxscore_batting": boxscore_batting_records,
            "boxscore_pitching": boxscore_pitching_records, "hits": hit_records,
            # game_ctx is stamped onto every pitch row, so historical consumers
            # recover it by de-duplicating pitches. A pregame payload has zero
            # pitch rows yet the model still needs venue/probables/umpire/weather,
            # so it is surfaced separately. run_ingestion ignores this key.
            "game_context": game_ctx,
        }


# ---------------------------------------------------------------------------
# MAIN EXECUTION LOOP
# ---------------------------------------------------------------------------
def run_ingestion(target_games: List[Dict[str, Any]], checkpoint: CheckpointManager,
                  engine: GumboIngestionEngine, is_retry: bool = False):
    pitch_buf: List[Dict[str, Any]] = []
    linescore_buf: List[Dict[str, Any]] = []
    runner_buf: List[Dict[str, Any]] = []
    player_buf: List[Dict[str, Any]] = []
    batting_buf: List[Dict[str, Any]] = []
    pitching_buf: List[Dict[str, Any]] = []
    hits_buf: List[Dict[str, Any]] = []

    # Games whose data is in the buffers but not yet written to storage.
    # Checkpointing happens ONLY after a successful flush — never before.
    pending_pks: List[Dict[str, Any]] = []

    def _flush_and_checkpoint(label: str = "threshold"):
        """Write all buffers to storage, then checkpoint. If the write fails,
        mark every pending game as failed so they are retried next run."""
        if not pending_pks:
            return
        try:
            logger.debug(f"[flush:{label}] Writing {len(pending_pks)} games across all tables...")
            _flush_by_season(pitch_buf, save_pitches)
            _flush_by_season(linescore_buf, save_linescores)
            _flush_by_season(runner_buf, save_runners)
            _flush_by_season(batting_buf, save_boxscore_batting)
            _flush_by_season(pitching_buf, save_boxscore_pitching)
            _flush_by_season(hits_buf, save_hits)
            save_players(player_buf)
            # All tables written — safe to checkpoint
            for pk_info in pending_pks:
                if pk_info["is_retry"]:
                    checkpoint.clear_retry_entry(pk_info["game_pk"])
                else:
                    checkpoint.mark_completed(pk_info["game_pk"])
            logger.debug(f"[flush:{label}] Checkpointed {len(pending_pks)} games as completed.")
        except Exception as e:
            error_str = str(e)
            logger.error(f"[flush:{label}] Save failed — marking {len(pending_pks)} pending games for retry: {error_str}", exc_info=True)
            for pk_info in pending_pks:
                checkpoint.mark_failed(pk_info["game_pk"], pk_info["season"],
                                       reason=type(e).__name__, error=error_str)
            raise
        finally:
            pitch_buf.clear(); linescore_buf.clear(); runner_buf.clear()
            player_buf.clear(); batting_buf.clear(); pitching_buf.clear()
            hits_buf.clear(); pending_pks.clear()

    label = "Retry Progress" if is_retry else "Scraping Progress"

    pbar = tqdm(total=len(target_games), desc=label)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="GumboWorker") as executor:
        # Submit games in rolling batches of SUBMIT_BATCH_SIZE so we never have
        # more than that many Future objects (and their full parse results) alive
        # at once. Submitting all 166k at once would pin all results in memory.
        game_iter = iter(target_games)
        in_flight: Dict = {}

        def _fill_queue():
            while len(in_flight) < SUBMIT_BATCH_SIZE:
                game = next(game_iter, None)
                if game is None:
                    break
                f = executor.submit(engine.extract_and_flatten_game, game)
                in_flight[f] = game

        _fill_queue()

        while in_flight:
            # Wait for the next completed future
            done_future = next(as_completed(in_flight))
            game = in_flight.pop(done_future)
            game_pk, season = game["game_pk"], game["season"]

            try:
                result = done_future.result()
                pitch_buf.extend(result.get("pitches", []))
                linescore_buf.extend(result.get("linescores", []))
                runner_buf.extend(result.get("runners", []))
                player_buf.extend(result.get("players", []))
                batting_buf.extend(result.get("boxscore_batting", []))
                pitching_buf.extend(result.get("boxscore_pitching", []))
                hits_buf.extend(result.get("hits", []))

                pending_pks.append({"game_pk": game_pk, "season": season, "is_retry": is_retry})
                pbar.update(1)
                pbar.set_postfix({"Pitches": len(pitch_buf), "Pending": len(pending_pks)})

                if (len(pitch_buf) >= CHUNK_FLUSH_THRESHOLD or
                        len(linescore_buf) >= LINESCORE_FLUSH_THRESHOLD or
                        len(runner_buf) >= RUNNER_FLUSH_THRESHOLD or
                        len(player_buf) >= PLAYER_FLUSH_THRESHOLD or
                        len(batting_buf) >= BOXSCORE_FLUSH_THRESHOLD or
                        len(pitching_buf) >= BOXSCORE_FLUSH_THRESHOLD or
                        len(hits_buf) >= HITS_FLUSH_THRESHOLD):
                    _flush_and_checkpoint("threshold")

            except Exception as e:
                error_str = str(e)
                logger.error(f"Worker thread failed for gamePk={game_pk}: {error_str}", exc_info=True)
                checkpoint.mark_failed(game_pk, season, reason=type(e).__name__, error=error_str)
                pbar.update(1)

            # Refill the queue now that a slot freed up
            _fill_queue()

    pbar.close()

    # Final flush for whatever remains in the buffers
    logger.debug("[final flush] Draining remaining buffers...")
    _flush_and_checkpoint("final")


def master_execution_loop(dry_run: bool = True, retry_mode: bool = False):
    checkpoint = CheckpointManager()
    engine = GumboIngestionEngine()

    if dry_run:
        discovered = engine.discover_schedule()
        logger.info(f"Total historical game contexts identified: {len(discovered)}")
        logger.info("Dry Run: Validating schema against 1 sample game.")
        if not discovered:
            return
        result = engine.extract_and_flatten_game(discovered[-1])
        logger.info("=== DRY RUN PARSE VALIDATION SUCCESS ===")
        logger.info(
            f"  pitches={len(result['pitches'])} | linescores={len(result['linescores'])} | "
            f"runners={len(result['runners'])} | players={len(result['players'])} | "
            f"boxscore_batting={len(result['boxscore_batting'])} | "
            f"boxscore_pitching={len(result['boxscore_pitching'])} | hits={len(result['hits'])}"
        )
        if result["pitches"]:
            logger.info(f"  Pitch columns ({len(result['pitches'][0])}): {list(result['pitches'][0].keys())}")
        return

    if retry_mode:
        retry_games = checkpoint.get_retry_games()
        if not retry_games:
            logger.info("Retry queue is empty — nothing to retry.")
            return
        logger.info(f"=== RETRY MODE: Processing {len(retry_games)} queued games ===")
        run_ingestion(retry_games, checkpoint, engine, is_retry=True)
        logger.info("=== RETRY RUN FINISHED ===")
        return

    discovered = engine.discover_schedule()
    total_games = len(discovered)
    logger.info(f"Total historical game contexts identified: {total_games}")

    pending = [g for g in discovered if not checkpoint.is_completed(g["game_pk"])]
    skipped_count = total_games - len(pending)
    if skipped_count:
        logger.info(f"Skipping {skipped_count} already-completed games. {len(pending)} remaining.")

    run_ingestion(sorted(pending, key=lambda x: x["game_pk"], reverse=True), checkpoint, engine)
    logger.info("=== INGESTION COMPLETELY FINISHED ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB GUMBO API Historical Data Ingestion Pipeline.")
    parser.add_argument("--live", action="store_true",
                        help="Disable dry run and start pulling production Parquet assets.")
    parser.add_argument("--retry", action="store_true",
                        help="Process only the games in the retry queue.")
    parser.add_argument("--local", action="store_true",
                        help="Write to local disk instead of S3 (useful for testing off EC2).")
    args = parser.parse_args()

    USE_S3 = not args.local
    is_dry_run = not args.live and not args.retry

    dest = f"s3://{S3_BUCKET}/{S3_PREFIX}/" if USE_S3 else f"{DATA_DIR}/"

    if args.retry:
        logger.info(f"=== INITIALIZING RETRY MODE | dest={dest} ===")
    elif not is_dry_run:
        logger.info(f"=== INITIALIZING FULL PRODUCTION INGESTION WORKFLOW | dest={dest} ===")
    else:
        logger.info("=== INITIALIZING GUMBO INGESTION IN DRY RUN MODE ===")

    master_execution_loop(dry_run=is_dry_run, retry_mode=args.retry)
