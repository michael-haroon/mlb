"""
MLB GUMBO Live Feed Daemon
--------------------------
Responsibilities:
  1. Poll live game feed — every 10s when live, every 60s when passive/delayed.
  2. Write data/live_state/{game_pk}.json on every poll for real-time inference.
  3. On game Final: call download_history.run_ingestion() to persist to S3/parquet.
  4. Daily enrichment at 08:00 UTC: fetch_weather.run_daily_weather() archive refresh.
  5. Forecast refresh loop: fetch_weather.run_forecast_refresh() every 6h (dedicated thread).

download_history.py owns ALL persistent storage and checkpoint tracking.
If this daemon crashes mid-game, run download_history.py standalone —
it will fetch any game not yet in checkpoint.json and fill the gap.
"""

import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

import requests

# Import download_history as a module so we reuse its storage layer entirely.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import download_history as dh

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
SCHEDULE_FETCH_HOUR_UTC = 8    # UTC hour for daily schedule reload
PASSIVE_POLL_INTERVAL   = 60   # seconds when no game is live
LIVE_POLL_INTERVAL      = 10   # seconds when a game is in Live state
FORECAST_REFRESH_INTERVAL = 6 * 3600  # ECMWF HRES runs at 00/06/12/18Z
MAX_RETRIES             = 7
BASE_BACKOFF            = 2.0  # exponential backoff base (seconds)
MAX_BACKOFF             = 120.0

DATA_DIR       = "data"
LOG_DIR        = os.path.join(DATA_DIR, "logs")
LIVE_STATE_DIR = os.path.join(DATA_DIR, "live_state")  # inference JSON files
MLB_BASE       = "https://statsapi.mlb.com"

os.makedirs(LOG_DIR,        exist_ok=True)
os.makedirs(LIVE_STATE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logger = logging.getLogger("GUMBO_LIVE")
logger.setLevel(logging.DEBUG)

_fh = logging.FileHandler(os.path.join(LOG_DIR, "gumbo_live.log"))
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s"
))
logger.addHandler(_fh)

_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter(
    "[GUMBO LIVE] %(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"
))
logger.addHandler(_ch)

# ---------------------------------------------------------------------------
# UTILITY HELPERS
# ---------------------------------------------------------------------------
def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default

def _str(value: Any, default: str = "None") -> str:
    return str(value) if value is not None else default

def _gn(node: Any, *keys) -> dict:
    """Navigate nested dicts safely; returns {} on any miss."""
    for key in keys:
        if not isinstance(node, dict):
            return {}
        node = node.get(key)
        if node is None:
            return {}
    return node if isinstance(node, dict) else {}

# ---------------------------------------------------------------------------
# GAME STATE TRACKER
# ---------------------------------------------------------------------------
class GameState:
    """Lightweight state bag for one tracked game. No data buffering here."""

    __slots__ = (
        "game_pk", "season", "game_date",
        "abstract_state",   # "Preview" | "Live" | "Final"
        "coded_state",      # single-char MLB code: "S", "I", "D", "F", ...
        "detailed_state",   # human-readable: "In Progress", "Rain Delay", ...
        "last_poll",
    )

    def __init__(self, game_pk: int, season: int, game_date: str):
        self.game_pk        = game_pk
        self.season         = season
        self.game_date      = game_date
        self.abstract_state = "Preview"
        self.coded_state    = "S"
        self.detailed_state = "Scheduled"
        self.last_poll      = 0.0

    def is_live(self)    -> bool: return self.abstract_state == "Live"
    def is_final(self)   -> bool: return self.abstract_state == "Final"
    def is_delayed(self) -> bool:
        return (self.coded_state in ("D", "U") or
                "delay" in self.detailed_state.lower())

# ---------------------------------------------------------------------------
# HTTP CLIENT
# ---------------------------------------------------------------------------
class MLBClient:
    """Minimal HTTP client for schedule + live feed polling only."""

    def __init__(self):
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=5, pool_maxsize=10)
        self.session.mount("https://", adapter)
        self.session.mount("http://",  adapter)
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Accept":     "application/json",
        })

    def get(self, path: str, params: Optional[Dict] = None) -> Optional[Dict]:
        url = f"{MLB_BASE}{path}"
        for attempt in range(MAX_RETRIES):
            try:
                t0   = time.time()
                resp = self.session.get(url, params=params, timeout=20)
                logger.debug(f"GET {url} → {resp.status_code} ({(time.time()-t0)*1000:.0f}ms)")

                if resp.status_code == 429:
                    wait = min(BASE_BACKOFF ** (attempt + 1), MAX_BACKOFF)
                    logger.warning(f"Rate-limited (429). Backing off {wait:.1f}s")
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    wait = min(BASE_BACKOFF ** attempt, MAX_BACKOFF)
                    logger.warning(f"Server error {resp.status_code}. Retry in {wait:.1f}s")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.Timeout:
                wait = min(BASE_BACKOFF ** attempt, MAX_BACKOFF)
                logger.warning(f"Timeout attempt {attempt+1}/{MAX_RETRIES}. Retry in {wait:.1f}s")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)

            except requests.exceptions.ConnectionError as exc:
                wait = min(BASE_BACKOFF ** attempt, MAX_BACKOFF)
                logger.warning(f"Connection error: {exc}. Retry in {wait:.1f}s")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)

            except requests.exceptions.RequestException as exc:
                logger.error(f"Unrecoverable request error on {url}: {exc}", exc_info=True)
                return None

        logger.error(f"Retry exhaustion ({MAX_RETRIES} attempts) for {url}")
        return None

# ---------------------------------------------------------------------------
# LIVE DAEMON
# ---------------------------------------------------------------------------
class LiveDaemon:
    def __init__(self):
        self.client = MLBClient()

        # Shared instances of download_history's engine and checkpoint manager.
        # When a game reaches Final, _poll_game calls dh.run_ingestion() with
        # these — it handles extraction, parquet writes, and checkpointing.
        # USE_S3 must be set on the dh module before these are constructed.
        self._dh_checkpoint = dh.CheckpointManager()
        self._dh_engine     = dh.GumboIngestionEngine()

        # game_pk → GameState; protected by _matrix_lock
        self._matrix:      Dict[int, GameState] = {}
        self._matrix_lock  = threading.Lock()
        self._shutdown     = threading.Event()

        self._seeded_today:        Set[int]      = set()
        self._last_schedule_date: Optional[str]  = None

    # ------------------------------------------------------------------ #
    #  SIGNAL HANDLING                                                     #
    # ------------------------------------------------------------------ #
    def _install_signal_handlers(self):
        def _handle(sig, _):
            logger.info(f"Signal {sig} received — graceful shutdown.")
            self._shutdown.set()
        signal.signal(signal.SIGTERM, _handle)
        signal.signal(signal.SIGINT,  _handle)

    # ------------------------------------------------------------------ #
    #  SCHEDULE LOADER                                                     #
    # ------------------------------------------------------------------ #
    def _fetch_schedule(self, date_str: str) -> List[Dict[str, Any]]:
        payload = self.client.get("/api/v1/schedule", params={"sportId": 1, "date": date_str})
        if not payload:
            return []
        games = []
        for date_node in payload.get("dates") or []:
            for g in date_node.get("games") or []:
                gk = g.get("gamePk")
                if not gk:
                    continue
                status = g.get("status") or {}
                season = _safe_int(
                    g.get("season") or date_str[:4],
                    datetime.now(timezone.utc).year,
                )
                games.append({
                    "game_pk":        int(gk),
                    "season":         season,
                    "game_date":      date_str,
                    "abstract_state": _str(status.get("abstractGameState"), "Preview"),
                    "coded_state":    _str(status.get("codedGameState"),    "S"),
                    "detailed_state": _str(status.get("detailedState"),     "Scheduled"),
                })
        return games

    def _seed_schedule(self):
        """
        Load today + tomorrow into the tracking matrix (36-hour window handles
        West Coast games crossing UTC midnight).
        Games already Final + checkpointed are skipped — they're already in S3.
        Games already in the matrix are left as-is (don't reset live state).
        """
        now_utc   = datetime.now(timezone.utc)
        today_str = now_utc.strftime("%Y-%m-%d")
        tmrw_str  = (now_utc + timedelta(days=1)).strftime("%Y-%m-%d")

        if self._last_schedule_date == today_str:
            return
        self._last_schedule_date = today_str
        self._seeded_today       = set()

        logger.info(f"[schedule] Loading games for {today_str} and {tmrw_str}")
        for date_str in (today_str, tmrw_str):
            games = self._fetch_schedule(date_str)
            logger.info(f"[schedule] {date_str}: {len(games)} games found.")
            with self._matrix_lock:
                for g in games:
                    gk = g["game_pk"]

                    if gk in self._matrix:
                        self._seeded_today.add(gk)
                        continue

                    # Already Final and already written to S3 — nothing to do.
                    if (g["abstract_state"] == "Final" and
                            self._dh_checkpoint.is_completed(gk)):
                        logger.debug(f"[schedule] gamePk={gk} Final+checkpointed — skipping.")
                        continue

                    gs = GameState(gk, g["season"], g["game_date"])
                    gs.abstract_state = g["abstract_state"]
                    gs.coded_state    = g["coded_state"]
                    gs.detailed_state = g["detailed_state"]
                    self._matrix[gk]  = gs
                    self._seeded_today.add(gk)
                    logger.info(
                        f"[schedule] Added gamePk={gk} season={g['season']} "
                        f"date={date_str} state={g['abstract_state']}"
                    )

    # ------------------------------------------------------------------ #
    #  SINGLE-GAME POLL                                                    #
    # ------------------------------------------------------------------ #
    def _poll_game(self, state: GameState):
        game_pk = state.game_pk
        t0      = time.time()

        payload = self.client.get(f"/api/v1.1/game/{game_pk}/feed/live")
        if not payload:
            logger.warning(f"[poll] gamePk={game_pk} — empty payload; skipping cycle.")
            return

        lag_ms = (time.time() - t0) * 1000

        game_data   = payload.get("gameData") or {}
        status_node = game_data.get("status") or {}

        new_abstract = _str(status_node.get("abstractGameState"), state.abstract_state)
        new_coded    = _str(status_node.get("codedGameState"),    state.coded_state)
        new_detailed = _str(status_node.get("detailedState"),     state.detailed_state)

        if new_abstract != state.abstract_state or new_coded != state.coded_state:
            logger.info(
                f"[state] gamePk={game_pk} | "
                f"{state.abstract_state}/{state.coded_state}/{state.detailed_state} → "
                f"{new_abstract}/{new_coded}/{new_detailed}"
            )

        state.abstract_state = new_abstract
        state.coded_state    = new_coded
        state.detailed_state = new_detailed
        state.last_poll      = time.time()

        # ---- FINAL: delegate to download_history for all persistent storage ----
        if state.is_final():
            if not self._dh_checkpoint.is_completed(game_pk):
                logger.info(
                    f"[final] gamePk={game_pk} — calling download_history.run_ingestion()"
                )
                try:
                    dh.run_ingestion(
                        target_games=[{"game_pk": game_pk, "season": state.season}],
                        checkpoint=self._dh_checkpoint,
                        engine=self._dh_engine,
                    )
                    logger.info(f"[final] gamePk={game_pk} — persisted to storage.")
                except Exception:
                    # download_history marks the game in its retry queue.
                    # Running download_history.py standalone will recover it.
                    logger.error(
                        f"[final] gamePk={game_pk} — run_ingestion failed. "
                        f"Recover by running: python download_history.py --live",
                        exc_info=True,
                    )
            else:
                logger.info(f"[final] gamePk={game_pk} already checkpointed — no write needed.")

            # Clean up inference file and remove from tracking matrix
            _remove_live_file(game_pk)
            with self._matrix_lock:
                self._matrix.pop(game_pk, None)
            return

        # ---- NON-FINAL: write inference snapshot, nothing else in memory ----
        _write_live_snapshot(game_pk, state, payload, lag_ms)

    # ------------------------------------------------------------------ #
    #  POLLING LOOP                                                        #
    # ------------------------------------------------------------------ #
    def _polling_loop(self):
        """
        Single thread. Dispatches each tracked game at the correct interval:
          - Live + not delayed → every LIVE_POLL_INTERVAL seconds
          - Everything else    → every PASSIVE_POLL_INTERVAL seconds
        """
        while not self._shutdown.is_set():
            now = time.time()

            with self._matrix_lock:
                game_states = list(self._matrix.values())

            if not game_states:
                self._shutdown.wait(timeout=PASSIVE_POLL_INTERVAL)
                continue

            live_games    = [gs for gs in game_states if gs.is_live() and not gs.is_delayed()]
            passive_games = [gs for gs in game_states if not gs.is_live() or gs.is_delayed()]

            for gs in live_games:
                if now - gs.last_poll >= LIVE_POLL_INTERVAL:
                    try:
                        self._poll_game(gs)
                    except Exception:
                        logger.error(f"[poll] Uncaught error gamePk={gs.game_pk}", exc_info=True)

            for gs in passive_games:
                if now - gs.last_poll >= PASSIVE_POLL_INTERVAL:
                    try:
                        self._poll_game(gs)
                    except Exception:
                        logger.error(f"[poll] Uncaught error gamePk={gs.game_pk}", exc_info=True)

            sleep_for = LIVE_POLL_INTERVAL if live_games else PASSIVE_POLL_INTERVAL
            self._shutdown.wait(timeout=sleep_for)

    # ------------------------------------------------------------------ #
    #  DAILY ENRICHMENT                                                    #
    # ------------------------------------------------------------------ #
    def _daily_enrichment(self):
        """Fetch standings, rosters, player stats, and platoon splits for today.

        Called once per day right after _seed_schedule at 08:00 UTC.
        A failure here never propagates — daemon stays alive regardless.
        """
        try:
            import daily_enrichment as de
            from datetime import timezone as tz
            now_utc  = datetime.now(tz.utc)
            date_str = now_utc.strftime("%Y-%m-%d")
            season   = now_utc.year
            logger.info(f"[enrichment] Starting daily enrichment for {date_str}")
            de.USE_S3 = dh.USE_S3  # inherit --local flag if set
            de.run_daily_enrichment(date_str, season)
            logger.info(f"[enrichment] Daily enrichment complete for {date_str}")
        except Exception:
            logger.warning("[enrichment] Daily enrichment failed — daemon continues", exc_info=True)

        # Weather is a separate try block: an enrichment failure above must not
        # skip it. Nothing called run_daily_weather() before, so the forecast
        # parquet went stale and fetch_live_weather silently returned zeros.
        try:
            import fetch_weather as fw
            logger.info("[weather] Starting daily weather refresh")
            fw.run_daily_weather(local=not dh.USE_S3)
            logger.info("[weather] Daily weather refresh complete")
        except Exception:
            logger.warning("[weather] Daily weather refresh failed — daemon continues", exc_info=True)

    # ------------------------------------------------------------------ #
    #  FORECAST REFRESH LOOP                                               #
    # ------------------------------------------------------------------ #
    def _forecast_refresh_loop(self):
        """Re-pull forecast products every FORECAST_REFRESH_INTERVAL.

        Separate from _schedule_loop's once-daily archive refresh: without this,
        a 01:00 UTC game prices off a forecast issued at 08:00 UTC the previous
        day (~17h lead). ECMWF HRES initialises at 00/06/12/18Z, so 6h is the
        shortest interval that can surface a new run.
        """
        while not self._shutdown.is_set():
            # Wait first — _daily_enrichment already fetched forecasts at startup.
            if self._shutdown.wait(timeout=FORECAST_REFRESH_INTERVAL):
                return
            try:
                import fetch_weather as fw
                logger.info("[weather] Starting 6-hourly forecast refresh")
                fw.run_forecast_refresh(local=not dh.USE_S3)
                logger.info("[weather] Forecast refresh complete")
            except Exception:
                logger.warning("[weather] Forecast refresh failed — daemon continues", exc_info=True)

    # ------------------------------------------------------------------ #
    #  SCHEDULE RESET LOOP                                                 #
    # ------------------------------------------------------------------ #
    def _schedule_loop(self):
        """Fires at 08:00 UTC daily to seed new games into the matrix."""
        while not self._shutdown.is_set():
            now_utc       = datetime.now(timezone.utc)
            next_seed_utc = now_utc.replace(
                hour=SCHEDULE_FETCH_HOUR_UTC, minute=0, second=0, microsecond=0
            )
            if now_utc >= next_seed_utc:
                self._seed_schedule()
                self._daily_enrichment()
                next_seed_utc += timedelta(days=1)

            wait_secs = (next_seed_utc - datetime.now(timezone.utc)).total_seconds()
            logger.info(
                f"[scheduler] Next schedule load in {wait_secs/3600:.2f}h "
                f"({next_seed_utc.strftime('%Y-%m-%d %H:%M UTC')})"
            )
            self._shutdown.wait(timeout=max(wait_secs, 0))

    # ------------------------------------------------------------------ #
    #  ENTRY POINT                                                         #
    # ------------------------------------------------------------------ #
    def run(self):
        self._install_signal_handlers()

        dest = f"s3://{dh.S3_BUCKET}/{dh.S3_PREFIX}/" if dh.USE_S3 else f"{DATA_DIR}/"
        logger.info("=" * 60)
        logger.info("MLB GUMBO Live Daemon starting.")
        logger.info(f"Persistent storage (via download_history): {dest}")
        logger.info(f"Inference files: {LIVE_STATE_DIR}/{{gamePk}}.json")
        logger.info(f"Poll intervals: live={LIVE_POLL_INTERVAL}s  passive={PASSIVE_POLL_INTERVAL}s")
        logger.info("Crash recovery: python download_history.py --live  (fills missing games)")
        logger.info("=" * 60)

        self._seed_schedule()
        self._daily_enrichment()

        poll_thread     = threading.Thread(target=self._polling_loop,  name="PollLoop",     daemon=True)
        schedule_thread = threading.Thread(target=self._schedule_loop, name="ScheduleLoop", daemon=True)
        weather_thread  = threading.Thread(target=self._forecast_refresh_loop, name="WeatherLoop", daemon=True)

        poll_thread.start()
        schedule_thread.start()
        weather_thread.start()

        logger.info("Daemon threads launched. Waiting for shutdown signal (SIGTERM / SIGINT).")
        self._shutdown.wait()

        logger.info("Shutdown requested — waiting for threads to drain (up to 30s).")
        poll_thread.join(timeout=30)
        schedule_thread.join(timeout=5)
        weather_thread.join(timeout=5)

        remaining = list(self._matrix.keys())
        if remaining:
            logger.warning(f"Stopped with {len(remaining)} in-flight games: {remaining}")
            logger.warning("Run: python download_history.py --live  to recover them.")
        logger.info("MLB GUMBO Live Daemon stopped.")


# ---------------------------------------------------------------------------
# INFERENCE FILE HELPERS  (module-level so they're easy to import/test)
# ---------------------------------------------------------------------------
def _write_live_snapshot(game_pk: int, state: GameState,
                         payload: Dict[str, Any], lag_ms: float):
    """
    Atomically overwrite data/live_state/{game_pk}.json with the current
    game snapshot. Inference code reads this file; it is always consistent
    (write to .tmp then os.replace — atomic on Linux/macOS).

    Schema intentionally kept loose — fields undefined when game hasn't
    started yet will be None. Callers must use .get() defensively.
    """
    live_data      = payload.get("liveData")  or {}
    linescore_node = live_data.get("linescore") or {}
    plays_node     = live_data.get("plays")     or {}
    all_plays      = plays_node.get("allPlays") or []
    current_play   = plays_node.get("currentPlay")

    snapshot = {
        # ---- identity ----
        "game_pk":        game_pk,
        "season":         state.season,
        "game_date":      state.game_date,
        "abstract_state": state.abstract_state,
        "coded_state":    state.coded_state,
        "detailed_state": state.detailed_state,
        "polled_at":      time.time(),
        "poll_lag_ms":    round(lag_ms, 1),

        # ---- score & situation ----
        "linescore": {
            "current_inning":       linescore_node.get("currentInning"),
            "current_inning_label": linescore_node.get("currentInningOrdinal"),
            "inning_half":          linescore_node.get("inningHalf"),
            "outs":                 linescore_node.get("outs"),
            "balls":                linescore_node.get("balls"),
            "strikes":              linescore_node.get("strikes"),
            "home_runs":  _safe_int(_gn(linescore_node, "teams", "home").get("runs"),   0),
            "away_runs":  _safe_int(_gn(linescore_node, "teams", "away").get("runs"),   0),
            "home_hits":  _safe_int(_gn(linescore_node, "teams", "home").get("hits"),   0),
            "away_hits":  _safe_int(_gn(linescore_node, "teams", "away").get("hits"),   0),
            "home_errors":_safe_int(_gn(linescore_node, "teams", "home").get("errors"), 0),
            "away_errors":_safe_int(_gn(linescore_node, "teams", "away").get("errors"), 0),
            "innings":    linescore_node.get("innings") or [],
        },

        # ---- current at-bat (the pitch-by-pitch node inference will use most) ----
        "current_play": current_play,

        # ---- summary counters ----
        "total_plays_completed": len(all_plays),

        # ---- teams (useful for model feature lookup) ----
        "home_team_id":   _safe_int(_gn(payload.get("gameData") or {}, "teams", "home").get("id"), -1),
        "away_team_id":   _safe_int(_gn(payload.get("gameData") or {}, "teams", "away").get("id"), -1),
        "home_team_abbr": _str(_gn(payload.get("gameData") or {}, "teams", "home").get("abbreviation"), "UNK"),
        "away_team_abbr": _str(_gn(payload.get("gameData") or {}, "teams", "away").get("abbreviation"), "UNK"),

        # ---- weather (present once game starts) ----
        "weather": payload.get("gameData", {}).get("weather"),
    }

    live_path = os.path.join(LIVE_STATE_DIR, f"{game_pk}.json")
    tmp_path  = live_path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(snapshot, f)
        os.replace(tmp_path, live_path)
        logger.debug(
            f"[poll] gamePk={game_pk} lag={lag_ms:.0f}ms "
            f"state={state.abstract_state}/{state.coded_state} "
            f"plays={len(all_plays)}"
        )
    except OSError:
        logger.warning(f"[poll] gamePk={game_pk} live state write failed", exc_info=True)


def _remove_live_file(game_pk: int):
    try:
        os.remove(os.path.join(LIVE_STATE_DIR, f"{game_pk}.json"))
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# ENTRY
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLB GUMBO Live Feed Daemon")
    parser.add_argument(
        "--local", action="store_true",
        help="Write Parquet to local disk instead of S3 (sets dh.USE_S3=False).",
    )
    cli_args = parser.parse_args()

    # Set the flag on the download_history module BEFORE constructing LiveDaemon,
    # because CheckpointManager.__init__ reads from S3 or local based on this flag.
    if cli_args.local:
        dh.USE_S3 = False
        logger.info("--local flag: storage directed to local disk via download_history.")

    LiveDaemon().run()
