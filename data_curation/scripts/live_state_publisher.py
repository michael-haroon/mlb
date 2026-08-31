"""
Live game-state publisher for the deep-learning stack.
------------------------------------------------------
Publishes a partial-game snapshot to S3 on EVERY observed state change — not
only at Final. The DL stack reprices in-game, so it needs the pitch rows that
exist *now*; before this module the only S3 write happened when a game went
Final, by which time repricing is worthless.

WHY the rows come from download_history.extract_and_flatten_game
----------------------------------------------------------------
Live rows must be schema-identical to the historical artifact the model was fit
on, or the live frame silently drifts (a renamed column, a different null
sentinel, title-case vs snake_case) and the model is fed garbage it cannot
complain about. Rather than reimplement extraction, this module hands the
already-polled GUMBO payload back into the *same* flattener the training
artifact was built with. Parity by construction — the same argument as
weather_asof's shared `assemble_asof_tensor`.

WHAT IS PUBLISHED (per game, per revision)
------------------------------------------
  deep_learning/live_state/date=YYYY-MM-DD/game_pk=NNNNNN/pitches.parquet
      Partial pitch table, GUMBO order, dh.SCHEMA_TYPE_MAP dtypes. Zero rows
      before first pitch — that is a valid state, not an error.
  deep_learning/live_state/date=YYYY-MM-DD/game_pk=NNNNNN/game_meta.parquet
      One row: the game context (venue, probables, umpires, records, weather).
      Regime flags are NOT applied here — feature_store._add_regime_flags is a
      pure function of game_date, so the reader applies it and this module
      stays free of a deep_learning import.
  deep_learning/live_state/date=YYYY-MM-DD/game_pk=NNNNNN/state.json
      Serving control block: status codes, situation, and — critically — the
      `available` map saying which pregame fields actually exist yet. Training
      always had them; at Scheduled they do not exist, and encoding an absent
      probable pitcher as 0.0 collides with a legitimate hash value. The
      consumer must mask, and this is how it knows to.
  deep_learning/live_state/date=YYYY-MM-DD/index.json
      Discovery: one entry per tracked game so a server polls a single object.

Ordering: `polled_at` (epoch seconds) is the ordering key. `revision` is a
per-process counter and resets on daemon restart — do not order on it.

WRITE PATH: publish() only enqueues. A single writer thread does the flatten
and the S3 PUTs, because the daemon's poll loop is single-threaded across all
tracked games: a 150ms PUT × 3 objects × 15 live games would push the 10s live
poll cadence out by seconds and make every game's snapshot stale. The queue
keeps only the LATEST pending payload per game, so a slow or failing S3 sheds
work instead of building an unbounded backlog of stale snapshots.
"""

import io
import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd

import download_history as dh

logger = logging.getLogger("GUMBO_LIVE")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
LIVE_STATE_PREFIX = "deep_learning/live_state"

# Republish an unchanged game this often so a consumer can distinguish "nothing
# is happening" from "the daemon died". Between pitches nothing about the game
# changes, so without this the writes would be pure waste; with it, staleness is
# bounded and detectable.
HEARTBEAT_SECONDS = 300

# Pregame fields the model consumes but which do not exist at Scheduled. GUMBO
# emits placeholders for some of them, so presence is checked by value, not by
# key existence. See _availability().
_ABSENT_STRINGS = {"", "None", "none", "null", "nan", "Unknown", "-1", "-1.0"}


def _present(value: Any) -> bool:
    """True when a GUMBO field carries real information.

    download_history's _str/_safe_int coerce misses to "None"/-1 rather than
    leaving them null, so a key-existence check would report everything present.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() not in _ABSENT_STRINGS
    if isinstance(value, (int, float)):
        return value not in (-1, -1.0)
    return True


def _availability(ctx: Dict[str, Any]) -> Dict[str, bool]:
    """Which pregame fields exist in THIS payload.

    The training artifact is built from Final payloads, where these are ~always
    populated (measured on game_meta 2022-2026: umpire_hp missing 0.02%,
    probable pitchers 0.20%, weather_temp 0.02%). At Scheduled they are 0%
    populated. A model that never saw them absent cannot be trusted to handle
    the 0.0 they encode to, so the consumer masks on this map instead.
    """
    return {
        "umpire_hp": _present(ctx.get("umpire_hp")),
        "umpire_1b": _present(ctx.get("umpire_1b")),
        "probable_pitcher_home_id": _present(ctx.get("probable_pitcher_home_id")),
        "probable_pitcher_away_id": _present(ctx.get("probable_pitcher_away_id")),
        "weather_temp": _present(ctx.get("weather_temp")),
        "weather_condition": _present(ctx.get("weather_condition")),
        "weather_wind": _present(ctx.get("weather_wind")),
        "attendance": _present(ctx.get("attendance")),
        "venue_id": _present(ctx.get("venue_id")),
        "game_datetime_utc": _present(ctx.get("game_datetime_utc")),
    }


def _lineup_ids(payload: Dict[str, Any]) -> Dict[str, list]:
    """Announced batting order per side, from boxscore.teams.{side}.battingOrder.

    Announced order is what exists before first pitch. It is NOT the same set as
    the batters who eventually appear (measured: 22.19 distinct batters/game
    mean, 87% of games exceed the 18 announced starters), so a consumer building
    a player slate pregame must use this and accept that it will grow.
    """
    box = ((payload.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
    out = {}
    for side in ("home", "away"):
        team = box.get(side) or {}
        order = team.get("battingOrder") or []
        out[f"{side}_batting_order"] = [int(p) for p in order if p is not None]
        # Announced starter; distinct from probable_pitcher_* which can be stale.
        pitchers = team.get("pitchers") or []
        out[f"{side}_starting_pitcher_id"] = int(pitchers[0]) if pitchers else -1
    return out


class LiveStatePublisher:
    """Serializes live snapshots to S3 (or local disk under --local).

    Storage destination follows download_history.USE_S3 so the daemon's --local
    flag reroutes DL live state too; otherwise a local test run would silently
    write to the production bucket.
    """

    def __init__(self, engine: "dh.GumboIngestionEngine",
                 prefix: str = LIVE_STATE_PREFIX,
                 heartbeat_seconds: int = HEARTBEAT_SECONDS):
        self._engine = engine
        self._prefix = prefix.rstrip("/")
        self._heartbeat = heartbeat_seconds

        # game_pk -> (meta, payload) awaiting publish; only the newest is kept.
        self._pending: Dict[int, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
        self._pending_lock = threading.Condition()
        self._shutdown = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # game_pk -> (coded_state, n_pitch_rows, published_at) of the last write.
        self._last: Dict[int, Tuple[str, int, float]] = {}
        # date -> {game_pk: index entry}; rewritten whenever a game publishes.
        self._index: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._revision: Dict[int, int] = {}

    # ------------------------------------------------------------------ #
    #  LIFECYCLE                                                          #
    # ------------------------------------------------------------------ #
    def start(self):
        self._thread = threading.Thread(target=self._writer_loop,
                                        name="LiveStateWriter", daemon=True)
        self._thread.start()
        logger.info(f"[live-state] Publisher started -> {self._destination()}")

    def stop(self, timeout: float = 30.0):
        self._shutdown.set()
        with self._pending_lock:
            self._pending_lock.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("[live-state] Publisher stopped.")

    def _destination(self) -> str:
        if dh.USE_S3:
            return f"s3://{dh.S3_BUCKET}/{self._prefix}/"
        return os.path.join(dh.DATA_DIR, self._prefix) + "/"

    # ------------------------------------------------------------------ #
    #  ENQUEUE (called from the poll loop — must not block)               #
    # ------------------------------------------------------------------ #
    def publish(self, game_pk: int, season: int, game_date: str,
                abstract_state: str, coded_state: str, detailed_state: str,
                payload: Dict[str, Any], poll_lag_ms: float,
                force: bool = False):
        """Queue a snapshot. Coalesces: a newer payload replaces an unwritten one.

        `force` bypasses the change filter — used at Final so the terminal
        snapshot is always published even if the pitch count did not move.
        """
        n_plays = len((((payload.get("liveData") or {}).get("plays") or {})
                       .get("allPlays") or []))
        prev = self._last.get(game_pk)
        if not force and prev is not None:
            prev_state, prev_plays, prev_at = prev
            unchanged = (prev_state == coded_state and prev_plays == n_plays)
            if unchanged and (time.time() - prev_at) < self._heartbeat:
                logger.debug(
                    f"[live-state] gamePk={game_pk} unchanged "
                    f"(state={coded_state} plays={n_plays}) — skipping publish."
                )
                return

        meta = {
            "game_pk": int(game_pk),
            "season": int(season),
            "game_date": game_date,
            "abstract_state": abstract_state,
            "coded_state": coded_state,
            "detailed_state": detailed_state,
            "polled_at": time.time(),
            "poll_lag_ms": round(poll_lag_ms, 1),
            "n_plays": n_plays,
            "force": force,
        }
        with self._pending_lock:
            if game_pk in self._pending:
                logger.debug(f"[live-state] gamePk={game_pk} coalesced a pending publish.")
            self._pending[game_pk] = (meta, payload)
            self._pending_lock.notify()

    # ------------------------------------------------------------------ #
    #  WRITER THREAD                                                      #
    # ------------------------------------------------------------------ #
    def _writer_loop(self):
        while True:
            with self._pending_lock:
                while not self._pending and not self._shutdown.is_set():
                    self._pending_lock.wait(timeout=5.0)
                if not self._pending and self._shutdown.is_set():
                    return
                # Drain everything queued; each game's newest payload only.
                batch = self._pending
                self._pending = {}

            for game_pk, (meta, payload) in batch.items():
                try:
                    self._write_one(meta, payload)
                except Exception:
                    # A failed publish must never kill the writer or the daemon.
                    # The next poll re-enqueues a fresher snapshot anyway, and
                    # download_history still owns the durable Final write.
                    logger.error(
                        f"[live-state] gamePk={game_pk} publish failed "
                        f"(state={meta.get('coded_state')})", exc_info=True
                    )

    def _write_one(self, meta: Dict[str, Any], payload: Dict[str, Any]):
        game_pk = meta["game_pk"]
        t0 = time.time()

        tables = self._engine.extract_and_flatten_game(
            {"game_pk": game_pk, "season": meta["season"]}, payload=payload
        )
        pitches = tables.get("pitches") or []
        ctx = tables.get("game_context") or {}

        rev = self._revision.get(game_pk, 0) + 1
        self._revision[game_pk] = rev

        base = f"{self._prefix}/date={meta['game_date']}/game_pk={game_pk}"

        # pitches.parquet is rewritten whole rather than appended. A partial game
        # is ~330 rows at most, so a full rewrite costs less than the read-concat
        # -dedupe an append would need, and it leaves the consumer a single
        # atomic object instead of a set of chunks it must reconcile.
        self._put_parquet(f"{base}/pitches.parquet", pitches, dh.SCHEMA_TYPE_MAP)
        self._put_parquet(f"{base}/game_meta.parquet",
                          [ctx] if ctx else [], None)

        linescore = (payload.get("liveData") or {}).get("linescore") or {}
        state_doc = {
            **{k: meta[k] for k in ("game_pk", "season", "game_date",
                                    "abstract_state", "coded_state",
                                    "detailed_state", "polled_at",
                                    "poll_lag_ms")},
            "revision": rev,
            "published_at": time.time(),
            # prefix_length the model can actually condition on right now. 0 is
            # the pregame case and is the batch-composition-sensitive path in
            # GameTransformer._team_readout — see
            # deep_learning/tests/test_pregame_readout_invariance.py.
            "n_pitch_rows": len(pitches),
            "n_plays": meta["n_plays"],
            # Weather tensor inputs. fetch_live_asof(venue_id, game_hour_utc)
            # needs exactly these two, so they travel with the snapshot rather
            # than forcing the server to re-fetch the schedule.
            "venue_id": ctx.get("venue_id"),
            "game_datetime_utc": ctx.get("game_datetime_utc"),
            "gumbo_weather": (payload.get("gameData") or {}).get("weather"),
            "situation": {
                "current_inning": linescore.get("currentInning"),
                "inning_half": linescore.get("inningHalf"),
                "outs": linescore.get("outs"),
                "balls": linescore.get("balls"),
                "strikes": linescore.get("strikes"),
                "home_runs": ((linescore.get("teams") or {}).get("home") or {}).get("runs"),
                "away_runs": ((linescore.get("teams") or {}).get("away") or {}).get("runs"),
            },
            "available": _availability(ctx),
            **_lineup_ids(payload),
        }
        self._put_json(f"{base}/state.json", state_doc)

        self._last[game_pk] = (meta["coded_state"], meta["n_plays"], time.time())
        self._seed_index(meta["game_date"])
        self._index[meta["game_date"]][game_pk] = {
            "game_pk": game_pk,
            "coded_state": meta["coded_state"],
            "detailed_state": meta["detailed_state"],
            "revision": rev,
            "n_pitch_rows": len(pitches),
            "polled_at": meta["polled_at"],
            "prefix": f"{base}/",
        }
        self._write_index(meta["game_date"])

        logger.debug(
            f"[live-state] gamePk={game_pk} rev={rev} state={meta['coded_state']} "
            f"pitch_rows={len(pitches)} write={1000 * (time.time() - t0):.0f}ms"
        )

    def _seed_index(self, game_date: str):
        """Load an existing index before the first write of the day.

        The index is rewritten from memory, so without this a daemon restart
        mid-slate would republish an index containing only the games polled
        since the restart — silently un-listing games that are still live.
        """
        if game_date in self._index:
            return
        self._index[game_date] = {}
        existing = self._get_json(f"{self._prefix}/date={game_date}/index.json")
        for entry in (existing or {}).get("games") or []:
            try:
                self._index[game_date][int(entry["game_pk"])] = entry
            except (KeyError, TypeError, ValueError):
                continue
        if self._index[game_date]:
            logger.info(
                f"[live-state] Seeded {game_date} index with "
                f"{len(self._index[game_date])} pre-existing games."
            )

    def _write_index(self, game_date: str):
        entries = self._index.get(game_date) or {}
        self._put_json(f"{self._prefix}/date={game_date}/index.json", {
            "game_date": game_date,
            "updated_at": time.time(),
            "games": sorted(entries.values(), key=lambda e: e["game_pk"]),
        })

    # ------------------------------------------------------------------ #
    #  STORAGE                                                            #
    # ------------------------------------------------------------------ #
    def _put_parquet(self, key: str, records: list,
                     schema: Optional[Dict[str, str]]):
        """Write a (possibly empty) parquet object.

        An empty frame is still written: the consumer must be able to tell "no
        pitches yet" (a real pregame state) from "the object is missing" (the
        publisher never ran, so do not trade).
        """
        df = pd.DataFrame(records)
        if schema is not None:
            df = dh._apply_schema(df, schema)
        if dh.USE_S3:
            buf = io.BytesIO()
            df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
            dh._get_s3().put_object(Bucket=dh.S3_BUCKET, Key=key, Body=buf.getvalue())
        else:
            full = os.path.join(dh.DATA_DIR, key)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            tmp = full + ".tmp"
            df.to_parquet(tmp, engine="pyarrow", compression="snappy", index=False)
            os.replace(tmp, full)

    def _get_json(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            if dh.USE_S3:
                obj = dh._get_s3().get_object(Bucket=dh.S3_BUCKET, Key=key)
                return json.loads(obj["Body"].read().decode("utf-8"))
            full = os.path.join(dh.DATA_DIR, key)
            if not os.path.exists(full):
                return None
            with open(full) as f:
                return json.load(f)
        except Exception:
            # A missing or malformed index is expected on the first game of a
            # day; it must not block the publish that would create it.
            logger.debug(f"[live-state] No readable object at {key}.")
            return None

    def _put_json(self, key: str, doc: Dict[str, Any]):
        body = json.dumps(doc, default=str).encode("utf-8")
        if dh.USE_S3:
            dh._get_s3().put_object(Bucket=dh.S3_BUCKET, Key=key, Body=body,
                                    ContentType="application/json")
        else:
            full = os.path.join(dh.DATA_DIR, key)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            tmp = full + ".tmp"
            with open(tmp, "wb") as f:
                f.write(body)
            os.replace(tmp, full)

    def forget(self, game_pk: int):
        """Drop in-memory bookkeeping once a game is done being tracked.

        The S3 objects are deliberately left in place — they are the only record
        of what the model was shown at each revision, which is what post-hoc P&L
        attribution needs. Retention is a separate lifecycle-policy concern.
        """
        self._last.pop(game_pk, None)
        self._revision.pop(game_pk, None)
        with self._pending_lock:
            self._pending.pop(game_pk, None)
