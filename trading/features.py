"""
pregame/trading/features.py
---------------------------
Feature pipeline manager for live trading.

Responsibilities:
1. Detect newly-finalized games via GUMBO schedule API (state-based, not poll-based)
2. Pull fresh raw data from S3 (where live_daemon.py deposits finalized games)
3. Run pregame/engineering/build.py to rebuild features incrementally
4. Signal the trading loop to reload models

Trigger model:
- On startup: compare GUMBO schedule (yesterday + today) against known game_pks
  in the feature store. Sync + rebuild if any Finals are missing.
- Each scan cycle (~60s): check GUMBO for new Final transitions. Trigger only
  when a game actually finishes — no wasteful periodic polling.
- Fallback safety net: rebuild if parquet exceeds FALLBACK_REFRESH_HOURS (24h).

The trading EC2 is self-contained: it pulls raw data and rebuilds locally.
No dependency on external pre-built artifacts.

Key constraint: tune_ratings=False during live trading. Rating parameters
were tuned at training time and saved to artifacts/. Re-tuning on every
rebuild would introduce instability mid-session.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import (
    S3_DATA_URI, FEATURES_MAX_AGE_HOURS,
    ARTIFACTS_DIR,
)

logger = logging.getLogger(__name__)

FEATURES_DIR = ARTIFACTS_DIR / "features"
FEATURES_PATH = FEATURES_DIR / "game_features.parquet"


class FeatureManager:
    """Manages the feature pipeline lifecycle for live trading."""

    def __init__(
        self,
        s3_uri: str = S3_DATA_URI,
        features_path: Path = FEATURES_PATH,
        local_cache: Optional[Path] = None,
    ):
        self._s3_uri = s3_uri
        self._features_path = features_path
        self._local_cache = local_cache or (ARTIFACTS_DIR / "raw_cache")
        self._features: Optional[pd.DataFrame] = None
        self._last_hash: Optional[str] = None
        self._last_rebuild: Optional[datetime] = None
        self._rebuild_lock = threading.Lock()
        # Separate from _rebuild_lock because _sync_s3 runs BEFORE that lock is taken.
        # Without a guard of its own, every refresh trigger launched another full
        # data-lake sync: 35 were observed in flight on the 2-vCPU trading box, load 51.
        self._sync_lock = threading.Lock()
        # Teams whose last completed game has not yet been incorporated into
        # game_features.parquet.  Populated by mark_teams_pending() on settlement;
        # cleared on successful refresh().
        self._teams_pending_rebuild: set[str] = set()
        self._pending_lock = threading.Lock()
        # State-based tracking: game_pks known to be in the feature store
        self._known_final_pks: set[int] = set()
        # GUMBO schedule cache for game context (synthetic row construction)
        self._game_context_cache: dict[str, list[dict]] = {}

    def load(self) -> pd.DataFrame:
        """Load game_features.parquet into memory. Call at startup.

        Also runs a startup completeness check: compares GUMBO schedule
        (yesterday + today) against the feature store and rebuilds if any
        finalized games are missing.
        """
        if not self._features_path.exists():
            logger.warning(f"Features not found at {self._features_path}; triggering rebuild")
            self.refresh()

        if self._features_path.exists():
            self._features = pd.read_parquet(self._features_path)
            self._last_hash = self._compute_hash()
            self._refresh_known_pks()
            logger.info(
                f"Loaded features: {len(self._features)} games, "
                f"latest game_date={self._features['game_date'].max()}"
            )
            # Startup completeness check
            missing = self._check_for_new_finals()
            if missing:
                logger.info(
                    f"Startup: {len(missing)} finalized game(s) missing from features — rebuilding"
                )
                self._sync_s3()
                if self._rebuild():
                    self._refresh_known_pks()
        else:
            logger.error("Features unavailable after rebuild attempt")
            self._features = pd.DataFrame()

        return self._features

    def get_features(self) -> pd.DataFrame:
        """Get current features DataFrame (already loaded)."""
        if self._features is None:
            return self.load()
        return self._features

    def mark_teams_pending(self, home_team: str, away_team: str) -> None:
        """Mark both teams as having an unprocessed settled game.

        Call this immediately after a settlement event, before triggering
        refresh_async.  Clears when refresh() completes successfully.
        """
        with self._pending_lock:
            self._teams_pending_rebuild.update([home_team, away_team])
        logger.debug(f"Teams pending rebuild: {self._teams_pending_rebuild}")

    def is_stale_for_game(self, home_team: str, away_team: str) -> bool:
        """Return True if features are not safe to use for this game.

        Two conditions trigger staleness:
        1. A rebuild is currently in flight (lock is held).
        2. Either team has a settled game not yet reflected in features
           (mark_teams_pending was called but refresh hasn't completed).
        """
        if self._rebuild_lock.locked():
            return True
        with self._pending_lock:
            return bool(self._teams_pending_rebuild & {home_team, away_team})

    def is_stale(self) -> bool:
        """Check if features need rebuild based on last successful rebuild time.

        Uses _last_rebuild (in-memory) if available, falling back to file mtime.
        This prevents the infinite-rebuild loop when mtime is old but no new
        games exist: the first rebuild sets _last_rebuild, subsequent scans
        check against that rather than the unchanged file mtime.
        """
        if not self._features_path.exists():
            return True

        if self._last_rebuild is not None:
            age_hours = (datetime.now(timezone.utc) - self._last_rebuild).total_seconds() / 3600.0
        else:
            mtime = datetime.fromtimestamp(self._features_path.stat().st_mtime, tz=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600.0

        if age_hours > FEATURES_MAX_AGE_HOURS:
            logger.info(f"Features are {age_hours:.1f}h old (max {FEATURES_MAX_AGE_HOURS}h)")
            return True

        return False

    def check_and_refresh(self) -> bool:
        """State-based refresh: check GUMBO for newly-finalized games.

        Returns True if features changed (caller should reload models).
        Called every scan cycle (~60s). Only triggers S3 sync + rebuild when
        a game has actually transitioned to Final since last check.

        Falls back to age-based rebuild if parquet exceeds FEATURES_MAX_AGE_HOURS.
        """
        # A rebuild already in flight has its own fresh sync behind it, and _rebuild's
        # "already in progress" path returns without calling _refresh_known_pks -- so the
        # finalized game stays unknown and is re-detected every cycle. Without this gate
        # each detection fired another sync, for as long as the build ran.
        if self._rebuild_lock.locked():
            logger.debug("Rebuild in progress; deferring refresh check")
            return False

        # Fallback safety net: age-based rebuild
        if self.is_stale():
            logger.info("Fallback trigger: features exceed max age")
            self._sync_s3()
            return self._rebuild()

        # State-based: check GUMBO for new Finals
        new_finals = self._check_for_new_finals()
        if not new_finals:
            return False

        logger.info(
            f"GUMBO detected {len(new_finals)} new Final game(s): "
            f"{[g['game_pk'] for g in new_finals]}"
        )

        # S3 sync to pull raw data deposited by live_daemon
        self._sync_s3()
        changed = self._rebuild()

        if changed:
            # Update known PKs from the refreshed feature store
            self._refresh_known_pks()

        return changed

    def _check_for_new_finals(self) -> list[dict]:
        """Query GUMBO schedule for games that are Final but not in our feature store."""
        from . import schedule as gumbo_schedule

        now_utc = datetime.now(timezone.utc)
        today_str = now_utc.strftime("%Y-%m-%d")
        yesterday_str = (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")

        # Populate known PKs on first call
        if not self._known_final_pks and self._features is not None:
            self._refresh_known_pks()

        new_finals = []
        for date_str in (yesterday_str, today_str):
            try:
                states = gumbo_schedule.get_game_states(date_str)
            except Exception as e:
                logger.debug(f"GUMBO schedule check failed for {date_str}: {e}")
                continue

            for g in states:
                if (g["abstract_state"] == "Final"
                        and g["game_pk"] is not None
                        and g["game_pk"] not in self._known_final_pks):
                    new_finals.append(g)

        return new_finals

    def _refresh_known_pks(self) -> None:
        """Update the set of game_pks known to be in the feature store."""
        if self._features is not None and "game_pk" in self._features.columns:
            self._known_final_pks = set(self._features["game_pk"].dropna().astype(int).values)
            logger.debug(f"Known final PKs: {len(self._known_final_pks)}")

    def get_game_context(self, away_team: str, home_team: str) -> Optional[dict]:
        """Get GUMBO game context for synthetic row construction."""
        from . import schedule as gumbo_schedule

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if today_str not in self._game_context_cache:
            try:
                self._game_context_cache[today_str] = (
                    gumbo_schedule.get_games_with_context(today_str)
                )
            except Exception as e:
                logger.warning(f"Failed to fetch game context: {e}")
                return None

        for g in self._game_context_cache.get(today_str, []):
            if g["home_abbr"] == home_team and g["away_abbr"] == away_team:
                return g

        return None

    def refresh(self) -> bool:
        """Sync S3 and rebuild features unconditionally.

        Returns True if features changed (caller should reload models).
        Thread-safe: only one rebuild runs at a time.
        """
        self._sync_s3()
        return self._rebuild()

    def _rebuild(self) -> bool:
        """Run the feature build pipeline. Thread-safe via rebuild lock."""
        if not self._rebuild_lock.acquire(blocking=False):
            logger.info("Rebuild already in progress, skipping")
            return False

        try:
            logger.info("Starting feature rebuild...")

            success = self._run_build()
            if not success:
                return False

            new_hash = self._compute_hash()
            if new_hash == self._last_hash:
                logger.info("Features unchanged after rebuild")
                self._last_rebuild = datetime.now(timezone.utc)
                with self._pending_lock:
                    self._teams_pending_rebuild.clear()
                return False

            new_features = pd.read_parquet(self._features_path)

            # Verify new parquet is a superset of the old one before accepting
            if self._features is not None:
                old_pks = set(self._features["game_pk"].dropna().astype(int))
                new_pks = set(new_features["game_pk"].dropna().astype(int))
                missing_pks = old_pks - new_pks
                if missing_pks:
                    logger.error(
                        f"REBUILD REJECTED: new parquet is missing {len(missing_pks)} "
                        f"game_pks that existed before. Keeping old features."
                    )
                    return False
                old_cols = set(self._features.columns)
                new_cols = set(new_features.columns)
                dropped_cols = old_cols - new_cols
                if dropped_cols:
                    # Reject if weather or other critical feature families are lost
                    critical_prefixes = ("air_density", "wind_toward", "wind_cross",
                                         "precip_", "humidity", "temperature_f")
                    critical_drops = {
                        c for c in dropped_cols
                        if any(c.startswith(p) for p in critical_prefixes)
                    }
                    if critical_drops:
                        logger.error(
                            f"REBUILD REJECTED: would drop {len(critical_drops)} "
                            f"protected columns: {sorted(critical_drops)[:10]}"
                        )
                        return False
                    logger.warning(
                        f"Rebuild dropped {len(dropped_cols)} columns: "
                        f"{sorted(dropped_cols)[:10]}"
                    )

            self._features = new_features
            self._last_hash = new_hash
            self._last_rebuild = datetime.now(timezone.utc)
            with self._pending_lock:
                self._teams_pending_rebuild.clear()
            logger.info(
                f"Features refreshed: {len(self._features)} games × "
                f"{len(self._features.columns)} cols, "
                f"latest={self._features['game_date'].max()}"
            )

            # Upload verified parquet to S3
            self._upload_features_to_s3()

            return True

        except Exception as e:
            logger.error(f"Feature rebuild failed: {e}")
            return False
        finally:
            self._rebuild_lock.release()

    def refresh_async(self, callback: Optional[callable] = None) -> None:
        """Run refresh in background thread. Calls callback(changed: bool) on completion."""
        def _run():
            self._sync_s3()
            changed = self._rebuild()
            if callback:
                callback(changed)

        threading.Thread(target=_run, daemon=True).start()

    def _sync_s3(self) -> bool:
        """aws s3 sync the delta. Returns True if any new files were downloaded.

        Skips rather than queues when a sync is already running: this is driven by a
        ~60s scan loop while a full-lake sync takes minutes, so waiting would just
        move the pile-up from the process table into the thread pool. Callers ignore
        the return value, so a skip is indistinguishable from "no new files" — which
        is the correct reading, since the in-flight sync is fetching them.
        """
        if not self._sync_lock.acquire(blocking=False):
            logger.info("S3 sync already in progress, skipping")
            return False
        try:
            return self._sync_s3_locked()
        finally:
            self._sync_lock.release()

    def _sync_s3_locked(self) -> bool:
        self._local_cache.mkdir(parents=True, exist_ok=True)
        # No --quiet so we can count transferred lines to detect new data.
        cmd = [
            "aws", "s3", "sync",
            self._s3_uri, str(self._local_cache),
        ]
        logger.info(f"S3 sync: {self._s3_uri} → {self._local_cache}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                logger.warning(f"S3 sync non-zero exit: {result.stderr[:200]}")
                return False
            new_files = [l for l in result.stdout.splitlines() if l.strip()]
            if new_files:
                logger.info(f"S3 sync: {len(new_files)} new file(s) downloaded")
            return bool(new_files)
        except subprocess.TimeoutExpired:
            logger.warning("S3 sync timed out (5min)")
            return False

    def _run_build(self) -> bool:
        """Run incremental feature build in-process.

        Loads raw data for only the current season, appends new game rows to a
        persistent game-frame checkpoint, then re-runs ratings + features on the
        full frame (~16k rows, ~55 MB). Stays within t4g.large memory envelope.
        """
        try:
            from classical_learning.engineering.build import build_features_incremental

            build_features_incremental(
                source=str(self._local_cache),
                output=FEATURES_DIR,
                tune_ratings=False,
            )
            logger.info("Incremental feature build completed successfully")
            return True
        except Exception as e:
            logger.error(f"Incremental feature build failed: {e}", exc_info=True)
            return False

    def _upload_features_to_s3(self) -> None:
        """Upload verified game_features.parquet to S3 artifact path."""
        s3_dest = f"{self._s3_uri.rstrip('/')}/artifacts/features/game_features.parquet"
        cmd = [
            "aws", "s3", "cp",
            str(self._features_path), s3_dest,
        ]
        logger.info(f"Uploading features to {s3_dest}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                logger.warning(f"S3 upload failed: {result.stderr[:200]}")
            else:
                logger.info("Features uploaded to S3 successfully")
        except subprocess.TimeoutExpired:
            logger.warning("S3 upload timed out (2min)")

    def _compute_hash(self) -> Optional[str]:
        """Hash the features parquet for change detection."""
        if not self._features_path.exists():
            return None
        h = hashlib.md5()
        with open(self._features_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
