"""
pregame/trading/features.py
---------------------------
Feature pipeline manager for live trading.

Responsibilities:
1. Detect when game_features.parquet is stale
2. Pull fresh raw data from S3 (where live_daemon.py deposits finalized games)
3. Run pregame/engineering/build.py to rebuild features
4. Signal the trading loop to reload models

The trading EC2 is self-contained: it pulls raw data and rebuilds locally.
No dependency on external pre-built artifacts.

Key constraint: tune_ratings=False during live trading. Rating parameters
were tuned at training time and saved to artifacts/. Re-tuning on every
rebuild would introduce instability mid-session.

Per-game staleness model:
- Settlement events call mark_teams_pending(home, away) before triggering a rebuild.
- is_stale_for_game(home, away) returns True while a rebuild is in flight OR while
  either team has a pending settled game not yet reflected in features.
- _sync_s3 detects new S3 files (without --quiet) so check_and_refresh can also
  trigger a rebuild on startup/restart if S3 has data the features don't yet have.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import (
    S3_DATA_URI, FEATURES_MAX_AGE_HOURS, FEATURES_PERIODIC_REFRESH_HOURS,
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
        # Teams whose last completed game has not yet been incorporated into
        # game_features.parquet.  Populated by mark_teams_pending() on settlement;
        # cleared on successful refresh().
        self._teams_pending_rebuild: set[str] = set()
        self._pending_lock = threading.Lock()

    def load(self) -> pd.DataFrame:
        """Load game_features.parquet into memory. Call at startup."""
        if not self._features_path.exists():
            logger.warning(f"Features not found at {self._features_path}; triggering rebuild")
            self.refresh()

        if self._features_path.exists():
            self._features = pd.read_parquet(self._features_path)
            self._last_hash = self._compute_hash()
            logger.info(
                f"Loaded features: {len(self._features)} games, "
                f"latest game_date={self._features['game_date'].max()}"
            )
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
        """Sync S3 and rebuild if features are stale, periodic interval is due,
        or S3 delivered new files.

        Returns True if features changed (caller should reload models).
        Runs in caller's thread — use refresh_async() for background.

        S3 sync only runs when at least one rebuild condition is already met, or
        when _last_rebuild is None (first run / restart). This avoids a network
        round-trip on every scan cycle.
        """
        age_stale = self.is_stale()
        periodic_due = False
        if self._last_rebuild is not None:
            hours_since = (datetime.now(timezone.utc) - self._last_rebuild).total_seconds() / 3600.0
            periodic_due = hours_since >= FEATURES_PERIODIC_REFRESH_HOURS
        first_run = self._last_rebuild is None

        if not (age_stale or periodic_due or first_run):
            return False

        new_s3_files = self._sync_s3()
        if not (age_stale or periodic_due or new_s3_files):
            # Sync ran but nothing changed and no other trigger — update timer.
            self._last_rebuild = datetime.now(timezone.utc)
            return False

        if new_s3_files and not age_stale and not periodic_due:
            logger.info("S3 had new files — triggering rebuild")

        return self._rebuild()

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

            self._features = pd.read_parquet(self._features_path)
            self._last_hash = new_hash
            self._last_rebuild = datetime.now(timezone.utc)
            with self._pending_lock:
                self._teams_pending_rebuild.clear()
            logger.info(
                f"Features refreshed: {len(self._features)} games, "
                f"latest={self._features['game_date'].max()}"
            )
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
        """aws s3 sync the delta. Returns True if any new files were downloaded."""
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
            from pregame.engineering.build import build_features_incremental

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

    def _compute_hash(self) -> Optional[str]:
        """Hash the features parquet for change detection."""
        if not self._features_path.exists():
            return None
        h = hashlib.md5()
        with open(self._features_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
