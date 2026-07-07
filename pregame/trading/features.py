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
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import threading
import time
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

    def is_stale(self) -> bool:
        """Check if features need rebuild based on age."""
        if not self._features_path.exists():
            return True

        mtime = datetime.fromtimestamp(self._features_path.stat().st_mtime, tz=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600.0

        if age_hours > FEATURES_MAX_AGE_HOURS:
            logger.info(f"Features are {age_hours:.1f}h old (max {FEATURES_MAX_AGE_HOURS}h)")
            return True

        return False

    def check_and_refresh(self) -> bool:
        """Check staleness and rebuild if needed. Returns True if features changed.

        Also runs on periodic schedule (every FEATURES_PERIODIC_REFRESH_HOURS).
        Runs in caller's thread — use refresh_async() for background.
        """
        if not self.is_stale():
            # Check periodic refresh
            if self._last_rebuild is not None:
                hours_since = (datetime.now(timezone.utc) - self._last_rebuild).total_seconds() / 3600.0
                if hours_since < FEATURES_PERIODIC_REFRESH_HOURS:
                    return False
        return self.refresh()

    def refresh(self) -> bool:
        """Pull fresh S3 data and rebuild features.

        Returns True if features changed (caller should reload models).
        Thread-safe: only one rebuild runs at a time.
        """
        if not self._rebuild_lock.acquire(blocking=False):
            logger.info("Rebuild already in progress, skipping")
            return False

        try:
            logger.info("Starting feature refresh...")

            # Step 1: Sync delta from S3
            self._sync_s3()

            # Step 2: Run build_features in subprocess
            success = self._run_build()
            if not success:
                return False

            # Step 3: Check if parquet actually changed
            new_hash = self._compute_hash()
            if new_hash == self._last_hash:
                logger.info("Features unchanged after rebuild")
                self._last_rebuild = datetime.now(timezone.utc)
                return False

            # Step 4: Reload into memory
            self._features = pd.read_parquet(self._features_path)
            self._last_hash = new_hash
            self._last_rebuild = datetime.now(timezone.utc)
            logger.info(
                f"Features refreshed: {len(self._features)} games, "
                f"latest={self._features['game_date'].max()}"
            )
            return True

        except Exception as e:
            logger.error(f"Feature refresh failed: {e}")
            return False
        finally:
            self._rebuild_lock.release()

    def refresh_async(self, callback: Optional[callable] = None) -> None:
        """Run refresh in background thread. Calls callback(changed: bool) on completion."""
        def _run():
            changed = self.refresh()
            if callback:
                callback(changed)

        threading.Thread(target=_run, daemon=True).start()

    def _sync_s3(self) -> None:
        """aws s3 sync the delta (only new files since last sync)."""
        self._local_cache.mkdir(parents=True, exist_ok=True)
        cmd = [
            "aws", "s3", "sync",
            self._s3_uri, str(self._local_cache),
            "--quiet",
        ]
        logger.info(f"S3 sync: {self._s3_uri} → {self._local_cache}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                logger.warning(f"S3 sync non-zero exit: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            logger.warning("S3 sync timed out (5min)")

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
