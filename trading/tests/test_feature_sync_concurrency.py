"""Regression tests for unbounded `aws s3 sync` fan-out in FeatureManager.

Observed in production 2026-08-30 on the live trading box (t4g.large, 2 vCPU):
35 concurrent `aws s3 sync s3://.../data` processes, all children of the trading
runner, load average 51 on two cores, SSH unresponsive.

Two defects compose into a self-amplifying loop:

  1. `_sync_s3` has no concurrency guard. `_rebuild` is protected by
     `_rebuild_lock`, but `refresh_async` / `check_and_refresh` call `_sync_s3`
     BEFORE that lock, so every trigger launches another full-data-lake sync no
     matter how many are already running.

  2. When `_rebuild` skips because the lock is held it returns without calling
     `_refresh_known_pks`, so `_known_final_pks` never learns about the finalized
     game. `_check_for_new_finals` therefore reports the same game every scan
     cycle (~60s) forever, and each report fires another ~5-minute sync.

A sync is slower than the interval that triggers it, so concurrency grows without
bound. The invariant these tests pin is: at most one sync subprocess in flight,
regardless of how often a refresh is triggered.
"""
from __future__ import annotations

import threading
import time

import pytest

from trading import features as fx


class _ConcurrencyProbe:
    """Stands in for subprocess.run, recording peak overlap instead of syncing.

    The real failure is *concurrency*, not call count: a guard that serialises
    callers is correct even if every caller eventually runs. So this tracks how
    many invocations are simultaneously in flight, which is the quantity that
    pegged the box.
    """

    def __init__(self, dwell: float = 0.20):
        self._dwell = dwell
        self._lock = threading.Lock()
        self.live = 0
        self.peak = 0
        self.calls = 0

    def __call__(self, *args, **kwargs):
        with self._lock:
            self.calls += 1
            self.live += 1
            self.peak = max(self.peak, self.live)
        # Hold the "process" open long enough that any unguarded caller overlaps.
        time.sleep(self._dwell)
        with self._lock:
            self.live -= 1

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        return _R()


@pytest.fixture()
def fm(tmp_path):
    return fx.FeatureManager(
        s3_uri="s3://unit-test-bucket/data",
        features_path=tmp_path / "game_features.parquet",
        local_cache=tmp_path / "raw_cache",
    )


def test_concurrent_sync_calls_do_not_stack(fm, monkeypatch):
    """Eight simultaneous refresh triggers must not put eight syncs on the box."""
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(fx.subprocess, "run", probe)

    threads = [threading.Thread(target=fm._sync_s3) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert probe.peak == 1, (
        f"{probe.peak} concurrent `aws s3 sync` processes in flight; each one walks "
        "the entire data lake, and on a 2-vCPU box this is what drove load to 51"
    )


def test_repeated_scan_cycles_while_rebuilding_launch_one_sync(fm, monkeypatch):
    """The 60s scan loop must not fire a fresh sync every cycle for the same game.

    Reproduces the production loop directly: a rebuild is already in progress (lock
    held), so every cycle sees the same un-ingested Final and re-triggers. Ten cycles
    stood for ten minutes of real time on the box.
    """
    probe = _ConcurrencyProbe(dwell=0.05)
    monkeypatch.setattr(fx.subprocess, "run", probe)
    monkeypatch.setattr(fm, "is_stale", lambda: False)
    # The same finalized game is reported every cycle -- that is the bug's premise,
    # not an artifact of the test: _rebuild's skip path never refreshes known PKs.
    monkeypatch.setattr(fm, "_check_for_new_finals", lambda: [{"game_pk": 777}])

    # Hold the rebuild lock so _rebuild takes its "already in progress" branch,
    # exactly as it did behind the long-running build on the box.
    assert fm._rebuild_lock.acquire(blocking=False)
    try:
        for _ in range(10):
            fm.check_and_refresh()
    finally:
        fm._rebuild_lock.release()

    # Either acceptable shape passes: defer the sync entirely while a build holds the
    # lock (0), or sync once and dedupe the rest (1). What must not happen is the
    # trigger re-arming every cycle, which is unbounded in the length of the build.
    assert probe.calls <= 1, (
        f"{probe.calls} syncs launched across 10 scan cycles for one unchanged game; "
        "a skipped rebuild must not re-arm the sync trigger"
    )
