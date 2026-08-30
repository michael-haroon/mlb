"""The fetcher must not persist a date whose shortfall came from transient failures.

Proven defect (2026-08-30): the shared-/tmp Herbie purge race documented at
fetch_nwp_asissued.HERBIE_SAVE_DIR exhausted every retry on a large share of tasks
between 07:43 and 08:23 UTC, producing 25 archive objects with task fill 0.17-0.81.
Every one was written as though complete, and because run_backfill() skips any date
whose S3 key already exists — a partial object is indistinguishable from a complete
one through head_object — those dates were permanently poisoned. Only an explicit
--force rerun, driven by a separate verifier sweep, recovered them.

PID-scoping the Herbie directory fixed that particular cause. These tests guard the
consequence, which any sustained transient-failure source reproduces.

The fix rests on a distinction the code already makes and then threw away:

  * fetch_issue_points_retrying returns None for a genuine archive gap (the GRIB was
    never published). That data does not exist; the partial object is the best
    obtainable and lead-fallback planning covers it. -> WRITE.
  * fetch_issue_points_retrying RAISES after exhausting retries on a transient error.
    That data does exist upstream and we simply failed to get it. -> DO NOT WRITE,
    leave the date absent so the next run retries it.

Leaving a date absent is only safe because check_coverage() in
verify_weather_archives.py fails on any population date with no object.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_curation" / "scripts"))

import fetch_nwp_asissued as fna  # noqa: E402

DATE = "2025-08-15"
_ISSUE = pd.Timestamp(f"{DATE}T18:00:00Z")
N_TASKS = 10
TASKS = {(_ISSUE, fxx) for fxx in range(N_TASKS)}


def _row(issue, fxx):
    return pd.DataFrame({
        "venue_id": [1],
        "issue_time_utc": [issue],
        "valid_time_utc": [issue + pd.Timedelta(hours=fxx)],
        "lead_hours": [fxx],
        "t2m_k": [295.0],
    })


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """run_backfill wired to one date with 10 planned tasks and no real S3/HRRR."""
    games = pd.DataFrame({
        "game_date": [pd.Timestamp(DATE)],
        "game_hour_utc": [_ISSUE],
        "venue_id": [1],
    })
    monkeypatch.setattr(fna, "load_population_games", lambda: games.copy())
    monkeypatch.setattr(fna, "load_venue_points",
                        lambda: pd.DataFrame({"venue_id": [1], "lat": [40.0], "lon": [-75.0]}))
    monkeypatch.setattr(fna, "plan_game_tasks", lambda gh: set(TASKS))
    monkeypatch.setattr(fna, "_s3_key_exists", lambda key: False)
    monkeypatch.setattr(fna, "HERBIE_SAVE_DIR", tmp_path / "herbie")

    writes: list[tuple[str, int]] = []
    monkeypatch.setattr(fna, "_write_parquet_s3",
                        lambda df, key: writes.append((key, len(df))))
    return writes


def _run(monkeypatch, behaviour):
    """behaviour(issue, fxx) -> DataFrame | None | raises TransientFetchError."""
    monkeypatch.setattr(
        fna, "fetch_issue_points_retrying",
        lambda issue, fxx, pts, all_pts=None, **kw: behaviour(issue, fxx))
    fna.run_backfill(DATE, DATE, workers=2)


def test_complete_date_is_written(harness, monkeypatch):
    """Control: all 10 tasks succeed -> the date is persisted."""
    _run(monkeypatch, lambda i, f: _row(i, f))
    assert len(harness) == 1, "a complete date must be written"
    assert harness[0][1] == N_TASKS


def test_transient_failure_shortfall_is_not_persisted(harness, monkeypatch):
    """THE DEFECT: 7 of 10 tasks lost to transient errors (fill 0.30).

    Writing here poisons the date permanently, because every later run skips it.
    """
    def behaviour(issue, fxx):
        if fxx >= 3:
            raise fna.TransientFetchError(f"503 on f{fxx:02d}")
        return _row(issue, fxx)

    _run(monkeypatch, behaviour)
    assert harness == [], (
        "a date that lost 70% of its tasks to transient errors was persisted as "
        "complete; it will be skipped by every subsequent run"
    )


def test_genuine_archive_gap_is_still_persisted(harness, monkeypatch):
    """Counterpart: the same 0.30 fill, but the data does not exist upstream.

    Refusing to write here would loop forever on dates that can never be filled.
    """
    _run(monkeypatch, lambda i, f: _row(i, f) if f < 3 else None)
    assert len(harness) == 1, (
        "a genuine upstream gap must still be archived — the missing tasks are "
        "unobtainable and lead-fallback planning handles them"
    )
    assert harness[0][1] == 3


def test_small_transient_shortfall_is_tolerated(harness, monkeypatch):
    """One lost task out of ten (fill 0.90) is above the write floor.

    Without a floor, a single permanently-raising task would keep a date absent
    forever, and check_coverage would alarm on it every run with no way to clear it.
    """
    def behaviour(issue, fxx):
        if fxx == 9:
            raise fna.TransientFetchError("one flaky task")
        return _row(issue, fxx)

    monkeypatch.setattr(fna, "MIN_WRITE_FILL", 0.80)
    _run(monkeypatch, behaviour)
    assert len(harness) == 1, "a 0.90-fill date should still be written at floor 0.80"


def test_write_floor_matches_the_verifier_report_floor():
    """The writer must refuse exactly what the completeness gate would reject.

    If the writer's floor were below the verifier's, the pipeline would persist dates
    that the gate then flags forever — the loop we just spent a morning unwinding.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                           / "data_curation" / "scripts"))
    import verify_weather_archives as vwa
    assert fna.MIN_WRITE_FILL == vwa.DATE_FILL_REPORT_FLOOR
