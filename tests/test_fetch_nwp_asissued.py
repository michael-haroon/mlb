"""Contract tests for HRRR as-issued extraction planning.

The as-of tensor's leakage guarantee rests on `freshest_issue` and
`plan_game_tasks`: an issue admitted at decision time t whose availability lies
after t reproduces exactly the lead-time leak this redesign removes.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_curation" / "scripts"))

from fetch_nwp_asissued import (  # noqa: E402
    HRRR_AVAILABILITY_LAG_MIN,
    apcp_search,
    freshest_issue,
    issue_available_time,
    plan_game_tasks,
)


GH = pd.Timestamp("2023-07-14 23:00", tz="UTC")  # typical 7pm ET first pitch


def test_freshest_issue_respects_availability_not_issue_time():
    """At 23:00 the 22Z issue is NOT available (lands 23:15 with a 75-min lag);
    the freshest usable issue is 21Z. Filtering on issue time would pick 22Z."""
    assert HRRR_AVAILABILITY_LAG_MIN == 75
    assert freshest_issue(GH) == pd.Timestamp("2023-07-14 21:00", tz="UTC")


def test_freshest_issue_just_after_availability_boundary():
    t = pd.Timestamp("2023-07-14 23:15", tz="UTC")
    assert freshest_issue(t) == pd.Timestamp("2023-07-14 22:00", tz="UTC")
    t = pd.Timestamp("2023-07-14 23:14", tz="UTC")
    assert freshest_issue(t) == pd.Timestamp("2023-07-14 21:00", tz="UTC")


def test_every_planned_task_is_available_at_some_decision_hour():
    """No planned (issue, fxx) may require information from after the last
    decision hour — availability must precede game_hour + 6h for everything."""
    for issue, fxx in plan_game_tasks(GH):
        assert issue_available_time(issue) <= GH + pd.Timedelta(hours=6)


def test_pregame_decision_has_full_window_coverage():
    """d=0 must be able to fill all 7 target hours (-1..5) from issues available
    at first pitch — the pregame row cannot be structurally empty."""
    tasks = plan_game_tasks(GH)
    primary = freshest_issue(GH)
    for h in range(-1, 6):
        valid = GH + pd.Timedelta(hours=h)
        fxx = int((valid - primary) / pd.Timedelta(hours=1))
        assert (primary, fxx) in tasks, f"target hour {h} not covered at d=0"


def test_fallback_issue_planned_for_every_decision():
    """A missing archive file must not hole the tensor: for each decision's
    primary issue, the preceding issue is also planned (at higher lead)."""
    tasks = plan_game_tasks(GH)
    issues = {i for i, _ in tasks}
    for d in range(7):
        primary = freshest_issue(GH + pd.Timedelta(hours=d))
        assert primary - pd.Timedelta(hours=1) in issues


def test_lead_monotonically_nonincreasing_for_fixed_target_hour():
    """The living window: as d advances, the freshest lead for a fixed future
    hour must never increase."""
    for h in range(-1, 6):
        valid = GH + pd.Timedelta(hours=h)
        prev_lead = None
        for d in range(7):
            issue = freshest_issue(GH + pd.Timedelta(hours=d))
            lead = (valid - issue) / pd.Timedelta(hours=1)
            if prev_lead is not None:
                assert lead <= prev_lead
            prev_lead = lead


def test_no_analysis_frames_planned():
    """fxx=0 has no 1-h APCP bucket and is an analysis, not a forecast."""
    assert all(fxx >= 1 for _, fxx in plan_game_tasks(GH))
    assert all(fxx <= 18 for _, fxx in plan_game_tasks(GH))


def test_task_count_is_bounded():
    """Volume sanity: one game should plan tens of tasks, not hundreds —
    the backfill's feasibility estimate depends on this."""
    n = len(plan_game_tasks(GH))
    assert 20 <= n <= 60, n


def test_midnight_utc_game_crosses_date_boundary():
    """Late west-coast game: issues come from the previous UTC day."""
    gh = pd.Timestamp("2023-07-15 02:00", tz="UTC")  # 7pm PT
    tasks = plan_game_tasks(gh)
    assert any(issue.date() == pd.Timestamp("2023-07-14").date() for issue, _ in tasks)


def test_apcp_search_embeds_the_one_hour_bucket():
    assert ":APCP:surface:1-2 hour acc" in apcp_search(2)
    assert ":APCP:surface:7-8 hour acc" in apcp_search(8)
    # the 0-fxx running total must NOT match the fxx=2 pattern
    assert "0-2 hour" not in apcp_search(2)


# ── Concurrency + transient-failure contract ─────────────────────────────────
# Regression guard for the shard-F incident (2026-08-30): three extraction
# processes on one box shared /tmp/herbie_nwp, so one process's between-dates
# purge deleted another's in-flight GRIB subsets. 1,319 ENOENT + 11 truncated
# reads dropped 41% of planned tasks, and because resume keys on S3 object
# existence the holes became permanent.


def test_save_dir_is_process_unique():
    """Two extractions on one host must not be able to purge each other's
    in-flight subsets. Process-unique save dirs make the race impossible rather
    than unlikely."""
    from fetch_nwp_asissued import HERBIE_SAVE_DIR
    import os

    assert str(os.getpid()) in str(HERBIE_SAVE_DIR), (
        f"{HERBIE_SAVE_DIR} is shared across processes on the same host")


def test_transient_fetch_failure_is_retried_not_silently_dropped(monkeypatch):
    """A download/decode failure must be retried. Returning None on the first
    exception is what turned S3 flakiness into permanent data loss."""
    import fetch_nwp_asissued as m

    calls = {"n": 0}

    def flaky(issue, fxx, points, all_points=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise m.TransientFetchError("End of resource reached")
        return pd.DataFrame({"venue_id": [1]})

    monkeypatch.setattr(m, "fetch_issue_points", flaky)
    out = m.fetch_issue_points_retrying(GH, 3, pd.DataFrame(), None, attempts=3)
    assert out is not None and calls["n"] == 3


def test_archive_gap_is_not_retried(monkeypatch):
    """A genuine archive gap (issue never existed) returns None immediately —
    retrying it would multiply wall-clock over the 2015 era's real holes."""
    import fetch_nwp_asissued as m

    calls = {"n": 0}

    def gap(issue, fxx, points, all_points=None):
        calls["n"] += 1
        return None

    monkeypatch.setattr(m, "fetch_issue_points", gap)
    assert m.fetch_issue_points_retrying(GH, 3, pd.DataFrame(), None, attempts=3) is None
    assert calls["n"] == 1


def test_exhausted_retries_raise_so_the_date_is_not_written_silently():
    """After the retry budget, the failure must surface — the caller counts it
    and refuses to treat the date as complete."""
    import fetch_nwp_asissued as m

    def always_fail(issue, fxx, points, all_points=None):
        raise m.TransientFetchError("boom")

    orig = m.fetch_issue_points
    m.fetch_issue_points = always_fail
    try:
        with pytest.raises(m.TransientFetchError):
            m.fetch_issue_points_retrying(GH, 3, pd.DataFrame(), None, attempts=2)
    finally:
        m.fetch_issue_points = orig
