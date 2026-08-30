"""Tests for the HRRR completeness gate's full-sweep mode.

The gate exists because a date whose extraction dropped tasks is INVISIBLE to a
normal rerun: run_backfill() skips any date whose S3 key already exists, and a
partially written object is indistinguishable from a complete one through
head_object. So the only way to repair such a date is to enumerate it and force it,
which requires covering every date rather than a sample.

Two defects these tests pin:
  1. `--sample 0` must mean "all dates". main() used to pass
     `max(args.sample, 60)`, which silently turned the release gate back into a
     60-date sample -- a downgrade with no error and no visible symptom.
  2. The repairable-date list must be written even when empty, so a consumer can
     distinguish "gate ran, nothing to repair" from "gate never ran".
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_curation" / "scripts"))

import fetch_nwp_asissued  # noqa: E402
import verify_weather_archives as vwa  # noqa: E402


# ── Fixtures: a 4-date archive where exactly one date lost upstream-present tasks ──

DATES = ["2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04"]
LOST_DATE = "2025-05-03"      # extraction dropped tasks that DO exist upstream
GAP_DATE = "2025-05-04"       # dropped tasks that do NOT exist upstream (real gap)

# 10 planned tasks per date: one issue hour, leads 0..9.
_ISSUE = pd.Timestamp("2025-05-01T18:00:00Z")
PLANNED = {(_ISSUE, fxx) for fxx in range(10)}


@pytest.fixture
def archive(monkeypatch, tmp_path):
    """Wire the verifier to an in-memory archive with a known loss pattern."""
    monkeypatch.setattr(vwa, "_list", lambda prefix: [
        (f"data/weather/source=hrrr_asissued/date={d}.parquet", 10_000) for d in DATES
    ])

    games = pd.DataFrame({
        "game_date": pd.to_datetime(DATES),
        "game_hour_utc": [_ISSUE] * len(DATES),
        "venue_id": [1] * len(DATES),
    })
    monkeypatch.setattr(fetch_nwp_asissued, "load_population_games", lambda: games.copy())
    monkeypatch.setattr(fetch_nwp_asissued, "plan_game_tasks", lambda gh: set(PLANNED))

    # _upstream_exists only sees (issue, fxx), so the two failure modes are separated
    # by WHICH leads each bad date is missing: leads >= 4 exist upstream, leads < 4 do
    # not. LOST_DATE therefore holds 0..3 and is missing the recoverable 4..9;
    # GAP_DATE holds 4..9 and is missing the genuinely-absent 0..3.
    HELD = {LOST_DATE: range(4), GAP_DATE: range(4, 10)}

    def fake_read(key: str) -> pd.DataFrame:
        tag = key.split("date=")[1].replace(".parquet", "")
        leads = list(HELD.get(tag, range(10)))
        return pd.DataFrame({
            "issue_time_utc": [_ISSUE] * len(leads),
            "lead_hours": leads,
        })

    monkeypatch.setattr(vwa, "_read", fake_read)
    monkeypatch.setattr(vwa, "_upstream_exists", lambda issue, fxx: fxx >= 4)

    monkeypatch.setattr(vwa, "_fails", [])
    return tmp_path


def _fail_text() -> str:
    return "\n".join(vwa._fails)


def test_full_sweep_covers_every_date(archive, capsys):
    """sample_n <= 0 must read all dates, not a random subset."""
    vwa.check_completeness(0, repair_out=str(archive / "repair.txt"))
    out = capsys.readouterr().out
    assert f"FULL sweep over all {len(DATES)} archived dates" in out


def test_full_sweep_finds_the_lost_date(archive):
    vwa.check_completeness(0, repair_out=str(archive / "repair.txt"))
    assert LOST_DATE in _fail_text(), "the recoverable date was not flagged"
    assert "EXIST upstream" in _fail_text()


def test_repair_list_contains_only_recoverable_dates(archive):
    """A genuine upstream gap must NOT be queued for repair -- forcing it would burn
    the rate-limited HRRR fetch budget re-downloading data that does not exist."""
    p = archive / "repair.txt"
    vwa.check_completeness(0, repair_out=str(p))
    listed = [ln for ln in p.read_text().splitlines() if ln.strip()]
    assert LOST_DATE in listed
    assert GAP_DATE not in listed, "a real archive gap was wrongly queued for repair"


def test_repair_list_is_written_even_when_empty(monkeypatch, archive):
    """Distinguishes 'gate ran, nothing to repair' from 'gate never ran'."""
    monkeypatch.setattr(vwa, "_read", lambda key: pd.DataFrame({
        "issue_time_utc": [_ISSUE] * 10, "lead_hours": list(range(10)),
    }))
    p = archive / "repair_empty.txt"
    vwa.check_completeness(0, repair_out=str(p))
    assert p.exists(), "repair list must be written even with zero repairable dates"
    assert [ln for ln in p.read_text().splitlines() if ln.strip()] == []


def test_threading_does_not_change_the_verdict(archive):
    """Same archive, 1 worker vs many, must produce the same repair set."""
    p1, p16 = archive / "w1.txt", archive / "w16.txt"
    vwa.check_completeness(0, workers=1, repair_out=str(p1))
    monkeypatch_fails = list(vwa._fails)
    vwa._fails.clear()
    vwa.check_completeness(0, workers=16, repair_out=str(p16))
    assert p1.read_text() == p16.read_text()
    assert len(monkeypatch_fails) == len(vwa._fails)


def test_sample_zero_is_not_floored_to_sixty_in_main():
    """Pins defect 1 at the call site: `max(args.sample, 60)` over a 0 sample would
    downgrade a full release gate to a 60-date spot check, silently."""
    src = inspect.getsource(vwa.main)
    tree = ast.parse(src.lstrip())
    for node in ast.walk(tree):
        # any bare `max(args.sample, 60)` passed straight into check_completeness
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "check_completeness"):
            for arg in node.args:
                assert not (isinstance(arg, ast.Call)
                            and isinstance(arg.func, ast.Name)
                            and arg.func.id == "max"), (
                    "check_completeness is called with max(...), which cannot "
                    "represent a full sweep (sample_n <= 0)"
                )


def test_positive_sample_still_gets_a_floor():
    """The floor is deliberate for sampled runs -- per-year medians over <60 dates are
    too noisy to act on. Keep it for positive samples."""
    src = inspect.getsource(vwa.main)
    assert "max(args.sample, 60)" in src, (
        "the positive-sample floor was removed along with the bug"
    )


# ── check_coverage: absent dates, classified ──────────────────────────────────
#
# A missing date is only a failure if its data exists upstream. run_backfill writes
# nothing at all when every planned task is an archive gap, which really happens in
# the 2015-2016 era, so a naive "missing == fail" gate would block those seasons'
# builds permanently.

@pytest.fixture
def coverage_env(monkeypatch, tmp_path):
    """Population has 3 dates; only the first is archived."""
    pop = ["2015-06-18", "2015-06-19", "2015-06-20"]
    games = pd.DataFrame({
        "game_date": pd.to_datetime(pop),
        "game_hour_utc": [_ISSUE] * 3,
        "venue_id": [1] * 3,
    })
    monkeypatch.setattr(fetch_nwp_asissued, "load_population_games", lambda: games.copy())
    monkeypatch.setattr(fetch_nwp_asissued, "plan_game_tasks", lambda gh: set(PLANNED))
    monkeypatch.setattr(vwa, "_list", lambda prefix: [
        ("data/weather/source=hrrr_asissued/date=2015-06-18.parquet", 10_000)])
    monkeypatch.setattr(vwa, "_fails", [])
    return tmp_path


def test_absent_date_fails_when_its_data_exists_upstream(monkeypatch, coverage_env):
    monkeypatch.setattr(vwa, "_upstream_exists", lambda issue, fxx: True)
    vwa.check_coverage(repair_out=str(coverage_env / "cov.txt"))
    assert "EXIST upstream" in _fail_text()
    listed = (coverage_env / "cov.txt").read_text().split()
    assert listed == ["2015-06-19", "2015-06-20"]


def test_absent_date_passes_when_upstream_has_nothing(monkeypatch, coverage_env, capsys):
    """The 2015-era genuine gap: no object, and no data to put in one."""
    monkeypatch.setattr(vwa, "_upstream_exists", lambda issue, fxx: False)
    vwa.check_coverage(repair_out=str(coverage_env / "cov.txt"))
    assert vwa._fails == [], f"a provably unobtainable date was failed: {_fail_text()}"
    out = capsys.readouterr().out
    assert "provably unobtainable" in out
    assert (coverage_env / "cov.txt").read_text().strip() == ""


def test_coverage_passes_when_everything_is_present(monkeypatch, coverage_env):
    monkeypatch.setattr(vwa, "_list", lambda prefix: [
        (f"data/weather/source=hrrr_asissued/date={d}.parquet", 10_000)
        for d in ("2015-06-18", "2015-06-19", "2015-06-20")])
    vwa.check_coverage()
    assert vwa._fails == []
