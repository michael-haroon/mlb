"""The pre/post artifact diff must not be able to report a false PASS.

This tool exists to prove the three builder fixes took effect, so its own failure mode is
the dangerous one: if the diff reports PASSED on an artifact where a fix silently no-oped,
it launders an unfixed tensor as verified and the whole verification chain becomes
decorative. These tests drive main() through a monkeypatched loader -- the S3 read is the
only impure part -- and assert on the exit code, because that is what a gate reads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_curation" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deep_learning"))

import diff_weather_asof_artifacts as d  # noqa: E402
from mlb_dl.weather_asof import (  # noqa: E402
    ASOF_CHANNELS,
    IMPOSSIBLE_ZERO_OBS_DIMS,
    N_DECISIONS,
    N_DIMS,
    N_OBS_DIMS,
    N_TARGET_HOURS,
    OFF_FCST,
    OFF_FCST_MASK,
    OFF_LEAD,
    OFF_OBS,
    OFF_OBS_MASK,
)

N_GAMES = 4
PKS = np.arange(400000, 400000 + N_GAMES)


def tensor(fill: float = 1013.0) -> np.ndarray:
    """Fully populated, all masked in -- the strictest case for a coverage comparison."""
    T = np.zeros((N_GAMES, N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS), np.float32)
    T[..., OFF_FCST:OFF_FCST_MASK] = fill
    T[..., OFF_FCST_MASK:OFF_OBS] = 1.0
    T[..., OFF_OBS:OFF_OBS_MASK] = fill
    T[..., OFF_OBS_MASK:OFF_LEAD] = 1.0
    T[..., OFF_LEAD] = 0.5
    # Visibility lives in metres and must sit under the ceiling by default, or every
    # baseline run would report the clamp as having failed.
    T[..., OFF_FCST + 11] = 16093.0
    T[..., OFF_OBS + 11] = 16093.0
    return T


def run(monkeypatch, old, new, argv_extra=()):
    """-> exit code from main()."""
    pairs = {"OLD": (old, PKS), "NEW": (new, PKS)}
    monkeypatch.setattr(d, "load", lambda key: pairs[key])
    monkeypatch.setattr(sys, "argv",
                        ["diff", "--season", "2015", "--old", "OLD", "--new", "NEW",
                         *argv_extra])
    with pytest.raises(SystemExit) as e:
        d.main()
    return e.value.code


def test_identical_artifacts_pass(monkeypatch):
    assert run(monkeypatch, tensor(), tensor()) == 0


def test_a_fix_that_worked_passes(monkeypatch):
    """The expected real case: old carries masked-in zeros, new does not."""
    old = tensor()
    for dim in IMPOSSIBLE_ZERO_OBS_DIMS:
        old[0, 6, 0, OFF_OBS + dim] = 0.0          # value absent, mask still claims it
    new = tensor()
    for dim in IMPOSSIBLE_ZERO_OBS_DIMS:
        new[0, 6, 0, OFF_OBS + dim] = 0.0
        new[0, 6, 0, OFF_OBS_MASK + dim] = 0.0     # honestly masked out
    assert run(monkeypatch, old, new) == 0


def test_a_fix_that_no_oped_fails(monkeypatch):
    """THE case this tool exists for. The defect is present in both artifacts, so the
    builder change did nothing -- reporting PASSED here would be the worst outcome."""
    old = new = tensor()
    old = old.copy()
    old[0, 6, 0, OFF_OBS + 13] = 0.0
    new = new.copy()
    new[0, 6, 0, OFF_OBS + 13] = 0.0               # still masked in
    assert run(monkeypatch, old, new) == 1


def test_a_fix_that_only_half_worked_fails(monkeypatch):
    """Partial credit is still a failure: fixing 4 of 5 dims leaves a live defect."""
    old = tensor()
    new = tensor()
    for dim in IMPOSSIBLE_ZERO_OBS_DIMS:
        old[0, 6, 0, OFF_OBS + dim] = 0.0
        new[0, 6, 0, OFF_OBS + dim] = 0.0
        new[0, 6, 0, OFF_OBS_MASK + dim] = 0.0
    new[0, 6, 0, OFF_OBS_MASK + IMPOSSIBLE_ZERO_OBS_DIMS[-1]] = 1.0   # one left unfixed
    assert run(monkeypatch, old, new) == 1


def test_visibility_still_above_the_ceiling_fails(monkeypatch):
    old = tensor()
    old[0, 0, 0, OFF_OBS + 11] = 112700.0
    new = tensor()
    new[0, 0, 0, OFF_OBS + 11] = 112700.0          # clamp never applied
    assert run(monkeypatch, old, new) == 1


def test_the_visibility_clamp_working_passes(monkeypatch):
    old = tensor()
    old[0, 0, 0, OFF_OBS + 11] = 112700.0
    new = tensor()                                  # clamped back to the ceiling
    new[0, 0, 0, OFF_OBS + 11] = 10.0 * 1609.34
    assert run(monkeypatch, old, new) == 0


# --- the real risk of the report-drop fix ------------------------------------
def test_a_large_coverage_loss_fails(monkeypatch):
    """Dropping whole METAR reports spends coverage to buy honesty. A fix that quietly
    halved obs coverage would be a worse bug than the one it repaired, and the audit
    alone would not catch it -- its obs floor is 85%, so a fall from 100% to 86% passes."""
    old = tensor()
    new = tensor()
    new[:, :, :, OFF_OBS_MASK + 9] = 0.0            # temperature coverage wiped
    assert run(monkeypatch, old, new) == 1


def test_a_small_coverage_loss_is_tolerated(monkeypatch):
    """The measured drop rate is ~0.02% of reports, so the gate must not fire on the
    fix's own intended effect."""
    old = tensor()
    new = tensor()
    new[0, 6, 0, OFF_OBS_MASK:OFF_LEAD] = 0.0       # one (game, d, h) cell of 196
    assert run(monkeypatch, old, new) == 0


def test_the_coverage_threshold_is_actually_enforced(monkeypatch):
    """Same artifacts, stricter threshold -> must flip to failing. Proves the tolerance
    is read rather than hardcoded past."""
    old = tensor()
    new = tensor()
    new[0, :, :, OFF_OBS_MASK + 9] = 0.0            # one game of four = 25% of that dim
    assert run(monkeypatch, old, new) == 1
    assert run(monkeypatch, old, new, ("--max-coverage-drop", "0.5")) == 0


def test_coverage_gain_is_not_reported_as_loss(monkeypatch):
    """A fix may legitimately add coverage; only a drop is a failure."""
    old = tensor()
    old[:, :, :, OFF_OBS_MASK + 9] = 0.0
    new = tensor()
    assert run(monkeypatch, old, new) == 0


def test_mismatched_game_sets_fail(monkeypatch):
    """Comparing tensors whose rows are different games makes every per-entry number
    meaningless, so this has to stop before any of them are printed."""
    old, new = tensor(), tensor()
    monkeypatch.setattr(d, "load",
                        lambda key: (old, PKS) if key == "OLD" else (new, PKS + 1))
    monkeypatch.setattr(sys, "argv",
                        ["diff", "--season", "2015", "--old", "OLD", "--new", "NEW"])
    with pytest.raises(SystemExit) as e:
        d.main()
    assert e.value.code == 1


def test_value_changes_are_reported_without_failing(monkeypatch, capsys):
    """A changed value in a dim nobody touched is the signal a fix reached too far. It is
    informational -- worth printing, not worth failing, since legitimate rebuilds move
    values when the upstream archive gains data."""
    old = tensor()
    new = tensor()
    new[..., OFF_FCST + 9] = 71.0
    assert run(monkeypatch, old, new) == 0
    out = capsys.readouterr().out
    assert "temperature_f" in out and "changed" in out
