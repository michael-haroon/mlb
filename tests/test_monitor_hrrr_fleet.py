"""The fleet monitor's alarms must actually fire.

This monitor is exception-only: silence means healthy. That design has one dangerous
failure mode -- a monitor that is silent because it is BROKEN is indistinguishable
from one that is silent because the fleet is fine. So every alarm gets a test that
proves it fires, and the healthy path gets a test that proves it stays quiet.

Two defects these tests pin, both found by running the alarms rather than reading them:

  1. STALLED could never fire. `newest_epoch` parsed S3's UTC LastModified with BSD
     `date -j`, which interprets a naive timestamp in the LOCAL zone. On a PDT laptop
     every object read as 7 hours in the future, so the computed age was about -420
     minutes and never cleared any positive threshold.
  2. STALLED was reported from inside the per-box loop, guarded by a hardcoded box
     name, so it was silently disabled for any fleet not containing that box.

The remote call is stubbed via $SSH, so these tests touch no EC2 instance. They do
issue one S3 list per pass to age the archive; that is a metadata-only call.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parent.parent / "data_curation" / "scripts"
          / "monitor_hrrr_fleet.sh")


def _stub(tmp_path: Path, name: str, procs: int, withheld: int = 0, rfail: int = 0) -> Path:
    """A fake $SSH that reports a fixed box state instead of contacting a box."""
    p = tmp_path / name
    p.write_text(
        "#!/bin/bash\n"
        f'echo "procs={procs}"\n'
        f'echo "withheld={withheld}"\n'
        f'echo "repairfail={rfail}"\n'
    )
    p.chmod(0o755)
    return p


def _run(tmp_path: Path, stub: Path, state: Path, **env):
    e = dict(os.environ, SSH=str(stub), STATE=str(state), ONESHOT="1",
             BOXES=env.pop("BOXES", "X:203.0.113.1"))
    e.update({k: str(v) for k, v in env.items()})
    r = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True,
                       env=e, timeout=180)
    return r.returncode, r.stdout


def test_script_is_syntactically_valid():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_healthy_fleet_is_silent(tmp_path):
    """The core contract: a working fleet produces NO output at all."""
    rc, out = _run(tmp_path, _stub(tmp_path, "busy", procs=5), tmp_path / "s",
                   STALL_MIN=25)
    assert out.strip() == "", f"healthy fleet emitted output: {out!r}"
    assert rc == 0


def test_stall_fires_when_archive_is_stale(tmp_path):
    """Pins defect 1: with a zero threshold the alarm must trip.

    Under the timezone bug the measured age was negative, so this stayed silent no
    matter how long the fleet had actually been stuck.
    """
    rc, out = _run(tmp_path, _stub(tmp_path, "busy", procs=5), tmp_path / "s",
                   STALL_MIN=0)
    assert "STALLED" in out, f"stall alarm did not fire: {out!r}"
    assert rc == 1


def test_stall_age_is_not_negative(tmp_path):
    """Directly guards the timezone handling: a negative age is nonsense and is the
    signature of parsing a UTC stamp as local time."""
    _, out = _run(tmp_path, _stub(tmp_path, "busy", procs=5), tmp_path / "s",
                  STALL_MIN=0)
    assert "-" not in out.split("object for ")[1], f"negative archive age: {out!r}"


def test_stall_fires_for_a_fleet_with_no_box_named_a(tmp_path):
    """Pins defect 2: the alarm was keyed to a hardcoded box name."""
    _, out = _run(tmp_path, _stub(tmp_path, "busy", procs=5), tmp_path / "s",
                  STALL_MIN=0, BOXES="X:203.0.113.1 Y:203.0.113.2")
    assert "STALLED" in out
    assert "2 box(es)" in out


def test_withheld_and_repairfail_fire(tmp_path):
    stub = _stub(tmp_path, "bad", procs=5, withheld=3, rfail=1)
    _, out = _run(tmp_path, stub, tmp_path / "s", STALL_MIN=25)
    assert "WITHHELD" in out and "REPAIRFAIL" in out


def test_withheld_is_reported_only_when_it_grows(tmp_path):
    """A withheld date stays in the log forever. Re-reporting it every interval would
    turn the exception-only monitor into a noise source and train the reader to ignore
    it, which is the whole failure this design exists to avoid."""
    stub = _stub(tmp_path, "bad", procs=5, withheld=3)
    state = tmp_path / "s"
    _, first = _run(tmp_path, stub, state, STALL_MIN=25)
    assert "WITHHELD" in first
    rc, second = _run(tmp_path, stub, state, STALL_MIN=25)
    assert second.strip() == "", f"unchanged withheld count re-reported: {second!r}"
    assert rc == 0


def test_unreachable_needs_two_consecutive_misses(tmp_path):
    """One ssh failure is a normal network blip; alarming on it would be noise."""
    p = tmp_path / "dead"
    p.write_text("#!/bin/bash\nexit 255\n")
    p.chmod(0o755)
    state = tmp_path / "s"
    _, first = _run(tmp_path, p, state, STALL_MIN=25)
    assert "UNREACHABLE" not in first, "alarmed on a single ssh failure"
    _, second = _run(tmp_path, p, state, STALL_MIN=25)
    assert "UNREACHABLE" in second and "x2" in second


def test_recovery_resets_the_miss_counter(tmp_path):
    """A blip, a recovery, then another blip must not reach the x2 threshold."""
    dead = tmp_path / "dead"
    dead.write_text("#!/bin/bash\nexit 255\n")
    dead.chmod(0o755)
    state = tmp_path / "s"
    _run(tmp_path, dead, state, STALL_MIN=25)                      # miss 1
    _run(tmp_path, _stub(tmp_path, "busy", procs=5), state, STALL_MIN=25)  # recover
    _, out = _run(tmp_path, dead, state, STALL_MIN=25)             # miss 1 again
    assert "UNREACHABLE" not in out, f"counter not reset on recovery: {out!r}"


def test_all_idle_reports_done_and_defers_to_the_coverage_gate(tmp_path):
    """DONE must not claim success -- a dead shard and a finished one look identical
    from process state, so the monitor hands off to the population-based gate."""
    rc, out = _run(tmp_path, _stub(tmp_path, "idle", procs=0), tmp_path / "s",
                   STALL_MIN=0, BOXES="X:203.0.113.1 Y:203.0.113.2")
    assert "DONE" in out and "coverage gate" in out
    assert "STALLED" not in out, "an idle fleet is not stalled"
    assert rc == 0


def test_documented_alarms_all_exist_in_the_code(tmp_path):
    """The header is the operator's contract. A documented alarm that was never
    implemented is worse than an undocumented one -- the reader waits for a line that
    can never arrive."""
    src = SCRIPT.read_text()
    header = src.split("set -u")[0]
    documented = {w for w in ("WITHHELD", "REPAIRFAIL", "UNREACHABLE", "STALLED", "DONE")
                  if w in header}
    body = src.split("set -u", 1)[1]
    for alarm in documented:
        assert f'say "{alarm}' in body or f"say \"{alarm}" in body or alarm in body, (
            f"{alarm} is documented in the header but never emitted"
        )
    assert "DEAD" not in header, (
        "the DEAD alarm is documented but not implemented; the idle case defers to "
        "the coverage gate instead"
    )
