"""The as-of weather toggle that makes the A/B a controlled experiment.

`append_weather_asof_to_prepared.py` mutates the prepared tensors IN PLACE: it writes
weather_asof.npy into each split dir and patches manifest.json with
has_weather_asof=True. Geometry is then inferred from that manifest, so once the append
runs there is no way to reproduce the control on the same directory — every later run
sees as-of weather. That makes the locked control (phase-1 best_val 4.95209) and the
treatment differ in TWO ways at once: the weather channels, and whichever prepared
directory each was read from.

The toggle removes that confound. Control and treatment read the identical prepared
tensors and differ by one flag, which is also what the weather ablation needs.
"""

import pytest

from mlb_dl.train_unified import _resolve_weather_geometry
from mlb_dl.weather_asof import ASOF_CHANNELS, N_TARGET_HOURS


class _Prepared:
    def __init__(self, has_asof: bool):
        self.manifest = {"has_weather_asof": has_asof, "rating_dim": 59}


class _Cached:
    """The from-frames/cached path attaches a per-game dict instead of a manifest."""
    manifest: dict = {}

    def __init__(self, by_pk):
        self._weather_asof_by_pk = by_pk


def test_appended_tensors_activate_asof_by_default():
    cfg, active = _resolve_weather_geometry(_Prepared(True), use_prepared=True)
    assert active
    assert (cfg.weather_tokens, cfg.weather_dim) == (N_TARGET_HOURS, ASOF_CHANNELS)


def test_toggle_forces_legacy_geometry_on_appended_tensors():
    """The control run: same directory, same appended npy, weather switched off."""
    legacy, _ = _resolve_weather_geometry(_Prepared(False), use_prepared=True)
    cfg, active = _resolve_weather_geometry(_Prepared(True), use_prepared=True,
                                            disable_asof=True)
    assert not active, "the flag must override the manifest, not defer to it"
    assert (cfg.weather_tokens, cfg.weather_dim) == (legacy.weather_tokens,
                                                     legacy.weather_dim), (
        "disabling as-of weather must reproduce the legacy geometry exactly, or the "
        "control checkpoint's architecture no longer matches the control run")


def test_toggle_also_overrides_the_cached_dataset_path():
    """Detection spans three data paths; a flag that only covers the prepared manifest
    would silently leave as-of weather on for the cached path."""
    _, active = _resolve_weather_geometry(_Cached({1: object()}), use_prepared=False,
                                          disable_asof=True)
    assert not active


def test_toggle_is_a_no_op_when_there_is_no_asof_data():
    """Passing the flag against legacy tensors must not change anything, so the same
    command line is safe to use before and after the append."""
    off, active_off = _resolve_weather_geometry(_Prepared(False), use_prepared=True,
                                                disable_asof=True)
    on, active_on = _resolve_weather_geometry(_Prepared(False), use_prepared=True)
    assert not active_off and not active_on
    assert (off.weather_tokens, off.weather_dim) == (on.weather_tokens, on.weather_dim)


def test_geometry_follows_what_the_dataset_will_actually_serve():
    """PreparedDataset resolves the flag into _has_weather_asof at construction, and the
    model is built from the geometry. If the geometry consulted the raw manifest instead,
    a dataset constructed with disable_asof would feed the legacy 4x22 tensor to a model
    built for 7x99 — the flag has to be passed to both calls or to neither, and this
    removes that requirement by making the dataset the single source of truth."""
    ds = _Prepared(True)
    ds._has_weather_asof = False           # what PreparedDataset(disable_asof=True) yields
    _, active = _resolve_weather_geometry(ds, use_prepared=True)   # flag NOT repeated here
    assert not active, "geometry trusted the manifest over the dataset's actual behaviour"


def test_prepared_dataset_flag_survives_a_manifest_claiming_asof(tmp_path):
    """The dataset half of the same invariant, exercised through the real constructor:
    with disable_asof the appended npy must not even be opened, so a control run works on
    a directory where only manifest.json advertises as-of weather."""
    from mlb_dl.precollate import PreparedDataset

    split = tmp_path / "train"
    split.mkdir()
    (split / "manifest.json").write_text(
        '{"has_weather_asof": true, "n_samples": 1, "n_games": 1, "rating_dim": 59}')
    # No .npy files exist at all. With the flag honoured, construction still has to fail
    # on some OTHER array -- but never on weather_asof.npy, which is the one the flag
    # promises not to touch.
    with pytest.raises(FileNotFoundError) as e:
        PreparedDataset(split, disable_asof=True)
    assert "weather_asof.npy" not in str(e.value), (
        "disable_asof still tried to load the as-of array")


@pytest.mark.parametrize("cmd", ["fit-unified", "evaluate"])
def test_both_commands_expose_the_flag(cmd):
    """fit-unified and evaluate must agree on geometry — a checkpoint trained with the
    flag cannot be scored without it, so the flag has to exist on both."""
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "mlb_dl.train_unified", cmd, "--help"],
                         capture_output=True, text=True)
    assert "--no-asof-weather" in out.stdout, f"{cmd} lacks the toggle:\n{out.stdout}"
