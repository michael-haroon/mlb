"""`evaluate` must resolve the SAME weather geometry as `fit-unified`.

The control run's held-out metrics have to be regenerated from its checkpoint
rather than retrained, and the A/B treatment will be evaluated the same way.
If evaluate builds a legacy 4x22 weather context while the checkpoint was
trained with the as-of 7x99 geometry, `load_state_dict` fails on a shape
mismatch — or worse, would silently score a differently-shaped model if the
projection happened to align. Both commands therefore go through one resolver.
"""

import types

import pytest

from mlb_dl.train_unified import _resolve_weather_geometry
from mlb_dl.weather_asof import ASOF_CHANNELS, N_TARGET_HOURS


class _PreparedStub:
    def __init__(self, has_asof, rating_dim=7):
        self.manifest = {"has_weather_asof": has_asof, "rating_dim": rating_dim}


def test_prepared_manifest_flag_selects_asof_geometry():
    cfg, active = _resolve_weather_geometry(_PreparedStub(True), use_prepared=True)
    assert active
    assert cfg.weather_tokens == N_TARGET_HOURS
    assert cfg.weather_dim == ASOF_CHANNELS


def test_prepared_manifest_without_flag_keeps_legacy_geometry():
    cfg, active = _resolve_weather_geometry(_PreparedStub(False), use_prepared=True)
    assert not active
    assert cfg.weather_tokens == 4 and cfg.weather_dim == 22


def test_built_dataset_attribute_selects_asof_geometry():
    """The from-frames and cached paths signal via the dataset attribute, not a
    manifest — a dict that is present but empty must NOT count as active."""
    ds = types.SimpleNamespace(_weather_asof_by_pk={101: object()})
    cfg, active = _resolve_weather_geometry(ds, use_prepared=False)
    assert active and cfg.weather_dim == ASOF_CHANNELS

    empty = types.SimpleNamespace(_weather_asof_by_pk={})
    cfg2, active2 = _resolve_weather_geometry(empty, use_prepared=False)
    assert not active2 and cfg2.weather_dim == 22


def test_dataset_with_no_weather_signal_at_all():
    cfg, active = _resolve_weather_geometry(types.SimpleNamespace(), use_prepared=False)
    assert not active and cfg.weather_dim == 22


def test_evaluate_parser_accepts_prepared_dir():
    """Without --prepared-dir, regenerating held-out metrics means the
    EBS-bound feature-store rebuild that cost the baseline run hours."""
    import sys
    from unittest import mock
    from mlb_dl import train_unified

    argv = ["train_unified", "evaluate", "--feature-store", "/fs",
            "--checkpoint", "/ckpt.pt", "--output", "/out",
            "--prepared-dir", "/mnt/fast/prepared_tensors",
            "--d-model", "384", "--n-layers", "6", "--n-heads", "12"]
    captured = {}
    with mock.patch.object(sys, "argv", argv), \
         mock.patch.object(train_unified, "_cmd_evaluate",
                           side_effect=lambda a: captured.update(vars(a))):
        train_unified.main()
    assert captured["prepared_dir"] == "/mnt/fast/prepared_tensors"
    assert captured["d_model"] == 384 and captured["n_layers"] == 6
