"""Phase 4 wiring tests: artifact loader standardization, prepared-path
slicing, and the model forward with the 7x99 weather geometry."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from mlb_dl.weather_asof import (
    ASOF_CHANNELS,
    N_DIMS,
    N_OBS_DIMS,
    N_DECISIONS,
    N_TARGET_HOURS,
    OFF_FCST,
    OFF_FCST_MASK,
    OFF_OBS,
    OFF_OBS_MASK,
    OFF_LEAD,
)
from mlb_dl.train_unified import _load_weather_asof_artifacts

CHANNEL_COLS = [f"wx_c{i:02d}" for i in range(ASOF_CHANNELS)]


def _synthetic_artifact(tmp_path, populate_masks=True):
    """One game's raw artifact + norm sidecar in a fake feature-store dir."""
    rng = np.random.default_rng(0)
    n_rows = N_DECISIONS * N_TARGET_HOURS
    arr = np.zeros((n_rows, ASOF_CHANNELS), dtype=np.float32)
    arr[:, OFF_FCST:OFF_FCST + N_DIMS] = rng.normal(10.0, 3.0, (n_rows, N_DIMS))
    arr[:, OFF_OBS:OFF_OBS + N_OBS_DIMS] = rng.normal(5.0, 2.0, (n_rows, N_OBS_DIMS))
    if populate_masks:
        arr[:, OFF_FCST_MASK:OFF_FCST_MASK + N_DIMS] = 1.0
        arr[:, OFF_OBS_MASK:OFF_OBS_MASK + N_OBS_DIMS] = 1.0
    else:
        # masked entries carry raw 0 values (assemble writes vec*mask)
        arr[:, OFF_FCST:OFF_FCST + N_DIMS] = 0.0
        arr[:, OFF_OBS:OFF_OBS + N_OBS_DIMS] = 0.0
    df = pd.DataFrame(arr, columns=CHANNEL_COLS)
    df.insert(0, "target_hour", np.tile(np.arange(-1, 6), N_DECISIONS))
    df.insert(0, "decision_hour", np.repeat(np.arange(7), N_TARGET_HOURS))
    df.insert(0, "game_pk", 777001)

    (tmp_path / "weather_asof").mkdir()
    df.to_parquet(tmp_path / "weather_asof" / "season=2023.parquet", index=False)
    stats = {
        "fcst_mean": [10.0] * N_DIMS, "fcst_std": [3.0] * N_DIMS,
        "obs_mean": [5.0] * N_OBS_DIMS, "obs_std": [2.0] * N_OBS_DIMS,
    }
    (tmp_path / "weather_asof_norm.json").write_text(json.dumps(stats))

    off = pd.DataFrame({"game_pk": [777001] * 4, "sequence_index": [0, 1, 2, 3],
                        "wx_hour_offset": np.array([0, 0, 1, 2], dtype=np.int8)})
    (tmp_path / "wx_hour_offset").mkdir()
    off.to_parquet(tmp_path / "wx_hour_offset" / "season=2023.parquet", index=False)
    return tmp_path


def test_loader_standardizes_and_reshapes(tmp_path):
    fs = _synthetic_artifact(tmp_path)
    asof, offsets = _load_weather_asof_artifacts(fs)
    T = asof[777001]
    assert T.shape == (N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS)
    # standardized populated values: mean~0 std~1 in each block
    f = T[:, :, OFF_FCST:OFF_FCST + N_DIMS]
    assert abs(f.mean()) < 0.15 and 0.8 < f.std() < 1.2
    o = T[:, :, OFF_OBS:OFF_OBS + N_OBS_DIMS]
    assert abs(o.mean()) < 0.15 and 0.8 < o.std() < 1.2
    # masks untouched
    assert (T[:, :, OFF_FCST_MASK:OFF_FCST_MASK + N_DIMS] == 1.0).all()
    assert offsets[777001].tolist() == [0, 0, 1, 2]


def test_loader_keeps_masked_entries_exact_zero(tmp_path):
    fs = _synthetic_artifact(tmp_path, populate_masks=False)
    asof, _ = _load_weather_asof_artifacts(fs)
    T = asof[777001]
    # (0 - mean)/std would be nonzero; the mask multiply must pin it to 0
    assert not T[:, :, OFF_FCST:OFF_FCST + N_DIMS].any()
    assert not T[:, :, OFF_OBS:OFF_OBS + N_OBS_DIMS].any()


def test_loader_absent_dir_returns_empty(tmp_path):
    asof, offsets = _load_weather_asof_artifacts(tmp_path)
    assert asof == {} and offsets == {}


def test_model_forward_with_asof_geometry():
    """GameTransformer must accept weather_temporal [B, 7, 99] when configured
    with the as-of geometry — the exact wiring fit-unified enables."""
    from mlb_dl.game_transformer import ContextConfig, GameTransformer

    cfg = ContextConfig()
    cfg.weather_tokens = N_TARGET_HOURS
    cfg.weather_dim = ASOF_CHANNELS
    model = GameTransformer(d_model=64, rating_dim=0, flat_feature_dim=30,
                            context_config=cfg, num_backbone_layers=1,
                            num_heads=4, d_ff=128, dropout=0.0)
    B = 2
    wx = torch.randn(B, N_TARGET_HOURS, ASOF_CHANNELS)
    compiler = model.context_compiler
    tokens = compiler.weather_proj(wx)
    assert tokens.shape == (B, N_TARGET_HOURS, 64)
    hour_offsets = torch.arange(wx.size(1)).unsqueeze(0).expand(B, -1)
    embedded = tokens + compiler.weather_hour_embed(hour_offsets)
    assert embedded.shape == (B, N_TARGET_HOURS, 64)
    assert torch.isfinite(embedded).all()


def test_legacy_geometry_still_default():
    from mlb_dl.game_transformer import ContextConfig

    cfg = ContextConfig()
    assert cfg.weather_tokens == 4 and cfg.weather_dim == 22
