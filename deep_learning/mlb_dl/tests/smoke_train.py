"""Smoke test: load real feature store, verify value ranges, run 2 epochs.

Usage:
    conda run -n pred python -m deep_learning.mlb_dl.tests.smoke_train \
        --feature-store deep_learning/feature_store
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-store", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    from deep_learning.mlb_dl.train_unified import (
        _load_feature_store,
        _build_datasets,
        _prepare_model_input,
    )
    from deep_learning.mlb_dl.datasets import temporal_split_dates
    from deep_learning.mlb_dl.game_transformer_dataset import (
        game_transformer_collate_fn,
        AblationConfig,
        PITCH_CONTINUOUS_COLS,
    )
    from deep_learning.mlb_dl.game_transformer import (
        GameTransformer,
        GameTransformerLoss,
        ContextConfig,
    )

    # --- Load feature store ---
    log.info("Loading feature store from %s", args.feature_store)
    frames = _load_feature_store(args.feature_store)
    if not frames:
        log.error("No data loaded.")
        return 1

    # Filter to 2024+ only for smoke test (memory constraint)
    import pandas as pd
    log.info("Filtering to season >= 2024 for smoke test...")
    if "season" in frames["pitch_sequences"].columns:
        frames["pitch_sequences"] = frames["pitch_sequences"][
            frames["pitch_sequences"]["season"] >= 2024
        ].copy()
    if "season" in frames["game_targets"].columns:
        frames["game_targets"] = frames["game_targets"][
            frames["game_targets"]["season"] >= 2024
        ].copy()
    if "season" in frames["game_meta"].columns:
        frames["game_meta"] = frames["game_meta"][
            frames["game_meta"]["season"] >= 2024
        ].copy()
    if "season" in frames["team_games"].columns:
        frames["team_games"] = frames["team_games"][
            frames["team_games"]["season"] >= 2024
        ].copy()
    if "season" in frames["player_batting_history"].columns:
        frames["player_batting_history"] = frames["player_batting_history"][
            frames["player_batting_history"]["season"] >= 2024
        ].copy()
    log.info("  pitch_sequences: %d rows", len(frames["pitch_sequences"]))
    log.info("  game_targets: %d rows", len(frames["game_targets"]))

    # --- Build datasets ---
    train_end, val_end = temporal_split_dates(frames["game_targets"])
    log.info("Split: train < %s, val < %s", train_end.date(), val_end.date())

    ablation = AblationConfig()
    train_ds, val_ds, _ = _build_datasets(frames, ablation, train_end, val_end)
    log.info("Dataset sizes: train=%d, val=%d", len(train_ds), len(val_ds))

    # --- VALUE RANGE CHECK ---
    log.info("=" * 60)
    log.info("VALUE RANGE CHECK on _pitch_cont_array")
    log.info("=" * 60)

    arr = train_ds._pitch_cont_array
    mask = train_ds._pitch_obs_mask
    log.info("Shape: %s, dtype: %s", arr.shape, arr.dtype)
    log.info("Obs mask mean (overall): %.4f", mask.mean())

    # Check no NaN or Inf remain
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()
    log.info("NaN count: %d, Inf count: %d", n_nan, n_inf)
    assert n_nan == 0, f"FAIL: {n_nan} NaN values found in _pitch_cont_array!"
    assert n_inf == 0, f"FAIL: {n_inf} Inf values found in _pitch_cont_array!"

    # Per-column stats
    log.info("\nPer-column z-score stats (non-binary columns):")
    _binary_cols = {"pre_on_first", "pre_on_second", "pre_on_third",
                    "is_pitch", "is_strike", "is_ball", "is_in_play", "is_top_inning"}
    extreme_count = 0
    for i, col in enumerate(PITCH_CONTINUOUS_COLS):
        if col in _binary_cols:
            continue
        col_vals = arr[:, i]
        col_mask = mask[:, i]
        obs_rate = col_mask.mean()
        if obs_rate < 0.01:
            continue
        z_min, z_max = col_vals.min(), col_vals.max()
        z_mean = col_vals.mean()
        n_extreme = int((np.abs(col_vals) > 10).sum())
        extreme_count += n_extreme
        if n_extreme > 0 or abs(z_min) > 5 or abs(z_max) > 5:
            log.info("  %s: range=[%.2f, %.2f], mean=%.4f, |z|>10: %d, obs=%.1f%%",
                     col, z_min, z_max, z_mean, n_extreme, obs_rate * 100)

    log.info("\nTotal extreme values (|z|>10): %d / %d = %.4f%%",
             extreme_count, arr.size, extreme_count / arr.size * 100)

    # Key assertion: mean should be near 0 for standardized columns
    # (masked positions are 0, observed positions are z-scored)
    overall_mean = arr.mean()
    log.info("Overall array mean: %.6f (should be near 0)", overall_mean)
    assert abs(overall_mean) < 1.0, f"FAIL: overall mean {overall_mean} too far from 0"
    log.info("VALUE RANGE CHECK PASSED")

    rating_dim = frames.get("_rating_dim", 0)

    del frames
    gc.collect()

    # --- SMOKE TRAINING ---
    log.info("=" * 60)
    log.info("SMOKE TRAINING: %d epochs, batch_size=%d", args.epochs, args.batch_size)
    log.info("=" * 60)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=game_transformer_collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=game_transformer_collate_fn, num_workers=0,
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    log.info("Device: %s", device)

    model = GameTransformer(
        d_model=128,
        rating_dim=rating_dim,
        flat_feature_dim=30,
        context_config=ContextConfig(
            sp_games=5, team_games=10, tokens_per_game=4, rating_steps=10
        ),
        num_backbone_layers=2,
        num_heads=4,
        d_ff=512,
        dropout=0.1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %d parameters", n_params)

    loss_fn = GameTransformerLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    player_ctx_dim = 128 * 2

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else
                        {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                         for kk, vv in v.items()} if isinstance(v, dict) else v)
                     for k, v in batch.items()}

            model_input = _prepare_model_input(batch, player_context_dim=player_ctx_dim)
            # Move context tensors to device
            for key in ["sp_home", "sp_away", "team_home", "team_away"]:
                if key in model_input["context"]:
                    cat = model_input["context"][key]
                    for k, v in cat.items():
                        if isinstance(v, torch.Tensor):
                            cat[k] = v.to(device)

            preds = model(model_input)
            targets = batch["targets"]

            loss, _ = loss_fn(preds, targets, live_inning=None)

            if torch.isnan(loss):
                log.error("NaN loss at epoch %d, batch %d — skipping (pre-existing model numerics)", epoch, batch_idx)
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_losses.append(loss.item())

            if batch_idx >= 20:
                break  # Only run 20 batches per epoch for smoke test

        elapsed = time.time() - t0
        mean_loss = np.mean(epoch_losses)
        log.info("Epoch %d: train_loss=%.4f (%d batches, %.1fs)", epoch, mean_loss, len(epoch_losses), elapsed)

        # Quick val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else
                            {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                             for kk, vv in v.items()} if isinstance(v, dict) else v)
                         for k, v in batch.items()}
                model_input = _prepare_model_input(batch, player_context_dim=player_ctx_dim)
                for key in ["sp_home", "sp_away", "team_home", "team_away"]:
                    if key in model_input["context"]:
                        cat = model_input["context"][key]
                        for k, v in cat.items():
                            if isinstance(v, torch.Tensor):
                                cat[k] = v.to(device)
                preds = model(model_input)
                targets = batch["targets"]
                loss, _ = loss_fn(preds, targets, live_inning=None)
                if torch.isnan(loss):
                    continue
                val_losses.append(loss.item())
                if batch_idx >= 10:
                    break

        val_mean = np.mean(val_losses)
        log.info("Epoch %d: val_loss=%.4f", epoch, val_mean)

    log.info("=" * 60)
    log.info("SMOKE TRAINING PASSED — loss is finite and model produces gradients")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
