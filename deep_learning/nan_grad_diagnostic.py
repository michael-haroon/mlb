"""Gradient explosion diagnostic: simulate training for N batches and track grad norms.

The single-batch diagnostic passed cleanly — NaN emerges from accumulated gradient
explosion over many training steps. This script:
1. Trains for up to 200 batches with the exact same setup as ec2_train_diagnostic.sh
2. Logs per-batch: loss, grad_norm (before clip), grad_norm (after clip), max weight
3. Stops as soon as NaN is detected and reports the last 10 batches before failure
4. Tests with different LR/grad_clip to identify safe training parameters
"""
import sys
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from mlb_dl.train_unified import _load_feature_store, _to_device, _prepare_model_input
from mlb_dl.datasets import Standardizer, temporal_split_dates
from mlb_dl.game_transformer import GameTransformer, GameTransformerLoss, ContextConfig
from mlb_dl.game_transformer_dataset import (
    AblationConfig, GameTransformerDataset, game_transformer_collate_fn,
    PITCH_CONTINUOUS_COLS,
)
from torch.utils.data import DataLoader

SEPARATOR = "=" * 70


def run_training_test(model, loss_fn, loader, device, lr, grad_clip, use_amp, max_batches, label):
    """Train for max_batches steps, logging grad norms. Returns history."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history = []
    nan_batch = None

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            model_input = _prepare_model_input(batch, player_context_dim=256)
            predictions = model(model_input)
            targets_with_mask = {**batch["targets"], "player_mask": batch.get("player_mask")}
            loss, task_losses = loss_fn(predictions, targets_with_mask, batch.get("live_inning"))

        loss_val = loss.item()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        # Measure grad norm BEFORE clipping
        grad_norm_pre = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float('inf')
        ).item()

        # Now actually clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        # Measure grad norm AFTER clipping
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norm_post = total_norm ** 0.5

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
            scale_val = scaler.get_scale()
        else:
            optimizer.step()
            scale_val = 1.0

        # Max absolute weight
        max_weight = max(p.abs().max().item() for p in model.parameters())

        # Check for NaN
        has_nan_loss = np.isnan(loss_val)
        has_nan_grad = np.isnan(grad_norm_pre)
        has_nan_weight = any(torch.isnan(p).any().item() for p in model.parameters())

        entry = {
            "batch": i,
            "loss": loss_val,
            "grad_norm_pre_clip": grad_norm_pre,
            "grad_norm_post_clip": grad_norm_post,
            "max_weight": max_weight,
            "amp_scale": scale_val,
            "nan_loss": has_nan_loss,
            "nan_grad": has_nan_grad,
            "nan_weight": has_nan_weight,
        }
        history.append(entry)

        if has_nan_loss or has_nan_grad or has_nan_weight:
            nan_batch = i
            break

        if i % 20 == 0:
            print(f"  [{label}] batch {i}: loss={loss_val:.4f}, "
                  f"grad_pre={grad_norm_pre:.2f}, grad_post={grad_norm_post:.2f}, "
                  f"max_w={max_weight:.4f}, scale={scale_val:.0f}")

    return history, nan_batch


def main():
    print(SEPARATOR)
    print(" GRADIENT EXPLOSION DIAGNOSTIC")
    print(SEPARATOR)

    # Load feature store
    print("\n[1/4] Loading feature store...")
    frames = _load_feature_store("artifacts/feature_store")
    train_end, val_end = temporal_split_dates(frames["game_targets"])

    # Build dataset
    print("[2/4] Building dataset...")
    pitch_df = frames["pitch_sequences"]
    train_pitches = pitch_df[pd.to_datetime(pitch_df["game_date"], errors="coerce") < train_end]
    available_cols = [c for c in PITCH_CONTINUOUS_COLS if c in train_pitches.columns]
    standardizer = Standardizer.fit(train_pitches, available_cols)
    del train_pitches

    new_feature_kwargs = {
        "weather_features": frames.get("weather_features"),
        "weather_temporal": frames.get("weather_temporal"),
        "venue_dimensions": frames.get("venue_dimensions"),
        "daily_stats": frames.get("daily_stats"),
        "game_features": frames.get("rating_sequences"),
    }

    ablation = AblationConfig()
    train_ds = GameTransformerDataset(
        pitch_sequences=frames["pitch_sequences"],
        game_targets=frames["game_targets"],
        game_meta=frames["game_meta"],
        team_games=frames["team_games"],
        player_batting_history=frames["player_batting_history"],
        standardizer=standardizer,
        ablation=ablation,
        split_end=train_end,
        **new_feature_kwargs,
    )
    print(f"  Dataset: {len(train_ds)} samples")

    # Free memory
    del frames, pitch_df
    gc.collect()

    device = torch.device("cuda")
    rating_dim = train_ds._rating_dim

    # Test configurations
    configs = [
        {"lr": 1.2e-3, "grad_clip": 5.0, "use_amp": True,  "batch_size": 128, "label": "CURRENT (LR=1.2e-3, clip=5, AMP)"},
        {"lr": 3e-4,   "grad_clip": 1.0, "use_amp": True,  "batch_size": 128, "label": "SAFE (LR=3e-4, clip=1, AMP)"},
        {"lr": 1e-4,   "grad_clip": 1.0, "use_amp": False, "batch_size": 32,  "label": "CONSERVATIVE (LR=1e-4, clip=1, FP32, bs=32)"},
    ]

    results = {}

    for cfg in configs:
        print(f"\n[3/4] Testing: {cfg['label']}")
        print(f"  LR={cfg['lr']}, grad_clip={cfg['grad_clip']}, AMP={cfg['use_amp']}, batch_size={cfg['batch_size']}")

        # Fresh model for each test
        context_config = ContextConfig()
        model = GameTransformer(
            d_model=128, rating_dim=rating_dim, flat_feature_dim=30,
            context_config=context_config, num_backbone_layers=2,
            num_heads=4, d_ff=512, dropout=0.1,
        ).to(device)

        loss_fn = GameTransformerLoss().to(device)

        loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True,
            collate_fn=game_transformer_collate_fn, num_workers=0,
        )

        history, nan_batch = run_training_test(
            model, loss_fn, loader, device,
            lr=cfg["lr"],
            grad_clip=cfg["grad_clip"],
            use_amp=cfg["use_amp"],
            max_batches=200,
            label=cfg["label"][:20],
        )

        results[cfg["label"]] = {"history": history, "nan_batch": nan_batch}

        if nan_batch is not None:
            print(f"\n  *** NaN at batch {nan_batch} ***")
            # Show last 5 entries before NaN
            start = max(0, nan_batch - 5)
            for entry in history[start:]:
                print(f"    batch {entry['batch']}: loss={entry['loss']:.4f}, "
                      f"grad_pre={entry['grad_norm_pre_clip']:.2f}, "
                      f"max_w={entry['max_weight']:.4f}, "
                      f"scale={entry['amp_scale']:.0f}, "
                      f"nan_l={entry['nan_loss']}, nan_g={entry['nan_grad']}, nan_w={entry['nan_weight']}")
        else:
            # Show summary stats
            losses = [h["loss"] for h in history]
            grads = [h["grad_norm_pre_clip"] for h in history]
            print(f"\n  Completed 200 batches successfully!")
            print(f"  Loss: start={losses[0]:.4f}, end={losses[-1]:.4f}, min={min(losses):.4f}, max={max(losses):.4f}")
            print(f"  Grad norm (pre-clip): mean={np.mean(grads):.2f}, max={max(grads):.2f}, "
                  f"p99={np.percentile(grads, 99):.2f}")

        # Cleanup
        del model, loss_fn, loader
        torch.cuda.empty_cache()
        gc.collect()

    # Summary
    print(f"\n{SEPARATOR}")
    print(" SUMMARY")
    print(SEPARATOR)
    for label, res in results.items():
        if res["nan_batch"] is not None:
            print(f"  FAIL: {label} — NaN at batch {res['nan_batch']}")
        else:
            losses = [h["loss"] for h in res["history"]]
            grads = [h["grad_norm_pre_clip"] for h in res["history"]]
            print(f"  PASS: {label} — 200 batches, loss {losses[0]:.3f}->{losses[-1]:.3f}, max_grad={max(grads):.1f}")

    print(SEPARATOR)


if __name__ == "__main__":
    main()
