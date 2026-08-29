"""Pinpoint exact NaN source in backward pass using anomaly detection.

Previous findings:
- Forward pass: finite loss (~8.5) on ANY batch
- Backward pass: NaN gradients on FIRST batch (shuffled)
- First 8 sequential samples work fine
- Conclusion: certain samples trigger an operation whose forward is finite
  but whose gradient is undefined (log(0), sqrt(0), 0^fractional, etc.)

This script uses torch.autograd.set_detect_anomaly(True) which throws a
RuntimeError identifying the exact operation + traceback.
"""
import sys
import gc
import traceback
import numpy as np
import pandas as pd
import torch

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


def main():
    print(SEPARATOR)
    print(" BACKWARD PASS NaN — ANOMALY DETECTION")
    print(SEPARATOR)

    print("\n[1/3] Loading feature store + building dataset...")
    frames = _load_feature_store("artifacts/feature_store")
    train_end, val_end = temporal_split_dates(frames["game_targets"])

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
    del frames, pitch_df
    gc.collect()

    device = torch.device("cuda")
    rating_dim = train_ds._rating_dim

    # Use batch_size=32 with shuffle to reproduce the NaN
    loader = DataLoader(
        train_ds, batch_size=32, shuffle=True,
        collate_fn=game_transformer_collate_fn, num_workers=0,
    )

    # Build model
    context_config = ContextConfig()
    model = GameTransformer(
        d_model=128, rating_dim=rating_dim, flat_feature_dim=30,
        context_config=context_config, num_backbone_layers=2,
        num_heads=4, d_ff=512, dropout=0.1,
    ).to(device)
    model.train()
    loss_fn = GameTransformerLoss().to(device)

    print("\n[2/3] Running forward+backward with anomaly detection (FP32, no AMP)...")
    print("  Using shuffled batch_size=32 to reproduce NaN condition...")

    batch = next(iter(loader))
    batch = _to_device(batch, device)

    torch.autograd.set_detect_anomaly(True)

    try:
        model_input = _prepare_model_input(batch, player_context_dim=256)
        predictions = model(model_input)
        targets_with_mask = {**batch["targets"], "player_mask": batch.get("player_mask")}
        loss, task_losses = loss_fn(predictions, targets_with_mask, batch.get("live_inning"))

        print(f"  Forward pass OK. Loss = {loss.item():.4f}")
        print(f"  Task losses:")
        for k, v in task_losses.items():
            print(f"    {k}: {v.item():.5f}")

        loss.backward()
        print("  Backward pass OK — no NaN detected on this batch.")

        # Check grad norm
        total_norm = 0.0
        nan_params = []
        for name, p in model.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    nan_params.append(name)
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        print(f"  Grad norm: {total_norm:.4f}")
        if nan_params:
            print(f"  *** NaN in gradients of: {nan_params[:10]} ***")

    except RuntimeError as e:
        print(f"\n  *** ANOMALY DETECTED ***")
        print(f"  Error: {e}")
        print(f"\n  Full traceback:")
        traceback.print_exc()

    # If the first batch didn't trigger it, try a few more
    print("\n[3/3] Testing 10 more batches to find one that triggers NaN...")
    torch.autograd.set_detect_anomaly(False)  # Disable for speed

    model2 = GameTransformer(
        d_model=128, rating_dim=rating_dim, flat_feature_dim=30,
        context_config=context_config, num_backbone_layers=2,
        num_heads=4, d_ff=512, dropout=0.1,
    ).to(device)
    model2.train()

    loader2 = DataLoader(
        train_ds, batch_size=32, shuffle=True,
        collate_fn=game_transformer_collate_fn, num_workers=0,
    )

    for i, batch in enumerate(loader2):
        if i >= 10:
            break
        batch = _to_device(batch, device)
        model2.zero_grad()

        model_input = _prepare_model_input(batch, player_context_dim=256)
        predictions = model2(model_input)
        targets_with_mask = {**batch["targets"], "player_mask": batch.get("player_mask")}
        loss, _ = loss_fn(predictions, targets_with_mask, batch.get("live_inning"))
        loss.backward()

        # Check for NaN grads
        has_nan = any(torch.isnan(p.grad).any().item() for p in model2.parameters() if p.grad is not None)
        grad_norm = sum(p.grad.norm(2).item()**2 for p in model2.parameters() if p.grad is not None) ** 0.5
        print(f"  Batch {i}: loss={loss.item():.4f}, grad_norm={grad_norm:.2f}, nan_grad={has_nan}")

        if has_nan:
            print(f"\n  Found NaN batch! Re-running with anomaly detection...")
            torch.autograd.set_detect_anomaly(True)
            model2.zero_grad()
            try:
                predictions = model2(model_input)
                targets_with_mask = {**batch["targets"], "player_mask": batch.get("player_mask")}
                loss, task_losses = loss_fn(predictions, targets_with_mask, batch.get("live_inning"))
                print(f"  Loss = {loss.item():.4f}")
                print(f"  Task losses: {', '.join(f'{k}={v.item():.4f}' for k,v in task_losses.items())}")
                loss.backward()
                print("  (anomaly detection didn't catch it this time)")
            except RuntimeError as e:
                print(f"\n  *** ANOMALY DETECTED ***")
                print(f"  Error: {e}")
                traceback.print_exc()
            break

    print(f"\n{SEPARATOR}")
    print(" DONE")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
