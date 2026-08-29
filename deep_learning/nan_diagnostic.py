"""NaN diagnostic: find exactly where NaN first appears in the forward pass.

Runs on EC2 with the actual feature store data. Checks:
1. Raw rating sequences for NaN (PRIME SUSPECT — no nan_to_num protection)
2. All dataset sample tensors for NaN/inf
3. Forward pass with per-layer hooks
4. Loss computation
5. Single backward pass with torch.autograd.set_detect_anomaly(True)
"""
import sys
import traceback

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


def main():
    print(SEPARATOR)
    print(" NaN DIAGNOSTIC — Finding exact source of NaN")
    print(SEPARATOR)

    # ===== Step 0: Load feature store =====
    print("\n[0/6] Loading feature store...")
    frames = _load_feature_store("artifacts/feature_store")
    train_end, val_end = temporal_split_dates(frames["game_targets"])
    print(f"  train_end={train_end}, val_end={val_end}")

    # ===== Step 1: Check rating sequences for NaN =====
    print(f"\n[1/6] Checking rating sequences for NaN...")
    rating_seqs = frames.get("rating_sequences")
    if rating_seqs and isinstance(rating_seqs, dict):
        total_entries = len(rating_seqs)
        nan_count = 0
        inf_count = 0
        nan_examples = []
        for key, arr in rating_seqs.items():
            n = np.isnan(arr).sum()
            i = np.isinf(arr).sum()
            if n > 0:
                nan_count += 1
                if len(nan_examples) < 5:
                    nan_examples.append((key, n, arr.size, arr.shape))
            if i > 0:
                inf_count += 1

        print(f"  Total entries: {total_entries}")
        print(f"  Entries with NaN: {nan_count} ({100*nan_count/max(total_entries,1):.1f}%)")
        print(f"  Entries with Inf: {inf_count}")
        if nan_examples:
            print(f"  *** RATING NaN CONFIRMED — Examples: ***")
            for key, n, total, shape in nan_examples:
                print(f"    {key}: NaN={n}/{total} shape={shape}")
            # Show which columns have NaN
            sample_key = nan_examples[0][0]
            sample_arr = rating_seqs[sample_key]
            nan_cols = np.where(np.isnan(sample_arr).any(axis=0))[0]
            print(f"    NaN columns (indices): {nan_cols.tolist()}")
    else:
        print("  No rating sequences loaded.")

    # ===== Step 2: Build standardizer and dataset =====
    print(f"\n[2/6] Building standardizer + dataset (train split)...")
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
    print(f"  Dataset: {len(train_ds)} samples, rating_dim={train_ds._rating_dim}")

    # ===== Step 3: Check individual samples for NaN =====
    print(f"\n[3/6] Checking 50 random samples for NaN inputs...")
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(train_ds), size=50)
    nan_report = {}

    def _check_tensor(key, t):
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            nc = torch.isnan(t).sum().item()
            ic = torch.isinf(t).sum().item()
            if nc > 0 or ic > 0:
                if key not in nan_report:
                    nan_report[key] = {"nan": 0, "inf": 0, "samples": []}
                nan_report[key]["nan"] += nc
                nan_report[key]["inf"] += ic
                nan_report[key]["samples"].append((int(idx), nc, ic, t.numel()))

    for idx in indices:
        sample = train_ds[int(idx)]
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                _check_tensor(k, v)
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    _check_tensor(f"{k}.{kk}", vv)

    if nan_report:
        print(f"  *** NaN/Inf FOUND IN {len(nan_report)} FIELDS: ***")
        for k, info in sorted(nan_report.items(), key=lambda x: -x[1]["nan"]):
            print(f"    {k}: total_NaN={info['nan']}, total_Inf={info['inf']}, "
                  f"affected_samples={len(info['samples'])}/50")
            if info["samples"]:
                ex = info["samples"][0]
                print(f"      e.g. sample {ex[0]}: {ex[1]} NaN, {ex[2]} Inf out of {ex[3]} elements")
    else:
        print("  All 50 samples clean — no NaN/Inf in inputs.")

    # ===== Step 4: Forward pass with hooks =====
    print(f"\n[4/6] Building batch and running forward pass with NaN hooks...")
    loader = DataLoader(train_ds, batch_size=8, shuffle=False,
                        collate_fn=game_transformer_collate_fn, num_workers=0)
    batch = next(iter(loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    batch = _to_device(batch, device)

    # Check batch tensors
    batch_nans = {}
    def _check_batch(d, prefix=""):
        for k, v in d.items():
            full_key = f"{prefix}{k}"
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                nc = torch.isnan(v).sum().item()
                ic = torch.isinf(v).sum().item()
                if nc > 0 or ic > 0:
                    batch_nans[full_key] = (nc, ic, v.numel(), v.shape)
            elif isinstance(v, dict):
                _check_batch(v, prefix=f"{k}.")

    _check_batch(batch)
    if batch_nans:
        print(f"  *** NaN/Inf IN BATCH ({len(batch_nans)} fields): ***")
        for k, (nc, ic, total, shape) in sorted(batch_nans.items()):
            print(f"    {k}: NaN={nc}, Inf={ic}, total={total}, shape={shape}")
    else:
        print("  Batch clean — no NaN/Inf.")

    # Build model
    rating_dim = train_ds._rating_dim
    context_config = ContextConfig()
    model = GameTransformer(
        d_model=128, rating_dim=rating_dim, flat_feature_dim=30,
        context_config=context_config, num_backbone_layers=2,
        num_heads=4, d_ff=512, dropout=0.1,
    ).to(device)
    model.train()  # Use train mode to match training behavior

    # Register hooks on every module
    nan_layers = []
    first_nan_layer = [None]  # mutable container

    def make_hook(name):
        def hook(module, input, output):
            if first_nan_layer[0] is not None:
                return  # Already found first NaN, skip rest
            outputs = output if isinstance(output, tuple) else (output,)
            for i, o in enumerate(outputs):
                if isinstance(o, torch.Tensor) and o.is_floating_point():
                    nc = torch.isnan(o).sum().item()
                    ic = torch.isinf(o).sum().item()
                    suffix = f"[{i}]" if isinstance(output, tuple) else ""
                    if nc > 0 or ic > 0:
                        entry = (f"{name}{suffix}", nc, ic, o.numel(), o.shape)
                        nan_layers.append(entry)
                        if first_nan_layer[0] is None:
                            first_nan_layer[0] = entry
                            # Also inspect inputs to this layer
                            inputs = input if isinstance(input, tuple) else (input,)
                            print(f"\n  *** FIRST NaN at: {name}{suffix} ***")
                            print(f"      Output: NaN={nc}, Inf={ic}, shape={o.shape}")
                            print(f"      Output stats (finite only): "
                                  f"min={o[o.isfinite()].min().item() if o.isfinite().any() else 'N/A'}, "
                                  f"max={o[o.isfinite()].max().item() if o.isfinite().any() else 'N/A'}")
                            for j, inp in enumerate(inputs):
                                if isinstance(inp, torch.Tensor) and inp.is_floating_point():
                                    inp_nan = torch.isnan(inp).sum().item()
                                    inp_inf = torch.isinf(inp).sum().item()
                                    print(f"      Input[{j}]: NaN={inp_nan}, Inf={inp_inf}, "
                                          f"shape={inp.shape}, "
                                          f"min={inp[inp.isfinite()].min().item() if inp.isfinite().any() else 'N/A'}, "
                                          f"max={inp[inp.isfinite()].max().item() if inp.isfinite().any() else 'N/A'}")
        return hook

    hooks = []
    for name, module in model.named_modules():
        if name:  # skip root module
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Forward pass
    with torch.no_grad():
        model_input = _prepare_model_input(batch, player_context_dim=256)

        # Check model_input for NaN
        print("\n  Model input NaN check:")
        input_nans = {}
        def _check_input(d, prefix=""):
            for k, v in d.items():
                full_key = f"{prefix}{k}"
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    nc = torch.isnan(v).sum().item()
                    ic = torch.isinf(v).sum().item()
                    if nc > 0 or ic > 0:
                        input_nans[full_key] = (nc, ic, v.numel(), v.shape)
                elif isinstance(v, dict):
                    _check_input(v, prefix=f"{k}.")

        _check_input(model_input)
        if input_nans:
            print(f"    *** NaN/Inf IN MODEL INPUT ({len(input_nans)} fields): ***")
            for k, (nc, ic, total, shape) in sorted(input_nans.items()):
                print(f"      {k}: NaN={nc}, Inf={ic}, total={total}, shape={shape}")
        else:
            print("    Model input clean.")

        try:
            predictions = model(model_input)
        except Exception as e:
            print(f"\n  *** FORWARD PASS FAILED: {e} ***")
            traceback.print_exc()
            for h in hooks:
                h.remove()
            return

    # Remove hooks
    for h in hooks:
        h.remove()

    if nan_layers:
        print(f"\n  NaN detected in {len(nan_layers)} layers total.")
        print(f"  First 10 NaN layers:")
        for name, nc, ic, total, shape in nan_layers[:10]:
            print(f"    {name}: NaN={nc}, Inf={ic}, total={total}, shape={shape}")
    else:
        print("\n  Forward pass clean — no NaN in any layer output.")

    # Check predictions
    print("\n  Prediction stats:")
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            nc = torch.isnan(v).sum().item()
            ic = torch.isinf(v).sum().item()
            flag = " *** NaN ***" if nc > 0 else (" *** Inf ***" if ic > 0 else "")
            finite = v[v.isfinite()]
            stats = f"min={finite.min().item():.4f}, max={finite.max().item():.4f}" if finite.numel() > 0 else "ALL NaN/Inf"
            print(f"    {k}: shape={v.shape}, NaN={nc}, Inf={ic}, {stats}{flag}")

    # ===== Step 5: Loss computation =====
    print(f"\n[5/6] Computing loss...")
    loss_fn = GameTransformerLoss().to(device)
    targets_with_mask = {**batch["targets"], "player_mask": batch.get("player_mask")}
    loss, task_losses = loss_fn(predictions, targets_with_mask, batch.get("live_inning"))
    print(f"  Total loss: {loss.item()}")
    for k, v in task_losses.items():
        val = v.item()
        flag = " *** NaN ***" if np.isnan(val) else (" *** Inf ***" if np.isinf(val) else "")
        print(f"    {k}: {val:.5f}{flag}")

    # ===== Step 6: Backward pass with anomaly detection =====
    if not np.isnan(loss.item()):
        print(f"\n[6/6] Loss is finite — running backward with anomaly detection...")
        # Re-run with gradients enabled
        model.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            model_input_grad = _prepare_model_input(batch, player_context_dim=256)
            predictions_grad = model(model_input_grad)
            targets_with_mask = {**batch["targets"], "player_mask": batch.get("player_mask")}
            loss_grad, _ = loss_fn(predictions_grad, targets_with_mask, batch.get("live_inning"))
            try:
                loss_grad.backward()
                print("  Backward pass clean — no anomaly detected.")
            except RuntimeError as e:
                print(f"  *** BACKWARD ANOMALY: {e} ***")
    else:
        print(f"\n[6/6] Loss is NaN — skipping backward (issue is in forward pass / loss).")

    # ===== Summary =====
    print(f"\n{SEPARATOR}")
    print(" DIAGNOSIS SUMMARY")
    print(SEPARATOR)

    if rating_seqs and nan_count > 0:
        print(f"  [ROOT CAUSE CANDIDATE] Rating sequences: {nan_count}/{total_entries} entries have NaN")
        print(f"  FIX: np.nan_to_num(arr, nan=0.0) on rating_sequences after loading")

    if nan_report:
        print(f"  [INPUT ISSUE] {len(nan_report)} sample fields have NaN/Inf")
        for k, info in sorted(nan_report.items(), key=lambda x: -x[1]["nan"])[:5]:
            print(f"    {k}: {info['nan']} total NaN across {len(info['samples'])}/50 samples")

    if first_nan_layer[0]:
        name = first_nan_layer[0][0]
        print(f"  [FIRST NaN LAYER] {name}")

    if not nan_report and not nan_layers and not np.isnan(loss.item()):
        print("  Model appears healthy on first batch.")
        print("  NaN may emerge later from gradient explosion — check grad norms.")

    print(SEPARATOR)


if __name__ == "__main__":
    main()
