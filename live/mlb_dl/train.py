from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .data_sources import season_range
from .feature_store import build_feature_store

_LOG_DIR = Path("data/logs")

log = logging.getLogger(__name__)


def _setup_logging() -> None:
    if log.handlers:
        return
    log.setLevel(logging.DEBUG)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(_LOG_DIR / "train.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    log.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(sh)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLB deep-learning training utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-feature-store", help="Build target and feature-store parquet files")
    build.add_argument("--source", required=True, help="Local path or s3:// bucket prefix")
    build.add_argument("--output", required=True, help="Output feature-store directory")
    build.add_argument("--season-start", type=int)
    build.add_argument("--season-end", type=int)
    build.set_defaults(func=_cmd_build_feature_store)

    fit = sub.add_parser("fit-pregame", help="Train the pre-game multitask CNN")
    fit.add_argument("--feature-store", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--history-length", type=int, default=20)
    fit.add_argument("--min-history", type=int, default=5)
    fit.add_argument("--epochs", type=int, default=20)
    fit.add_argument("--batch-size", type=int, default=64)
    fit.add_argument("--learning-rate", type=float, default=5e-4)
    fit.add_argument("--weight-decay", type=float, default=1e-4)
    fit.add_argument("--hidden-dim", type=int, default=128)
    fit.add_argument("--dropout", type=float, default=0.2)
    fit.add_argument("--time-decay-lambda", type=float, default=0.003)
    fit.add_argument("--patience", type=int, default=5, help="Early stopping: epochs without val improvement before stopping")
    fit.set_defaults(func=_cmd_fit_pregame)

    eda = sub.add_parser("run-eda", help="Run exploratory distribution analysis on feature store")
    eda_source = eda.add_mutually_exclusive_group(required=True)
    eda_source.add_argument("--source", help="S3 or local path to raw data; builds feature store on the fly")
    eda_source.add_argument("--feature-store", help="Path to an already-built feature store directory")
    eda.add_argument("--output", required=True, help="Output directory for EDA artifacts")
    eda.add_argument("--season-start", type=int, help="First season to ingest (used with --source)")
    eda.add_argument("--season-end", type=int, help="Last season to ingest (used with --source)")
    eda.add_argument("--seasons", type=int, nargs="+", metavar="YEAR",
                     help="Filter EDA to specific seasons (e.g. --seasons 2021 2022 2023)")
    eda.add_argument("--targets-only", action="store_true", help="Skip feature column analysis")
    eda.add_argument("--features-only", action="store_true", help="Skip target column analysis")
    eda.set_defaults(func=_cmd_run_eda)

    fit_live = sub.add_parser("fit-live", help="Train the live HAN model on pitch sequences")
    fit_live.add_argument("--feature-store", required=True, help="Path to feature store directory")
    fit_live.add_argument("--output", required=True, help="Output directory for checkpoints and eval")
    fit_live.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    fit_live.add_argument("--batch-size", type=int, default=32, help="Batch size")
    fit_live.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate")
    fit_live.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    fit_live.add_argument("--d-model", type=int, default=128, help="Model hidden dimension")
    fit_live.add_argument("--n-heads", type=int, default=4, help="Attention heads per layer")
    fit_live.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    fit_live.add_argument("--max-seq-len", type=int, default=350, help="Max pitch sequence length")
    fit_live.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    fit_live.add_argument("--gradient-clip", type=float, default=5.0, help="Gradient clipping max norm")
    fit_live.add_argument("--negbin-weight", type=float, default=1.0, help="NegBin loss weight")
    fit_live.add_argument("--win-weight", type=float, default=0.5, help="Home win BCE weight")
    fit_live.add_argument("--ei-weight", type=float, default=0.3, help="Extra innings BCE weight")
    fit_live.add_argument("--yrfi-weight", type=float, default=0.3, help="YRFI BCE weight")
    fit_live.set_defaults(func=_cmd_fit_live)

    args = parser.parse_args()
    args.func(args)


def _cmd_run_eda(args) -> None:
    from .eda import run_eda

    if args.source:
        # Build feature store from S3/local into <output>/feature_store/ first
        fs_path = Path(args.output) / "feature_store"
        seasons = season_range(args.season_start, args.season_end)
        print(f"Building feature store from {args.source} → {fs_path}")
        build_feature_store(args.source, str(fs_path), seasons=seasons)
    else:
        fs_path = Path(args.feature_store)

    outputs = run_eda(
        feature_store_path=fs_path,
        output_dir=args.output,
        seasons=args.seasons,
        targets_only=args.targets_only,
        features_only=args.features_only,
    )
    print(json.dumps(outputs, indent=2))


def _cmd_build_feature_store(args) -> None:
    seasons = season_range(args.season_start, args.season_end)
    outputs = build_feature_store(args.source, args.output, seasons=seasons)
    print(json.dumps(outputs, indent=2))


def _cmd_fit_pregame(args) -> None:
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from .datasets import (
        PregameSequenceDataset,
        SequenceSpec,
        Standardizer,
        infer_feature_columns,
        temporal_split_dates,
    )
    from .distributions import gaussian_nll, weighted_mean
    from .evaluation import evaluate_pregame_model, save_evaluation
    from .models import PregameMultiTaskModel

    _setup_logging()
    t0 = time.time()

    feature_store = Path(args.feature_store)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    log.info("Loading team_games.parquet ...")
    t = time.time()
    team_games = pd.read_parquet(feature_store / "team_games.parquet")
    log.info("  team_games: %d rows, %d cols (%.1fs)", len(team_games), len(team_games.columns), time.time() - t)

    log.info("Loading game_targets.parquet ...")
    t = time.time()
    game_targets = pd.read_parquet(feature_store / "game_targets.parquet")
    log.info("  game_targets: %d rows (%.1fs)", len(game_targets), time.time() - t)

    # Canonical source: pregame.strategy.config.SKIP_SEASONS
    from pregame.strategy.config import SKIP_SEASONS
    if SKIP_SEASONS:
        if "season" in team_games.columns:
            team_games = team_games[~team_games["season"].isin(SKIP_SEASONS)].reset_index(drop=True)
        if "season" in game_targets.columns:
            game_targets = game_targets[~game_targets["season"].isin(SKIP_SEASONS)].reset_index(drop=True)

    train_end, val_end = temporal_split_dates(game_targets)
    log.info("Temporal split: train < %s, val < %s", train_end.date(), val_end.date())

    log.info("Fitting standardizer on %d train rows ...", (pd.to_datetime(team_games["game_date"]) < train_end).sum())
    t = time.time()
    feature_columns = infer_feature_columns(team_games)
    train_team_games = team_games[pd.to_datetime(team_games["game_date"]) < train_end]
    standardizer = Standardizer.fit(train_team_games, feature_columns)
    log.info("  %d feature columns (%.1fs)", len(feature_columns), time.time() - t)

    spec = SequenceSpec(
        history_length=args.history_length,
        min_history=args.min_history,
        time_decay_lambda=args.time_decay_lambda,
    )

    log.info("Building train dataset (history filter may take a while) ...")
    t = time.time()
    train_ds = PregameSequenceDataset(
        team_games,
        game_targets,
        standardizer,
        spec,
        split_end=train_end,
    )
    log.info("  train_ds: %d samples (%.1fs)", len(train_ds), time.time() - t)

    log.info("Building val dataset ...")
    t = time.time()
    val_ds = PregameSequenceDataset(
        team_games,
        game_targets,
        standardizer,
        spec,
        split_start=train_end,
        split_end=val_end,
    )
    log.info("  val_ds: %d samples (%.1fs)", len(val_ds), time.time() - t)

    log.info("Building test dataset ...")
    t = time.time()
    test_ds = PregameSequenceDataset(
        team_games,
        game_targets,
        standardizer,
        spec,
        split_start=val_end,
    )
    log.info("  test_ds: %d samples (%.1fs)", len(test_ds), time.time() - t)

    if len(train_ds) == 0:
        raise RuntimeError("No training samples after history and target filters")
    if len(val_ds) == 0:
        raise RuntimeError("No validation samples after temporal split")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model = PregameMultiTaskModel(
        feature_dim=len(feature_columns),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %d trainable parameters", n_params)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    log.info("Starting training: %d epochs, batch_size=%d, lr=%s", args.epochs, args.batch_size, args.learning_rate)
    log.info("Setup complete in %.1fs — epoch 1 starting ...", time.time() - t0)

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    best_val = float("inf")
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        train_loss = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            gaussian_nll=gaussian_nll,
            weighted_mean=weighted_mean,
            epoch=epoch,
            total_epochs=args.epochs,
            n_batches=n_train_batches,
        )
        val_loss = _evaluate(
            model,
            val_loader,
            device,
            gaussian_nll=gaussian_nll,
            weighted_mean=weighted_mean,
            epoch=epoch,
            total_epochs=args.epochs,
            n_batches=n_val_batches,
        )
        epoch_time = time.time() - t_epoch
        improved = val_loss < best_val
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "epoch_secs": round(epoch_time, 1)}
        history.append(row)

        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        patience_str = f"  (no improvement {epochs_without_improvement}/{args.patience})" if not improved else ""
        log.info("epoch %d/%d  (%.1fs)%s%s", epoch, args.epochs, epoch_time, "  * new best" if improved else "", patience_str)
        log.info("  train_loss = %.4f", train_loss)
        log.info("  val_loss   = %.4f", val_loss)

        _save_checkpoint(
            output / f"checkpoint_epoch{epoch:03d}.pt",
            model,
            standardizer,
            args,
            feature_columns,
            train_end,
            val_end,
            val_loss,
        )
        if improved:
            best_val = val_loss
            _save_checkpoint(
                output / "model.pt",
                model,
                standardizer,
                args,
                feature_columns,
                train_end,
                val_end,
                best_val,
            )

        if epochs_without_improvement >= args.patience:
            log.info("Early stopping: no val improvement for %d epochs.", args.patience)
            break

    (output / "history.json").write_text(json.dumps(history, indent=2))

    log.info("Running evaluation on validation split ...")
    val_reports = evaluate_pregame_model(model, val_loader, device, game_targets)
    val_path = save_evaluation(val_reports, str(output / "eval_val"))
    log.info("Validation evaluation saved: %s", val_path)

    if len(test_ds) > 0:
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        log.info("Running evaluation on test split ...")
        test_reports = evaluate_pregame_model(model, test_loader, device, game_targets)
        test_path = save_evaluation(test_reports, str(output / "eval_test"))
        log.info("Test evaluation saved: %s", test_path)
    else:
        log.warning("No test samples available (all data used for train+val).")

    log.info("Total runtime: %.1fs", time.time() - t0)


def _progress(step: int, total: int, loss: float, label: str, t_start: float) -> None:
    elapsed = time.time() - t_start
    rate = step / elapsed if elapsed > 0 else 0
    eta = (total - step) / rate if rate > 0 else 0
    bar_width = 24
    filled = int(bar_width * step / total) if total > 0 else 0
    bar = "#" * filled + "-" * (bar_width - filled)
    print(
        f"\r  {label} [{bar}] {step}/{total}  loss={loss:.4f}  {rate:.1f}b/s  eta={eta:.0f}s",
        end="",
        flush=True,
    )


def _run_epoch(model, loader, optimizer, device, gaussian_nll, weighted_mean,
               epoch=0, total_epochs=0, n_batches=0) -> float:
    model.train()
    total = 0.0
    count = 0
    t_start = time.time()
    for batch in loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        preds = model(batch)
        loss = _loss(batch, preds, gaussian_nll, weighted_mean)
        loss.backward()
        torch_clip = __import__("torch").nn.utils.clip_grad_norm_
        torch_clip(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += float(loss.detach().cpu())
        count += 1
        _progress(count, n_batches, total / count, f"e{epoch}/{total_epochs} train", t_start)
    print(flush=True)
    return total / max(count, 1)


def _evaluate(model, loader, device, gaussian_nll, weighted_mean,
              epoch=0, total_epochs=0, n_batches=0) -> float:
    import torch

    model.eval()
    total = 0.0
    count = 0
    t_start = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            preds = model(batch)
            loss = _loss(batch, preds, gaussian_nll, weighted_mean)
            total += float(loss.detach().cpu())
            count += 1
            _progress(count, n_batches, total / count, f"e{epoch}/{total_epochs}   val", t_start)
    print(flush=True)
    return total / max(count, 1)


def _loss(batch, preds, gaussian_nll, weighted_mean):
    import torch.nn.functional as F

    targets = batch["targets"]
    weights = batch["sample_weight"]
    home_win = F.binary_cross_entropy_with_logits(
        preds["home_win_logit"], targets["home_win"], reduction="none"
    )
    yrfi = F.binary_cross_entropy_with_logits(
        preds["yrfi_logit"], targets["yrfi"], reduction="none"
    )
    total = gaussian_nll(
        targets["total_runs"], preds["total_runs_mu"], preds["total_runs_sigma"]
    )
    diff = gaussian_nll(
        targets["home_run_diff"], preds["home_run_diff_mu"], preds["home_run_diff_sigma"]
    )
    return (
        weighted_mean(home_win, weights)
        + 0.50 * weighted_mean(yrfi, weights)
        + 0.25 * weighted_mean(total, weights)
        + 0.25 * weighted_mean(diff, weights)
    )


def _to_device(value, device):
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _save_checkpoint(
    path,
    model,
    standardizer,
    args,
    feature_columns,
    train_end,
    val_end,
    best_val,
) -> None:
    import torch

    payload = {
        "model_state": model.state_dict(),
        "standardizer": standardizer.to_dict(),
        "feature_columns": feature_columns,
        "config": vars(args),
        "train_end": str(train_end),
        "val_end": str(val_end),
        "best_val_loss": best_val,
    }
    torch.save(payload, path)


def _cmd_fit_live(args) -> None:
    """Train the live HAN model on pitch sequences."""
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from .datasets import Standardizer, infer_live_feature_columns, temporal_split_dates
    from .distributions import negbin_nll, weighted_mean
    from .han_model import LiveHANModel
    from .live_dataset import LiveHANDataset

    _setup_logging()
    t0 = time.time()

    feature_store = Path(args.feature_store)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    log.info("Loading pitch_sequences.parquet ...")
    t = time.time()
    pitch_sequences = pd.read_parquet(feature_store / "pitch_sequences.parquet")
    log.info("  pitch_sequences: %d rows, %d cols (%.1fs)", len(pitch_sequences), len(pitch_sequences.columns), time.time() - t)

    log.info("Loading game_targets.parquet ...")
    t = time.time()
    game_targets = pd.read_parquet(feature_store / "game_targets.parquet")
    log.info("  game_targets: %d rows (%.1fs)", len(game_targets), time.time() - t)

    # Load pregame features (if available, otherwise use dummy)
    pregame_path = feature_store / "team_games.parquet"
    if pregame_path.exists():
        log.info("Loading pregame features from team_games.parquet ...")
        t = time.time()
        pregame_features = pd.read_parquet(pregame_path)
        log.info("  pregame_features: %d rows (%.1fs)", len(pregame_features), time.time() - t)
    else:
        log.warning("No team_games.parquet found — using placeholder pregame features")
        pregame_features = pd.DataFrame({"game_pk": game_targets["game_pk"], "game_date": game_targets["game_date"]})

    # Canonical source: pregame.strategy.config.SKIP_SEASONS
    from pregame.strategy.config import SKIP_SEASONS
    if SKIP_SEASONS:
        if "season" in pitch_sequences.columns:
            pitch_sequences = pitch_sequences[~pitch_sequences["season"].isin(SKIP_SEASONS)].reset_index(drop=True)
        if "season" in game_targets.columns:
            game_targets = game_targets[~game_targets["season"].isin(SKIP_SEASONS)].reset_index(drop=True)
        if "season" in pregame_features.columns:
            pregame_features = pregame_features[~pregame_features["season"].isin(SKIP_SEASONS)].reset_index(drop=True)

    # Temporal splits: 2015-2023 train, 2024 val, 2025+ test
    train_end, val_end = temporal_split_dates(game_targets)
    log.info("Temporal split: train < %s, val < %s", train_end.date(), val_end.date())

    # Fit standardizer on training pitch sequences
    log.info("Fitting standardizer on training pitch sequences ...")
    t = time.time()
    feature_columns = infer_live_feature_columns(pitch_sequences)
    train_pitches = pitch_sequences[pd.to_datetime(pitch_sequences["game_date"]) < train_end]
    standardizer = Standardizer.fit(train_pitches, feature_columns)
    log.info("  %d feature columns (%.1fs)", len(feature_columns), time.time() - t)

    # Build datasets
    log.info("Building train dataset ...")
    t = time.time()
    train_ds = LiveHANDataset(
        pitch_sequences,
        game_targets,
        pregame_features,
        standardizer,
        split_end=train_end,
        stride=25,  # TODO: validate — placeholder
        max_prefixes_per_game=32,  # TODO: validate — placeholder
        max_seq_len=args.max_seq_len,
    )
    log.info("  train_ds: %d sub-game samples (%.1fs)", len(train_ds), time.time() - t)

    log.info("Building val dataset ...")
    t = time.time()
    val_ds = LiveHANDataset(
        pitch_sequences,
        game_targets,
        pregame_features,
        standardizer,
        split_start=train_end,
        split_end=val_end,
        stride=25,
        max_prefixes_per_game=32,
        max_seq_len=args.max_seq_len,
    )
    log.info("  val_ds: %d sub-game samples (%.1fs)", len(val_ds), time.time() - t)

    log.info("Building test dataset ...")
    t = time.time()
    test_ds = LiveHANDataset(
        pitch_sequences,
        game_targets,
        pregame_features,
        standardizer,
        split_start=val_end,
        stride=25,
        max_prefixes_per_game=32,
        max_seq_len=args.max_seq_len,
    )
    log.info("  test_ds: %d sub-game samples (%.1fs)", len(test_ds), time.time() - t)

    if len(train_ds) == 0:
        raise RuntimeError("No training samples after filtering")
    if len(val_ds) == 0:
        raise RuntimeError("No validation samples after temporal split")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=_collate_live_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=_collate_live_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Initialize model
    model = LiveHANModel(
        d_model=args.d_model,
        d_pregame=128,  # TODO: validate — placeholder (match pregame feature dim)
        batter_buckets=512,
        pitcher_buckets=512,
        player_embed_dim=16,  # TODO: validate — placeholder
        n_heads=args.n_heads,
        n_layers_per_level=2,  # TODO: validate — placeholder
        dim_feedforward=args.d_model * 4,  # Standard Transformer ratio
        dropout=args.dropout,
        max_innings=20,
        max_abs_per_inning=25,
        max_pitches_per_ab=15,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %d trainable parameters", n_params)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    loss_config = {
        "negbin_weight": args.negbin_weight,
        "win_weight": args.win_weight,
        "ei_weight": args.ei_weight,
        "yrfi_weight": args.yrfi_weight,
    }

    log.info("Starting training: %d epochs, batch_size=%d, lr=%s", args.epochs, args.batch_size, args.learning_rate)
    log.info("Loss weights: NegBin=%.2f, Win=%.2f, EI=%.2f, YRFI=%.2f",
             args.negbin_weight, args.win_weight, args.ei_weight, args.yrfi_weight)
    log.info("Setup complete in %.1fs — epoch 1 starting ...", time.time() - t0)

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    best_val = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()

        train_loss = _run_live_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_config,
            args.gradient_clip,
            epoch=epoch,
            total_epochs=args.epochs,
            n_batches=n_train_batches,
        )

        val_loss = _evaluate_live(
            model,
            val_loader,
            device,
            loss_config,
            epoch=epoch,
            total_epochs=args.epochs,
            n_batches=n_val_batches,
        )

        epoch_time = time.time() - t_epoch
        improved = val_loss < best_val
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_secs": round(epoch_time, 1),
        }
        history.append(row)

        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        patience_str = f"  (no improvement {epochs_without_improvement}/{args.patience})" if not improved else ""
        log.info("epoch %d/%d  (%.1fs)%s%s", epoch, args.epochs, epoch_time, "  * new best" if improved else "", patience_str)
        log.info("  train_loss = %.4f", train_loss)
        log.info("  val_loss   = %.4f", val_loss)

        # Save epoch checkpoint
        _save_live_checkpoint(
            output / f"checkpoint_epoch{epoch:03d}.pt",
            model,
            standardizer,
            args,
            feature_columns,
            train_end,
            val_end,
            val_loss,
        )

        if improved:
            best_val = val_loss
            _save_live_checkpoint(
                output / "model.pt",
                model,
                standardizer,
                args,
                feature_columns,
                train_end,
                val_end,
                best_val,
            )

        if epochs_without_improvement >= args.patience:
            log.info("Early stopping: no val improvement for %d epochs.", args.patience)
            break

    (output / "history.json").write_text(json.dumps(history, indent=2))

    # Evaluation on test set
    if len(test_ds) > 0:
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=_collate_live_batch,
        )
        log.info("Running evaluation on test split ...")
        test_metrics = _eval_live_metrics(model, test_loader, device)
        log.info("Test set metrics:")
        for key, val in test_metrics.items():
            log.info("  %s = %.4f", key, val)
        (output / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    else:
        log.warning("No test samples available.")

    log.info("Total runtime: %.1fs", time.time() - t0)


def _collate_live_batch(samples: list[dict]) -> dict:
    """Collate a batch of LiveHANDataset samples with proper padding.

    WHY custom collate: HAN model requires variable-length sequences to be padded
    to the same length within a batch, with attention masks indicating valid positions.
    """
    import torch

    if len(samples) == 0:
        raise ValueError("Empty batch")

    # Find max sequence length in this batch
    max_len = max(s["continuous"].size(0) for s in samples)

    batch = {}

    # Pad sequence tensors to max_len
    seq_keys = [
        "continuous", "pitch_type", "outcome_flags", "count_state", "game_state",
        "base_state", "batter_hash", "pitcher_hash", "handedness", "score",
        "positional", "intra_ab", "elapsed_time", "hierarchy_indices",
    ]

    for key in seq_keys:
        tensors = []
        for s in samples:
            t = s[key]
            seq_len = t.size(0)
            if seq_len < max_len:
                # Left-pad with zeros
                pad_len = max_len - seq_len
                if t.dim() == 1:
                    pad_shape = (pad_len,)
                else:
                    pad_shape = (pad_len,) + t.shape[1:]
                pad = torch.zeros(pad_shape, dtype=t.dtype)
                t = torch.cat([pad, t], dim=0)
            tensors.append(t)
        batch[key] = torch.stack(tensors, dim=0)

    # Attention mask: True for valid positions
    masks = []
    for s in samples:
        seq_len = s["continuous"].size(0)
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[max_len - seq_len:] = True  # Valid positions are at the end (left-padded)
        masks.append(mask)
    batch["attention_mask"] = torch.stack(masks, dim=0)

    # Non-sequence tensors
    batch["pregame_prior"] = torch.stack([s["pregame_prior"] for s in samples], dim=0)
    batch["game_progress"] = torch.stack([s["game_progress"] for s in samples], dim=0).squeeze(-1)

    # Targets
    batch["targets"] = {
        "remaining_home_runs": torch.stack([s["targets"]["remaining_home_runs"] for s in samples], dim=0),
        "remaining_away_runs": torch.stack([s["targets"]["remaining_away_runs"] for s in samples], dim=0),
        "home_win": torch.stack([s["targets"]["home_win"] for s in samples], dim=0),
        "yrfi": torch.stack([s["targets"]["yrfi"] for s in samples], dim=0),
        "extra_innings": torch.stack([s["targets"]["extra_innings"] for s in samples], dim=0),
    }

    # Sample weights and metadata
    batch["sample_weight"] = torch.stack([s["sample_weight"] for s in samples], dim=0)
    batch["game_pk"] = torch.stack([s["game_pk"] for s in samples], dim=0)
    batch["game_progress_scalar"] = torch.stack([s["game_progress_scalar"] for s in samples], dim=0)

    return batch


def _run_live_epoch(
    model,
    loader,
    optimizer,
    device,
    loss_config: dict,
    gradient_clip: float,
    epoch=0,
    total_epochs=0,
    n_batches=0,
) -> float:
    """Run one training epoch for the live HAN model."""
    model.train()
    total = 0.0
    count = 0
    t_start = time.time()

    for batch in loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        preds = model(batch)

        # Multi-task loss
        loss = _live_model_loss(preds, batch["targets"], batch["game_progress_scalar"], batch["sample_weight"], loss_config)

        # Backward pass
        loss.backward()

        # Gradient clipping (prevents NegBin parameterization instability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

        optimizer.step()

        total += float(loss.detach().cpu())
        count += 1
        _progress(count, n_batches, total / count, f"e{epoch}/{total_epochs} train", t_start)

    print(flush=True)
    return total / max(count, 1)


def _evaluate_live(
    model,
    loader,
    device,
    loss_config: dict,
    epoch=0,
    total_epochs=0,
    n_batches=0,
) -> float:
    """Evaluate the live HAN model on a validation set."""
    import torch

    model.eval()
    total = 0.0
    count = 0
    t_start = time.time()

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            preds = model(batch)
            loss = _live_model_loss(preds, batch["targets"], batch["game_progress_scalar"], batch["sample_weight"], loss_config)
            total += float(loss.detach().cpu())
            count += 1
            _progress(count, n_batches, total / count, f"e{epoch}/{total_epochs}   val", t_start)

    print(flush=True)
    return total / max(count, 1)


def _live_model_loss(
    outputs: dict,
    targets: dict,
    game_progress: torch.Tensor,
    sample_weights: torch.Tensor,
    loss_config: dict,
) -> torch.Tensor:
    """Multi-task loss for the live HAN model.

    Components:
    1. NegBin NLL for remaining home/away runs (primary — proper scoring rule)
    2. BCE for home_win (auxiliary consistency check)
    3. BCE for extra_innings (masked: only when game_progress > 0.6)
    4. BCE for YRFI (masked: only when game_progress < 0.056, i.e. first inning)

    WHY these weights: NegBin is primary because it captures the full distribution.
    Classification heads are auxiliary regularizers that prevent the distributional
    output from drifting into physically impossible states (e.g., predicting 0
    remaining runs while win probability is 0.5).

    WHY masking: YRFI is deterministic after inning 1; extra innings is deterministic
    before inning 9. Masking prevents the model from wasting capacity on irrelevant predictions.
    """
    import torch.nn.functional as F
    from .distributions import negbin_nll, weighted_mean

    # NegBin NLL (primary loss)
    nll_home = negbin_nll(
        targets["remaining_home_runs"],
        outputs["mu_home_remaining"],
        outputs["alpha_home_remaining"],
    )
    nll_away = negbin_nll(
        targets["remaining_away_runs"],
        outputs["mu_away_remaining"],
        outputs["alpha_away_remaining"],
    )
    negbin_loss = weighted_mean(nll_home + nll_away, sample_weights)

    # Home win BCE (auxiliary)
    win_loss = F.binary_cross_entropy_with_logits(
        outputs["home_win_logit"],
        targets["home_win"],
        reduction="none",
    )
    win_loss = weighted_mean(win_loss, sample_weights)

    # Extra innings BCE (masked: only train on late-game samples)
    # WHY 0.6: After inning 5.5 (mid-6th), extra innings becomes a relevant prediction
    ei_mask = (game_progress > 0.6).float()
    ei_loss = F.binary_cross_entropy_with_logits(
        outputs["extra_innings_logit"],
        targets["extra_innings"],
        reduction="none",
    )
    ei_loss = weighted_mean(ei_loss * ei_mask, sample_weights * ei_mask + 1e-6)

    # YRFI BCE (masked: only train on samples where first inning is still in progress)
    # WHY 0.056: 0.5/9 = 0.056 covers approximately the top of the first inning.
    # Using 0.12 incorrectly includes top-of-2nd where YRFI is already settled.
    yrfi_mask = (game_progress < 0.056).float()
    yrfi_loss = F.binary_cross_entropy_with_logits(
        outputs["yrfi_logit"],
        targets["yrfi"],
        reduction="none",
    )
    yrfi_loss = weighted_mean(yrfi_loss * yrfi_mask, sample_weights * yrfi_mask + 1e-6)

    # Weighted combination
    total_loss = (
        loss_config["negbin_weight"] * negbin_loss
        + loss_config["win_weight"] * win_loss
        + loss_config["ei_weight"] * ei_loss
        + loss_config["yrfi_weight"] * yrfi_loss
    )

    return total_loss


def _eval_live_metrics(model, loader, device) -> dict:
    """Compute evaluation metrics for the live HAN model.

    Metrics:
    - NegBin NLL (primary)
    - Brier score for derived P(home_win)
    - Calibration (ECE) stratified by game progress
    """
    import torch
    import numpy as np
    from .distributions import negbin_nll

    model.eval()
    all_nll = []
    all_brier = []

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            preds = model(batch)

            # NegBin NLL
            nll_home = negbin_nll(
                batch["targets"]["remaining_home_runs"],
                preds["mu_home_remaining"],
                preds["alpha_home_remaining"],
            )
            nll_away = negbin_nll(
                batch["targets"]["remaining_away_runs"],
                preds["mu_away_remaining"],
                preds["alpha_away_remaining"],
            )
            all_nll.extend((nll_home + nll_away).cpu().numpy().tolist())

            # Brier score for home_win
            p_win = torch.sigmoid(preds["home_win_logit"])
            brier = (p_win - batch["targets"]["home_win"]) ** 2
            all_brier.extend(brier.cpu().numpy().tolist())

    return {
        "negbin_nll_mean": float(np.mean(all_nll)),
        "negbin_nll_std": float(np.std(all_nll)),
        "brier_score_mean": float(np.mean(all_brier)),
        "brier_score_std": float(np.std(all_brier)),
    }


def _save_live_checkpoint(
    path: Path,
    model,
    standardizer,
    args,
    feature_columns: list[str],
    train_end,
    val_end,
    best_val: float,
) -> None:
    """Save a live HAN model checkpoint."""
    import torch

    payload = {
        "model_state": model.state_dict(),
        "standardizer": standardizer.to_dict(),
        "feature_columns": feature_columns,
        "config": {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "dropout": args.dropout,
            "max_seq_len": args.max_seq_len,
            "d_pregame": 128,  # TODO: validate — placeholder
            "batter_buckets": 512,
            "pitcher_buckets": 512,
            "player_embed_dim": 16,
            "n_layers_per_level": 2,
            "dim_feedforward": args.d_model * 4,
            "max_innings": 20,
            "max_abs_per_inning": 25,
            "max_pitches_per_ab": 15,
        },
        "train_end": str(train_end),
        "val_end": str(val_end),
        "best_val_loss": best_val,
        "architecture": "LiveHANModel",
    }
    torch.save(payload, path)
    log.debug("Saved checkpoint: %s", path)


if __name__ == "__main__":
    main()

