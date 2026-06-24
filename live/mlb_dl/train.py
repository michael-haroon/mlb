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

    raw_eda = sub.add_parser("run-raw-eda", help="Run EDA directly on raw PITCHES table (Workflow 1)")
    raw_eda.add_argument("--source", required=True, help="S3 or local path to raw data")
    raw_eda.add_argument("--output", required=True, help="Output directory for EDA artifacts")
    raw_eda.add_argument("--season-start", type=int, help="First season to ingest")
    raw_eda.add_argument("--season-end", type=int, help="Last season to ingest")
    raw_eda.set_defaults(func=_cmd_run_raw_eda)

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


def _cmd_run_raw_eda(args) -> None:
    from .raw_eda import run_raw_eda

    seasons = season_range(args.season_start, args.season_end)
    outputs = run_raw_eda(
        source_uri=args.source,
        output_dir=args.output,
        seasons=seasons,
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


if __name__ == "__main__":
    main()

