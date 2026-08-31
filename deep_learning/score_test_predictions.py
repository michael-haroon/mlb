#!/usr/bin/env python3.11
"""Dump per-SAMPLE test-split predictions from a GameTransformer checkpoint.

WHY THIS EXISTS: every DL metric recorded so far pools all ~15 samples per game into one
number, and only 6.55% of those samples are pregame (`prefix_length == 0`). The remaining
93.45% are mid-game cuts, 47.6% of which have >=200 pitches already played. Two of the six
phase-1 heads -- `negbin_home` and `negbin_away`, whose target is runs REMAINING -- are 73%
of the objective and are near-determined on those late cuts (20.5% of all samples have zero
runs left to predict). So the pooled number is dominated by scoreboard reading, and no
pooled comparison against a pregame-only classical model is meaningful.

Dumping predictions per sample rather than printing metrics is deliberate: the same artifact
answers "how good is the DL model pregame", "how does each head decay with prefix length",
and "how does it compare to the classical ensemble on identical games" without re-running the
GPU pass for each question.

The pregame path is also STRUCTURALLY different, not just rarer: `_team_readout` mean-pools
the context tokens when T=0 and takes the last live token otherwise. Pregame predictions
therefore come from a readout that receives 6.55% of the gradient signal.

Usage (on a box with the prepared tensors and the checkpoint):
  python3.11 deep_learning/score_test_predictions.py \
      --prepared-dir /mnt/fast/prepared_tensors \
      --checkpoint /mnt/fast/ab_runs/control/final_model.pt \
      --d-model 384 --n-layers 6 --n-heads 12 \
      --output /mnt/fast/pregame_cmp/control_test_preds.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlb_dl.game_transformer import GameTransformer  # noqa: E402
from mlb_dl.precollate import load_prepared_datasets, prepared_collate_fn  # noqa: E402
from mlb_dl.train_unified import (  # noqa: E402
    _prepare_model_input, _resolve_weather_geometry, _to_device,
)

log = logging.getLogger("score_test")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-asof-weather", action="store_true")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    train_ds, val_ds, test_ds = load_prepared_datasets(
        args.prepared_dir, disable_asof=args.no_asof_weather)
    ds = test_ds if args.split == "test" else val_ds
    del train_ds
    log.info("%s split: %d samples, %d games", args.split, len(ds), len(ds._game_pks))

    rating_dim = ds.manifest.get("rating_dim", 0)
    context_config, _ = _resolve_weather_geometry(
        ds, True, disable_asof=args.no_asof_weather)

    model = GameTransformer(
        d_model=args.d_model,
        rating_dim=rating_dim,
        flat_feature_dim=30,
        context_config=context_config,
        num_backbone_layers=args.n_layers,
        num_heads=args.n_heads,
        d_ff=args.d_model * 4,
        dropout=0.0,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    # torch.compile prefixes every key with "_orig_mod."; loading such a checkpoint into a
    # bare module fails on every key at once, which reads like an architecture mismatch.
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # Loud, not fatal: the loss module's log_weights live in the checkpoint of some runs
        # and a genuine geometry mismatch must not be mistaken for that.
        log.warning("load_state_dict: %d missing, %d unexpected", len(missing), len(unexpected))
        for k in list(missing)[:10]:
            log.warning("  missing: %s", k)
        for k in list(unexpected)[:10]:
            log.warning("  unexpected: %s", k)
        if len(missing) > 0.05 * len(list(model.state_dict())):
            log.error("more than 5%% of parameters unmatched — refusing to score a "
                      "half-initialised model")
            return 1
    model.eval()

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=prepared_collate_fn, num_workers=args.num_workers)

    rows: list[dict] = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            gpk = batch["game_pk"].numpy()
            plen = batch["prefix_length"].numpy()
            tg = batch["targets"]
            # Reuse the trainer's own batch prep rather than moving tensors by hand: the model
            # consumes a "context" key that `_prepare_model_input` assembles from the flat
            # collate output, and player_context_dim must match how the checkpoint was trained.
            dev_batch = _to_device(batch, device)
            model_input = _prepare_model_input(
                dev_batch, player_context_dim=args.d_model * 2)
            out = model(model_input)
            rows.append(pd.DataFrame({
                "game_pk": gpk,
                "prefix_length": plen,
                # Predictions
                "p_home_win": torch.sigmoid(out["home_win_logit"]).float().cpu().numpy(),
                "p_yrfi": torch.sigmoid(out["yrfi_logit"]).float().cpu().numpy(),
                "p_extra_innings": torch.sigmoid(
                    out["extra_innings_logit"]).float().cpu().numpy(),
                "mu_home": out["mu_home"].float().cpu().numpy(),
                "mu_away": out["mu_away"].float().cpu().numpy(),
                "alpha_home": out["alpha_home"].float().cpu().numpy(),
                "alpha_away": out["alpha_away"].float().cpu().numpy(),
                # Targets. home_win/yrfi/extra_innings/total_runs are GAME-level (constant
                # across a game's prefixes); the two *_remaining are per-sample.
                "y_home_win": tg["home_win"].numpy(),
                "y_yrfi": tg["yrfi"].numpy(),
                "y_extra_innings": tg["extra_innings"].numpy(),
                "y_total_runs": tg["total_runs"].numpy(),
                "y_home_runs_remaining": tg["home_runs_remaining"].numpy(),
                "y_away_runs_remaining": tg["away_runs_remaining"].numpy(),
                "yrfi_mask": batch["yrfi_mask"].numpy(),
            }))
            if bi % 100 == 0:
                log.info("  batch %d/%d", bi, len(loader))

    df = pd.concat(rows, ignore_index=True)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outp, index=False)
    log.info("wrote %s: %d rows, %d games, %d pregame rows",
             outp, len(df), df["game_pk"].nunique(), int((df["prefix_length"] == 0).sum()))

    # Sanity that must hold or the join downstream is meaningless: exactly one pregame row
    # per game, and it is the only row where runs-remaining equals the final total.
    n_pre = int((df["prefix_length"] == 0).sum())
    if n_pre != df["game_pk"].nunique():
        log.error("pregame rows (%d) != games (%d) — the prefix-0 subset is not one per game",
                  n_pre, df["game_pk"].nunique())
        return 1
    pre = df[df["prefix_length"] == 0]
    resid = (pre["y_home_runs_remaining"] + pre["y_away_runs_remaining"]
             - pre["y_total_runs"]).abs()
    log.info("pregame check: max |remaining_sum - total_runs| = %.6f (expect 0)", resid.max())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
