#!/usr/bin/env python3.11
"""Dump per-PLAYER-SLOT test-split predictions from a GameTransformer checkpoint.

WHY THIS EXISTS: `score_test_predictions.py` dumps only the six GAME-level heads. The whole
point of the phases-2/3 curriculum was the PLAYER heads (hits / HR / pitcher-K / H+R+RBI / SB),
which received gradient for the first time in phase 2 and have NEVER been scored on held-out
data. Without this, "did the phases help the player props" is unanswerable.

MASKING IS LOAD-BEARING: player slots are padded to MAX_PLAYERS and padding is 0, not -1
(`precollate.py` `player_mask` == 1 on real slots). Scoring the padded slots silently craters
every metric — this is the exact bug recorded in `project_player_mask_eval_bug_2026_08_30.md`.
This script writes ONE row per (game, prefix, VALID player slot) and drops padding via the mask
before anything is scored downstream.

Player targets are GAME-level (constant across a game's prefixes — `precollate.py:814-823`
indexes them by game, not sample), while the player-head predictions come from the prefix's
`game_repr` and therefore vary with prefix_length. Keeping `prefix_length` per row lets the
downstream analysis bucket the decay curve and, in particular, isolate the honest pregame test
(prefix_length == 0), which mirrors the single-game player-performance task in the literature.

Usage (on a box with the prepared tensors and the checkpoint):
  python3.11 deep_learning/score_player_heads.py \
      --prepared-dir /mnt/fast/prepared_tensors \
      --checkpoint   /mnt/fast/phases23/phase3/best.pt \
      --d-model 384 --n-layers 6 --n-heads 8 --no-asof-weather \
      --output /mnt/fast/phases23/phase3_player_preds.parquet
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

log = logging.getLogger("score_player")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=8)
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
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
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

    player_context_dim = args.d_model * 2  # player_context_tokens=2 (train_unified default)
    rows: list[dict] = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            gpk = batch["game_pk"].numpy()          # [B]
            plen = batch["prefix_length"].numpy()   # [B]
            tg = batch["targets"]
            dev_batch = _to_device(batch, device)
            model_input = _prepare_model_input(dev_batch, player_context_dim=player_context_dim)
            out = model(model_input)

            # All player tensors are [B, P]. Flatten to [B*P] and keep valid slots only.
            pm = batch["player_mask"].numpy()                       # [B, P] 1=real 0=pad
            B, P = pm.shape
            # P(1+ hit) = 1 - P(0 hits); hits_categorical is [B, P, 5].
            p_hit = (1.0 - out["hits_categorical"][..., 0]).float().cpu().numpy()
            p_hr = out["hr_prob"].float().cpu().numpy()
            p_sb = torch.sigmoid(out["stolen_bases_logit"]).float().cpu().numpy()
            k_mu = out["pitcher_k_mu"].float().cpu().numpy()
            hrbi_mu = out["h_r_rbi_mu"].float().cpu().numpy()

            game_col = np.repeat(gpk, P)
            plen_col = np.repeat(plen, P)
            slot_col = np.tile(np.arange(P), B)
            frame = pd.DataFrame({
                "game_pk": game_col,
                "prefix_length": plen_col,
                "player_slot": slot_col,
                "player_mask": pm.reshape(-1),
                "p_hit": p_hit.reshape(-1),
                "p_hr": p_hr.reshape(-1),
                "p_sb": p_sb.reshape(-1),
                "k_mu": k_mu.reshape(-1),
                "hrbi_mu": hrbi_mu.reshape(-1),
                "y_hits": tg["player_hits"].numpy().reshape(-1),
                "y_hr": tg["player_hr"].numpy().reshape(-1),
                "y_so": tg["player_so"].numpy().reshape(-1),
                "y_hrbi": tg["player_hrbi"].numpy().reshape(-1),
                "y_sb": tg["player_sb"].numpy().reshape(-1),
            })
            # Drop padding here so no downstream step can accidentally score it.
            rows.append(frame[frame["player_mask"] == 1].drop(columns=["player_mask"]))
            if bi % 100 == 0:
                log.info("  batch %d/%d", bi, len(loader))

    df = pd.concat(rows, ignore_index=True)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outp, index=False)
    log.info("wrote %s: %d valid-slot rows, %d games, %d pregame rows",
             outp, len(df), df["game_pk"].nunique(), int((df["prefix_length"] == 0).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
