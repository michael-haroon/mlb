#!/usr/bin/env python3.11
"""Cache the frozen trunk's context representation for every pregame row of a split.

WHY THIS EXISTS. Pregame sits at zero skill on all four pregame quantities (home_win BSS
-0.0041, LogLoss 0.69337 vs ln2 = 0.69315). Two causes were fixed -- the readout branch that
serving uses was batch-level so it never trained (`ab619fd`), and there was no pregame-only
metric to notice (`a6df6f7`). What is still unknown, and what decides whether a ~5h retrain is
worth paying for, is which of these two the binding constraint is:

  (a) the information IS in the context tokens and only the `ctx_pool -> head` readout failed
      to extract it, or
  (b) the pregame information set itself is too narrow (5 prior starts / 10 prior team games is
      a ~2-week window; the classical stack gets season-to-date and multi-season priors).

A frozen-trunk probe separates them. This script does the expensive half once: one forward pass
over the pregame rows, caching the pooled context representation. `fit_pregame_probe.py` then
fits probes on that matrix in seconds, so probe capacity can be varied without ever re-running
the GPU pass.

TWO PROPERTIES MAKE THIS CHEAP AND EXACT.

1. No model edit is needed. `model.backbone` is a real submodule whose forward output *is*
   `backbone_out`, so a forward hook yields it and `backbone_out[:, :num_context, :].mean(1)`
   is bit-exact with `ctx_pool` in `GameTransformer._team_readout`. Verified at runtime, not
   assumed -- see `_assert_reproduces_own_head`.
2. The prefix-LM mask forbids context->live attention, so the context rows of `backbone_out` do
   not depend on whether live tokens are present. A pregame row's representation is therefore
   identical alone or beside live rows, which is what lets the pregame rows be extracted as
   their own subset without changing a single number.

Using a pre-`ab619fd` checkpoint is VALID here even though its pregame readout is untrained:
the probe discards the head and refits it. The trunk is what the 293,646 live examples trained,
and the trunk is what is under test.

Usage:
  python3.11 deep_learning/extract_pregame_repr.py \
      --prepared-dir /mnt/fast/prepared_tensors \
      --checkpoint /mnt/fast/sweep_A/phase1/best.pt \
      --d-model 384 --n-layers 6 --n-heads 8 \
      --splits train val test \
      --output /mnt/fast/pregame_probe/sweep_A_phase1.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Repo root, not `deep_learning/`, so this module is importable as
# `deep_learning.extract_pregame_repr` from the test suite AND runnable as a script. Inserting
# `deep_learning/` instead (as `score_test_predictions.py` does) would load a SECOND copy of the
# `mlb_dl` package tree under a different name when the tests import this file, and the two
# copies' `GameTransformer` classes would not be the same object.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deep_learning.mlb_dl.game_transformer import (  # noqa: E402
    ContextConfig, GameTransformer,
)
from deep_learning.mlb_dl.precollate import (  # noqa: E402
    load_prepared_datasets, prepared_collate_fn,
)
from deep_learning.mlb_dl.train_unified import (  # noqa: E402
    _prepare_model_input, _resolve_weather_geometry, _to_device,
)

log = logging.getLogger("extract_pregame")

# Segment order is fixed by the append order in `ContextCompiler.forward`: the four game
# categories, then flat, then weather, then the two rating sides. Sizes are derived from the
# config rather than hardcoded because the as-of weather arm has 7 weather tokens (151 total)
# where legacy has 4 (148), and a wrong boundary here would silently mix segments.
SEGMENT_ORDER = ("sp_home", "sp_away", "team_home", "team_away",
                 "flat_features", "weather", "rating_home", "rating_away")


def context_segment_bounds(
    cfg: ContextConfig, rating_dim: int, num_context: int,
) -> list[tuple[str, int, int]]:
    """Half-open [start, end) token bounds per context segment.

    Asserted against the ACTUAL `num_context` rather than `cfg.total_context_tokens`, because
    those two can disagree: `ContextCompiler.forward` skips a game category whose key is absent
    and emits rating tokens only when `rating_dim > 0`, while `cfg.total_context_tokens`
    unconditionally counts all of them. `_team_readout` has a latent bug of exactly this shape
    (it subtracts `cfg.rating_tokens` even when no rating tokens were emitted), so this
    function refuses to guess.
    """
    sizes: list[tuple[str, int]] = [
        ("sp_home", cfg.sp_tokens),
        ("sp_away", cfg.sp_tokens),
        ("team_home", cfg.team_tokens),
        ("team_away", cfg.team_tokens),
        ("flat_features", cfg.flat_feature_tokens),
        ("weather", cfg.weather_tokens),
    ]
    if rating_dim > 0:
        sizes.append(("rating_home", cfg.rating_steps))
        sizes.append(("rating_away", cfg.rating_steps))

    total = sum(n for _, n in sizes)
    if total != num_context:
        raise ValueError(
            f"segment sizes sum to {total} but the compiler emitted {num_context} context "
            f"tokens; the boundary table is stale. sizes={sizes}"
        )

    bounds: list[tuple[str, int, int]] = []
    start = 0
    for name, n in sizes:
        bounds.append((name, start, start + n))
        start += n
    return bounds


def select_pregame_indices(
    prefix_length, sample_to_game, n_games: int,
) -> np.ndarray:
    """Sample indices of the `prefix_length == 0` rows, one per game or it raises.

    The probe's train/select/report partition is by GAME (fit on train, tune on val, report on
    test). That is only true if each game contributes exactly one pregame row: a duplicate would
    put correlated residuals inside a split while leaving the row count plausible, and a missing
    one would silently shrink the population. So the check is on DISTINCT games, not on the count.
    """
    idx = np.flatnonzero(np.asarray(prefix_length) == 0)
    games = np.asarray(sample_to_game)[idx]
    n_distinct = len(np.unique(games))
    if n_distinct != len(idx) or len(idx) != n_games:
        raise ValueError(
            f"prefix_length==0 subset is not one row per game: {len(idx)} pregame rows covering "
            f"{n_distinct} distinct games, but the split has {n_games} games"
        )
    return idx


class _TrunkTap:
    """Captures `backbone_out` and the context-token count from module forward hooks.

    Hooks both `context_compiler` (whose output width is the authoritative `num_context` --
    config arithmetic is not, per `context_segment_bounds`) and `backbone` (whose output is
    `backbone_out` itself, returned as a `(x, kv_cache)` tuple).
    """

    def __init__(self, model: GameTransformer):
        self.backbone_out: torch.Tensor | None = None
        self.num_context: int | None = None
        self._handles = [
            model.context_compiler.register_forward_hook(self._on_context),
            model.backbone.register_forward_hook(self._on_backbone),
        ]

    def _on_context(self, _module, _inputs, output):
        self.num_context = int(output.size(1))

    def _on_backbone(self, _module, _inputs, output):
        self.backbone_out = output[0] if isinstance(output, tuple) else output

    def clear(self) -> None:
        self.backbone_out = None
        self.num_context = None

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def pool_context(
    backbone_out: torch.Tensor, num_context: int,
    bounds: list[tuple[str, int, int]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return (global ctx_pool [B, d_model], per-segment pools).

    The global pool is the mean over ALL context tokens, matching
    `_team_readout`'s `ctx_pool` exactly. Per-segment pools cost one slice each and are what
    make a null result actionable: they say WHICH part of the information set is dead rather
    than only that the whole of it is.
    """
    ctx = backbone_out[:, :num_context, :]
    ctx_pool = ctx.mean(dim=1)
    seg_pools = {name: ctx[:, s:e, :].mean(dim=1) for name, s, e in bounds}
    return ctx_pool, seg_pools


def _assert_reproduces_own_head(
    model: GameTransformer, ctx_pool: torch.Tensor, out: dict[str, torch.Tensor],
) -> float:
    """THE correctness gate: the tapped tensor must BE the readout the heads consumed.

    If the hook captured the wrong tensor, or the context slice bound is off, or the pool runs
    over the wrong axis, every probe number downstream is meaningless while still looking
    perfectly plausible. Pushing the tapped `ctx_pool` back through the checkpoint's own
    `head_home_win` and comparing against the model's emitted logit is the one check that
    catches all three at once.

    Only valid on all-pregame batches, where `_team_readout` returns `ctx_pool` for every row.

    Replays the PREGAME head. Pregame and live readouts were untied into separate parameters, so
    `head_home_win` is no longer the head a prefix_length=0 row is priced by; replaying it here
    would fail this gate (measured max |delta| 2.6e-01) even though the tapped tensor is correct.
    Kept as a hard gate rather than relaxed: the point is to catch a wrong tensor, and it can only
    do that if it replays the head the model actually used.
    """
    replayed = model.head_home_win_pregame(ctx_pool).squeeze(-1)
    delta = (replayed - out["home_win_logit"]).abs().max().item()
    if not (delta < 1e-4):
        raise AssertionError(
            f"tapped ctx_pool does not reproduce the model's own home_win_logit "
            f"(max |delta| = {delta:.3e}). The cached features are NOT the readout the heads "
            f"consume, so no probe fit on them would mean anything."
        )
    return delta


class ResolvedGeometry(NamedTuple):
    d_model: int
    n_layers: int
    n_heads: int
    no_asof_weather: bool


def resolve_geometry(checkpoint, args, require: bool = False) -> tuple[ResolvedGeometry, str]:
    """Prefer the training run's own `run_config.json` over anything passed on the command line.

    WHY. `n_heads` cannot be validated against the checkpoint: MultiheadAttention's projections
    are shaped by d_model alone, so a wrong head count loads cleanly, survives the >5%-missing
    tripwire below, survives the head-reproduction gate (which only proves the right TENSOR was
    tapped inside whatever forward pass ran) -- and still changes every output. This function
    exists because the first pass at this probe read the A/B control at n_heads=12, inferred from
    an argparse default, when it had been trained at 8; the entire control column was void and
    nothing anywhere had complained.

    Searches the checkpoint's own directory and its parent, because runs write
    `run_config.json` at the run root while checkpoints land in `<run>/phase1/best.pt`.
    """
    ckpt = Path(checkpoint)
    for d in (ckpt.parent, ckpt.parent.parent):
        f = d / "run_config.json"
        if not f.is_file():
            continue
        rec = json.loads(f.read_text())
        g = ResolvedGeometry(
            d_model=int(rec["d_model"]), n_layers=int(rec["n_layers"]),
            n_heads=int(rec["n_heads"]),
            # `asof_active` is the resolved truth; `no_asof_weather` was only the request.
            no_asof_weather=not bool(rec.get("asof_active", not rec.get("no_asof_weather", False))))
        cli = ResolvedGeometry(args.d_model, args.n_layers, args.n_heads,
                               bool(getattr(args, "no_asof_weather", False)))
        if g != cli:
            # Loud on purpose: a caller whose flags were overridden must learn it here, not
            # discover it in a result table months later.
            log.warning("run_config.json at %s OVERRIDES the command line: %s -> %s", f, cli, g)
        log.info("geometry source: %s", f)
        return g, "run_config"

    msg = (f"no run_config.json beside {ckpt} or in {ckpt.parent.parent} — this checkpoint "
           "predates config persistence, so n_heads cannot be verified against it")
    if require:
        raise SystemExit(msg)
    # Not fatal, because every checkpoint trained before this landed is in exactly this state.
    # But the caller is told the value is a claim, not a fact.
    log.warning("%s; trusting --n-heads=%d UNVERIFIED — confirm it against the launcher script "
                "that trained this run, not against an argparse default", msg, args.n_heads)
    return ResolvedGeometry(args.d_model, args.n_layers, args.n_heads,
                            bool(getattr(args, "no_asof_weather", False))), "cli_unverified"


def _build_model(ds, args, device) -> GameTransformer:
    """Construct + load exactly as `score_test_predictions.py` does.

    Kept structurally identical (same geometry resolver, same prefix strip, same 5% tripwire)
    so a checkpoint that scores there loads here, and any divergence is a bug in one of the two
    rather than an open question about which is right.
    """
    # args carries the RESOLVED geometry: main() reconciles it against the run's own
    # run_config.json before any dataset is opened, so there is one source of truth here.
    rating_dim = ds.manifest.get("rating_dim", 0)
    context_config, asof_active = _resolve_weather_geometry(
        ds, True, disable_asof=args.no_asof_weather)
    log.info("geometry: d_model=%d n_layers=%d n_heads=%d rating_dim=%d weather=%dx%d "
             "asof_active=%s -> %d context tokens", args.d_model, args.n_layers, args.n_heads,
             rating_dim, context_config.weather_tokens, context_config.weather_dim,
             asof_active, context_config.total_context_tokens)

    model = GameTransformer(
        d_model=args.d_model,
        rating_dim=rating_dim,
        flat_feature_dim=30,
        context_config=context_config,
        num_backbone_layers=args.n_layers,
        num_heads=args.n_heads,
        d_ff=args.d_model * 4,
        dropout=0.0,  # features must be deterministic, so no dropout at extraction
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    # torch.compile prefixes every key with "_orig_mod."; loading such a checkpoint into a
    # bare module fails on every key at once, which reads like an architecture mismatch.
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # Loud, not fatal: some runs' checkpoints carry GameTransformerLoss.log_weights.
        log.warning("load_state_dict: %d missing, %d unexpected", len(missing), len(unexpected))
        for k in list(missing)[:10]:
            log.warning("  missing: %s", k)
        for k in list(unexpected)[:10]:
            log.warning("  unexpected: %s", k)
        if len(missing) > 0.05 * len(list(model.state_dict())):
            raise SystemExit(
                "more than 5% of parameters unmatched — refusing to extract features from a "
                "half-initialised model")
    model.eval()
    return model


@torch.no_grad()
def extract_split(model, ds, split: str, args, device) -> dict[str, np.ndarray]:
    """Cache pooled context features + labels for the `prefix_length == 0` rows of one split."""
    n_games = len(ds._game_pks)
    pregame_idx = select_pregame_indices(ds._prefix_length, ds._sample_to_game, n_games)
    log.info("%s: %d pregame rows out of %d samples (%d games)",
             split, len(pregame_idx), len(ds), n_games)

    loader = DataLoader(
        Subset(ds, pregame_idx.tolist()), batch_size=args.batch_size, shuffle=False,
        collate_fn=prepared_collate_fn, num_workers=args.num_workers)

    tap = _TrunkTap(model)
    bounds: list[tuple[str, int, int]] | None = None
    acc: dict[str, list[np.ndarray]] = {}
    worst_gate = 0.0

    try:
        for bi, batch in enumerate(loader):
            if int(batch["prefix_length"].sum()) != 0:
                raise SystemExit("non-pregame row leaked into the pregame subset")

            dev_batch = _to_device(batch, device)
            tap.clear()
            model_input = _prepare_model_input(dev_batch, player_context_dim=args.d_model * 2)
            out = model(model_input)

            if tap.backbone_out is None or tap.num_context is None:
                raise SystemExit("forward hooks did not fire — module layout changed")
            if bounds is None:
                cfg = model.context_config
                bounds = context_segment_bounds(
                    cfg, ds.manifest.get("rating_dim", 0), tap.num_context)
                log.info("context segments: %s",
                         ", ".join(f"{n}[{s}:{e})" for n, s, e in bounds))

            ctx_pool, seg_pools = pool_context(tap.backbone_out, tap.num_context, bounds)
            worst_gate = max(worst_gate, _assert_reproduces_own_head(model, ctx_pool, out))

            chunk = {"ctx_pool": ctx_pool}
            chunk.update({f"seg_{k}": v for k, v in seg_pools.items()})
            # The checkpoint's OWN head outputs, so this artifact can reproduce the recorded
            # collapse numbers and prove the extraction is scoring the same model.
            chunk["p_home_win"] = torch.sigmoid(out["home_win_logit"])
            chunk["p_yrfi"] = torch.sigmoid(out["yrfi_logit"])
            chunk["p_extra_innings"] = torch.sigmoid(out["extra_innings_logit"])
            chunk["mu_home"] = out["mu_home"]
            chunk["mu_away"] = out["mu_away"]
            for name, tensor in chunk.items():
                acc.setdefault(name, []).append(tensor.float().cpu().numpy())

            tg = batch["targets"]
            labels = {
                "game_pk": batch["game_pk"],
                "y_home_win": tg["home_win"],
                "y_yrfi": tg["yrfi"],
                "y_extra_innings": tg["extra_innings"],
                "y_total_runs": tg["total_runs"],
                "y_home_runs_remaining": tg["home_runs_remaining"],
                "y_away_runs_remaining": tg["away_runs_remaining"],
                "yrfi_mask": batch["yrfi_mask"],
            }
            for name, tensor in labels.items():
                acc.setdefault(name, []).append(
                    tensor.reshape(-1).float().cpu().numpy())

            if bi % 50 == 0:
                log.info("  %s batch %d/%d", split, bi, len(loader))
    finally:
        tap.remove()

    log.info("%s: head-reproduction gate passed, worst |delta| = %.3e", split, worst_gate)
    packed = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    packed["game_pk"] = packed["game_pk"].astype(np.int64)
    # At prefix 0 nothing has been observed, so runs-remaining IS the final total. A drift here
    # means the subset is not actually pregame.
    resid = np.abs(packed["y_home_runs_remaining"] + packed["y_away_runs_remaining"]
                   - packed["y_total_runs"]).max()
    if resid > 1e-4:
        raise SystemExit(
            f"{split}: max |remaining_sum - total_runs| = {resid:.6f}, expected 0 at prefix 0")
    return packed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True, help="destination .npz")
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-asof-weather", action="store_true")
    ap.add_argument("--require-run-config", action="store_true",
                    help="refuse to run unless the checkpoint's run_config.json is present, so "
                         "n_heads is a fact rather than a claim")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    choices=["train", "val", "test"])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s checkpoint=%s", device, args.checkpoint)

    # Reconcile geometry BEFORE opening any dataset, because --no-asof-weather also selects which
    # weather geometry the prepared tensors are read with.
    geom, geom_source = resolve_geometry(args.checkpoint, args, require=args.require_run_config)
    args.d_model, args.n_layers, args.n_heads = geom.d_model, geom.n_layers, geom.n_heads
    args.no_asof_weather = geom.no_asof_weather
    log.info("geometry source=%s -> %s", geom_source, geom)

    splits = dict(zip(("train", "val", "test"),
                      load_prepared_datasets(args.prepared_dir,
                                             disable_asof=args.no_asof_weather)))
    # Geometry comes from a split that is definitely present; all three share it.
    model = _build_model(splits[args.splits[0]], args, device)

    out: dict[str, np.ndarray] = {}
    for split in args.splits:
        packed = extract_split(model, splits[split], split, args, device)
        for k, v in packed.items():
            out[f"{split}/{k}"] = v

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out["meta/d_model"] = np.array(args.d_model)
    out["meta/n_heads"] = np.array(args.n_heads)
    out["meta/n_layers"] = np.array(args.n_layers)
    # Recorded so a result table can always be traced back to whether the head count was read
    # from the run or merely asserted by the caller.
    out["meta/geometry_source"] = np.array(geom_source)
    out["meta/checkpoint"] = np.array(str(args.checkpoint))
    out["meta/asof_disabled"] = np.array(bool(args.no_asof_weather))
    out["meta/segments"] = np.array(list(SEGMENT_ORDER))
    np.savez_compressed(outp, **out)
    log.info("wrote %s (%.1f MB)", outp, outp.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
