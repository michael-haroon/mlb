"""Per-head calibration vs discrimination diagnosis for a GameTransformer checkpoint.

Motivation (2026-08-30): the control run's held-out metrics showed
`player_hr_brier` 0.2069 against a base rate of 0.1159 with a predicted mean of
0.4174. A constant predictor that always quotes the base rate scores
p(1-p) = 0.1025, so the HR head is TWICE as bad as quoting a constant — and
`player_sb_brier` 0.1524 vs a 0.0635 base rate (constant 0.0595) is the same
story. For a market maker that is disqualifying: you cannot quote a price from a
probability that is 3.6x too high.

The cause is not a coding error. `GameTransformerLoss._focal_bce` uses
alpha=0.75 on positives and gamma=2 (Lin et al. 2017), which optimizes a
DELIBERATELY REWEIGHTED distribution. Focal loss is not a proper scoring rule,
so its minimizer is not the true conditional probability — inflated scores are
the designed behaviour, and it is the right choice for detection/ranking and the
wrong one for quoting.

That leaves one question that changes the remedy completely:

  * high AUC + bad Brier  -> the head LEARNED the signal and only the scale is
    wrong. Remedy is a post-hoc calibration map, cheap and no retrain.
  * AUC ~ 0.5             -> the head learned nothing and recalibration cannot
    manufacture information. Remedy is architectural.

So this script reports, per binary head: base rate, predicted mean, Brier, the
constant-predictor Brier, Brier Skill Score against that constant, AUC, and the
Brier after a Platt map FIT ON VAL AND APPLIED TO TEST. Fitting on val is what
makes the recalibrated number honest — a map fit on test would be leakage and
would flatter every head.

Usage (on the training box, with prepared tensors):
  python -m diagnose_head_calibration --checkpoint OUT/final_model.pt \
      --prepared-dir /mnt/fast/prepared_tensors --d-model 384 --n-layers 6 \
      --n-heads 12 --output OUT/head_calibration.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mlb_dl.game_transformer import GameTransformer
from mlb_dl.train_unified import (
    _player_valid,
    _prepare_model_input,
    _resolve_weather_geometry,
    _to_device,
)

log = logging.getLogger("head_cal")

# (name, prediction key, target key, pred is already a probability, per-player,
#  target is a count needing binarisation)
# Two traps, both hit in practice:
#  * per-player heads must be masked by player_mask, not a target sentinel —
#    padding slots are stored as 0, so `y >= 0` admits every one of them;
#  * player_hr/player_sb targets are COUNTS (max 4 / 3) while the heads predict
#    P(1+ event), so they must be binarised or Brier and its p(1-p) baseline are
#    both meaningless.
BINARY_HEADS = [
    ("home_win", "home_win_logit", "home_win", False, False, False),
    ("yrfi", "yrfi_logit", "yrfi", False, False, False),
    ("extra_innings", "extra_innings_logit", "extra_innings", False, False, False),
    ("player_hr", "hr_prob", "player_hr", True, True, True),
    ("player_sb", "stolen_bases_logit", "player_sb", False, True, True),
]

_EPS = 1e-6


def _to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _collect(model, loader, device, player_context_dim) -> dict[str, np.ndarray]:
    """Raw per-sample scores and targets for every binary head."""
    model.eval()
    want_pred = {k for _, k, _, _, _, _ in BINARY_HEADS}
    want_tgt = {k for _, _, k, _, _, _ in BINARY_HEADS}
    acc: dict[str, list] = {}
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            preds = model(_prepare_model_input(batch, player_context_dim=player_context_dim))
            for k in want_pred:
                if k in preds:
                    acc.setdefault(f"p::{k}", []).append(preds[k].detach().float().cpu().numpy().ravel())
            for k in want_tgt:
                if k in batch["targets"]:
                    acc.setdefault(f"t::{k}", []).append(
                        batch["targets"][k].detach().float().cpu().numpy().ravel())
            if batch.get("player_mask") is not None:
                acc.setdefault("player_mask", []).append(
                    batch["player_mask"].detach().float().cpu().numpy().ravel())
    return {k: np.concatenate(v) for k, v in acc.items()}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Overflow-free logistic. np.exp(-x) overflows for x < -745, and an
    undamped Newton step can easily propose coefficients that reach there."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.clip(x, 0, None))),
                    np.exp(np.clip(x, None, 0)) / (1.0 + np.exp(np.clip(x, None, 0))))


def _neg_loglik(a: float, b: float, z: np.ndarray, y: np.ndarray) -> float:
    """-log L via logaddexp, which is exact for large |t| where log(1+e^t)
    would overflow."""
    t = a * z + b
    return float(np.sum(np.logaddexp(0.0, t) - y * t))


def _platt(z_fit: np.ndarray, y_fit: np.ndarray, z_apply: np.ndarray) -> np.ndarray:
    """2-parameter logistic map (Platt 1999) fit by DAMPED Newton.

    Hand-rolled rather than sklearn so the diagnosis carries no dependency the
    training image might lack. The damping is not decoration: undamped Newton on
    a well-separated score overshoots into the flat tails, where the Hessian is
    ~0 and the next step is enormous. Measured on synthetic focal-style inflation
    it diverged and produced a Brier WORSE than the uncalibrated score (skill
    -1.17), i.e. it would have reported "recalibration cannot help" for a head
    that recalibrates fine. Backtracking on the actual objective makes each step
    a guaranteed improvement.
    """
    a, b = 1.0, 0.0
    f = _neg_loglik(a, b, z_fit, y_fit)
    for _ in range(200):
        p = _sigmoid(a * z_fit + b)
        w = np.clip(p * (1 - p), 1e-12, None)
        r = y_fit - p
        g = np.array([np.dot(r, z_fit), r.sum()])                  # d(logL)/d(a,b)
        H = np.array([[np.dot(w * z_fit, z_fit), np.dot(w, z_fit)],
                      [np.dot(w, z_fit), w.sum()]])                # -d2(logL)
        H[0, 0] += 1e-9
        H[1, 1] += 1e-9
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        for _ in range(60):
            na, nb = a + t * step[0], b + t * step[1]
            nf = _neg_loglik(na, nb, z_fit, y_fit)
            if np.isfinite(nf) and nf <= f:
                break
            t *= 0.5
        else:
            break
        moved = abs(na - a) + abs(nb - b)
        a, b, f = na, nb, nf
        if moved < 1e-10:
            break
    return _sigmoid(a * z_apply + b)


def _isotonic(z_fit: np.ndarray, y_fit: np.ndarray, z_apply: np.ndarray) -> np.ndarray:
    """Non-parametric monotone calibration by pool-adjacent-violators.

    Platt is linear in the logit, so it cannot repair a distortion that varies
    with p — and focal loss with gamma=2 produces exactly that, because the
    (1-p_t)^gamma factor reweights differently at every probability level. The
    isotonic map is the monotone-optimal recalibration, so it is the decisive
    test: by the Brier calibration-refinement decomposition, a score with any
    discrimination at all must score no worse than the constant baseline once
    isotonically calibrated. If isotonic STILL loses to the constant, the problem
    is not calibration.
    """
    order = np.argsort(z_fit, kind="mergesort")
    x, y = z_fit[order], y_fit[order].astype(np.float64)
    # Pool EQUAL scores into one block before PAVA. Isotonic regression is only
    # unique on distinct x; leaving ties as separate blocks lets them land in
    # different blocks with different means, and the step lookup then returns an
    # arbitrary one of them. Not hypothetical: hr_prob saturates, so _to_logit
    # clips whole runs to the same +/-13.8, and a 4-point tied case resolved to
    # {0, 0, 1} instead of the correct constant 0.5.
    bp, starts = np.unique(x, return_index=True)
    sums = np.add.reduceat(y, starts)
    cnts = np.diff(np.append(starts, len(y))).astype(np.float64)
    # PAVA over (sum, count) blocks; merge backwards while means decrease.
    means: list[float] = []
    weights: list[float] = []
    edges: list[float] = []
    for s, w, xv in zip(sums, cnts, bp):
        means.append(s / w)
        weights.append(w)
        edges.append(xv)
        while len(means) > 1 and means[-2] > means[-1]:
            m2, w2 = means.pop(), weights.pop()
            e2 = edges.pop()
            m1, w1 = means.pop(), weights.pop()
            edges.pop()
            wn = w1 + w2
            means.append((m1 * w1 + m2 * w2) / wn)
            weights.append(wn)
            edges.append(e2)          # block ends at the later score
    bp = np.asarray(edges)
    mv = np.clip(np.asarray(means), _EPS, 1 - _EPS)
    # Right-continuous step lookup: each test score takes its block's mean.
    idx = np.searchsorted(bp, z_apply, side="left")
    idx = np.clip(idx, 0, len(mv) - 1)
    return mv[idx]


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U), tie-corrected via average ranks.

    Calibration-invariant by construction: any monotone rescaling of `s` leaves
    it unchanged, which is exactly why it separates "wrong scale" from
    "no signal".
    """
    pos, neg = y == 1, y == 0
    n_p, n_n = int(pos.sum()), int(neg.sum())
    if n_p == 0 or n_n == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[pos].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n)


def analyse(val: dict, test: dict) -> dict:
    out = {}
    for name, pkey, tkey, is_prob, is_player, is_count in BINARY_HEADS:
        pk, tk = f"p::{pkey}", f"t::{tkey}"
        if pk not in test or tk not in test:
            continue
        y_te, s_te = test[tk], test[pk]
        n = min(len(y_te), len(s_te))
        y_te, s_te = y_te[:n], s_te[:n]
        # Mask exactly as the loss did. For player heads that means player_mask;
        # the target sentinel alone is a no-op because padding is stored as 0.
        m = _player_valid(y_te, test.get("player_mask")[:n] if
                          (is_player and test.get("player_mask") is not None) else None)
        y_te, s_te = y_te[m], s_te[m]
        if is_count:
            y_te = (y_te > 0).astype(np.float64)
        if len(y_te) == 0 or y_te.max() == y_te.min():
            continue
        p_te = s_te if is_prob else _sigmoid(s_te)
        z_te = _to_logit(p_te) if is_prob else s_te

        base = float(y_te.mean())
        brier = float(np.mean((p_te - y_te) ** 2))
        const = base * (1 - base)
        rec = {
            "n": int(len(y_te)),
            "base_rate": round(base, 5),
            "pred_mean": round(float(p_te.mean()), 5),
            "pred_mean_over_base": round(float(p_te.mean()) / base, 3) if base > 0 else None,
            "brier": round(brier, 5),
            "brier_constant_baseline": round(const, 5),
            "bss_vs_constant": round((const - brier) / const, 4) if const > 0 else None,
            "auc": round(_auc(y_te, s_te), 4),
        }
        # Recalibrate only if val carries the same head, so the map never sees test.
        if pk in val and tk in val:
            y_va, s_va = val[tk], val[pk]
            k = min(len(y_va), len(s_va))
            y_va, s_va = y_va[:k], s_va[:k]
            mv = _player_valid(y_va, val.get("player_mask")[:k] if
                               (is_player and val.get("player_mask") is not None) else None)
            y_va, s_va = y_va[mv], s_va[mv]
            if is_count:
                y_va = (y_va > 0).astype(np.float64)
            if len(y_va) and y_va.max() > y_va.min():
                z_va = _to_logit(s_va) if is_prob else s_va
                p_cal = _platt(z_va, y_va, z_te)
                b_cal = float(np.mean((p_cal - y_te) ** 2))
                rec["brier_platt_val_fit"] = round(b_cal, 5)
                rec["bss_after_platt"] = round((const - b_cal) / const, 4) if const > 0 else None
                rec["pred_mean_after_platt"] = round(float(p_cal.mean()), 5)
                p_iso = _isotonic(z_va, y_va, z_te)
                b_iso = float(np.mean((p_iso - y_te) ** 2))
                rec["brier_isotonic_val_fit"] = round(b_iso, 5)
                rec["bss_after_isotonic"] = round((const - b_iso) / const, 4) if const > 0 else None
                rec["pred_mean_after_isotonic"] = round(float(p_iso.mean()), 5)
        # The verdict is what makes this actionable: it names the remedy. Isotonic
        # is the monotone optimum, so it — not Platt — decides whether the
        # failure is calibration or something deeper.
        best = rec.get("bss_after_isotonic", rec.get("bss_after_platt", rec["bss_vs_constant"]))
        if not np.isfinite(rec["auc"]):
            rec["verdict"] = "UNSCORABLE"
        elif rec["auc"] < 0.55:
            rec["verdict"] = "NO SIGNAL — recalibration cannot help; architectural"
        elif best is not None and best <= 0:
            rec["verdict"] = ("RANKS BUT NO MONOTONE MAP BEATS CONSTANT — "
                              "suspect val/test shift or a broken valid mask")
        elif rec["bss_vs_constant"] <= 0:
            rec["verdict"] = "MISCALIBRATED ONLY — post-hoc monotone map recovers skill"
        else:
            rec["verdict"] = "OK"
        out[name] = rec
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prepared-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--dump-scores", action="store_true",
                    help="Also save raw val/test scores as .scores.npz")
    args = ap.parse_args()

    from mlb_dl.precollate import load_prepared_datasets, prepared_collate_fn

    _, val_ds, test_ds = load_prepared_datasets(args.prepared_dir)
    rating_dim = test_ds.manifest.get("rating_dim", 0)
    ctx, asof = _resolve_weather_geometry(test_ds, use_prepared=True)
    log.info("val=%d test=%d rating_dim=%d asof=%s",
             len(val_ds), len(test_ds), rating_dim, asof)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GameTransformer(
        d_model=args.d_model, rating_dim=rating_dim, flat_feature_dim=30,
        context_config=ctx, num_backbone_layers=args.n_layers,
        num_heads=args.n_heads, d_ff=args.d_model * 4, dropout=0.0,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)

    def loader(ds):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          collate_fn=prepared_collate_fn,
                          num_workers=args.num_workers, pin_memory=True)

    pcd = args.d_model * 2
    log.info("collecting val scores")
    val = _collect(model, loader(val_ds), device, pcd)
    log.info("collecting test scores")
    test = _collect(model, loader(test_ds), device, pcd)

    report = analyse(val, test)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    # Persist raw scores so any follow-up calibration question is answered
    # offline: the GPU forward pass is the expensive part, the analysis is not.
    if args.dump_scores:
        np.savez_compressed(out.with_suffix(".scores.npz"),
                            **{f"val_{k}": v for k, v in val.items()},
                            **{f"test_{k}": v for k, v in test.items()})
        log.info("raw scores -> %s", out.with_suffix(".scores.npz"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
