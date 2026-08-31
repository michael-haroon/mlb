#!/usr/bin/env python3.11
"""Compare the DL model's PREGAME predictions against the classical ensemble, same games.

WHY A DEDICATED SCRIPT: `train_unified evaluate` reports one pooled number per head over all
~15 samples per game, of which only 6.55% are pregame. The classical stack is pregame-only. A
pooled-vs-pregame subtraction measures the information set, not the model -- that is exactly
the defect that produced the retracted "+41.74% on home_win". This script pins both sides to
prefix_length == 0 and to an identical game_pk set before computing anything.

TWO CONFOUNDS THAT CANNOT BE REMOVED, ONLY STATED:

 1. TRAINING RECENCY FAVOURS CLASSICAL. `generate_loyo_splits` trains each fold on all
    STRICTLY PRIOR seasons, so a 2026 classical prediction is fit on 2015-2025. The DL train
    cut is 2024-08-03 (quantile over distinct game dates). Classical therefore carries ~1.3
    extra years of data into the same games. Both are honest out-of-sample -- neither leaks --
    but if DL loses, some of the gap is staleness rather than architecture, and if DL wins it
    wins from behind.

 2. ONLY total_runs IS JOINABLE TODAY. `train.py` writes `oof_game_pks_<target>_<tier>.npy`,
    but only the total_runs run was recent enough to have it; the other ten targets' OOF
    arrays have a different row count (32,069 vs 31,179) and no key array, so their
    predictions cannot be attached to games at all. Those rows print as BLOCKED rather than
    being silently approximated by row order -- assuming row order across two different
    populations is what caused the calibration OOF misalignment bug.

Usage:
  conda run -n pred python deep_learning/compare_pregame_vs_classical.py \
      --dl-preds /path/control_test_preds.parquet \
      --classical-models classical_learning/artifacts/models \
      --game-targets s3://.../game_targets.parquet
"""

from __future__ import annotations

import argparse
import importlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

# sys.path[0] is this file's directory (deep_learning/), not the repo root, so the
# `classical_learning` package the ensemble pickles reference is not importable without this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Targets the classical stack trains. Only those with a key array can be joined.
CLASSICAL_TARGETS = [
    "total_runs", "home_win", "yrfi", "extra_innings", "away_runs", "home_runs",
    "home_run_diff", "first_5_home_win", "first_5_total_runs", "first_5_home_run_diff",
]


def _load_classical_ensemble(models_dir: Path, target: str, tier: str = "A") -> dict | None:
    """Unpickle an ensemble saved before the pregame/ -> classical_learning/ rename.

    The pickles embed fully-qualified module paths (`pregame.strategy...`) that stopped
    resolving at commit 0031e2d. The classes themselves did not move, only the package, so
    aliasing the old name onto the new one in sys.modules is sufficient. Resolved lazily
    because pickle names missing submodules one at a time.
    """
    p = models_dir / f"ensemble_{target}_{tier}.pkl"
    if not p.exists():
        return None
    import classical_learning
    sys.modules.setdefault("pregame", classical_learning)
    for _ in range(50):
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except ModuleNotFoundError as e:
            new = e.name.replace("pregame", "classical_learning", 1)
            try:
                sys.modules[e.name] = importlib.import_module(new)
            except Exception:
                print(f"  ! cannot alias {e.name} -> {new}; ensemble unreadable")
                return None
    return None


def _classical_oof(models_dir: Path, target: str, tier: str = "A"
                   ) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """(game_pks, weighted OOF prediction, member names) or None if not joinable."""
    gpk_path = models_dir / f"oof_game_pks_{target}_{tier}.npy"
    if not gpk_path.exists():
        return None
    gpks = np.load(gpk_path)

    ens = _load_classical_ensemble(models_dir, target, tier)
    if ens is None:
        return None
    members = list(ens["members"])
    weights = np.asarray(ens["weights"], dtype=float)

    stack, kept_w, kept_m = [], [], []
    for m, w in zip(members, weights):
        f = models_dir / f"oof_{target}_{m}_{tier}.npy"
        if not f.exists():
            print(f"  ! missing member OOF {f.name}; dropping from the blend")
            continue
        a = np.load(f)
        if a.shape[0] != gpks.shape[0]:
            # Row-order alignment across differing populations is exactly the failure that
            # corrupted the CalibrationBundle isotonic fit. Refuse rather than guess.
            print(f"  ! {f.name} has {a.shape[0]} rows but the key array has "
                  f"{gpks.shape[0]} — refusing to align by position")
            return None
        stack.append(a); kept_w.append(w); kept_m.append(m)
    if not stack:
        return None
    W = np.asarray(kept_w, float)
    W = W / W.sum()
    pred = np.einsum("i,ij->j", W, np.vstack(stack))
    return gpks, pred, kept_m


def _negbin_nll(y: np.ndarray, mu: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """NegBin-2 NLL, matching mlb_dl.game_transformer.negbin_nll's parameterisation.

    alpha is the DISPERSION (variance = mu + alpha*mu^2), so r = 1/alpha. Written out rather
    than imported so this script has no torch dependency and can run on the laptop.
    """
    alpha = np.clip(alpha, 1e-6, None)
    r = 1.0 / alpha
    return -(gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
             + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))


def _reg_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    e = p - y
    return {"n": len(y), "MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e**2).mean())),
            "bias": float(e.mean())}


def _clf_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    brier = float(((p - y) ** 2).mean())
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    base = float(y.mean())
    brier_const = float(((base - y) ** 2).mean())
    # 10-bin equal-width ECE. Equal-width rather than equal-count so empty bins are visible
    # as coverage gaps instead of being silently merged.
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    ece = 0.0
    for b in range(10):
        m = idx == b
        if m.sum():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"n": len(y), "Brier": brier, "LogLoss": ll, "ECE": float(ece),
            "base_rate": base, "Brier_const": brier_const,
            "BSS_vs_const": float(1 - brier / brier_const) if brier_const > 0 else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl-preds", required=True, help="parquet from score_test_predictions.py")
    ap.add_argument("--classical-models", default="classical_learning/artifacts/models")
    ap.add_argument("--tier", default="A")
    ap.add_argument("--out", default=None, help="optional parquet of the joined per-game rows")
    args = ap.parse_args()

    models_dir = Path(args.classical_models)
    dl = pd.read_parquet(args.dl_preds)
    pre = dl[dl["prefix_length"] == 0].copy()
    if len(pre) != pre["game_pk"].nunique():
        print(f"FATAL: {len(pre)} prefix-0 rows over {pre['game_pk'].nunique()} games")
        return 1
    print(f"DL predictions: {len(dl):,} samples over {dl['game_pk'].nunique():,} games")
    print(f"  pregame (prefix_length == 0): {len(pre):,} rows "
          f"({100*len(pre)/len(dl):.2f}% of samples)\n")

    # ---- 1. How the DL heads decay with prefix length -----------------------
    # This is the decomposition no pooled metric can show, and the reason a pooled DL number
    # cannot be compared to a pregame classical number.
    print("=" * 78)
    print("DL heads BY PREFIX BUCKET (the pooled metric is a weighted mix of these rows)")
    print("=" * 78)
    buckets = [("pregame (0)", dl["prefix_length"] == 0),
               ("live <75", (dl["prefix_length"] > 0) & (dl["prefix_length"] < 75)),
               ("live 75-199", (dl["prefix_length"] >= 75) & (dl["prefix_length"] < 200)),
               ("live >=200", dl["prefix_length"] >= 200)]
    print(f"{'bucket':14s}{'n':>8}{'HW Brier':>10}{'HW BSS':>9}{'XI Brier':>10}"
          f"{'runs MAE':>10}{'runs NLL':>10}")
    for lab, m in buckets:
        d = dl[m]
        hw = _clf_metrics(d["y_home_win"].to_numpy(float), d["p_home_win"].to_numpy(float))
        xi = _clf_metrics(d["y_extra_innings"].to_numpy(float),
                          d["p_extra_innings"].to_numpy(float))
        yrem = (d["y_home_runs_remaining"] + d["y_away_runs_remaining"]).to_numpy(float)
        mrem = (d["mu_home"] + d["mu_away"]).to_numpy(float)
        nll = _negbin_nll(d["y_home_runs_remaining"].to_numpy(float),
                          d["mu_home"].to_numpy(float),
                          d["alpha_home"].to_numpy(float)).mean()
        print(f"{lab:14s}{len(d):>8,}{hw['Brier']:>10.5f}{hw['BSS_vs_const']:>9.4f}"
              f"{xi['Brier']:>10.5f}{np.abs(mrem-yrem).mean():>10.4f}{nll:>10.4f}")
    print()

    # ---- 2. Pregame head-by-head vs classical ------------------------------
    print("=" * 78)
    print("PREGAME vs CLASSICAL, identical games")
    print("=" * 78)
    joined = pre[["game_pk"]].copy()
    any_ok = False

    # total_runs: DL E[total] = mu_home + mu_away; classical predicts the total directly.
    cl = _classical_oof(models_dir, "total_runs", args.tier)
    if cl is None:
        print("total_runs   BLOCKED: no usable oof_game_pks_total_runs_"
              f"{args.tier}.npy + member OOF set")
    else:
        gpks, cpred, members = cl
        cdf = pd.DataFrame({"game_pk": gpks, "cl_total_runs": cpred}).dropna()
        j = pre.merge(cdf, on="game_pk", how="inner")
        y = j["y_total_runs"].to_numpy(float)
        dlp = (j["mu_home"] + j["mu_away"]).to_numpy(float)
        clp = j["cl_total_runs"].to_numpy(float)
        # Naive reference: predict the mean of the compared games. Any model that cannot beat
        # this has no point-forecast skill on this slice at all.
        naive = np.full_like(y, y.mean())
        print(f"total_runs   n={len(j):,} games "
              f"({100*len(j)/len(pre):.1f}% of the DL test split); "
              f"classical members={members}")
        for name, p in [("  DL pregame", dlp), ("  classical", clp), ("  slice mean", naive)]:
            m = _reg_metrics(y, p)
            print(f"{name:14s} MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  bias={m['bias']:+.4f}")
        d_mae = _reg_metrics(y, dlp)["MAE"] - _reg_metrics(y, clp)["MAE"]
        print(f"  -> DL minus classical MAE: {d_mae:+.4f} "
              f"({'DL better' if d_mae < 0 else 'classical better'})")
        # Paired test: the two models score the SAME games, so the per-game absolute-error
        # difference is paired and a one-sample t on that difference is the right screen.
        # Assumption: the mean of ~2k paired differences is approximately normal by CLT. The
        # differences themselves are NOT normal (heavy right tail on |error|), which is why
        # the test is on the mean and not on the raw errors.
        diff = np.abs(dlp - y) - np.abs(clp - y)
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        print(f"  paired mean |err| diff = {diff.mean():+.4f} +/- {1.96*se:.4f} (95% CI), "
              f"t={diff.mean()/se:+.2f}")
        joined = joined.merge(j[["game_pk", "y_total_runs", "cl_total_runs"]],
                              on="game_pk", how="left")
        joined["dl_total_runs"] = pre.merge(
            j[["game_pk"]], on="game_pk", how="left").assign(
            v=(pre["mu_home"] + pre["mu_away"]).to_numpy())["v"].to_numpy()
        any_ok = True

    for t in [x for x in CLASSICAL_TARGETS if x != "total_runs"]:
        if not (models_dir / f"oof_game_pks_{t}_{args.tier}.npy").exists():
            print(f"{t:12s} BLOCKED: no oof_game_pks_{t}_{args.tier}.npy — the OOF "
                  f"predictions exist but cannot be attached to games")

    # ---- 3. DL pregame standalone, for every head --------------------------
    print()
    print("=" * 78)
    print("DL PREGAME standalone (no classical counterpart joinable yet)")
    print("=" * 78)
    for lab, ycol, pcol in [("home_win", "y_home_win", "p_home_win"),
                            ("yrfi", "y_yrfi", "p_yrfi"),
                            ("extra_innings", "y_extra_innings", "p_extra_innings")]:
        sub = pre
        if lab == "yrfi":
            # yrfi_mask is 1.0 at prefix 0 by construction, but assert rather than assume:
            # a future sampler change that emits a prefix past the 1st inning as "pregame"
            # would silently score a masked-out head.
            assert (pre["yrfi_mask"] > 0.5).all(), "yrfi_mask must be 1 at prefix 0"
        m = _clf_metrics(sub[ycol].to_numpy(float), sub[pcol].to_numpy(float))
        print(f"{lab:16s} n={m['n']:,}  Brier={m['Brier']:.5f}  LogLoss={m['LogLoss']:.5f}  "
              f"ECE={m['ECE']:.5f}  base={m['base_rate']:.4f}  BSS={m['BSS_vs_const']:+.4f}")

    if args.out and any_ok:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        joined.to_parquet(args.out, index=False)
        print(f"\nwrote joined per-game rows: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
