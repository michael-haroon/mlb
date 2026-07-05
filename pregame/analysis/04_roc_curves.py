#!/usr/bin/env python3
"""ROC curve visualizer for all classification targets.

Produces one 2-panel figure per target:
  Left  — standard ROC (TPR vs FPR), one curve per model family + ensemble blend
  Right — TPR and FPR as functions of decision threshold

Usage:
    conda run -n pred python -m pregame.analysis.04_roc_curves \
        --models /tmp/mlb_artifacts/models \
        --features /tmp/mlb_artifacts/features/game_features.parquet \
        --out /tmp/roc_all.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pregame.strategy.config import TARGETS_CLASSIFICATION
from pregame.strategy.ensemble import load_ensemble_oof

PALETTE = [
    "#4c78a8", "#f58518", "#e45756", "#72b7b2",
    "#54a24b", "#b279a2", "#ff9da6", "#9d755d",
    "#bab0ac", "#eeca3b", "#1f77b4", "#d62728",
    "#9467bd", "#8c564b",
]
ENSEMBLE_COLOUR = "#000000"


def _load_target(models_dir: Path, features_path: Path, target: str, tier: str):
    df = pd.read_parquet(features_path, columns=[target, "season"])
    y_series = df[target].dropna()
    y_true = y_series.values

    oof_matrix = load_ensemble_oof(models_dir, target, tier)
    if not oof_matrix:
        raise FileNotFoundError(f"No OOF files for {target}/{tier}")

    n = len(y_true)
    aligned = {}
    for fam, arr in oof_matrix.items():
        aligned[fam] = arr[:n] if len(arr) >= n else np.pad(arr, (0, n - len(arr)), constant_values=np.nan)

    no2020 = (df.loc[y_series.index, "season"] != 2020).values
    y_true = y_true[no2020]
    aligned = {fam: arr[no2020] for fam, arr in aligned.items()}

    # Load ensemble pickle to recover member weights and add a blend curve
    import pickle
    pkl_path = models_dir / f"ensemble_{target}_{tier}.pkl"
    ensemble_curve = None
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)
        members = bundle.get("members", [])
        weights = np.array(bundle.get("weights", []))
        member_bundles = {mb["family"]: mb for mb in bundle.get("member_bundles", [])}

        valid_mask = ~np.isnan(y_true)
        for m in members:
            if m in aligned:
                valid_mask &= ~np.isnan(aligned[m])

        if valid_mask.sum() >= 50 and all(m in aligned for m in members):
            cols = []
            for m in members:
                col = aligned[m][valid_mask]
                iso = member_bundles.get(m, {}).get("isotonic_calibrator")
                if iso is not None:
                    col = iso.predict(col)
                cols.append(col)
            blend = np.column_stack(cols) @ weights
            ensemble_curve = (blend, y_true[valid_mask])

    return aligned, y_true, ensemble_curve


def _plot_one_target(
    models_dir: Path, features_path: Path, target: str, tier: str, ax_roc, ax_fp
):
    try:
        oof, y_true, ensemble_curve = _load_target(models_dir, features_path, target, tier)
    except FileNotFoundError as e:
        ax_roc.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax_roc.transAxes, fontsize=8)
        return

    # Sort families by AUC descending (after inversion) so the best models are first in legend
    def _auc(arr):
        valid = ~np.isnan(y_true) & ~np.isnan(arr)
        if valid.sum() < 30:
            return 0.0
        a = roc_auc_score(y_true[valid], arr[valid])
        return max(a, 1.0 - a)  # invert if needed

    sorted_families = sorted(oof.items(), key=lambda kv: _auc(kv[1]), reverse=True)

    legend_handles = []

    for i, (name, scores) in enumerate(sorted_families):
        colour = PALETTE[i % len(PALETTE)]
        valid = ~np.isnan(y_true) & ~np.isnan(scores)
        if valid.sum() < 30:
            continue
        yv, sv = y_true[valid], scores[valid]
        raw_auc = roc_auc_score(yv, sv)
        inverted = raw_auc < 0.5
        if inverted:
            sv = 1.0 - sv
            raw_auc = 1.0 - raw_auc

        fpr, tpr, thresholds = roc_curve(yv, sv)
        label = f"{name}{'*' if inverted else ''}  (AUC={raw_auc:.3f})"

        ax_roc.plot(fpr, tpr, color=colour, lw=1.4, alpha=0.75)
        if len(thresholds) > 1:
            ax_fp.plot(thresholds[1:], tpr[1:], color=colour, lw=1.4, linestyle="-", alpha=0.7)
            ax_fp.plot(thresholds[1:], fpr[1:], color=colour, lw=1.4, linestyle="--", alpha=0.5)

        legend_handles.append(mpatches.Patch(color=colour, label=label))

    # Ensemble blend curve on top
    if ensemble_curve is not None:
        blend, yt = ensemble_curve
        ens_auc = roc_auc_score(yt, blend)
        if ens_auc < 0.5:
            blend = 1.0 - blend
            ens_auc = 1.0 - ens_auc
        fpr_e, tpr_e, thr_e = roc_curve(yt, blend)
        ax_roc.plot(fpr_e, tpr_e, color=ENSEMBLE_COLOUR, lw=2.5, alpha=0.95, zorder=10)
        if len(thr_e) > 1:
            ax_fp.plot(thr_e[1:], tpr_e[1:], color=ENSEMBLE_COLOUR, lw=2.5, linestyle="-", alpha=0.9, zorder=10)
            ax_fp.plot(thr_e[1:], fpr_e[1:], color=ENSEMBLE_COLOUR, lw=2.5, linestyle="--", alpha=0.7, zorder=10)
        legend_handles.insert(0, mpatches.Patch(
            color=ENSEMBLE_COLOUR,
            label=f"ENSEMBLE  (AUC={ens_auc:.3f})",
        ))

    # Diagonal + iso-AUC reference lines
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.7, label="Coin-flip (AUC=0.50)")
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.set_aspect("equal")
    ax_roc.set_xlabel("False Positive Rate", fontsize=9)
    ax_roc.set_ylabel("True Positive Rate", fontsize=9)
    ax_roc.set_title(f"{target}  |  ROC (OOF, no 2020)", fontsize=10, fontweight="bold")
    ax_roc.legend(handles=legend_handles, loc="lower right", fontsize=6.5,
                  framealpha=0.9, title="* = scores inverted", title_fontsize=6)

    ax_fp.set_xlim(0, 1)
    ax_fp.set_ylim(-0.02, 1.02)
    ax_fp.set_xlabel("Decision Threshold  (score ≥ θ)", fontsize=9)
    ax_fp.set_ylabel("Rate", fontsize=9)
    ax_fp.set_title(f"{target}  |  TPR (—) & FPR (--) vs θ", fontsize=10, fontweight="bold")
    solid = plt.Line2D([0], [0], color="grey", lw=1.5, linestyle="-", label="TPR")
    dashed = plt.Line2D([0], [0], color="grey", lw=1.5, linestyle="--", label="FPR")
    ax_fp.legend(handles=[solid, dashed], loc="upper right", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="/tmp/mlb_artifacts/models")
    ap.add_argument("--features", default="/tmp/mlb_artifacts/features/game_features.parquet")
    ap.add_argument("--tier", default="A", choices=["A", "B", "C"])
    ap.add_argument("--out", default="/tmp/roc_all.png")
    args = ap.parse_args()

    models_dir = Path(args.models)
    features_path = Path(args.features)
    targets = TARGETS_CLASSIFICATION  # ["home_win", "yrfi", "first_5_home_win", "extra_innings"]

    n = len(targets)
    fig, axes = plt.subplots(n, 2, figsize=(14, 6 * n))
    fig.suptitle("ROC Curves — all classification targets  (OOF, tier A, 2015-2025 excl. 2020)",
                 fontsize=13, fontweight="bold", y=1.005)

    for row, target in enumerate(targets):
        _plot_one_target(models_dir, features_path, target, args.tier,
                         axes[row, 0], axes[row, 1])

    plt.tight_layout()
    out = Path(args.out)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
