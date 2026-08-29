"""Calibration analysis: reliability diagrams, ECE, Brier decomposition, confidence tiers."""
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Resolve project root so this runs as `python -m pregame.analysis.calibration_analysis`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "pregame" / "artifacts" / "models"
FEATURES_PATH = PROJECT_ROOT / "pregame" / "artifacts" / "features" / "game_features.parquet"
OUT_DIR = PROJECT_ROOT / "pregame" / "analysis"
PDF_PATH = OUT_DIR / "calibration_report.pdf"
TIER = "A"

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"LOW": "#e05c5c", "MEDIUM": "#f0a030", "HIGH": "#4a90d9"}


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_ece(y_true, y_pred, n_bins=12):
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(y_true)
    ece = 0.0
    bin_stats = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_pred >= lo) & (y_pred < hi)
        if i == n_bins - 1:
            mask = (y_pred >= lo) & (y_pred <= hi)
        n = mask.sum()
        if n == 0:
            bin_stats.append({"mid": (lo + hi) / 2, "actual": np.nan, "pred": np.nan, "n": 0, "err": 0.0})
            continue
        actual = y_true[mask].mean()
        pred = y_pred[mask].mean()
        err = pred - actual  # signed: positive = overconfident
        ece += n / total * abs(err)
        bin_stats.append({"mid": (lo + hi) / 2, "actual": actual, "pred": pred, "n": n, "err": err})
    return ece, bin_stats


def brier_decomposition(y_true, y_pred):
    """Murphy (1973) decomposition: BS = reliability - resolution + uncertainty."""
    n = len(y_true)
    bs = np.mean((y_pred - y_true) ** 2)
    clim = y_true.mean()
    bs_clim = clim * (1 - clim)  # uncertainty term

    # Bin-wise reliability and resolution
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    reliability = 0.0
    resolution = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_pred >= lo) & (y_pred < hi) if i < n_bins - 1 else (y_pred >= lo) & (y_pred <= hi)
        if mask.sum() == 0:
            continue
        nk = mask.sum()
        ok = y_true[mask].mean()
        fk = y_pred[mask].mean()
        reliability += nk / n * (fk - ok) ** 2
        resolution += nk / n * (ok - clim) ** 2

    bss = 1.0 - bs / bs_clim if bs_clim > 0 else 0.0
    return {"brier": bs, "reliability": reliability, "resolution": resolution,
            "uncertainty": bs_clim, "bss": bss}


# ── data loading ──────────────────────────────────────────────────────────────

def load_target_data(target):
    """Return (y_true, oof_raw_dict, weights, bundle) or None on failure."""
    pkl_path = MODELS_DIR / f"ensemble_{target}_{TIER}.pkl"
    if not pkl_path.exists():
        print(f"  [SKIP] {pkl_path.name} not found")
        return None

    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)

    task = bundle.get("task", "classification")
    if task != "classification":
        print(f"  [SKIP] {target}: task={task} (regression not analyzed here)")
        return None

    cal = bundle.get("calibration")
    member_bundles = bundle.get("member_bundles", [])
    weights = np.array(bundle.get("weights", []))

    # Normalize weights in case of floating-point drift
    if weights.sum() > 0:
        weights = weights / weights.sum()

    # Load per-model OOF arrays
    oof_dict = {}
    for mb in member_bundles:
        family = mb["family"]
        oof_path = MODELS_DIR / f"oof_{target}_{family}_{TIER}.npy"
        if not oof_path.exists():
            print(f"  [WARN] Missing OOF: {oof_path.name}")
            continue
        oof_dict[family] = np.load(oof_path)

    if not oof_dict:
        print(f"  [SKIP] {target}: no OOF arrays found")
        return None

    # Load y_true from features parquet — same logic as cli.py _run_ensemble
    if not FEATURES_PATH.exists():
        print(f"  [SKIP] Features parquet not found: {FEATURES_PATH}")
        return None

    df = pd.read_parquet(FEATURES_PATH, columns=[target, "season"])
    y_series = df[target].dropna()
    y_all = y_series.values

    n = len(y_all)
    # Align OOF arrays to y length (same padding as cli.py)
    aligned_oof = {}
    families_in_bundle = [mb["family"] for mb in member_bundles if mb["family"] in oof_dict]
    for fam in families_in_bundle:
        arr = oof_dict[fam]
        aligned = arr[-n:] if len(arr) >= n else np.pad(arr, (0, n - len(arr)), constant_values=np.nan)
        aligned_oof[fam] = aligned

    # Exclude 2020 (structural outlier — see cli.py comment for rationale)
    no2020 = (df.loc[y_series.index, "season"] != 2020).values
    y_true = y_all[no2020]
    aligned_oof = {fam: arr[no2020] for fam, arr in aligned_oof.items()}

    # Weights are aligned to member_bundles order
    fam_to_weight = {mb["family"]: w for mb, w in zip(member_bundles, weights)
                     if mb["family"] in aligned_oof}
    used_fams = list(fam_to_weight.keys())
    used_weights = np.array([fam_to_weight[f] for f in used_fams])
    if used_weights.sum() > 0:
        used_weights /= used_weights.sum()

    # Build valid-row mask (no NaN in y or any used OOF)
    valid = ~np.isnan(y_true)
    for fam in used_fams:
        valid &= ~np.isnan(aligned_oof[fam])

    y_v = y_true[valid]
    oof_matrix = np.column_stack([aligned_oof[f][valid] for f in used_fams])  # (n, k)

    # Apply per-model isotonic calibrators (mirrors cli.py calibration step)
    mb_map = {mb["family"]: mb for mb in member_bundles}
    cal_matrix = np.empty_like(oof_matrix)
    for i, fam in enumerate(used_fams):
        iso = mb_map.get(fam, {}).get("isotonic_calibrator")
        if iso is not None:
            cal_matrix[:, i] = iso.predict(np.clip(oof_matrix[:, i], 0.01, 0.99))
        else:
            cal_matrix[:, i] = oof_matrix[:, i]

    # Ensemble OOF: weighted mean of calibrated per-model predictions
    oof_blend = cal_matrix @ used_weights
    oof_std = cal_matrix.std(axis=1)

    # Final calibration via CalibrationBundle isotonic (if present)
    oof_calibrated = oof_blend.copy()
    if cal is not None and cal.isotonic is not None:
        oof_calibrated = cal.isotonic.predict(np.clip(oof_blend, 0.01, 0.99))

    return {
        "target": target,
        "y_true": y_v,
        "oof_raw": oof_blend,
        "oof_calibrated": oof_calibrated,
        "oof_std": oof_std,
        "cal": cal,
        "n": int(valid.sum()),
        "families": used_fams,
        "weights": used_weights,
    }


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_reliability(ax, y_true, y_pred, title, n_bins=12):
    ece, bin_stats = compute_ece(y_true, y_pred, n_bins)
    mids = [b["mid"] for b in bin_stats]
    actuals = [b["actual"] for b in bin_stats]
    counts = [b["n"] for b in bin_stats]

    valid_bins = [(m, a, c) for m, a, c in zip(mids, actuals, counts) if not np.isnan(a)]
    if not valid_bins:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return ece, bin_stats

    xs, ys, ns = zip(*valid_bins)
    max_n = max(ns)
    widths = [0.07 * (n / max_n) ** 0.5 for n in ns]

    for x, y, w in zip(xs, ys, widths):
        ax.bar(x, y, width=w, color="#4a90d9", alpha=0.6, align="center")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax.plot(xs, ys, "o-", color="#4a90d9", ms=5, lw=1.5, label="Model")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=10)
    ax.legend(fontsize=8)
    return ece, bin_stats


def plot_signed_ece(ax, bin_stats, title):
    valid = [(b["mid"], b["err"], b["n"]) for b in bin_stats if not np.isnan(b["err"])]
    if not valid:
        return
    xs, errs, ns = zip(*valid)
    colors = ["#e05c5c" if e > 0 else "#4a90d9" for e in errs]
    ax.bar(xs, errs, width=0.06, color=colors, alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Predicted probability bin"); ax.set_ylabel("Signed error (pred − actual)")
    ax.set_title(f"{title}\n+ = overconfident, − = underconfident", fontsize=10)


def plot_tiers(ax, y_true, y_pred, oof_std, cal, title, n_bins=10):
    """Separate reliability curves per confidence tier."""
    if cal is None or cal.std_p33 is None:
        ax.text(0.5, 0.5, "no tier info", ha="center", va="center", transform=ax.transAxes)
        return {}

    p33, p67 = cal.std_p33, cal.std_p67
    # HIGH confidence = LOW std (ensemble agrees)
    tier_masks = {
        "HIGH": oof_std <= p33,
        "MEDIUM": (oof_std > p33) & (oof_std <= p67),
        "LOW": oof_std > p67,
    }

    tier_eces = {}
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    for tier, mask in tier_masks.items():
        if mask.sum() < 30:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        ece, bin_stats = compute_ece(yt, yp, n_bins)
        valid_bins = [(b["mid"], b["actual"]) for b in bin_stats if not np.isnan(b["actual"])]
        if not valid_bins:
            continue
        xs, ys = zip(*valid_bins)
        ax.plot(xs, ys, "o-", color=COLORS[tier], ms=4, lw=1.5,
                label=f"{tier} (n={mask.sum()}, ECE={ece:.4f})")
        tier_eces[tier] = ece

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.set_title(f"{title}\nCalibration by confidence tier", fontsize=10)
    ax.legend(fontsize=8)
    return tier_eces


def plot_sharpness(ax, y_pred, title):
    ax.hist(y_pred, bins=30, range=(0, 1), color="#4a90d9", alpha=0.7, edgecolor="white")
    ax.axvline(0.5, color="gray", lw=1, ls="--", alpha=0.6)
    std = y_pred.std()
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Count")
    ax.set_title(f"{title}\nSharpness (σ = {std:.4f})", fontsize=10)


# ── main ──────────────────────────────────────────────────────────────────────

def run():
    from classical_learning.strategy.config import TARGETS_CLASSIFICATION

    # Discover classification ensemble bundles
    discovered = []
    for pkl in sorted(MODELS_DIR.glob(f"ensemble_*_{TIER}.pkl")):
        stem = pkl.stem  # ensemble_{target}_{tier}
        target = stem[len("ensemble_"):-len(f"_{TIER}")]
        if target in TARGETS_CLASSIFICATION:
            discovered.append(target)

    if not discovered:
        print(f"No classification ensemble bundles found in {MODELS_DIR}")
        print("Run `conda run -n pred python -m pregame.cli ensemble` first.")
        return

    print(f"Found {len(discovered)} classification targets: {discovered}\n")

    summary_rows = []

    with PdfPages(PDF_PATH) as pdf:
        for target in discovered:
            print(f"Processing: {target}")
            data = load_target_data(target)
            if data is None:
                continue

            y_true = data["y_true"]
            oof_raw = data["oof_raw"]
            oof_cal = data["oof_calibrated"]
            oof_std = data["oof_std"]
            cal = data["cal"]
            n = data["n"]

            ece_raw, bin_stats_raw = compute_ece(y_true, oof_raw)
            ece_cal, bin_stats_cal = compute_ece(y_true, oof_cal)
            brier_raw = brier_decomposition(y_true, oof_raw)
            brier_cal = brier_decomposition(y_true, oof_cal)
            sharpness = oof_cal.std()

            print(f"  n={n}  ECE_raw={ece_raw:.4f}  ECE_cal={ece_cal:.4f}  "
                  f"Brier={brier_cal['brier']:.4f}  BSS={brier_cal['bss']:.4f}  "
                  f"sharpness={sharpness:.4f}")

            # ── Figure 1: reliability + signed ECE (raw vs calibrated) ────────
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"{target.upper()} — Calibration Report (n={n})", fontsize=13)

            plot_reliability(axes[0, 0], y_true, oof_raw, "Raw ensemble OOF")
            plot_reliability(axes[0, 1], y_true, oof_cal, "After CalibrationBundle isotonic")
            plot_signed_ece(axes[1, 0], bin_stats_raw, "Signed ECE — raw")
            plot_signed_ece(axes[1, 1], bin_stats_cal, "Signed ECE — calibrated")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.savefig(OUT_DIR / f"cal_{target}_reliability.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

            # ── Figure 2: tiers + sharpness ───────────────────────────────────
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            fig.suptitle(f"{target.upper()} — Tiers & Sharpness", fontsize=13)

            tier_eces = plot_tiers(axes[0], y_true, oof_cal, oof_std, cal,
                                   "Confidence tiers (calibrated)")
            plot_sharpness(axes[1], oof_raw, "Raw predictions")
            plot_sharpness(axes[2], oof_cal, "Calibrated predictions")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.savefig(OUT_DIR / f"cal_{target}_tiers.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

            summary_rows.append({
                "target": target,
                "n": n,
                "ECE_raw": round(ece_raw, 4),
                "ECE_cal": round(ece_cal, 4),
                "Brier": round(brier_cal["brier"], 4),
                "BSS": round(brier_cal["bss"], 4),
                "Resolution": round(brier_cal["resolution"], 4),
                "Reliability": round(brier_cal["reliability"], 4),
                "Sharpness": round(sharpness, 4),
                "ECE_HIGH": round(tier_eces.get("HIGH", float("nan")), 4),
                "ECE_MED": round(tier_eces.get("MEDIUM", float("nan")), 4),
                "ECE_LOW": round(tier_eces.get("LOW", float("nan")), 4),
            })

        # ── Summary page ──────────────────────────────────────────────────────
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            fig, ax = plt.subplots(figsize=(14, max(3, len(summary_rows) * 0.8 + 2)))
            ax.axis("off")
            tbl = ax.table(
                cellText=df_summary.values,
                colLabels=df_summary.columns,
                cellLoc="center",
                loc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1, 1.5)
            ax.set_title("Calibration Summary", fontsize=12, pad=20)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nPDF saved: {PDF_PATH}")
    print("\n── Summary ──────────────────────────────────────────────────────────")
    if summary_rows:
        print(pd.DataFrame(summary_rows).to_string(index=False))
    else:
        print("No targets processed.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    run()
