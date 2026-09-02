"""Variance decomposition: MC noise vs temporal instability.

After running multi-repeat importance (n_repeats=30), this script decomposes
the total fold-to-fold variance into:
  σ²_MC     = mean across folds of [within-fold variance from 30 repeats]
  σ²_temporal = variance across folds of [fold means]
  R(f)      = σ²_MC / (σ²_MC + σ²_temporal)  — fraction reducible by multi-rep

Usage:
    python3.11 scripts/analyze_variance_decomposition.py --target home_win
    python3.11 scripts/analyze_variance_decomposition.py --all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.strategy.config import ALL_TARGETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BASE_DIR = Path("data/importance")


def decompose_variance(target: str) -> pd.DataFrame | None:
    """Compute variance decomposition for one target.

    Requires both *_raw.csv (fold means) and *_repeat_sd.csv (within-fold SD).
    """
    target_dir = BASE_DIR / target
    results = []

    for method in ("mda", "desub_mda"):
        raw_path = target_dir / f"importance_{method}_raw.csv"
        sd_path = target_dir / f"importance_{method}_repeat_sd.csv"

        if not raw_path.exists() or not sd_path.exists():
            log.warning(f"  [{target}] Missing {method} raw or repeat_sd — skipping")
            continue

        raw = pd.read_csv(raw_path, index_col=0)
        repeat_sd = pd.read_csv(sd_path, index_col=0)

        common_cols = raw.columns.intersection(repeat_sd.columns)
        if len(common_cols) == 0:
            log.warning(f"  [{target}] No common features between raw and repeat_sd for {method}")
            continue

        for feat in common_cols:
            fold_means = raw[feat].dropna().values
            fold_sds = repeat_sd[feat].dropna().values

            if len(fold_means) < 2 or len(fold_sds) == 0:
                continue

            # σ²_MC = mean of within-fold variances (SD² from repeats)
            sigma2_mc = np.mean(fold_sds ** 2)
            # σ²_temporal = variance of fold means
            sigma2_temporal = np.var(fold_means, ddof=1)
            # Total observed variance (what single-permutation runs saw)
            sigma2_total = sigma2_mc + sigma2_temporal
            # Fraction attributable to MC noise
            r_mc = sigma2_mc / sigma2_total if sigma2_total > 0 else 0.0

            results.append({
                "feature": feat,
                "method": method,
                "sigma2_mc": sigma2_mc,
                "sigma2_temporal": sigma2_temporal,
                "sigma2_total": sigma2_total,
                "R_mc": r_mc,
                "mean_importance": fold_means.mean(),
            })

    if not results:
        return None

    df = pd.DataFrame(results)
    return df.sort_values("R_mc", ascending=False)


def summarize(df: pd.DataFrame, target: str):
    """Print diagnostic summary."""
    log.info(f"\n{'='*70}")
    log.info(f"Variance Decomposition: {target}")
    log.info(f"{'='*70}")

    for method in df["method"].unique():
        sub = df[df["method"] == method]
        log.info(f"\n  {method.upper()} ({len(sub)} features):")
        log.info(f"    Median R_mc (MC fraction):  {sub['R_mc'].median():.3f}")
        log.info(f"    Mean R_mc:                  {sub['R_mc'].mean():.3f}")
        log.info(f"    Features with R_mc > 0.5:   {(sub['R_mc'] > 0.5).sum()} "
                 f"({(sub['R_mc'] > 0.5).mean()*100:.1f}%)")
        log.info(f"    Features with R_mc > 0.8:   {(sub['R_mc'] > 0.8).sum()} "
                 f"({(sub['R_mc'] > 0.8).mean()*100:.1f}%)")

        # Top features by MC noise fraction
        top_mc = sub.nlargest(10, "R_mc")
        log.info(f"\n    Top-10 highest MC noise fraction:")
        for _, row in top_mc.iterrows():
            log.info(f"      {row['feature']:50s} R_mc={row['R_mc']:.3f} "
                     f"imp={row['mean_importance']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", type=str)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    targets = ALL_TARGETS if args.all else [args.target]

    for target in targets:
        df = decompose_variance(target)
        if df is None:
            log.warning(f"[{target}] No data available — run multi-repeat importance first")
            continue

        summarize(df, target)

        # Save
        out_path = BASE_DIR / target / "variance_decomposition.csv"
        df.to_csv(out_path, index=False)
        log.info(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
