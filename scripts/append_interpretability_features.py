"""Append interpretability-derived features to game_features.parquet on S3.

Computes Tier 1 (interaction features from H-stat/SHAP findings) and
Tier 2 (ALE-backed clipping and binning) as new columns.

Usage on EC2:
    python3.11 scripts/append_interpretability_features.py
    python3.11 scripts/append_interpretability_features.py --dry-run  # validate only, don't upload

Usage locally:
    conda run -n pred python scripts/append_interpretability_features.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import boto3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BUCKET = "mlb-265753586044-us-east-1-an"
FEATURES_KEY = "data/features/game_features.parquet"

# --- Tier 1: Interaction features ---
# Source columns for platoon divergence
PLATOON_PAIRS = [
    ("home_sp_fip_vs_rhh_roll5", "home_sp_fip_vs_lhh_roll5", "home_sp_platoon_divergence_roll5"),
    ("away_sp_fip_vs_rhh_roll5", "away_sp_fip_vs_lhh_roll5", "away_sp_platoon_divergence_roll5"),
    ("home_sp_fip_vs_rhh_roll10", "home_sp_fip_vs_lhh_roll10", "home_sp_platoon_divergence_roll10"),
    ("away_sp_fip_vs_rhh_roll10", "away_sp_fip_vs_lhh_roll10", "away_sp_platoon_divergence_roll10"),
]

# --- Tier 2: Clipping targets ---
CLIP_FEATURES = [
    "home_sp_kpct_vs_lhh_roll5",
    "home_sp_kpct_vs_rhh_roll5",
    "away_sp_kpct_vs_lhh_roll5",
    "away_sp_kpct_vs_rhh_roll5",
    "home_sp_kpct_vs_lhh_roll10",
    "home_sp_kpct_vs_rhh_roll10",
    "away_sp_kpct_vs_lhh_roll10",
    "away_sp_kpct_vs_rhh_roll10",
    "away_sp_kbb_diff_vs_lhh_roll10",
    "away_sp_kbb_diff_vs_lhh_roll5",
    "home_sp_kbb_diff_vs_lhh_roll10",
    "home_sp_kbb_diff_vs_lhh_roll5",
    "home_sp_kbb_diff_vs_rhh_roll10",
    "away_sp_kbb_diff_vs_rhh_roll10",
    "sp_era_diff",
]


def compute_tier1(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Tier 1 interaction features."""
    new_cols = {}

    # 1. SP Platoon Divergence: |FIP_vs_RHH - FIP_vs_LHH|
    for col_rhh, col_lhh, out_name in PLATOON_PAIRS:
        new_cols[out_name] = (df[col_rhh] - df[col_lhh]).abs()

    # Differential between home and away platoon divergence
    new_cols["diff_sp_platoon_divergence_roll5"] = (
        new_cols["home_sp_platoon_divergence_roll5"]
        - new_cols["away_sp_platoon_divergence_roll5"]
    )
    new_cols["diff_sp_platoon_divergence_roll10"] = (
        new_cols["home_sp_platoon_divergence_roll10"]
        - new_cols["away_sp_platoon_divergence_roll10"]
    )

    # 2. Market-Rating Disagreement: consensus_prob - expit(diff_massey_inn3)
    # diff_massey_inn3 ranges [-9.55, 10.84] — use logistic to map to probability
    from scipy.special import expit

    massey_implied = expit(df["diff_massey_inn3"].values)
    disagreement = df["consensus_home_win_prob"].values - massey_implied
    new_cols["market_massey_inn3_disagreement"] = pd.Series(
        disagreement, index=df.index
    )

    # 3. Division-Strength Interaction: elo_prob_x_same_division * sign(pythag_2nd_diff)
    new_cols["div_strength_interaction"] = (
        df["elo_prob_x_same_division"] * np.sign(df["pythag_2nd_diff"])
    )

    return pd.DataFrame(new_cols, index=df.index)


def compute_tier2(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Tier 2 clipped/binned features."""
    new_cols = {}

    # Clipping at 5th/95th percentile
    for col in CLIP_FEATURES:
        if col not in df.columns:
            log.warning(f"Tier 2 clip target missing: {col}")
            continue
        p5 = df[col].quantile(0.05)
        p95 = df[col].quantile(0.95)
        new_cols[f"{col}_clipped"] = df[col].clip(lower=p5, upper=p95)
        log.info(f"  Clipped {col}: [{p5:.4f}, {p95:.4f}]")

    # Weather temperature clipping at fixed thresholds (ALE-backed)
    new_cols["weather_temp_clipped"] = df["weather_temp"].clip(lower=55.0, upper=85.0)

    # Binning: elo_prob_x_same_division > 0.57
    new_cols["elo_prob_same_div_above57"] = (
        df["elo_prob_x_same_division"] > 0.57
    ).astype(float)

    # Binning: sign of pythag_2nd_diff (-1, 0, +1)
    new_cols["pythag_2nd_diff_sign"] = np.sign(df["pythag_2nd_diff"]).astype(float)

    return pd.DataFrame(new_cols, index=df.index)


def validate(df_orig: pd.DataFrame, df_new: pd.DataFrame) -> bool:
    """Run all validation checks. Returns True if all pass."""
    passed = True
    n_orig = len(df_orig)
    n_new = len(df_new)

    # 1. Row count preservation
    if n_new != n_orig:
        log.error(f"FAIL: Row count changed: {n_orig} -> {n_new}")
        passed = False
    else:
        log.info(f"PASS: Row count preserved ({n_orig})")

    # 2. Platoon divergence >= 0
    for col in df_new.columns:
        if "platoon_divergence" in col and "diff_" not in col:
            violations = (df_new[col].dropna() < 0).sum()
            if violations > 0:
                log.error(f"FAIL: {col} has {violations} negative values")
                passed = False
            else:
                log.info(f"PASS: {col} >= 0 (all non-NaN values)")

    # 3. Market disagreement range
    col = "market_massey_inn3_disagreement"
    if col in df_new.columns:
        vals = df_new[col].dropna()
        if vals.min() < -1.0 or vals.max() > 1.0:
            log.error(f"FAIL: {col} outside [-1, 1]: [{vals.min():.4f}, {vals.max():.4f}]")
            passed = False
        else:
            log.info(f"PASS: {col} in [{vals.min():.4f}, {vals.max():.4f}]")

    # 4. Division interaction range (should be in [-1, 1] since elo_prob is [0,1] * sign is [-1,1])
    col = "div_strength_interaction"
    if col in df_new.columns:
        vals = df_new[col].dropna()
        if vals.min() < -1.0 - 1e-9 or vals.max() > 1.0 + 1e-9:
            log.error(f"FAIL: {col} outside [-1, 1]: [{vals.min():.4f}, {vals.max():.4f}]")
            passed = False
        else:
            log.info(f"PASS: {col} in [{vals.min():.4f}, {vals.max():.4f}]")

    # 5. Clipped features within bounds
    for col in df_new.columns:
        if col.endswith("_clipped"):
            source_col = col.replace("_clipped", "")
            if source_col in df_orig.columns:
                vals = df_new[col].dropna()
                if col == "weather_temp_clipped":
                    lo, hi = 55.0, 85.0
                else:
                    lo = df_orig[source_col].quantile(0.05)
                    hi = df_orig[source_col].quantile(0.95)
                below = (vals < lo - 1e-9).sum()
                above = (vals > hi + 1e-9).sum()
                if below > 0 or above > 0:
                    log.error(f"FAIL: {col} has {below} below {lo:.4f} and {above} above {hi:.4f}")
                    passed = False
                else:
                    log.info(f"PASS: {col} within [{lo:.4f}, {hi:.4f}]")

    # 6. Binned features have only expected values
    col = "elo_prob_same_div_above57"
    if col in df_new.columns:
        unique = set(df_new[col].dropna().unique())
        if not unique.issubset({0.0, 1.0}):
            log.error(f"FAIL: {col} has unexpected values: {unique - {0.0, 1.0}}")
            passed = False
        else:
            log.info(f"PASS: {col} values are {{0, 1}}")

    col = "pythag_2nd_diff_sign"
    if col in df_new.columns:
        unique = set(df_new[col].dropna().unique())
        if not unique.issubset({-1.0, 0.0, 1.0}):
            log.error(f"FAIL: {col} has unexpected values: {unique - {-1.0, 0.0, 1.0}}")
            passed = False
        else:
            log.info(f"PASS: {col} values are {{-1, 0, 1}}")

    # 7. NaN propagation check — new features should never have FEWER NaN than their sources
    nan_checks = {
        "home_sp_platoon_divergence_roll5": ["home_sp_fip_vs_rhh_roll5", "home_sp_fip_vs_lhh_roll5"],
        "away_sp_platoon_divergence_roll5": ["away_sp_fip_vs_rhh_roll5", "away_sp_fip_vs_lhh_roll5"],
        "home_sp_platoon_divergence_roll10": ["home_sp_fip_vs_rhh_roll10", "home_sp_fip_vs_lhh_roll10"],
        "away_sp_platoon_divergence_roll10": ["away_sp_fip_vs_rhh_roll10", "away_sp_fip_vs_lhh_roll10"],
        "market_massey_inn3_disagreement": ["consensus_home_win_prob", "diff_massey_inn3"],
    }
    for new_col, sources in nan_checks.items():
        if new_col not in df_new.columns:
            continue
        # Expected NaN = union of NaN in any source
        source_nan_mask = pd.Series(False, index=df_orig.index)
        for src in sources:
            if src in df_orig.columns:
                source_nan_mask |= df_orig[src].isna()
        expected_nan = source_nan_mask.sum()
        actual_nan = df_new[new_col].isna().sum()
        if actual_nan < expected_nan:
            log.error(
                f"FAIL: {new_col} has FEWER NaN ({actual_nan}) than sources ({expected_nan})"
            )
            passed = False
        else:
            log.info(f"PASS: {new_col} NaN={actual_nan} >= source NaN={expected_nan}")

    return passed


def print_summary(df_new: pd.DataFrame, df_orig: pd.DataFrame):
    """Print summary statistics for all new columns."""
    log.info("")
    log.info("=" * 90)
    log.info("NEW FEATURE SUMMARY STATISTICS")
    log.info("=" * 90)
    log.info(
        f"{'Column':50s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'NaN':>6s} {'NaN%':>6s}"
    )
    log.info("-" * 90)
    for col in sorted(df_new.columns):
        s = df_new[col]
        nan_count = s.isna().sum()
        nan_pct = 100 * nan_count / len(s)
        log.info(
            f"{col:50s} {s.mean():8.4f} {s.std():8.4f} {s.min():8.4f} {s.max():8.4f} "
            f"{nan_count:6d} {nan_pct:5.1f}%"
        )
    log.info("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Append interpretability features")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, don't upload")
    args = parser.parse_args()

    t0 = time.time()

    # Download parquet from S3
    log.info(f"Downloading s3://{BUCKET}/{FEATURES_KEY} ...")
    s3 = boto3.client("s3")
    tmp_in = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    s3.download_fileobj(
        BUCKET, FEATURES_KEY, tmp_in
    )
    tmp_in.close()
    log.info(f"Downloaded to {tmp_in.name}")

    # Load
    df = pd.read_parquet(tmp_in.name)
    n_rows = len(df)
    n_cols_orig = len(df.columns)
    log.info(f"Loaded: {n_rows:,} rows × {n_cols_orig} cols")

    # Check for already-computed features (idempotency)
    tier1_markers = ["sp_platoon_divergence", "market_massey_inn3_disagreement", "div_strength_interaction"]
    tier2_markers = ["_clipped", "elo_prob_same_div_above57", "pythag_2nd_diff_sign"]
    existing_new = [c for c in df.columns if any(m in c for m in tier1_markers + tier2_markers)]
    if existing_new:
        log.info(f"Dropping {len(existing_new)} previously computed columns for recompute: {existing_new[:5]}...")
        df = df.drop(columns=existing_new)

    # Verify source columns exist
    required_sources = [
        "home_sp_fip_vs_rhh_roll5", "home_sp_fip_vs_lhh_roll5",
        "away_sp_fip_vs_rhh_roll5", "away_sp_fip_vs_lhh_roll5",
        "home_sp_fip_vs_rhh_roll10", "home_sp_fip_vs_lhh_roll10",
        "away_sp_fip_vs_rhh_roll10", "away_sp_fip_vs_lhh_roll10",
        "consensus_home_win_prob", "diff_massey_inn3",
        "elo_prob_x_same_division", "pythag_2nd_diff",
        "weather_temp",
    ]
    missing = [c for c in required_sources if c not in df.columns]
    if missing:
        log.error(f"Missing required source columns: {missing}")
        sys.exit(1)

    # Compute features
    log.info("Computing Tier 1 interaction features...")
    tier1 = compute_tier1(df)
    log.info(f"  +{len(tier1.columns)} Tier 1 columns")

    log.info("Computing Tier 2 clipped/binned features...")
    tier2 = compute_tier2(df)
    log.info(f"  +{len(tier2.columns)} Tier 2 columns")

    # Combine
    all_new = pd.concat([tier1, tier2], axis=1)
    log.info(f"Total new columns: {len(all_new.columns)}")

    # Validate
    log.info("")
    log.info("=" * 50)
    log.info("VALIDATION")
    log.info("=" * 50)
    valid = validate(df, all_new)
    print_summary(all_new, df)

    if not valid:
        log.error("VALIDATION FAILED — not uploading")
        sys.exit(1)

    log.info("")
    log.info("ALL VALIDATION CHECKS PASSED")

    if args.dry_run:
        log.info("--dry-run: skipping upload")
        elapsed = time.time() - t0
        log.info(f"Dry run complete in {elapsed:.1f}s")
        return

    # Append to dataframe and write back
    result = pd.concat([df, all_new], axis=1)
    assert len(result) == n_rows, f"Row count mismatch: {len(result)} != {n_rows}"

    tmp_out = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp_out.close()
    result.to_parquet(tmp_out.name, index=False, engine="pyarrow")
    log.info(f"Written to {tmp_out.name}: {len(result):,} rows × {len(result.columns)} cols")

    # Upload to S3
    log.info(f"Uploading to s3://{BUCKET}/{FEATURES_KEY} ...")
    s3.upload_file(tmp_out.name, BUCKET, FEATURES_KEY)
    elapsed = time.time() - t0
    log.info(
        f"Done: +{len(all_new.columns)} new columns appended "
        f"({n_cols_orig} → {len(result.columns)} total, {elapsed:.1f}s)"
    )


if __name__ == "__main__":
    main()
