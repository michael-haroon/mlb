"""Append H-stat-derived interaction features to game_features.parquet on S3.

Computes 6 pairwise interaction products identified by H-statistic analysis
across 8 LOYO folds. Each interaction is a simple element-wise product of
two pre-computed source columns.

Usage on EC2:
    python3.11 scripts/append_interaction_features.py
    python3.11 scripts/append_interaction_features.py --dry-run

Usage locally:
    conda run -n pred python scripts/append_interaction_features.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time

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

ELO_CENTER = 1500.0

REQUIRED_SOURCES = [
    "diff_massey_inn9",
    "diff_massey_inn6",
    "diff_massey_full",
    "home_team_woba_vs_rhp_roll200pa",
    "home_elo",
    "away_elo",
    "elo_diff",
    "diff_all_roll20_whip",
    "log5_prob_season",
    "pythag_2nd_diff",
    "consensus_prob",
]

INTERACTION_COLS = [
    "diff_massey_inn9_x_home_team_woba_vs_rhp_roll200pa",
    "diff_massey_inn6_x_home_elo_centered",
    "diff_all_roll20_whip_x_log5_prob_season",
    "away_elo_centered_x_home_team_woba_vs_rhp_roll200pa",
    "diff_massey_full_x_elo_diff",
    "pythag_2nd_diff_x_consensus_prob",
]

IDEMPOTENCY_MARKERS = [
    "_x_home_team_woba_vs_rhp_roll200pa",
    "_x_home_elo_centered",
    "_x_log5_prob_season",
    "_x_elo_diff",
    "_x_consensus_prob",
]

RANGE_CHECKS = {
    "diff_massey_inn9_x_home_team_woba_vs_rhp_roll200pa": (-15.0, 15.0),
    "diff_massey_inn6_x_home_elo_centered": (-4000.0, 4000.0),
    "diff_all_roll20_whip_x_log5_prob_season": (-3.0, 3.0),
    "away_elo_centered_x_home_team_woba_vs_rhp_roll200pa": (-150.0, 150.0),
    "diff_massey_full_x_elo_diff": (-5000.0, 5000.0),
    "pythag_2nd_diff_x_consensus_prob": (-2.0, 2.0),
}

SOURCE_PAIRS = {
    "diff_massey_inn9_x_home_team_woba_vs_rhp_roll200pa": (
        "diff_massey_inn9", "home_team_woba_vs_rhp_roll200pa"),
    "diff_massey_inn6_x_home_elo_centered": (
        "diff_massey_inn6", "home_elo"),
    "diff_all_roll20_whip_x_log5_prob_season": (
        "diff_all_roll20_whip", "log5_prob_season"),
    "away_elo_centered_x_home_team_woba_vs_rhp_roll200pa": (
        "away_elo", "home_team_woba_vs_rhp_roll200pa"),
    "diff_massey_full_x_elo_diff": (
        "diff_massey_full", "elo_diff"),
    "pythag_2nd_diff_x_consensus_prob": (
        "pythag_2nd_diff", "consensus_prob"),
}


def verify_source_columns(df: pd.DataFrame) -> bool:
    """Check source column distributions are sensible before computing interactions."""
    passed = True
    n = len(df)

    checks = {
        "diff_massey_inn9": {"min": -25, "max": 25, "max_nan_pct": 0.30},
        "diff_massey_inn6": {"min": -25, "max": 25, "max_nan_pct": 0.30},
        "diff_massey_full": {"min": -25, "max": 25, "max_nan_pct": 0.30},
        "home_team_woba_vs_rhp_roll200pa": {"min": 0.1, "max": 0.6, "max_nan_pct": 0.20},
        "home_elo": {"min": 1200, "max": 1800, "max_nan_pct": 0.01},
        "away_elo": {"min": 1200, "max": 1800, "max_nan_pct": 0.01},
        "elo_diff": {"min": -500, "max": 500, "max_nan_pct": 0.01},
        "diff_all_roll20_whip": {"min": -5, "max": 5, "max_nan_pct": 0.15},
        "log5_prob_season": {"min": 0.0, "max": 1.0, "max_nan_pct": 0.05},
        "pythag_2nd_diff": {"min": -1.5, "max": 1.5, "max_nan_pct": 0.05},
        "consensus_prob": {"min": 0.0, "max": 1.0, "max_nan_pct": 0.05},
    }

    for col, bounds in checks.items():
        s = df[col]
        nan_pct = s.isna().sum() / n
        vals = s.dropna()

        if nan_pct > bounds["max_nan_pct"]:
            log.warning(
                f"  SOURCE CHECK: {col} has {nan_pct:.1%} NaN "
                f"(threshold {bounds['max_nan_pct']:.0%})"
            )

        if len(vals) == 0:
            log.error(f"  SOURCE FAIL: {col} is entirely NaN")
            passed = False
            continue

        if vals.min() < bounds["min"] or vals.max() > bounds["max"]:
            log.error(
                f"  SOURCE FAIL: {col} range [{vals.min():.4f}, {vals.max():.4f}] "
                f"outside expected [{bounds['min']}, {bounds['max']}]"
            )
            passed = False
        else:
            log.info(
                f"  SOURCE OK: {col} — range [{vals.min():.4f}, {vals.max():.4f}], "
                f"NaN={nan_pct:.1%}, mean={vals.mean():.4f}"
            )

    return passed


def compute_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 6 interaction features. Returns new-column-only DataFrame."""
    new = {}

    new["diff_massey_inn9_x_home_team_woba_vs_rhp_roll200pa"] = (
        df["diff_massey_inn9"] * df["home_team_woba_vs_rhp_roll200pa"]
    )

    new["diff_massey_inn6_x_home_elo_centered"] = (
        df["diff_massey_inn6"] * (df["home_elo"] - ELO_CENTER)
    )

    new["diff_all_roll20_whip_x_log5_prob_season"] = (
        df["diff_all_roll20_whip"] * df["log5_prob_season"]
    )

    new["away_elo_centered_x_home_team_woba_vs_rhp_roll200pa"] = (
        (df["away_elo"] - ELO_CENTER) * df["home_team_woba_vs_rhp_roll200pa"]
    )

    new["diff_massey_full_x_elo_diff"] = (
        df["diff_massey_full"] * df["elo_diff"]
    )

    new["pythag_2nd_diff_x_consensus_prob"] = (
        df["pythag_2nd_diff"] * df["consensus_prob"]
    )

    return pd.DataFrame(new, index=df.index).astype("float32")


def validate(df_orig: pd.DataFrame, df_new: pd.DataFrame) -> bool:
    """Validate computed interaction features. Returns True if all checks pass."""
    passed = True
    n_orig = len(df_orig)
    n_new = len(df_new)

    if n_new != n_orig:
        log.error(f"FAIL: Row count {n_orig} → {n_new}")
        passed = False

    for col, (lo, hi) in RANGE_CHECKS.items():
        if col not in df_new.columns:
            continue
        vals = df_new[col].dropna()
        if len(vals) == 0:
            log.error(f"FAIL: {col} is entirely NaN")
            passed = False
            continue
        if vals.min() < lo or vals.max() > hi:
            log.error(
                f"FAIL: {col} outside [{lo}, {hi}]: "
                f"[{vals.min():.4f}, {vals.max():.4f}]"
            )
            passed = False
        else:
            log.info(f"PASS: {col} in [{vals.min():.4f}, {vals.max():.4f}]")

    for col, (src_a, src_b) in SOURCE_PAIRS.items():
        if col not in df_new.columns:
            continue
        source_nan = df_orig[src_a].isna() | df_orig[src_b].isna()
        expected_nan = source_nan.sum()
        actual_nan = df_new[col].isna().sum()
        if actual_nan < expected_nan:
            log.error(
                f"FAIL: {col} NaN={actual_nan} < source NaN={expected_nan}"
            )
            passed = False
        else:
            log.info(f"PASS: {col} NaN={actual_nan} ≥ source NaN={expected_nan}")

    for col in df_new.columns:
        if df_new[col].dtype != np.float32:
            log.error(f"FAIL: {col} dtype is {df_new[col].dtype}, expected float32")
            passed = False

    return passed


def print_summary(df_new: pd.DataFrame) -> None:
    """Print descriptive stats for new columns."""
    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY STATISTICS")
    log.info("=" * 60)
    for col in df_new.columns:
        s = df_new[col]
        vals = s.dropna()
        log.info(
            f"  {col}:\n"
            f"    count={len(vals):,}  NaN={s.isna().sum():,}  "
            f"mean={vals.mean():.6f}  std={vals.std():.6f}\n"
            f"    min={vals.min():.6f}  p25={vals.quantile(0.25):.6f}  "
            f"p50={vals.quantile(0.50):.6f}  p75={vals.quantile(0.75):.6f}  "
            f"max={vals.max():.6f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate only, don't upload to S3",
    )
    args = parser.parse_args()

    t0 = time.time()
    log.info("Appending H-stat interaction features to game_features.parquet")

    # Download from S3
    s3 = boto3.client("s3")
    log.info(f"Downloading s3://{BUCKET}/{FEATURES_KEY} ...")
    tmp_in = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    s3.download_fileobj(BUCKET, FEATURES_KEY, tmp_in)
    tmp_in.close()
    log.info(f"Downloaded to {tmp_in.name}")

    df = pd.read_parquet(tmp_in.name)
    n_rows = len(df)
    n_cols_orig = len(df.columns)
    log.info(f"Loaded: {n_rows:,} rows × {n_cols_orig} cols")

    # Idempotency: drop previously computed interaction columns
    existing_new = [
        c for c in df.columns
        if any(m in c for m in IDEMPOTENCY_MARKERS)
    ]
    if existing_new:
        log.info(
            f"Dropping {len(existing_new)} previously computed columns "
            f"for recompute: {existing_new}"
        )
        df = df.drop(columns=existing_new)

    # Verify source columns exist
    missing = [c for c in REQUIRED_SOURCES if c not in df.columns]
    if missing:
        log.error(f"Missing required source columns: {missing}")
        sys.exit(1)
    log.info(f"All {len(REQUIRED_SOURCES)} source columns present")

    # Verify source column values are sensible
    log.info("")
    log.info("=" * 60)
    log.info("SOURCE COLUMN VERIFICATION")
    log.info("=" * 60)
    sources_ok = verify_source_columns(df)
    if not sources_ok:
        log.error("SOURCE VERIFICATION FAILED — aborting")
        sys.exit(1)
    log.info("All source columns pass sanity checks")

    # Compute interactions
    log.info("")
    log.info("Computing 6 interaction features...")
    df_new = compute_interactions(df)
    log.info(f"  +{len(df_new.columns)} interaction columns computed")

    # Validate output
    log.info("")
    log.info("=" * 60)
    log.info("OUTPUT VALIDATION")
    log.info("=" * 60)
    valid = validate(df, df_new)
    print_summary(df_new)

    if not valid:
        log.error("VALIDATION FAILED — not uploading")
        sys.exit(1)

    log.info("")
    log.info("ALL VALIDATION CHECKS PASSED")

    if args.dry_run:
        elapsed = time.time() - t0
        log.info(f"--dry-run: skipping upload ({elapsed:.1f}s)")
        return

    # Append and write back
    result = pd.concat([df, df_new], axis=1)
    assert len(result) == n_rows, f"Row count mismatch: {len(result)} != {n_rows}"

    tmp_out = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp_out.close()
    result.to_parquet(tmp_out.name, index=False, engine="pyarrow")
    log.info(
        f"Written to {tmp_out.name}: {len(result):,} rows × {len(result.columns)} cols"
    )

    log.info(f"Uploading to s3://{BUCKET}/{FEATURES_KEY} ...")
    s3.upload_file(tmp_out.name, BUCKET, FEATURES_KEY)
    elapsed = time.time() - t0
    log.info(
        f"Done: +{len(df_new.columns)} new columns appended "
        f"({n_cols_orig} → {len(result.columns)} total, {elapsed:.1f}s)"
    )


if __name__ == "__main__":
    main()
