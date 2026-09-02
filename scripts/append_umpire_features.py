"""Append umpire tendency features to the master game_features.parquet on S3.

Run: conda run -n pred python scripts/append_umpire_features.py

Steps:
1. Load game_features from S3 (source of truth)
2. Join umpire_2b from pitches table (not in existing parquet)
3. Compute 6 umpire features via _umpire_features()
4. Verify: NaN rates, value ranges, no column corruption
5. Append new columns and write back to S3

Features added:
- ump_hp_rpg_factor: HP umpire runs-per-game factor (ratio to league avg)
- ump_hp_bb_per_game: HP umpire career walks per game
- ump_hp_k_per_game: HP umpire career strikeouts per game
- ump_hp_called_strike_pct: HP umpire called strike percentage (zone size)
- ump_2b_sb_per_game: 2B umpire career stolen bases per game
- ump_2b_cs_per_game: 2B umpire career caught stealing per game
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deep_learning"))

from classical_learning.engineering.feature_engineering import _umpire_features

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

S3_BUCKET = "mlb-265753586044-us-east-1-an"
S3_FEATURES_KEY = f"{S3_BUCKET}/data/features/game_features.parquet"
S3_DATA = f"s3://{S3_BUCKET}/data"

EXPECTED_UMP_COLS = [
    "ump_hp_rpg_factor",
    "ump_hp_bb_per_game",
    "ump_hp_k_per_game",
    "ump_hp_called_strike_pct",
    "ump_2b_sb_per_game",
    "ump_2b_cs_per_game",
]


def join_pitch_derived_columns(games: pd.DataFrame) -> pd.DataFrame:
    """Join umpire_2b and game_called_strike_pct from pitches table.

    Both require the pitch-level table:
    - umpire_2b: not in boxscore, only in per-pitch metadata
    - game_called_strike_pct: Called Strike / (Called Strike + Ball) per game,
      since the boxscore endpoint doesn't break down strikes by call type
    """
    need_ump2b = "umpire_2b" not in games.columns or games["umpire_2b"].notna().sum() == 0
    need_csp = "game_called_strike_pct" not in games.columns or games["game_called_strike_pct"].eq(0).all()

    if not need_ump2b and not need_csp:
        log.info("  umpire_2b and game_called_strike_pct already present")
        return games

    log.info("Loading pitches table for umpire_2b + called_strike_pct (~2 min)...")
    from mlb_dl.data_sources import ParquetCatalog, season_range

    catalog = ParquetCatalog(S3_DATA)
    seasons = season_range(2015, 2026)
    pitches = catalog.read_table(
        "pitches", columns=["game_pk", "umpire_2b", "pitch_call"], seasons=seasons
    )
    log.info(f"  Pitches loaded: {len(pitches):,} rows")

    # --- umpire_2b ---
    if need_ump2b:
        ump2b = (
            pitches.drop_duplicates("game_pk")[["game_pk", "umpire_2b"]]
            .set_index("game_pk")
        )
        if "umpire_2b" in games.columns:
            games = games.drop(columns=["umpire_2b"])
        games = games.join(ump2b, on="game_pk", how="left")
        coverage = games["umpire_2b"].notna().sum()
        log.info(f"  umpire_2b joined: {coverage:,}/{len(games):,} ({100*coverage/len(games):.1f}%)")

    # --- game_called_strike_pct ---
    if need_csp:
        ump_decisions = pitches[pitches["pitch_call"].isin(["Called Strike", "Ball"])].copy()
        log.info(f"  Umpire decisions (Called Strike + Ball): {len(ump_decisions):,} pitches")
        per_game = ump_decisions.groupby("game_pk")["pitch_call"].agg(
            called_strikes=lambda x: (x == "Called Strike").sum(),
            balls=lambda x: (x == "Ball").sum(),
        )
        per_game["game_called_strike_pct"] = (
            per_game["called_strikes"] / (per_game["called_strikes"] + per_game["balls"])
        )
        csp_series = per_game[["game_called_strike_pct"]]

        if "game_called_strike_pct" in games.columns:
            games = games.drop(columns=["game_called_strike_pct"])
        games = games.join(csp_series, on="game_pk", how="left")
        coverage = games["game_called_strike_pct"].notna().sum()
        log.info(f"  game_called_strike_pct joined: {coverage:,}/{len(games):,} ({100*coverage/len(games):.1f}%)")
        log.info(f"  mean={games['game_called_strike_pct'].mean():.4f}, std={games['game_called_strike_pct'].std():.4f}")

    del pitches
    return games


def verify_features(original: pd.DataFrame, result: pd.DataFrame, new_cols: list[str]) -> bool:
    """Run sanity checks on the appended features. Returns True if all pass."""
    passed = True

    # 1. Row count unchanged
    if len(result) != len(original):
        log.error(f"FAIL: Row count changed {len(original):,} → {len(result):,}")
        return False
    log.info(f"  [PASS] Row count preserved: {len(result):,}")

    # 2. Original columns not corrupted (spot-check numeric sums)
    sample_cols = [c for c in original.columns
                   if original[c].dtype in (np.float32, np.float64) and original[c].notna().any()][:10]
    for c in sample_cols:
        orig_sum = original[c].sum()
        new_sum = result[c].sum()
        if abs(orig_sum - new_sum) > 1e-3:
            log.error(f"FAIL: Column {c} corrupted: sum {orig_sum:.4f} → {new_sum:.4f}")
            passed = False
    if passed:
        log.info(f"  [PASS] Original columns intact (spot-checked {len(sample_cols)} numeric cols)")

    # 3. New columns exist
    missing = [c for c in new_cols if c not in result.columns]
    if missing:
        log.error(f"FAIL: Missing expected columns: {missing}")
        passed = False
    else:
        log.info(f"  [PASS] All {len(new_cols)} umpire columns present")

    # 4. NaN rates and value ranges
    for c in new_cols:
        if c not in result.columns:
            continue
        s = result[c]
        nan_pct = s.isna().mean() * 100
        non_null = s.notna().sum()

        if non_null == 0:
            log.error(f"FAIL: {c} is ALL NaN — feature computation broken")
            passed = False
            continue

        valid = s.dropna()
        if "pct" in c:
            bad = ((valid < 0) | (valid > 1)).sum()
            if bad > 0:
                log.error(f"FAIL: {c} has {bad} values outside [0,1]")
                passed = False
        elif "factor" in c:
            if valid.min() < 0:
                log.error(f"FAIL: {c} has negative values (min={valid.min():.4f})")
                passed = False
            if valid.max() > 5.0:
                log.error(f"FAIL: {c} has extreme values (max={valid.max():.4f})")
                passed = False
        elif "per_game" in c:
            if valid.min() < 0:
                log.error(f"FAIL: {c} has negative values (min={valid.min():.4f})")
                passed = False

        log.info(
            f"  {'[PASS]' if non_null > 0 else '[FAIL]'} {c}: "
            f"NaN={nan_pct:.1f}%, non-null={non_null:,}, "
            f"mean={s.mean():.4f}, min={s.min():.4f}, max={s.max():.4f}"
        )

    # 5. Temporal safety: first 20 games per umpire should be NaN (min_periods=20)
    if "ump_hp_bb_per_game" in result.columns and "umpire_hp" in result.columns:
        first_20 = result.groupby("umpire_hp").head(20)
        early_nan_rate = first_20["ump_hp_bb_per_game"].isna().mean()
        if early_nan_rate < 0.9:
            log.warning(f"  [WARN] Early-game NaN rate only {early_nan_rate:.1%} — possible min_periods issue")
        else:
            log.info(f"  [PASS] Temporal safety: first-20-games NaN rate = {early_nan_rate:.1%}")

    return passed


def main():
    t_start = time.time()
    fs = s3fs.S3FileSystem()

    # --- Step 1: Load ---
    log.info(f"Loading s3://{S3_FEATURES_KEY}...")
    games = pd.read_parquet(fs.open(S3_FEATURES_KEY, "rb"))
    log.info(f"  Shape: {games.shape[0]:,} rows × {games.shape[1]} cols")
    original = games.copy()

    # Idempotent: drop existing ump columns if re-running
    existing_ump = [c for c in games.columns if c.startswith("ump_")]
    if existing_ump:
        log.info(f"  Dropping existing ump columns for recompute: {existing_ump}")
        games = games.drop(columns=existing_ump)

    # --- Step 2: Join umpire_2b + game_called_strike_pct from pitches ---
    games = join_pitch_derived_columns(games)

    # --- Step 3: Compute features ---
    log.info("Computing umpire features (expanding means + shift(1))...")
    result = _umpire_features(games)
    new_ump_cols = [c for c in result.columns if c.startswith("ump_")]
    log.info(f"  Computed {len(new_ump_cols)} columns: {new_ump_cols}")

    # --- Step 4: Build output (append only — don't touch original columns) ---
    games_out = original.copy()
    # Drop pre-existing ump columns for idempotency
    drop_cols = [c for c in games_out.columns if c.startswith("ump_")]
    if "umpire_2b" in drop_cols:
        drop_cols.remove("umpire_2b")
    if drop_cols:
        games_out = games_out.drop(columns=drop_cols)

    # Attach umpire_2b if not already there
    if "umpire_2b" not in games_out.columns:
        games_out["umpire_2b"] = games["umpire_2b"].values

    # Attach game_called_strike_pct (source column for incremental builds)
    if "game_called_strike_pct" not in games_out.columns:
        games_out["game_called_strike_pct"] = games["game_called_strike_pct"].values

    # Attach computed features
    for c in new_ump_cols:
        games_out[c] = result[c].values

    # --- Step 5: Verify ---
    log.info("Running verification checks...")
    checks_passed = verify_features(original, games_out, EXPECTED_UMP_COLS)

    if not checks_passed:
        log.error("VERIFICATION FAILED — aborting write.")
        log.error("  Backup exists at: s3://mlb-265753586044-us-east-1-an/data/features/backup/")
        sys.exit(1)

    # --- Step 6: Write ---
    log.info(f"All checks passed. Writing to s3://{S3_FEATURES_KEY}...")
    with fs.open(S3_FEATURES_KEY, "wb") as f:
        games_out.to_parquet(f, index=False, engine="pyarrow")

    elapsed = time.time() - t_start
    log.info(
        f"Done in {elapsed:.1f}s: {games_out.shape[0]:,} rows × {games_out.shape[1]} cols "
        f"(+{len(new_ump_cols)} ump features + umpire_2b)"
    )


if __name__ == "__main__":
    main()
