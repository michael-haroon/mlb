"""Append Massey per-inning ratings + pending feature groups to game_features.parquet.

Computes and appends:
  1. Massey per-inning ratings (inn1–inn9 + full) — from linescore
  2. Colley per-inning ratings — win/loss variant
  3. Bullpen workload (BFI) — already implemented, never run
  4. Manager tendencies — already implemented, never run
  5. Bat strength (hit distance, TB/hit) — already implemented, never run

All features are T0 (use only prior completed games): strictly temporal-safe.

Usage (EC2):
    python3.12 scripts/append_massey_and_pending_features.py

Usage (local):
    conda run -n pred python scripts/append_massey_and_pending_features.py

Verification: prints shape, NaN rates, and distribution stats before and after.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deep_learning"))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FEATURES_PATH = ROOT / "classical_learning" / "artifacts" / "features" / "game_features.parquet"
S3_SOURCE = "s3://mlb-265753586044-us-east-1-an/data"
SEASON_START = 2015
SEASON_END = 2026


def verify_parquet(df: pd.DataFrame, label: str) -> dict:
    """Print and return verification stats for the parquet."""
    stats = {
        "rows": len(df),
        "cols": len(df.columns),
        "total_nan": int(df.isna().sum().sum()),
        "nan_pct": float(df.isna().mean().mean()) * 100,
    }
    log.info(f"[{label}] {stats['rows']:,} rows × {stats['cols']} cols | NaN: {stats['nan_pct']:.2f}%")
    return stats


def verify_new_columns(df: pd.DataFrame, new_cols: list[str], label: str) -> None:
    """Detailed verification of newly appended columns."""
    log.info(f"\n[{label}] Verifying {len(new_cols)} new columns:")
    for c in sorted(new_cols):
        s = df[c]
        nan_pct = s.isna().mean() * 100
        if s.notna().any():
            desc = f"min={s.min():.3f} mean={s.mean():.3f} max={s.max():.3f}"
        else:
            desc = "ALL NaN"
        status = "OK" if nan_pct < 95 else "WARN"
        log.info(f"  [{status}] {c}: NaN={nan_pct:.1f}% {desc}")


def compute_massey_features(existing: pd.DataFrame) -> pd.DataFrame | None:
    """Compute Massey per-inning + full diff features from linescore."""
    from mlb_dl.data_sources import ParquetCatalog, season_range
    from classical_learning.engineering.massey_ratings import (
        prepare_linescore_cumulative,
        build_pregame_massey_features,
        MasseyDesign,
        MASSEY_TARGETS,
    )
    from classical_learning.engineering.constants import LINESCORE_COLUMNS

    catalog = ParquetCatalog(S3_SOURCE)
    seasons = season_range(SEASON_START, SEASON_END)

    log.info("Loading linescore from S3 for Massey computation...")
    linescore = catalog.read_table("linescore", columns=LINESCORE_COLUMNS, seasons=seasons)
    log.info(f"  linescore: {len(linescore):,} rows, {linescore['game_pk'].nunique()} games")

    if linescore.empty:
        log.warning("No linescore data — skipping Massey features")
        return None

    # Game metadata from existing features
    meta = existing[["game_pk", "season", "game_date", "home_team_id", "away_team_id"]].copy()
    meta = meta.drop_duplicates("game_pk")

    log.info("Preparing cumulative inning margins...")
    cumulative = prepare_linescore_cumulative(linescore, meta)
    log.info(f"  Cumulative: {cumulative.shape[0]} games")
    del linescore

    if cumulative.empty:
        log.warning("Empty cumulative frame — skipping Massey features")
        return None

    # Fit with default HA design; refit every game-date for maximum accuracy
    design = MasseyDesign("massey", include_home_advantage=True, min_games=30)
    log.info("Computing pregame Massey features (temporal-safe, per-inning + full)...")
    t0 = time.time()
    massey_features = build_pregame_massey_features(
        cumulative,
        designs=[design],
        targets=list(MASSEY_TARGETS),
        min_prior_games=30,
        refit_interval=1,
    )
    elapsed = time.time() - t0
    log.info(f"  Massey features: {massey_features.shape} in {elapsed:.1f}s")

    # Extract diff columns
    diff_cols = [c for c in massey_features.columns if c.startswith("diff_")]
    if not diff_cols:
        log.warning("No Massey diff columns produced")
        return None

    # Join onto existing by game_pk
    merge_cols = ["game_pk"] + diff_cols
    result = existing[["game_pk"]].merge(
        massey_features[merge_cols], on="game_pk", how="left"
    )

    # Verify temporal safety: early-season games should have NaN
    for c in diff_cols[:1]:
        first_valid = result[c].first_valid_index()
        if first_valid is not None:
            pct_filled = result[c].notna().mean() * 100
            log.info(f"  {c}: first valid at row {first_valid}, fill rate {pct_filled:.1f}%")

    return result[diff_cols]


def compute_colley_features(existing: pd.DataFrame) -> pd.DataFrame | None:
    """Compute Colley per-inning + full diff features from linescore.

    Colley uses win/loss per inning (who was ahead after inning N) instead of margin.
    """
    from mlb_dl.data_sources import ParquetCatalog, season_range
    from classical_learning.engineering.massey_ratings import (
        prepare_linescore_cumulative,
        fit_massey_inning,
        MasseyDesign,
        INNINGS,
    )
    from classical_learning.engineering.constants import LINESCORE_COLUMNS

    catalog = ParquetCatalog(S3_SOURCE)
    seasons = season_range(SEASON_START, SEASON_END)

    log.info("Loading linescore from S3 for Colley computation...")
    linescore = catalog.read_table("linescore", columns=LINESCORE_COLUMNS, seasons=seasons)
    if linescore.empty:
        log.warning("No linescore data — skipping Colley features")
        return None

    meta = existing[["game_pk", "season", "game_date", "home_team_id", "away_team_id"]].copy()
    meta = meta.drop_duplicates("game_pk")
    cumulative = prepare_linescore_cumulative(linescore, meta)
    del linescore

    if cumulative.empty:
        return None

    # Convert margins to binary wins for Colley: sign(margin) mapped to 0/1
    targets = [f"inn{i}" for i in INNINGS] + ["full"]
    for target in targets:
        margin_col = f"margin_{target}"
        if margin_col in cumulative.columns:
            cumulative[f"win_{target}"] = np.where(
                cumulative[margin_col] > 0, 1.0,
                np.where(cumulative[margin_col] < 0, 0.0, 0.5)
            )

    # Colley system: C = 2I + M, b = 1 + 0.5*(wins - losses)
    # Simpler: fit Massey with win (1/0/0.5) as target instead of margin
    all_rows: list[dict] = []
    design = MasseyDesign("colley", include_home_advantage=True, min_games=30)

    log.info("Computing Colley ratings per season (temporal-safe)...")
    for season, season_games in cumulative.groupby("season", sort=True):
        season_games = season_games.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
        dates = np.sort(season_games["game_date"].unique())

        cached_ratings: dict[str, pd.Series] = {}
        last_fit_idx = -1

        for date_idx, game_date in enumerate(dates):
            prior = season_games[season_games["game_date"] < game_date]

            if len(prior) < 30:
                today = season_games[season_games["game_date"] == game_date]
                for _, game in today.iterrows():
                    all_rows.append({"game_pk": game["game_pk"], "season": season})
                continue

            if date_idx - last_fit_idx >= 1 or not cached_ratings:
                cached_ratings = {}
                for target in targets:
                    win_col = f"win_{target}"
                    if win_col not in prior.columns:
                        continue
                    # Use win as a "margin" in the Massey solver
                    prior_copy = prior.copy()
                    prior_copy[f"margin_{target}"] = prior_copy[win_col]
                    try:
                        fit = fit_massey_inning(prior_copy, target, design, season=season)
                        if not fit.ratings.empty:
                            rating_col = fit.ratings.columns[2]
                            cached_ratings[f"colley_{target}"] = fit.ratings.set_index("team_id")[rating_col]
                    except Exception:
                        pass
                last_fit_idx = date_idx

            today = season_games[season_games["game_date"] == game_date]
            for _, game in today.iterrows():
                row: dict = {"game_pk": game["game_pk"], "season": season}
                home_id = int(game["home_team_id"])
                away_id = int(game["away_team_id"])
                for key, ratings_s in cached_ratings.items():
                    try:
                        row[f"diff_{key}"] = float(ratings_s.loc[home_id]) - float(ratings_s.loc[away_id])
                    except (KeyError, TypeError):
                        row[f"diff_{key}"] = np.nan
                all_rows.append(row)

    colley_df = pd.DataFrame(all_rows)
    diff_cols = [c for c in colley_df.columns if c.startswith("diff_")]
    if not diff_cols:
        return None

    result = existing[["game_pk"]].merge(
        colley_df[["game_pk"] + diff_cols], on="game_pk", how="left"
    )
    log.info(f"  Colley features: {len(diff_cols)} columns")
    return result[diff_cols]


def compute_pending_pitch_features(existing: pd.DataFrame) -> list[pd.DataFrame]:
    """Compute BFI, manager tendencies, and bat strength from pitches."""
    from classical_learning.engineering.data_loader import load_pitches_raw
    from classical_learning.engineering.pitch_level_features import (
        _compute_bat_strength_features,
        _compute_bullpen_workload_features,
        _compute_manager_tendency_features,
    )

    has_bullpen = any("bullpen_pitches_last3d" in c for c in existing.columns)
    has_mgr = any("mgr_pitchers_used" in c for c in existing.columns)
    has_bat_str = any("bat_avg_hit_distance" in c or "bat_tb_per_hit" in c for c in existing.columns)

    if has_bullpen and has_mgr and has_bat_str:
        log.info("All pitch-derived features already present — skipping")
        return []

    log.info("Loading pitches_raw from S3 for BFI/manager/bat-strength...")
    pitches_raw = load_pitches_raw(S3_SOURCE, SEASON_START, SEASON_END)
    pitches = pitches_raw[
        (pitches_raw["game_type_code"] == "R") & (pitches_raw["season"] != 2020)
    ].copy()
    pitches = pitches.sort_values(
        ["game_pk", "at_bat_index", "pitch_number"], na_position="last"
    ).reset_index(drop=True)
    del pitches_raw
    log.info(f"  {len(pitches):,} regular-season pitch rows")

    gf_cols = ["game_pk", "game_date", "home_team_id", "away_team_id",
               "probable_pitcher_home_id", "probable_pitcher_away_id"]
    gf_cols = [c for c in gf_cols if c in existing.columns]
    game_frame = existing[gf_cols].copy()

    blocks: list[pd.DataFrame] = []

    if not has_bullpen:
        log.info("Computing bullpen workload (BFI) features...")
        bwl = _compute_bullpen_workload_features(pitches, game_frame)
        if len(bwl.columns) > 1:
            blocks.append(bwl.drop(columns=["game_pk"]))
            log.info(f"  +{len(bwl.columns) - 1} bullpen columns")

    if not has_mgr:
        log.info("Computing manager tendency features...")
        mgr = _compute_manager_tendency_features(pitches, game_frame)
        if len(mgr.columns) > 1:
            blocks.append(mgr.drop(columns=["game_pk"]))
            log.info(f"  +{len(mgr.columns) - 1} manager columns")

    if not has_bat_str:
        log.info("Computing bat strength features...")
        bstr = _compute_bat_strength_features(pitches, game_frame)
        if len(bstr.columns) > 1:
            blocks.append(bstr.drop(columns=["game_pk"]))
            log.info(f"  +{len(bstr.columns) - 1} bat strength columns")

    del pitches
    return blocks


def main():
    t_start = time.time()

    # --- Load existing parquet ---
    log.info(f"Loading existing features from {FEATURES_PATH}...")
    existing = pd.read_parquet(FEATURES_PATH)
    pre_stats = verify_parquet(existing, "PRE-APPEND")
    original_cols = set(existing.columns)

    all_new_blocks: list[pd.DataFrame] = []

    # --- 1. Massey per-inning ratings ---
    has_massey = any("diff_massey_" in c for c in existing.columns)
    if not has_massey:
        massey_block = compute_massey_features(existing)
        if massey_block is not None:
            all_new_blocks.append(massey_block)
    else:
        log.info("Massey features already present — skipping")

    # --- 2. Colley per-inning ratings ---
    has_colley = any("diff_colley_" in c for c in existing.columns)
    if not has_colley:
        colley_block = compute_colley_features(existing)
        if colley_block is not None:
            all_new_blocks.append(colley_block)
    else:
        log.info("Colley features already present — skipping")

    # --- 3. BFI + Manager + Bat Strength ---
    pitch_blocks = compute_pending_pitch_features(existing)
    all_new_blocks.extend(pitch_blocks)

    # --- Append and verify ---
    if not all_new_blocks:
        log.info("No new features to append — parquet unchanged")
        return

    new_col_count = sum(len(b.columns) for b in all_new_blocks)
    log.info(f"\nAppending {new_col_count} new columns to parquet...")

    result = pd.concat([existing] + all_new_blocks, axis=1)

    # Integrity checks
    assert len(result) == len(existing), f"Row count changed: {len(existing)} → {len(result)}"
    for col in original_cols:
        if result[col].dtype in (np.float32, np.float64, float):
            orig_sum = existing[col].sum()
            new_sum = result[col].sum()
            if not (pd.isna(orig_sum) and pd.isna(new_sum)):
                assert abs(orig_sum - new_sum) < 1e-3, f"Column {col} corrupted: {orig_sum} → {new_sum}"

    # Post-append verification
    post_stats = verify_parquet(result, "POST-APPEND")
    new_cols = [c for c in result.columns if c not in original_cols]
    verify_new_columns(result, new_cols, "NEW FEATURES")

    # Data availability check: new features should be NaN for early-season games
    # (temporal safety — not enough prior games to fit ratings)
    if "diff_massey_full" in result.columns:
        early_season = result.groupby("season").head(30)
        massey_nan_rate = early_season["diff_massey_full"].isna().mean()
        log.info(f"\n[TEMPORAL SAFETY] Early-season (first 30 games/season) diff_massey_full NaN rate: {massey_nan_rate:.1%}")
        if massey_nan_rate < 0.5:
            log.error("POSSIBLE LEAKAGE: Massey features are too dense early in season!")
            sys.exit(1)

    # Write
    log.info(f"\nWriting to {FEATURES_PATH}...")
    result.to_parquet(FEATURES_PATH, index=False, engine="pyarrow")

    elapsed = time.time() - t_start
    log.info(
        f"\nDone: {len(result):,} rows × {len(result.columns)} cols "
        f"(+{len(new_cols)} new features, {elapsed:.1f}s total)"
    )
    log.info(f"New columns: {sorted(new_cols)}")


if __name__ == "__main__":
    main()
