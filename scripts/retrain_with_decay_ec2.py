"""Retrain LightGBM with exponential decay weighting + compare with/without new features.

Usage (on EC2):
    python3.11 scripts/retrain_with_decay_ec2.py --target home_win
    python3.11 scripts/retrain_with_decay_ec2.py --target home_win --lambda-decay 0.003
    python3.11 scripts/retrain_with_decay_ec2.py --target yrfi --test-season 2025

Locally (dry-run):
    conda run -n pred python scripts/retrain_with_decay_ec2.py --target home_win --dry-run

This script:
1. Downloads game_features.parquet from S3 (must include 26 new interpretability features)
2. Splits into train (all seasons < test_season) and test (test_season)
3. Computes exponential decay sample weights: w_i = exp(-λ * age_days_i)
4. Trains LightGBM WITH new features (full model)
5. Trains LightGBM WITHOUT new features (baseline)
6. Compares held-out test loss (log_loss, Brier, AUC)
7. Runs TreeSHAP on full model to rank new feature importance
8. Uploads results to S3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.strategy.config import TARGETS_CLASSIFICATION, SKIP_SEASONS
from classical_learning.strategy.data import _select_pregame_features, _PREGAME_FEATURE_PREFIXES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BUCKET = "mlb-265753586044-us-east-1-an"
FEATURES_KEY = "data/features/game_features.parquet"
OUTPUT_PREFIX = "classical_learning/artifacts/retrain_decay"

# 26 new features from interpretability analysis (Tier 1 + Tier 2)
NEW_FEATURES = [
    "home_sp_platoon_divergence_roll5",
    "away_sp_platoon_divergence_roll5",
    "home_sp_platoon_divergence_roll10",
    "away_sp_platoon_divergence_roll10",
    "diff_sp_platoon_divergence_roll5",
    "diff_sp_platoon_divergence_roll10",
    "market_massey_inn3_disagreement",
    "div_strength_interaction",
    "home_sp_fip_vs_rhh_roll5_clipped",
    "home_sp_fip_vs_lhh_roll5_clipped",
    "away_sp_fip_vs_rhh_roll5_clipped",
    "away_sp_fip_vs_lhh_roll5_clipped",
    "home_sp_fip_vs_rhh_roll10_clipped",
    "home_sp_fip_vs_lhh_roll10_clipped",
    "away_sp_fip_vs_rhh_roll10_clipped",
    "away_sp_fip_vs_lhh_roll10_clipped",
    "elo_prob_x_same_division_clipped",
    "pythag_2nd_diff_clipped",
    "consensus_home_win_prob_clipped",
    "diff_massey_inn3_clipped",
    "weather_temp_clipped",
    "elo_prob_same_div_above57",
    "pythag_2nd_diff_sign",
]


def download_features(s3) -> Path:
    log.info(f"Downloading s3://{BUCKET}/{FEATURES_KEY} ...")
    obj = s3.get_object(Bucket=BUCKET, Key=FEATURES_KEY)
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.write(obj["Body"].read())
    tmp.close()
    log.info(f"Downloaded to {tmp.name}")
    return Path(tmp.name)


def compute_exponential_decay_weights(
    dates: pd.Series,
    lambda_decay: float = 0.002,
) -> np.ndarray:
    """Exponential decay: w_i = exp(-λ * age_days_i).

    age_days measured from the most recent date in the series.
    Weights normalized to sum to n_samples (preserves effective gradient scale).

    With λ=0.002 and half-life = ln(2)/λ ≈ 347 days (~1 year):
      - Same season: weight ~0.7-1.0
      - 1 year old: weight ~0.5
      - 2 years old: weight ~0.25
      - 3 years old: weight ~0.12
      - 5 years old: weight ~0.03

    This gives heavy focus on 2024+ when test_season=2026.
    """
    max_date = dates.max()
    age_days = (max_date - dates).dt.days.values.astype(np.float64)
    raw_weights = np.exp(-lambda_decay * age_days)
    normalized = raw_weights * len(raw_weights) / raw_weights.sum()
    return normalized


def select_features(df: pd.DataFrame) -> list[str]:
    """Use the same prefix allowlist as the production pipeline."""
    return _select_pregame_features(df)


def train_lightgbm(X_train, y_train, sample_weight, task, params=None):
    """Train a LightGBM model with given data and weights."""
    import lightgbm as lgb

    default_params = {
        "n_estimators": 800,
        "learning_rate": 0.03,
        "max_depth": 5,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    if task == "classification":
        model = lgb.LGBMClassifier(objective="binary", **default_params)
    else:
        model = lgb.LGBMRegressor(objective="regression", **default_params)

    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def evaluate_model(model, X_test, y_test, task):
    """Compute test-set metrics."""
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    if task == "classification":
        preds = model.predict_proba(X_test)[:, 1]
        preds_clipped = np.clip(preds, 0.01, 0.99)
        metrics = {
            "log_loss": float(log_loss(y_test, preds_clipped)),
            "brier_score": float(brier_score_loss(y_test, preds_clipped)),
            "accuracy": float(accuracy_score(y_test, (preds >= 0.5).astype(int))),
            "n_test": int(len(y_test)),
        }
        if len(np.unique(y_test)) > 1:
            metrics["auc_roc"] = float(roc_auc_score(y_test, preds_clipped))
        return metrics, preds
    else:
        preds = model.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "r2": float(r2_score(y_test, preds)),
            "n_test": int(len(y_test)),
        }
        return metrics, preds


def run_shap_analysis(model, X_train, feature_names, top_n=50):
    """Run TreeSHAP and return global importance ranking."""
    import shap

    log.info(f"Computing TreeSHAP values ({len(X_train)} samples, {len(feature_names)} features)...")
    t0 = time.time()

    subsample_n = min(2000, len(X_train))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_train), subsample_n, replace=False)
    X_sub = X_train.iloc[idx] if hasattr(X_train, "iloc") else X_train[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    elapsed = time.time() - t0
    log.info(f"SHAP done in {elapsed:.1f}s. Top 10:")
    for _, row in importance_df.head(10).iterrows():
        log.info(f"  {row['feature']}: {row['mean_abs_shap']:.6f}")

    return importance_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--test-season", type=int, default=2026,
                        help="Season to hold out for testing (default: 2026)")
    parser.add_argument("--lambda-decay", type=float, default=0.002,
                        help="Exponential decay rate per day (default: 0.002, half-life ~347 days)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run locally without S3 upload")
    args = parser.parse_args()

    target = args.target
    test_season = args.test_season
    lambda_decay = args.lambda_decay
    task = "classification" if target in TARGETS_CLASSIFICATION else "regression"

    log.info(f"{'='*70}")
    log.info(f"Target: {target} | Task: {task} | Test season: {test_season}")
    log.info(f"Lambda decay: {lambda_decay} | Half-life: {np.log(2)/lambda_decay:.0f} days")
    log.info(f"{'='*70}")

    # --- Download data ---
    if args.dry_run:
        local_path = Path("/tmp/mlb_interp/game_features.parquet")
        if not local_path.exists():
            log.error("Dry-run requires local parquet at /tmp/mlb_interp/game_features.parquet")
            sys.exit(1)
        features_path = local_path
        s3 = None
    else:
        s3 = boto3.client("s3")
        features_path = download_features(s3)

    df = pd.read_parquet(features_path)
    log.info(f"Loaded {len(df):,} games × {len(df.columns)} columns")

    # --- Verify new features exist ---
    present_new = [f for f in NEW_FEATURES if f in df.columns]
    missing_new = [f for f in NEW_FEATURES if f not in df.columns]
    if missing_new:
        log.warning(f"Missing {len(missing_new)} new features (run append_interpretability_features.py first):")
        for f in missing_new[:5]:
            log.warning(f"  {f}")
        if len(present_new) == 0:
            log.error("No new features found — cannot run comparison. Exiting.")
            sys.exit(1)
    log.info(f"New features present: {len(present_new)}/{len(NEW_FEATURES)}")

    # --- Filter and split ---
    df = df[~df["season"].isin(SKIP_SEASONS)].reset_index(drop=True)
    df = df[df[target].notna()].reset_index(drop=True)
    df = df[df["season"] >= 2015].reset_index(drop=True)
    log.info(f"After filtering: {len(df):,} games, seasons {df['season'].min()}-{df['season'].max()}")

    train_mask = df["season"] < test_season
    test_mask = df["season"] == test_season

    if test_mask.sum() == 0:
        log.error(f"No data for test season {test_season}!")
        sys.exit(1)

    df_train = df[train_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)
    log.info(f"Train: {len(df_train):,} games ({df_train['season'].min()}-{df_train['season'].max()})")
    log.info(f"Test:  {len(df_test):,} games (season {test_season})")

    # --- Select features ---
    all_features = select_features(df)
    log.info(f"Total pregame features after allowlist: {len(all_features)}")

    features_with_new = [f for f in all_features if f in df_train.columns]
    features_without_new = [f for f in features_with_new if f not in set(present_new)]
    log.info(f"Features WITH new: {len(features_with_new)}")
    log.info(f"Features WITHOUT new: {len(features_without_new)}")
    log.info(f"Delta: {len(features_with_new) - len(features_without_new)} new features")

    # --- Compute exponential decay weights ---
    if "target_game_date" in df_train.columns:
        train_dates = pd.to_datetime(df_train["target_game_date"])
    elif "game_date" in df_train.columns:
        train_dates = pd.to_datetime(df_train["game_date"])
    else:
        log.warning("No date column found — computing decay from season midpoints")
        max_season = df_train["season"].max()
        age_days = (max_season - df_train["season"]) * 365 + 180
        raw_weights = np.exp(-lambda_decay * age_days.values.astype(np.float64))
        sample_weights = raw_weights * len(raw_weights) / raw_weights.sum()
        train_dates = None

    if train_dates is not None:
        sample_weights = compute_exponential_decay_weights(train_dates, lambda_decay)

    # Log weight distribution by season
    for s in sorted(df_train["season"].unique()):
        mask = df_train["season"] == s
        w_mean = sample_weights[mask].mean()
        w_sum_pct = sample_weights[mask].sum() / sample_weights.sum() * 100
        log.info(f"  Season {s}: mean_weight={w_mean:.3f}, total_share={w_sum_pct:.1f}%")

    # --- Prepare data ---
    X_train_full = df_train[features_with_new].copy()
    X_train_base = df_train[features_without_new].copy()
    X_test_full = df_test[features_with_new].copy()
    X_test_base = df_test[features_without_new].copy()
    y_train = df_train[target].values
    y_test = df_test[target].values

    # Drop columns with >95% NaN
    nan_pct_full = X_train_full.isna().mean()
    valid_full = nan_pct_full[nan_pct_full < 0.95].index.tolist()
    X_train_full = X_train_full[valid_full]
    X_test_full = X_test_full[valid_full]

    nan_pct_base = X_train_base.isna().mean()
    valid_base = nan_pct_base[nan_pct_base < 0.95].index.tolist()
    X_train_base = X_train_base[valid_base]
    X_test_base = X_test_base[valid_base]

    log.info(f"After NaN filter: full={X_train_full.shape[1]} features, base={X_train_base.shape[1]} features")

    # --- Train models ---
    log.info("\n" + "="*70)
    log.info("TRAINING: Full model (WITH new features + exponential decay)")
    log.info("="*70)
    t0 = time.time()
    model_full = train_lightgbm(X_train_full, y_train, sample_weights, task)
    log.info(f"Full model trained in {time.time()-t0:.1f}s")

    log.info("\n" + "="*70)
    log.info("TRAINING: Baseline model (WITHOUT new features + exponential decay)")
    log.info("="*70)
    t0 = time.time()
    model_base = train_lightgbm(X_train_base, y_train, sample_weights, task)
    log.info(f"Baseline model trained in {time.time()-t0:.1f}s")

    # --- Also train without decay for comparison ---
    log.info("\n" + "="*70)
    log.info("TRAINING: Full model (WITH new features, UNIFORM weights)")
    log.info("="*70)
    uniform_weights = np.ones(len(y_train))
    t0 = time.time()
    model_full_uniform = train_lightgbm(X_train_full, y_train, uniform_weights, task)
    log.info(f"Full model (uniform) trained in {time.time()-t0:.1f}s")

    # --- Evaluate ---
    log.info("\n" + "="*70)
    log.info("EVALUATION on held-out test set")
    log.info("="*70)

    metrics_full, preds_full = evaluate_model(model_full, X_test_full, y_test, task)
    metrics_base, preds_base = evaluate_model(model_base, X_test_base, y_test, task)
    metrics_full_uniform, preds_uniform = evaluate_model(model_full_uniform, X_test_full, y_test, task)

    log.info(f"\nFull model (decay):    {metrics_full}")
    log.info(f"Baseline (no new, decay): {metrics_base}")
    log.info(f"Full model (uniform):  {metrics_full_uniform}")

    # Compute deltas
    primary = "log_loss" if task == "classification" else "mae"
    delta_features = metrics_full[primary] - metrics_base[primary]
    delta_decay = metrics_full[primary] - metrics_full_uniform[primary]

    log.info(f"\n--- RESULTS ---")
    log.info(f"New features effect (full-decay vs base-decay): {delta_features:+.6f} {primary}")
    log.info(f"Decay effect (full-decay vs full-uniform):      {delta_decay:+.6f} {primary}")
    if task == "classification":
        brier_delta = metrics_full["brier_score"] - metrics_base["brier_score"]
        log.info(f"Brier score delta (features): {brier_delta:+.6f}")

    # --- TreeSHAP on full model ---
    log.info("\n" + "="*70)
    log.info("SHAP ANALYSIS: Full model with decay")
    log.info("="*70)
    shap_df = run_shap_analysis(model_full, X_train_full, list(X_train_full.columns))

    # Highlight new features in SHAP ranking
    shap_df["is_new_feature"] = shap_df["feature"].isin(set(present_new))
    new_in_shap = shap_df[shap_df["is_new_feature"]].copy()
    log.info(f"\nNew features in SHAP ranking:")
    for _, row in new_in_shap.iterrows():
        rank = shap_df.index[shap_df["feature"] == row["feature"]].tolist()[0] + 1
        log.info(f"  Rank {rank:3d}: {row['feature']} (|SHAP|={row['mean_abs_shap']:.6f})")

    # --- Compile results ---
    results = {
        "config": {
            "target": target,
            "task": task,
            "test_season": test_season,
            "lambda_decay": lambda_decay,
            "half_life_days": round(np.log(2) / lambda_decay, 1),
            "n_train": len(df_train),
            "n_test": len(df_test),
            "n_features_full": X_train_full.shape[1],
            "n_features_base": X_train_base.shape[1],
            "n_new_features_present": len(present_new),
        },
        "metrics": {
            "full_decay": metrics_full,
            "baseline_decay": metrics_base,
            "full_uniform": metrics_full_uniform,
        },
        "deltas": {
            "new_features_effect": {primary: float(delta_features)},
            "decay_effect": {primary: float(delta_decay)},
        },
        "weight_distribution": {},
        "shap_top_50": shap_df.head(50)[["feature", "mean_abs_shap", "is_new_feature"]].to_dict("records"),
        "new_features_shap": new_in_shap[["feature", "mean_abs_shap"]].to_dict("records"),
    }

    # Add Brier deltas for classification
    if task == "classification":
        results["deltas"]["new_features_effect"]["brier_score"] = float(
            metrics_full["brier_score"] - metrics_base["brier_score"])
        results["deltas"]["decay_effect"]["brier_score"] = float(
            metrics_full["brier_score"] - metrics_full_uniform["brier_score"])
        if "auc_roc" in metrics_full and "auc_roc" in metrics_base:
            results["deltas"]["new_features_effect"]["auc_roc"] = float(
                metrics_full["auc_roc"] - metrics_base["auc_roc"])

    # Weight distribution
    for s in sorted(df_train["season"].unique()):
        mask = df_train["season"] == s
        results["weight_distribution"][str(s)] = {
            "mean_weight": float(sample_weights[mask].mean()),
            "total_share_pct": float(sample_weights[mask].sum() / sample_weights.sum() * 100),
            "n_games": int(mask.sum()),
        }

    # --- Save / Upload ---
    results_json = json.dumps(results, indent=2, default=str)
    log.info(f"\n{'='*70}")
    log.info("FULL RESULTS:")
    log.info(results_json)

    if not args.dry_run and s3 is not None:
        # Upload results JSON
        key = f"{OUTPUT_PREFIX}/{target}/retrain_decay_results.json"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(results_json)
            f.flush()
            s3.upload_file(f.name, BUCKET, key)
        log.info(f"Uploaded results to s3://{BUCKET}/{key}")

        # Upload SHAP rankings
        shap_key = f"{OUTPUT_PREFIX}/{target}/shap_decay_importance.csv"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            shap_df.to_csv(f.name, index=False)
            s3.upload_file(f.name, BUCKET, shap_key)
        log.info(f"Uploaded SHAP to s3://{BUCKET}/{shap_key}")
    else:
        out_dir = Path("/tmp/mlb_retrain_decay")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"retrain_decay_{target}.json", "w") as f:
            f.write(results_json)
        shap_df.to_csv(out_dir / f"shap_decay_{target}.csv", index=False)
        log.info(f"Saved results to {out_dir}")

    log.info("\nDONE.")


if __name__ == "__main__":
    main()
