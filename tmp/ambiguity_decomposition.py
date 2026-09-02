"""Ambiguity Decomposition (Krogh & Vedelsby 1995) for ensemble pruning.

For each candidate model:
1. Compute ensemble MSE WITH this model (equal-weight average)
2. Compute ensemble MSE WITHOUT this model
3. Delta = MSE_without - MSE_with (positive = model helps)
4. Bootstrap 95% CI on delta over games

A model is genuinely redundant iff delta <= 0 OR CI includes 0.

Additionally computes:
- Per-model R² on held-out
- Ensemble R² vs best single model R²
- Prediction correlation AFTER mean-centering (removes trivial offset)

Uses production routing (per-family feature sets from routing_report.json).
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from classical_learning.strategy.data import load_features
from classical_learning.strategy.models import build_model
from classical_learning.strategy.config import NEEDS_IMPUTATION, NEEDS_SCALING

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET = sys.argv[1] if len(sys.argv) > 1 else "total_runs"
HOLDOUT_SEASONS = [2025, 2026]
CLASSIFICATION_TARGETS = {"home_win", "yrfi", "first_5_home_win", "extra_innings"}
TASK = "classification" if TARGET in CLASSIFICATION_TARGETS else "regression"

N_BOOTSTRAP = 2000
CI_ALPHA = 0.05

# Use away_runs routing report for regression targets (only one available)
ROUTING_REPORT_PATH = (
    PROJECT_ROOT / "data" / "importance" / "away_runs" / "filtered" / "routing_report.json"
)

# For classification, use the home_win routing report
if TASK == "classification":
    ROUTING_REPORT_PATH = (
        PROJECT_ROOT / "pregame" / "artifacts" / "importance" / "home_win" / "filtered" / "routing_report.json"
    )

# Models to evaluate (all tree + linear families)
ALL_FAMILIES = [
    "lightgbm", "xgboost", "catboost", "hist_gradient_boosting",
    "random_forest", "extra_trees",
    "ridge", "lasso", "elasticnet",
]


def load_routing_report() -> dict[str, list[str]]:
    """Load per-family feature sets from routing report."""
    with open(ROUTING_REPORT_PATH) as f:
        report = json.load(f)
    return report["per_family"]


def compute_temporal_weights(seasons_train: pd.Series) -> np.ndarray:
    """Linear decay from 0.05 (oldest) to 1.0 (newest), normalized to sum=N."""
    unique = sorted(seasons_train.unique())
    n_seasons = len(unique)
    weight_map = {s: 0.05 + 0.95 * i / max(n_seasons - 1, 1) for i, s in enumerate(unique)}
    weights = seasons_train.map(weight_map).values.astype(np.float64)
    weights *= len(weights) / weights.sum()
    return weights


def impute_and_scale(X: pd.DataFrame, train_mask: np.ndarray, test_mask: np.ndarray,
                     family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply family-appropriate preprocessing."""
    X_train = X[train_mask].copy()
    X_test = X[test_mask].copy()

    if family in NEEDS_IMPUTATION:
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)

    if family in NEEDS_SCALING:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        cols = X_train.columns
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=cols, index=X_test.index)

    return X_train, X_test


def train_single_model(family: str, X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, weights: np.ndarray) -> np.ndarray | None:
    """Train one model and return predictions on test."""
    import inspect

    try:
        model = build_model(family, TASK)
        fit_kwargs = {}
        sig = inspect.signature(model.fit)
        if "sample_weight" in sig.parameters:
            fit_kwargs["sample_weight"] = weights

        model.fit(X_train.values, y_train.values, **fit_kwargs)

        if TASK == "regression":
            return model.predict(X_test.values)
        else:
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X_test.values)[:, 1]
            elif hasattr(model, "decision_function"):
                from scipy.special import expit
                return expit(model.decision_function(X_test.values))
            return None
    except Exception as e:
        log.error(f"  {family} FAILED: {e}")
        return None


def bootstrap_delta(y_true: np.ndarray, preds_with: np.ndarray,
                    preds_without: np.ndarray, n_boot: int = N_BOOTSTRAP) -> dict:
    """Bootstrap the MSE delta (without - with) and return CI."""
    n = len(y_true)
    rng = np.random.default_rng(42)

    mse_with = np.mean((y_true - preds_with) ** 2)
    mse_without = np.mean((y_true - preds_without) ** 2)
    delta = mse_without - mse_with

    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        mse_w_b = np.mean((y_b - preds_with[idx]) ** 2)
        mse_wo_b = np.mean((y_b - preds_without[idx]) ** 2)
        deltas[b] = mse_wo_b - mse_w_b

    ci_lo = np.percentile(deltas, 100 * CI_ALPHA / 2)
    ci_hi = np.percentile(deltas, 100 * (1 - CI_ALPHA / 2))

    return {
        "delta": float(delta),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "significant": ci_lo > 0,  # True = model significantly helps
    }


def main():
    # Load data
    features_path = PROJECT_ROOT / "tmp" / "game_features.parquet"
    if not features_path.exists():
        import subprocess
        s3_uri = "s3://mlb-265753586044-us-east-1-an/artifacts/features/game_features.parquet"
        log.info(f"Downloading {s3_uri}")
        subprocess.run(["aws", "s3", "cp", s3_uri, str(features_path)], check=True)

    X_all, y, seasons, game_pks = load_features(features_path, TARGET, data_mode="2016+")

    # Load routing
    routing = load_routing_report()

    # Split
    train_mask = ~seasons.isin(HOLDOUT_SEASONS)
    test_mask = seasons.isin(HOLDOUT_SEASONS)
    y_train = y[train_mask]
    y_test = y[test_mask].values
    seasons_train = seasons[train_mask]
    weights = compute_temporal_weights(seasons_train)

    log.info(f"Target: {TARGET} ({TASK})")
    log.info(f"Train: {train_mask.sum()} games, Test: {test_mask.sum()} games")
    log.info(f"Routing report: {ROUTING_REPORT_PATH}")

    # Train all models with their routed feature sets
    predictions = {}
    for family in ALL_FAMILIES:
        # Get routed features, intersect with available columns
        routed_feats = routing.get(family, [])
        available = [f for f in routed_feats if f in X_all.columns]
        if not available:
            log.warning(f"  {family}: no routed features available in parquet — SKIP")
            continue

        X_family = X_all[available]
        X_train, X_test_df = impute_and_scale(X_family, train_mask.values, test_mask.values, family)

        log.info(f"  Training {family} on {len(available)} routed features...")
        preds = train_single_model(family, X_train, y_train, X_test_df, weights)

        if preds is not None:
            predictions[family] = preds
            r2 = 1 - np.sum((y_test - preds) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
            mse = np.mean((y_test - preds) ** 2)
            log.info(f"    {family}: MSE={mse:.4f}, R²={r2:.4f}")

    if len(predictions) < 2:
        log.error("Need at least 2 models for ambiguity decomposition")
        return

    # Compute ensemble (equal-weight average of all models)
    model_names = list(predictions.keys())
    pred_matrix = np.column_stack([predictions[m] for m in model_names])
    ensemble_pred = pred_matrix.mean(axis=1)

    ensemble_mse = np.mean((y_test - ensemble_pred) ** 2)
    ensemble_r2 = 1 - np.sum((y_test - ensemble_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)

    # Per-model metrics
    model_metrics = {}
    for m in model_names:
        mse = np.mean((y_test - predictions[m]) ** 2)
        r2 = 1 - np.sum((y_test - predictions[m]) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
        model_metrics[m] = {"mse": float(mse), "r2": float(r2)}

    best_single = min(model_metrics, key=lambda m: model_metrics[m]["mse"])

    log.info(f"\n{'='*70}")
    log.info(f"Ensemble MSE={ensemble_mse:.4f}, R²={ensemble_r2:.4f}")
    log.info(f"Best single model: {best_single} MSE={model_metrics[best_single]['mse']:.4f}, R²={model_metrics[best_single]['r2']:.4f}")
    log.info(f"Diversity benefit: ensemble MSE is {(model_metrics[best_single]['mse'] - ensemble_mse):.4f} lower than best single")
    log.info(f"{'='*70}")

    # Ambiguity Decomposition: leave-one-out from ensemble
    ambiguity_results = {}
    for i, drop_model in enumerate(model_names):
        # Ensemble WITHOUT this model
        remaining_idx = [j for j in range(len(model_names)) if j != i]
        if not remaining_idx:
            continue
        ensemble_without = pred_matrix[:, remaining_idx].mean(axis=1)

        result = bootstrap_delta(y_test, ensemble_pred, ensemble_without)
        result["model"] = drop_model
        result["model_mse"] = model_metrics[drop_model]["mse"]
        result["model_r2"] = model_metrics[drop_model]["r2"]
        ambiguity_results[drop_model] = result

        status = "HELPS (keep)" if result["significant"] else "REDUNDANT (safe to drop)"
        log.info(
            f"  Drop {drop_model:25s}: Δ={result['delta']:+.5f} "
            f"CI=[{result['ci_lo']:+.5f}, {result['ci_hi']:+.5f}] → {status}"
        )

    # Mean-centered prediction correlations
    log.info(f"\n{'='*70}")
    log.info("Pairwise prediction correlation AFTER mean-centering:")
    log.info(f"{'='*70}")

    centered = {}
    for m in model_names:
        centered[m] = predictions[m] - predictions[m].mean()

    corr_matrix = {}
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a, b = model_names[i], model_names[j]
            r, _ = pearsonr(centered[a], centered[b])
            corr_matrix[f"{a} vs {b}"] = round(float(r), 4)
            log.info(f"  {a:25s} vs {b:25s}: r={r:.4f}")

    # Write results
    output = {
        "target": TARGET,
        "task": TASK,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "ensemble": {
            "mse": float(ensemble_mse),
            "r2": float(ensemble_r2),
            "n_models": len(model_names),
        },
        "best_single_model": {
            "name": best_single,
            "mse": model_metrics[best_single]["mse"],
            "r2": model_metrics[best_single]["r2"],
        },
        "diversity_benefit_mse": float(model_metrics[best_single]["mse"] - ensemble_mse),
        "per_model": model_metrics,
        "ambiguity_decomposition": ambiguity_results,
        "mean_centered_correlations": corr_matrix,
        "models_to_drop": [m for m, r in ambiguity_results.items() if not r["significant"]],
        "models_to_keep": [m for m, r in ambiguity_results.items() if r["significant"]],
    }

    output_path = PROJECT_ROOT / "tmp" / f"ambiguity_decomposition_{TARGET}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nResults written to {output_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"AMBIGUITY DECOMPOSITION — {TARGET} ({TASK})")
    print(f"{'='*80}")
    print(f"\nEnsemble: MSE={ensemble_mse:.4f}, R²={ensemble_r2:.4f} ({len(model_names)} models)")
    print(f"Best single: {best_single} MSE={model_metrics[best_single]['mse']:.4f}, R²={model_metrics[best_single]['r2']:.4f}")
    print(f"Diversity benefit: {(model_metrics[best_single]['mse'] - ensemble_mse):.4f} MSE reduction")

    print(f"\n{'─'*80}")
    print(f"{'Model':<25} {'MSE':>8} {'R²':>8} {'Δ(drop)':>10} {'95% CI':>22} {'Verdict':>12}")
    print(f"{'─'*80}")
    for m in sorted(model_names, key=lambda x: ambiguity_results[x]["delta"], reverse=True):
        r = ambiguity_results[m]
        verdict = "KEEP" if r["significant"] else "DROP"
        print(
            f"{m:<25} {model_metrics[m]['mse']:>8.4f} {model_metrics[m]['r2']:>8.4f} "
            f"{r['delta']:>+10.5f} [{r['ci_lo']:>+.5f}, {r['ci_hi']:>+.5f}] "
            f"{'✓ KEEP' if r['significant'] else '✗ DROP':>12}"
        )

    print(f"\n{'─'*80}")
    print("Mean-centered prediction correlations (offset removed):")
    print(f"{'─'*80}")
    for pair, r in sorted(corr_matrix.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pair:<55} r={r:.4f}")


if __name__ == "__main__":
    main()
