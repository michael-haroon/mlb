"""Ensemble redundancy analysis: identify model pairs with correlated predictions/errors.

Downloads game_features.parquet from S3, trains all models within each architectural
family on the same (non-toxic) feature set, then computes pairwise prediction and
error correlation on held-out 2025-2026 data. Recommends drops for pairs that exceed
both thresholds: prediction_corr > 0.95 AND error_corr > 0.90.

Uses only features that survived the importance gate (i.e., NOT in noise/absorbed sets).
"""
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import brier_score_loss

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from classical_learning.strategy.data import load_features
from classical_learning.strategy.models import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET = sys.argv[1] if len(sys.argv) > 1 else "home_win"
HOLDOUT_SEASONS = [2025, 2026]

CLASSIFICATION_TARGETS = {"home_win", "yrfi", "first_5_home_win", "extra_innings"}
TASK = "classification" if TARGET in CLASSIFICATION_TARGETS else "regression"

# Model families grouped by architectural similarity
if TASK == "classification":
    FAMILIES = {
        "bagged_trees": ["random_forest", "extra_trees"],
        "gradient_boosting": ["lightgbm", "xgboost", "catboost", "hist_gradient_boosting"],
        "linear_l1_l2": ["lasso", "elasticnet", "ridge"],
        "linear_classification": ["logistic_regression", "sgd", "bagging_logreg"],
    }
else:
    FAMILIES = {
        "bagged_trees": ["random_forest", "extra_trees"],
        "gradient_boosting": ["lightgbm", "xgboost", "catboost", "hist_gradient_boosting"],
        "linear_l1_l2": ["lasso", "elasticnet", "ridge"],
    }

# Features excluded by importance gate (noise + absorbed)
EXCLUDED_FEATURES = frozenset([
    "away_days_rest", "away_games_last_7d", "diff_roll10_avg", "diff_roll10_obp",
    "diff_roll10_rd_std", "diff_roll20_rd_std", "diff_roll5_avg", "diff_roll5_babip",
    "diff_roll5_obp", "diff_roll5_rd_std", "home_days_rest", "home_games_last_7d",
    "is_dome", "is_doubleheader", "rule_3batter_minimum", "rule_shift_ban_pitch_clock",
    "rule_universal_dh", "sum_roll10_k_rate", "sum_roll5_k9", "sum_roll5_k_rate", "temp_f",
])

PRED_CORR_THRESHOLD = 0.95
ERROR_CORR_THRESHOLD = 0.90


def download_features() -> Path:
    """Download game_features.parquet from S3 to a local temp path."""
    import subprocess

    local_path = PROJECT_ROOT / "tmp" / "game_features.parquet"
    if local_path.exists():
        log.info(f"Using cached {local_path}")
        return local_path

    s3_uri = "s3://mlb-265753586044-us-east-1-an/artifacts/features/game_features.parquet"
    log.info(f"Downloading {s3_uri} -> {local_path}")
    subprocess.run(["aws", "s3", "cp", s3_uri, str(local_path)], check=True)
    return local_path


def filter_non_toxic(X: pd.DataFrame) -> pd.DataFrame:
    """Remove features that the importance gate classified as noise or absorbed."""
    kept = [c for c in X.columns if c not in EXCLUDED_FEATURES]
    dropped = [c for c in X.columns if c in EXCLUDED_FEATURES]
    log.info(f"Kept {len(kept)} features, dropped {len(dropped)} toxic/absorbed features")
    return X[kept]


def train_and_predict(
    family_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray | None:
    """Train a model and return predictions on test set."""
    try:
        model = build_model(family_name, TASK)
        fit_kwargs = {}
        if sample_weight is not None and hasattr(model, "fit"):
            import inspect
            sig = inspect.signature(model.fit)
            if "sample_weight" in sig.parameters:
                fit_kwargs["sample_weight"] = sample_weight

        model.fit(X_train.values, y_train.values, **fit_kwargs)

        if TASK == "classification":
            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(X_test.values)[:, 1]
            elif hasattr(model, "decision_function"):
                from scipy.special import expit
                preds = expit(model.decision_function(X_test.values))
            else:
                log.warning(f"{family_name}: no predict_proba or decision_function")
                return None
        else:
            preds = model.predict(X_test.values)

        return preds
    except Exception as e:
        log.error(f"{family_name} failed: {e}")
        return None


def compute_temporal_weights(seasons_train: pd.Series) -> np.ndarray:
    """Linear decay from 0.05 (oldest) to 1.0 (newest), normalized to sum=N."""
    unique = sorted(seasons_train.unique())
    n_seasons = len(unique)
    weight_map = {s: 0.05 + 0.95 * i / max(n_seasons - 1, 1) for i, s in enumerate(unique)}
    weights = seasons_train.map(weight_map).values.astype(np.float64)
    weights *= len(weights) / weights.sum()
    return weights


def impute_for_linear(X: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray):
    """Median imputation using train-only statistics."""
    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    return X_train, X_test


def scale_for_linear(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """StandardScaler fit on train, transform both."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled


def analyze_family(
    group_name: str,
    family_names: list[str],
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
) -> dict:
    """Train all models in a family group, compute pairwise correlations."""
    from classical_learning.strategy.config import NEEDS_SCALING, NEEDS_IMPUTATION

    train_mask = ~seasons.isin(HOLDOUT_SEASONS)
    test_mask = seasons.isin(HOLDOUT_SEASONS)

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    seasons_train = seasons.iloc[train_idx]

    weights = compute_temporal_weights(seasons_train)

    log.info(f"\n{'='*60}")
    log.info(f"Family group: {group_name}")
    log.info(f"Models: {family_names}")
    log.info(f"Train: {len(train_idx)} games, Test: {len(test_idx)} games")
    log.info(f"{'='*60}")

    predictions = {}
    scores = {}

    for fam in family_names:
        needs_impute = fam in NEEDS_IMPUTATION
        needs_scale = fam in NEEDS_SCALING

        if needs_impute:
            X_tr, X_te = impute_for_linear(X, train_idx, test_idx)
        else:
            X_tr = X.iloc[train_idx]
            X_te = X.iloc[test_idx]

        if needs_scale:
            X_tr, X_te = scale_for_linear(X_tr, X_te)

        log.info(f"  Training {fam}...")
        probs = train_and_predict(fam, X_tr, y_train, X_te, weights if not needs_scale else None)

        if probs is not None:
            predictions[fam] = probs
            if TASK == "classification":
                scores[fam] = brier_score_loss(y_test.values, probs)
                log.info(f"    {fam}: Brier={scores[fam]:.5f}")
            else:
                scores[fam] = float(np.mean((y_test.values - probs) ** 2))
                log.info(f"    {fam}: MSE={scores[fam]:.5f}")
        else:
            log.warning(f"    {fam}: SKIPPED (training failed)")

    # Pairwise analysis
    results = {
        "group": group_name,
        "models": {},
        "pairs": [],
        "recommendations": {},
    }

    for fam in predictions:
        metric_name = "brier" if TASK == "classification" else "mse"
        results["models"][fam] = {metric_name: scores[fam]}

    fam_list = list(predictions.keys())
    for i in range(len(fam_list)):
        for j in range(i + 1, len(fam_list)):
            a, b = fam_list[i], fam_list[j]
            pred_corr, _ = pearsonr(predictions[a], predictions[b])

            # Error correlation: both models' squared errors on same games
            errors_a = (y_test.values - predictions[a]) ** 2
            errors_b = (y_test.values - predictions[b]) ** 2
            error_corr, _ = pearsonr(errors_a, errors_b)

            pair_info = {
                "model_a": a,
                "model_b": b,
                "prediction_correlation": round(pred_corr, 5),
                "error_correlation": round(error_corr, 5),
                "score_a": round(scores[a], 5),
                "score_b": round(scores[b], 5),
                "redundant": pred_corr > PRED_CORR_THRESHOLD and error_corr > ERROR_CORR_THRESHOLD,
            }
            results["pairs"].append(pair_info)

            if pair_info["redundant"]:
                # Drop the one with worse Brier score
                worse = a if scores[a] > scores[b] else b
                better = b if worse == a else a
                results["recommendations"][worse] = {
                    "action": "DROP",
                    "reason": (
                        f"Redundant with {better} "
                        f"(pred_corr={pred_corr:.4f}, error_corr={error_corr:.4f}). "
                        f"{metric_name}: {scores[worse]:.5f} vs {scores[better]:.5f}"
                    ),
                }
                if better not in results["recommendations"]:
                    results["recommendations"][better] = {
                        "action": "KEEP",
                        "reason": f"Better Brier than redundant pair {worse}",
                    }

            log.info(
                f"  {a} vs {b}: pred_corr={pred_corr:.4f}, "
                f"error_corr={error_corr:.4f} {'*** REDUNDANT ***' if pair_info['redundant'] else ''}"
            )

    return results


def main():
    # Download/load features
    features_path = download_features()
    X, y, seasons, game_pks = load_features(features_path, TARGET, data_mode="2016+")

    # Remove toxic features
    X = filter_non_toxic(X)

    log.info(f"Final feature matrix: {X.shape[0]} games × {X.shape[1]} features")
    log.info(f"Holdout seasons: {HOLDOUT_SEASONS}")
    log.info(f"Holdout games: {seasons.isin(HOLDOUT_SEASONS).sum()}")

    all_results = {}
    for group_name, family_names in FAMILIES.items():
        result = analyze_family(group_name, family_names, X, y, seasons)
        all_results[group_name] = result

    # Aggregate recommendations
    final_output = {
        "thresholds": {
            "prediction_correlation": PRED_CORR_THRESHOLD,
            "error_correlation": ERROR_CORR_THRESHOLD,
        },
        "family_results": all_results,
        "global_recommendations": {},
    }

    for group_name, result in all_results.items():
        for model, rec in result["recommendations"].items():
            final_output["global_recommendations"][model] = rec

    # Write output (convert numpy types for JSON)
    output_path = PROJECT_ROOT / "tmp" / f"redundancy_analysis_{TARGET}.json"

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return super().default(obj)

    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2, cls=NumpyEncoder)
    log.info(f"\nResults written to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"ENSEMBLE REDUNDANCY ANALYSIS — {TARGET} ({TASK})")
    print("=" * 70)
    for group_name, result in all_results.items():
        print(f"\n▸ {group_name.upper()}")
        for pair in result["pairs"]:
            flag = " ← REDUNDANT" if pair["redundant"] else ""
            print(
                f"  {pair['model_a']:25s} vs {pair['model_b']:25s} | "
                f"pred_r={pair['prediction_correlation']:.4f}  err_r={pair['error_correlation']:.4f}"
                f"{flag}"
            )

    print("\n" + "-" * 70)
    print("RECOMMENDATIONS")
    print("-" * 70)
    recs = final_output["global_recommendations"]
    if not recs:
        print("  No redundant pairs found — all models contribute orthogonal information.")
    for model, rec in sorted(recs.items(), key=lambda x: x[1]["action"]):
        print(f"  {rec['action']:5s} {model:25s} — {rec['reason']}")


if __name__ == "__main__":
    main()
