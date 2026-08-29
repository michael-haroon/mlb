"""Configuration for the MLB pregame strategy module.

Structural parameters only — all model hyperparameters are Optuna-tuned.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
TARGETS_CLASSIFICATION = ["home_win", "yrfi", "first_5_home_win", "extra_innings"]
TARGETS_REGRESSION = [
    "home_run_diff", "total_runs", "home_runs", "away_runs",
    "first_5_home_run_diff", "first_5_total_runs",
]

ALL_TARGETS = TARGETS_CLASSIFICATION + TARGETS_REGRESSION

# SFI class-weight overrides keyed by target name.
# class_weight="balanced" rescales per-class loss so each class contributes
# equally; for near-50% targets (home_win, yrfi, first_5_home_win: 1.0–1.2x
# imbalance) it is a no-op and their null ≈ coin-flip (−0.693) is correct.
# For extra_innings (7.3% positive, 12.8x imbalance), "balanced" pushes OOS
# predictions toward 0.5 — a no-signal tree scores ≈ −0.693 while the true
# base-rate null is ≈ −0.260, so every feature fails the null test.
# class_weight=None lets the tree learn the base rate; no-signal features
# converge to null OOS; signal features correctly exceed it.
SFI_CLASS_WEIGHT_OVERRIDES: dict[str, str | None] = {
    "extra_innings": None,
}

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
SKIP_SEASONS: list[int] = [2020]  # COVID shortened season — structural outlier
SHORT_SEASON = {2020: 60}
LOYO_MIN_TRAIN_SEASONS = 3

# ---------------------------------------------------------------------------
# Optuna HPO
# ---------------------------------------------------------------------------
OPTUNA_N_TRIALS = 100
OPTUNA_INNER_CV_SPLITS = 3
OPTUNA_PRUNER = "MedianPruner"
OPTUNA_PRUNER_STARTUP_TRIALS = 10
OPTUNA_SAMPLER = "TPESampler"
OPTUNA_SEED = 42

# ---------------------------------------------------------------------------
# Ensemble (structural, not model-specific)
# ---------------------------------------------------------------------------
MAX_ENSEMBLE_SIZE = 50
MAX_CORRELATION = 0.95
METRIC_TOLERANCE = 1.20

# ---------------------------------------------------------------------------
# ECE Calibration
# ---------------------------------------------------------------------------
ECE_N_BINS = 15

# ---------------------------------------------------------------------------
# Model families and their NaN handling requirements
# ---------------------------------------------------------------------------
TREE_MODELS = {
    "lightgbm", "xgboost", "catboost", "random_forest",
    "extra_trees", "hist_gradient_boosting", "ydf_oblique_gbt",
}
LINEAR_MODELS = {
    "logistic_regression", "ridge", "lasso", "elasticnet",
    "sgd", "knn", "lda", "qda", "gaussian_nb", "mlp", "bagging_logreg",
}
# AdaBoost uses DecisionTree base estimators but sklearn's implementation
# does NOT handle NaN natively (unlike HistGradientBoosting/XGBoost/LightGBM).
# It must receive imputed data.
NEEDS_IMPUTATION = LINEAR_MODELS | {"adaboost", "extra_trees", "random_forest"}
NEEDS_SCALING = {"logistic_regression", "ridge", "lasso", "elasticnet", "sgd", "knn", "mlp", "bagging_logreg"}

# ---------------------------------------------------------------------------
# Data availability tiers for era-stratified ensemble
# ---------------------------------------------------------------------------
TIER_A_MIN_SEASON = 2015  # Full Statcast
TIER_B_MIN_SEASON = 2008  # PITCHf/x only
TIER_C_MIN_SEASON = None  # All seasons (game-level aggregates)

# ---------------------------------------------------------------------------
# Feature subsets (populated after importance analysis)
# ---------------------------------------------------------------------------
FEATURE_SUBSETS = {
    "ratings_only": None,       # Set after build
    "rolling_short": None,
    "rolling_medium": None,
    "rolling_long": None,
    "pitching": None,
    "context": None,
    "momentum": None,
    "efficiency": None,
    "matchup": None,
    "all_survivors": None,
    "ratings_plus_momentum": None,
    "short_window": None,
    "long_window": None,
}

# ---------------------------------------------------------------------------
# Feature importance routing
# ---------------------------------------------------------------------------
from pathlib import Path

# Importance artifacts directory — train.py auto-detects filter_report.csv
# and applies per-family feature routing when the file exists.
IMPORTANCE_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "importance"

# Sizing artifacts directory — train.py reads sizing_curve_{target}.json from here.
SIZING_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "sizing"

# Rigorous PCA cross-check results (3-method: MDI, MDA, SFI).
# Contains kendall_tau.json per target. Used by feature routing to detect
# whether MDA is structurally grounded (enables cluster-first routing).
PCA_CROSSCHECK_DIR = Path(__file__).resolve().parents[2] / "data" / "importance"
