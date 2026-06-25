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

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
SKIP_SEASONS: list[int] = []
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
    "extra_trees", "hist_gradient_boosting", "adaboost",
}
LINEAR_MODELS = {
    "logistic_regression", "ridge", "lasso", "elasticnet",
    "sgd", "knn", "lda", "qda", "gaussian_nb", "mlp", "bagging_logreg",
}

# Tree models accept NaN natively; linear models need imputation + scaling
NEEDS_IMPUTATION = LINEAR_MODELS
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
