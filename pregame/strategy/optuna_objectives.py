"""Optuna objective functions for per-model-family hyperparameter optimization.

Every model hyperparameter is tuned empirically — no heuristic defaults are
accepted without validation. Each objective function defines the full search
space for its model family and evaluates via inner temporal CV.
"""
from __future__ import annotations

import gc
import logging
from typing import Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from .config import OPTUNA_INNER_CV_SPLITS
from .models import build_model

log = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_objective(
    family: str,
    task: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weights: pd.Series | None = None,
    needs_scaling: bool = False,
) -> Callable[[optuna.Trial], float]:
    """Create an Optuna objective function for a specific model family.

    Parameters
    ----------
    family : str
        Model family name.
    task : str
        "classification" or "regression".
    X_train : pd.DataFrame
        Training features (from LOYO training portion, already imputed if needed).
    y_train : pd.Series
        Training targets.
    sample_weights : pd.Series, optional
        Temporal sample weights.
    needs_scaling : bool
        If True, fit a StandardScaler on each inner fold's training rows and
        apply it to that fold's validation rows. Scaling must happen per fold
        to avoid fitting on rows that are temporally ahead of the val rows.

    Returns
    -------
    Callable that accepts an optuna.Trial and returns the objective value.
    """
    suggest_fn = SUGGEST_FUNCTIONS.get(family)
    if suggest_fn is None:
        raise ValueError(f"No Optuna objective for family {family!r}")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial, task)

        # Also tune sample weight lambda
        weight_lambda = trial.suggest_float("weight_lambda", 0.01, 0.5)

        tscv = TimeSeriesSplit(n_splits=OPTUNA_INNER_CV_SPLITS)
        scores = []

        for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
            X_tr = X_train.iloc[tr_idx]
            X_va = X_train.iloc[va_idx]
            y_tr = y_train.iloc[tr_idx]
            y_va = y_train.iloc[va_idx]

            # Per-fold scaling: fit only on this fold's training rows so val
            # rows are never used to compute scale statistics.
            if needs_scaling:
                from sklearn.preprocessing import StandardScaler
                fold_scaler = StandardScaler()
                X_tr = pd.DataFrame(
                    fold_scaler.fit_transform(X_tr),
                    columns=X_tr.columns, index=X_tr.index,
                )
                X_va = pd.DataFrame(
                    fold_scaler.transform(X_va),
                    columns=X_va.columns, index=X_va.index,
                )

            # Recompute weights for this inner fold
            if sample_weights is not None:
                w_tr = sample_weights.iloc[tr_idx]
            else:
                w_tr = None

            try:
                model = build_model(family, task, params)

                # Fit with sample weights if supported
                fit_kwargs = {}
                if w_tr is not None and hasattr(model, "fit"):
                    import inspect
                    sig = inspect.signature(model.fit)
                    if "sample_weight" in sig.parameters:
                        fit_kwargs["sample_weight"] = w_tr.values

                model.fit(X_tr, y_tr, **fit_kwargs)

                if task == "classification":
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_va)[:, 1]
                    else:
                        proba = model.decision_function(X_va)
                        proba = 1.0 / (1.0 + np.exp(-proba))  # sigmoid
                    score = log_loss(y_va, proba.clip(0.01, 0.99))
                else:
                    preds = model.predict(X_va)
                    score = mean_absolute_error(y_va, preds)

                scores.append(score)
            except Exception as e:
                log.debug(f"Trial {trial.number} fold {fold_idx} failed: {e}")
                return float("inf")
            finally:
                del model

            # Pruning: report intermediate value
            trial.report(np.mean(scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    return objective


# ---------------------------------------------------------------------------
# Per-family suggest functions
# ---------------------------------------------------------------------------

def _suggest_lightgbm(trial: optuna.Trial, task: str) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]),
    }


def _suggest_xgboost(trial: optuna.Trial, task: str) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
    }


def _suggest_catboost(trial: optuna.Trial, task: str) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30.0, log=True),
    }


def _suggest_random_forest(trial: optuna.Trial, task: str) -> dict:
    max_depth = trial.suggest_categorical("max_depth_type", ["none", "limited"])
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "max_depth": None if max_depth == "none" else trial.suggest_int("max_depth", 5, 40),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 50),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.8]),
    }


def _suggest_extra_trees(trial: optuna.Trial, task: str) -> dict:
    max_depth = trial.suggest_categorical("max_depth_type", ["none", "limited"])
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "max_depth": None if max_depth == "none" else trial.suggest_int("max_depth", 5, 40),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 50),
    }


def _suggest_hist_gradient_boosting(trial: optuna.Trial, task: str) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100),
    }


def _suggest_adaboost(trial: optuna.Trial, task: str) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
        "base_max_depth": trial.suggest_int("base_max_depth", 1, 5),
    }


def _suggest_logistic_regression(trial: optuna.Trial, task: str) -> dict:
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    params = {
        "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
        "penalty": penalty,
    }
    if penalty == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.1, 0.9)
        params["solver"] = "saga"
    elif penalty == "l1":
        params["solver"] = "saga"
    else:
        params["solver"] = "lbfgs"
    return params


def _suggest_ridge(trial: optuna.Trial, task: str) -> dict:
    return {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}


def _suggest_lasso(trial: optuna.Trial, task: str) -> dict:
    return {"alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True)}


def _suggest_elasticnet(trial: optuna.Trial, task: str) -> dict:
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
        "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
    }


def _suggest_sgd(trial: optuna.Trial, task: str) -> dict:
    if task == "classification":
        loss = trial.suggest_categorical("loss", ["log_loss", "modified_huber"])
    else:
        loss = trial.suggest_categorical("loss", ["huber", "epsilon_insensitive", "squared_epsilon_insensitive"])
    return {
        "loss": loss,
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
    }


def _suggest_knn(trial: optuna.Trial, task: str) -> dict:
    return {"n_neighbors": trial.suggest_int("n_neighbors", 3, 100)}


def _suggest_lda(trial: optuna.Trial, task: str) -> dict:
    solver = trial.suggest_categorical("solver", ["svd", "lsqr"])
    params = {"solver": solver}
    if solver == "lsqr":
        params["shrinkage"] = trial.suggest_categorical("shrinkage", [None, "auto"])
    return params


def _suggest_qda(trial: optuna.Trial, task: str) -> dict:
    return {"reg_param": trial.suggest_float("reg_param", 0.0, 1.0)}


def _suggest_gaussian_nb(trial: optuna.Trial, task: str) -> dict:
    return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-3, log=True)}


def _suggest_mlp(trial: optuna.Trial, task: str) -> dict:
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f"n_units_l{i}", 32, 512))
    return {
        "hidden_layer_sizes": tuple(layers),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-5, 1e-2, log=True),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
    }


def _suggest_bagging_logreg(trial: optuna.Trial, task: str) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 20, 150),
        "max_samples": trial.suggest_float("max_samples", 0.5, 0.95),
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SUGGEST_FUNCTIONS = {
    "lightgbm": _suggest_lightgbm,
    "xgboost": _suggest_xgboost,
    "catboost": _suggest_catboost,
    "random_forest": _suggest_random_forest,
    "extra_trees": _suggest_extra_trees,
    "hist_gradient_boosting": _suggest_hist_gradient_boosting,
    "adaboost": _suggest_adaboost,
    "logistic_regression": _suggest_logistic_regression,
    "ridge": _suggest_ridge,
    "lasso": _suggest_lasso,
    "elasticnet": _suggest_elasticnet,
    "sgd": _suggest_sgd,
    "knn": _suggest_knn,
    "lda": _suggest_lda,
    "qda": _suggest_qda,
    "gaussian_nb": _suggest_gaussian_nb,
    "mlp": _suggest_mlp,
    "bagging_logreg": _suggest_bagging_logreg,
}
