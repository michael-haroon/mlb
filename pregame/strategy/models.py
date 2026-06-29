"""Model builder for 40+ architectures across tree, linear, and neural families.

All hyperparameters are placeholder defaults — actual values come from Optuna
tuning in optuna_objectives.py. This module only handles instantiation.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def build_model(family: str, task: str, params: dict[str, Any] | None = None):
    """Instantiate a model given family name, task type, and Optuna-tuned params.

    Parameters
    ----------
    family : str
        Model family name (e.g., "lightgbm", "random_forest", "logistic_regression").
    task : str
        "classification" or "regression".
    params : dict, optional
        Optuna-tuned hyperparameters. If None, uses safe minimal defaults.

    Returns
    -------
    Fitted sklearn-compatible estimator (has .fit() and .predict()/.predict_proba()).
    """
    params = params or {}
    builder = MODEL_BUILDERS.get(family)
    if builder is None:
        raise ValueError(f"Unknown model family: {family!r}. Available: {sorted(MODEL_BUILDERS)}")
    return builder(task, params)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_lightgbm(task: str, params: dict):
    import lightgbm as lgb

    common = {
        "n_estimators": params.get("n_estimators", 600),
        "learning_rate": params.get("learning_rate", 0.03),
        "max_depth": params.get("max_depth", 4),
        "num_leaves": params.get("num_leaves", 15),
        "min_child_samples": params.get("min_child_samples", 20),
        "subsample": params.get("subsample", 0.8),
        "colsample_bytree": params.get("colsample_bytree", 0.8),
        "reg_alpha": params.get("reg_alpha", 0.5),
        "reg_lambda": params.get("reg_lambda", 2.0),
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }

    if params.get("boosting_type"):
        common["boosting_type"] = params["boosting_type"]

    if task == "classification":
        return lgb.LGBMClassifier(objective="binary", **common)
    else:
        return lgb.LGBMRegressor(objective="huber", **common)


def _build_xgboost(task: str, params: dict):
    import xgboost as xgb

    common = {
        "n_estimators": params.get("n_estimators", 600),
        "learning_rate": params.get("learning_rate", 0.03),
        "max_depth": params.get("max_depth", 4),
        "min_child_weight": params.get("min_child_weight", 20),
        "subsample": params.get("subsample", 0.8),
        "colsample_bytree": params.get("colsample_bytree", 0.8),
        "reg_alpha": params.get("reg_alpha", 0.5),
        "reg_lambda": params.get("reg_lambda", 2.0),
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    if params.get("booster"):
        common["booster"] = params["booster"]

    if task == "classification":
        return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    else:
        return xgb.XGBRegressor(objective="reg:pseudohuberror", **common)


def _build_catboost(task: str, params: dict):
    from catboost import CatBoostClassifier, CatBoostRegressor

    common = {
        "iterations": params.get("n_estimators", 600),
        "learning_rate": params.get("learning_rate", 0.03),
        "depth": params.get("max_depth", 6),
        "l2_leaf_reg": params.get("l2_leaf_reg", 3.0),
        "random_seed": 42,
        "verbose": 0,
        "thread_count": -1,
    }

    if task == "classification":
        return CatBoostClassifier(loss_function="Logloss", **common)
    else:
        return CatBoostRegressor(loss_function="Huber:delta=1.0", **common)


def _build_random_forest(task: str, params: dict):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    common = {
        "n_estimators": params.get("n_estimators", 500),
        "max_depth": params.get("max_depth", None),
        "min_samples_leaf": params.get("min_samples_leaf", 5),
        "max_features": params.get("max_features", "sqrt"),
        "random_state": 42,
        "n_jobs": -1,
    }

    if task == "classification":
        return RandomForestClassifier(**common)
    else:
        return RandomForestRegressor(**common)


def _build_extra_trees(task: str, params: dict):
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

    common = {
        "n_estimators": params.get("n_estimators", 500),
        "max_depth": params.get("max_depth", None),
        "min_samples_leaf": params.get("min_samples_leaf", 5),
        "random_state": 42,
        "n_jobs": -1,
    }

    if task == "classification":
        return ExtraTreesClassifier(**common)
    else:
        return ExtraTreesRegressor(**common)


def _build_hist_gradient_boosting(task: str, params: dict):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    common = {
        "max_iter": params.get("n_estimators", 300),
        "max_depth": params.get("max_depth", 5),
        "learning_rate": params.get("learning_rate", 0.05),
        "min_samples_leaf": params.get("min_samples_leaf", 20),
        "random_state": 42,
    }

    if task == "classification":
        return HistGradientBoostingClassifier(loss="log_loss", **common)
    else:
        return HistGradientBoostingRegressor(loss="squared_error", **common)


def _build_adaboost(task: str, params: dict):
    from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    n_est = params.get("n_estimators", 100)
    lr = params.get("learning_rate", 0.5)

    if task == "classification":
        base = DecisionTreeClassifier(max_depth=params.get("base_max_depth", 3))
        return AdaBoostClassifier(estimator=base, n_estimators=n_est, learning_rate=lr, random_state=42)
    else:
        base = DecisionTreeRegressor(max_depth=params.get("base_max_depth", 3))
        return AdaBoostRegressor(estimator=base, n_estimators=n_est, learning_rate=lr, random_state=42)


def _build_logistic_regression(task: str, params: dict):
    from sklearn.linear_model import LogisticRegression, Ridge

    if task == "classification":
        l1_ratio = params.get("l1_ratio", 0.0)
        # sklearn 1.8+: penalty= is deprecated. l1_ratio alone controls regularization type:
        # 0.0 → pure L2 (lbfgs), 1.0 → pure L1 (saga), (0,1) → ElasticNet (saga).
        solver = "saga" if l1_ratio > 0 else "lbfgs"
        return LogisticRegression(
            C=params.get("C", 0.1),
            l1_ratio=l1_ratio,
            solver=solver,
            max_iter=2000,
            random_state=42,
        )
    else:
        return Ridge(alpha=params.get("alpha", 1.0), random_state=42)


def _build_ridge(task: str, params: dict):
    from sklearn.linear_model import Ridge, RidgeClassifier

    alpha = params.get("alpha", 1.0)
    if task == "classification":
        return RidgeClassifier(alpha=alpha)
    else:
        return Ridge(alpha=alpha)


def _build_lasso(task: str, params: dict):
    from sklearn.linear_model import Lasso, LogisticRegression

    if task == "classification":
        # l1_ratio=1.0 → pure L1; penalty= removed (deprecated in sklearn 1.8)
        return LogisticRegression(C=1.0 / max(params.get("alpha", 1.0), 1e-6),
                                  l1_ratio=1.0, solver="saga", max_iter=2000, random_state=42)
    else:
        return Lasso(alpha=params.get("alpha", 1.0), max_iter=5000, random_state=42)


def _build_elasticnet(task: str, params: dict):
    from sklearn.linear_model import ElasticNet, LogisticRegression

    if task == "classification":
        # penalty= removed (deprecated sklearn 1.8); l1_ratio in (0,1) → ElasticNet via saga
        return LogisticRegression(
            C=params.get("C", 0.1),
            l1_ratio=params.get("l1_ratio", 0.5),
            solver="saga",
            max_iter=2000,
            random_state=42,
        )
    else:
        return ElasticNet(
            alpha=params.get("alpha", 1.0),
            l1_ratio=params.get("l1_ratio", 0.5),
            max_iter=5000,
            random_state=42,
        )


def _build_sgd(task: str, params: dict):
    from sklearn.linear_model import SGDClassifier, SGDRegressor

    if task == "classification":
        return SGDClassifier(
            loss=params.get("loss", "log_loss"),
            alpha=params.get("alpha", 1e-4),
            max_iter=5000,
            # early_stopping disabled: sklearn uses a random internal holdout which
            # violates temporal ordering — future rows leak into early-stop decisions.
            early_stopping=False,
            random_state=42,
        )
    else:
        return SGDRegressor(
            loss=params.get("loss", "huber"),
            alpha=params.get("alpha", 1e-4),
            max_iter=5000,
            early_stopping=False,
            random_state=42,
        )


def _build_knn(task: str, params: dict):
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    k = params.get("n_neighbors", 20)
    if task == "classification":
        return KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
    else:
        return KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)


def _build_lda(task: str, params: dict):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    if task != "classification":
        # LDA is classification-only; fall back to Ridge for regression
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)

    return LinearDiscriminantAnalysis(
        solver=params.get("solver", "svd"),
        shrinkage=params.get("shrinkage", None),
    )


def _build_qda(task: str, params: dict):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    if task != "classification":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)

    return QuadraticDiscriminantAnalysis(reg_param=params.get("reg_param", 0.0))


def _build_gaussian_nb(task: str, params: dict):
    from sklearn.naive_bayes import GaussianNB

    if task != "classification":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)

    return GaussianNB(var_smoothing=params.get("var_smoothing", 1e-9))


def _build_mlp(task: str, params: dict):
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    hidden = params.get("hidden_layer_sizes", (128, 64))
    common = {
        "hidden_layer_sizes": hidden,
        "learning_rate_init": params.get("learning_rate_init", 0.001),
        "alpha": params.get("alpha", 1e-4),
        "batch_size": params.get("batch_size", 256),
        "max_iter": 1000,
        # early_stopping disabled: sklearn's internal validation split is random,
        # not temporally ordered — future rows would leak into early-stop decisions.
        "early_stopping": False,
        "random_state": 42,
    }

    if task == "classification":
        return MLPClassifier(**common)
    else:
        return MLPRegressor(**common)


def _build_bagging_logreg(task: str, params: dict):
    from sklearn.ensemble import BaggingClassifier, BaggingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    if task == "classification":
        base = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        return BaggingClassifier(
            estimator=base,
            n_estimators=params.get("n_estimators", 50),
            max_samples=params.get("max_samples", 0.8),
            random_state=42,
            n_jobs=-1,
        )
    else:
        base = Ridge(alpha=1.0)
        return BaggingRegressor(
            estimator=base,
            n_estimators=params.get("n_estimators", 50),
            max_samples=params.get("max_samples", 0.8),
            random_state=42,
            n_jobs=-1,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_BUILDERS = {
    "lightgbm": _build_lightgbm,
    "xgboost": _build_xgboost,
    "catboost": _build_catboost,
    "random_forest": _build_random_forest,
    "extra_trees": _build_extra_trees,
    "hist_gradient_boosting": _build_hist_gradient_boosting,
    "adaboost": _build_adaboost,
    "logistic_regression": _build_logistic_regression,
    "ridge": _build_ridge,
    "lasso": _build_lasso,
    "elasticnet": _build_elasticnet,
    "sgd": _build_sgd,
    "knn": _build_knn,
    "lda": _build_lda,
    "qda": _build_qda,
    "gaussian_nb": _build_gaussian_nb,
    "mlp": _build_mlp,
    "bagging_logreg": _build_bagging_logreg,
}


def list_families() -> list[str]:
    """Return all registered model family names."""
    return sorted(MODEL_BUILDERS.keys())
