"""Feature-count sizing curve: per-family empirical S* determination.

For each model family, sweeps feature counts (ordered by composite_rank)
to find the optimal count for that specific model type. Uses fixed-param
models on a held-out sizing fold (most recent season) so S* selection
doesn't leak into final LOYO evaluation.

Output: sizing_curve_{target}.json with per-family S* values.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet, Lasso, LogisticRegression, Ridge, RidgeClassifier,
    SGDClassifier, SGDRegressor,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from .config import IMPORTANCE_DIR, NEEDS_IMPUTATION, NEEDS_SCALING, TARGETS_CLASSIFICATION
from .data import compute_temporal_weights, generate_loyo_splits, load_features

log = logging.getLogger(__name__)

# Coarse grid for initial sweep — every 10 features up to 100, then 20s
COARSE_STEP = 10
COARSE_STEP_LARGE = 20
COARSE_THRESHOLD = 100


def _build_fixed_model(family: str, task: str):
    """Build a fixed-param model for sizing (no HPO, fast evaluation)."""
    if family in ("lightgbm", "catboost", "xgboost", "hist_gradient_boosting"):
        if task == "classification":
            return HistGradientBoostingClassifier(
                max_iter=300, max_leaf_nodes=31, learning_rate=0.05,
                min_samples_leaf=50, l2_regularization=1.0,
                max_depth=None, random_state=42,
            )
        return HistGradientBoostingRegressor(
            max_iter=300, max_leaf_nodes=31, learning_rate=0.05,
            min_samples_leaf=50, l2_regularization=1.0,
            max_depth=None, random_state=42,
        )

    if family in ("random_forest", "extra_trees"):
        cls = ExtraTreesClassifier if family == "extra_trees" else RandomForestClassifier
        reg = ExtraTreesRegressor if family == "extra_trees" else RandomForestRegressor
        if task == "classification":
            return cls(n_estimators=200, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1)
        return reg(n_estimators=200, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1)

    if family == "ydf_oblique_gbt":
        from .models import YDFObliqueClassifier, YDFObliqueRegressor
        if task == "classification":
            return YDFObliqueClassifier(
                num_trees=200, max_depth=6, shrinkage=0.1,
                sparse_oblique_projection_density_factor=2.0,
            )
        return YDFObliqueRegressor(
            num_trees=200, max_depth=6, shrinkage=0.1,
            sparse_oblique_projection_density_factor=2.0,
        )

    if family == "adaboost":
        if task == "classification":
            return AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        return AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    if family == "mlp":
        if task == "classification":
            return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True)
        return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True)

    if family == "logistic_regression":
        if task == "classification":
            return LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs", random_state=42)
        return Ridge(alpha=1.0)

    if family == "ridge":
        if task == "classification":
            return RidgeClassifier(alpha=1.0)
        return Ridge(alpha=1.0)

    if family == "lasso":
        if task == "classification":
            return LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=2000, random_state=42)
        return Lasso(alpha=1.0, max_iter=5000)

    if family == "elasticnet":
        if task == "classification":
            return LogisticRegression(penalty="elasticnet", l1_ratio=0.5, solver="saga", C=0.1, max_iter=2000, random_state=42)
        return ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000)

    if family == "sgd":
        if task == "classification":
            return SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=5000, early_stopping=False, random_state=42)
        return SGDRegressor(loss="squared_error", alpha=1e-4, max_iter=5000, early_stopping=False, random_state=42)

    if family == "bagging_logreg":
        if task == "classification":
            return BaggingClassifier(
                estimator=LogisticRegression(C=0.1, max_iter=1000, random_state=42),
                n_estimators=50, max_samples=0.8, random_state=42, n_jobs=-1,
            )
        return BaggingRegressor(
            estimator=Ridge(alpha=1.0),
            n_estimators=50, max_samples=0.8, random_state=42, n_jobs=-1,
        )

    if family == "knn":
        if task == "classification":
            return KNeighborsClassifier(n_neighbors=20, weights="distance", n_jobs=-1)
        return KNeighborsRegressor(n_neighbors=20, weights="distance", n_jobs=-1)

    if family == "lda":
        if task == "classification":
            return LinearDiscriminantAnalysis(solver="svd")
        return Ridge(alpha=1.0)  # LDA is clf-only

    if family == "qda":
        if task == "classification":
            return QuadraticDiscriminantAnalysis(reg_param=0.0)
        return Ridge(alpha=1.0)  # QDA is clf-only

    if family == "gaussian_nb":
        if task == "classification":
            return GaussianNB()
        return Ridge(alpha=1.0)  # GaussianNB is clf-only

    # Unknown family fallback — should not be reached in production
    if task == "classification":
        return HistGradientBoostingClassifier(
            max_iter=300, max_leaf_nodes=31, learning_rate=0.05,
            min_samples_leaf=50, l2_regularization=1.0,
            max_depth=None, random_state=42,
        )
    return HistGradientBoostingRegressor(
        max_iter=300, max_leaf_nodes=31, learning_rate=0.05,
        min_samples_leaf=50, l2_regularization=1.0,
        max_depth=None, random_state=42,
    )


def _generate_cutoffs(max_features: int) -> list[int]:
    """Generate a sweep from 5 to max_features with adaptive step size."""
    cutoffs = []
    n = 5
    while n < COARSE_THRESHOLD and n < max_features:
        cutoffs.append(n)
        n += COARSE_STEP
    while n < max_features:
        cutoffs.append(n)
        n += COARSE_STEP_LARGE
    cutoffs.append(max_features)
    return cutoffs


def _compute_metrics_simple(y_true, y_pred, task: str) -> float:
    """Return primary metric (lower = better)."""
    from sklearn.metrics import log_loss, mean_absolute_error
    if task == "classification":
        return log_loss(y_true, np.clip(y_pred, 1e-7, 1 - 1e-7))
    return mean_absolute_error(y_true, y_pred)


def run_sizing_curve(
    features_path: Path,
    target: str,
    output_dir: Path,
    data_mode: str = "2015+",
    fine_grained: bool = False,
    importance_dir: Path | None = None,
) -> dict:
    """Run per-family sizing curves for one target.

    For each model family, sweeps the full feature range ordered by
    architectural priority and finds the optimal count. Stores per-family S*
    values so training can cap each family independently.

    Parameters
    ----------
    features_path : Path
        Path to game_features.parquet.
    target : str
        Target column name.
    output_dir : Path
        Directory to write sizing_curve_{target}.json.
    data_mode : str
        "2015+" or "all".
    fine_grained : bool
        If True, refine around coarse optimum with single-feature resolution.
    importance_dir : Path, optional
        Override importance directory. Defaults to IMPORTANCE_DIR from config.

    Returns
    -------
    dict with per-family S* values and the global (hist_gradient_boosting) curve.
    """
    from ..analysis.feature_routing import get_feature_set_uncapped

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task = "classification" if target in TARGETS_CLASSIFICATION else "regression"
    log.info(f"[sizing] target={target}, task={task}, mode={data_mode}")

    # Load data
    X, y, seasons, game_pks = load_features(features_path, target, data_mode)

    # Load feature_report.csv for composite_rank ordering
    imp_dir = importance_dir if importance_dir is not None else IMPORTANCE_DIR
    report_path = imp_dir / target / "filtered" / "feature_report.csv"
    if not report_path.exists():
        log.error(f"[sizing] No feature_report.csv at {report_path}")
        return {"target": target, "error": "no_feature_report"}

    report = pd.read_csv(report_path, index_col="feature")
    if "composite_rank" not in report.columns:
        log.error(f"[sizing] feature_report.csv missing composite_rank column")
        return {"target": target, "error": "no_composite_rank"}

    available = set(X.columns)
    max_features = len([f for f in report.index if f in available])
    log.info(f"[sizing] {max_features} ranked features available")

    if max_features < 5:
        return {"target": target, "error": "insufficient_features"}

    # Generate LOYO splits — use last as sizing fold
    splits = generate_loyo_splits(seasons)
    if len(splits) < 3:
        return {"target": target, "error": "insufficient_splits"}

    sizing_split = splits[-1]
    log.info(f"[sizing] Sizing fold: season {sizing_split.val_season}")

    # Pre-slice data for the sizing fold
    y_train = y.iloc[sizing_split.train_idx]
    y_val = y.iloc[sizing_split.val_idx]
    train_seasons = seasons.iloc[sizing_split.train_idx]
    sample_weights = compute_temporal_weights(train_seasons)

    # Size ALL 19 families — uncapped ordering lets the curve explore beyond
    # the routing ceiling so S* can grow OR shrink for every model type.
    ALL_FAMILIES = [
        "lightgbm", "catboost", "xgboost", "hist_gradient_boosting",
        "random_forest", "extra_trees", "ydf_oblique_gbt", "adaboost", "mlp",
        "logistic_regression", "ridge", "lasso", "elasticnet",
        "sgd", "bagging_logreg", "knn", "lda", "qda", "gaussian_nb",
    ]

    per_family_results = {}

    for family in ALL_FAMILIES:
        family_ranked = get_feature_set_uncapped(family, report)
        # Filter to features actually present in the data matrix
        family_ranked = [f for f in family_ranked if f in available]
        family_max = len(family_ranked)

        if family_max < 5:
            log.info(f"[sizing:{family}] Only {family_max} routed features — using all")
            per_family_results[family] = {"optimal_S": family_max, "curve": []}
            continue

        # Generate cutoffs for this family's feature space
        cutoffs = _generate_cutoffs(family_max)

        log.info(f"[sizing:{family}] Sweeping {len(cutoffs)} cutoffs in [5, {family_max}]")

        curve = []
        needs_impute = family in NEEDS_IMPUTATION
        needs_scale  = family in NEEDS_SCALING

        def _eval_cutoff(n_feat: int) -> dict | None:
            selected = family_ranked[:n_feat]
            X_tr_s = X.iloc[sizing_split.train_idx][selected]
            X_va_s = X.iloc[sizing_split.val_idx][selected]
            if needs_impute:
                X_tr_s = X_tr_s.fillna(0)
                X_va_s = X_va_s.fillna(0)
            if needs_scale:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr_s)
                X_va_s = scaler.transform(X_va_s)
            model = _build_fixed_model(family, task)
            try:
                if hasattr(model, "fit") and "sample_weight" in str(type(model).fit.__code__.co_varnames):
                    model.fit(X_tr_s, y_train, sample_weight=sample_weights.values)
                else:
                    model.fit(X_tr_s, y_train)
            except Exception as e:
                log.debug(f"[sizing:{family}] n={n_feat} fit failed: {e}")
                return None

            if task == "classification":
                _pred = lambda arr: model.predict_proba(arr)[:, 1] if hasattr(model, "predict_proba") else model.predict(arr)
            else:
                _pred = model.predict

            val_loss   = _compute_metrics_simple(y_val.values,   _pred(X_va_s), task)
            train_loss = _compute_metrics_simple(y_train.values, _pred(X_tr_s), task)
            return {
                "n_features": n_feat,
                "val_loss":   round(val_loss,   6),
                "train_loss": round(train_loss, 6),
                "overfit_gap": round(val_loss - train_loss, 6),
            }

        for n_feat in cutoffs:
            pt = _eval_cutoff(n_feat)
            if pt is not None:
                curve.append(pt)

        if not curve:
            per_family_results[family] = {"optimal_S": family_max, "curve": []}
            continue

        # Find coarse optimum
        best_point = min(curve, key=lambda r: r["val_loss"])
        coarse_S = best_point["n_features"]

        # Fine-grained refinement around coarse optimum
        if fine_grained and family_max > 20:
            coarse_idx = cutoffs.index(coarse_S) if coarse_S in cutoffs else 0
            lower = cutoffs[coarse_idx - 1] if coarse_idx > 0 else max(3, coarse_S - 10)
            upper = cutoffs[coarse_idx + 1] if coarse_idx < len(cutoffs) - 1 else min(family_max, coarse_S + 10)
            already = {r["n_features"] for r in curve}
            fine_range = [n for n in range(lower, upper + 1) if n not in already]

            log.info(f"[sizing:{family}] Fine pass: {len(fine_range)} points in [{lower}, {upper}]")

            for n_feat in fine_range:
                pt = _eval_cutoff(n_feat)
                if pt is not None:
                    curve.append(pt)

            curve.sort(key=lambda r: r["n_features"])

        # Final S* for this family
        best_point = min(curve, key=lambda r: r["val_loss"])
        optimal_S  = best_point["n_features"]
        all_point  = curve[-1]

        per_family_results[family] = {
            "optimal_S":             optimal_S,
            "optimal_val_loss":      best_point["val_loss"],
            "optimal_train_loss":    best_point["train_loss"],
            "optimal_overfit_gap":   best_point["overfit_gap"],
            "all_features_val_loss":   all_point["val_loss"],
            "all_features_train_loss": all_point["train_loss"],
            "all_features_overfit_gap": all_point["overfit_gap"],
            "routed_features": family_max,
            "curve": curve,
        }

        log.info(
            f"[sizing:{family}] S*={optimal_S}/{family_max} "
            f"(val={best_point['val_loss']:.5f} tr={best_point['train_loss']:.5f} "
            f"gap={best_point['overfit_gap']:+.5f} | "
            f"all: val={all_point['val_loss']:.5f} tr={all_point['train_loss']:.5f} "
            f"gap={all_point['overfit_gap']:+.5f})"
        )

    # Summary uses hist_gradient_boosting as the "headline" S*
    hgb = per_family_results.get("hist_gradient_boosting", {})

    summary = {
        "target": target,
        "task": task,
        "sizing_fold_season": int(sizing_split.val_season),
        "train_seasons": [int(s) for s in sizing_split.train_seasons],
        "primary_metric": "log_loss" if task == "classification" else "mae",
        "total_ranked_features": max_features,
        "per_family": per_family_results,
        "optimal_S": hgb.get("optimal_S", max_features),
        "optimal_val_loss": hgb.get("optimal_val_loss", 0),
        "all_features_val_loss": hgb.get("all_features_val_loss", 0),
        "degradation_from_all": round(
            hgb.get("all_features_val_loss", 0) - hgb.get("optimal_val_loss", 0), 6
        ),
    }

    out_path = output_dir / f"sizing_curve_{target}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"[sizing] Per-family S* written to {out_path}")
    for fam, res in sorted(per_family_results.items()):
        if res.get("optimal_S"):
            log.info(f"  {fam:<25} S*={res['optimal_S']}/{res.get('routed_features', '?')}")

    return summary
