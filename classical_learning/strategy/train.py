"""LOYO cross-validation training loop with Optuna inner HPO.

Two-phase per model family:
1. Optuna HPO on training portion (inner temporal CV)
2. LOYO evaluation with best params (outer temporal CV)
"""
from __future__ import annotations

import gc
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd


def _json_default(obj):
    """JSON serializer for numpy types that json.dump can't handle."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

from .config import (
    OPTUNA_N_TRIALS,
    OPTUNA_PRUNER_STARTUP_TRIALS,
    OPTUNA_SEED,
    SIZING_DIR,
    TARGETS_CLASSIFICATION,
)
from .data import (
    PreparedData,
    compute_temporal_weights,
    generate_loyo_splits,
    load_features,
    prepare_fold,
)
from .evaluate import compute_metrics
from .models import build_model, list_families
from .optuna_objectives import create_objective

log = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_target(
    features_path: Path,
    target: str,
    output_dir: Path,
    data_mode: str = "2015+",
    families: list[str] | None = None,
    n_trials: int = OPTUNA_N_TRIALS,
    tier: str = "A",
    importance_dir: Path | None = None,
) -> dict:
    """Train all model families for one target via LOYO CV with Optuna HPO.

    Parameters
    ----------
    features_path : Path
        Path to game_features.parquet.
    target : str
        Target column name.
    output_dir : Path
        Directory to write model artifacts.
    data_mode : str
        "2015+" or "all".
    families : list[str], optional
        Model families to train. Defaults to all registered families.
    n_trials : int
        Optuna trials per family.
    tier : str
        Data availability tier ("A", "B", or "C").
    importance_dir : Path, optional
        Override importance directory. Defaults to IMPORTANCE_DIR from config.

    Returns
    -------
    dict
        Summary of training results per family.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task = "classification" if target in TARGETS_CLASSIFICATION else "regression"
    log.info(f"Training target={target}, task={task}, mode={data_mode}, tier={tier}")

    # Load data
    X, y, seasons, game_pks = load_features(features_path, target, data_mode)
    splits = generate_loyo_splits(seasons)

    # Auto-detect importance filter: use it if feature_report.csv exists
    filter_report = None
    from .config import IMPORTANCE_DIR, PCA_CROSSCHECK_DIR
    imp_dir = importance_dir if importance_dir is not None else IMPORTANCE_DIR
    report_path = imp_dir / target / "filtered" / "feature_report.csv"
    if report_path.exists():
        filter_report = pd.read_csv(report_path, index_col="feature")
        log.info(f"  Importance filter loaded: {len(filter_report)} features from {report_path}")
    else:
        log.info(f"  No importance filter found at {report_path} — using all features.")

    # Load PCA cross-check results for MDA regime detection
    import json as _json_kt
    pca_crosscheck = None
    crosscheck_path = PCA_CROSSCHECK_DIR / target / "kendall_tau.json"
    if crosscheck_path.exists():
        with open(crosscheck_path) as _f_kt:
            pca_crosscheck = _json_kt.load(_f_kt)
        log.info(f"  PCA crosscheck loaded from {crosscheck_path}")
    mda_importance_dir = PCA_CROSSCHECK_DIR / target

    # Auto-detect per-family sizing caps from sizing_curve_{target}.json.
    # Each family gets its own S* determined by its own model type.
    per_family_sizing = {}
    sizing_path = SIZING_DIR / f"sizing_curve_{target}.json"
    if sizing_path.exists():
        import json as _json
        with open(sizing_path) as _f:
            sizing_data = _json.load(_f)
        if "per_family" in sizing_data:
            for fam, fam_data in sizing_data["per_family"].items():
                if "optimal_S" in fam_data:
                    per_family_sizing[fam] = fam_data["optimal_S"]
            log.info(f"  Per-family sizing loaded: {len(per_family_sizing)} families from {sizing_path.name}")
        elif "optimal_S" in sizing_data and "error" not in sizing_data:
            # Legacy format: single global S* — apply only to hist_gradient_boosting
            per_family_sizing["hist_gradient_boosting"] = sizing_data["optimal_S"]
            log.info(f"  Legacy sizing cap: S*={sizing_data['optimal_S']} (hist_gb only)")

    if families is None:
        families = list_families()

    results = {}

    for family in families:
        # Resolve per-family feature set BEFORE HPO so we skip early and HPO
        # runs on the same filtered feature matrix that LOYO evaluation will use.
        importance_features = None
        if family in per_family_sizing and filter_report is not None:
            from ..analysis.feature_routing import get_feature_set_uncapped
            all_ordered = get_feature_set_uncapped(family, filter_report)
            all_ordered = [f for f in all_ordered if f in set(X.columns)]
            family_s = per_family_sizing[family]
            importance_features = all_ordered[:family_s]
            log.info(f"  [{family}] Empirical S*={family_s}: {len(importance_features)} features (uncapped)")
        elif filter_report is not None:
            from ..analysis.feature_routing import get_feature_set
            importance_features = get_feature_set(family, filter_report)
            log.info(f"  [{family}] Routing-based: {len(importance_features)} features (capped)")

        if importance_features is not None and len(importance_features) == 0:
            log.warning(f"  [{family}] Skipping: importance filter returned 0 features for this target")
            # Remove stale OOF from prior runs so ensemble won't pick it up
            stale_oof = output_dir / f"oof_{target}_{family}_{tier}.npy"
            if stale_oof.exists():
                stale_oof.unlink()
                log.info(f"  [{family}] Removed stale OOF: {stale_oof.name}")
            results[family] = {"status": "no_features"}
            continue

        X_hpo = X[importance_features] if importance_features is not None else X

        log.info(f"  [{family}] Starting Optuna HPO ({n_trials} trials)...")
        t0 = time.time()

        try:
            best_params = _run_optuna_hpo(family, task, X_hpo, y, seasons, splits, n_trials)
        except Exception as e:
            log.warning(f"  [{family}] Optuna HPO failed: {e}")
            results[family] = {"status": "hpo_failed", "error": str(e)}
            continue

        log.info(f"  [{family}] HPO done in {time.time() - t0:.1f}s. Running LOYO evaluation...")

        # Phase 2: LOYO evaluation with best params
        oof_predictions = np.full(len(y), np.nan)
        fold_metrics = []

        for split in splits:
            model = None
            prepared = None
            try:
                prepared = prepare_fold(X, y, seasons, split, family, tier=tier,
                                        importance_features=importance_features)
                model = build_model(family, task, best_params)

                # Fit with sample weights
                fit_kwargs = {}
                if hasattr(model, "fit"):
                    import inspect
                    sig = inspect.signature(model.fit)
                    if "sample_weight" in sig.parameters:
                        fit_kwargs["sample_weight"] = prepared.sample_weights.values

                model.fit(prepared.X_train, prepared.y_train, **fit_kwargs)

                # Predict
                if task == "classification":
                    if hasattr(model, "predict_proba"):
                        preds = model.predict_proba(prepared.X_val)[:, 1]
                    else:
                        dec = model.decision_function(prepared.X_val)
                        preds = 1.0 / (1.0 + np.exp(-dec))
                else:
                    preds = model.predict(prepared.X_val)

                oof_predictions[split.val_idx] = preds

                # Fold metrics
                metrics = compute_metrics(prepared.y_val.values, preds, task)
                metrics["val_season"] = int(split.val_season)

                # Train metrics for overfit detection — prefixed to avoid collisions
                if task == "classification":
                    if hasattr(model, "predict_proba"):
                        train_preds = model.predict_proba(prepared.X_train)[:, 1]
                    else:
                        dec = model.decision_function(prepared.X_train)
                        train_preds = 1.0 / (1.0 + np.exp(-dec))
                else:
                    train_preds = model.predict(prepared.X_train)
                train_metrics = compute_metrics(prepared.y_train.values, train_preds, task)
                for k, v in train_metrics.items():
                    if k != "n_valid":
                        metrics[f"train_{k}"] = v

                primary = "log_loss" if task == "classification" else "mae"
                gap = metrics.get(primary, 0) - metrics.get(f"train_{primary}", 0)
                log.debug(
                    f"  [{family}] fold {split.val_season}: "
                    f"val_{primary}={metrics.get(primary, 0):.4f} "
                    f"train_{primary}={metrics.get(f'train_{primary}', 0):.4f} "
                    f"gap={gap:+.4f}"
                )

                fold_metrics.append(metrics)

            except Exception as e:
                log.warning(f"  [{family}] Fold {split.val_season} failed: {e}")
            finally:
                # Guard against UnboundLocalError when prepare_fold/build_model throws
                # before either variable is assigned — del on unbound local propagates
                # out of finally and kills the entire training run.
                if model is not None:
                    del model
                if prepared is not None:
                    del prepared
                gc.collect()

        # Aggregate metrics
        if fold_metrics:
            agg_metrics = _aggregate_fold_metrics(fold_metrics)
            elapsed = time.time() - t0

            # Resolve actual feature list for reproducibility tracking
            feature_columns = importance_features if importance_features is not None else list(X.columns)

            results[family] = {
                "status": "success",
                "best_params": best_params,
                "feature_columns": feature_columns,
                "fold_metrics": fold_metrics,
                "aggregate_metrics": agg_metrics,
                "elapsed_secs": round(elapsed, 1),
            }

            # Save OOF predictions and game_pk index for key-based alignment
            oof_path = output_dir / f"oof_{target}_{family}_{tier}.npy"
            np.save(oof_path, oof_predictions)
            gpk_path = output_dir / f"oof_game_pks_{target}_{tier}.npy"
            if not gpk_path.exists():
                np.save(gpk_path, game_pks.values)

            # Save best params + feature list so we know exactly what this model used
            params_path = output_dir / f"params_{target}_{family}_{tier}.json"
            params_payload = {"best_params": best_params, "feature_columns": feature_columns}
            with open(params_path, "w") as f:
                json.dump(params_payload, f, indent=2, default=str)

            primary = "log_loss" if task == "classification" else "mae"
            mean_val = agg_metrics.get(primary)
            mean_train = agg_metrics.get(f"train_{primary}")
            if mean_val is not None and mean_train is not None:
                mean_gap = mean_val - mean_train
                fit_label = (
                    "OVERFIT" if mean_gap > 0.10
                    else "underfit" if mean_gap < 0.01 and mean_val > 0.35
                    else "ok"
                )
                log.info(
                    f"  [{family}] fit={fit_label} "
                    f"val_{primary}={mean_val:.4f} "
                    f"train_{primary}={mean_train:.4f} "
                    f"gap={mean_gap:+.4f}"
                )
            log.info(f"  [{family}] Done: {agg_metrics}")
        else:
            results[family] = {"status": "all_folds_failed"}

    # Save summary
    summary_path = output_dir / f"training_summary_{target}_{tier}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)

    return results


def _run_optuna_hpo(
    family: str,
    task: str,
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    splits: list,
    n_trials: int,
) -> dict:
    """Run Optuna HPO on the training portion of the most recent split."""
    # Use the latest split's training data for HPO
    latest_split = splits[-1]
    X_train = X.iloc[latest_split.train_idx]
    y_train = y.iloc[latest_split.train_idx]
    train_seasons = seasons.iloc[latest_split.train_idx]
    sample_weights = compute_temporal_weights(train_seasons)

    # Handle NaN imputation for models that cannot accept missing values.
    # Scaling is intentionally NOT done here — it is applied per inner fold
    # inside create_objective so each fold's scaler is fit only on that fold's
    # training rows. A single scaler fit here would use statistics from rows
    # that are temporally ahead of earlier inner-fold validation rows.
    from .config import NEEDS_IMPUTATION, NEEDS_SCALING
    from .data import _semantic_impute
    if family in NEEDS_IMPUTATION:
        X_train = _semantic_impute(X_train)

    objective = create_objective(
        family, task, X_train, y_train, sample_weights,
        needs_scaling=(family in NEEDS_SCALING),
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=OPTUNA_PRUNER_STARTUP_TRIALS),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    # Drop study before returning — it holds all 100 trial objects in memory
    del study
    gc.collect()
    return best_params


def _aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Compute weighted mean of fold metrics (recent seasons weighted higher)."""
    if not fold_metrics:
        return {}

    seasons = [m["val_season"] for m in fold_metrics]
    max_season = max(seasons)
    min_season = min(seasons)
    span = max(max_season - min_season, 1)

    # Weight recent seasons higher
    weights = [(m["val_season"] - min_season) / span + 0.1 for m in fold_metrics]
    total_w = sum(weights)

    agg = {}
    numeric_keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float))
                    and k != "val_season"]

    for key in numeric_keys:
        values = [m.get(key, 0) for m in fold_metrics]
        agg[key] = sum(v * w for v, w in zip(values, weights)) / total_w
        agg[f"{key}_per_season"] = {int(m["val_season"]): m.get(key) for m in fold_metrics}

    return agg
