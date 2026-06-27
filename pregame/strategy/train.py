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
    X, y, seasons = load_features(features_path, target, data_mode)
    splits = generate_loyo_splits(seasons)

    if families is None:
        families = list_families()

    results = {}

    for family in families:
        log.info(f"  [{family}] Starting Optuna HPO ({n_trials} trials)...")
        t0 = time.time()

        try:
            best_params = _run_optuna_hpo(family, task, X, y, seasons, splits, n_trials)
        except Exception as e:
            log.warning(f"  [{family}] Optuna HPO failed: {e}")
            results[family] = {"status": "hpo_failed", "error": str(e)}
            continue

        log.info(f"  [{family}] HPO done in {time.time() - t0:.1f}s. Running LOYO evaluation...")

        # Phase 2: LOYO evaluation with best params
        oof_predictions = np.full(len(y), np.nan)
        fold_metrics = []

        for split in splits:
            try:
                prepared = prepare_fold(X, y, seasons, split, family, tier=tier)
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
                fold_metrics.append(metrics)

            except Exception as e:
                log.warning(f"  [{family}] Fold {split.val_season} failed: {e}")
            finally:
                # Release fitted model and fold data; prepared holds train+val matrices
                del model, prepared
                gc.collect()

        # Aggregate metrics
        if fold_metrics:
            agg_metrics = _aggregate_fold_metrics(fold_metrics)
            elapsed = time.time() - t0

            results[family] = {
                "status": "success",
                "best_params": best_params,
                "fold_metrics": fold_metrics,
                "aggregate_metrics": agg_metrics,
                "elapsed_secs": round(elapsed, 1),
            }

            # Save OOF predictions
            oof_path = output_dir / f"oof_{target}_{family}_{tier}.npy"
            np.save(oof_path, oof_predictions)

            # Save best params
            params_path = output_dir / f"params_{target}_{family}_{tier}.json"
            with open(params_path, "w") as f:
                json.dump(best_params, f, indent=2, default=str)

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
