"""Orthogonality-based ensemble construction with SLSQP weight optimization.

1. Filter candidates by metric quality
2. Compute pairwise correlation of OOF predictions
3. Greedy forward selection maximizing diversity
4. SLSQP weight optimization
5. Refit selected members on full training data and persist to pickle
"""
from __future__ import annotations

import gc
import inspect
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import MAX_CORRELATION, MAX_ENSEMBLE_SIZE, METRIC_TOLERANCE, NEEDS_IMPUTATION, NEEDS_SCALING, TARGETS_CLASSIFICATION

log = logging.getLogger(__name__)


def build_ensemble(
    oof_matrix: dict[str, np.ndarray],
    y_true: np.ndarray,
    metrics: dict[str, dict],
    task: str,
    max_size: int = MAX_ENSEMBLE_SIZE,
    max_correlation: float = MAX_CORRELATION,
    metric_tolerance: float = METRIC_TOLERANCE,
) -> dict:
    """Build diversity-optimized ensemble from candidate OOF predictions.

    Parameters
    ----------
    oof_matrix : dict[str, np.ndarray]
        Mapping of candidate_name → OOF predictions array.
    y_true : np.ndarray
        Ground truth for the OOF predictions.
    metrics : dict[str, dict]
        Mapping of candidate_name → metric dict (from evaluate.py).
    task : str
        "classification" or "regression".
    max_size : int
        Maximum ensemble members.
    max_correlation : float
        Orthogonality threshold (reject if max corr to any existing member > this).
    metric_tolerance : float
        Keep candidates within this factor of the best metric.

    Returns
    -------
    dict with keys: members, weights, correlation_matrix, ensemble_metrics
    """
    if not oof_matrix:
        return {"members": [], "weights": [], "error": "no candidates"}

    # --- Step 1: Filter candidates ---
    # Classification: drop models in the AUC dead zone [0.49, 0.51] after inversion.
    # These have no structural signal — calibration and SLSQP cannot recover them,
    # and including them adds optimizer variance for zero expected gain.
    # SLSQP's non-negativity + sum-to-one constraints act as algorithmic pruning for
    # everything else: redundant models get w=0, weak-but-uncorrelated models get
    # a small non-zero weight. No log_loss cutoff needed post-AUC filter.
    #
    # Regression: no pre-filter. SLSQP handles pruning entirely.
    if task == "classification":
        from sklearn.metrics import roc_auc_score
        viable = {}
        for name, arr in oof_matrix.items():
            valid = ~np.isnan(y_true) & ~np.isnan(arr)
            if valid.sum() < 50:
                log.info(f"  {name}: skipped (< 50 valid OOF rows)")
                continue
            try:
                auc = roc_auc_score(y_true[valid], arr[valid])
            except Exception:
                log.info(f"  {name}: skipped (AUC computation failed)")
                continue
            # Invert if signal is flipped
            if auc < 0.5:
                auc = 1.0 - auc
                log.info(f"  {name}: AUC < 0.5 — signal inverted (AUC now {auc:.4f})")
            if 0.49 <= auc <= 0.51:
                log.info(f"  {name}: AUC={auc:.4f} in dead zone [0.49, 0.51] — excluded")
                continue
            viable[name] = auc
        log.info(f"Filtering: {len(oof_matrix)} candidates → {len(viable)} survivors (AUC dead-zone filter)")
    else:
        # For regression all candidates with OOF arrays are viable; SLSQP prunes.
        viable = {name: 0.0 for name in oof_matrix}
        log.info(f"Regression: passing all {len(viable)} candidates to SLSQP")

    if not viable:
        return {"members": [], "weights": [], "error": "all candidates in AUC dead zone"}

    # --- Step 2: Correlation matrix of OOF predictions ---
    viable_names = sorted(viable)
    n_viable = len(viable_names)

    # Stack predictions into matrix (n_samples × n_candidates)
    valid_mask = ~np.isnan(y_true)
    for name in viable_names:
        valid_mask &= ~np.isnan(oof_matrix[name])

    pred_matrix = np.column_stack([oof_matrix[name][valid_mask] for name in viable_names])
    corr_matrix = np.corrcoef(pred_matrix.T)

    # --- Step 3: Greedy forward selection ---
    # Seed with the model that has the strongest standalone signal.
    # Classification: highest AUC (viable values are AUC scores).
    # Regression: viable values are all 0.0 (no ranking); fall back to first name.
    if task == "classification":
        best_candidate = max(viable_names, key=lambda n: viable[n])
    else:
        best_candidate = viable_names[0]
    selected = [best_candidate]
    remaining = [n for n in viable_names if n != best_candidate]

    while len(selected) < max_size and remaining:
        best_next = None
        best_max_corr = 1.0

        for candidate in remaining:
            cand_idx = viable_names.index(candidate)
            # Max correlation with any already-selected member
            max_corr_to_selected = max(
                abs(corr_matrix[cand_idx, viable_names.index(s)])
                for s in selected
            )

            if max_corr_to_selected < best_max_corr:
                best_max_corr = max_corr_to_selected
                best_next = candidate

        if best_next is None or best_max_corr > max_correlation:
            break

        selected.append(best_next)
        remaining.remove(best_next)
        log.info(f"  Added {best_next} (max_corr={best_max_corr:.3f}, ensemble size={len(selected)})")

    log.info(f"Selected {len(selected)} ensemble members")

    # --- Step 4: SLSQP weight optimization ---
    selected_preds = np.column_stack([oof_matrix[name][valid_mask] for name in selected])
    y_valid = y_true[valid_mask]

    # For classification, calibrate each model's OOF with isotonic regression before
    # optimizing weights. This ensures SLSQP sees calibrated probabilities — the same
    # transformation applied at inference — so the optimized weights reflect production
    # behavior rather than raw uncalibrated score scales.
    if task == "classification":
        from .calibration import fit_isotonic_per_model
        calibrated_preds = np.column_stack([
            fit_isotonic_per_model(selected_preds[:, i], y_valid).predict(selected_preds[:, i])
            for i in range(selected_preds.shape[1])
        ])
        weights = _optimize_weights(calibrated_preds, y_valid, task)
    else:
        calibrated_preds = selected_preds
        weights = _optimize_weights(selected_preds, y_valid, task)

    # --- Compute ensemble metrics ---
    # Use calibrated_preds for classification so reported metrics reflect true
    # production behavior (post-isotonic blend), not raw OOF scores.
    ensemble_pred = calibrated_preds @ weights
    from .evaluate import compute_metrics
    ensemble_metrics = compute_metrics(y_valid, ensemble_pred, task)

    return {
        "members": selected,
        "weights": weights.tolist(),
        "correlation_matrix": {
            name: {other: float(corr_matrix[viable_names.index(name), viable_names.index(other)])
                   for other in selected}
            for name in selected
        },
        "ensemble_metrics": ensemble_metrics,
        "candidate_count": n_viable,
    }


def _optimize_weights(
    pred_matrix: np.ndarray,
    y_true: np.ndarray,
    task: str,
) -> np.ndarray:
    """Optimize ensemble weights via SLSQP: minimize loss s.t. Σw=1, w≥0."""
    n_models = pred_matrix.shape[1]

    if task == "classification":
        def objective(w):
            blended = pred_matrix @ w
            blended = np.clip(blended, 0.01, 0.99)
            return -np.mean(y_true * np.log(blended) + (1 - y_true) * np.log(1 - blended))
    else:
        def objective(w):
            blended = pred_matrix @ w
            return np.mean(np.abs(y_true - blended))

    # Constraints: sum to 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    # Bounds: each weight in [0, 1]
    bounds = [(0.0, 1.0)] * n_models
    # Initial: equal weights
    w0 = np.ones(n_models) / n_models

    try:
        result = minimize(objective, w0, method="SLSQP", bounds=bounds,
                          constraints=constraints, options={"maxiter": 1000, "ftol": 1e-10})
        if result.success:
            return result.x
    except Exception as e:
        log.warning(f"Weight optimization failed: {e}")

    # Fallback: equal weights
    log.warning("Falling back to equal weights")
    return np.ones(n_models) / n_models


def predict_ensemble(
    X: np.ndarray,
    models: list,
    weights: np.ndarray,
    task: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ensemble prediction with uncertainty estimate.

    Returns
    -------
    tuple of (blended prediction, ensemble std across members)
    """
    preds = []
    for model in models:
        if task == "classification":
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X)[:, 1]
            else:
                dec = model.decision_function(X)
                p = 1.0 / (1.0 + np.exp(-dec))
        else:
            p = model.predict(X)
        preds.append(p)

    pred_matrix = np.column_stack(preds)
    blended = pred_matrix @ weights
    ensemble_std = pred_matrix.std(axis=1)

    return blended, ensemble_std


def load_ensemble_oof(
    models_dir: Path,
    target: str,
    tier: str,
) -> dict[str, np.ndarray]:
    """Load all saved OOF arrays for a target into a name → array dict.

    Returns a dict keyed by family name (e.g. "lightgbm") mapping to the
    OOF prediction array saved by train_target().
    """
    models_dir = Path(models_dir)
    oof_files = sorted(models_dir.glob(f"oof_{target}_*_{tier}.npy"))

    result = {}
    for f in oof_files:
        # Filename pattern: oof_{target}_{family}_{tier}.npy
        # Strip leading oof_{target}_ and trailing _{tier}.npy
        stem = f.stem  # e.g. oof_home_win_lightgbm_A
        prefix = f"oof_{target}_"
        suffix = f"_{tier}"
        family = stem[len(prefix):]
        if family.endswith(suffix):
            family = family[: -len(suffix)]
        result[family] = np.load(f)
        log.debug(f"Loaded OOF for {family}: shape={result[family].shape}")

    log.info(f"Loaded {len(result)} OOF arrays for target={target} tier={tier}")
    return result


def _prefilter_and_invert(
    oof_matrix: dict[str, np.ndarray],
    y_true: np.ndarray,
    task: str,
) -> dict[str, np.ndarray]:
    """Invert predictions for models with AUC < 0.5; drop dead-zone models (0.49–0.51).

    A model with AUC < 0.5 has inverted signal — (1-p) recovers its information.
    Dead-zone models (AUC ≈ 0.50 after inversion) contribute only variance.
    Regression models are returned unchanged.
    """
    if task != "classification":
        return dict(oof_matrix)

    from sklearn.metrics import roc_auc_score

    result = {}
    for name, arr in oof_matrix.items():
        valid = ~np.isnan(y_true) & ~np.isnan(arr)
        if valid.sum() < 50:
            result[name] = arr
            continue
        try:
            auc = roc_auc_score(y_true[valid], arr[valid])
        except Exception:
            result[name] = arr
            continue

        if auc < 0.5:
            auc = 1.0 - auc
            arr = 1.0 - arr
            log.info(f"  {name}: AUC<0.5 — inverted predictions (AUC now {auc:.4f})")

        if 0.49 <= auc <= 0.51:
            log.info(f"  {name}: AUC={auc:.4f} in dead zone [0.49,0.51] — excluded")
            continue

        result[name] = arr

    log.info(f"_prefilter_and_invert: {len(oof_matrix)} → {len(result)} candidates")
    return result


def _fit_meta_learner(
    pred_matrix: np.ndarray,
    y_true: np.ndarray,
    task: str,
) -> np.ndarray:
    """Constrained meta-learner weights via non-negative regression.

    Classification: LogisticRegression with non-negative constraint enforced via
    positive=True. fit_intercept=False because calibrated OOFs already handle bias.
    Regression: scipy.optimize.nnls for non-negative least squares.
    Returns weight vector normalized to sum=1.
    """
    # Both tasks: non-negative least squares (NNLS) for non-negative weight constraint.
    # For classification, NNLS on the probability matrix is equivalent to minimizing
    # squared error with w≥0; it reliably returns non-negative weights without requiring
    # a `positive=True` flag that varies across sklearn versions.
    from scipy.optimize import nnls
    w, _ = nnls(pred_matrix, y_true)

    total = w.sum()
    if total < 1e-10:
        return np.ones(pred_matrix.shape[1]) / pred_matrix.shape[1]
    return w / total


def compare_ensemble_methods(
    oof_matrix: dict[str, np.ndarray],
    y_true: np.ndarray,
    task: str,
    metrics: dict[str, dict],
) -> dict:
    """Run 8 calibration × ensemble combinations on the same OOF data.

    Combinations: {none, platt, isotonic, temperature} × {slsqp, meta}.
    All combinations share the same candidate selection (greedy diversity filter)
    to keep the comparison fair. Calibrators are fit per-model on the full OOF
    (valid since OOF predictions are already held-out).

    Returns dict keyed by "{cal}+{ens}" with members, weights, ensemble_metrics.
    """
    from .calibration import (
        apply_platt, apply_temperature,
        fit_isotonic_per_model, fit_platt, fit_temperature,
    )
    from .evaluate import compute_metrics

    # Step 1: inversion filter (classification only)
    filtered_oof = _prefilter_and_invert(oof_matrix, y_true, task)
    if not filtered_oof:
        return {"error": "all candidates filtered out"}

    # Step 2: metric quality filter (same logic as build_ensemble)
    primary = "log_loss" if task == "classification" else "mae"
    candidate_scores = {
        name: metrics[name][primary]
        for name in filtered_oof
        if name in metrics and primary in metrics[name]
    }
    if not candidate_scores:
        # fall back: use all filtered candidates
        candidate_scores = {name: 0.0 for name in filtered_oof}

    best_score = min(candidate_scores.values())
    threshold = best_score * METRIC_TOLERANCE
    if task == "classification":
        threshold = min(threshold, 0.6931 + 0.005)  # absolute coinflip floor
    viable_names = sorted(
        name for name, s in candidate_scores.items() if s <= threshold
    )

    # Step 3: greedy diversity selection (same as build_ensemble)
    valid_mask = ~np.isnan(y_true)
    for name in viable_names:
        valid_mask &= ~np.isnan(filtered_oof[name])
    y_valid = y_true[valid_mask]

    if len(viable_names) == 0 or valid_mask.sum() < 50:
        return {"error": "insufficient valid samples after filtering"}

    pred_matrix_raw = np.column_stack([filtered_oof[n][valid_mask] for n in viable_names])
    corr_matrix = np.corrcoef(pred_matrix_raw.T)

    best_candidate = min(viable_names, key=lambda n: candidate_scores.get(n, 0))
    selected = [best_candidate]
    remaining = [n for n in viable_names if n != best_candidate]

    while len(selected) < MAX_ENSEMBLE_SIZE and remaining:
        best_next, best_max_corr = None, 1.0
        for candidate in remaining:
            ci = viable_names.index(candidate)
            max_corr = max(
                abs(corr_matrix[ci, viable_names.index(s)]) for s in selected
            )
            if max_corr < best_max_corr:
                best_max_corr, best_next = max_corr, candidate
        if best_next is None or best_max_corr > MAX_CORRELATION:
            break
        selected.append(best_next)
        remaining.remove(best_next)

    log.info(f"compare_ensemble_methods: {len(selected)} members selected: {selected}")

    # Raw OOF matrix for selected members (valid rows only)
    raw_preds = {n: filtered_oof[n][valid_mask] for n in selected}

    # Step 4: fit calibrators per model on full valid OOF
    calibrators: dict[str, dict] = {}
    for name in selected:
        p = raw_preds[name]
        calibrators[name] = {}
        if task == "classification":
            calibrators[name]["platt"] = fit_platt(p, y_valid)
            calibrators[name]["temperature"] = fit_temperature(p, y_valid)
            calibrators[name]["isotonic"] = fit_isotonic_per_model(p, y_valid)

    def _apply_cal(name: str, method: str) -> np.ndarray:
        p = raw_preds[name]
        if method == "none":
            return p
        elif method == "platt":
            clf, lo, hi = calibrators[name]["platt"]
            return apply_platt(p, clf, lo, hi)
        elif method == "temperature":
            T = calibrators[name]["temperature"]
            return apply_temperature(p, T)
        elif method == "isotonic":
            ir = calibrators[name]["isotonic"]
            return ir.predict(p)
        return p

    results = {}
    cal_methods = ["none", "platt", "isotonic", "temperature"] if task == "classification" else ["none"]
    ens_methods = ["slsqp", "meta"]

    for cal in cal_methods:
        # Build calibrated prediction matrix for this cal method
        cal_preds = np.column_stack([_apply_cal(n, cal) for n in selected])

        for ens in ens_methods:
            key = f"{cal}+{ens}"
            try:
                if ens == "slsqp":
                    w = _optimize_weights(cal_preds, y_valid, task)
                else:
                    w = _fit_meta_learner(cal_preds, y_valid, task)

                blend = cal_preds @ w
                blend = np.clip(blend, 1e-7, 1 - 1e-7) if task == "classification" else blend
                em = compute_metrics(y_valid, blend, task)
                results[key] = {
                    "members": selected,
                    "weights": {n: float(w[i]) for i, n in enumerate(selected)},
                    "ensemble_metrics": em,
                    "n_members": len(selected),
                }
            except Exception as e:
                log.warning(f"  {key} failed: {e}")
                results[key] = {"error": str(e)}

    return results


def fit_and_save_ensemble(
    features_path: Path,
    models_dir: Path,
    target: str,
    tier: str,
    members: list[str],
    weights: list[float],
    data_mode: str = "2015+",
    output_path: Path | None = None,
    oof_matrix: dict | None = None,
) -> Path:
    """Refit selected ensemble members on the full training set and save to pickle.

    After build_ensemble() selects members and optimizes weights from OOF arrays,
    this function refits each selected model on the full dataset (all LOYO folds
    combined) and bundles everything into a deployable pickle.

    Parameters
    ----------
    features_path : Path
        Path to game_features.parquet.
    models_dir : Path
        Directory containing params_{target}_{family}_{tier}.json files.
    target : str
        Target column name.
    tier : str
        Data tier ("A", "B", or "C").
    members : list[str]
        Family names selected by build_ensemble().
    weights : list[float]
        Blend weights aligned with members.
    data_mode : str
        "2015+" or "all".
    output_path : Path, optional
        Destination pickle path. Defaults to models_dir/ensemble_{target}_{tier}.pkl.
    oof_matrix : dict[str, np.ndarray], optional
        Raw OOF arrays (2020-excluded, aligned to y) keyed by family name.
        When provided and task is classification, one IsotonicRegression is fit
        per member on its OOF to match the pre-blend calibration applied during
        weight optimization.

    Returns
    -------
    Path
        Path to the saved ensemble pickle.
    """
    from sklearn.preprocessing import StandardScaler

    from .data import _semantic_impute, compute_temporal_weights, load_features
    from .models import build_model

    models_dir = Path(models_dir)
    output_path = output_path or models_dir / f"ensemble_{target}_{tier}.pkl"
    task = "classification" if target in TARGETS_CLASSIFICATION else "regression"

    # Drop negligible-weight members — SLSQP may return tiny floats (e.g. 1e-8) rather
    # than exact zeros; threshold at 1% so rounding artifacts don't survive into the pickle
    nonzero = [(m, w) for m, w in zip(members, weights) if w >= 0.01]
    if len(nonzero) < len(members):
        dropped = [m for m, w in zip(members, weights) if w == 0]
        log.info(f"Dropping {len(dropped)} zero-weight members: {dropped}")
        members, weights = zip(*nonzero)
        members, weights = list(members), list(weights)

    log.info(f"Refitting {len(members)} ensemble members for {target} on full training data")

    X, y, seasons = load_features(features_path, target, data_mode)

    # 2020 was a 60-game shortened season played under pandemic protocols: no fans,
    # neutral-site bubble games, universal DH for the first time, and dramatically
    # compressed schedule. Every distributional property (run environment, rest
    # patterns, win rates) is a structural outlier — never refit ensemble members
    # on it, as it would corrupt all models' learned weights and biases.
    no2020 = (seasons != 2020).values
    X = X[no2020].reset_index(drop=True)
    y = y[no2020].reset_index(drop=True)
    seasons = seasons[no2020].reset_index(drop=True)

    sample_weights = compute_temporal_weights(seasons)

    # Load importance filter if present — mirrors train_target() behavior
    filter_report = None
    from .config import IMPORTANCE_DIR
    report_path = IMPORTANCE_DIR / target / "filtered" / "feature_report.csv"
    if report_path.exists():
        filter_report = pd.read_csv(report_path, index_col="feature")
        log.info(f"  Importance filter loaded: {len(filter_report)} features")

    member_bundles = []

    for family in members:
        params_file = models_dir / f"params_{target}_{family}_{tier}.json"
        if not params_file.exists():
            raise FileNotFoundError(
                f"Params file not found: {params_file}. Run train first."
            )
        with open(params_file) as f:
            best_params = json.load(f)

        # Resolve feature set (mirrors prepare_fold logic in train.py)
        importance_features = None
        if filter_report is not None:
            from ..analysis.feature_routing import get_feature_set
            importance_features = get_feature_set(family, filter_report)

        X_fit = X[importance_features] if importance_features is not None else X
        feature_columns = list(X_fit.columns)

        # Imputation then scaling — both fit only on training data (the full set here)
        if family in NEEDS_IMPUTATION:
            X_fit = _semantic_impute(X_fit)

        scaler = None
        if family in NEEDS_SCALING:
            scaler = StandardScaler()
            X_arr = scaler.fit_transform(X_fit.values)
        else:
            X_arr = X_fit.values

        model = build_model(family, task, best_params)
        fit_kwargs = {}
        if "sample_weight" in inspect.signature(model.fit).parameters:
            fit_kwargs["sample_weight"] = sample_weights.values

        model.fit(X_arr, y.values, **fit_kwargs)

        # Fit per-model isotonic calibrator on the OOF predictions so that inference
        # mirrors the calibration applied during SLSQP weight optimization. Regression
        # targets skip this — isotonic calibration is classification-only.
        isotonic_cal = None
        if task == "classification" and oof_matrix is not None and family in oof_matrix:
            from .calibration import fit_isotonic_per_model
            oof_arr = oof_matrix[family]
            # oof_arr is aligned to the no-2020 y; guard against any residual length
            # mismatch from truncation at the call site with [:len(y)].
            oof_slice = oof_arr[:len(y)]
            valid = ~np.isnan(y.values) & ~np.isnan(oof_slice)
            if valid.sum() >= 50:
                isotonic_cal = fit_isotonic_per_model(oof_slice[valid], y.values[valid])
                log.info(f"  Fitted isotonic calibrator for {family} on {valid.sum()} OOF rows")
            else:
                log.warning(
                    f"  {family}: only {valid.sum()} valid OOF rows — "
                    "skipping isotonic calibrator (need ≥50)"
                )

        member_bundles.append({
            "family": family,
            "model": model,
            "scaler": scaler,
            "feature_columns": feature_columns,
            "needs_imputation": family in NEEDS_IMPUTATION,
            "isotonic_calibrator": isotonic_cal,
        })

        log.info(f"  Fitted {family} on {X_arr.shape[0]} rows")
        del X_arr
        gc.collect()

    bundle = {
        "target": target,
        "task": task,
        "tier": tier,
        "members": members,
        "weights": np.array(weights),
        "member_bundles": member_bundles,
    }

    with open(output_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    log.info(f"Ensemble saved to {output_path}")
    return output_path
