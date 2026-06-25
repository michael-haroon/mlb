"""Orchestrate feature importance analysis.

Runs MDI, MDA, SFI on the feature store, builds routing tables, and
saves results to artifacts/importance/.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..strategy.config import LOYO_MIN_TRAIN_SEASONS, TARGETS_CLASSIFICATION
from ..strategy.data import compute_temporal_weights, load_features
from .feature_importance import compute_mda, compute_mdi, compute_sfi, denoise_correlation_matrix
from .feature_routing import build_feature_subsets, route_features

log = logging.getLogger(__name__)


def run_importance_analysis(
    features_path: Path,
    output_dir: Path,
    target: str,
    data_mode: str = "2015+",
    n_estimators_mdi: int = 500,
    n_estimators_mda: int = 300,
) -> dict:
    """Run full feature importance pipeline for one target.

    Parameters
    ----------
    features_path : Path
        Path to game_features.parquet.
    output_dir : Path
        Directory to write importance artifacts.
    target : str
        Target column name.
    data_mode : str
        "2015+" or "all".

    Returns
    -------
    dict with routing tables and summary statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task = "classification" if target in TARGETS_CLASSIFICATION else "regression"
    log.info(f"Running importance analysis: target={target}, task={task}, mode={data_mode}")

    t0 = time.time()

    # Load features
    X, y, seasons = load_features(features_path, target, data_mode)
    sample_weight = compute_temporal_weights(seasons)

    log.info(f"Features: {X.shape[1]} columns, {len(X):,} samples")

    # Drop columns with >95% NaN (uninformative)
    nan_pct = X.isna().mean()
    valid_cols = nan_pct[nan_pct < 0.95].index.tolist()
    X = X[valid_cols]
    log.info(f"After NaN filter: {X.shape[1]} columns")

    # --- MDI ---
    log.info("Computing MDI...")
    mdi = compute_mdi(X, y, task, sample_weight, n_estimators=n_estimators_mdi)
    mdi.to_csv(output_dir / f"importance_mdi_{target}.csv", index=False)

    # --- MDA ---
    log.info("Computing MDA...")
    mda = compute_mda(X, y, seasons, task, sample_weight, n_estimators=n_estimators_mda)
    mda.to_csv(output_dir / f"importance_mda_{target}.csv", index=False)

    # --- SFI ---
    log.info("Computing SFI...")
    sfi = compute_sfi(X, y, seasons, task)
    sfi.to_csv(output_dir / f"importance_sfi_{target}.csv", index=False)

    # --- Correlation denoising ---
    log.info("Denoising correlation matrix...")
    X_filled = X.fillna(X.median())
    denoised_corr = denoise_correlation_matrix(X_filled)

    # --- Feature routing ---
    log.info("Routing features...")
    routing = route_features(mdi, mda, sfi)
    feature_subsets = build_feature_subsets(X.columns.tolist(), routing)

    # Save routing
    routing_path = output_dir / f"feature_routing_{target}.json"
    with open(routing_path, "w") as f:
        json.dump(routing, f, indent=2)

    subsets_path = output_dir / f"feature_subsets_{target}.json"
    with open(subsets_path, "w") as f:
        json.dump(feature_subsets, f, indent=2)

    elapsed = time.time() - t0
    log.info(f"Importance analysis complete in {elapsed:.1f}s")

    return {
        "target": target,
        "n_features_input": X.shape[1],
        "n_accepted": len(routing.get("accepted", [])),
        "n_rejected": len(routing.get("rejected", [])),
        "n_subsets": len(feature_subsets),
        "elapsed_secs": round(elapsed, 1),
    }


def _get_loyo_splits(seasons: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate LOYO train/val index pairs for importance analysis."""
    unique_seasons = sorted(seasons.unique())
    splits = []

    for val_season in unique_seasons:
        train_seasons = [s for s in unique_seasons if s < val_season]
        if len(train_seasons) < LOYO_MIN_TRAIN_SEASONS:
            continue

        train_idx = np.where(seasons.isin(train_seasons))[0]
        val_idx = np.where(seasons == val_season)[0]
        splits.append((train_idx, val_idx))

    return splits
