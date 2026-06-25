"""Data loading, splitting, and NaN handling for the strategy module.

Implements the MNAR-aware imputation framework:
- Tree models receive NaN natively (with binary observation masks)
- Linear models receive MICE-imputed + proxy-substituted features
- Temporal sample weighting applied regardless of imputation strategy
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

from .config import (
    LOYO_MIN_TRAIN_SEASONS,
    NEEDS_IMPUTATION,
    NEEDS_SCALING,
    SKIP_SEASONS,
    TIER_A_MIN_SEASON,
    TIER_B_MIN_SEASON,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statcast features and their pre-2015 proxies
# ---------------------------------------------------------------------------
STATCAST_PROXY_MAP = {
    # statcast_col: (proxy_col, proxy_quality, scale_factor)
    # scale_factor converts proxy to approximate Statcast units
    # These are populated during feature discovery; placeholders for now
}

# Features that only exist post-2015 (MNAR when missing)
STATCAST_ONLY_FEATURES: list[str] = []  # Populated at runtime from feature store


@dataclass
class PreparedData:
    """Container for train/val data prepared for a specific model family."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    sample_weights: pd.Series
    feature_columns: list[str]
    imputer: Optional[IterativeImputer] = None
    scaler: Optional[StandardScaler] = None
    observation_masks: Optional[pd.DataFrame] = None


@dataclass
class LOYOSplit:
    """One fold of Leave-One-Year-Out cross-validation."""
    val_season: int
    train_seasons: list[int]
    train_idx: np.ndarray
    val_idx: np.ndarray


def load_features(
    features_path: Path,
    target: str,
    data_mode: str = "2015+",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load game features and extract target column.

    Parameters
    ----------
    features_path : Path
        Path to game_features.parquet.
    target : str
        Target column name.
    data_mode : str
        "2015+" or "all" — controls season filtering.

    Returns
    -------
    tuple of (features DataFrame, target Series, season Series)
    """
    df = pd.read_parquet(features_path)
    log.info(f"Loaded {len(df):,} games × {len(df.columns)} columns from {features_path}")

    # Filter by data mode
    if data_mode == "2015+":
        df = df[df["season"] >= 2015].reset_index(drop=True)
        log.info(f"Filtered to 2015+: {len(df):,} games")
    elif data_mode == "all":
        pass  # use everything
    else:
        raise ValueError(f"Unknown data_mode: {data_mode!r}. Use '2015+' or 'all'.")

    # Filter to non-null target
    if target not in df.columns:
        raise ValueError(f"Target {target!r} not in columns. Available: {sorted(df.columns)}")

    valid_mask = df[target].notna()
    df = df[valid_mask].reset_index(drop=True)
    log.info(f"Valid target rows: {len(df):,}")

    # Separate target and metadata
    meta_cols = ["game_pk", "game_date", "season", "home_team_id", "away_team_id"]
    target_cols = [c for c in df.columns if c.startswith("home_win") or c.startswith("away_win")
                   or c.startswith("total_runs") or c.startswith("home_run") or c.startswith("yrfi")
                   or c.startswith("nrfi") or c.startswith("first_5") or c.startswith("extra_innings")
                   or c.startswith("target_")]
    non_feature_cols = set(meta_cols + target_cols)
    feature_cols = [c for c in df.columns if c not in non_feature_cols
                    and df[c].dtype in ("float32", "float64", "int32", "int64")]

    features = df[feature_cols]
    targets = df[target]
    seasons = df["season"]

    return features, targets, seasons


def generate_loyo_splits(
    seasons: pd.Series,
    skip_seasons: Optional[list[int]] = None,
) -> list[LOYOSplit]:
    """Generate Leave-One-Year-Out cross-validation splits.

    Each fold uses all prior seasons as training and one season as validation.
    Respects temporal ordering (no future data in training).
    """
    if skip_seasons is None:
        skip_seasons = SKIP_SEASONS

    unique_seasons = sorted(seasons.unique())
    splits = []

    for val_season in unique_seasons:
        if val_season in skip_seasons:
            continue

        train_seasons = [s for s in unique_seasons if s < val_season and s not in skip_seasons]

        if len(train_seasons) < LOYO_MIN_TRAIN_SEASONS:
            continue

        train_idx = np.where(seasons.isin(train_seasons))[0]
        val_idx = np.where(seasons == val_season)[0]

        splits.append(LOYOSplit(
            val_season=val_season,
            train_seasons=train_seasons,
            train_idx=train_idx,
            val_idx=val_idx,
        ))

    log.info(f"Generated {len(splits)} LOYO splits (val seasons: {[s.val_season for s in splits]})")
    return splits


def prepare_fold(
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    split: LOYOSplit,
    model_family: str,
    tier: str = "A",
) -> PreparedData:
    """Prepare data for one LOYO fold with model-appropriate NaN handling.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series
        Full target vector.
    seasons : pd.Series
        Season labels.
    split : LOYOSplit
        Train/val indices.
    model_family : str
        Model family name (determines NaN handling strategy).
    tier : str
        "A" (Statcast), "B" (PITCHf/x), or "C" (aggregates only).
    """
    X_train = X.iloc[split.train_idx].copy()
    X_val = X.iloc[split.val_idx].copy()
    y_train = y.iloc[split.train_idx].copy()
    y_val = y.iloc[split.val_idx].copy()
    train_seasons = seasons.iloc[split.train_idx]

    # --- Tier-based feature filtering ---
    feature_cols = list(X_train.columns)
    if tier == "C":
        # Drop Statcast-only features for maximum robustness
        feature_cols = [c for c in feature_cols if c not in STATCAST_ONLY_FEATURES]
    elif tier == "B":
        # Drop features that require post-2015 Statcast data
        feature_cols = [c for c in feature_cols if c not in STATCAST_ONLY_FEATURES
                        or "_pitchfx_" in c or "_velocity_" in c]

    X_train = X_train[feature_cols]
    X_val = X_val[feature_cols]

    # --- Binary observation masks (for tree models with MNAR features) ---
    observation_masks = None
    if model_family in ("lightgbm", "xgboost", "catboost", "random_forest",
                        "extra_trees", "hist_gradient_boosting"):
        # Add _observed binary indicators for columns with >5% NaN in training
        nan_pct = X_train.isna().mean()
        mnar_cols = nan_pct[nan_pct > 0.05].index.tolist()
        if mnar_cols:
            for col in mnar_cols:
                X_train[f"{col}_observed"] = X_train[col].notna().astype("float32")
                X_val[f"{col}_observed"] = X_val[col].notna().astype("float32")
            observation_masks = X_train[[f"{c}_observed" for c in mnar_cols]]
            feature_cols = list(X_train.columns)

    # --- Imputation (only for linear models) ---
    imputer = None
    scaler = None

    if model_family in NEEDS_IMPUTATION:
        # MICE imputation: fit on training fold ONLY
        imputer = IterativeImputer(max_iter=10, random_state=42, sample_posterior=False)
        X_train_arr = imputer.fit_transform(X_train)
        X_val_arr = imputer.transform(X_val)
        X_train = pd.DataFrame(X_train_arr, columns=feature_cols, index=X_train.index)
        X_val = pd.DataFrame(X_val_arr, columns=feature_cols, index=X_val.index)

    if model_family in NEEDS_SCALING:
        scaler = StandardScaler()
        X_train_arr = scaler.fit_transform(X_train)
        X_val_arr = scaler.transform(X_val)
        X_train = pd.DataFrame(X_train_arr, columns=feature_cols, index=X_train.index)
        X_val = pd.DataFrame(X_val_arr, columns=feature_cols, index=X_val.index)

    # --- Temporal sample weighting ---
    sample_weights = compute_temporal_weights(train_seasons)

    return PreparedData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        sample_weights=sample_weights,
        feature_columns=feature_cols,
        imputer=imputer,
        scaler=scaler,
        observation_masks=observation_masks,
    )


def compute_temporal_weights(
    seasons: pd.Series,
    min_weight: float = 0.05,
) -> pd.Series:
    """Compute temporal sample weights — recent seasons weighted much higher.

    Linear interpolation from min_weight (oldest) to 1.0 (newest), then
    normalized so weights sum to n_samples.

    The min_weight=0.05 means pre-2015 games get ~5% of the weight of a
    recent game during gradient updates. This mitigates imputation noise
    in older seasons without discarding them entirely.
    """
    min_season = seasons.min()
    max_season = seasons.max()

    if min_season == max_season:
        return pd.Series(1.0, index=seasons.index)

    # Linear interpolation
    raw_weights = (seasons - min_season) / (max_season - min_season)
    raw_weights = raw_weights.clip(lower=min_weight)

    # Normalize to sum to n_samples
    weights = raw_weights * len(raw_weights) / raw_weights.sum()
    return weights
