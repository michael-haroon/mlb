"""Data loading, splitting, and NaN handling for the strategy module.

Implements semantic imputation:
- Tree models (LightGBM, XGBoost, etc.) receive NaN natively with binary
  observation masks for MNAR-aware splitting.
- All other models (AdaBoost, linear, neural) receive semantically imputed
  features using domain-correct priors per feature group — no iterative
  estimation that can overflow or introduce lookahead.
- Temporal sample weighting applied regardless of imputation strategy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
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

    Uses a strict allowlist of pregame-knowable feature prefixes to prevent
    leakage from postgame box score columns (runs, hits, pitcher stats from
    the game being predicted).

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
    features_path = Path(features_path)
    if features_path.is_dir():
        raise ValueError(
            f"features_path is a directory: {features_path!r}. "
            "Pass the path to a specific .parquet file (e.g. pregame/artifacts/game_features.parquet)."
        )
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

    # Select features using strict pregame allowlist
    feature_cols = _select_pregame_features(df)
    features = df[feature_cols]
    targets = df[target]
    seasons = df["season"]

    log.info(f"Selected {len(feature_cols)} pregame features")
    return features, targets, seasons


# Pregame-knowable feature prefixes. Any numeric column matching one of these
# prefixes is safe to use as a model input — it was observable BEFORE first pitch.
# This is an allowlist, not a blocklist: unlisted columns are excluded by default
# to prevent leakage from postgame box score data.
_PREGAME_FEATURE_PREFIXES = (
    # Rolling stats from prior games (all use shift(1) internally)
    "home_roll", "away_roll",
    # Differentials and sums of rolling features
    "diff_roll", "sum_roll",
    # Schedule context
    "home_days_rest", "away_days_rest",
    "home_games_last_7d", "away_games_last_7d",
    # Win/loss streaks (computed from prior games via shift(1))
    "home_win_streak", "away_win_streak",
    # Rating systems (Elo, Wolfe, Pythagorean, SRS, BaseRuns) — all from prior games
    "home_elo", "away_elo",
    "home_wolfe", "away_wolfe", "wolfe_diff", "wolfe_prob",
    "home_pythag_1st", "home_pythag_2nd", "away_pythag_1st", "away_pythag_2nd",
    "pythag_1st_diff", "pythag_2nd_diff",
    "home_srs", "away_srs", "srs_diff",
    "bsr_offense_diff", "bsr_defense_diff",
    # BSR rolling ratings — exact names only, NOT the *_game variants which
    # are computed from this game's box score
    "home_bsr_offense", "home_bsr_defense",
    "away_bsr_offense", "away_bsr_defense",
    # Matchup probability estimates from rating systems
    "log5_prob", "consensus_home_win_prob", "consensus_home_win_std",
    # Head-to-head from prior meetings
    "h2h_",
    # Venue and weather (known before game)
    "park_factor", "temp_f", "is_dome", "is_night_game", "is_doubleheader",
    "venue_capacity", "venue_latitude", "venue_longitude",
    # Starting pitcher season-level stats (from prior starts, not this game)
    "sp_home_season_era", "sp_home_season_whip",
    "sp_away_season_era", "sp_away_season_whip",
    "sp_era_diff", "sp_whip_diff",
    "sp_home_is_lefty", "sp_away_is_lefty",
    # Structural regime flags
    "rule_",
)


# Columns that match a prefix above but are actually postgame leakers.
# These are excluded even if they match _PREGAME_FEATURE_PREFIXES.
#
# sp_*_season_era / season_whip: MLB boxscore API returns seasonStats after the
#   game completes, so these always include the current outing. Proven via
#   S3 data — 70% of first-start rows had season_era == game_era exactly.
#   Safe replacements (expanding cumulative prior starts) are computed in
#   feature_engineering._starting_pitcher_features() and carry the same names.
#
# home_wins / home_win_pct / away_*: post-game standings from the API.
#   Pre-game win% is captured by home_roll*_winpct (shift(1) rolling).
_POSTGAME_EXCLUSIONS = frozenset({
    "home_bsr_offense_game", "home_bsr_defense_game", "home_bsr_game",
    "away_bsr_offense_game", "away_bsr_defense_game", "away_bsr_game",
    # Raw API season stats — always post-game; safe versions computed from game logs
    "sp_home_season_era_raw", "sp_away_season_era_raw",
    "sp_home_season_whip_raw", "sp_away_season_whip_raw",
    # Post-game standings columns (not extracted from game_builder anymore, but
    # guarded here in case a stale parquet still carries them)
    "home_wins", "home_losses", "home_win_pct",
    "away_wins", "away_losses", "away_win_pct",
})


def _select_pregame_features(df: pd.DataFrame) -> list[str]:
    """Return only columns that are known before first pitch.

    Uses a strict prefix allowlist. Postgame box score columns (runs, hits,
    pitcher game stats, W/L records including today) are excluded even if
    numeric, to prevent target leakage.
    """
    selected = []
    for col in df.columns:
        if df[col].dtype not in ("float32", "float64", "int32", "int64"):
            continue
        if col in _POSTGAME_EXCLUSIONS:
            continue
        for prefix in _PREGAME_FEATURE_PREFIXES:
            if col.startswith(prefix) or col == prefix:
                selected.append(col)
                break
    return selected


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

    # --- Semantic imputation (for models that cannot handle NaN) ---
    scaler = None

    if model_family in NEEDS_IMPUTATION:
        X_train = _semantic_impute(X_train)
        X_val = _semantic_impute(X_val)

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


# ---------------------------------------------------------------------------
# Semantic imputation — domain-correct priors per feature group
# ---------------------------------------------------------------------------

# Mapping: column name pattern → fill value.
# Order matters: first match wins. More specific patterns must come before
# general ones (e.g., "h2h_home_winrate" before generic "roll" catch-all).
_IMPUTATION_RULES: list[tuple[callable, float]] = [
    # Probability features → 0.5 (no-edge prior)
    (lambda c: "winrate" in c or "log5_prob" in c or "consensus_home_win_prob" in c, 0.5),
    # Park factor → 1.0 (league-average multiplier by definition)
    (lambda c: c == "park_factor", 1.0),
    # Days rest → 7 (offseason proxy for first game of season)
    (lambda c: "days_rest" in c, 7.0),
    # SP season ERA → 4.50 (replacement-level prior, 2015-2024 MLB average ~4.2-4.5)
    (lambda c: "season_era" in c or c == "sp_era_diff", 4.50),
    # SP season WHIP → 1.30 (replacement-level prior)
    (lambda c: "season_whip" in c or c == "sp_whip_diff", 1.30),
    # Venue coordinates → 0 (non-informative; these rows are neutral-site games)
    (lambda c: "venue_latitude" in c or "venue_longitude" in c, 0.0),
]

# Default: everything else (rolling stats, streaks, H2H rd_mean, SRS, games_last_7d,
# differentials, sums) → 0. Rationale: these are accumulation features where NaN
# means "no prior data to aggregate." Zero represents absence of signal, not a
# measured zero outcome. For zero-centered stats (SRS, run diff), 0 is also the
# league-average prior.
_DEFAULT_FILL = 0.0


def _semantic_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN with domain-correct static priors.

    No data-dependent estimation (no MICE, no training-set mean) — each fill
    value is a fixed semantic prior derived from the feature's definition.
    This avoids lookahead bias and BayesianRidge overflow.
    """
    if not df.isna().any().any():
        return df

    df = df.copy()
    nan_cols = df.columns[df.isna().any()].tolist()

    for col in nan_cols:
        fill_value = _DEFAULT_FILL
        for matcher, value in _IMPUTATION_RULES:
            if matcher(col):
                fill_value = value
                break
        df[col] = df[col].fillna(fill_value)

    return df
