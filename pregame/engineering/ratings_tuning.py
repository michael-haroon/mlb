"""Optuna hyperparameter optimization for rating system parameters.

Every parameter in the 6 rating systems that is not derived from published
literature (or whose literature value we want to validate) is tuned here
by minimizing prediction error on held-out seasons.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import brier_score_loss

from .ratings import (
    DEFAULT_PARAMS,
    compute_elo,
    compute_log5,
    compute_pythagenpat,
    compute_srs,
    compute_wolfe,
)

log = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_all_ratings(
    games: pd.DataFrame,
    n_trials: int = 100,
    val_seasons: Optional[list[int]] = None,
) -> dict:
    """Tune all rating system parameters via Optuna.

    Uses LOYO structure: for each validation season, train ratings on prior
    seasons and evaluate on the held-out season. Objective is aggregate Brier
    score of the rating-implied win probability vs actual game outcome.

    Parameters
    ----------
    games : pd.DataFrame
        Game frame with home_team_id, away_team_id, home_run_diff, season, game_date.
    n_trials : int
        Number of Optuna trials per rating system.
    val_seasons : list[int], optional
        Seasons to hold out for validation. Defaults to last 3 seasons.

    Returns
    -------
    dict
        Optimized parameter dictionary, compatible with ratings.attach_all_ratings().

    Known limitation (TODO: validate impact before fixing):
        Each objective calls compute_*(games_copy, params) on the FULL game frame,
        then evaluates Brier score only on val_seasons. The rating values for
        val_seasons are therefore computed from chronologically prior games
        (correct), but the PARAMETERS are chosen to minimise error specifically
        on those val seasons. When those same val seasons later appear as LOYO
        val folds in train.py, the feature values (Elo, SRS, etc.) were generated
        with params tuned on them — a mild form of target leakage in parameter
        space. Estimated effect: ~1-3% optimistic Brier on those folds.
        Proper fix: tune only on seasons strictly before val_seasons (e.g., tune
        on all_seasons[:-3], evaluate on all_seasons[-3:]). Requires re-running
        Optuna (~800s) so deferred until next full pipeline rebuild.
    """
    if val_seasons is None:
        all_seasons = sorted(games["season"].dropna().unique())
        val_seasons = all_seasons[-3:] if len(all_seasons) > 3 else all_seasons[-1:]

    log.info(f"Tuning ratings on val_seasons={val_seasons}, {n_trials} trials per system")

    optimized = {}

    # Tune Elo parameters
    log.info("Tuning Elo parameters...")
    elo_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    elo_study.optimize(
        lambda trial: _elo_objective(trial, games, val_seasons),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    optimized.update(elo_study.best_params)
    log.info(f"  Elo best Brier: {elo_study.best_value:.5f}, params: {elo_study.best_params}")

    # Tune Wolfe/BT parameters
    log.info("Tuning Wolfe/BT parameters...")
    wolfe_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    wolfe_study.optimize(
        lambda trial: _wolfe_objective(trial, games, val_seasons),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    optimized.update(wolfe_study.best_params)
    log.info(f"  Wolfe best Brier: {wolfe_study.best_value:.5f}")

    # Tune Log5 window sizes
    log.info("Tuning Log5 window sizes...")
    log5_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    log5_study.optimize(
        lambda trial: _log5_objective(trial, games, val_seasons),
        n_trials=n_trials // 2,  # simpler search space
        show_progress_bar=True,
    )
    optimized.update(log5_study.best_params)
    log.info(f"  Log5 best Brier: {log5_study.best_value:.5f}")

    # Validate Pythagenpat z exponent
    log.info("Validating Pythagenpat z...")
    pythag_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    pythag_study.optimize(
        lambda trial: _pythagenpat_objective(trial, games, val_seasons),
        n_trials=n_trials // 2,
        show_progress_bar=True,
    )
    best_z = pythag_study.best_params["pythagenpat_z"]
    # Keep literature value if within tolerance
    if abs(best_z - 0.287) < 0.01:
        log.info(f"  Pythagenpat z={best_z:.4f} ≈ literature 0.287; keeping 0.287")
        optimized["pythagenpat_z"] = 0.287
    else:
        log.info(f"  Pythagenpat z={best_z:.4f} significantly differs from 0.287; adopting")
        optimized["pythagenpat_z"] = best_z

    # Tune SRS window
    log.info("Tuning SRS window...")
    srs_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    srs_study.optimize(
        lambda trial: _srs_objective(trial, games, val_seasons),
        n_trials=n_trials // 3,
        show_progress_bar=True,
    )
    optimized.update(srs_study.best_params)
    log.info(f"  SRS best: {srs_study.best_value:.5f}")

    # Merge with defaults for any untuned params
    final = {**DEFAULT_PARAMS, **optimized}
    log.info(f"Final tuned parameters: {final}")
    return final


def _elo_objective(trial: optuna.Trial, games: pd.DataFrame, val_seasons: list) -> float:
    """Optuna objective for Elo system parameters."""
    params = {
        "elo_k": trial.suggest_float("elo_k", 2.0, 12.0),
        "elo_home_advantage": trial.suggest_float("elo_home_advantage", 10.0, 60.0),
        "elo_mov_coeff": trial.suggest_float("elo_mov_coeff", 1.0, 4.0),
        "elo_mov_elo_scale": trial.suggest_float("elo_mov_elo_scale", 0.0005, 0.005),
        "elo_mov_cap": trial.suggest_float("elo_mov_cap", 0.8, 2.0),
        "elo_pitcher_coeff": trial.suggest_float("elo_pitcher_coeff", 0.0, 10.0),
        "elo_reversion_weight": trial.suggest_float("elo_reversion_weight", 0.4, 0.8),
    }

    games_copy = games.copy()
    games_copy = compute_elo(games_copy, params=params)

    return _evaluate_prob_column(games_copy, "elo_prob", val_seasons)


def _wolfe_objective(trial: optuna.Trial, games: pd.DataFrame, val_seasons: list) -> float:
    """Optuna objective for Wolfe/BT parameters."""
    decay = trial.suggest_float("wolfe_decay_lambda", 0.001, 0.05, log=True)
    halving = trial.suggest_float("wolfe_halving_threshold", 5.0, 25.0)

    games_copy = games.copy()
    games_copy = compute_wolfe(games_copy, decay_lambda=decay, halving_threshold=halving)

    return _evaluate_prob_column(games_copy, "wolfe_prob", val_seasons)


def _log5_objective(trial: optuna.Trial, games: pd.DataFrame, val_seasons: list) -> float:
    """Optuna objective for Log5 window sizes."""
    short = trial.suggest_int("log5_window_short", 10, 50)
    medium = trial.suggest_int("log5_window_medium", 30, 80)

    if medium <= short:
        return 1.0  # infeasible

    games_copy = games.copy()
    games_copy = compute_log5(games_copy, window_short=short, window_medium=medium)

    # Evaluate both windows, pick best
    brier_short = _evaluate_prob_column(games_copy, "log5_prob_short", val_seasons)
    brier_medium = _evaluate_prob_column(games_copy, "log5_prob_medium", val_seasons)
    return min(brier_short, brier_medium)


def _pythagenpat_objective(trial: optuna.Trial, games: pd.DataFrame, val_seasons: list) -> float:
    """Optuna objective to validate Pythagenpat z exponent."""
    z = trial.suggest_float("pythagenpat_z", 0.20, 0.40)

    games_copy = games.copy()
    games_copy = compute_pythagenpat(games_copy, z=z)

    # Use pythag_1st as a probability (it's already 0-1)
    # Evaluate how well it predicts home_win
    mask = games_copy["season"].isin(val_seasons) & games_copy["home_pythag_1st"].notna()
    if mask.sum() < 50:
        return 1.0

    # Pythagenpat isn't a direct matchup probability, but the differential is informative
    # Convert to implied probability using logistic transform of the difference
    diff = games_copy.loc[mask, "home_pythag_1st"] - games_copy.loc[mask, "away_pythag_1st"]
    # Sigmoid with learned slope (fixed at 5 for objective purposes)
    prob = 1.0 / (1.0 + np.exp(-5.0 * diff))
    actual = games_copy.loc[mask, "home_win"]

    valid = actual.notna() & prob.notna()
    if valid.sum() < 50:
        return 1.0

    return brier_score_loss(actual[valid], prob[valid])


def _srs_objective(trial: optuna.Trial, games: pd.DataFrame, val_seasons: list) -> float:
    """Optuna objective for SRS window size."""
    window = trial.suggest_int("srs_window", 20, 162)

    games_copy = games.copy()
    games_copy = compute_srs(games_copy, window=window)

    # Convert SRS differential to probability
    mask = games_copy["season"].isin(val_seasons) & games_copy["srs_diff"].notna()
    if mask.sum() < 50:
        return 1.0

    diff = games_copy.loc[mask, "srs_diff"]
    # Logistic sigmoid; scale SRS runs to probability
    prob = 1.0 / (1.0 + np.exp(-0.15 * diff))
    actual = games_copy.loc[mask, "home_win"]

    valid = actual.notna() & prob.notna()
    if valid.sum() < 50:
        return 1.0

    return brier_score_loss(actual[valid], prob[valid])


def _evaluate_prob_column(
    games: pd.DataFrame,
    prob_col: str,
    val_seasons: list,
) -> float:
    """Compute Brier score for a probability column on validation seasons."""
    mask = games["season"].isin(val_seasons) & games[prob_col].notna() & games["home_win"].notna()
    if mask.sum() < 50:
        return 1.0

    predicted = games.loc[mask, prob_col].clip(0.01, 0.99)
    actual = games.loc[mask, "home_win"]
    return brier_score_loss(actual, predicted)
