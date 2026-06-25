"""Six predictive team rating systems for MLB.

Each rating is computed with strict temporal safety: only games prior to the
current game inform the current game's features (shift(1) or expanding window
on chronologically sorted data).

Rating system parameters are NOT hardcoded heuristics. All tunable parameters
are set via the `params` dict which is optimized by Optuna in ratings_tuning.py.
Default values are from published literature (cited inline) and serve only as
initial search points for Optuna.

Systems implemented:
1. BaseRuns (BsR) — context-neutral run estimation (Smyth, 1990s)
2. Pythagenpat — dynamic-exponent win expectation (Smyth/Patriot, 2004)
3. Simple Rating System (SRS) — iterative least-squares (Baseball-Reference)
4. Pitcher-Adjusted Elo — Markovian logistic updates (FiveThirtyEight style)
5. Wolfe/Bradley-Terry MLE — recency-weighted maximum likelihood
6. Log5 — matchup probability from win percentages (Bill James)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.optimize import minimize

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default parameters (from literature). Overridden by Optuna-tuned values.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    # Pythagenpat: z=0.287 (Smyth/Patriot; Baseball Prospectus 2004)
    "pythagenpat_z": 0.287,
    # Elo: FiveThirtyEight MLB defaults
    "elo_k": 4.0,
    "elo_home_advantage": 24.0,
    "elo_mov_coeff": 2.2,
    "elo_mov_elo_scale": 0.001,
    "elo_mov_cap": 1.25,
    "elo_pitcher_coeff": 4.7,
    "elo_reversion_weight": 0.6,
    # Wolfe/BT: decay and diminishing returns
    "wolfe_decay_lambda": 0.005,
    "wolfe_halving_threshold": 14.0,
    # Log5: rolling window sizes (games)
    "log5_window_short": 20,
    "log5_window_medium": 40,
    # SRS: window and convergence
    "srs_window": 162,
    "srs_tol": 1e-6,
    "srs_max_iter": 100,
}


def attach_all_ratings(
    games: pd.DataFrame,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """Compute and attach all 6 rating systems to the game frame.

    Parameters
    ----------
    games : pd.DataFrame
        Output of game_builder.build_game_frame(). Must be sorted by game_date.
    params : dict, optional
        Optuna-tuned parameters. Falls back to DEFAULT_PARAMS for any missing key.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    log.info("Computing BaseRuns...")
    games = compute_baseruns(games)

    log.info("Computing Pythagenpat (3 tiers)...")
    games = compute_pythagenpat(games, z=p["pythagenpat_z"])

    log.info("Computing SRS...")
    games = compute_srs(games, window=p["srs_window"], tol=p["srs_tol"], max_iter=p["srs_max_iter"])

    log.info("Computing Elo...")
    games = compute_elo(games, params=p)

    log.info("Computing Wolfe/Bradley-Terry...")
    games = compute_wolfe(games, decay_lambda=p["wolfe_decay_lambda"],
                          halving_threshold=p["wolfe_halving_threshold"])

    log.info("Computing Log5...")
    games = compute_log5(games, window_short=p["log5_window_short"],
                         window_medium=p["log5_window_medium"])

    # Implied probability consensus: mean of all rating-system win probabilities
    prob_cols = [c for c in games.columns if c.endswith("_prob") and "home" not in c.lower()]
    if prob_cols:
        games["consensus_prob"] = games[prob_cols].mean(axis=1)

    return games


# ---------------------------------------------------------------------------
# 1. BaseRuns (David Smyth)
# ---------------------------------------------------------------------------

def compute_baseruns(games: pd.DataFrame) -> pd.DataFrame:
    """Compute BaseRuns for offense and defense (pitcher-allowed BsR).

    Uses expanding mean of per-game BsR components for temporal safety.
    """
    for side in ("home", "away"):
        games[f"{side}_bsr_game"] = _baseruns_single_game(games, side)

    # Pitcher-allowed BsR: compute from the opposing team's batting against this side's pitching
    # home defense = away offense that game, away defense = home offense that game
    games["home_bsr_defense_game"] = games["away_bsr_game"]
    games["away_bsr_defense_game"] = games["home_bsr_game"]

    # Expanding mean with shift(1) — only prior games inform current game
    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        if team_col not in games.columns:
            continue

        for stat, src in [("bsr_offense", f"{side}_bsr_game"),
                          ("bsr_defense", f"{side}_bsr_defense_game")]:
            games[f"{side}_{stat}"] = (
                games.groupby(team_col)[src]
                .transform(lambda s: s.expanding().mean().shift(1))
            )

    # Differentials
    if "home_bsr_offense" in games.columns and "away_bsr_offense" in games.columns:
        games["bsr_offense_diff"] = games["home_bsr_offense"] - games["away_bsr_offense"]
        games["bsr_defense_diff"] = games["home_bsr_defense"] - games["away_bsr_defense"]

    return games


def _baseruns_single_game(games: pd.DataFrame, side: str) -> pd.Series:
    """Apply BsR formula to a single game's batting stats.

    BsR = (A * B) / (B + C) + D
    See: Predictive MLB Team Ratings document for full derivation.
    """
    H = games.get(f"{side}_H", pd.Series(0, index=games.index))
    BB = games.get(f"{side}_BB", pd.Series(0, index=games.index))
    HBP = games.get(f"{side}_HBP", pd.Series(0, index=games.index))
    HR = games.get(f"{side}_HR", pd.Series(0, index=games.index))
    IBB = games.get(f"{side}_IBB", pd.Series(0, index=games.index))
    TB = games.get(f"{side}_TB", pd.Series(0, index=games.index))
    SB = games.get(f"{side}_SB", pd.Series(0, index=games.index))
    CS = games.get(f"{side}_CS", pd.Series(0, index=games.index))
    GDP = games.get(f"{side}_GDP", pd.Series(0, index=games.index))
    PA = games.get(f"{side}_PA", pd.Series(0, index=games.index))
    SH = games.get(f"{side}_SH", pd.Series(0, index=games.index))
    SF = games.get(f"{side}_SF", pd.Series(0, index=games.index))

    # Ensure numeric
    for arr in [H, BB, HBP, HR, IBB, TB, SB, CS, GDP, PA, SH, SF]:
        arr = pd.to_numeric(arr, errors="coerce").fillna(0)

    A = H + BB + HBP - HR - 0.5 * IBB
    B = 1.1 * (1.4 * TB - 0.6 * H - 3 * HR + 0.1 * (BB + HBP - IBB) + 0.9 * (SB - CS - GDP))
    C = PA - A - HR + CS + GDP - SH - SF
    D = HR

    denom = B + C
    bsr = np.where(denom != 0, (A * B) / denom + D, D)
    return pd.Series(bsr, index=games.index, dtype="float32")


# ---------------------------------------------------------------------------
# 2. Pythagenpat (Smyth/Patriot)
# ---------------------------------------------------------------------------

def compute_pythagenpat(games: pd.DataFrame, z: float = 0.287) -> pd.DataFrame:
    """Compute Pythagenpat expected win% at 3 tiers.

    Tier 1: actual runs scored/allowed
    Tier 2: BsR-estimated runs (removes sequencing luck)
    Tier 3: SOS-adjusted tier 2 (not implemented until SRS is computed)
    """
    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        if team_col not in games.columns:
            continue

        # Expanding cumulative RS/RA with shift(1)
        rs_col = f"{side}_bat_game_runs" if f"{side}_bat_game_runs" in games.columns else None
        ra_col = f"{'away' if side == 'home' else 'home'}_bat_game_runs"
        ra_col = ra_col if ra_col in games.columns else None

        if rs_col and ra_col:
            rs_cum = games.groupby(team_col)[rs_col].transform(
                lambda s: s.expanding().sum().shift(1)
            )
            ra_cum = games.groupby(team_col)[ra_col].transform(
                lambda s: s.expanding().sum().shift(1)
            )
            gp = games.groupby(team_col).cumcount()  # games played (0-indexed)

            games[f"{side}_pythag_1st"] = _pythagenpat_formula(rs_cum, ra_cum, gp, z)

        # Tier 2: use expanding BsR as estimated runs
        bsr_off = f"{side}_bsr_offense"
        bsr_def = f"{side}_bsr_defense"
        if bsr_off in games.columns and bsr_def in games.columns:
            # BsR values are already expanding means; multiply by games played for cumulative
            gp = games.groupby(team_col).cumcount().clip(lower=1)
            rs_bsr = games[bsr_off] * gp
            ra_bsr = games[bsr_def] * gp
            games[f"{side}_pythag_2nd"] = _pythagenpat_formula(rs_bsr, ra_bsr, gp, z)

    # Tier 3 (SOS-adjusted) is computed after SRS is available
    # Differentials
    for tier in ("1st", "2nd"):
        h = f"home_pythag_{tier}"
        a = f"away_pythag_{tier}"
        if h in games.columns and a in games.columns:
            games[f"pythag_{tier}_diff"] = games[h] - games[a]

    return games


def _pythagenpat_formula(
    rs: pd.Series, ra: pd.Series, games_played: pd.Series, z: float
) -> pd.Series:
    """W% = RS^x / (RS^x + RA^x) where x = RPG^z."""
    rpg = (rs + ra) / games_played.clip(lower=1)
    exponent = rpg ** z

    # Avoid division by zero
    ratio = np.where(ra > 0, rs / ra, 1.0)
    ratio_exp = np.power(np.abs(ratio).clip(min=1e-10), exponent)
    win_pct = ratio_exp / (1 + ratio_exp)
    return pd.Series(win_pct, index=rs.index, dtype="float32").clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# 3. Simple Rating System (SRS)
# ---------------------------------------------------------------------------

def compute_srs(
    games: pd.DataFrame,
    window: int = 162,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> pd.DataFrame:
    """Compute SRS ratings per season-to-date using iterative least-squares.

    For temporal safety, each game's SRS features use only results from
    prior games within the same season.
    """
    if "season" not in games.columns:
        return games

    srs_home = pd.Series(np.nan, index=games.index, dtype="float32")
    srs_away = pd.Series(np.nan, index=games.index, dtype="float32")

    for season, season_games in games.groupby("season"):
        idx = season_games.index
        n_games = len(season_games)

        if n_games < 10:
            continue

        teams = pd.concat([
            season_games["home_team_id"], season_games["away_team_id"]
        ]).dropna().unique()
        team_to_idx = {t: i for i, t in enumerate(sorted(teams))}
        n_teams = len(teams)

        # Compute SRS iteratively for each game (expanding window)
        ratings_cache = {}
        for pos in range(n_games):
            game_idx = idx[pos]
            # Use games 0..pos-1 (prior games only)
            prior = season_games.iloc[max(0, pos - window):pos]

            if len(prior) < 5:
                continue

            ratings = _solve_srs(prior, team_to_idx, n_teams, tol, max_iter)
            if ratings is not None:
                ratings_cache[game_idx] = ratings
                h_team = season_games.at[game_idx, "home_team_id"]
                a_team = season_games.at[game_idx, "away_team_id"]

                if h_team in team_to_idx:
                    srs_home.at[game_idx] = ratings[team_to_idx[h_team]]
                if a_team in team_to_idx:
                    srs_away.at[game_idx] = ratings[team_to_idx[a_team]]

    games["home_srs"] = srs_home
    games["away_srs"] = srs_away
    games["srs_diff"] = srs_home - srs_away
    return games


def _solve_srs(
    prior_games: pd.DataFrame,
    team_to_idx: dict,
    n_teams: int,
    tol: float,
    max_iter: int,
) -> Optional[np.ndarray]:
    """Solve SRS via iterative method: r = y + S*r, mean-centered."""
    n = len(prior_games)
    if n < 5:
        return None

    # Build schedule matrix and margin vector
    games_per_team = np.zeros(n_teams)
    margin_sum = np.zeros(n_teams)
    schedule_matrix = np.zeros((n_teams, n_teams))

    for _, row in prior_games.iterrows():
        h = row.get("home_team_id")
        a = row.get("away_team_id")
        if h not in team_to_idx or a not in team_to_idx:
            continue

        h_idx = team_to_idx[h]
        a_idx = team_to_idx[a]

        # Use run differential from targets or raw runs
        if "home_run_diff" in row.index and pd.notna(row["home_run_diff"]):
            margin = float(row["home_run_diff"])
        elif "home_runs" in row.index and "away_runs" in row.index:
            margin = float(row.get("home_runs", 0)) - float(row.get("away_runs", 0))
        else:
            continue

        games_per_team[h_idx] += 1
        games_per_team[a_idx] += 1
        margin_sum[h_idx] += margin
        margin_sum[a_idx] -= margin
        schedule_matrix[h_idx, a_idx] += 1
        schedule_matrix[a_idx, h_idx] += 1

    # Average margin per game
    valid = games_per_team > 0
    if valid.sum() < 3:
        return None

    avg_margin = np.zeros(n_teams)
    avg_margin[valid] = margin_sum[valid] / games_per_team[valid]

    # Normalize schedule matrix to proportions
    S = np.zeros((n_teams, n_teams))
    for i in range(n_teams):
        if games_per_team[i] > 0:
            S[i] = schedule_matrix[i] / games_per_team[i]

    # Iterative solution: r_{k+1} = avg_margin + S @ r_k, then center
    r = avg_margin.copy()
    for _ in range(max_iter):
        r_new = avg_margin + S @ r
        r_new -= r_new.mean()
        if np.max(np.abs(r_new - r)) < tol:
            return r_new.astype("float32")
        r = r_new

    return r.astype("float32")


# ---------------------------------------------------------------------------
# 4. Pitcher-Adjusted Elo
# ---------------------------------------------------------------------------

def compute_elo(games: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Compute Elo ratings with pitcher adjustments, updated after each game.

    All parameters are Optuna-tuned. Literature defaults (FiveThirtyEight)
    serve as initial values for the search.
    """
    K = params.get("elo_k", 4.0)
    H = params.get("elo_home_advantage", 24.0)
    K_mov = params.get("elo_mov_coeff", 2.2)
    K_elo_scale = params.get("elo_mov_elo_scale", 0.001)
    K_cap = params.get("elo_mov_cap", 1.25)
    pitcher_coeff = params.get("elo_pitcher_coeff", 4.7)
    reversion = params.get("elo_reversion_weight", 0.6)

    # Initialize all teams at 1500
    team_elos: dict[int, float] = {}
    # Track rolling game score per pitcher for pitcher adjustment
    team_avg_pitcher: dict[int, list] = {}
    last_season = None

    elo_home_arr = np.full(len(games), np.nan, dtype="float32")
    elo_away_arr = np.full(len(games), np.nan, dtype="float32")
    elo_prob_arr = np.full(len(games), np.nan, dtype="float32")

    for i, row in games.iterrows():
        h_team = row.get("home_team_id")
        a_team = row.get("away_team_id")
        season = row.get("season")

        if pd.isna(h_team) or pd.isna(a_team):
            continue

        h_team = int(h_team)
        a_team = int(a_team)

        # Between-season reversion
        if last_season is not None and season != last_season:
            for team in list(team_elos.keys()):
                team_elos[team] = reversion * team_elos[team] + (1 - reversion) * 1500.0
            team_avg_pitcher.clear()
        last_season = season

        # Initialize unseen teams
        if h_team not in team_elos:
            team_elos[h_team] = 1500.0
        if a_team not in team_elos:
            team_elos[a_team] = 1500.0

        h_elo = team_elos[h_team]
        a_elo = team_elos[a_team]

        # Pitcher adjustment (uses FIP or game score from prior starts)
        h_adj = _pitcher_adjustment(row, "home", team_avg_pitcher, h_team, pitcher_coeff)
        a_adj = _pitcher_adjustment(row, "away", team_avg_pitcher, a_team, pitcher_coeff)

        h_elo_adj = h_elo + h_adj
        a_elo_adj = a_elo + a_adj

        # Pre-game probability
        prob_home = 1.0 / (1.0 + 10.0 ** ((a_elo_adj - (h_elo_adj + H)) / 400.0))

        elo_home_arr[i] = h_elo
        elo_away_arr[i] = a_elo
        elo_prob_arr[i] = prob_home

        # Post-game update
        run_diff = row.get("home_run_diff")
        if pd.isna(run_diff):
            continue

        result = 1.0 if run_diff > 0 else (0.0 if run_diff < 0 else 0.5)
        abs_diff = abs(float(run_diff))
        elo_diff = abs(h_elo - a_elo)

        # Margin of victory multiplier (log-scaled, capped)
        mov_mult = min(
            np.log(abs_diff + 1) * K_mov / (K_elo_scale * elo_diff + K_mov),
            K_cap,
        )

        update = K * mov_mult * (result - prob_home)
        team_elos[h_team] = h_elo + update
        team_elos[a_team] = a_elo - update

    games["home_elo"] = elo_home_arr
    games["away_elo"] = elo_away_arr
    games["elo_prob"] = elo_prob_arr
    games["elo_diff"] = elo_home_arr - elo_away_arr
    return games


def _pitcher_adjustment(
    row: pd.Series,
    side: str,
    team_avg_pitcher: dict,
    team_id: int,
    coeff: float,
) -> float:
    """Compute pitcher Elo adjustment from rolling game score differential."""
    # Use season ERA as proxy for pitcher quality relative to team
    era_col = f"sp_{side}_season_era"
    if era_col in row.index and pd.notna(row[era_col]):
        era = float(row[era_col])
        # Track team's average pitcher ERA
        if team_id not in team_avg_pitcher:
            team_avg_pitcher[team_id] = []
        team_avg_pitcher[team_id].append(era)
        team_avg = np.mean(team_avg_pitcher[team_id][-20:])  # rolling 20 starts
        # Lower ERA = better pitcher → positive adjustment
        # Scale: 1 ERA unit ≈ coeff Elo points advantage
        return coeff * (team_avg - era)

    return 0.0


# ---------------------------------------------------------------------------
# 5. Wolfe/Bradley-Terry MLE
# ---------------------------------------------------------------------------

def compute_wolfe(
    games: pd.DataFrame,
    decay_lambda: float = 0.005,
    halving_threshold: float = 14.0,
) -> pd.DataFrame:
    """Compute Wolfe-style Bradley-Terry MLE ratings with recency weighting.

    Refits periodically (every 50 games) within each season for tractability.
    """
    if "season" not in games.columns:
        return games

    wolfe_home = pd.Series(np.nan, index=games.index, dtype="float32")
    wolfe_away = pd.Series(np.nan, index=games.index, dtype="float32")
    wolfe_prob = pd.Series(np.nan, index=games.index, dtype="float32")

    for season, season_games in games.groupby("season"):
        idx = season_games.index
        n_games = len(season_games)

        teams = pd.concat([
            season_games["home_team_id"], season_games["away_team_id"]
        ]).dropna().unique()
        team_to_idx = {int(t): i for i, t in enumerate(sorted(teams))}
        n_teams = len(teams)

        if n_teams < 5:
            continue

        # Refit every 50 games for computational tractability
        last_fit_ratings = np.zeros(n_teams)
        refit_interval = 50

        for pos in range(n_games):
            game_idx = idx[pos]

            # Refit BT model using all prior games this season
            if pos > 0 and pos % refit_interval == 0:
                prior = season_games.iloc[:pos]
                ratings = _fit_bradley_terry(
                    prior, team_to_idx, n_teams, decay_lambda, halving_threshold
                )
                if ratings is not None:
                    last_fit_ratings = ratings

            h_team = season_games.at[game_idx, "home_team_id"]
            a_team = season_games.at[game_idx, "away_team_id"]

            if pd.isna(h_team) or pd.isna(a_team):
                continue

            h_team = int(h_team)
            a_team = int(a_team)

            if h_team in team_to_idx and a_team in team_to_idx:
                h_r = last_fit_ratings[team_to_idx[h_team]]
                a_r = last_fit_ratings[team_to_idx[a_team]]
                wolfe_home.at[game_idx] = h_r
                wolfe_away.at[game_idx] = a_r
                # BT probability
                prob = 1.0 / (1.0 + np.exp(-(h_r - a_r)))
                wolfe_prob.at[game_idx] = prob

    games["home_wolfe"] = wolfe_home
    games["away_wolfe"] = wolfe_away
    games["wolfe_prob"] = wolfe_prob
    games["wolfe_diff"] = wolfe_home - wolfe_away
    return games


def _fit_bradley_terry(
    prior_games: pd.DataFrame,
    team_to_idx: dict,
    n_teams: int,
    decay_lambda: float,
    halving_threshold: float,
) -> Optional[np.ndarray]:
    """Fit Bradley-Terry model via MLE with recency weighting."""
    if len(prior_games) < 10:
        return None

    # Build weighted wins matrix
    W = np.zeros((n_teams, n_teams))
    max_idx = len(prior_games) - 1

    for pos, (_, row) in enumerate(prior_games.iterrows()):
        h = row.get("home_team_id")
        a = row.get("away_team_id")
        rd = row.get("home_run_diff")

        if pd.isna(h) or pd.isna(a) or pd.isna(rd):
            continue

        h = int(h)
        a = int(a)
        if h not in team_to_idx or a not in team_to_idx:
            continue

        h_idx = team_to_idx[h]
        a_idx = team_to_idx[a]

        # Recency weight: exponential decay by games elapsed
        games_ago = max_idx - pos
        weight = np.exp(-decay_lambda * games_ago)

        # Score-based weight with halving function for blowouts
        abs_diff = abs(float(rd))
        if abs_diff > halving_threshold:
            score_weight = halving_threshold + (abs_diff - halving_threshold) * 0.5
        else:
            score_weight = abs_diff
        # Normalize score weight to [0.5, 2.0] range
        score_mult = 0.5 + min(score_weight / halving_threshold, 1.5)

        combined_weight = weight * score_mult

        if rd > 0:
            W[h_idx, a_idx] += combined_weight
        elif rd < 0:
            W[a_idx, h_idx] += combined_weight
        else:
            W[h_idx, a_idx] += combined_weight * 0.5
            W[a_idx, h_idx] += combined_weight * 0.5

    # MLE: minimize negative log-likelihood
    def neg_ll(theta):
        full_theta = np.zeros(n_teams)
        full_theta[1:] = theta
        nll = 0.0
        for i in range(n_teams):
            for j in range(n_teams):
                if W[i, j] > 0:
                    diff = full_theta[i] - full_theta[j]
                    # Numerically stable log-sigmoid
                    if diff > 0:
                        nll -= W[i, j] * (diff - np.log(1 + np.exp(diff)))
                    else:
                        nll -= W[i, j] * (-np.log(1 + np.exp(-diff)))
        return nll

    try:
        result = minimize(neg_ll, np.zeros(n_teams - 1), method="L-BFGS-B",
                          options={"maxiter": 200, "ftol": 1e-8})
        ratings = np.zeros(n_teams)
        ratings[1:] = result.x
        ratings -= ratings.mean()
        return ratings.astype("float32")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 6. Log5
# ---------------------------------------------------------------------------

def compute_log5(
    games: pd.DataFrame,
    window_short: int = 20,
    window_medium: int = 40,
) -> pd.DataFrame:
    """Compute Log5 matchup probabilities from rolling win percentages.

    P(A>B) = p_A(1-p_B) / (p_A(1-p_B) + p_B(1-p_A))
    """
    if "home_team_id" not in games.columns:
        return games

    # Compute rolling win% for each team at each game (shift(1))
    # We need to track win/loss per team chronologically
    all_teams = pd.concat([games["home_team_id"], games["away_team_id"]]).dropna().unique()

    # Build a team-game timeline
    team_winpct: dict[int, list] = {int(t): [] for t in all_teams}

    # Pre-compute results per team
    for _, row in games.iterrows():
        h = row.get("home_team_id")
        a = row.get("away_team_id")
        rd = row.get("home_run_diff")

        if pd.isna(h) or pd.isna(a) or pd.isna(rd):
            continue

        h, a = int(h), int(a)
        h_win = 1.0 if rd > 0 else (0.5 if rd == 0 else 0.0)
        team_winpct[h].append(h_win)
        team_winpct[a].append(1.0 - h_win)

    # Now compute rolling win% per game with temporal safety
    # Track position per team
    team_pos: dict[int, int] = {int(t): 0 for t in all_teams}

    log5_short = np.full(len(games), np.nan, dtype="float32")
    log5_medium = np.full(len(games), np.nan, dtype="float32")
    log5_season = np.full(len(games), np.nan, dtype="float32")

    # Need season tracking for season-to-date
    team_season_results: dict[tuple, list] = {}

    for i, row in games.iterrows():
        h = row.get("home_team_id")
        a = row.get("away_team_id")
        season = row.get("season")

        if pd.isna(h) or pd.isna(a):
            continue

        h, a = int(h), int(a)

        # Get prior win% for home team
        h_results = team_winpct.get(h, [])
        a_results = team_winpct.get(a, [])
        h_pos = team_pos.get(h, 0)
        a_pos = team_pos.get(a, 0)

        # Short window
        h_short = _rolling_mean(h_results, h_pos, window_short)
        a_short = _rolling_mean(a_results, a_pos, window_short)
        if h_short is not None and a_short is not None:
            log5_short[i] = _log5_formula(h_short, a_short)

        # Medium window
        h_med = _rolling_mean(h_results, h_pos, window_medium)
        a_med = _rolling_mean(a_results, a_pos, window_medium)
        if h_med is not None and a_med is not None:
            log5_medium[i] = _log5_formula(h_med, a_med)

        # Season-to-date
        h_season_key = (h, season)
        a_season_key = (a, season)
        h_std = team_season_results.get(h_season_key, [])
        a_std = team_season_results.get(a_season_key, [])
        if len(h_std) >= 5 and len(a_std) >= 5:
            h_s = np.mean(h_std)
            a_s = np.mean(a_std)
            log5_season[i] = _log5_formula(h_s, a_s)

        # Update positions AFTER computing features (temporal safety)
        rd = row.get("home_run_diff")
        if not pd.isna(rd):
            h_win = 1.0 if rd > 0 else (0.5 if rd == 0 else 0.0)
            team_pos[h] = h_pos + 1
            team_pos[a] = a_pos + 1
            team_season_results.setdefault(h_season_key, []).append(h_win)
            team_season_results.setdefault(a_season_key, []).append(1.0 - h_win)

    games["log5_prob_short"] = log5_short
    games["log5_prob_medium"] = log5_medium
    games["log5_prob_season"] = log5_season
    return games


def _rolling_mean(results: list, current_pos: int, window: int) -> Optional[float]:
    """Compute rolling mean of last `window` results before current_pos."""
    if current_pos < 5:
        return None
    start = max(0, current_pos - window)
    subset = results[start:current_pos]
    if len(subset) < 5:
        return None
    return np.mean(subset)


def _log5_formula(p_a: float, p_b: float) -> float:
    """Log5: P(A beats B) given win probabilities p_a, p_b."""
    p_a = np.clip(p_a, 0.01, 0.99)
    p_b = np.clip(p_b, 0.01, 0.99)
    num = p_a * (1 - p_b)
    denom = p_a * (1 - p_b) + p_b * (1 - p_a)
    if denom == 0:
        return 0.5
    return num / denom
