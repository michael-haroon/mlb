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

    Builds a unified per-team timeline so expanding BsR sees ALL games
    regardless of home/away side.
    """
    if "home_team_id" not in games.columns or "away_team_id" not in games.columns:
        return games

    for side in ("home", "away"):
        games[f"{side}_bsr_game"] = _baseruns_single_game(games, side)

    # Build unified timeline: one row per (team, game) with offense/defense BsR.
    parts = []
    for side in ("home", "away"):
        opp_side = "away" if side == "home" else "home"
        sub = pd.DataFrame({
            "team_id": games[f"{side}_team_id"],
            "frame_idx": games.index,
            "bsr_offense": games[f"{side}_bsr_game"].values,
            "bsr_defense": games[f"{opp_side}_bsr_game"].values,
            "side": side,
        })
        parts.append(sub)

    timeline = pd.concat(parts, ignore_index=True)
    timeline = timeline.sort_values("frame_idx").reset_index(drop=True)

    # Expanding mean across ALL games per team
    for stat in ("bsr_offense", "bsr_defense"):
        timeline[f"_{stat}"] = (
            timeline.groupby("team_id")[stat]
            .transform(lambda s: s.expanding().mean().shift(1))
        )

    # Map back to game frame
    for side in ("home", "away"):
        side_rows = timeline[timeline["side"] == side].set_index("frame_idx")
        games[f"{side}_bsr_offense"] = side_rows["_bsr_offense"].reindex(games.index)
        games[f"{side}_bsr_defense"] = side_rows["_bsr_defense"].reindex(games.index)

    # Differentials and sums
    if "home_bsr_offense" in games.columns and "away_bsr_offense" in games.columns:
        games["bsr_offense_diff"] = games["home_bsr_offense"] - games["away_bsr_offense"]
        games["bsr_offense_sum"] = games["home_bsr_offense"] + games["away_bsr_offense"]
        games["bsr_defense_diff"] = games["home_bsr_defense"] - games["away_bsr_defense"]
        games["bsr_defense_sum"] = games["home_bsr_defense"] + games["away_bsr_defense"]

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

    Tier 1: actual runs scored/allowed (unified timeline)
    Tier 2: BsR-estimated runs (removes sequencing luck)
    Tier 3: SOS-adjusted tier 2 (not implemented until SRS is computed)
    """
    if "home_team_id" not in games.columns or "away_team_id" not in games.columns:
        return games

    # --- Tier 1: Unified timeline for actual runs ---
    rs_home = f"home_bat_game_runs"
    rs_away = f"away_bat_game_runs"
    if rs_home in games.columns and rs_away in games.columns:
        parts = []
        for side in ("home", "away"):
            opp_side = "away" if side == "home" else "home"
            sub = pd.DataFrame({
                "team_id": games[f"{side}_team_id"],
                "frame_idx": games.index,
                "rs": games[f"{side}_bat_game_runs"].values,
                "ra": games[f"{opp_side}_bat_game_runs"].values,
                "side": side,
            })
            parts.append(sub)

        timeline = pd.concat(parts, ignore_index=True)
        timeline = timeline.sort_values("frame_idx").reset_index(drop=True)

        timeline["_rs_cum"] = timeline.groupby("team_id")["rs"].transform(
            lambda s: s.expanding().sum().shift(1))
        timeline["_ra_cum"] = timeline.groupby("team_id")["ra"].transform(
            lambda s: s.expanding().sum().shift(1))
        timeline["_gp"] = timeline.groupby("team_id").cumcount()

        timeline["_pythag_1st"] = _pythagenpat_formula(
            timeline["_rs_cum"], timeline["_ra_cum"], timeline["_gp"], z)

        for side in ("home", "away"):
            side_rows = timeline[timeline["side"] == side].set_index("frame_idx")
            games[f"{side}_pythag_1st"] = side_rows["_pythag_1st"].reindex(games.index)

    # --- Tier 2: use expanding BsR (already unified from compute_baseruns fix) ---
    for side in ("home", "away"):
        bsr_off = f"{side}_bsr_offense"
        bsr_def = f"{side}_bsr_defense"
        if bsr_off in games.columns and bsr_def in games.columns:
            # BsR expanding means × games played gives cumulative estimated runs.
            # Games played comes from the unified timeline.
            gp_col = f"{side}_pythag_1st"  # proxy: if tier 1 exists, timeline was built
            if "_gp" not in dir():
                # Rebuild unified GP count for tier 2
                parts_gp = []
                for s in ("home", "away"):
                    sub = pd.DataFrame({
                        "team_id": games[f"{s}_team_id"],
                        "frame_idx": games.index,
                        "side": s,
                    })
                    parts_gp.append(sub)
                tl_gp = pd.concat(parts_gp, ignore_index=True).sort_values("frame_idx")
                tl_gp["_gp"] = tl_gp.groupby("team_id").cumcount().clip(lower=1)
                for s in ("home", "away"):
                    sr = tl_gp[tl_gp["side"] == s].set_index("frame_idx")
                    games[f"_{s}_gp"] = sr["_gp"].reindex(games.index)
            gp = games.get(f"_{side}_gp", games.groupby(f"{side}_team_id").cumcount().clip(lower=1))
            rs_bsr = games[bsr_off] * gp
            ra_bsr = games[bsr_def] * gp
            games[f"{side}_pythag_2nd"] = _pythagenpat_formula(rs_bsr, ra_bsr, gp, z)

    # Clean up temp columns
    for side in ("home", "away"):
        col = f"_{side}_gp"
        if col in games.columns:
            games = games.drop(columns=[col])

    # Differentials and sums
    for tier in ("1st", "2nd"):
        h = f"home_pythag_{tier}"
        a = f"away_pythag_{tier}"
        if h in games.columns and a in games.columns:
            games[f"pythag_{tier}_diff"] = games[h] - games[a]
            games[f"pythag_{tier}_sum"] = games[h] + games[a]

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

    Incremental sliding-window: instead of re-building the schedule matrix
    from scratch for each game (O(n²) per trial), we maintain running
    margin_sum/games_per_team/schedule_matrix arrays and add the entering
    game + subtract the exiting game as the window advances.  This reduces
    the per-game cost from O(window) to O(1), making the total season cost
    O(n) matrix additions rather than O(n·window).
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

        # Pre-extract columns as numpy arrays to avoid per-row pandas overhead
        h_teams = season_games["home_team_id"].values
        a_teams = season_games["away_team_id"].values
        if "home_run_diff" in season_games.columns:
            margins_raw = season_games["home_run_diff"].values
        elif "home_runs" in season_games.columns and "away_runs" in season_games.columns:
            margins_raw = (season_games["home_runs"].values
                           - season_games["away_runs"].values)
        else:
            continue

        # Convert teams to integer indices (-1 = unknown)
        h_idx_arr = np.array([team_to_idx.get(t, -1) for t in h_teams])
        a_idx_arr = np.array([team_to_idx.get(t, -1) for t in a_teams])
        margins_arr = np.where(
            pd.isna(margins_raw), np.nan, margins_raw.astype(float)
        )

        # Sliding-window state (starts empty, adding one game at a time)
        games_per_team = np.zeros(n_teams)
        margin_sum = np.zeros(n_teams)
        schedule_matrix = np.zeros((n_teams, n_teams))

        def _add_game(pos, sign=1.0):
            h, a, m = h_idx_arr[pos], a_idx_arr[pos], margins_arr[pos]
            if h == -1 or a == -1 or np.isnan(m):
                return
            games_per_team[h] += sign
            games_per_team[a] += sign
            margin_sum[h] += sign * m
            margin_sum[a] -= sign * m
            schedule_matrix[h, a] += sign
            schedule_matrix[a, h] += sign

        for pos in range(n_games):
            game_idx = idx[pos]

            # Add game entering the window tail (game at pos-1 just became prior)
            if pos > 0:
                _add_game(pos - 1)
            # Evict game that fell outside the window
            if pos > window:
                _add_game(pos - window - 1, sign=-1.0)

            if games_per_team.sum() < 5 or (games_per_team > 0).sum() < 3:
                continue

            ratings = _solve_srs_from_state(
                games_per_team, margin_sum, schedule_matrix, n_teams, tol, max_iter
            )
            if ratings is None:
                continue

            h_team = h_teams[pos]
            a_team = a_teams[pos]
            if h_team in team_to_idx:
                srs_home.at[game_idx] = ratings[team_to_idx[h_team]]
            if a_team in team_to_idx:
                srs_away.at[game_idx] = ratings[team_to_idx[a_team]]

    games["home_srs"] = srs_home
    games["away_srs"] = srs_away
    games["srs_diff"] = srs_home - srs_away
    games["srs_sum"] = srs_home + srs_away
    return games


def _solve_srs_from_state(
    games_per_team: np.ndarray,
    margin_sum: np.ndarray,
    schedule_matrix: np.ndarray,
    n_teams: int,
    tol: float,
    max_iter: int,
) -> Optional[np.ndarray]:
    """Solve SRS given pre-accumulated state arrays (no iterrows rebuild)."""
    valid = games_per_team > 0
    if valid.sum() < 3:
        return None

    avg_margin = np.zeros(n_teams)
    avg_margin[valid] = margin_sum[valid] / games_per_team[valid]

    # Normalised schedule proportions — vectorised row division
    S = np.zeros((n_teams, n_teams))
    row_counts = games_per_team[:, None]
    nonzero = games_per_team > 0
    S[nonzero] = schedule_matrix[nonzero] / row_counts[nonzero]

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
    games["elo_sum"] = elo_home_arr + elo_away_arr
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
        if team_id not in team_avg_pitcher:
            team_avg_pitcher[team_id] = []
        # Compute team baseline BEFORE appending current pitcher to avoid self-inclusion
        team_avg = np.mean(team_avg_pitcher[team_id][-20:]) if team_avg_pitcher[team_id] else era
        team_avg_pitcher[team_id].append(era)
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
                prob = 1.0 / (1.0 + np.exp(-np.clip(h_r - a_r, -500, 500)))
                wolfe_prob.at[game_idx] = prob

    games["home_wolfe"] = wolfe_home
    games["away_wolfe"] = wolfe_away
    games["wolfe_prob"] = wolfe_prob
    games["wolfe_diff"] = wolfe_home - wolfe_away
    games["wolfe_sum"] = wolfe_home + wolfe_away
    return games


def _fit_bradley_terry(
    prior_games: pd.DataFrame,
    team_to_idx: dict,
    n_teams: int,
    decay_lambda: float,
    halving_threshold: float,
) -> Optional[np.ndarray]:
    """Fit Bradley-Terry model via MM algorithm with recency weighting.

    The Minorization-Maximization (Hunter 2004) update is:
        p_i^{(t+1)} = W_i / Σ_j (W_ij + W_ji) / (p_i^(t) + p_j^(t))
    where p_i are strength parameters (not log-strengths) and W_ij is the
    weighted win count of i over j.  Each iteration is a single vectorised
    division — no scipy optimizer, no finite-difference gradient evaluation.
    Convergence: typically 30-80 iterations vs L-BFGS-B's 200 function evals
    each of which previously ran an O(n_teams²) Python double-loop.

    Reference: Hunter (2004), "MM algorithms for generalized Bradley-Terry
    models", Annals of Statistics 32(1), pp. 384-406.
    """
    if len(prior_games) < 10:
        return None

    # Build weighted wins matrix using vectorised numpy operations
    h_ids = prior_games["home_team_id"].values
    a_ids = prior_games["away_team_id"].values
    rds = prior_games["home_run_diff"].values if "home_run_diff" in prior_games.columns else None

    if rds is None:
        return None

    n = len(prior_games)
    max_idx = n - 1

    # Recency weights: exp(-λ * games_ago), computed once as a vector
    games_ago = np.arange(n - 1, -1, -1, dtype=float)  # [n-1, n-2, ..., 0]
    decay_weights = np.exp(-decay_lambda * games_ago)

    W = np.zeros((n_teams, n_teams))

    for pos in range(n):
        h = h_ids[pos]
        a = a_ids[pos]
        rd = rds[pos]

        if pd.isna(h) or pd.isna(a) or pd.isna(rd):
            continue
        h, a = int(h), int(a)
        if h not in team_to_idx or a not in team_to_idx:
            continue

        h_idx = team_to_idx[h]
        a_idx = team_to_idx[a]
        rd_f = float(rd)

        abs_diff = abs(rd_f)
        if abs_diff > halving_threshold:
            score_weight = halving_threshold + (abs_diff - halving_threshold) * 0.5
        else:
            score_weight = abs_diff
        score_mult = 0.5 + min(score_weight / halving_threshold, 1.5)

        w = decay_weights[pos] * score_mult

        if rd_f > 0:
            W[h_idx, a_idx] += w
        elif rd_f < 0:
            W[a_idx, h_idx] += w
        else:
            W[h_idx, a_idx] += w * 0.5
            W[a_idx, h_idx] += w * 0.5

    # MM iterations — each step is fully vectorised
    # p_i^{new} = W_i / sum_j (W_ij+W_ji)/(p_i+p_j)
    # Work in strength space (p > 0), convert to log-ratings at the end.
    p = np.ones(n_teams)
    W_total = W + W.T  # symmetric total-games matrix

    wins = W.sum(axis=1)  # W_i = total weighted wins for team i (constant across iters)
    for _ in range(100):
        # Denominator: sum_j W_total[i,j] / (p[i] + p[j])  for each i.
        # Guard against 0/0 on diagonal entries where W_total[i,i]=0 but
        # p[i]+p[i] could be 0 for a winless team; add eps to avoid the warning.
        pair_sum = p[:, None] + p[None, :] + 1e-300
        denom = (W_total / pair_sum).sum(axis=1)

        new_p = p.copy()
        mask = denom > 0
        new_p[mask] = wins[mask] / denom[mask]
        # Normalise so mean(p) = 1 (equivalent to mean-centring log-p)
        new_p /= new_p.mean()

        if np.max(np.abs(new_p - p)) < 1e-8:
            p = new_p
            break
        p = new_p

    # Convert to log-ratings (mean-centred) for consistency with Elo/SRS scale
    ratings = np.log(np.clip(p, 1e-10, None))
    ratings -= ratings.mean()
    return ratings.astype("float32")


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
