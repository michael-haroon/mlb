"""Feature engineering beyond rating systems.

Computes rolling statistics, contextual features, differentials, and
matchup-specific features. All temporal features use shift(1) to prevent
look-ahead bias.

Performance note: all functions accumulate new columns in a local dict and
call pd.concat once at the end, avoiding the O(n²) fragmented-DataFrame
copies that pandas emits PerformanceWarning about when you do hundreds of
individual `df[col] = ...` assignments on the same frame.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _compute_pregame_pitcher_era(games: pd.DataFrame) -> pd.DataFrame:
    """Compute sp_*_season_era / season_whip from prior starts before ratings run.

    build.py calls this between game frame construction and attach_all_ratings
    so that compute_elo's pitcher adjustment has valid pre-game ERA/WHIP.
    _starting_pitcher_features (called later in engineer_features) skips
    recomputing if these columns already exist.
    """
    new_cols: dict[str, pd.Series] = {}
    for side in ("home", "away"):
        pid_col = f"sp_{side}_id"
        er_col  = f"sp_{side}_game_earned_runs"
        ip_col  = f"sp_{side}_game_innings_pitched"
        h_col   = f"sp_{side}_game_hits"
        bb_col  = f"sp_{side}_game_bb"

        if pid_col not in games.columns:
            continue

        pid = games[pid_col]
        er  = pd.to_numeric(games.get(er_col,  pd.Series(np.nan, index=games.index)), errors="coerce")
        ip  = pd.to_numeric(games.get(ip_col,  pd.Series(np.nan, index=games.index)), errors="coerce")
        h   = pd.to_numeric(games.get(h_col,   pd.Series(np.nan, index=games.index)), errors="coerce")
        bb  = pd.to_numeric(games.get(bb_col,  pd.Series(np.nan, index=games.index)), errors="coerce")

        cum_er = er.groupby(pid).transform(lambda s: s.expanding().sum().shift(1))
        cum_ip = ip.groupby(pid).transform(lambda s: s.expanding().sum().shift(1))
        cum_h  =  h.groupby(pid).transform(lambda s: s.expanding().sum().shift(1))
        cum_bb = bb.groupby(pid).transform(lambda s: s.expanding().sum().shift(1))

        safe_ip = cum_ip.replace(0, np.nan)
        new_cols[f"sp_{side}_season_era"]  = (cum_er / safe_ip * 9.0).astype("float32")
        new_cols[f"sp_{side}_season_whip"] = ((cum_h + cum_bb) / safe_ip).astype("float32")

    if not new_cols:
        return games
    return pd.concat([games, pd.DataFrame(new_cols, index=games.index)], axis=1)


def engineer_features(games: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps to the game frame.

    Parameters
    ----------
    games : pd.DataFrame
        Game frame with raw stats, targets, and rating features attached.
        Must be sorted by game_date.

    Returns
    -------
    pd.DataFrame
        Game frame with all engineered features added.
    """
    log.info("Engineering rolling batting stats...")
    games = _rolling_batting_stats(games)

    log.info("Engineering rolling pitching stats...")
    games = _rolling_pitching_stats(games)

    log.info("Engineering win/loss momentum...")
    games = _momentum_features(games)

    log.info("Engineering rest and schedule density...")
    games = _rest_and_schedule(games)

    log.info("Engineering park factors...")
    games = _park_factors(games)

    log.info("Engineering weather features...")
    games = _weather_features(games)

    log.info("Engineering starting pitcher features...")
    games = _starting_pitcher_features(games)

    log.info("Engineering head-to-head features...")
    games = _head_to_head(games)

    log.info("Engineering differentials and sums...")
    games = _differentials_and_sums(games)

    log.info("Engineering consensus probability...")
    games = _consensus_probability(games)

    log.info(f"Feature engineering complete: {len(games.columns)} total columns")
    return games


# ---------------------------------------------------------------------------
# Rolling batting stats
# ---------------------------------------------------------------------------

def _rolling_batting_stats(games: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling batting averages at 5, 10, 20 game windows."""
    windows = [5, 10, 20]
    new_cols: dict[str, pd.Series] = {}

    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        if team_col not in games.columns:
            continue

        ab = games.get(f"{side}_bat_game_ab", pd.Series(0, index=games.index))
        h = games.get(f"{side}_H", pd.Series(0, index=games.index))
        bb = games.get(f"{side}_BB", pd.Series(0, index=games.index))
        hbp = games.get(f"{side}_HBP", pd.Series(0, index=games.index))
        hr = games.get(f"{side}_HR", pd.Series(0, index=games.index))
        sf = games.get(f"{side}_SF", pd.Series(0, index=games.index))
        tb = games.get(f"{side}_TB", pd.Series(0, index=games.index))
        so = games.get(f"{side}_bat_game_so", pd.Series(0, index=games.index))
        pa = games.get(f"{side}_PA", pd.Series(0, index=games.index))

        safe_ab = ab.replace(0, np.nan)
        safe_pa = pa.replace(0, np.nan)
        babip_denom = (ab - so - hr + sf).replace(0, np.nan)

        per_game = {
            f"{side}_game_avg":     h / safe_ab,
            f"{side}_game_obp":     (h + bb + hbp) / safe_pa,
            f"{side}_game_slg":     tb / safe_ab,
            f"{side}_game_iso":     (tb - h) / safe_ab,
            f"{side}_game_hr_rate": hr / safe_pa,
            f"{side}_game_k_rate":  so / safe_pa,
            f"{side}_game_bb_rate": bb / safe_pa,
            f"{side}_game_babip":   (h - hr) / babip_denom,
        }
        new_cols.update(per_game)

        for w in windows:
            for col, series in per_game.items():
                stat_name = col.replace(f"{side}_game_", "")
                roll_name = f"{side}_roll{w}_{stat_name}"
                new_cols[roll_name] = (
                    series.groupby(games[team_col])
                    .transform(lambda s: s.rolling(w, min_periods=max(3, w // 2)).mean().shift(1))
                )

        # OPS composite — built after rolling obp/slg exist in new_cols
        for w in windows:
            obp_col = f"{side}_roll{w}_obp"
            slg_col = f"{side}_roll{w}_slg"
            if obp_col in new_cols and slg_col in new_cols:
                new_cols[f"{side}_roll{w}_ops"] = new_cols[obp_col] + new_cols[slg_col]

    return pd.concat([games, pd.DataFrame(new_cols, index=games.index)], axis=1)


# ---------------------------------------------------------------------------
# Rolling pitching stats
# ---------------------------------------------------------------------------

def _rolling_pitching_stats(games: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling pitching stats at 5, 10, 20 game windows."""
    windows = [5, 10, 20]
    new_cols: dict[str, pd.Series] = {}

    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        if team_col not in games.columns:
            continue

        ip = games.get(f"{side}_pit_game_innings_pitched", pd.Series(0, index=games.index))
        hits = games.get(f"{side}_pit_game_hits", pd.Series(0, index=games.index))
        er = games.get(f"{side}_pit_game_earned_runs", pd.Series(0, index=games.index))
        bb = games.get(f"{side}_pit_game_bb", pd.Series(0, index=games.index))
        so = games.get(f"{side}_pit_game_so", pd.Series(0, index=games.index))
        hr = games.get(f"{side}_pit_game_hr", pd.Series(0, index=games.index))

        safe_ip = ip.replace(0, np.nan)
        fip_const = 3.10

        per_game = {
            f"{side}_game_era":  (er / safe_ip) * 9.0,
            f"{side}_game_whip": (hits + bb) / safe_ip,
            f"{side}_game_k9":   (so / safe_ip) * 9.0,
            f"{side}_game_bb9":  (bb / safe_ip) * 9.0,
            f"{side}_game_hr9":  (hr / safe_ip) * 9.0,
            f"{side}_game_fip":  (13 * hr + 3 * bb - 2 * so) / safe_ip + fip_const,
        }
        new_cols.update(per_game)

        for w in windows:
            for col, series in per_game.items():
                stat_name = col.replace(f"{side}_game_", "")
                new_cols[f"{side}_roll{w}_{stat_name}"] = (
                    series.groupby(games[team_col])
                    .transform(lambda s: s.rolling(w, min_periods=max(3, w // 2)).mean().shift(1))
                )

    return pd.concat([games, pd.DataFrame(new_cols, index=games.index)], axis=1)


# ---------------------------------------------------------------------------
# Momentum features
# ---------------------------------------------------------------------------

def _momentum_features(games: pd.DataFrame) -> pd.DataFrame:
    """Win streaks, run differential trends, rolling std."""
    new_cols: dict[str, pd.Series] = {}

    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        if team_col not in games.columns:
            continue

        win_col = "home_win" if side == "home" else "away_win"

        if win_col in games.columns:
            for w in (10, 20):
                new_cols[f"{side}_roll{w}_winpct"] = (
                    games.groupby(team_col)[win_col]
                    .transform(lambda s: s.rolling(w, min_periods=5).mean().shift(1))
                )
            new_cols[f"{side}_win_streak"] = (
                games.groupby(team_col)[win_col]
                .transform(lambda s: _streak_length(s).shift(1))
            )

        rd_col = f"{side}_bat_game_runs"
        opp_side = "away" if side == "home" else "home"
        opp_rd_col = f"{opp_side}_bat_game_runs"

        if rd_col in games.columns and opp_rd_col in games.columns:
            game_rd = games[rd_col] - games[opp_rd_col]
            for w in (5, 10, 20):
                new_cols[f"{side}_roll{w}_rd_mean"] = (
                    games.groupby(team_col)[rd_col]
                    .transform(lambda s: s.rolling(w, min_periods=3).mean().shift(1))
                ) - (
                    games.groupby(team_col)[opp_rd_col]
                    .transform(lambda s: s.rolling(w, min_periods=3).mean().shift(1))
                )
                new_cols[f"{side}_roll{w}_rd_std"] = (
                    game_rd.groupby(games[team_col])
                    .transform(lambda s: s.rolling(w, min_periods=3).std().shift(1))
                )

    return pd.concat([games, pd.DataFrame(new_cols, index=games.index)], axis=1)


def _streak_length(series: pd.Series) -> pd.Series:
    """Compute current streak length (positive for wins, negative for losses)."""
    streaks = pd.Series(0, index=series.index, dtype="int32")
    current = 0
    for i, val in series.items():
        if pd.isna(val):
            streaks.at[i] = current
            continue
        if val == 1:
            current = max(0, current) + 1
        elif val == 0:
            current = min(0, current) - 1
        else:
            current = 0
        streaks.at[i] = current
    return streaks


# ---------------------------------------------------------------------------
# Rest and schedule density
# ---------------------------------------------------------------------------

def _rest_and_schedule(games: pd.DataFrame) -> pd.DataFrame:
    """Days rest, games in last 7 days, doubleheader flag."""
    if "game_date" not in games.columns:
        return games

    new_cols: dict[str, pd.Series] = {}
    gd = pd.to_datetime(games["game_date"], errors="coerce")

    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        if team_col not in games.columns:
            continue

        # Days since last game
        new_cols[f"{side}_days_rest"] = (
            gd.groupby(games[team_col])
            .transform(lambda s: s.diff().dt.days)
        )

        # Games in last 7 days: rolling("7D") requires a DatetimeIndex, so we
        # temporarily set the index, apply the window, then reset.
        # fillna(0) because an empty 7-day lookback window means zero games
        # played — not unknown data. Occurs on long homestands, All-Star break,
        # or when a team hasn't appeared on a given side for >7 days.
        def _games_last_7d(s: pd.Series) -> pd.Series:
            s_dt = s.copy()
            s_dt.index = s_dt.values  # DatetimeIndex = the dates themselves
            counts = s_dt.rolling("7D", closed="left").count()
            counts.index = s.index  # restore original integer index
            return counts

        new_cols[f"{side}_games_last_7d"] = (
            gd.groupby(games[team_col]).transform(_games_last_7d).fillna(0)
        )

    if "double_header" in games.columns:
        new_cols["is_doubleheader"] = (games["double_header"] == "Y").astype("float32")
    elif "game_number" in games.columns:
        new_cols["is_doubleheader"] = (games["game_number"] > 1).astype("float32")

    return pd.concat([games, pd.DataFrame(new_cols, index=games.index)], axis=1)


# ---------------------------------------------------------------------------
# Park factors
# ---------------------------------------------------------------------------

def _park_factors(games: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling park factor from prior games only.

    Both numerator and denominator use expanding means with shift(1).
    The previous groupby("season").transform("mean") for league_avg leaked
    the full season's run scoring into every row, proven via S3: game 1 of
    2023 had league_avg = 9.55 (full-season mean) on the day it was played.
    """
    if "venue_id" not in games.columns or "total_runs" not in games.columns:
        return games

    # Season-to-date league average excluding the current game.
    league_avg = (
        games.groupby("season")["total_runs"]
        .transform(lambda s: s.expanding(min_periods=5).mean().shift(1))
    )
    venue_avg = (
        games.groupby("venue_id")["total_runs"]
        .transform(lambda s: s.expanding(min_periods=10).mean().shift(1))
    )
    park_factor = venue_avg / league_avg.clip(lower=1.0)
    return pd.concat([games, pd.DataFrame({"park_factor": park_factor}, index=games.index)], axis=1)


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

def _weather_features(games: pd.DataFrame) -> pd.DataFrame:
    """Parse weather into numeric features."""
    new_cols: dict[str, pd.Series] = {}
    if "weather_temp" in games.columns:
        new_cols["temp_f"] = pd.to_numeric(games["weather_temp"], errors="coerce")
    if "venue_roof_type" in games.columns:
        new_cols["is_dome"] = games["venue_roof_type"].isin(["Dome", "Retractable"]).astype("float32")
    if "day_night" in games.columns:
        new_cols["is_night_game"] = (games["day_night"] == "night").astype("float32")
    if not new_cols:
        return games
    return pd.concat([games, pd.DataFrame(new_cols, index=games.index)], axis=1)


# ---------------------------------------------------------------------------
# Starting pitcher features
# ---------------------------------------------------------------------------

def _starting_pitcher_features(games: pd.DataFrame) -> pd.DataFrame:
    """Compute starting pitcher quality features from prior starts only.

    ERA and WHIP use expanding cumulative game logs with shift(1). If
    _compute_pregame_pitcher_era already ran (called by build.py before
    attach_all_ratings so Elo has valid pitcher adjustments), the ERA/WHIP
    columns already exist and are not recomputed — only diffs and handedness
    are added.
    """
    new_cols: dict[str, pd.Series] = {}

    era_already_computed = (
        "sp_home_season_era" in games.columns and
        "sp_away_season_era" in games.columns
    )

    if not era_already_computed:
        # Fallback: compute here if build.py pre-compute step was skipped
        games = _compute_pregame_pitcher_era(games)

    for side in ("home", "away"):
        hand_col = f"sp_{side}_hand"
        if hand_col in games.columns:
            new_cols[f"sp_{side}_is_lefty"] = (games[hand_col] == "L").astype("float32")

    h_era  = games.get("sp_home_season_era")
    a_era  = games.get("sp_away_season_era")
    h_whip = games.get("sp_home_season_whip")
    a_whip = games.get("sp_away_season_whip")

    if h_era is not None and a_era is not None:
        new_cols["sp_era_diff"]  = (a_era  - h_era).astype("float32")
    if h_whip is not None and a_whip is not None:
        new_cols["sp_whip_diff"] = (a_whip - h_whip).astype("float32")

    if not new_cols:
        return games
    return pd.concat([games, pd.DataFrame(new_cols, index=games.index)], axis=1)


# ---------------------------------------------------------------------------
# Head-to-head
# ---------------------------------------------------------------------------

def _head_to_head(games: pd.DataFrame) -> pd.DataFrame:
    """H2H record and run differential in last N meetings between these teams."""
    if "home_team_id" not in games.columns or "away_team_id" not in games.columns:
        return games

    h = games["home_team_id"].astype(str)
    a = games["away_team_id"].astype(str)
    matchup_key = np.where(h < a, h + "_" + a, a + "_" + h)

    new_cols: dict[str, pd.Series] = {}
    if "home_win" in games.columns:
        new_cols["h2h_home_winrate_10"] = (
            games.groupby(matchup_key)["home_win"]
            .transform(lambda s: s.rolling(10, min_periods=3).mean().shift(1))
        )
    if "home_run_diff" in games.columns:
        new_cols["h2h_rd_mean_10"] = (
            games.groupby(matchup_key)["home_run_diff"]
            .transform(lambda s: s.rolling(10, min_periods=3).mean().shift(1))
        )

    if not new_cols:
        return games
    return pd.concat([games, pd.DataFrame(new_cols, index=games.index)], axis=1)


# ---------------------------------------------------------------------------
# Differentials and sums
# ---------------------------------------------------------------------------

def _differentials_and_sums(games: pd.DataFrame) -> pd.DataFrame:
    """Compute home - away differentials and home + away sums for all numeric rolling features."""
    new_cols: dict[str, pd.Series] = {}
    for h_col in (c for c in games.columns if c.startswith("home_roll")):
        suffix = h_col[len("home_"):]
        a_col = f"away_{suffix}"
        if a_col in games.columns:
            new_cols[f"diff_{suffix}"] = games[h_col] - games[a_col]
            new_cols[f"sum_{suffix}"] = games[h_col] + games[a_col]

    if not new_cols:
        return games
    return pd.concat([games, pd.DataFrame(new_cols, index=games.index)], axis=1)


# ---------------------------------------------------------------------------
# Consensus probability
# ---------------------------------------------------------------------------

def _consensus_probability(games: pd.DataFrame) -> pd.DataFrame:
    """Compute mean of all rating-system implied probabilities."""
    prob_cols = [c for c in games.columns if "_prob" in c and games[c].dtype in ("float32", "float64")]
    if not prob_cols:
        return games
    prob_block = games[prob_cols]
    new_cols = {
        "consensus_home_win_prob": prob_block.mean(axis=1),
        "consensus_home_win_std": prob_block.std(axis=1),
    }
    return pd.concat([games, pd.DataFrame(new_cols, index=games.index)], axis=1)
