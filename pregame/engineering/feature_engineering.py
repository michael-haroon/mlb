"""Feature engineering beyond rating systems.

Computes rolling statistics, contextual features, differentials, and
matchup-specific features. All temporal features use shift(1) to prevent
look-ahead bias.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


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

    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        if team_col not in games.columns:
            continue

        # Derived rate stats per game
        ab = games.get(f"{side}_bat_game_ab", pd.Series(0, index=games.index))
        h = games.get(f"{side}_H", pd.Series(0, index=games.index))
        bb = games.get(f"{side}_BB", pd.Series(0, index=games.index))
        hbp = games.get(f"{side}_HBP", pd.Series(0, index=games.index))
        hr = games.get(f"{side}_HR", pd.Series(0, index=games.index))
        sf = games.get(f"{side}_SF", pd.Series(0, index=games.index))
        tb = games.get(f"{side}_TB", pd.Series(0, index=games.index))
        so = games.get(f"{side}_bat_game_so", pd.Series(0, index=games.index))
        pa = games.get(f"{side}_PA", pd.Series(0, index=games.index))

        # Per-game rate stats (avoid division by zero)
        safe_ab = ab.replace(0, np.nan)
        safe_pa = pa.replace(0, np.nan)

        games[f"{side}_game_avg"] = h / safe_ab
        games[f"{side}_game_obp"] = (h + bb + hbp) / safe_pa
        games[f"{side}_game_slg"] = tb / safe_ab
        games[f"{side}_game_iso"] = (tb - h) / safe_ab
        games[f"{side}_game_hr_rate"] = hr / safe_pa
        games[f"{side}_game_k_rate"] = so / safe_pa
        games[f"{side}_game_bb_rate"] = bb / safe_pa

        # BABIP = (H - HR) / (AB - SO - HR + SF)
        babip_denom = (ab - so - hr + sf).replace(0, np.nan)
        games[f"{side}_game_babip"] = (h - hr) / babip_denom

        # Rolling averages with shift(1)
        rate_cols = [
            f"{side}_game_avg", f"{side}_game_obp", f"{side}_game_slg",
            f"{side}_game_iso", f"{side}_game_hr_rate", f"{side}_game_k_rate",
            f"{side}_game_bb_rate", f"{side}_game_babip",
        ]

        for w in windows:
            for col in rate_cols:
                if col in games.columns:
                    stat_name = col.replace(f"{side}_game_", "")
                    games[f"{side}_roll{w}_{stat_name}"] = (
                        games.groupby(team_col)[col]
                        .transform(lambda s: s.rolling(w, min_periods=max(3, w // 2)).mean().shift(1))
                    )

        # OPS composite
        for w in windows:
            obp_col = f"{side}_roll{w}_obp"
            slg_col = f"{side}_roll{w}_slg"
            if obp_col in games.columns and slg_col in games.columns:
                games[f"{side}_roll{w}_ops"] = games[obp_col] + games[slg_col]

    return games


# ---------------------------------------------------------------------------
# Rolling pitching stats
# ---------------------------------------------------------------------------

def _rolling_pitching_stats(games: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling pitching stats at 5, 10, 20 game windows."""
    windows = [5, 10, 20]

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

        # ERA per game (scaled to 9 innings)
        games[f"{side}_game_era"] = (er / safe_ip) * 9.0
        # WHIP
        games[f"{side}_game_whip"] = (hits + bb) / safe_ip
        # K/9 and BB/9
        games[f"{side}_game_k9"] = (so / safe_ip) * 9.0
        games[f"{side}_game_bb9"] = (bb / safe_ip) * 9.0
        games[f"{side}_game_hr9"] = (hr / safe_ip) * 9.0

        # FIP = (13*HR + 3*BB - 2*K) / IP + constant
        # FIP constant ≈ 3.10 (league average, varies by year; using 3.10 as baseline)
        fip_const = 3.10
        games[f"{side}_game_fip"] = (13 * hr + 3 * bb - 2 * so) / safe_ip + fip_const

        pit_cols = [
            f"{side}_game_era", f"{side}_game_whip", f"{side}_game_k9",
            f"{side}_game_bb9", f"{side}_game_hr9", f"{side}_game_fip",
        ]

        for w in windows:
            for col in pit_cols:
                if col in games.columns:
                    stat_name = col.replace(f"{side}_game_", "")
                    games[f"{side}_roll{w}_{stat_name}"] = (
                        games.groupby(team_col)[col]
                        .transform(lambda s: s.rolling(w, min_periods=max(3, w // 2)).mean().shift(1))
                    )

    return games


# ---------------------------------------------------------------------------
# Momentum features
# ---------------------------------------------------------------------------

def _momentum_features(games: pd.DataFrame) -> pd.DataFrame:
    """Win streaks, run differential trends, rolling std."""
    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        if team_col not in games.columns:
            continue

        # Binary win indicator
        if side == "home":
            win_col = "home_win"
        else:
            win_col = "away_win"

        if win_col in games.columns:
            # Rolling win% (10 and 20 games)
            for w in (10, 20):
                games[f"{side}_roll{w}_winpct"] = (
                    games.groupby(team_col)[win_col]
                    .transform(lambda s: s.rolling(w, min_periods=5).mean().shift(1))
                )

            # Win streak (consecutive wins heading into game)
            games[f"{side}_win_streak"] = (
                games.groupby(team_col)[win_col]
                .transform(lambda s: _streak_length(s).shift(1))
            )

        # Run differential rolling mean and std
        rd_col = f"{side}_bat_game_runs"
        opp_side = "away" if side == "home" else "home"
        opp_rd_col = f"{opp_side}_bat_game_runs"

        if rd_col in games.columns and opp_rd_col in games.columns:
            game_rd = games[rd_col] - games[opp_rd_col]
            for w in (5, 10, 20):
                games[f"{side}_roll{w}_rd_mean"] = (
                    games.groupby(team_col)[rd_col]
                    .transform(lambda s: s.rolling(w, min_periods=3).mean().shift(1))
                ) - (
                    games.groupby(team_col)[opp_rd_col]
                    .transform(lambda s: s.rolling(w, min_periods=3).mean().shift(1))
                )

                games[f"{side}_roll{w}_rd_std"] = (
                    game_rd.groupby(games[team_col])
                    .transform(lambda s: s.rolling(w, min_periods=3).std().shift(1))
                )

    return games


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

    gd = pd.to_datetime(games["game_date"], errors="coerce")

    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        if team_col not in games.columns:
            continue

        # Days since last game
        games[f"{side}_days_rest"] = (
            gd.groupby(games[team_col])
            .transform(lambda s: s.diff().dt.days)
        )

        # Games in last 7 days
        games[f"{side}_games_last_7d"] = (
            gd.groupby(games[team_col])
            .transform(lambda s: s.rolling("7D", closed="left").count())
        )

    # Doubleheader indicator
    if "double_header" in games.columns:
        games["is_doubleheader"] = (games["double_header"] == "Y").astype("float32")
    elif "game_number" in games.columns:
        games["is_doubleheader"] = (games["game_number"] > 1).astype("float32")

    return games


# ---------------------------------------------------------------------------
# Park factors
# ---------------------------------------------------------------------------

def _park_factors(games: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling park factor from prior season data at each venue."""
    if "venue_id" not in games.columns or "total_runs" not in games.columns:
        return games

    # Simple park factor: avg total runs at this venue / league avg
    # Expanding mean with shift for temporal safety
    league_avg = games.groupby("season")["total_runs"].transform("mean")
    games["_venue_runs"] = games["total_runs"]

    # Per-venue expanding mean (all prior games at this venue)
    venue_avg = (
        games.groupby("venue_id")["_venue_runs"]
        .transform(lambda s: s.expanding(min_periods=10).mean().shift(1))
    )

    # Park factor relative to league average
    games["park_factor"] = venue_avg / league_avg.clip(lower=1.0)
    games.drop(columns=["_venue_runs"], inplace=True)

    return games


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

def _weather_features(games: pd.DataFrame) -> pd.DataFrame:
    """Parse weather into numeric features."""
    if "weather_temp" in games.columns:
        games["temp_f"] = pd.to_numeric(games["weather_temp"], errors="coerce")

    if "venue_roof_type" in games.columns:
        games["is_dome"] = games["venue_roof_type"].isin(["Dome", "Retractable"]).astype("float32")

    if "day_night" in games.columns:
        games["is_night_game"] = (games["day_night"] == "night").astype("float32")

    return games


# ---------------------------------------------------------------------------
# Starting pitcher features
# ---------------------------------------------------------------------------

def _starting_pitcher_features(games: pd.DataFrame) -> pd.DataFrame:
    """Compute starting pitcher quality features relative to team baseline."""
    for side in ("home", "away"):
        era_col = f"sp_{side}_season_era"
        if era_col in games.columns:
            games[era_col] = pd.to_numeric(games[era_col], errors="coerce")

        whip_col = f"sp_{side}_season_whip"
        if whip_col in games.columns:
            games[whip_col] = pd.to_numeric(games[whip_col], errors="coerce")

        # Pitcher handedness as binary
        hand_col = f"sp_{side}_hand"
        if hand_col in games.columns:
            games[f"sp_{side}_is_lefty"] = (games[hand_col] == "L").astype("float32")

    # SP quality differential
    if "sp_home_season_era" in games.columns and "sp_away_season_era" in games.columns:
        # Lower ERA = better; so diff = away_era - home_era (positive = home SP advantage)
        games["sp_era_diff"] = games["sp_away_season_era"] - games["sp_home_season_era"]

    if "sp_home_season_whip" in games.columns and "sp_away_season_whip" in games.columns:
        games["sp_whip_diff"] = games["sp_away_season_whip"] - games["sp_home_season_whip"]

    return games


# ---------------------------------------------------------------------------
# Head-to-head
# ---------------------------------------------------------------------------

def _head_to_head(games: pd.DataFrame) -> pd.DataFrame:
    """H2H record and run differential in last N meetings between these teams."""
    if "home_team_id" not in games.columns or "away_team_id" not in games.columns:
        return games

    # Create matchup key (ordered pair so A vs B == B vs A)
    h = games["home_team_id"].astype(str)
    a = games["away_team_id"].astype(str)
    games["_matchup_key"] = np.where(h < a, h + "_" + a, a + "_" + h)

    # H2H home wins in this matchup (last 10 meetings)
    if "home_win" in games.columns:
        games["h2h_home_winrate_10"] = (
            games.groupby("_matchup_key")["home_win"]
            .transform(lambda s: s.rolling(10, min_periods=3).mean().shift(1))
        )

    if "home_run_diff" in games.columns:
        games["h2h_rd_mean_10"] = (
            games.groupby("_matchup_key")["home_run_diff"]
            .transform(lambda s: s.rolling(10, min_periods=3).mean().shift(1))
        )

    games.drop(columns=["_matchup_key"], inplace=True)
    return games


# ---------------------------------------------------------------------------
# Differentials and sums
# ---------------------------------------------------------------------------

def _differentials_and_sums(games: pd.DataFrame) -> pd.DataFrame:
    """Compute home - away differentials and home + away sums for all numeric rolling features."""
    home_prefix = "home_roll"
    away_prefix = "away_roll"

    home_cols = [c for c in games.columns if c.startswith(home_prefix)]

    for h_col in home_cols:
        suffix = h_col[len("home_"):]
        a_col = f"away_{suffix}"
        if a_col in games.columns:
            games[f"diff_{suffix}"] = games[h_col] - games[a_col]
            games[f"sum_{suffix}"] = games[h_col] + games[a_col]

    return games


# ---------------------------------------------------------------------------
# Consensus probability
# ---------------------------------------------------------------------------

def _consensus_probability(games: pd.DataFrame) -> pd.DataFrame:
    """Compute mean of all rating-system implied probabilities."""
    prob_cols = [c for c in games.columns if "_prob" in c and games[c].dtype in ("float32", "float64")]
    if prob_cols:
        games["consensus_home_win_prob"] = games[prob_cols].mean(axis=1)
        games["consensus_home_win_std"] = games[prob_cols].std(axis=1)
    return games
