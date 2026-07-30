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

    Accumulates ALL prior starts for a pitcher regardless of home/away side,
    then maps the cumulative ERA/WHIP back to each game row using
    probable_pitcher_{side}_id (pre-game announcement) to avoid lookahead from
    the post-game sp_{side}_id. Falls back to sp_{side}_id when probable
    pitcher columns are absent.
    """
    # Collect (pitcher_id, frame_idx, er, ip, h, bb) from both sides into one timeline.
    parts = []
    for side in ("home", "away"):
        pid_col = f"sp_{side}_id"
        er_col  = f"sp_{side}_game_earned_runs"
        ip_col  = f"sp_{side}_game_innings_pitched"
        h_col   = f"sp_{side}_game_hits"
        bb_col  = f"sp_{side}_game_bb"

        if pid_col not in games.columns:
            continue

        mask = games[pid_col].notna()
        sub = pd.DataFrame({
            "pitcher_id": games.loc[mask, pid_col],
            "frame_idx":  games.index[mask],
            "er":  pd.to_numeric(games.loc[mask, er_col]  if er_col in games.columns else np.nan, errors="coerce"),
            "ip":  pd.to_numeric(games.loc[mask, ip_col]  if ip_col in games.columns else np.nan, errors="coerce"),
            "h":   pd.to_numeric(games.loc[mask, h_col]   if h_col  in games.columns else np.nan, errors="coerce"),
            "bb":  pd.to_numeric(games.loc[mask, bb_col]  if bb_col in games.columns else np.nan, errors="coerce"),
            "side": side,
        })
        parts.append(sub)

    if not parts:
        return games

    timeline = pd.concat(parts, ignore_index=True)
    timeline = timeline.sort_values("frame_idx").reset_index(drop=True)

    # Expanding cumulative sums (NO shift) — includes current start's stats.
    # Exclusion of the current game is handled at lookup time via searchsorted.
    for col in ("er", "ip", "h", "bb"):
        timeline[f"cum_{col}"] = (
            timeline.groupby("pitcher_id")[col]
            .transform(lambda s: s.expanding().sum())
        )

    safe_ip = timeline["cum_ip"].replace(0, np.nan)
    timeline["_era"]  = (timeline["cum_er"] / safe_ip * 9.0).astype("float32")
    timeline["_whip"] = ((timeline["cum_h"] + timeline["cum_bb"]) / safe_ip).astype("float32")

    # Build lookup: for each pitcher, sorted array of (frame_idx, era, whip).
    pitcher_history: dict = {}
    for pid, grp in timeline.groupby("pitcher_id"):
        arr = grp[["frame_idx", "_era", "_whip"]].values
        pitcher_history[pid] = arr

    # Map back using probable_pitcher (pre-game) when available, else sp_{side}_id.
    # searchsorted(side="left") - 1 finds the latest start STRICTLY BEFORE this frame,
    # which correctly excludes the current game if the pitcher starts it.
    new_cols: dict[str, pd.Series] = {}
    for side in ("home", "away"):
        prob_col = f"probable_pitcher_{side}_id"
        sp_col = f"sp_{side}_id"
        lookup_col = prob_col if prob_col in games.columns else sp_col
        if lookup_col not in games.columns:
            continue

        lookup_ids = games[lookup_col].values
        frame_idxs = games.index.values
        era_vals = np.full(len(games), np.nan, dtype="float32")
        whip_vals = np.full(len(games), np.nan, dtype="float32")

        for i in range(len(games)):
            pid = lookup_ids[i]
            if pd.isna(pid) or pid not in pitcher_history:
                continue
            hist = pitcher_history[pid]
            # Latest start strictly before this frame position
            pos = np.searchsorted(hist[:, 0], frame_idxs[i], side="left") - 1
            if pos >= 0:
                era_vals[i] = hist[pos, 1]
                whip_vals[i] = hist[pos, 2]

        new_cols[f"sp_{side}_season_era"] = pd.Series(era_vals, index=games.index)
        new_cols[f"sp_{side}_season_whip"] = pd.Series(whip_vals, index=games.index)

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

    log.info("Engineering EWMA features...")
    games = _ewma_features(games)

    log.info("Engineering unified (overall) rolling stats...")
    games = _unified_rolling_stats(games)

    log.info("Engineering win/loss momentum...")
    games = _momentum_features(games)

    log.info("Engineering rest and schedule density...")
    games = _rest_and_schedule(games)

    log.info("Engineering park factors...")
    games = _park_factors(games)

    log.info("Engineering weather features...")
    games = _weather_features(games)

    log.info("Engineering air density index...")
    games = _air_density_features(games)

    log.info("Engineering starting pitcher features...")
    games = _starting_pitcher_features(games)

    log.info("Engineering schedule context flags...")
    games = _schedule_context(games)

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
# EWMA features
# ---------------------------------------------------------------------------

def _ewma_features(games: pd.DataFrame) -> pd.DataFrame:
    """Exponentially weighted moving averages with game-index halflife.

    Unlike rolling windows which weight all observations equally, EWMA gives
    exponentially more weight to recent games. Uses game-index distance (not
    calendar days) consistent with the codebase's decay philosophy — avoids
    penalizing offseason gaps.

    halflife=15 means a game 15 starts ago gets half the weight of the most
    recent game.
    """
    halflife = 15  # games, not days — TODO: validate — placeholder
    new_cols: dict[str, pd.Series] = {}

    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        if team_col not in games.columns:
            continue

        # Per-game rates to EWMA — source columns created by _rolling_batting/pitching_stats
        stats_to_ewma = {
            f"{side}_game_avg": f"{side}_ewma_avg",
            f"{side}_game_obp": f"{side}_ewma_obp",
            f"{side}_game_slg": f"{side}_ewma_slg",
            f"{side}_game_era": f"{side}_ewma_era",
            f"{side}_game_whip": f"{side}_ewma_whip",
            f"{side}_game_k9": f"{side}_ewma_k9",
            f"{side}_game_fip": f"{side}_ewma_fip",
        }

        for src_col, dst_col in stats_to_ewma.items():
            source = games.get(src_col)
            if source is None:
                # Check new_cols in case it was added by a prior step in this function
                source = new_cols.get(src_col)
            if source is None:
                continue
            new_cols[dst_col] = (
                source.groupby(games[team_col])
                .transform(lambda s: s.ewm(halflife=halflife, min_periods=5).mean().shift(1))
                .astype("float32")
            )

        # OPS composite from EWMA components
        obp_key = f"{side}_ewma_obp"
        slg_key = f"{side}_ewma_slg"
        if obp_key in new_cols and slg_key in new_cols:
            new_cols[f"{side}_ewma_ops"] = (new_cols[obp_key] + new_cols[slg_key]).astype("float32")

    # Home - Away differentials and sums for all EWMA stats
    for stat in ("avg", "obp", "slg", "ops", "era", "whip", "k9", "fip"):
        h_col = f"home_ewma_{stat}"
        a_col = f"away_ewma_{stat}"
        if h_col in new_cols and a_col in new_cols:
            new_cols[f"diff_ewma_{stat}"] = (new_cols[h_col] - new_cols[a_col]).astype("float32")
            new_cols[f"sum_ewma_{stat}"] = (new_cols[h_col] + new_cols[a_col]).astype("float32")

    if not new_cols:
        return games
    return pd.concat([games, pd.DataFrame(new_cols, index=games.index)], axis=1)


# ---------------------------------------------------------------------------
# Unified (overall) rolling stats
# ---------------------------------------------------------------------------

def _unified_rolling_stats(games: pd.DataFrame) -> pd.DataFrame:
    """Rolling and EWMA stats across ALL team games (home + away combined).

    Complements per-side features by giving the model a view of overall team
    form with double the sample size and no staleness gaps from road trips.
    Uses the same melt-and-map pattern as momentum/rest.
    """
    if "home_team_id" not in games.columns or "away_team_id" not in games.columns:
        return games

    # Melt per-game rates from both sides into a unified timeline.
    # Source columns are the per-game rates already computed by _rolling_batting/pitching_stats.
    bat_stats = ["avg", "obp", "slg", "iso", "hr_rate", "k_rate", "bb_rate"]
    pit_stats = ["era", "whip", "k9", "fip"]
    all_stats = bat_stats + pit_stats

    parts = []
    for side in ("home", "away"):
        sub = pd.DataFrame({
            "team_id": games[f"{side}_team_id"],
            "frame_idx": games.index,
            "side": side,
        })
        for stat in all_stats:
            src_col = f"{side}_game_{stat}"
            if src_col in games.columns:
                sub[stat] = games[src_col].values
        parts.append(sub)

    timeline = pd.concat(parts, ignore_index=True)
    timeline = timeline.sort_values("frame_idx").reset_index(drop=True)

    # Rolling windows on unified timeline
    halflife = 15  # games (true games now, not same-side-games)
    new_cols: dict[str, pd.Series] = {}

    for stat in all_stats:
        if stat not in timeline.columns:
            continue
        for w in (10, 20):
            timeline[f"_roll{w}_{stat}"] = (
                timeline.groupby("team_id")[stat]
                .transform(lambda s, _w=w: s.rolling(_w, min_periods=max(3, _w // 2)).mean().shift(1))
            )
        timeline[f"_ewma_{stat}"] = (
            timeline.groupby("team_id")[stat]
            .transform(lambda s: s.ewm(halflife=halflife, min_periods=5).mean().shift(1))
            .astype("float32")
        )

    # OPS composites
    for w in (10, 20):
        obp = f"_roll{w}_obp"
        slg = f"_roll{w}_slg"
        if obp in timeline.columns and slg in timeline.columns:
            timeline[f"_roll{w}_ops"] = timeline[obp] + timeline[slg]
    if "_ewma_obp" in timeline.columns and "_ewma_slg" in timeline.columns:
        timeline["_ewma_ops"] = timeline["_ewma_obp"] + timeline["_ewma_slg"]

    # Map back to game frame by (frame_idx, side)
    unified_stats = [c for c in timeline.columns if c.startswith("_roll") or c.startswith("_ewma")]
    for side in ("home", "away"):
        side_rows = timeline[timeline["side"] == side].set_index("frame_idx")
        for col in unified_stats:
            feat_name = f"{side}_all{col}"  # e.g. home_all_roll10_avg, home_all_ewma_era
            if col in side_rows.columns:
                new_cols[feat_name] = side_rows[col].reindex(games.index)

    # Differentials and sums on unified stats
    for col in unified_stats:
        h_col = f"home_all{col}"
        a_col = f"away_all{col}"
        if h_col in new_cols and a_col in new_cols:
            new_cols[f"diff_all{col}"] = new_cols[h_col] - new_cols[a_col]
            new_cols[f"sum_all{col}"] = new_cols[h_col] + new_cols[a_col]

    if not new_cols:
        return games
    return pd.concat([games, pd.DataFrame(new_cols, index=games.index)], axis=1)


# ---------------------------------------------------------------------------
# Momentum features
# ---------------------------------------------------------------------------

def _momentum_features(games: pd.DataFrame) -> pd.DataFrame:
    """Win streaks, run differential trends, rolling std.

    Builds a unified per-team timeline (combining home and away rows) so that
    momentum features reflect the team's overall recent state, not just their
    performance on one side.
    """
    if "home_team_id" not in games.columns or "away_team_id" not in games.columns:
        return games

    # Build unified timeline: one row per (team, game), with win/loss and run diff.
    parts = []
    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        opp_side = "away" if side == "home" else "home"
        win_col = "home_win" if side == "home" else "away_win"
        runs_col = f"{side}_bat_game_runs"
        opp_runs_col = f"{opp_side}_bat_game_runs"

        sub = pd.DataFrame({"team_id": games[team_col], "frame_idx": games.index, "side": side})
        if win_col in games.columns:
            sub["win"] = games[win_col].values
        if runs_col in games.columns and opp_runs_col in games.columns:
            sub["runs_for"] = games[runs_col].values
            sub["runs_against"] = games[opp_runs_col].values
            sub["rd"] = sub["runs_for"] - sub["runs_against"]
        parts.append(sub)

    timeline = pd.concat(parts, ignore_index=True)
    timeline = timeline.sort_values("frame_idx").reset_index(drop=True)

    new_cols: dict[str, pd.Series] = {}

    # Rolling win% and win streak on unified timeline
    if "win" in timeline.columns:
        for w in (10, 20):
            timeline[f"_roll{w}_winpct"] = (
                timeline.groupby("team_id")["win"]
                .transform(lambda s: s.rolling(w, min_periods=5).mean().shift(1))
            )
        timeline["_win_streak"] = (
            timeline.groupby("team_id")["win"]
            .transform(lambda s: _streak_length(s).shift(1))
        )

    # Rolling run differential
    if "rd" in timeline.columns:
        for w in (5, 10, 20):
            timeline[f"_roll{w}_rd_mean"] = (
                timeline.groupby("team_id")["rd"]
                .transform(lambda s: s.rolling(w, min_periods=3).mean().shift(1))
            )
            timeline[f"_roll{w}_rd_std"] = (
                timeline.groupby("team_id")["rd"]
                .transform(lambda s: s.rolling(w, min_periods=3).std().shift(1))
            )

    # Map back to game frame by (frame_idx, side)
    for side in ("home", "away"):
        side_rows = timeline[timeline["side"] == side].set_index("frame_idx")
        if "win" in timeline.columns:
            for w in (10, 20):
                col = f"_roll{w}_winpct"
                if col in side_rows.columns:
                    new_cols[f"{side}_roll{w}_winpct"] = side_rows[col].reindex(games.index)
            if "_win_streak" in side_rows.columns:
                new_cols[f"{side}_win_streak"] = side_rows["_win_streak"].reindex(games.index)
        if "rd" in timeline.columns:
            for w in (5, 10, 20):
                for suffix in ("rd_mean", "rd_std"):
                    col = f"_roll{w}_{suffix}"
                    if col in side_rows.columns:
                        new_cols[f"{side}_roll{w}_{suffix}"] = side_rows[col].reindex(games.index)

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
    """Days rest, games in last 7 days, doubleheader flag.

    Builds a unified per-team timeline (combining home and away rows) so that
    rest/schedule features see ALL games a team plays, not just same-side games.
    """
    if "game_date" not in games.columns:
        return games
    if "home_team_id" not in games.columns or "away_team_id" not in games.columns:
        return games

    new_cols: dict[str, pd.Series] = {}

    # Build unified timeline: one row per (team, game) with game_date.
    gd = pd.to_datetime(games["game_date"], errors="coerce")
    parts = []
    for side in ("home", "away"):
        sub = pd.DataFrame({
            "team_id": games[f"{side}_team_id"],
            "frame_idx": games.index,
            "game_date": gd.values,
            "side": side,
        })
        parts.append(sub)

    timeline = pd.concat(parts, ignore_index=True)
    timeline = timeline.sort_values("frame_idx").reset_index(drop=True)
    timeline["game_date"] = pd.to_datetime(timeline["game_date"])

    # Days since last game (any side)
    timeline["_days_rest"] = (
        timeline.groupby("team_id")["game_date"]
        .transform(lambda s: s.diff().dt.days)
    )

    # Games in last 7 days (any side)
    def _games_last_7d(s: pd.Series) -> pd.Series:
        # pandas >=3.0.5 requires a strictly monotonic datetime index for
        # time-based rolling. Sort by date first, roll, then unsort back to
        # original row order. Sorting positionally (argsort) avoids reindex
        # failures caused by duplicate dates (doubleheaders).
        sort_order = s.argsort(kind="stable")
        unsort_order = sort_order.argsort(kind="stable")
        s_sorted = s.iloc[sort_order]
        s_dt = s_sorted.copy()
        s_dt.index = s_dt.values  # monotonic datetime index
        counts = s_dt.rolling("7D", closed="left").count()
        return pd.Series(counts.values[unsort_order], index=s.index)

    timeline["_games_7d"] = (
        timeline.groupby("team_id")["game_date"]
        .transform(_games_last_7d)
        .fillna(0)
    )

    # Map back to game frame by (frame_idx, side)
    for side in ("home", "away"):
        side_rows = timeline[timeline["side"] == side].set_index("frame_idx")
        new_cols[f"{side}_days_rest"] = side_rows["_days_rest"].reindex(games.index)
        new_cols[f"{side}_games_last_7d"] = side_rows["_games_7d"].reindex(games.index)

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
    League average is restricted to regular-season games only — spring training
    run-scoring rates (shorter outings, split-squad, different rosters) would
    contaminate early regular-season park factors otherwise.
    """
    if "venue_id" not in games.columns or "total_runs" not in games.columns:
        return games

    is_regular = games.get("game_type_code") == "R"
    total_runs = games["total_runs"].copy()

    # Mask out non-regular-season games from the league average denominator.
    # They still get a park_factor (via venue_avg which is all-time), but they
    # don't contribute to the league scoring baseline.
    runs_for_league = total_runs.where(is_regular, other=np.nan)

    league_avg = (
        runs_for_league.groupby(games["season"])
        .transform(lambda s: s.expanding(min_periods=5).mean().shift(1))
    )
    venue_avg = (
        total_runs.groupby(games["venue_id"])
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
# Air Density Index
# ---------------------------------------------------------------------------

# MLB Stats API venue_id → elevation (feet above sea level).
# Only venues with elevation >400ft included; all others default to 0 (sea level,
# ADI ≈ 1.0). Elevation affects batted-ball carry — lower air density at altitude
# reduces drag, increasing HR distance and total run scoring.
# TODO: validate — placeholder (venue IDs from MLB Stats API; elevations from USGS)
_VENUE_ELEVATIONS_FT: dict[int, int] = {
    19: 5280,    # Coors Field (Denver) — extreme outlier
    15: 1082,    # Chase Field (Phoenix)
    5325: 551,   # Globe Life Field (Arlington)
    7: 750,      # Kauffman Stadium (Kansas City)
    3312: 815,   # Target Field (Minneapolis)
    2889: 466,   # Busch Stadium (St. Louis)
    2602: 480,   # Great American Ball Park (Cincinnati)
    4705: 1050,  # Truist Park (Atlanta)
    2700: 1555,  # Salt River Fields (spring training, Scottsdale)
}


def _air_density_features(games: pd.DataFrame) -> pd.DataFrame:
    """Air Density Index from venue elevation via ISA lapse-rate formula.

    ADI = rho(h) / rho(sea_level) — the fraction of sea-level air density
    at the venue's elevation. Lower ADI → less drag → ball carries farther.
    Coors (ADI≈0.85) is the extreme; most venues are >0.98.
    """
    if "venue_id" not in games.columns:
        return games

    elevation_ft = games["venue_id"].map(_VENUE_ELEVATIONS_FT).fillna(0)
    # ISA (International Standard Atmosphere) lapse-rate formula:
    # rho/rho0 = (1 - L*h/T0)^(g*M/(R*L) - 1)
    # L=0.0065 K/m, T0=288.15K, g=9.80665, M=0.0289644 kg/mol, R=8.31447 J/(mol·K)
    # Exponent = g*M/(R*L) - 1 = 4.2558
    # Coefficient per foot: L * 0.3048 / T0 = 6.8756e-6
    adi = ((1.0 - 6.8756e-6 * elevation_ft) ** 4.2558).astype("float32")

    return pd.concat([games, pd.DataFrame({"air_density_index": adi}, index=games.index)], axis=1)


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
        new_cols["sp_era_diff"] = (a_era - h_era).astype("float32")
        new_cols["sp_era_sum"] = (a_era + h_era).astype("float32")
    if h_whip is not None and a_whip is not None:
        new_cols["sp_whip_diff"] = (a_whip - h_whip).astype("float32")
        new_cols["sp_whip_sum"] = (a_whip + h_whip).astype("float32")

    if not new_cols:
        return games
    return pd.concat([games, pd.DataFrame(new_cols, index=games.index)], axis=1)


# ---------------------------------------------------------------------------
# Schedule context flags + interactions
# ---------------------------------------------------------------------------

def _schedule_context(games: pd.DataFrame) -> pd.DataFrame:
    """Matchup-type flags and pre-computed interaction terms.

    Flags: is_same_division, is_same_league (binary).
    Interactions: elo_prob * flag, consensus_prob * flag (for linear models
    that cannot discover split interactions natively).

    Invariant: is_same_division=1 implies is_same_league=1. Different-league
    teams cannot be in the same division even if division_id values collide.
    """
    new_cols: dict[str, pd.Series] = {}

    has_league = "home_league_id" in games.columns and "away_league_id" in games.columns
    has_division = "home_division_id" in games.columns and "away_division_id" in games.columns

    if has_league:
        same_league = (games["home_league_id"] == games["away_league_id"]).astype("float32")
        # NaN on either side → NaN (pandas == with NaN → False, but we want NaN)
        either_null = games["home_league_id"].isna() | games["away_league_id"].isna()
        same_league = same_league.where(~either_null, other=np.nan).astype("float32")
    else:
        same_league = pd.Series(np.nan, index=games.index, dtype="float32")

    if has_division and has_league:
        same_division = (
            (games["home_division_id"] == games["away_division_id"])
            & (games["home_league_id"] == games["away_league_id"])
        ).astype("float32")
        either_null_div = (
            games["home_division_id"].isna() | games["away_division_id"].isna()
            | games["home_league_id"].isna() | games["away_league_id"].isna()
        )
        same_division = same_division.where(~either_null_div, other=np.nan).astype("float32")
    else:
        same_division = pd.Series(np.nan, index=games.index, dtype="float32")

    new_cols["is_same_league"] = same_league
    new_cols["is_same_division"] = same_division

    # Interaction terms for linear models
    elo_prob = games.get("elo_prob")
    consensus_prob = games.get("consensus_home_win_prob")

    if elo_prob is not None:
        new_cols["elo_prob_x_same_league"] = (elo_prob * same_league).astype("float32")
        new_cols["elo_prob_x_same_division"] = (elo_prob * same_division).astype("float32")

    if consensus_prob is not None:
        new_cols["consensus_prob_x_same_league"] = (consensus_prob * same_league).astype("float32")
        new_cols["consensus_prob_x_same_division"] = (consensus_prob * same_division).astype("float32")

    return pd.concat([games, pd.DataFrame(new_cols, index=games.index)], axis=1)


# ---------------------------------------------------------------------------
# Head-to-head (DEPRECATED — retained for reference, no longer called)
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
