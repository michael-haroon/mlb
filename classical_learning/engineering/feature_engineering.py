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

    log.info("Engineering umpire tendency features...")
    games = _umpire_features(games)

    log.info("Engineering weather features...")
    games = _weather_features(games)

    log.info("Engineering air density index...")
    games = _air_density_features(games)

    log.info("Engineering starting pitcher features...")
    games = _starting_pitcher_features(games)

    log.info("Engineering schedule context flags...")
    games = _schedule_context(games)

    log.info("Engineering baserunning features...")
    games = _baserunning_features(games)

    log.info("Engineering defense & stranding features...")
    games = _defense_stranding_features(games)

    log.info("Engineering pennant race features...")
    games = _pennant_race_features(games)

    log.info("Engineering postseason flag...")
    games = _postseason_flag(games)

    log.info("Engineering head-to-head history...")
    games = _head_to_head_features(games)

    log.info("Engineering travel and timezone features...")
    games = _travel_features(games)

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
# Umpire tendency
# ---------------------------------------------------------------------------


def _umpire_features(games: pd.DataFrame) -> pd.DataFrame:
    """Umpire tendency features from career expanding means with shift(1).

    HP umpire zone size affects walks, strikeouts, and run environment.
    2B umpire adjudicates steals of 2nd (86% of all attempts, Kruskal-Wallis
    H=258, p<1e-6 between-umpire variance verified on 2015-2026 RUNNERS data).
    Career-spanning: zones are stable biomechanical traits with slow drift.
    """
    new_cols: dict[str, pd.Series] = {}
    is_regular = games.get("game_type_code") == "R"

    # ── HP Umpire ────────────────────────────────────────────────────────
    if "umpire_hp" in games.columns and "total_runs" in games.columns:
        total_runs = games["total_runs"].astype("float64")
        runs_reg = total_runs.where(is_regular, other=np.nan)

        # RPG factor: umpire career average / league season average
        league_avg = (
            runs_reg.groupby(games["season"])
            .transform(lambda s: s.expanding(min_periods=5).mean().shift(1))
        )
        ump_avg = (
            runs_reg.groupby(games["umpire_hp"])
            .transform(lambda s: s.expanding(min_periods=20).mean().shift(1))
        )
        new_cols["ump_hp_rpg_factor"] = ump_avg / league_avg.clip(lower=1.0)

        # BB per game
        if "home_BB" in games.columns and "away_BB" in games.columns:
            total_bb = (games["home_BB"] + games["away_BB"]).astype("float64")
            bb_reg = total_bb.where(is_regular, other=np.nan)
            new_cols["ump_hp_bb_per_game"] = (
                bb_reg.groupby(games["umpire_hp"])
                .transform(lambda s: s.expanding(min_periods=20).mean().shift(1))
            )

        # K per game
        if "home_bat_game_so" in games.columns and "away_bat_game_so" in games.columns:
            total_k = (games["home_bat_game_so"] + games["away_bat_game_so"]).astype("float64")
            k_reg = total_k.where(is_regular, other=np.nan)
            new_cols["ump_hp_k_per_game"] = (
                k_reg.groupby(games["umpire_hp"])
                .transform(lambda s: s.expanding(min_periods=20).mean().shift(1))
            )

        # Called strike % — purest zone-size proxy.
        # Per-game ratio: Called Strike / (Called Strike + Ball) from pitch-level data.
        # Accepts pre-computed `game_called_strike_pct` (aggregated from pitches table)
        # since the boxscore endpoint doesn't break down strikes by call type.
        if "game_called_strike_pct" in games.columns:
            csp = games["game_called_strike_pct"].astype("float64").where(is_regular, other=np.nan)
            new_cols["ump_hp_called_strike_pct"] = (
                csp.groupby(games["umpire_hp"])
                .transform(lambda s: s.expanding(min_periods=20).mean().shift(1))
            )

    # ── 2B Umpire ────────────────────────────────────────────────────────
    if "umpire_2b" in games.columns:
        if "home_SB" in games.columns and "away_SB" in games.columns:
            total_sb = (games["home_SB"] + games["away_SB"]).astype("float64")
            sb_reg = total_sb.where(is_regular, other=np.nan)
            new_cols["ump_2b_sb_per_game"] = (
                sb_reg.groupby(games["umpire_2b"])
                .transform(lambda s: s.expanding(min_periods=20).mean().shift(1))
            )

        if "home_CS" in games.columns and "away_CS" in games.columns:
            total_cs = (games["home_CS"] + games["away_CS"]).astype("float64")
            cs_reg = total_cs.where(is_regular, other=np.nan)
            new_cols["ump_2b_cs_per_game"] = (
                cs_reg.groupby(games["umpire_2b"])
                .transform(lambda s: s.expanding(min_periods=20).mean().shift(1))
            )

    if not new_cols:
        return games
    return pd.concat([games, pd.DataFrame(new_cols, index=games.index)], axis=1)


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

def _weather_features(games: pd.DataFrame) -> pd.DataFrame:
    """Parse weather into numeric features.

    If ERA5-derived weather features already exist (from weather.py),
    only fills is_dome and is_night_game. Otherwise falls back to
    GUMBO's crude weather_temp field.
    """
    new_cols: dict[str, pd.Series] = {}
    # Only use GUMBO temp as fallback when ERA5 features are absent
    if "temperature_f" not in games.columns and "weather_temp" in games.columns:
        new_cols["temp_f"] = pd.to_numeric(games["weather_temp"], errors="coerce")
    if "venue_roof_type" in games.columns and "is_dome" not in games.columns:
        new_cols["is_dome"] = games["venue_roof_type"].isin(["Dome", "Retractable"]).astype("float32")
    if "day_night" in games.columns and "is_night_game" not in games.columns:
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

    Fallback when ERA5-derived air_density_ratio is already present
    (computed by weather.py with actual temp/pressure/humidity).
    The ERA5 version is strictly better — this only runs if weather
    features were not attached (e.g. no weather data available).
    """
    if "air_density_ratio" in games.columns:
        # ERA5-derived version already computed; create legacy alias for compatibility
        if "air_density_index" not in games.columns:
            games["air_density_index"] = games["air_density_ratio"]
        return games

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
# Postseason flag (Group E)
# ---------------------------------------------------------------------------

def _postseason_flag(games: pd.DataFrame) -> pd.DataFrame:
    """Binary indicator for postseason games.

    Uses game_type_code if available: D=Division Series, L=League Championship,
    W=World Series, F=Wild Card. Falls back to month >= 10 if code absent.
    """
    if "is_postseason" in games.columns:
        return games  # already computed

    new_cols: dict[str, pd.Series] = {}

    if "game_type_code" in games.columns:
        new_cols["is_postseason"] = (
            games["game_type_code"].isin({"D", "L", "W", "F"}).astype("float32")
        )
    elif "game_date" in games.columns:
        # Fallback: October/November months are postseason
        dates = pd.to_datetime(games["game_date"])
        new_cols["is_postseason"] = (dates.dt.month >= 10).astype("float32")
    else:
        return games

    return pd.concat([games, pd.DataFrame(new_cols, index=games.index)], axis=1)


# ---------------------------------------------------------------------------
# Head-to-head history (Group C) — revived and modernised from deprecated _head_to_head
# ---------------------------------------------------------------------------

def _head_to_head_features(games: pd.DataFrame) -> pd.DataFrame:
    """H2H record, total runs, and run differential in last N meetings.

    Canonical matchup key is symmetric (min_id + "_" + max_id) so that
    A-at-B and B-at-A share the same history.

    min_periods=5 prevents NaN noise when teams have rarely faced each other
    (e.g. early-season inter-league). Cross-season meetings are included
    (no season reset) because the signal is cumulative matchup history.
    """
    if "home_team_id" not in games.columns or "away_team_id" not in games.columns:
        return games
    if "home_win" not in games.columns:
        return games

    h = games["home_team_id"].astype(str)
    a = games["away_team_id"].astype(str)
    # Canonical key: smaller ID first, so A-vs-B == B-vs-A
    matchup_key = np.where(h < a, h + "_" + a, a + "_" + h)

    new_cols: dict[str, pd.Series] = {}

    for N, mp in ((10, 5), (20, 10)):
        new_cols[f"h2h_home_winpct_last{N}"] = (
            games.groupby(matchup_key)["home_win"]
            .transform(lambda s, _N=N, _mp=mp: s.rolling(_N, min_periods=_mp).mean().shift(1))
            .astype("float32")
        )

    has_scores = "home_runs" in games.columns and "away_runs" in games.columns
    if has_scores:
        total_runs = games["home_runs"].add(games["away_runs"])
        run_diff   = games["home_runs"].sub(games["away_runs"])

        new_cols["h2h_avg_total_runs_last10"] = (
            total_runs.groupby(matchup_key)
            .transform(lambda s: s.rolling(10, min_periods=5).mean().shift(1))
            .astype("float32")
        )
        new_cols["h2h_avg_rd_last10"] = (
            run_diff.groupby(matchup_key)
            .transform(lambda s: s.rolling(10, min_periods=5).mean().shift(1))
            .astype("float32")
        )

    if not new_cols:
        return games
    return pd.concat([games, pd.DataFrame(new_cols, index=games.index)], axis=1)


# ---------------------------------------------------------------------------
# Travel and timezone fatigue (Group D)
# ---------------------------------------------------------------------------

def _travel_features(games: pd.DataFrame) -> pd.DataFrame:
    """Per-team travel distance and timezone shift since last game.

    Haversine distance between consecutive venue lat/lngs for each team
    (home and away games interleaved). Timezone offset estimated from
    longitude (UTC ≈ lon/15) — crude but captures dominant US coast-to-coast
    ~3h shift without requiring external timezone lookup.

    travel_fatigue_flag: 1 if km > 1500 AND it's a day game (is_night_game=0),
    which proxies the "flew overnight for a day game" scenario.
    """
    if not all(c in games.columns for c in ("venue_latitude", "venue_longitude",
                                             "home_team_id", "away_team_id")):
        log.warning("Missing venue lat/lon or team IDs — skipping travel features")
        return games

    def _haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                      lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """Vectorised Haversine distance in km."""
        R = 6371.0
        lat1r, lon1r, lat2r, lon2r = (np.radians(np.asarray(x, dtype=float))
                                       for x in (lat1, lon1, lat2, lon2))
        dlat = lat2r - lat1r
        dlon = lon2r - lon1r
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
        return 2.0 * R * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))

    lat_vals = games["venue_latitude"].values.astype(float)
    lon_vals = games["venue_longitude"].values.astype(float)
    gdate    = games["game_date"].values if "game_date" in games.columns else games.index.values

    # Build one row per (team, game) — each team appears once as home or away
    parts = []
    for side in ("home", "away"):
        parts.append(pd.DataFrame({
            "team_id":   games[f"{side}_team_id"].values,
            "game_date": gdate,
            "frame_idx": games.index.values,
            "lat":       lat_vals,
            "lon":       lon_vals,
            "_side":     side,
        }))

    tl = pd.concat(parts, ignore_index=True)
    tl = tl.sort_values(["team_id", "game_date", "frame_idx"]).reset_index(drop=True)

    # Previous venue for each team (shift within team group)
    tl["_prev_lat"] = tl.groupby("team_id")["lat"].shift(1)
    tl["_prev_lon"] = tl.groupby("team_id")["lon"].shift(1)

    has_prev = tl["_prev_lat"].notna()
    tl["_travel_km"] = np.where(
        has_prev,
        _haversine_km(
            tl["lat"].values,
            tl["lon"].values,
            tl["_prev_lat"].fillna(0.0).values,
            tl["_prev_lon"].fillna(0.0).values,
        ),
        np.nan,
    )

    # UTC offset approximation: lon / 15 (captures 3h west-to-east shift)
    # TODO: validate — placeholder (ignores DST, actual timezone boundaries)
    tl["_utc_off"]      = tl["lon"] / 15.0
    tl["_prev_utc_off"] = tl.groupby("team_id")["_utc_off"].shift(1)
    _raw_delta          = (tl["_utc_off"] - tl["_prev_utc_off"]).abs()
    # Circadian disruption is the minimum of going either direction around the clock.
    # Without wrap-around, US→Tokyo gives 17h instead of the correct 7h.
    tl["_tz_delta"]     = _raw_delta.where(_raw_delta <= 12.0, 24.0 - _raw_delta)

    new_cols: dict[str, pd.Series] = {}

    for side in ("home", "away"):
        side_rows = tl[tl["_side"] == side].set_index("frame_idx")
        new_cols[f"{side}_travel_km_since_last_game"] = (
            side_rows["_travel_km"].reindex(games.index).astype("float32")
        )
        new_cols[f"{side}_timezone_delta"] = (
            side_rows["_tz_delta"].reindex(games.index).astype("float32")
        )

    # Travel fatigue flag: > 1500 km travel AND day game (flew overnight)
    has_night = "is_night_game" in games.columns
    for side in ("home", "away"):
        km_col = f"{side}_travel_km_since_last_game"
        if km_col not in new_cols:
            continue
        big_travel = (new_cols[km_col].fillna(0) > 1500).astype("float32")
        if has_night:
            day_game = (games["is_night_game"] == 0).astype("float32")
            new_cols[f"{side}_travel_fatigue_flag"] = (big_travel * day_game).astype("float32")
        else:
            new_cols[f"{side}_travel_fatigue_flag"] = big_travel.astype("float32")

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


# ---------------------------------------------------------------------------
# Baserunning features (from runners aggregation in game_builder)
# ---------------------------------------------------------------------------

def _baserunning_features(games: pd.DataFrame) -> pd.DataFrame:
    """Rolling baserunning efficiency metrics per team.

    Uses runners-derived game-level columns (sb_attempts, sb_successes,
    extra_base_taken, etc.) from game_builder._aggregate_runners().
    Falls back to box score SB/CS columns if runners data absent.
    """
    if "home_team_id" not in games.columns:
        return games

    # Determine data source: runners-derived or box score fallback
    has_runners = "home_sb_attempts" in games.columns
    new_cols: dict[str, pd.Series] = {}

    parts = []
    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        sub = pd.DataFrame({"team_id": games[team_col], "frame_idx": games.index, "side": side})

        if has_runners:
            sub["sb_attempts"] = games[f"{side}_sb_attempts"].values
            sub["sb_successes"] = games[f"{side}_sb_successes"].values
            sub["extra_base_taken"] = games[f"{side}_extra_base_taken"].values
            sub["extra_base_opps"] = games[f"{side}_extra_base_opps"].values
            sub["first_to_third"] = games[f"{side}_first_to_third"].values
            sub["score_from_second"] = games[f"{side}_score_from_second"].values
        else:
            sb_col = f"{side}_bat_game_sb" if f"{side}_bat_game_sb" in games.columns else None
            cs_col = f"{side}_bat_game_cs" if f"{side}_bat_game_cs" in games.columns else None
            if sb_col and cs_col:
                sub["sb_successes"] = pd.to_numeric(games[sb_col], errors="coerce").values
                sub["sb_attempts"] = (
                    pd.to_numeric(games[sb_col], errors="coerce") +
                    pd.to_numeric(games[cs_col], errors="coerce")
                ).values
            else:
                return games

        if "game_date" in games.columns:
            sub["_season"] = pd.to_datetime(games["game_date"]).dt.year.values
        else:
            sub["_season"] = 0
        parts.append(sub)

    timeline = pd.concat(parts, ignore_index=True)
    timeline = timeline.sort_values("frame_idx").reset_index(drop=True)

    # Compute rates per game
    safe_att = timeline["sb_attempts"].replace(0, np.nan)
    timeline["_sb_rate"] = timeline["sb_successes"] / safe_att

    if has_runners:
        safe_opps = timeline["extra_base_opps"].replace(0, np.nan)
        timeline["_extra_base_rate"] = timeline["extra_base_taken"] / safe_opps

    # Rolling with shift(1)
    rate_cols = ["_sb_rate", "sb_attempts"]
    if has_runners:
        rate_cols += ["_extra_base_rate", "first_to_third", "score_from_second"]

    for w in (10, 20):
        for col in rate_cols:
            if col not in timeline.columns:
                continue
            timeline[f"{col}_roll{w}"] = (
                timeline.groupby(["team_id", "_season"])[col]
                .transform(lambda s, _w=w: s.rolling(_w, min_periods=5).mean().shift(1))
            )

    # Map back to game frame
    for side in ("home", "away"):
        side_rows = timeline[timeline["side"] == side].set_index("frame_idx")
        for w in (10, 20):
            new_cols[f"{side}_roll{w}_sb_success_rate"] = side_rows[f"_sb_rate_roll{w}"].reindex(games.index)
            new_cols[f"{side}_roll{w}_sb_attempts_pg"] = side_rows[f"sb_attempts_roll{w}"].reindex(games.index)
            if has_runners:
                new_cols[f"{side}_roll{w}_extra_base_rate"] = side_rows["_extra_base_rate_roll" + str(w)].reindex(games.index)
                new_cols[f"{side}_roll{w}_first_to_third"] = side_rows[f"first_to_third_roll{w}"].reindex(games.index)
                new_cols[f"{side}_roll{w}_score_from_second"] = side_rows[f"score_from_second_roll{w}"].reindex(games.index)

    # Differentials
    for w in (10, 20):
        h_sb = f"home_roll{w}_sb_success_rate"
        a_sb = f"away_roll{w}_sb_success_rate"
        if h_sb in new_cols and a_sb in new_cols:
            new_cols[f"diff_roll{w}_sb_success_rate"] = new_cols[h_sb] - new_cols[a_sb]

    result = pd.DataFrame(new_cols, index=games.index).astype("float32")
    return pd.concat([games, result], axis=1)


# ---------------------------------------------------------------------------
# Defense and stranding features (from linescore errors/LOB)
# ---------------------------------------------------------------------------

def _defense_stranding_features(games: pd.DataFrame) -> pd.DataFrame:
    """Rolling defensive metrics: errors per game, LOB per game, stranding rate."""
    if "home_total_errors" not in games.columns:
        return games

    new_cols: dict[str, pd.Series] = {}

    parts = []
    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        if team_col not in games.columns:
            return games
        # When this team is fielding (pitching), errors are committed BY them
        # and LOB reflects runners THEY stranded on defense.
        # Errors: this team's defensive errors
        # LOB: runners left on base by the opposing offense (but stranding is
        # a pitcher/defense stat — high LOB = good defense/pitching).
        sub = pd.DataFrame({"team_id": games[team_col], "frame_idx": games.index, "side": side})
        sub["errors"] = pd.to_numeric(games[f"{side}_total_errors"], errors="coerce").values
        opp_side = "away" if side == "home" else "home"
        sub["lob"] = pd.to_numeric(games[f"{opp_side}_total_lob"], errors="coerce").values

        runs_col = f"{opp_side}_bat_game_runs"
        if runs_col in games.columns:
            opp_runs = pd.to_numeric(games[runs_col], errors="coerce").values
            safe_denom = (sub["lob"] + opp_runs).replace(0, np.nan)
            sub["strand_rate"] = sub["lob"] / safe_denom

        if "game_date" in games.columns:
            sub["_season"] = pd.to_datetime(games["game_date"]).dt.year.values
        else:
            sub["_season"] = 0
        parts.append(sub)

    timeline = pd.concat(parts, ignore_index=True)
    timeline = timeline.sort_values("frame_idx").reset_index(drop=True)

    cols_to_roll = ["errors", "lob"]
    if "strand_rate" in timeline.columns:
        cols_to_roll.append("strand_rate")

    for w in (10, 20):
        for col in cols_to_roll:
            timeline[f"{col}_roll{w}"] = (
                timeline.groupby(["team_id", "_season"])[col]
                .transform(lambda s, _w=w: s.rolling(_w, min_periods=5).mean().shift(1))
            )

    for side in ("home", "away"):
        side_rows = timeline[timeline["side"] == side].set_index("frame_idx")
        for w in (10, 20):
            new_cols[f"{side}_roll{w}_errors_pg"] = side_rows[f"errors_roll{w}"].reindex(games.index)
            new_cols[f"{side}_roll{w}_lob_pg"] = side_rows[f"lob_roll{w}"].reindex(games.index)
            if "strand_rate" in cols_to_roll:
                new_cols[f"{side}_roll{w}_strand_rate"] = side_rows[f"strand_rate_roll{w}"].reindex(games.index)

    result = pd.DataFrame(new_cols, index=games.index).astype("float32")
    return pd.concat([games, result], axis=1)


# ---------------------------------------------------------------------------
# Pennant race / standings context
# ---------------------------------------------------------------------------

def _pennant_race_features(
    games: pd.DataFrame, standings: pd.DataFrame = None
) -> pd.DataFrame:
    """Standings context features sourced from daily standings snapshots.

    Uses prior-day standings lookup (game_date - 1 day) to avoid leaking the
    current game's result. Division leaders have games_back=0.0 in the API.
    Season progress uses games_played from the pitches table with shift(1).
    """
    if "home_team_id" not in games.columns:
        return games

    new_cols: dict[str, pd.Series] = {}

    # --- Games-back from standings snapshots (prior-day lookup, no shift needed) ---
    if standings is not None and not standings.empty and "game_date" in games.columns:
        game_dates = pd.to_datetime(games["game_date"])
        lookup_dates = (game_dates - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")

        standings_lookup = standings[["team_id", "date", "games_back", "wild_card_games_back"]].copy()
        standings_lookup["date"] = standings_lookup["date"].astype(str)

        for side in ("home", "away"):
            team_col = f"{side}_team_id"
            keys = pd.DataFrame({
                "team_id": games[team_col].values,
                "date": lookup_dates.values,
                "_idx": games.index,
            })
            merged = keys.merge(standings_lookup, on=["team_id", "date"], how="left")
            merged = merged.drop_duplicates("_idx").set_index("_idx").reindex(games.index)

            new_cols[f"{side}_div_games_back"] = merged["games_back"].values
            new_cols[f"{side}_wc_games_back"] = merged["wild_card_games_back"].values
            new_cols[f"{side}_in_contention"] = (
                pd.to_numeric(merged["games_back"], errors="coerce") <= 5.0
            ).astype("float32").values

        if "home_div_games_back" in new_cols and "away_div_games_back" in new_cols:
            h = pd.to_numeric(pd.Series(new_cols["home_div_games_back"]), errors="coerce")
            a = pd.to_numeric(pd.Series(new_cols["away_div_games_back"]), errors="coerce")
            new_cols["diff_div_games_back"] = (h - a).values

    # --- Season progress from games_played (shift(1) per team per season) ---
    has_gp = "home_games_played" in games.columns
    if has_gp and "game_date" in games.columns:
        season_col = pd.to_datetime(games["game_date"]).dt.year

        parts = []
        for side in ("home", "away"):
            team_col = f"{side}_team_id"
            sub = pd.DataFrame({
                "team_id": games[team_col].values,
                "frame_idx": games.index,
                "side": side,
                "_season": season_col.values,
                "games_played": pd.to_numeric(
                    games[f"{side}_games_played"], errors="coerce"
                ).values,
            })
            parts.append(sub)

        timeline = pd.concat(parts, ignore_index=True)
        timeline = timeline.sort_values("frame_idx").reset_index(drop=True)
        timeline["gp_shifted"] = (
            timeline.groupby(["team_id", "_season"])["games_played"]
            .transform(lambda s: s.shift(1))
        )

        home_rows = timeline[timeline["side"] == "home"].set_index("frame_idx")
        gp = home_rows["gp_shifted"].reindex(games.index)
        new_cols["season_progress"] = (gp / 162.0).clip(0, 1).values

    if not new_cols:
        return games

    result = pd.DataFrame(new_cols, index=games.index).astype("float32")
    return pd.concat([games, result], axis=1)
