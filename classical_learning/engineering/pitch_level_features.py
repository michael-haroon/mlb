"""Pitch-level feature families for pregame ML.

Five feature families derived from raw pitch data:
  1. TTO (Times Through Order) velocity decay and release-point stability
  2. K-BB% splits by batter handedness
  3. FIP splits by batter handedness
  4. Platoon wOBA splits (batter vs L/R pitcher, aggregated to team)
  5. Pitch-mix matchup score (pitcher freq × batter wOBA vs pitch type)

All rolling aggregations use shift(1) at game level to prevent leakage.
Only regular season games (game_type_code == "R") are processed.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# FanGraphs run values averaged 2015-2024 — stable enough as static constants.
# Source: FanGraphs guts page (linear weights, wOBA scale).
WOBA_WEIGHTS: dict[str, float] = {
    "walk":      0.690,
    "hbp":       0.720,
    "single":    0.880,
    "double":    1.260,
    "triple":    1.590,
    "home_run":  2.080,
}

# Events that count in the wOBA denominator (PA).
# Excludes: sacrifice bunts, interference, catcher-obstruction, sac_fly_double_play
# (non-standard; rare enough that omitting doesn't materially affect wOBA).
WOBA_PA_DENOM_EVENTS: frozenset[str] = frozenset({
    "walk", "intent_walk", "hit_by_pitch",
    "single", "double", "triple", "home_run",
    "strikeout", "strikeout_double_play",
    "field_out", "grounded_into_double_play", "force_out",
    "double_play", "fielders_choice", "fielders_choice_out",
    "sac_fly",
})

# Pitch types modeled explicitly; everything else collapses to "other".
TRACKED_PITCH_TYPES: tuple[str, ...] = ("FF", "SL", "CH", "CU", "FC", "SI", "FS", "ST")

# FIP ERA-scale constant (league-calibrated; standard value).
FIP_CONSTANT = 3.10


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_pitch_level_features(
    pitches_raw: pd.DataFrame,
    game_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Compute TTO, K-BB% splits, FIP splits, platoon wOBA, and pitch-mix matchup features.

    Parameters
    ----------
    pitches_raw : pd.DataFrame
        Raw pitches table with one row per pitch event.
    game_frame : pd.DataFrame
        Game-level frame with probable_pitcher_home_id, probable_pitcher_away_id,
        home_team_id, away_team_id, game_pk, game_date.

    Returns
    -------
    pd.DataFrame
        game_frame with new pitch-level features joined in.
    """
    # Only regular season — postseason platoon splits behave differently
    # (small-sample roster manipulation), and including them adds noise.
    # 2020 excluded: 60-game pandemic season with structural anomalies (no fans,
    # universal DH, neutral-site rules) that contaminate rolling windows for 2021
    # starters/batters. Consistent with SKIP_SEASONS in strategy/config.py.
    pitches = pitches_raw[
        (pitches_raw["game_type_code"] == "R") & (pitches_raw["season"] != 2020)
    ].copy()
    log.info(f"pitch_level_features: {len(pitches):,} regular-season pitch rows (2020 excluded)")

    # Sort once; all downstream functions assume temporal order within pitcher/batter.
    pitches = pitches.sort_values(
        ["game_pk", "at_bat_index", "pitch_number"],
        na_position="last",
    ).reset_index(drop=True)

    log.info("Computing TTO (velocity decay / release stability) features...")
    tto_feats = _compute_tto_features(pitches, game_frame)

    log.info("Computing K-BB% splits by batter handedness...")
    kbb_feats = _compute_kbb_splits(pitches, game_frame)

    log.info("Computing FIP splits by batter handedness...")
    fip_feats = _compute_fip_splits(pitches, game_frame)

    log.info("Computing platoon wOBA splits...")
    woba_feats = _compute_woba_splits(pitches, game_frame)

    log.info("Computing pitch-mix matchup scores...")
    pitchmix_feats = _compute_pitchmix_matchup(pitches, game_frame)

    log.info("Computing batted ball quality features...")
    batted_ball_feats = _compute_batted_ball_features(pitches, game_frame)

    log.info("Computing spin & movement features...")
    spin_feats = _compute_spin_movement_features(pitches, game_frame)

    log.info("Computing command & plate discipline features...")
    command_feats = _compute_command_features(pitches, game_frame)

    log.info("Computing spray direction features...")
    spray_feats = _compute_spray_features(pitches, game_frame)

    log.info("Computing platoon composition features...")
    platoon_feats = _compute_platoon_composition(pitches, game_frame)

    log.info("Computing bat strength features (hit distance, TB/hit)...")
    bat_strength_feats = _compute_bat_strength_features(pitches, game_frame)

    log.info("Computing bullpen workload features...")
    bullpen_feats = _compute_bullpen_workload_features(pitches, game_frame)

    log.info("Computing manager tendency features (pitchers used, bunt rate)...")
    mgr_feats = _compute_manager_tendency_features(pitches, game_frame)

    # Join all feature blocks onto game_frame by game_pk.
    result = game_frame.copy()
    for feats in (tto_feats, kbb_feats, fip_feats, woba_feats, pitchmix_feats,
                  batted_ball_feats, spin_feats, command_feats, spray_feats, platoon_feats,
                  bat_strength_feats, bullpen_feats, mgr_feats):
        if feats is not None and len(feats) > 0:
            result = result.merge(feats, on="game_pk", how="left")

    log.info(
        f"pitch_level_features complete: added "
        f"{len(result.columns) - len(game_frame.columns)} columns"
    )
    return result


# ---------------------------------------------------------------------------
# Feature 1: TTO velocity decay and release-point stability
# ---------------------------------------------------------------------------

def _compute_tto_features(
    pitches: pd.DataFrame,
    game_frame: pd.DataFrame,
) -> pd.DataFrame:
    """TTO risk: velocity decay and release-point wobble per pitcher start."""
    # Only actual pitches have reliable velo/release data.
    p = pitches[pitches["is_pitch"] == True].copy()  # noqa: E712

    # Assign a pitch sequence number within each game+pitcher appearance
    # so we can slice early (1-25) vs late (75+) pitches.
    p["_pitch_seq"] = (
        p.groupby(["game_pk", "pitcher_id"]).cumcount() + 1
    )

    # Compute per-start summary stats.
    def _start_stats(grp: pd.DataFrame) -> pd.Series:
        velo = grp["release_speed"]
        early = velo[grp["_pitch_seq"] <= 25].mean()
        late  = velo[grp["_pitch_seq"] >= 75].mean()
        return pd.Series({
            "velo_decay":      late - early,   # negative = arm tired
            "release_x_std":   grp["coord_x0"].std(),
            "release_z_std":   grp["coord_z0"].std(),
        })

    starts = (
        p.groupby(["pitcher_id", "game_pk"])
        .apply(_start_stats, include_groups=False)
        .reset_index()
    )

    # Attach game_date so we can sort chronologically per pitcher.
    game_dates = game_frame[["game_pk", "game_date"]].drop_duplicates()
    starts = starts.merge(game_dates, on="game_pk", how="left")
    starts = starts.sort_values(["pitcher_id", "game_date", "game_pk"])

    # Rolling 5 and 10 starts with shift(1) — never include the current start.
    stat_cols = ["velo_decay", "release_x_std", "release_z_std"]
    for w in (5, 10):
        for col in stat_cols:
            roll_name = f"_tto_{col}_roll{w}"
            starts[roll_name] = (
                starts.groupby("pitcher_id")[col]
                .transform(lambda s, _w=w: s.rolling(_w, min_periods=max(2, _w // 2)).mean().shift(1))
            )

    new_cols: dict[str, pd.Series] = {}

    for side, pid_col in (("home", "probable_pitcher_home_id"), ("away", "probable_pitcher_away_id")):
        if pid_col not in game_frame.columns:
            continue
        sp_map = game_frame[["game_pk", pid_col]].rename(columns={pid_col: "pitcher_id"})
        roll_cols = [f"_tto_{c}_roll{w}" for w in (5, 10) for c in stat_cols]
        merged = sp_map.merge(
            starts[["game_pk", "pitcher_id"] + roll_cols],
            on=["game_pk", "pitcher_id"], how="left",
        ).set_index("game_pk").reindex(game_frame["game_pk"].values)

        for w in (5, 10):
            for col in stat_cols:
                feat_name = f"{side}_sp_tto_{col}_roll{w}"
                new_cols[feat_name] = merged[f"_tto_{col}_roll{w}"].values

    out = pd.DataFrame(new_cols, index=game_frame.index)
    out["game_pk"] = game_frame["game_pk"].values
    return out.astype({c: "float32" for c in out.columns if c != "game_pk"})


# ---------------------------------------------------------------------------
# Feature 2: K-BB% splits by batter handedness
# ---------------------------------------------------------------------------

_K_EVENTS   = frozenset({"strikeout", "strikeout_double_play"})
_BB_EVENTS  = frozenset({"walk", "intent_walk"})
_PA_EVENTS  = _K_EVENTS | _BB_EVENTS | frozenset({
    "single", "double", "triple", "home_run", "hit_by_pitch",
    "field_out", "grounded_into_double_play", "force_out",
    "double_play", "fielders_choice", "fielders_choice_out",
    "sac_fly", "sac_bunt",
})


def _compute_kbb_splits(
    pitches: pd.DataFrame,
    game_frame: pd.DataFrame,
) -> pd.DataFrame:
    """K%, BB%, K-BB% per pitcher vs L/R batters, rolling 5 and 10 games."""
    # PA terminals: last pitch of each at-bat (non-null event_type on is_pitch rows).
    pa = pitches[
        pitches["is_pitch"] == True  # noqa: E712
    ].dropna(subset=["event_type"]).copy()

    # Keep only events that count as a PA (avoid defensive indifference, etc.)
    pa = pa[pa["event_type"].isin(_PA_EVENTS)]

    # Collapse multiple pitch rows within a single PA to one row.
    pa = pa.drop_duplicates(subset=["pitcher_id", "bat_side_code", "game_pk", "at_bat_index"], keep="first")

    pa["_is_k"]  = pa["event_type"].isin(_K_EVENTS).astype("int8")
    pa["_is_bb"] = pa["event_type"].isin(_BB_EVENTS).astype("int8")

    # Per pitcher × batter handedness × game: count K, BB, PA.
    game_splits = (
        pa.groupby(["pitcher_id", "bat_side_code", "game_pk"])
        .agg(k=("_is_k", "sum"), bb=("_is_bb", "sum"), pa=("_is_k", "count"))
        .reset_index()
    )

    game_dates = game_frame[["game_pk", "game_date"]].drop_duplicates()
    game_splits = game_splits.merge(game_dates, on="game_pk", how="left")
    game_splits = game_splits.sort_values(["pitcher_id", "bat_side_code", "game_date", "game_pk"])

    new_cols: dict[str, pd.Series] = {}

    for hand in ("L", "R"):
        hand_tag = "lhh" if hand == "L" else "rhh"
        sub = game_splits[game_splits["bat_side_code"] == hand].copy()

        for w in (5, 10):
            roll_k  = sub.groupby("pitcher_id")["k"].transform(
                lambda s, _w=w: s.rolling(_w, min_periods=max(2, _w // 2)).sum().shift(1))
            roll_bb = sub.groupby("pitcher_id")["bb"].transform(
                lambda s, _w=w: s.rolling(_w, min_periods=max(2, _w // 2)).sum().shift(1))
            roll_pa = sub.groupby("pitcher_id")["pa"].transform(
                lambda s, _w=w: s.rolling(_w, min_periods=max(2, _w // 2)).sum().shift(1))

            safe_pa = roll_pa.replace(0, np.nan)
            sub[f"_kpct_roll{w}"]  = roll_k  / safe_pa
            sub[f"_bbpct_roll{w}"] = roll_bb / safe_pa
            sub[f"_kbb_roll{w}"]   = sub[f"_kpct_roll{w}"] - sub[f"_bbpct_roll{w}"]

        for side, pid_col in (("home", "probable_pitcher_home_id"), ("away", "probable_pitcher_away_id")):
            if pid_col not in game_frame.columns:
                continue
            sp_map = game_frame[["game_pk", pid_col]].rename(columns={pid_col: "pitcher_id"})
            merge_cols = ["game_pk", "pitcher_id"] + [
                f"_{m}_roll{w}" for w in (5, 10) for m in ("kpct", "bbpct", "kbb")
            ]
            merged = sp_map.merge(
                sub[merge_cols], on=["game_pk", "pitcher_id"], how="left",
            ).set_index("game_pk").reindex(game_frame["game_pk"].values)

            for w in (5, 10):
                for metric, src in (("kpct", "kpct"), ("bbpct", "bbpct"), ("kbb_diff", "kbb")):
                    new_cols[f"{side}_sp_{metric}_vs_{hand_tag}_roll{w}"] = merged[f"_{src}_roll{w}"].values

    out = pd.DataFrame(new_cols, index=game_frame.index)
    out["game_pk"] = game_frame["game_pk"].values
    return out.astype({c: "float32" for c in out.columns if c != "game_pk"})


# ---------------------------------------------------------------------------
# Feature 3: FIP splits by batter handedness
# ---------------------------------------------------------------------------

_FIP_HR_EVENTS  = frozenset({"home_run"})
# Standard FIP excludes intentional walks (IBB) from the BB component —
# IBB are manager decisions, not pitcher skill. Source: FanGraphs FIP definition.
_FIP_BB_EVENTS  = frozenset({"walk"})
_FIP_HBP_EVENTS = frozenset({"hit_by_pitch"})
_FIP_K_EVENTS   = _K_EVENTS
_OUT_EVENTS = frozenset({
    "strikeout", "strikeout_double_play", "field_out",
    "grounded_into_double_play", "force_out", "double_play",
    "fielders_choice_out", "sac_bunt", "sac_fly",
})
_DP_EVENTS = frozenset({"grounded_into_double_play", "double_play", "strikeout_double_play"})


def _compute_fip_splits(
    pitches: pd.DataFrame,
    game_frame: pd.DataFrame,
) -> pd.DataFrame:
    """FIP vs L/R batters per pitcher, rolling 5 and 10 games."""
    pa = pitches[
        pitches["is_pitch"] == True  # noqa: E712
    ].dropna(subset=["event_type"]).copy()

    # Collapse multiple pitch rows within a single PA to one row.
    pa = pa.drop_duplicates(subset=["pitcher_id", "bat_side_code", "game_pk", "at_bat_index"], keep="first")

    pa["_hr"]  = pa["event_type"].isin(_FIP_HR_EVENTS).astype("int8")
    pa["_bb"]  = pa["event_type"].isin(_FIP_BB_EVENTS).astype("int8")
    pa["_hbp"] = pa["event_type"].isin(_FIP_HBP_EVENTS).astype("int8")
    pa["_k"]   = pa["event_type"].isin(_FIP_K_EVENTS).astype("int8")
    # Count outs: double-play events = 2 outs, single-out events = 1.
    pa["_out"] = pa["event_type"].isin(_OUT_EVENTS).astype("int8")
    pa.loc[pa["event_type"].isin(_DP_EVENTS), "_out"] = 2

    game_splits = (
        pa.groupby(["pitcher_id", "bat_side_code", "game_pk"])
        .agg(hr=("_hr", "sum"), bb=("_bb", "sum"), hbp=("_hbp", "sum"), k=("_k", "sum"), outs=("_out", "sum"))
        .reset_index()
    )
    # IP = outs / 3
    game_splits["ip"] = game_splits["outs"] / 3.0

    game_dates = game_frame[["game_pk", "game_date"]].drop_duplicates()
    game_splits = game_splits.merge(game_dates, on="game_pk", how="left")
    game_splits = game_splits.sort_values(["pitcher_id", "bat_side_code", "game_date", "game_pk"])

    new_cols: dict[str, pd.Series] = {}

    for hand in ("L", "R"):
        hand_tag = "lhh" if hand == "L" else "rhh"
        sub = game_splits[game_splits["bat_side_code"] == hand].copy()

        for w in (5, 10):
            roll_hr  = sub.groupby("pitcher_id")["hr"].transform(
                lambda s, _w=w: s.rolling(_w, min_periods=max(2, _w // 2)).sum().shift(1))
            roll_bb  = sub.groupby("pitcher_id")["bb"].transform(
                lambda s, _w=w: s.rolling(_w, min_periods=max(2, _w // 2)).sum().shift(1))
            roll_hbp = sub.groupby("pitcher_id")["hbp"].transform(
                lambda s, _w=w: s.rolling(_w, min_periods=max(2, _w // 2)).sum().shift(1))
            roll_k   = sub.groupby("pitcher_id")["k"].transform(
                lambda s, _w=w: s.rolling(_w, min_periods=max(2, _w // 2)).sum().shift(1))
            roll_ip  = sub.groupby("pitcher_id")["ip"].transform(
                lambda s, _w=w: s.rolling(_w, min_periods=max(2, _w // 2)).sum().shift(1))

            safe_ip = roll_ip.replace(0, np.nan)
            # Standard FIP: (13*HR + 3*(BB+HBP) - 2*K) / IP + C. Source: FanGraphs.
            sub[f"_fip_roll{w}"] = (13 * roll_hr + 3 * (roll_bb + roll_hbp) - 2 * roll_k) / safe_ip + FIP_CONSTANT

        for side, pid_col in (("home", "probable_pitcher_home_id"), ("away", "probable_pitcher_away_id")):
            if pid_col not in game_frame.columns:
                continue
            sp_map = game_frame[["game_pk", pid_col]].rename(columns={pid_col: "pitcher_id"})
            merge_cols = ["game_pk", "pitcher_id"] + [f"_fip_roll{w}" for w in (5, 10)]
            merged = sp_map.merge(
                sub[merge_cols], on=["game_pk", "pitcher_id"], how="left",
            ).set_index("game_pk").reindex(game_frame["game_pk"].values)

            for w in (5, 10):
                new_cols[f"{side}_sp_fip_vs_{hand_tag}_roll{w}"] = merged[f"_fip_roll{w}"].values

    out = pd.DataFrame(new_cols, index=game_frame.index)
    out["game_pk"] = game_frame["game_pk"].values
    return out.astype({c: "float32" for c in out.columns if c != "game_pk"})


# ---------------------------------------------------------------------------
# Feature 4: Platoon wOBA splits (batter vs L/R pitcher, team-aggregated)
# ---------------------------------------------------------------------------

# Map event_type strings to wOBA weight keys.
_EVENT_TO_WOBA_KEY: dict[str, str] = {
    "walk":          "walk",
    "intent_walk":   "walk",
    "hit_by_pitch":  "hbp",
    "single":        "single",
    "double":        "double",
    "triple":        "triple",
    "home_run":      "home_run",
}


def _compute_woba_splits(
    pitches: pd.DataFrame,
    game_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Batter platoon wOBA vs L/R pitcher, rolled over 100/200 PA, aggregated to team."""
    pa = pitches[
        pitches["is_pitch"] == True  # noqa: E712
    ].dropna(subset=["event_type"]).copy()

    # Keep only countable PA events.
    pa = pa[pa["event_type"].isin(WOBA_PA_DENOM_EVENTS)]

    pa["_woba_num"] = pa["event_type"].map(_EVENT_TO_WOBA_KEY).map(WOBA_WEIGHTS).fillna(0.0)

    # game_date is already a column in the pitches table (PITCH_LEVEL_COLUMNS).
    # No merge needed — sorting directly avoids a game_date_x/game_date_y conflict.
    pa = pa.sort_values(["batter_id", "pitch_hand_code", "game_date", "game_pk", "at_bat_index"])

    # Collapse multiple pitch rows within a single PA to one row (event_type is
    # duplicated across all pitches in the same PA). Preserves all distinct PAs so
    # rolling(100) operates on 100 actual plate appearances, not 100 games.
    pa = pa.drop_duplicates(subset=["batter_id", "pitch_hand_code", "game_pk", "at_bat_index"], keep="first")

    new_cols: dict[str, pd.Series] = {}

    for hand in ("L", "R"):
        hand_tag = "lhp" if hand == "L" else "rhp"
        sub = pa[pa["pitch_hand_code"] == hand].copy()
        # pandas 3.0: groupby().apply() on empty DF returns empty DF, not empty Series —
        # assigning to a single column slot raises ValueError.
        if len(sub) == 0:
            continue

        for window_pa in (100, 200):
            def _roll_woba(grp: pd.DataFrame, _w: int = window_pa) -> pd.Series:
                num   = grp["_woba_num"].rolling(_w, min_periods=max(10, _w // 5)).sum().shift(1)
                denom = pd.Series(1, index=grp.index).rolling(_w, min_periods=max(10, _w // 5)).sum().shift(1)
                return num / denom.replace(0, np.nan)

            sub[f"_woba_roll{window_pa}pa"] = (
                sub.groupby("batter_id", group_keys=False)
                .apply(_roll_woba, include_groups=False)
            )

        # home batters bat in the bottom half (half_inning == "bottom").
        # away batters bat in the top half (half_inning == "top").
        sub_home = sub[sub["half_inning"] == "bottom"].copy()
        sub_home["_team_id"] = sub_home["home_team_id"]
        sub_away = sub[sub["half_inning"] == "top"].copy()
        sub_away["_team_id"] = sub_away["away_team_id"]

        batter_woba = pd.concat([sub_home, sub_away], ignore_index=False)

        for window_pa in (100, 200):
            col = f"_woba_roll{window_pa}pa"
            # Take each batter's last PA in the game (highest at_bat_index) — all PA
            # rows carry the same shift(1)-lagged rolling wOBA, so "last" just picks a
            # representative. Then average across batters for the team-game aggregate.
            valid = batter_woba[batter_woba[col].notna()]
            batter_game_woba = (
                valid.sort_values("at_bat_index")
                .groupby(["_team_id", "game_pk", "batter_id"])[col]
                .last()
                .reset_index()
            )
            team_game_woba = (
                batter_game_woba.groupby(["_team_id", "game_pk"])[col]
                .mean()
                .reset_index()
                .rename(columns={col: f"_team_woba_{hand_tag}_roll{window_pa}pa"})
            )

            for side in ("home", "away"):
                team_col = f"{side}_team_id"
                if team_col not in game_frame.columns:
                    continue

                game_team = game_frame[["game_pk", team_col]].rename(columns={team_col: "_team_id"})
                merged = game_team.merge(team_game_woba, on=["_team_id", "game_pk"], how="left")
                merged = merged.set_index("game_pk").reindex(game_frame["game_pk"].values)

                feat_name = f"{side}_team_woba_vs_{hand_tag}_roll{window_pa}pa"
                new_cols[feat_name] = merged[f"_team_woba_{hand_tag}_roll{window_pa}pa"].values

    out = pd.DataFrame(new_cols, index=game_frame.index)
    out["game_pk"] = game_frame["game_pk"].values
    return out.astype({c: "float32" for c in out.columns if c != "game_pk"})


# ---------------------------------------------------------------------------
# Feature 5: Pitch-mix matchup score
# ---------------------------------------------------------------------------

def _compute_pitchmix_matchup(
    pitches: pd.DataFrame,
    game_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Pitch-mix matchup: pitcher type-freq × batter wOBA vs that type, team-aggregated.

    Step A: pitcher pitch-type frequency profile, rolling 10 starts.
    Step B: batter wOBA against each pitch type, rolling 200 PA per type.
    Step C: dot-product → matchup score → mean across team lineup.
    """
    p = pitches[pitches["is_pitch"] == True].copy()  # noqa: E712

    # Normalize pitch type: collapse anything outside TRACKED_PITCH_TYPES to "other".
    p["_ptype"] = p["pitch_type"].where(p["pitch_type"].isin(TRACKED_PITCH_TYPES), other="other")

    # game_date is already present in the pitches table — no merge needed.
    game_dates = game_frame[["game_pk", "game_date"]].drop_duplicates()

    # -----------------------------------------------------------------------
    # Step A: pitcher pitch-type frequency per start, rolling 10 starts.
    # -----------------------------------------------------------------------
    total_per_start = (
        p.groupby(["pitcher_id", "game_pk"])["_ptype"]
        .count()
        .reset_index()
        .rename(columns={"_ptype": "_total"})
    )
    type_per_start = (
        p.groupby(["pitcher_id", "game_pk", "_ptype"])["is_pitch"]
        .count()
        .reset_index()
        .rename(columns={"is_pitch": "_count"})
    )
    type_per_start = type_per_start.merge(total_per_start, on=["pitcher_id", "game_pk"])
    type_per_start["_freq"] = type_per_start["_count"] / type_per_start["_total"].replace(0, np.nan)

    # Attach game_date from game_frame (type_per_start is already aggregated, has no game_date).
    type_per_start = type_per_start.merge(game_dates, on="game_pk", how="left")
    type_per_start = type_per_start.sort_values(["pitcher_id", "_ptype", "game_date", "game_pk"])

    # Rolling 10 starts with shift(1) per pitcher per pitch type.
    type_per_start["_freq_roll10"] = (
        type_per_start.groupby(["pitcher_id", "_ptype"])["_freq"]
        .transform(lambda s: s.rolling(10, min_periods=3).mean().shift(1))
    )

    # Staleness cap: zero out pitch types not thrown in >5 consecutive prior starts.
    # When a pitcher drops a type, the rolling freq retains a stale non-zero value;
    # this forces it to zero so post-normalization redistributes mass to active types.
    def _starts_since_thrown(s: pd.Series) -> pd.Series:
        """Count consecutive starts with zero usage, per (pitcher, ptype) group."""
        was_thrown = (s > 0).values
        result = np.zeros(len(s), dtype="int32")
        gap = 0
        for i in range(len(was_thrown)):
            if was_thrown[i]:
                gap = 0
            else:
                gap += 1
            result[i] = gap
        return pd.Series(result, index=s.index)

    type_per_start["_gap"] = (
        type_per_start.groupby(["pitcher_id", "_ptype"])["_count"]
        .transform(_starts_since_thrown)
    )
    # shift(1) within group: at prediction time we see the gap as of the PRIOR start
    type_per_start["_gap_shifted"] = (
        type_per_start.groupby(["pitcher_id", "_ptype"])["_gap"]
        .shift(1)
        .fillna(0)
    )
    stale_mask = type_per_start["_gap_shifted"] > 5
    type_per_start.loc[stale_mask, "_freq_roll10"] = 0.0

    # Pivot to one row per (pitcher_id, game_pk): columns = each pitch type freq.
    all_types = list(TRACKED_PITCH_TYPES) + ["other"]
    pitcher_profile = (
        type_per_start[["pitcher_id", "game_pk", "_ptype", "_freq_roll10"]]
        .pivot_table(index=["pitcher_id", "game_pk"], columns="_ptype", values="_freq_roll10")
        .reset_index()
    )
    pitcher_profile.columns.name = None
    # Ensure all type columns exist (some pitchers never throw certain types).
    for pt in all_types:
        if pt not in pitcher_profile.columns:
            pitcher_profile[pt] = np.nan

    # Normalize frequency profile rows to sum to 1.0. Independent per-type rolling
    # windows violate the probability simplex when pitchers add/drop pitch types.
    freq_cols = [c for c in all_types if c in pitcher_profile.columns]
    row_sums = pitcher_profile[freq_cols].sum(axis=1).replace(0, np.nan)
    pitcher_profile[freq_cols] = pitcher_profile[freq_cols].div(row_sums, axis=0)

    # -----------------------------------------------------------------------
    # Step B: batter wOBA against each pitch type, rolling 200 PA per type.
    # -----------------------------------------------------------------------
    pa = p.dropna(subset=["event_type"]).copy()
    pa = pa[pa["event_type"].isin(WOBA_PA_DENOM_EVENTS)]
    pa["_woba_num"] = pa["event_type"].map(_EVENT_TO_WOBA_KEY).map(WOBA_WEIGHTS).fillna(0.0)

    # game_date already in pa (from pitches table); sort directly.
    pa = pa.sort_values(["batter_id", "_ptype", "game_date", "game_pk", "at_bat_index"])

    batter_type_woba_parts = []
    for pt in all_types:
        sub = pa[pa["_ptype"] == pt].copy()
        if sub.empty:
            continue

        def _roll_woba_200(grp: pd.DataFrame) -> pd.Series:
            num   = grp["_woba_num"].rolling(200, min_periods=20).sum().shift(1)
            denom = pd.Series(1, index=grp.index).rolling(200, min_periods=20).sum().shift(1)
            return num / denom.replace(0, np.nan)

        sub[f"_bwoba_{pt}"] = (
            sub.groupby("batter_id", group_keys=False)
            .apply(_roll_woba_200, include_groups=False)
        )
        key_cols = ["batter_id", "game_pk", "half_inning", "home_team_id", "away_team_id"]
        # Drop duplicates before appending — a batter can have multiple pitches per PA of
        # type `pt` in one game, so the same (batter_id, game_pk) appears multiple times.
        # An outer merge across all pitch types without deduplication produces a cartesian
        # product that blows up rows by up to N_pitches_per_PA × N_types.
        part = (
            sub[key_cols + [f"_bwoba_{pt}"]]
            .drop_duplicates(subset=key_cols)
            .copy()
        )
        batter_type_woba_parts.append(part)

    if not batter_type_woba_parts:
        return pd.DataFrame({"game_pk": game_frame["game_pk"]})

    batter_woba_df = batter_type_woba_parts[0]
    key_cols = ["batter_id", "game_pk", "half_inning", "home_team_id", "away_team_id"]
    for df in batter_type_woba_parts[1:]:
        batter_woba_df = batter_woba_df.merge(df, on=key_cols, how="outer")

    # -----------------------------------------------------------------------
    # League-average wOBA per pitch type — computed from the dataset itself.
    # Used as fill value when a batter has insufficient history for a pitch type.
    # Uses most recent 3 seasons available in the data to stay within the current
    # regime (post-humidor 2022). Structural breaks at 2021/2022/2023 make a
    # pooled 2015-2024 mean invalid for FF/SL/CH/CU/FC/SI (Chow test p<0.001).
    # -----------------------------------------------------------------------
    if "season" in pa.columns:
        recent_seasons = sorted(pa["season"].unique())[-3:]
        pa_recent = pa[pa["season"].isin(recent_seasons)]
    else:
        pa_recent = pa
    league_avg_woba_by_type: dict[str, float] = {}
    for pt in all_types:
        sub = pa_recent[pa_recent["_ptype"] == pt]
        if len(sub) > 0:
            league_avg_woba_by_type[pt] = float(sub["_woba_num"].mean())
        else:
            league_avg_woba_by_type[pt] = 0.320
    log.debug(f"League-avg wOBA by type (recent 3 seasons {list(recent_seasons) if 'season' in pa.columns else 'all'}): "
              f"{league_avg_woba_by_type}")

    # -----------------------------------------------------------------------
    # Step C: matchup score per batter in each game.
    # -----------------------------------------------------------------------
    # home offense faces away SP (home batters bat in "bottom" half).
    # away offense faces home SP (away batters bat in "top" half).
    new_cols: dict[str, pd.Series] = {}

    for offense_side, def_pid_col, half in (
        ("home", "probable_pitcher_away_id",  "bottom"),
        ("away", "probable_pitcher_home_id",  "top"),
    ):
        if def_pid_col not in game_frame.columns:
            continue

        batter_side = batter_woba_df[batter_woba_df["half_inning"] == half].copy()
        if offense_side == "home":
            batter_side["_team_id"] = batter_side["home_team_id"]
        else:
            batter_side["_team_id"] = batter_side["away_team_id"]

        # Attach pitcher profile for the opposing SP.
        sp_map = game_frame[["game_pk", def_pid_col]].rename(columns={def_pid_col: "pitcher_id"})
        batter_side = batter_side.merge(sp_map, on="game_pk", how="left")
        profile_renamed = pitcher_profile.rename(columns={pt: f"_spfreq_{pt}" for pt in all_types})
        batter_side = batter_side.merge(profile_renamed, on=["game_pk", "pitcher_id"], how="left")

        # Dot-product: sum over pitch types of (sp_freq × batter_woba_vs_type).
        # Fill missing batter wOBA with expanding league-average for that pitch type
        # rather than 0.0 — a batter with no history is average, not terrible.
        # Per-type expanding mean accounts for structural breaks (deadened ball 2021,
        # humidor 2022) that shift league wOBA by pitch type over time.
        matchup_scores = pd.Series(0.0, index=batter_side.index)
        any_valid_freq = pd.Series(False, index=batter_side.index)
        for pt in all_types:
            bwoba_col  = f"_bwoba_{pt}"
            spfreq_col = f"_spfreq_{pt}"
            if bwoba_col in batter_side.columns and spfreq_col in batter_side.columns:
                fill_val = league_avg_woba_by_type.get(pt, 0.320)
                freq = batter_side[spfreq_col]
                any_valid_freq = any_valid_freq | freq.notna()
                matchup_scores = matchup_scores + (
                    batter_side[bwoba_col].fillna(fill_val) * freq.fillna(0.0)
                )
        # No SP profile → NaN (not 0.0 which models read as "worst matchup")
        batter_side["_matchup_score"] = matchup_scores.where(any_valid_freq, np.nan)

        # Mean batter matchup score per game.
        team_matchup = (
            batter_side.groupby("game_pk")["_matchup_score"]
            .mean()
            .reset_index()
            .rename(columns={"_matchup_score": f"_{offense_side}_matchup"})
        )

        merged = (
            game_frame[["game_pk"]].merge(team_matchup, on="game_pk", how="left")
            .set_index("game_pk")
            .reindex(game_frame["game_pk"].values)
        )
        new_cols[f"{offense_side}_team_pitchmix_matchup_score_roll10"] = (
            merged[f"_{offense_side}_matchup"].values
        )

    out = pd.DataFrame(new_cols, index=game_frame.index)
    out["game_pk"] = game_frame["game_pk"].values
    return out.astype({c: "float32" for c in out.columns if c != "game_pk"})


# ---------------------------------------------------------------------------
# Feature 6: Batted ball quality (Statcast EV, LA, barrel, trajectory)
# ---------------------------------------------------------------------------

def _compute_batted_ball_features(
    pitches: pd.DataFrame,
    game_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Batted ball quality: barrel rate, hard hit%, EV, ground/fly/line drive rates."""
    if "is_in_play" not in pitches.columns or "hit_launch_speed" not in pitches.columns:
        log.warning("Missing is_in_play or hit_launch_speed — skipping batted ball features")
        return pd.DataFrame({"game_pk": game_frame["game_pk"]})

    bip = pitches[
        (pitches["is_in_play"] == True) &  # noqa: E712
        pitches["hit_launch_speed"].notna()
    ].copy()

    if bip.empty:
        log.warning("No batted ball data available — skipping batted ball features")
        return pd.DataFrame({"game_pk": game_frame["game_pk"]})

    ev = bip["hit_launch_speed"]
    la = bip["hit_launch_angle"]

    # Full Statcast barrel definition: speed-dependent LA zone
    la_min = np.clip(26.0 - 2.0 * (ev - 98.0), 8.0, 26.0)
    la_max = np.clip(30.0 + 2.0 * (ev - 98.0), 30.0, 50.0)
    bip["_is_barrel"] = ((ev >= 98.0) & (la >= la_min) & (la <= la_max)).astype("float32")
    bip["_is_hard_hit"] = (ev >= 95.0).astype("float32")
    bip["_is_sweet_spot"] = ((la >= 8.0) & (la <= 32.0)).astype("float32")

    traj = bip["hit_trajectory"].str.lower() if "hit_trajectory" in bip.columns else pd.Series("", index=bip.index)
    bip["_is_gb"] = traj.isin(["ground_ball", "bunt_grounder"]).astype("float32")
    bip["_is_fb"] = traj.isin(["fly_ball", "popup", "bunt_popup"]).astype("float32")
    bip["_is_ld"] = traj.isin(["line_drive"]).astype("float32")

    game_dates = game_frame[["game_pk", "game_date"]].drop_duplicates()

    new_cols: dict[str, np.ndarray] = {}

    # --- SP-level: aggregate per pitcher per game, rolling 5/10 starts ---
    sp_game = bip.groupby(["pitcher_id", "game_pk"]).agg(
        _barrel_rate=("_is_barrel", "mean"),
        _hard_hit_pct=("_is_hard_hit", "mean"),
        _avg_ev=("hit_launch_speed", "mean"),
        _gb_rate=("_is_gb", "mean"),
        _fb_rate=("_is_fb", "mean"),
    ).reset_index()
    sp_game = sp_game.merge(game_dates, on="game_pk", how="left")
    sp_game = sp_game.sort_values(["pitcher_id", "game_date", "game_pk"])

    sp_stats = ["_barrel_rate", "_hard_hit_pct", "_avg_ev", "_gb_rate", "_fb_rate"]
    for w in (5, 10):
        mp = max(2, w // 2)
        for col in sp_stats:
            sp_game[f"{col}_roll{w}"] = (
                sp_game.groupby("pitcher_id")[col]
                .transform(lambda s, _w=w, _mp=mp: s.rolling(_w, min_periods=_mp).mean().shift(1))
            )

    for side, pid_col in (("home", "probable_pitcher_home_id"), ("away", "probable_pitcher_away_id")):
        if pid_col not in game_frame.columns:
            continue
        sp_map = game_frame[["game_pk", pid_col]].rename(columns={pid_col: "pitcher_id"})
        roll_cols = [f"{c}_roll{w}" for w in (5, 10) for c in sp_stats]
        merged = sp_map.merge(
            sp_game[["game_pk", "pitcher_id"] + roll_cols],
            on=["game_pk", "pitcher_id"], how="left",
        ).set_index("game_pk").reindex(game_frame["game_pk"].values)

        for w in (5, 10):
            for col in sp_stats:
                feat_name = f"{side}_sp{col}_allowed_roll{w}"
                new_cols[feat_name] = merged[f"{col}_roll{w}"].values

    # --- Team batting-level: aggregate per batting team per game ---
    bip["_batting_team_id"] = np.where(
        bip["half_inning"] == "bottom",
        bip["home_team_id"],
        bip["away_team_id"],
    )
    team_game = bip.groupby(["_batting_team_id", "game_pk"]).agg(
        _barrel_rate=("_is_barrel", "mean"),
        _hard_hit_pct=("_is_hard_hit", "mean"),
        _avg_ev=("hit_launch_speed", "mean"),
        _sweet_spot_pct=("_is_sweet_spot", "mean"),
        _gb_rate=("_is_gb", "mean"),
        _fb_rate=("_is_fb", "mean"),
        _ld_rate=("_is_ld", "mean"),
    ).reset_index()
    team_game = team_game.merge(game_dates, on="game_pk", how="left")
    team_game = team_game.sort_values(["_batting_team_id", "game_date", "game_pk"])

    team_stats = ["_barrel_rate", "_hard_hit_pct", "_avg_ev", "_sweet_spot_pct",
                  "_gb_rate", "_fb_rate", "_ld_rate"]
    for w in (10, 20):
        mp = max(3, w // 3)
        for col in team_stats:
            team_game[f"{col}_roll{w}"] = (
                team_game.groupby("_batting_team_id")[col]
                .transform(lambda s, _w=w, _mp=mp: s.rolling(_w, min_periods=_mp).mean().shift(1))
            )

    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        if team_col not in game_frame.columns:
            continue
        game_team = game_frame[["game_pk", team_col]].rename(columns={team_col: "_batting_team_id"})
        roll_cols = [f"{c}_roll{w}" for w in (10, 20) for c in team_stats]
        merged = game_team.merge(
            team_game[["_batting_team_id", "game_pk"] + roll_cols],
            on=["_batting_team_id", "game_pk"], how="left",
        ).set_index("game_pk").reindex(game_frame["game_pk"].values)

        for w in (10, 20):
            for col in team_stats:
                feat_name = f"{side}_bat{col}_roll{w}"
                new_cols[feat_name] = merged[f"{col}_roll{w}"].values

    # --- Differentials (home batting - away batting) ---
    for w in (10, 20):
        for col in ("_barrel_rate", "_hard_hit_pct", "_avg_ev"):
            h = new_cols.get(f"home_bat{col}_roll{w}")
            a = new_cols.get(f"away_bat{col}_roll{w}")
            if h is not None and a is not None:
                new_cols[f"diff_bat{col}_roll{w}"] = h - a

    out = pd.DataFrame(new_cols, index=game_frame.index)
    out["game_pk"] = game_frame["game_pk"].values
    return out.astype({c: "float32" for c in out.columns if c != "game_pk"})


# ---------------------------------------------------------------------------
# Feature 7: Spin & movement (Statcast, SP-level)
# ---------------------------------------------------------------------------

def _compute_spin_movement_features(
    pitches: pd.DataFrame,
    game_frame: pd.DataFrame,
) -> pd.DataFrame:
    """SP spin rate, movement, extension, and velo retention."""
    if "spin_rate" not in pitches.columns:
        log.warning("Missing spin_rate column — skipping spin/movement features")
        return pd.DataFrame({"game_pk": game_frame["game_pk"]})

    p = pitches[
        (pitches["is_pitch"] == True) &  # noqa: E712
        pitches["spin_rate"].notna()
    ].copy()

    if p.empty:
        log.warning("No spin data available — skipping spin/movement features")
        return pd.DataFrame({"game_pk": game_frame["game_pk"]})

    game_dates = game_frame[["game_pk", "game_date"]].drop_duplicates()

    # Per-pitcher per-game aggregation
    p["_pfx_x_abs"] = p["pfx_x"].abs()
    p["_velo_retention"] = np.where(
        p["release_speed"].notna() & (p["release_speed"] > 0),
        p["end_speed"] / p["release_speed"],
        np.nan,
    )

    # Overall stats per start
    sp_game = p.groupby(["pitcher_id", "game_pk"]).agg(
        _avg_spin=("spin_rate", "mean"),
        _avg_pfx_x_abs=("_pfx_x_abs", "mean"),
        _avg_pfx_z=("pfx_z", "mean"),
        _avg_extension=("extension", "mean"),
        _avg_velo_retention=("_velo_retention", "mean"),
    ).reset_index()

    # Per pitch-type spin (FF, SL, CH — most common + informative)
    for pt in ("FF", "SL", "CH"):
        pt_data = p[p["pitch_type"] == pt].groupby(["pitcher_id", "game_pk"])["spin_rate"].mean()
        pt_data = pt_data.reset_index().rename(columns={"spin_rate": f"_avg_spin_{pt}"})
        sp_game = sp_game.merge(pt_data, on=["pitcher_id", "game_pk"], how="left")

    sp_game = sp_game.merge(game_dates, on="game_pk", how="left")
    sp_game = sp_game.sort_values(["pitcher_id", "game_date", "game_pk"])

    stat_cols = ["_avg_spin", "_avg_pfx_x_abs", "_avg_pfx_z", "_avg_extension",
                 "_avg_velo_retention", "_avg_spin_FF", "_avg_spin_SL", "_avg_spin_CH"]
    available_stats = [c for c in stat_cols if c in sp_game.columns]

    for w in (5, 10):
        mp = 3 if w == 5 else 5
        for col in available_stats:
            sp_game[f"{col}_roll{w}"] = (
                sp_game.groupby("pitcher_id")[col]
                .transform(lambda s, _w=w, _mp=mp: s.rolling(_w, min_periods=_mp).mean().shift(1))
            )

    # Spin trend: ratio of short-term to long-term rolling (fatigue/decline detector)
    if "_avg_spin" in sp_game.columns:
        sp_game["_spin_trend_roll3"] = (
            sp_game.groupby("pitcher_id")["_avg_spin"]
            .transform(lambda s: s.rolling(3, min_periods=2).mean().shift(1))
        )
        sp_game["_spin_trend"] = np.where(
            sp_game["_avg_spin_roll10"].notna() & (sp_game["_avg_spin_roll10"] > 0),
            sp_game["_spin_trend_roll3"] / sp_game["_avg_spin_roll10"],
            np.nan,
        )

    new_cols: dict[str, np.ndarray] = {}

    for side, pid_col in (("home", "probable_pitcher_home_id"), ("away", "probable_pitcher_away_id")):
        if pid_col not in game_frame.columns:
            continue
        sp_map = game_frame[["game_pk", pid_col]].rename(columns={pid_col: "pitcher_id"})
        roll_cols = [f"{c}_roll{w}" for w in (5, 10) for c in available_stats]
        if "_spin_trend" in sp_game.columns:
            roll_cols.append("_spin_trend")
        existing_roll = [c for c in roll_cols if c in sp_game.columns]

        merged = sp_map.merge(
            sp_game[["game_pk", "pitcher_id"] + existing_roll],
            on=["game_pk", "pitcher_id"], how="left",
        ).set_index("game_pk").reindex(game_frame["game_pk"].values)

        for w in (5, 10):
            for col in available_stats:
                rc = f"{col}_roll{w}"
                if rc in merged.columns:
                    new_cols[f"{side}_sp{col}_roll{w}"] = merged[rc].values
        if "_spin_trend" in merged.columns:
            new_cols[f"{side}_sp_spin_trend"] = merged["_spin_trend"].values

    out = pd.DataFrame(new_cols, index=game_frame.index)
    out["game_pk"] = game_frame["game_pk"].values
    return out.astype({c: "float32" for c in out.columns if c != "game_pk"})


# ---------------------------------------------------------------------------
# Feature 8: Command & plate discipline
# ---------------------------------------------------------------------------

_SWING_CALLS = frozenset({
    "Swinging Strike", "Swinging Strike (Blocked)", "Foul", "Foul Tip",
    "Foul Bunt", "Missed Bunt", "In play, out(s)", "In play, no out", "In play, run(s)",
})
_WHIFF_CALLS = frozenset({"Swinging Strike", "Swinging Strike (Blocked)"})
_CALLED_STRIKE_CALLS = frozenset({"Called Strike"})


def _compute_command_features(
    pitches: pd.DataFrame,
    game_frame: pd.DataFrame,
) -> pd.DataFrame:
    """SP command (zone%, chase induced, whiff rate, CSW%) + team batting discipline."""
    p = pitches[pitches["is_pitch"] == True].copy()  # noqa: E712

    if p.empty or "pitch_call" not in p.columns:
        log.warning("No pitch_call data — skipping command features")
        return pd.DataFrame({"game_pk": game_frame["game_pk"]})

    game_dates = game_frame[["game_pk", "game_date"]].drop_duplicates()

    # Classify zones and outcomes
    zone = pd.to_numeric(p["zone_location"], errors="coerce")
    p["_in_zone"] = ((zone >= 1) & (zone <= 9)).astype("float32")
    p["_out_zone"] = (~((zone >= 1) & (zone <= 9)) & zone.notna()).astype("float32")
    p["_is_swing"] = p["pitch_call"].isin(_SWING_CALLS).astype("float32")
    p["_is_whiff"] = p["pitch_call"].isin(_WHIFF_CALLS).astype("float32")
    p["_is_called_strike"] = p["pitch_call"].isin(_CALLED_STRIKE_CALLS).astype("float32")
    p["_is_csw"] = (p["_is_whiff"] + p["_is_called_strike"]).clip(upper=1.0)

    # First pitch of at-bat
    p["_is_first_pitch"] = (p["pitch_number"] == 1).astype("float32")
    p["_first_pitch_strike"] = (
        p["_is_first_pitch"] * (p["_in_zone"] + p["_is_swing"]).clip(upper=1.0)
    )

    # --- SP-level aggregation ---
    sp_agg = p.groupby(["pitcher_id", "game_pk"]).agg(
        _zone_pct=("_in_zone", "mean"),
        _whiff_rate=("_is_whiff", "mean"),
        _csw_pct=("_is_csw", "mean"),
        _total_pitches=("_is_swing", "count"),
    ).reset_index()

    # First pitch strike pct: mean of _first_pitch_strike among first pitches only
    fp = p[p["_is_first_pitch"] == 1.0]
    fp_agg = fp.groupby(["pitcher_id", "game_pk"]).agg(
        _first_pitch_strike_pct=("_first_pitch_strike", "mean"),
    ).reset_index()
    sp_agg = sp_agg.merge(fp_agg, on=["pitcher_id", "game_pk"], how="left")

    # Chase rate induced: swings at O-zone pitches / O-zone pitches
    o_zone = p[p["_out_zone"] == 1.0]
    chase = o_zone.groupby(["pitcher_id", "game_pk"]).agg(
        _o_zone_swings=("_is_swing", "sum"),
        _o_zone_pitches=("_out_zone", "count"),
    ).reset_index()
    chase["_chase_rate_induced"] = (
        chase["_o_zone_swings"] / chase["_o_zone_pitches"].replace(0, np.nan)
    )
    sp_agg = sp_agg.merge(
        chase[["pitcher_id", "game_pk", "_chase_rate_induced"]],
        on=["pitcher_id", "game_pk"], how="left",
    )

    sp_agg = sp_agg.merge(game_dates, on="game_pk", how="left")
    sp_agg = sp_agg.sort_values(["pitcher_id", "game_date", "game_pk"])

    sp_stats = ["_zone_pct", "_first_pitch_strike_pct", "_whiff_rate",
                "_csw_pct", "_chase_rate_induced"]
    for w in (5, 10):
        mp = 3 if w == 5 else 5
        for col in sp_stats:
            sp_agg[f"{col}_roll{w}"] = (
                sp_agg.groupby("pitcher_id")[col]
                .transform(lambda s, _w=w, _mp=mp: s.rolling(_w, min_periods=_mp).mean().shift(1))
            )

    new_cols: dict[str, np.ndarray] = {}

    for side, pid_col in (("home", "probable_pitcher_home_id"), ("away", "probable_pitcher_away_id")):
        if pid_col not in game_frame.columns:
            continue
        sp_map = game_frame[["game_pk", pid_col]].rename(columns={pid_col: "pitcher_id"})
        roll_cols = [f"{c}_roll{w}" for w in (5, 10) for c in sp_stats]
        merged = sp_map.merge(
            sp_agg[["game_pk", "pitcher_id"] + roll_cols],
            on=["game_pk", "pitcher_id"], how="left",
        ).set_index("game_pk").reindex(game_frame["game_pk"].values)

        for w in (5, 10):
            for col in sp_stats:
                new_cols[f"{side}_sp{col}_roll{w}"] = merged[f"{col}_roll{w}"].values

    # --- Team batting-level: discipline metrics ---
    p["_batting_team_id"] = np.where(
        p["half_inning"] == "bottom",
        p["home_team_id"],
        p["away_team_id"],
    )

    # Team chase: swing rate on O-zone pitches
    team_o_zone = p[p["_out_zone"] == 1.0]
    team_chase = team_o_zone.groupby(["_batting_team_id", "game_pk"]).agg(
        _team_chase_rate=("_is_swing", "mean"),
    ).reset_index()

    # Team whiff + contact
    team_all = p.groupby(["_batting_team_id", "game_pk"]).agg(
        _team_whiff_rate=("_is_whiff", "mean"),
        _swings=("_is_swing", "sum"),
        _whiffs=("_is_whiff", "sum"),
    ).reset_index()
    team_all["_team_contact_rate"] = np.where(
        team_all["_swings"] > 0,
        1.0 - team_all["_whiffs"] / team_all["_swings"],
        np.nan,
    )

    team_game = team_chase.merge(
        team_all[["_batting_team_id", "game_pk", "_team_whiff_rate", "_team_contact_rate"]],
        on=["_batting_team_id", "game_pk"], how="outer",
    )
    team_game = team_game.merge(game_dates, on="game_pk", how="left")
    team_game = team_game.sort_values(["_batting_team_id", "game_date", "game_pk"])

    team_stats = ["_team_chase_rate", "_team_whiff_rate", "_team_contact_rate"]
    for w in (10, 20):
        mp = max(3, w // 3)
        for col in team_stats:
            if col in team_game.columns:
                team_game[f"{col}_roll{w}"] = (
                    team_game.groupby("_batting_team_id")[col]
                    .transform(lambda s, _w=w, _mp=mp: s.rolling(_w, min_periods=_mp).mean().shift(1))
                )

    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        if team_col not in game_frame.columns:
            continue
        game_team = game_frame[["game_pk", team_col]].rename(columns={team_col: "_batting_team_id"})
        roll_cols = [f"{c}_roll{w}" for w in (10, 20) for c in team_stats
                     if f"{c}_roll{w}" in team_game.columns]
        merged = game_team.merge(
            team_game[["_batting_team_id", "game_pk"] + roll_cols],
            on=["_batting_team_id", "game_pk"], how="left",
        ).set_index("game_pk").reindex(game_frame["game_pk"].values)

        for w in (10, 20):
            for col in team_stats:
                rc = f"{col}_roll{w}"
                if rc in merged.columns:
                    new_cols[f"{side}{col}_roll{w}"] = merged[rc].values

    # Differentials
    for w in (10, 20):
        for col in team_stats:
            h = new_cols.get(f"home{col}_roll{w}")
            a = new_cols.get(f"away{col}_roll{w}")
            if h is not None and a is not None:
                new_cols[f"diff{col}_roll{w}"] = h - a

    out = pd.DataFrame(new_cols, index=game_frame.index)
    out["game_pk"] = game_frame["game_pk"].values
    return out.astype({c: "float32" for c in out.columns if c != "game_pk"})


# ---------------------------------------------------------------------------
# Feature 9: Spray direction (team batting pull/center/oppo)
# ---------------------------------------------------------------------------

def _compute_spray_features(
    pitches: pd.DataFrame,
    game_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Team batting spray direction: pull%, center%, oppo% by batter handedness."""
    if "hit_coord_x" not in pitches.columns or "hit_coord_y" not in pitches.columns:
        log.warning("Missing hit_coord columns — skipping spray features")
        return pd.DataFrame({"game_pk": game_frame["game_pk"]})

    if "is_in_play" not in pitches.columns:
        log.warning("Missing is_in_play — skipping spray features")
        return pd.DataFrame({"game_pk": game_frame["game_pk"]})

    bip = pitches[
        (pitches["is_in_play"] == True) &  # noqa: E712
        pitches["hit_coord_x"].notna() &
        pitches["hit_coord_y"].notna()
    ].copy()

    if bip.empty:
        log.warning("No spray coord data — skipping spray features")
        return pd.DataFrame({"game_pk": game_frame["game_pk"]})

    game_dates = game_frame[["game_pk", "game_date"]].drop_duplicates()

    # Spray angle from home plate (pixel coords: home at ~(125, 200), y inverted)
    dx = bip["hit_coord_x"] - 125.0
    dy = 200.0 - bip["hit_coord_y"]
    angle = np.degrees(np.arctan2(dx, dy))

    # Pull/oppo relative to batter hand
    is_rhh = bip["bat_side_code"] == "R"
    bip["_is_pull"] = np.where(is_rhh, angle < -15.0, angle > 15.0).astype("float32")
    bip["_is_oppo"] = np.where(is_rhh, angle > 15.0, angle < -15.0).astype("float32")
    bip["_is_center"] = ((angle >= -15.0) & (angle <= 15.0)).astype("float32")

    # Team assignment via half_inning
    bip["_batting_team_id"] = np.where(
        bip["half_inning"] == "bottom",
        bip["home_team_id"],
        bip["away_team_id"],
    )

    team_game = bip.groupby(["_batting_team_id", "game_pk"]).agg(
        _pull_pct=("_is_pull", "mean"),
        _center_pct=("_is_center", "mean"),
        _oppo_pct=("_is_oppo", "mean"),
    ).reset_index()
    team_game = team_game.merge(game_dates, on="game_pk", how="left")
    team_game = team_game.sort_values(["_batting_team_id", "game_date", "game_pk"])

    spray_stats = ["_pull_pct", "_center_pct", "_oppo_pct"]
    for w in (10, 20):
        mp = max(3, w // 3)
        for col in spray_stats:
            team_game[f"{col}_roll{w}"] = (
                team_game.groupby("_batting_team_id")[col]
                .transform(lambda s, _w=w, _mp=mp: s.rolling(_w, min_periods=_mp).mean().shift(1))
            )

    new_cols: dict[str, np.ndarray] = {}

    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        if team_col not in game_frame.columns:
            continue
        game_team = game_frame[["game_pk", team_col]].rename(columns={team_col: "_batting_team_id"})
        roll_cols = [f"{c}_roll{w}" for w in (10, 20) for c in spray_stats]
        merged = game_team.merge(
            team_game[["_batting_team_id", "game_pk"] + roll_cols],
            on=["_batting_team_id", "game_pk"], how="left",
        ).set_index("game_pk").reindex(game_frame["game_pk"].values)

        for w in (10, 20):
            for col in spray_stats:
                new_cols[f"{side}_bat{col}_roll{w}"] = merged[f"{col}_roll{w}"].values

    out = pd.DataFrame(new_cols, index=game_frame.index)
    out["game_pk"] = game_frame["game_pk"].values
    return out.astype({c: "float32" for c in out.columns if c != "game_pk"})


# ---------------------------------------------------------------------------
# Feature 10: Platoon composition (team LHH% → platoon advantage index)
# ---------------------------------------------------------------------------

def _compute_platoon_composition(
    pitches: pd.DataFrame,
    game_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Team LHH/RHH PA composition + platoon advantage index vs opposing SP hand."""
    # Count PAs by batter hand per team per game (PA = one per at_bat_index)
    pa = pitches.drop_duplicates(["game_pk", "at_bat_index"]).copy()

    if pa.empty or "bat_side_code" not in pa.columns:
        log.warning("No PA/bat_side data — skipping platoon composition")
        return pd.DataFrame({"game_pk": game_frame["game_pk"]})

    game_dates = game_frame[["game_pk", "game_date"]].drop_duplicates()

    pa["_batting_team_id"] = np.where(
        pa["half_inning"] == "bottom",
        pa["home_team_id"],
        pa["away_team_id"],
    )
    pa["_is_lhh"] = (pa["bat_side_code"] == "L").astype("float32")

    team_game = pa.groupby(["_batting_team_id", "game_pk"]).agg(
        _lhh_pct=("_is_lhh", "mean"),
    ).reset_index()
    team_game = team_game.merge(game_dates, on="game_pk", how="left")
    team_game = team_game.sort_values(["_batting_team_id", "game_date", "game_pk"])

    # Rolling 20 games
    team_game["_lhh_pct_roll20"] = (
        team_game.groupby("_batting_team_id")["_lhh_pct"]
        .transform(lambda s: s.rolling(20, min_periods=7).mean().shift(1))
    )

    new_cols: dict[str, np.ndarray] = {}

    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        if team_col not in game_frame.columns:
            continue
        game_team = game_frame[["game_pk", team_col]].rename(columns={team_col: "_batting_team_id"})
        merged = game_team.merge(
            team_game[["_batting_team_id", "game_pk", "_lhh_pct_roll20"]],
            on=["_batting_team_id", "game_pk"], how="left",
        ).set_index("game_pk").reindex(game_frame["game_pk"].values)

        new_cols[f"{side}_bat_lhh_pct_roll20"] = merged["_lhh_pct_roll20"].values

    # Platoon advantage index: high LHH% is advantageous vs RHP and vice versa
    for side, opp_side in (("home", "away"), ("away", "home")):
        sp_hand_col = f"sp_{opp_side}_hand"
        lhh_col = f"{side}_bat_lhh_pct_roll20"
        if sp_hand_col in game_frame.columns and lhh_col in new_cols:
            lhh = new_cols[lhh_col]
            sp_hand = game_frame[sp_hand_col].values
            # vs RHP: advantage = LHH% (more lefties vs righty pitcher)
            # vs LHP: advantage = 1 - LHH% (more righties vs lefty pitcher)
            adv = np.where(
                sp_hand == "R", lhh,
                np.where(sp_hand == "L", 1.0 - lhh, np.nan),
            )
            new_cols[f"{side}_platoon_advantage_index"] = adv

    out = pd.DataFrame(new_cols, index=game_frame.index)
    out["game_pk"] = game_frame["game_pk"].values
    return out.astype({c: "float32" for c in out.columns if c != "game_pk"})


# ---------------------------------------------------------------------------
# Feature 11: Bat strength — hit distance and total-bases-per-hit (Group A)
# ---------------------------------------------------------------------------

def _compute_bat_strength_features(
    pitches: pd.DataFrame,
    game_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Rolling bat-strength metrics derived from BIP outcomes.

    A1: avg_hit_distance — mean tracked distance of all balls in play per
        (batting team, game), rolling 10 and 20 games.  Range ~150-220 ft.
    A2: tb_per_hit — total bases per hit (Single=1 … HR=4).  Guards hits=0.
        Rolling 10 and 20 games.  Range 1.0-4.0, typical 1.3-1.8.

    xwOBA (A3) is skipped: estimated_woba_using_speedangle is absent from the
    GUMBO feed parquet (Savant-only model output, see data recon).
    """
    new_cols: dict[str, np.ndarray] = {}
    game_dates = game_frame[["game_pk", "game_date"]].drop_duplicates()

    # ── A1: Average hit distance ────────────────────────────────────────────
    if "hit_total_distance" in pitches.columns:
        bip_dist = pitches[
            pitches["hit_total_distance"].notna() &
            (pitches["hit_total_distance"] > 0)
        ].copy()

        bip_dist["_batting_team_id"] = np.where(
            bip_dist["half_inning"] == "bottom",
            bip_dist["home_team_id"],
            bip_dist["away_team_id"],
        )

        dist_game = (
            bip_dist.groupby(["_batting_team_id", "game_pk"])["hit_total_distance"]
            .mean()
            .reset_index(name="_avg_hit_dist")
        )
        dist_game = dist_game.merge(game_dates, on="game_pk", how="left")
        dist_game = dist_game.sort_values(["_batting_team_id", "game_date", "game_pk"])

        for w in (10, 20):
            mp = max(3, w // 2)
            dist_game[f"_avg_hit_dist_roll{w}"] = (
                dist_game.groupby("_batting_team_id")["_avg_hit_dist"]
                .transform(lambda s, _w=w, _mp=mp: s.rolling(_w, min_periods=_mp).mean().shift(1))
            )

        for side in ("home", "away"):
            team_col = f"{side}_team_id"
            if team_col not in game_frame.columns:
                continue
            game_team = game_frame[["game_pk", team_col]].rename(
                columns={team_col: "_batting_team_id"}
            )
            roll_cols = [f"_avg_hit_dist_roll{w}" for w in (10, 20)]
            merged = game_team.merge(
                dist_game[["_batting_team_id", "game_pk"] + roll_cols],
                on=["_batting_team_id", "game_pk"], how="left",
            ).set_index("game_pk").reindex(game_frame["game_pk"].values)
            for w in (10, 20):
                new_cols[f"{side}_bat_avg_hit_distance_roll{w}"] = (
                    merged[f"_avg_hit_dist_roll{w}"].values
                )

        # Differential (home − away)
        for w in (10, 20):
            h = new_cols.get(f"home_bat_avg_hit_distance_roll{w}")
            a = new_cols.get(f"away_bat_avg_hit_distance_roll{w}")
            if h is not None and a is not None:
                new_cols[f"diff_bat_avg_hit_distance_roll{w}"] = h - a
    else:
        log.warning("Missing hit_total_distance — skipping bat avg-hit-distance features")

    # ── A2: Total bases per hit ─────────────────────────────────────────────
    # event_type is snake_case in this parquet (confirmed by existing kbb_splits usage)
    if "event_type" in pitches.columns:
        TB_MAP: dict[str, int] = {
            "single": 1, "double": 2, "triple": 3, "home_run": 4,
        }
        # Terminal at-bat rows that are hits (deduped to one per at-bat)
        hit_rows = pitches[
            pitches["event_type"].isin(TB_MAP)
        ].drop_duplicates(subset=["game_pk", "at_bat_index"]).copy()

        hit_rows["_tb"] = hit_rows["event_type"].map(TB_MAP)
        hit_rows["_batting_team_id"] = np.where(
            hit_rows["half_inning"] == "bottom",
            hit_rows["home_team_id"],
            hit_rows["away_team_id"],
        )

        tb_game = hit_rows.groupby(["_batting_team_id", "game_pk"]).agg(
            _total_bases=("_tb", "sum"),
            _hits=("_tb", "count"),
        ).reset_index()
        # Guard hits == 0 (cannot happen given filter above, but explicit)
        tb_game["_tb_per_hit"] = (
            tb_game["_total_bases"] / tb_game["_hits"].replace(0, np.nan)
        )
        tb_game = tb_game.merge(game_dates, on="game_pk", how="left")
        tb_game = tb_game.sort_values(["_batting_team_id", "game_date", "game_pk"])

        for w in (10, 20):
            mp = max(3, w // 2)
            tb_game[f"_tb_per_hit_roll{w}"] = (
                tb_game.groupby("_batting_team_id")["_tb_per_hit"]
                .transform(lambda s, _w=w, _mp=mp: s.rolling(_w, min_periods=_mp).mean().shift(1))
            )

        for side in ("home", "away"):
            team_col = f"{side}_team_id"
            if team_col not in game_frame.columns:
                continue
            game_team = game_frame[["game_pk", team_col]].rename(
                columns={team_col: "_batting_team_id"}
            )
            roll_cols = [f"_tb_per_hit_roll{w}" for w in (10, 20)]
            merged = game_team.merge(
                tb_game[["_batting_team_id", "game_pk"] + roll_cols],
                on=["_batting_team_id", "game_pk"], how="left",
            ).set_index("game_pk").reindex(game_frame["game_pk"].values)
            for w in (10, 20):
                new_cols[f"{side}_bat_tb_per_hit_roll{w}"] = (
                    merged[f"_tb_per_hit_roll{w}"].values
                )

        for w in (10, 20):
            h = new_cols.get(f"home_bat_tb_per_hit_roll{w}")
            a = new_cols.get(f"away_bat_tb_per_hit_roll{w}")
            if h is not None and a is not None:
                new_cols[f"diff_bat_tb_per_hit_roll{w}"] = h - a
    else:
        log.warning("Missing event_type — skipping bat TB-per-hit features")

    if not new_cols:
        return pd.DataFrame({"game_pk": game_frame["game_pk"]})

    out = pd.DataFrame(new_cols, index=game_frame.index)
    out["game_pk"] = game_frame["game_pk"].values
    return out.astype({c: "float32" for c in out.columns if c != "game_pk"})


# ---------------------------------------------------------------------------
# Feature 12: Bullpen workload (Group B)
# ---------------------------------------------------------------------------

def _compute_bullpen_workload_features(
    pitches: pd.DataFrame,
    game_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Rolling bullpen workload: RP pitch counts and appearances in prior 3/7 days.

    SP identification: probable_pitcher_{side}_id from game_frame.  Any pitcher
    appearing in a game who is NOT the identified probable starter is a reliever.
    If no SP is identified for a team-game, that game is skipped (conservative).

    Windows use calendar-day sums (not game counts) to handle off-days and
    doubleheaders correctly.  Binary search gives O(n log n) per team.
    B3 (innings) skipped: PITCHES lacks per-pitcher IP; requires BOXSCORE_PITCHING.
    """
    if "is_pitch" not in pitches.columns or "half_inning" not in pitches.columns:
        log.warning("Missing is_pitch or half_inning — skipping bullpen workload features")
        return pd.DataFrame({"game_pk": game_frame["game_pk"]})

    p = pitches[pitches["is_pitch"] == True].copy()  # noqa: E712

    # Pitching team: top half → home team pitches; bottom → away team pitches
    p["_pitching_team_id"] = np.where(
        p["half_inning"] == "top",
        p["home_team_id"],
        p["away_team_id"],
    )

    # Pitch counts per pitcher per team per game
    pitch_counts = (
        p.groupby(["_pitching_team_id", "pitcher_id", "game_pk"])
        .size()
        .reset_index(name="_n_pitches")
    )

    # Build SP lookup from game_frame probable pitcher columns
    sp_parts = []
    for side, pid_col in (("home", "probable_pitcher_home_id"),
                          ("away", "probable_pitcher_away_id")):
        team_col = f"{side}_team_id"
        if pid_col not in game_frame.columns or team_col not in game_frame.columns:
            continue
        sub = game_frame[["game_pk", team_col, pid_col]].dropna(subset=[pid_col]).copy()
        sub = sub.rename(columns={team_col: "_pitching_team_id", pid_col: "_sp_id"})
        sp_parts.append(sub)

    if not sp_parts:
        log.warning("No probable_pitcher columns in game_frame — skipping bullpen features")
        return pd.DataFrame({"game_pk": game_frame["game_pk"]})

    sp_df = pd.concat(sp_parts, ignore_index=True)

    # Mark relievers: pitcher != identified SP (and SP must be known)
    pitch_counts = pitch_counts.merge(sp_df, on=["game_pk", "_pitching_team_id"], how="left")
    pitch_counts["_is_rp"] = (
        pitch_counts["_sp_id"].notna() &
        (pitch_counts["pitcher_id"] != pitch_counts["_sp_id"])
    )

    rp = pitch_counts[pitch_counts["_is_rp"]].copy()

    # Aggregate per (pitching team, game)
    rp_game = rp.groupby(["_pitching_team_id", "game_pk"]).agg(
        _rp_pitches=("_n_pitches", "sum"),
        _rp_appearances=("pitcher_id", "nunique"),
    ).reset_index()

    game_dates = game_frame[["game_pk", "game_date"]].drop_duplicates()
    rp_game = rp_game.merge(game_dates, on="game_pk", how="left")
    rp_game["game_date"] = pd.to_datetime(rp_game["game_date"])
    rp_game = rp_game.sort_values(["_pitching_team_id", "game_date"])

    # Per-team date-window cumulative sums using binary-search (O(n log n))
    def _date_window_sum(dates_arr: np.ndarray,
                         vals_arr: np.ndarray,
                         window_days: int) -> np.ndarray:
        """Sum of vals_arr in prior `window_days` calendar days (exclusive of today)."""
        n = len(dates_arr)
        cumsum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(vals_arr))])
        result = np.empty(n, dtype=float)
        td = np.timedelta64(window_days, "D")
        for i in range(n):
            d = dates_arr[i]
            lo = d - td
            j_lo = int(np.searchsorted(dates_arr, lo, side="left"))
            j_hi = int(np.searchsorted(dates_arr, d, side="left"))  # excludes today
            result[i] = cumsum[j_hi] - cumsum[j_lo]
        return result

    result_parts = []
    for team_id, grp in rp_game.groupby("_pitching_team_id"):
        grp = grp.sort_values("game_date").copy()
        dates_arr = grp["game_date"].values.astype("datetime64[D]")
        grp["_rp_pitches_3d"] = _date_window_sum(dates_arr, grp["_rp_pitches"].values, 3)
        grp["_rp_pitches_7d"] = _date_window_sum(dates_arr, grp["_rp_pitches"].values, 7)
        grp["_rp_apps_3d"]    = _date_window_sum(dates_arr, grp["_rp_appearances"].values, 3)
        result_parts.append(grp)

    if not result_parts:
        return pd.DataFrame({"game_pk": game_frame["game_pk"]})

    rp_windowed = pd.concat(result_parts, ignore_index=True)

    new_cols: dict[str, np.ndarray] = {}
    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        if team_col not in game_frame.columns:
            continue
        game_team = game_frame[["game_pk", team_col]].rename(
            columns={team_col: "_pitching_team_id"}
        )
        win_cols = ["_pitching_team_id", "game_pk",
                    "_rp_pitches_3d", "_rp_pitches_7d", "_rp_apps_3d"]
        merged = game_team.merge(
            rp_windowed[[c for c in win_cols if c in rp_windowed.columns]],
            on=["_pitching_team_id", "game_pk"], how="left",
        ).set_index("game_pk").reindex(game_frame["game_pk"].values)

        if "_rp_pitches_3d" in merged.columns:
            new_cols[f"{side}_bullpen_pitches_last3d"]     = merged["_rp_pitches_3d"].values
        if "_rp_apps_3d" in merged.columns:
            new_cols[f"{side}_bullpen_appearances_last3d"] = merged["_rp_apps_3d"].values
        if "_rp_pitches_7d" in merged.columns:
            new_cols[f"{side}_bullpen_pitches_last7d"]     = merged["_rp_pitches_7d"].values

    if not new_cols:
        return pd.DataFrame({"game_pk": game_frame["game_pk"]})

    out = pd.DataFrame(new_cols, index=game_frame.index)
    out["game_pk"] = game_frame["game_pk"].values
    return out.astype({c: "float32" for c in out.columns if c != "game_pk"})


# ---------------------------------------------------------------------------
# Feature 13: Manager tendencies (Group F)
# ---------------------------------------------------------------------------

def _compute_manager_tendency_features(
    pitches: pd.DataFrame,
    game_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Rolling manager tendency metrics.

    F1: mgr_pitchers_used_roll10 — avg distinct pitchers per game (10-game window).
        Captures bullpen management style.
    F2: mgr_bunt_rate_roll20 — bunt attempts / PA (20-game window).
        Bunt detection: event_type == 'sac_bunt' OR hit_trajectory in
        {bunt_grounder, bunt_popup} (snake_case per parquet schema).
    """
    new_cols: dict[str, np.ndarray] = {}
    game_dates = game_frame[["game_pk", "game_date"]].drop_duplicates()

    # ── F1: Distinct pitchers used ──────────────────────────────────────────
    if "half_inning" in pitches.columns and "pitcher_id" in pitches.columns:
        p = pitches.dropna(subset=["pitcher_id", "game_pk"]).copy()
        p["_pitching_team_id"] = np.where(
            p["half_inning"] == "top",
            p["home_team_id"],
            p["away_team_id"],
        )

        pitchers_used = (
            p.groupby(["_pitching_team_id", "game_pk"])["pitcher_id"]
            .nunique()
            .reset_index(name="_n_pitchers")
        )
        pitchers_used = pitchers_used.merge(game_dates, on="game_pk", how="left")
        pitchers_used = pitchers_used.sort_values(
            ["_pitching_team_id", "game_date", "game_pk"]
        )

        pitchers_used["_n_pitchers_roll10"] = (
            pitchers_used.groupby("_pitching_team_id")["_n_pitchers"]
            .transform(lambda s: s.rolling(10, min_periods=5).mean().shift(1))
        )

        for side in ("home", "away"):
            team_col = f"{side}_team_id"
            if team_col not in game_frame.columns:
                continue
            game_team = game_frame[["game_pk", team_col]].rename(
                columns={team_col: "_pitching_team_id"}
            )
            merged = game_team.merge(
                pitchers_used[["_pitching_team_id", "game_pk", "_n_pitchers_roll10"]],
                on=["_pitching_team_id", "game_pk"], how="left",
            ).set_index("game_pk").reindex(game_frame["game_pk"].values)
            new_cols[f"{side}_mgr_pitchers_used_roll10"] = merged["_n_pitchers_roll10"].values
    else:
        log.warning("Missing half_inning or pitcher_id — skipping pitchers-used feature")

    # ── F2: Bunt rate ───────────────────────────────────────────────────────
    if "half_inning" in pitches.columns:
        # One row per at-bat via dedup on at_bat_index
        pa = pitches.drop_duplicates(subset=["game_pk", "at_bat_index"]).copy()
        if pa.empty:
            log.warning("No PA rows — skipping bunt rate feature")
        else:
            pa["_batting_team_id"] = np.where(
                pa["half_inning"] == "bottom",
                pa["home_team_id"],
                pa["away_team_id"],
            )

            pa["_is_bunt"] = pd.Series(False, index=pa.index)
            if "event_type" in pa.columns:
                pa["_is_bunt"] |= pa["event_type"].isin({"sac_bunt"})
            if "hit_trajectory" in pa.columns:
                pa["_is_bunt"] |= pa["hit_trajectory"].isin({"bunt_grounder", "bunt_popup"})
            pa["_is_bunt"] = pa["_is_bunt"].astype("float32")

            bunt_game = pa.groupby(["_batting_team_id", "game_pk"]).agg(
                _bunt_rate=("_is_bunt", "mean"),
            ).reset_index()
            bunt_game = bunt_game.merge(game_dates, on="game_pk", how="left")
            bunt_game = bunt_game.sort_values(
                ["_batting_team_id", "game_date", "game_pk"]
            )

            bunt_game["_bunt_rate_roll20"] = (
                bunt_game.groupby("_batting_team_id")["_bunt_rate"]
                .transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1))
            )

            for side in ("home", "away"):
                team_col = f"{side}_team_id"
                if team_col not in game_frame.columns:
                    continue
                game_team = game_frame[["game_pk", team_col]].rename(
                    columns={team_col: "_batting_team_id"}
                )
                merged = game_team.merge(
                    bunt_game[["_batting_team_id", "game_pk", "_bunt_rate_roll20"]],
                    on=["_batting_team_id", "game_pk"], how="left",
                ).set_index("game_pk").reindex(game_frame["game_pk"].values)
                new_cols[f"{side}_mgr_bunt_rate_roll20"] = merged["_bunt_rate_roll20"].values
    else:
        log.warning("Missing half_inning — skipping bunt rate feature")

    if not new_cols:
        return pd.DataFrame({"game_pk": game_frame["game_pk"]})

    out = pd.DataFrame(new_cols, index=game_frame.index)
    out["game_pk"] = game_frame["game_pk"].values
    return out.astype({c: "float32" for c in out.columns if c != "game_pk"})
