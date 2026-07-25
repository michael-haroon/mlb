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

    # Join all feature blocks onto game_frame by game_pk.
    result = game_frame.copy()
    for feats in (tto_feats, kbb_feats, fip_feats, woba_feats, pitchmix_feats):
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
        for pt in all_types:
            bwoba_col  = f"_bwoba_{pt}"
            spfreq_col = f"_spfreq_{pt}"
            if bwoba_col in batter_side.columns and spfreq_col in batter_side.columns:
                fill_val = league_avg_woba_by_type.get(pt, 0.320)
                matchup_scores = matchup_scores + (
                    batter_side[bwoba_col].fillna(fill_val) * batter_side[spfreq_col].fillna(0.0)
                )
        batter_side["_matchup_score"] = matchup_scores

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
