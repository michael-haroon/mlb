from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Comparison = Literal["above", "below", "exactly", "at_least", "between"]

TRAINABLE = "trainable"
SETTLES_LAST_FAIR = "settles_last_fair"
NO_APPEARANCE = "no_appearance"

GAME_TARGET_COLUMNS = [
    "home_win",
    "away_win",
    "yrfi",
    "nrfi",
    "extra_innings",
    "total_runs",
    "home_runs",
    "away_runs",
    "home_run_diff",
    "away_run_diff",
    "first_5_total_runs",
    "first_5_home_runs",
    "first_5_away_runs",
    "first_5_home_run_diff",
    "first_5_home_win",
    "first_5_away_win",
    "first_5_tie",
]

PLAYER_BATTING_TARGET_COLUMNS = [
    "game_hits",
    "game_hr",
    "game_hits_runs_rbi",
    "game_total_bases",
]

PLAYER_PITCHING_TARGET_COLUMNS = ["game_so"]


@dataclass(frozen=True)
class MarketLine:
    name: str
    target_column: str
    comparison: Comparison
    count: float
    upper_count: float | None = None


def market_specs() -> list[dict]:
    """Return the market families covered by TARGETS.md.

    Lines with arbitrary exchange thresholds are priced from distribution
    parameters at inference time; fixed binary targets are trained directly.
    """

    return [
        {"family": "moneyline", "side": "home", "target": "home_win", "distribution": "bernoulli"},
        {"family": "moneyline", "side": "away", "target": "away_win", "distribution": "bernoulli"},
        {"family": "spread", "side": "home", "target": "home_run_diff", "distribution": "gaussian"},
        {"family": "spread", "side": "away", "target": "away_run_diff", "distribution": "gaussian"},
        {"family": "total_runs", "side": "game", "target": "total_runs", "distribution": "count"},
        {"family": "team_total", "side": "home", "target": "home_runs", "distribution": "count"},
        {"family": "team_total", "side": "away", "target": "away_runs", "distribution": "count"},
        {"family": "first_5_spread", "side": "home", "target": "first_5_home_run_diff", "distribution": "gaussian"},
        {"family": "first_5_spread", "side": "away", "target": "first_5_away_run_diff", "distribution": "gaussian"},
        {"family": "first_5_total", "side": "game", "target": "first_5_total_runs", "distribution": "count"},
        {"family": "yrfi", "side": "game", "target": "yrfi", "distribution": "bernoulli"},
        {"family": "nrfi", "side": "game", "target": "nrfi", "distribution": "bernoulli"},
        {"family": "first_5_winner", "side": "home", "target": "first_5_home_win", "distribution": "bernoulli"},
        {"family": "first_5_winner", "side": "away", "target": "first_5_away_win", "distribution": "bernoulli"},
        {"family": "first_5_winner", "side": "tie", "target": "first_5_tie", "distribution": "bernoulli"},
        {"family": "extra_innings", "side": "game", "target": "extra_innings", "distribution": "bernoulli"},
        {"family": "player_home_runs", "side": "player", "target": "game_hr", "distribution": "count"},
        {"family": "player_strikeouts", "side": "player", "target": "game_so", "distribution": "count"},
        {"family": "player_hits", "side": "player", "target": "game_hits", "distribution": "count"},
        {"family": "player_hits_runs_rbi", "side": "player", "target": "game_hits_runs_rbi", "distribution": "count"},
        {"family": "player_total_bases", "side": "player", "target": "game_total_bases", "distribution": "count"},
    ]


def build_game_targets(linescore_df, game_meta_df=None):
    """Build game-level market targets from official inning linescores.

    Extra innings are included in full-game totals, matching Kalshi's default
    rule unless a market explicitly names a shorter period.
    """

    import numpy as np
    import pandas as pd

    required = {"game_pk", "season", "inning", "home_runs", "away_runs"}
    missing = required.difference(linescore_df.columns)
    if missing:
        raise ValueError(f"linescore_df missing columns: {sorted(missing)}")

    if linescore_df.empty:
        return pd.DataFrame()

    df = linescore_df.copy()
    for col in ("inning", "home_runs", "away_runs"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    grouped = df.groupby(["game_pk", "season"], sort=False)
    targets = grouped.agg(
        innings_played=("inning", "max"),
        home_runs=("home_runs", "sum"),
        away_runs=("away_runs", "sum"),
    ).reset_index()

    first_5 = _sum_period(df, max_inning=5, prefix="first_5")
    first_1 = _sum_period(df, max_inning=1, prefix="first_1")
    regulation = _sum_period(df, max_inning=9, prefix="regulation")

    targets = targets.merge(first_5, on=["game_pk", "season"], how="left")
    targets = targets.merge(first_1, on=["game_pk", "season"], how="left")
    targets = targets.merge(regulation, on=["game_pk", "season"], how="left")

    for prefix in ("first_5", "first_1", "regulation"):
        for side in ("home", "away"):
            col = f"{prefix}_{side}_runs"
            targets[col] = targets[col].fillna(0).astype(int)
        targets[f"{prefix}_total_runs"] = (
            targets[f"{prefix}_home_runs"] + targets[f"{prefix}_away_runs"]
        )
        targets[f"{prefix}_home_run_diff"] = (
            targets[f"{prefix}_home_runs"] - targets[f"{prefix}_away_runs"]
        )

    targets["total_runs"] = targets["home_runs"] + targets["away_runs"]
    targets["home_team_total_runs"] = targets["home_runs"]
    targets["away_team_total_runs"] = targets["away_runs"]
    targets["home_run_diff"] = targets["home_runs"] - targets["away_runs"]
    targets["away_run_diff"] = -targets["home_run_diff"]
    targets["first_5_away_run_diff"] = -targets["first_5_home_run_diff"]
    targets["home_win"] = (targets["home_run_diff"] > 0).astype("float32")
    targets["away_win"] = (targets["home_run_diff"] < 0).astype("float32")
    targets["extra_innings"] = (targets["innings_played"] > 9).astype("float32")
    targets["yrfi"] = (targets["first_1_total_runs"] > 0).astype("float32")
    targets["nrfi"] = 1.0 - targets["yrfi"]
    targets["first_5_home_win"] = (targets["first_5_home_run_diff"] > 0).astype("float32")
    targets["first_5_away_win"] = (targets["first_5_home_run_diff"] < 0).astype("float32")
    targets["first_5_tie"] = (targets["first_5_home_run_diff"] == 0).astype("float32")
    targets["regulation_tie"] = (
        targets["regulation_home_runs"] == targets["regulation_away_runs"]
    ).astype("float32")

    targets["shortened_or_called"] = (targets["innings_played"] < 9).astype("float32")
    targets["target_status"] = TRAINABLE

    if game_meta_df is not None and not game_meta_df.empty:
        meta_cols = [
            col
            for col in [
                "game_pk",
                "game_date",
                "game_datetime_utc",
                "home_team_id",
                "away_team_id",
                "venue_id",
                "venue_name",
                "day_night",
                "weather_temp",
                "weather_condition",
                "weather_wind",
                "game_type_code",
                "double_header",
                "game_number",
                "probable_pitcher_home_id",
                "probable_pitcher_away_id",
                "umpire_hp",
            ]
            if col in game_meta_df.columns
        ]
        meta = game_meta_df[meta_cols].drop_duplicates("game_pk")
        targets = targets.merge(meta, on="game_pk", how="left")

    numeric_cols = targets.select_dtypes(include=[np.number]).columns
    targets[numeric_cols] = targets[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return targets


def build_player_batting_targets(boxscore_batting_df):
    """Build player batting prop targets.

    Rows with no plate appearance are retained but marked non-trainable because
    scratched/non-starting contracts may settle to last fair price rather than 0.
    """

    import pandas as pd

    if boxscore_batting_df.empty:
        return pd.DataFrame()

    df = boxscore_batting_df.copy()
    int_cols = [
        "game_ab",
        "game_runs",
        "game_hits",
        "game_doubles",
        "game_triples",
        "game_hr",
        "game_rbi",
        "game_bb",
        "game_hbp",
        "game_sac",
        "game_sf",
        "game_so",
    ]
    for col in int_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["game_singles"] = (
        df["game_hits"] - df["game_doubles"] - df["game_triples"] - df["game_hr"]
    ).clip(lower=0)
    df["game_total_bases"] = (
        df["game_singles"]
        + 2 * df["game_doubles"]
        + 3 * df["game_triples"]
        + 4 * df["game_hr"]
    )
    df["game_hits_runs_rbi"] = df["game_hits"] + df["game_runs"] + df["game_rbi"]
    df["plate_appearances_est"] = (
        df["game_ab"] + df["game_bb"] + df["game_hbp"] + df["game_sac"] + df["game_sf"]
    )
    df["target_status"] = TRAINABLE
    if "is_substitute" in df.columns:
        df.loc[df["is_substitute"].fillna(False).astype(bool), "target_status"] = SETTLES_LAST_FAIR
    df.loc[df["plate_appearances_est"] <= 0, "target_status"] = NO_APPEARANCE

    keep = [
        "game_pk",
        "season",
        "player_id",
        "player_name",
        "side",
        "batting_order",
        "is_substitute",
        "plate_appearances_est",
        "game_hits",
        "game_hr",
        "game_runs",
        "game_rbi",
        "game_so",
        "game_total_bases",
        "game_hits_runs_rbi",
        "target_status",
    ]
    return df[[col for col in keep if col in df.columns]]


def build_player_pitching_targets(boxscore_pitching_df):
    """Build pitcher prop targets."""

    import pandas as pd

    if boxscore_pitching_df.empty:
        return pd.DataFrame()

    df = boxscore_pitching_df.copy()
    for col in ("game_so", "game_pitches_thrown", "game_innings_pitched"):
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["target_status"] = TRAINABLE
    if "is_starter" in df.columns:
        df.loc[~df["is_starter"].fillna(False).astype(bool), "target_status"] = SETTLES_LAST_FAIR
    df.loc[
        (df["game_pitches_thrown"] <= 0) & (df["game_innings_pitched"] <= 0),
        "target_status",
    ] = NO_APPEARANCE

    keep = [
        "game_pk",
        "season",
        "player_id",
        "player_name",
        "side",
        "is_starter",
        "game_innings_pitched",
        "game_pitches_thrown",
        "game_so",
        "target_status",
    ]
    return df[[col for col in keep if col in df.columns]]


def resolve_market_line(values, line: MarketLine):
    """Resolve a target column against a Kalshi-style comparison operator."""

    lower = line.count
    if line.comparison == "above":
        return values[line.target_column] > lower
    if line.comparison == "below":
        return values[line.target_column] < lower
    if line.comparison == "exactly":
        return values[line.target_column] == lower
    if line.comparison == "at_least":
        return values[line.target_column] >= lower
    if line.comparison == "between":
        if line.upper_count is None:
            raise ValueError("between comparison requires upper_count")
        return (values[line.target_column] >= lower) & (
            values[line.target_column] <= line.upper_count
        )
    raise ValueError(f"Unsupported comparison: {line.comparison}")


def _sum_period(linescore_df, max_inning: int, prefix: str):
    period = linescore_df[linescore_df["inning"] <= max_inning]
    out = period.groupby(["game_pk", "season"], sort=False).agg(
        **{
            f"{prefix}_home_runs": ("home_runs", "sum"),
            f"{prefix}_away_runs": ("away_runs", "sum"),
        }
    )
    return out.reset_index()
