"""Massey rating system for MLB using per-inning score differentials.

The core system is X β = y, where y is the score margin (home - away) after
a given inning (or end of game). Each row of X has +1 for the home team, -1
for the away team, plus an optional home-advantage column.

Produces 10 rating variants per team per season snapshot:
  - massey_inn1 through massey_inn9: margin through each inning
  - massey_full: final score margin (including extras)

Temporal safety: fit_massey_inning and build_pregame_massey_features use only
games strictly prior to the target date — no lookahead.

Reference: Massey (1997), "Statistical Models Applied to the Rating of Sports
Teams", Bluefield College.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

INNINGS = tuple(range(1, 10))  # 1 through 9
MASSEY_TARGETS = (*[f"inn{i}" for i in INNINGS], "full")


@dataclass(frozen=True)
class MasseyDesign:
    """A single Massey permutation for MLB."""

    name: str
    include_home_advantage: bool = True
    min_games: int = 20

    @property
    def rating_column(self) -> str:
        return self.name

    @property
    def rank_column(self) -> str:
        return f"{self.name}_rank"


@dataclass
class MasseyFit:
    """Solved ratings plus diagnostics for one design/season/target."""

    design: MasseyDesign
    season: int | str
    as_of_date: pd.Timestamp | None
    target: str
    teams: list[int]
    ratings: pd.DataFrame
    coefficients: dict[str, float]
    n_games: int
    solver: str
    rank: int
    components: list[list[int]]
    warnings: list[str] = field(default_factory=list)


DEFAULT_DESIGNS: tuple[MasseyDesign, ...] = (
    MasseyDesign("massey"),
    MasseyDesign("massey_no_ha", include_home_advantage=False),
)


def _team_components(games: pd.DataFrame, teams: list[int]) -> list[list[int]]:
    """Find connected components in the schedule graph via BFS."""
    graph: dict[int, set[int]] = {t: set() for t in teams}
    for row in games[["home_team_id", "away_team_id"]].itertuples(index=False):
        h, a = int(row.home_team_id), int(row.away_team_id)
        graph.setdefault(h, set()).add(a)
        graph.setdefault(a, set()).add(h)

    seen: set[int] = set()
    components: list[list[int]] = []
    for team in teams:
        if team in seen:
            continue
        queue: deque[int] = deque([team])
        seen.add(team)
        comp: list[int] = []
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nbr in graph.get(cur, ()):
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append(nbr)
        components.append(sorted(comp))
    return components


def prepare_linescore_cumulative(
    linescore: pd.DataFrame,
    game_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Pivot linescore rows into one row per game with cumulative margins.

    Returns a DataFrame with columns:
        game_pk, season, game_date, home_team_id, away_team_id,
        margin_inn1, margin_inn2, ..., margin_inn9, margin_full
    where margin_innN is the cumulative (home - away) runs through inning N.
    """
    required_ls = {"game_pk", "inning", "home_runs", "away_runs"}
    missing_ls = required_ls - set(linescore.columns)
    if missing_ls:
        raise ValueError(f"linescore missing columns: {sorted(missing_ls)}")

    required_meta = {"game_pk", "game_date", "home_team_id", "away_team_id"}
    missing_meta = required_meta - set(game_meta.columns)
    if missing_meta:
        raise ValueError(f"game_meta missing columns: {sorted(missing_meta)}")

    ls = linescore.copy()
    ls["inning"] = pd.to_numeric(ls["inning"], errors="coerce")
    ls["home_runs"] = pd.to_numeric(ls["home_runs"], errors="coerce").fillna(0)
    ls["away_runs"] = pd.to_numeric(ls["away_runs"], errors="coerce").fillna(0)
    ls = ls.dropna(subset=["inning"])
    ls["inning"] = ls["inning"].astype(int)

    # Deduplicate: batch files can overlap
    ls = ls.drop_duplicates(subset=["game_pk", "inning"])

    ls = ls.sort_values(["game_pk", "inning"])
    ls["margin"] = ls["home_runs"] - ls["away_runs"]

    # Cumulative margin per game
    ls["cum_margin"] = ls.groupby("game_pk")["margin"].cumsum()

    # Pivot: one column per inning (1-9), plus full game
    result_parts = []

    for inn in INNINGS:
        inn_data = ls[ls["inning"] == inn][["game_pk", "cum_margin"]].copy()
        inn_data = inn_data.rename(columns={"cum_margin": f"margin_inn{inn}"})
        result_parts.append(inn_data)

    # Full game margin: max inning's cumulative value
    full = ls.groupby("game_pk")["cum_margin"].last().reset_index()
    full = full.rename(columns={"cum_margin": "margin_full"})
    result_parts.append(full)

    # Merge all inning margins
    out = result_parts[0]
    for part in result_parts[1:]:
        out = out.merge(part, on="game_pk", how="outer")

    # Attach metadata
    meta = game_meta[["game_pk", "season", "game_date", "home_team_id", "away_team_id"]].copy()
    meta = meta.drop_duplicates("game_pk")
    meta["game_date"] = pd.to_datetime(meta["game_date"], errors="coerce")
    out = out.merge(meta, on="game_pk", how="inner")

    out = out.dropna(subset=["game_date", "home_team_id", "away_team_id"])
    out["home_team_id"] = out["home_team_id"].astype(int)
    out["away_team_id"] = out["away_team_id"].astype(int)

    return out.sort_values(["season", "game_date", "game_pk"]).reset_index(drop=True)


def fit_massey_inning(
    games: pd.DataFrame,
    target: str,
    design: MasseyDesign = DEFAULT_DESIGNS[0],
    *,
    season: int | str | None = None,
    as_of_date: pd.Timestamp | None = None,
) -> MasseyFit:
    """Fit one Massey design for one inning target (or full game).

    Parameters
    ----------
    games : pd.DataFrame
        Output of prepare_linescore_cumulative(). Must have margin_* columns.
    target : str
        One of 'inn1', 'inn2', ..., 'inn9', 'full'.
    design : MasseyDesign
        Controls home-advantage inclusion and minimum games.
    season : int, optional
        Restrict to this season.
    as_of_date : pd.Timestamp, optional
        Use only games on or before this date.
    """
    margin_col = f"margin_{target}"
    if margin_col not in games.columns:
        raise ValueError(f"Column {margin_col} not found. Available: {[c for c in games.columns if 'margin' in c]}")

    df = games.copy()
    if season is not None:
        df = df[df["season"] == season]
    if as_of_date is not None:
        df = df[df["game_date"] <= as_of_date]

    df = df.dropna(subset=[margin_col, "home_team_id", "away_team_id"])
    rating_name = f"{design.name}_{target}"

    if len(df) < design.min_games:
        empty = pd.DataFrame(columns=["season", "team_id", rating_name, f"{rating_name}_rank"])
        return MasseyFit(
            design=design, season=season, as_of_date=as_of_date,
            target=target, teams=[], ratings=empty, coefficients={},
            n_games=len(df), solver="empty", rank=0, components=[],
            warnings=[f"Only {len(df)} games < min_games={design.min_games}"],
        )

    teams = sorted(
        set(df["home_team_id"].astype(int).tolist())
        | set(df["away_team_id"].astype(int).tolist())
    )
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    # Build design matrix columns
    extra_columns: list[str] = []
    if design.include_home_advantage:
        extra_columns.append("home_advantage")

    n_cols = n_teams + len(extra_columns)
    n_games = len(df)

    # Vectorized normal equation assembly
    home_ids = df["home_team_id"].map(team_to_idx).values.astype(int)
    away_ids = df["away_team_id"].map(team_to_idx).values.astype(int)
    y = df[margin_col].values.astype(float)

    X = np.zeros((n_games, n_cols), dtype=float)
    X[np.arange(n_games), home_ids] = 1.0
    X[np.arange(n_games), away_ids] = -1.0

    if design.include_home_advantage:
        X[:, n_teams] = 1.0

    # Normal equations: M = X.T X, p = X.T y
    normal = X.T @ X
    target_vec = X.T @ y

    # Apply sum-to-zero constraint per connected component
    constrained = normal.copy()
    constrained_target = target_vec.copy()
    components = _team_components(df, teams)

    if len(components) > 1:
        log.info("[massey] %s/%s: %d disconnected components — constraining each",
                 design.name, target, len(components))

    for comp in components:
        row_idx = team_to_idx[comp[-1]]
        constrained[row_idx, :] = 0.0
        constrained[row_idx, [team_to_idx[t] for t in comp]] = 1.0
        constrained_target[row_idx] = 0.0

    # Condition number check + ridge if ill-conditioned
    cond = float(np.linalg.cond(constrained)) if n_cols > 0 else float("inf")
    if cond > 1e12:
        lambda_reg = 0.01
        np.fill_diagonal(constrained, constrained.diagonal() + lambda_reg)
        log.info("[massey] %s/%s: applied ridge λ=%.4f (cond was %.2e)",
                 design.name, target, lambda_reg, cond)

    # Solve
    warnings: list[str] = []
    try:
        beta = np.linalg.solve(constrained, constrained_target)
        solver = "numpy.linalg.solve"
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(constrained, constrained_target, rcond=None)
        solver = "numpy.linalg.lstsq"
        warnings.append("Singular after constraints; used least-squares fallback.")

    matrix_rank = int(np.linalg.matrix_rank(normal)) if n_cols > 0 else 0

    log.debug("[massey] %s/%s season=%s games=%d teams=%d solver=%s range=[%.3f, %.3f]",
              design.name, target, season, n_games, n_teams, solver,
              float(beta[:n_teams].min()), float(beta[:n_teams].max()))

    # Build output
    columns = [f"team_{t}" for t in teams] + extra_columns
    coefficients = {columns[i]: float(beta[i]) for i in range(n_cols)}

    rating_values = beta[:n_teams]
    ratings = pd.DataFrame({
        "season": season if season is not None else df["season"].iloc[0],
        "team_id": teams,
        rating_name: rating_values,
    })
    ratings[f"{rating_name}_rank"] = (
        ratings[rating_name].rank(ascending=False, method="min").astype(int)
    )

    return MasseyFit(
        design=design,
        season=season if season is not None else df["season"].iloc[0],
        as_of_date=as_of_date,
        target=target,
        teams=teams,
        ratings=ratings,
        coefficients=coefficients,
        n_games=n_games,
        solver=solver,
        rank=matrix_rank,
        components=components,
        warnings=warnings,
    )


def build_massey_season_ratings(
    games: pd.DataFrame,
    *,
    designs: Iterable[MasseyDesign] = (DEFAULT_DESIGNS[0],),
    targets: Iterable[str] = MASSEY_TARGETS,
    as_of_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[tuple, MasseyFit]]:
    """Fit all Massey designs × targets for each season.

    Returns:
        ratings_df: one row per (season, team) with columns for each rating variant.
        fits: diagnostic objects keyed by (season, design_name, target).
    """
    targets = list(targets)
    designs = list(designs)
    frames: list[pd.DataFrame] = []
    fits: dict[tuple, MasseyFit] = {}

    for season, season_games in games.groupby("season", sort=True):
        season_ratings: pd.DataFrame | None = None

        for design in designs:
            for target in targets:
                fit = fit_massey_inning(
                    season_games, target, design,
                    season=season, as_of_date=as_of_date,
                )
                fits[(season, design.name, target)] = fit

                if fit.ratings.empty:
                    continue

                if season_ratings is None:
                    season_ratings = fit.ratings
                else:
                    season_ratings = season_ratings.merge(
                        fit.ratings, on=["season", "team_id"], how="outer",
                    )

        if season_ratings is not None:
            frames.append(season_ratings)

    ratings_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return ratings_df, fits


def build_pregame_massey_features(
    games: pd.DataFrame,
    *,
    designs: Iterable[MasseyDesign] = (DEFAULT_DESIGNS[0],),
    targets: Iterable[str] = MASSEY_TARGETS,
    min_prior_games: int = 30,
    refit_interval: int = 1,
) -> pd.DataFrame:
    """Build temporally-safe Massey diff features for each game.

    For every game date, ratings are fit using ONLY games from earlier dates
    in the same season. The output diff is (home_rating - away_rating) for
    each target/design combination.

    Parameters
    ----------
    games : pd.DataFrame
        Output of prepare_linescore_cumulative().
    designs : iterable of MasseyDesign
        Rating variants to compute.
    targets : iterable of str
        Which margin targets to solve for.
    min_prior_games : int
        Skip until this many prior games exist in the season.
    refit_interval : int
        Refit every N game-dates (1 = daily, reduces cost if set higher).
    """
    targets = list(targets)
    designs = list(designs)
    all_rows: list[dict] = []

    for season, season_games in games.groupby("season", sort=True):
        season_games = season_games.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
        dates = season_games["game_date"].unique()
        dates = np.sort(dates)

        cached_ratings: dict[str, pd.DataFrame] | None = None
        last_fit_idx = -refit_interval  # force fit on first eligible date

        for date_idx, game_date in enumerate(dates):
            prior = season_games[season_games["game_date"] < game_date]

            if len(prior) < min_prior_games:
                # Not enough history yet — emit rows with NaN diffs
                today = season_games[season_games["game_date"] == game_date]
                for _, game in today.iterrows():
                    row = _base_row(game, season)
                    all_rows.append(row)
                continue

            # Refit if interval elapsed
            if (date_idx - last_fit_idx) >= refit_interval or cached_ratings is None:
                cached_ratings = {}
                for design in designs:
                    for target in targets:
                        fit = fit_massey_inning(
                            prior, target, design, season=season,
                        )
                        if not fit.ratings.empty:
                            key = f"{design.name}_{target}"
                            cached_ratings[key] = fit.ratings.set_index("team_id")[fit.ratings.columns[2]]
                last_fit_idx = date_idx

            # Emit features for today's games
            today = season_games[season_games["game_date"] == game_date]
            for _, game in today.iterrows():
                row = _base_row(game, season)

                home_id = int(game["home_team_id"])
                away_id = int(game["away_team_id"])

                if cached_ratings:
                    for key, ratings_series in cached_ratings.items():
                        try:
                            h_rating = float(ratings_series.loc[home_id])
                            a_rating = float(ratings_series.loc[away_id])
                            row[f"diff_{key}"] = h_rating - a_rating
                        except (KeyError, TypeError):
                            row[f"diff_{key}"] = np.nan

                all_rows.append(row)

    result = pd.DataFrame(all_rows)
    if not result.empty:
        result = result.sort_values(["season", "game_date", "game_pk"]).reset_index(drop=True)

    n_diff = len([c for c in result.columns if c.startswith("diff_")])
    log.info("[massey] Built %d games × %d diff features", len(result), n_diff)
    return result


def _base_row(game: pd.Series, season) -> dict:
    """Extract the identifying columns for one game row."""
    return {
        "game_pk": game["game_pk"],
        "season": season,
        "game_date": game["game_date"],
        "home_team_id": int(game["home_team_id"]),
        "away_team_id": int(game["away_team_id"]),
    }


def attach_massey_ratings(
    game_features: pd.DataFrame,
    massey_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Massey diff features onto the game features frame.

    Joins on game_pk. Only diff_* columns are attached to avoid duplicating
    metadata columns.
    """
    diff_cols = [c for c in massey_features.columns if c.startswith("diff_")]
    if not diff_cols:
        log.warning("[massey] No diff columns found in massey_features")
        return game_features

    merge_cols = ["game_pk"] + diff_cols
    return game_features.merge(
        massey_features[merge_cols], on="game_pk", how="left",
    )
