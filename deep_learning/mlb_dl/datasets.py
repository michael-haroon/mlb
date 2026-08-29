from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from typing import Iterable


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
    "first_5_away_run_diff",
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

IDENTITY_COLUMNS = {
    "game_pk",
    "season",
    "side",
    "game_date",
    "player_id",
    "batter_id",
    "pitcher_id",
    "team_id",
    "opponent_team_id",
    "home_team_id",
    "away_team_id",
    "venue_id",
    "sequence_index",
    "play_index",
    "at_bat_index",
    "pitch_sequence_index",
}


@dataclass
class SequenceSpec:
    history_length: int = 20
    min_history: int = 5
    # --- game-index-based dual decay ---
    # λ_intra: decay rate within a season (per game played by the team).
    # A value of 0.015 gives half-weight after ~46 games (~28% of a season),
    # capturing post-trade-deadline momentum shifts without wiping out
    # well-represented early-season games.
    intra_season_lambda: float = 0.015
    # λ_inter: decay rate applied once per full season crossed.
    # A value of 0.30 gives the prior season ~74% relative weight vs the
    # current season at game 1, declining to ~55% by mid-season.
    inter_season_lambda: float = 0.30
    # Legacy calendar-day lambda kept for backward compatibility with
    # any external code that reads this field; not used internally.
    time_decay_lambda: float = 0.003
    live_stride: int = 25
    live_max_prefixes_per_game: int = 32
    hash_bucket_count: int = 50000


@dataclass
class Standardizer:
    feature_columns: list[str]
    mean: dict[str, float]
    std: dict[str, float]

    @classmethod
    def fit(cls, frame, feature_columns: Iterable[str]):
        import numpy as np

        cols = list(feature_columns)
        means = {}
        stds = {}
        for col in cols:
            values = frame[col].astype("float32").to_numpy()
            finite = values[np.isfinite(values)]
            means[col] = float(np.mean(finite)) if finite.size else 0.0
            std = float(np.std(finite)) if finite.size else 1.0
            stds[col] = std if std > 1e-6 else 1.0
        return cls(feature_columns=cols, mean=means, std=stds)

    def transform(self, frame):
        import numpy as np

        values = frame[self.feature_columns].astype("float32").to_numpy(copy=True)
        mask = np.isfinite(values).astype("float32")
        for idx, col in enumerate(self.feature_columns):
            values[:, idx] = (values[:, idx] - self.mean[col]) / self.std[col]
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return values.astype("float32"), mask.astype("float32")

    def to_dict(self) -> dict:
        return {
            "feature_columns": self.feature_columns,
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            feature_columns=list(data["feature_columns"]),
            mean={str(k): float(v) for k, v in data["mean"].items()},
            std={str(k): float(v) for k, v in data["std"].items()},
        )


# ---------------------------------------------------------------------------
# Flat game-level dataset (no sequence history)
# ---------------------------------------------------------------------------

class FlatGameDataset:
    """Dataset for flat game-level features (one row per game).

    Used with PregameFlatModel when features already encode temporal patterns
    (e.g. rolling/EWMA features from the classical feature store).
    """

    def __init__(
        self,
        game_features,
        feature_columns: list[str],
        standardizer: Standardizer,
        split_start=None,
        split_end=None,
    ):
        import numpy as np
        import pandas as pd
        import torch

        df = game_features.copy()
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

        # Filter trainable
        if "target_status" in df.columns:
            df = df[df["target_status"].eq("trainable")]

        # Temporal split
        if split_start is not None:
            df = df[df["game_date"] >= pd.Timestamp(split_start)]
        if split_end is not None:
            df = df[df["game_date"] < pd.Timestamp(split_end)]

        # Require all targets present
        target_cols = ["home_win", "yrfi", "extra_innings", "total_runs", "home_run_diff"]
        df = df.dropna(subset=[c for c in target_cols if c in df.columns])

        # Standardize features
        values, mask = standardizer.transform(df)
        # Combine: features = values * mask (NaN positions stay 0)
        self._features = torch.from_numpy(values * mask)

        # Targets
        self._targets = {
            "home_win": torch.from_numpy(df["home_win"].astype("float32").to_numpy().copy()),
            "yrfi": torch.from_numpy(df["yrfi"].astype("float32").to_numpy().copy()),
            "extra_innings": torch.from_numpy(df["extra_innings"].astype("float32").to_numpy().copy()) if "extra_innings" in df.columns else torch.zeros(len(df)),
            "total_runs": torch.from_numpy(df["total_runs"].astype("float32").to_numpy().copy()),
            "home_run_diff": torch.from_numpy(df["home_run_diff"].astype("float32").to_numpy().copy()),
        }

        # Sample weights (game-index decay from most recent game)
        max_date = df["game_date"].max()
        age_days = (max_date - df["game_date"]).dt.days.clip(lower=0).to_numpy()
        self._weights = torch.from_numpy(np.exp(-0.003 * age_days).astype("float32"))

        self._game_pks = df["game_pk"].to_numpy() if "game_pk" in df.columns else np.arange(len(df))
        self._n = len(df)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": self._features[idx],
            "targets": {k: v[idx] for k, v in self._targets.items()},
            "sample_weight": self._weights[idx],
        }


# ---------------------------------------------------------------------------
# Game-index-based decay
# ---------------------------------------------------------------------------

def build_team_game_index(team_games) -> dict[int, dict]:
    """Build a per-team sequential game index for decay weight computation.

    Returns a dict keyed by team_id.  Each value is a dict:
        {
            "dates":   sorted list[pd.Timestamp],
            "seasons": list[int],            # parallel to dates
            "indices": list[int],            # monotone global game index
        }

    The global index is simply the zero-based rank of each game across all
    seasons for that team.  Season boundaries carry no special penalty: game 162
    of season N has index k, and game 1 of season N+1 has index k+1.  The decay
    weight for a history row is then:

        w = exp(-λ_intra * (current_index - row_index))

    multiplied by a cross-season penalty applied once per season boundary
    crossed:

        w *= exp(-λ_inter * seasons_crossed)

    where seasons_crossed = number of distinct season increments between
    row_index and current_index.
    """
    import pandas as pd

    by_team: dict[int, dict] = {}
    if team_games.empty or "team_id" not in team_games.columns:
        return by_team

    tg = team_games.dropna(subset=["team_id", "game_date"]).copy()
    tg["game_date"] = pd.to_datetime(tg["game_date"], errors="coerce")
    tg = tg.dropna(subset=["game_date"])

    for team_id, grp in tg.groupby("team_id", sort=False):
        grp_sorted = grp.sort_values("game_date").drop_duplicates("game_pk")
        dates = grp_sorted["game_date"].tolist()
        seasons = (
            grp_sorted["season"].tolist()
            if "season" in grp_sorted.columns
            else [None] * len(dates)
        )
        by_team[int(team_id)] = {
            "dates": dates,
            "seasons": seasons,
            "indices": list(range(len(dates))),
        }
    return by_team


def compute_game_decay_weight(
    team_entry: dict,
    current_date,
    spec: SequenceSpec,
) -> list[float]:
    """Return a decay weight for every game in team_entry that precedes current_date.

    Weight for game at global index i when the current game is at index N:

        Δ = N - i   (games elapsed)
        S = seasons crossed between game i and game N
        w = exp(-λ_intra * Δ) * exp(-λ_inter * S)

    Games from a prior season are penalised once per season boundary, not once
    per 365 days.  This eliminates the winter penalty from the original
    calendar-day implementation while still down-weighting stale franchise
    context.
    """
    import math

    dates = team_entry["dates"]
    seasons = team_entry["seasons"]

    # Find the index of the last game strictly before current_date.
    current_idx = 0
    for k, d in enumerate(dates):
        if d < current_date:
            current_idx = k + 1
        else:
            break

    if current_idx == 0:
        return []

    # The current game's "season" is the season of the most recent prior game
    # (a safe proxy when the target game season equals the last prior game season
    # or N+1).
    current_season = seasons[current_idx - 1]

    weights = []
    for i in range(current_idx):
        delta = current_idx - 1 - i  # games elapsed (0 = most recent prior game)
        row_season = seasons[i]
        if current_season is not None and row_season is not None:
            seasons_crossed = max(int(current_season) - int(row_season), 0)
        else:
            seasons_crossed = 0
        w = math.exp(-spec.intra_season_lambda * delta) * math.exp(-spec.inter_season_lambda * seasons_crossed)
        weights.append(w)
    return weights


class PregameSequenceDataset:
    """PyTorch dataset that yields home/away prior-game sequences.

    Sample weights are computed from sequential game indices rather than
    calendar days, eliminating the offseason winter penalty that
    artificially decays prior-season games by ~5 months of dead time.
    """

    def __init__(
        self,
        team_games,
        game_targets,
        standardizer: Standardizer,
        spec: SequenceSpec,
        split_start=None,
        split_end=None,
    ):
        import pandas as pd
        from torch.utils.data import Dataset

        if not isinstance(self, Dataset):
            pass

        self.team_games = team_games.copy()
        self.game_targets = game_targets.copy()
        self.standardizer = standardizer
        self.spec = spec

        self.team_games["game_date"] = pd.to_datetime(self.team_games["game_date"], errors="coerce")
        self.game_targets["game_date"] = pd.to_datetime(
            self.game_targets["game_date"], errors="coerce"
        )

        samples = self.game_targets[self.game_targets["target_status"].eq("trainable")].copy()
        samples = samples.dropna(subset=["game_date", "home_team_id", "away_team_id"])
        for col in GAME_TARGET_COLUMNS:
            samples = samples.dropna(subset=[col])

        if split_start is not None:
            samples = samples[samples["game_date"] >= pd.Timestamp(split_start)]
        if split_end is not None:
            samples = samples[samples["game_date"] < pd.Timestamp(split_end)]

        # Build per-team sorted history and sequential game index for decay.
        self.by_team = {
            int(team_id): group.sort_values("game_date").reset_index(drop=True)
            for team_id, group in self.team_games.dropna(subset=["team_id"]).groupby("team_id")
        }
        self.team_game_index = build_team_game_index(self.team_games)

        # Precompute standardized arrays per team so __getitem__ only needs
        # a searchsorted + slice instead of a pandas boolean scan + transform.
        import numpy as np
        self.team_arrays: dict[int, tuple] = {}  # team_id -> (dates_ns, values, mask)
        for team_id, df in self.by_team.items():
            vals, msk = self.standardizer.transform(df)
            self.team_arrays[team_id] = (
                df["game_date"].to_numpy(dtype="datetime64[ns]"),
                vals,   # (n_games, feature_dim) float32
                msk,    # (n_games, feature_dim) float32
            )

        # Build a lookup table: (team_id, game_date) → number of prior games for that team.
        # cumcount() on the sorted-by-date group gives 0,1,2,... so the value at row k is
        # exactly the count of games strictly before that date (i.e. available history).
        team_history_lut = (
            self.team_games.dropna(subset=["team_id", "game_date"])[["team_id", "game_date"]]
            .assign(team_id=lambda df: df["team_id"].astype(int))
            .sort_values(["team_id", "game_date"])
            .assign(prior_count=lambda df: df.groupby("team_id").cumcount())
        )

        samples_reset = samples.reset_index(drop=True).copy()
        samples_reset["home_team_id"] = samples_reset["home_team_id"].astype(int)
        samples_reset["away_team_id"] = samples_reset["away_team_id"].astype(int)

        # merge_asof on home team: for each sample game_date find the largest lut game_date
        # that is strictly less (direction='backward'), giving the prior_count at that point.
        home_lut = team_history_lut.rename(columns={"team_id": "home_team_id", "prior_count": "_home_prior"})
        away_lut = team_history_lut.rename(columns={"team_id": "away_team_id", "prior_count": "_away_prior"})

        merged = pd.merge_asof(
            samples_reset.sort_values("game_date"),
            home_lut.sort_values("game_date"),
            on="game_date",
            by="home_team_id",
            direction="backward",
        )
        merged = pd.merge_asof(
            merged,
            away_lut.sort_values("game_date"),
            on="game_date",
            by="away_team_id",
            direction="backward",
        )

        # prior_count at the matched row = games up to and including that date;
        # add 1 to convert "index of last game seen" to count of games before sample date.
        merged["_home_prior"] = merged["_home_prior"].fillna(-1).astype(int) + 1
        merged["_away_prior"] = merged["_away_prior"].fillna(-1).astype(int) + 1

        valid_mask = (merged["_home_prior"] >= spec.min_history) & (merged["_away_prior"] >= spec.min_history)
        self.samples = (
            merged.loc[valid_mask]
            .drop(columns=["_home_prior", "_away_prior"])
            .sort_index()
            .reset_index(drop=True)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        import numpy as np
        import torch

        row = self.samples.iloc[idx]
        home_values, home_mask, home_pad = self._team_sequence(row["home_team_id"], row["game_date"])
        away_values, away_mask, away_pad = self._team_sequence(row["away_team_id"], row["game_date"])

        # Sample weight: average of the most-recent game weight for home and away.
        # The most recent prior game always has Δ=0 so its weight is dominated by
        # the inter-season term; a mid-season target with only intra-season history
        # gets weight ≈ 1.0 for the last game, decaying back through the sequence.
        home_w = _last_prior_game_weight(self.team_game_index, row["home_team_id"], row["game_date"], self.spec)
        away_w = _last_prior_game_weight(self.team_game_index, row["away_team_id"], row["game_date"], self.spec)
        sample_weight = float((home_w + away_w) / 2.0)

        return {
            "home_values": torch.from_numpy(home_values),
            "home_mask": torch.from_numpy(home_mask),
            "home_padding": torch.from_numpy(home_pad),
            "away_values": torch.from_numpy(away_values),
            "away_mask": torch.from_numpy(away_mask),
            "away_padding": torch.from_numpy(away_pad),
            "targets": {
                col: torch.tensor(float(row[col]), dtype=torch.float32)
                for col in GAME_TARGET_COLUMNS
                if col in row.index
            },
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
            "game_pk": torch.tensor(int(row["game_pk"]), dtype=torch.long),
        }

    def _team_sequence(self, team_id, game_date):
        import numpy as np

        feature_dim = len(self.standardizer.feature_columns)
        entry = self.team_arrays.get(int(team_id))
        if entry is None:
            n_prior = 0
            values = np.zeros((0, feature_dim), dtype="float32")
            mask = np.zeros((0, feature_dim), dtype="float32")
        else:
            dates_ns, all_vals, all_mask = entry
            cut = int(np.searchsorted(dates_ns, np.datetime64(game_date, "ns"), side="left"))
            start = max(cut - self.spec.history_length, 0)
            values = all_vals[start:cut]
            mask = all_mask[start:cut]
            n_prior = len(values)

        return _left_pad(values, mask, self.spec.history_length, feature_dim)


def _last_prior_game_weight(
    team_game_index: dict[int, dict],
    team_id,
    current_date,
    spec: SequenceSpec,
) -> float:
    """Return the decay weight of the single most recent prior game.

    Used as the per-sample weight passed to the training loss.  The most recent
    prior game always has Δ=0, so its weight is entirely determined by how many
    season boundaries separate it from the target game's context season:

        w = exp(-λ_inter * seasons_crossed_at_last_game)

    A same-season last game returns 1.0.  A game from the prior season returns
    exp(-λ_inter).
    """
    import math

    entry = team_game_index.get(int(team_id) if team_id is not None else -1)
    if entry is None:
        return 1.0

    dates = entry["dates"]
    seasons = entry["seasons"]

    last_idx = -1
    for k, d in enumerate(dates):
        if d < current_date:
            last_idx = k
        else:
            break

    if last_idx < 0:
        return 1.0

    current_season = seasons[last_idx]
    seasons_crossed = 0
    # Look back to find the first game in the history window (up to history_length)
    # and measure the season span — this is the weight for that anchor game.
    # For the sample_weight we only care about the last game (Δ=0).
    row_season = seasons[last_idx]
    if current_season is not None and row_season is not None:
        seasons_crossed = max(int(current_season) - int(row_season), 0)

    return math.exp(-spec.inter_season_lambda * seasons_crossed)


def infer_feature_columns(team_games) -> list[str]:
    numeric = team_games.select_dtypes(include=["number", "bool"]).columns
    return [col for col in numeric if col not in IDENTITY_COLUMNS]


def infer_player_feature_columns(player_history) -> list[str]:
    numeric = player_history.select_dtypes(include=["number", "bool"]).columns
    excluded = IDENTITY_COLUMNS | set(PLAYER_BATTING_TARGET_COLUMNS) | set(PLAYER_PITCHING_TARGET_COLUMNS)
    return [col for col in numeric if col not in excluded]


def infer_live_feature_columns(pitch_sequences) -> list[str]:
    numeric = pitch_sequences.select_dtypes(include=["number", "bool"]).columns
    excluded = IDENTITY_COLUMNS | {
        "home_team_id",
        "away_team_id",
        "pre_on_first_id",
        "pre_on_second_id",
        "pre_on_third_id",
        "post_on_first_id",
        "post_on_second_id",
        "post_on_third_id",
    }
    return [col for col in numeric if col not in excluded]


class PregamePlayerDataset:
    """Prior player-game history dataset for pregame player props."""

    def __init__(
        self,
        player_history,
        target_columns: list[str],
        standardizer: Standardizer,
        spec: SequenceSpec,
        split_start=None,
        split_end=None,
    ):
        import pandas as pd

        self.player_history = player_history.copy()
        self.player_history["game_date"] = pd.to_datetime(
            self.player_history["game_date"], errors="coerce"
        )
        self.target_columns = target_columns
        self.standardizer = standardizer
        self.spec = spec
        self.max_date = self.player_history["game_date"].max()

        samples = self.player_history[self.player_history["target_status"].eq("trainable")].copy()
        samples = samples.dropna(subset=["game_date", "player_id"])
        for col in target_columns:
            samples = samples.dropna(subset=[col])
        if split_start is not None:
            samples = samples[samples["game_date"] >= pd.Timestamp(split_start)]
        if split_end is not None:
            samples = samples[samples["game_date"] < pd.Timestamp(split_end)]

        by_player = {
            int(player_id): group.sort_values("game_date").reset_index(drop=True)
            for player_id, group in self.player_history.dropna(subset=["player_id"]).groupby("player_id")
        }
        valid_indices = []
        reset = samples.reset_index(drop=True)
        for idx, row in reset.iterrows():
            if _history_count(by_player, row["player_id"], row["game_date"]) >= spec.min_history:
                valid_indices.append(idx)
        self.samples = reset.iloc[valid_indices].reset_index(drop=True)
        self.by_player = by_player

        # Build player game index for decay (same sequential logic as team games).
        self.player_game_index = _build_player_game_index(self.player_history)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        import numpy as np
        import torch

        row = self.samples.iloc[idx]
        values, mask, padding = self._player_sequence(row["player_id"], row["game_date"])
        sample_weight = float(
            _last_prior_game_weight(self.player_game_index, row["player_id"], row["game_date"], self.spec)
        )
        player_id = int(row["player_id"])
        return {
            "values": torch.from_numpy(values),
            "mask": torch.from_numpy(mask),
            "padding": torch.from_numpy(padding),
            "player_hash": torch.tensor(_hash_bucket(player_id, self.spec.hash_bucket_count), dtype=torch.long),
            "targets": {
                col: torch.tensor(float(row[col]), dtype=torch.float32) for col in self.target_columns
            },
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
            "game_pk": torch.tensor(int(row["game_pk"]), dtype=torch.long),
            "player_id": torch.tensor(player_id, dtype=torch.long),
        }

    def _player_sequence(self, player_id, game_date):
        import numpy as np

        feature_dim = len(self.standardizer.feature_columns)
        hist = self.by_player.get(int(player_id))
        if hist is None:
            values = np.zeros((0, feature_dim), dtype="float32")
            mask = np.zeros((0, feature_dim), dtype="float32")
        else:
            prior = hist[hist["game_date"] < game_date].tail(self.spec.history_length)
            values, mask = self.standardizer.transform(prior)
        return _left_pad(values, mask, self.spec.history_length, feature_dim)


def _build_player_game_index(player_history) -> dict[int, dict]:
    """Same sequential index structure as build_team_game_index, keyed by player_id."""
    import pandas as pd

    by_player: dict[int, dict] = {}
    if player_history.empty or "player_id" not in player_history.columns:
        return by_player

    ph = player_history.dropna(subset=["player_id", "game_date"]).copy()
    ph["game_date"] = pd.to_datetime(ph["game_date"], errors="coerce")
    ph = ph.dropna(subset=["game_date"])

    for player_id, grp in ph.groupby("player_id", sort=False):
        grp_sorted = grp.sort_values("game_date").drop_duplicates("game_pk")
        dates = grp_sorted["game_date"].tolist()
        seasons = (
            grp_sorted["season"].tolist()
            if "season" in grp_sorted.columns
            else [None] * len(dates)
        )
        by_player[int(player_id)] = {
            "dates": dates,
            "seasons": seasons,
            "indices": list(range(len(dates))),
        }
    return by_player


class LiveGameSequenceDataset:
    """Pitch-prefix dataset for live in-game repricing."""

    def __init__(
        self,
        pitch_sequences,
        game_targets,
        standardizer: Standardizer,
        spec: SequenceSpec,
        split_start=None,
        split_end=None,
    ):
        import pandas as pd

        self.pitch_sequences = pitch_sequences.copy()
        self.game_targets = game_targets.copy()
        self.pitch_sequences["game_date"] = pd.to_datetime(
            self.pitch_sequences["game_date"], errors="coerce"
        )
        self.game_targets["game_date"] = pd.to_datetime(self.game_targets["game_date"], errors="coerce")
        self.standardizer = standardizer
        self.spec = spec
        self.max_date = self.game_targets["game_date"].max()

        targets = self.game_targets[self.game_targets["target_status"].eq("trainable")].copy()
        targets = targets.dropna(subset=["game_pk", "game_date"])
        for col in GAME_TARGET_COLUMNS:
            targets = targets.dropna(subset=[col])
        if split_start is not None:
            targets = targets[targets["game_date"] >= pd.Timestamp(split_start)]
        if split_end is not None:
            targets = targets[targets["game_date"] < pd.Timestamp(split_end)]
        target_by_game = {int(row.game_pk): row for row in targets.itertuples(index=False)}

        self.by_game = {
            int(game_pk): group.sort_values("sequence_index").reset_index(drop=True)
            for game_pk, group in self.pitch_sequences.groupby("game_pk")
            if int(game_pk) in target_by_game
        }
        self.samples = []
        for game_pk, group in self.by_game.items():
            if len(group) == 0:
                continue
            positions = list(range(0, len(group), max(spec.live_stride, 1)))
            if positions[-1] != len(group) - 1:
                positions.append(len(group) - 1)
            if len(positions) > spec.live_max_prefixes_per_game:
                step = max(len(positions) // spec.live_max_prefixes_per_game, 1)
                positions = positions[::step][: spec.live_max_prefixes_per_game]
            for end_pos in positions:
                self.samples.append((game_pk, int(end_pos)))
        self.target_by_game = target_by_game

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        import numpy as np
        import torch

        game_pk, end_pos = self.samples[idx]
        seq = self.by_game[game_pk].iloc[: end_pos + 1].tail(self.spec.history_length)
        values, mask = self.standardizer.transform(seq)
        values, mask, padding = _left_pad(
            values,
            mask,
            self.spec.history_length,
            len(self.standardizer.feature_columns),
        )
        target = self.target_by_game[game_pk]
        # Live dataset retains calendar-day weighting since pitch sequences
        # do not have a meaningful game-index structure at the pitch level.
        age_days = max((self.max_date - target.game_date).days, 0)
        sample_weight = float(np.exp(-self.spec.time_decay_lambda * age_days))

        batter_hashes = _hash_sequence(seq.get("batter_id"), self.spec.history_length, self.spec.hash_bucket_count)
        pitcher_hashes = _hash_sequence(seq.get("pitcher_id"), self.spec.history_length, self.spec.hash_bucket_count)
        pitch_type_hashes = _hash_sequence(seq.get("pitch_type"), self.spec.history_length, 256)

        return {
            "values": torch.from_numpy(values),
            "mask": torch.from_numpy(mask),
            "padding": torch.from_numpy(padding),
            "batter_hashes": torch.tensor(batter_hashes, dtype=torch.long),
            "pitcher_hashes": torch.tensor(pitcher_hashes, dtype=torch.long),
            "pitch_type_hashes": torch.tensor(pitch_type_hashes, dtype=torch.long),
            "targets": {
                col: torch.tensor(float(getattr(target, col)), dtype=torch.float32)
                for col in GAME_TARGET_COLUMNS
            },
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
            "game_pk": torch.tensor(int(game_pk), dtype=torch.long),
            "end_sequence_index": torch.tensor(int(end_pos), dtype=torch.long),
        }


def temporal_split_dates(game_targets, train_fraction=0.80, val_fraction=0.10, min_date=None):
    import pandas as pd

    parsed_dates = pd.to_datetime(game_targets["game_date"], errors="coerce").dropna()
    min_ts = pd.Timestamp(min_date) if min_date is not None else None
    if min_ts is not None:
        parsed_dates = parsed_dates[parsed_dates >= min_ts]

    dates = parsed_dates.sort_values().drop_duplicates().to_list()
    if len(dates) < 10:
        suffix = f" on or after {min_ts.date()}" if min_ts is not None else ""
        raise ValueError(
            f"Need at least 10 distinct game dates{suffix} for temporal train/val/test split"
        )

    train_end = dates[int(len(dates) * train_fraction)]
    val_end = dates[int(len(dates) * (train_fraction + val_fraction))]
    return train_end, val_end


def _left_pad(values, mask, history_length: int, feature_dim: int):
    import numpy as np

    pad_count = history_length - len(values)
    if pad_count > 0:
        values = np.vstack([np.zeros((pad_count, feature_dim), dtype="float32"), values])
        mask = np.vstack([np.zeros((pad_count, feature_dim), dtype="float32"), mask])
        padding = np.concatenate(
            [np.zeros(pad_count, dtype="float32"), np.ones(len(values) - pad_count, dtype="float32")]
        )
    else:
        values = values[-history_length:]
        mask = mask[-history_length:]
        padding = np.ones(history_length, dtype="float32")
    return values.astype("float32"), mask.astype("float32"), padding.astype("float32")


def _hash_bucket(value, bucket_count: int) -> int:
    if value is None:
        return 0
    text = str(value)
    if text.lower() in {"nan", "none", ""}:
        return 0
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % max(bucket_count - 1, 1) + 1


def _hash_sequence(values, history_length: int, bucket_count: int):
    import numpy as np

    if values is None:
        raw = []
    else:
        raw = list(values)
    hashed = [_hash_bucket(value, bucket_count) for value in raw[-history_length:]]
    pad_count = history_length - len(hashed)
    if pad_count > 0:
        hashed = [0] * pad_count + hashed
    return np.asarray(hashed[-history_length:], dtype="int64")


def _history_count(by_team: dict[int, object], team_id, game_date: datetime) -> int:
    try:
        hist = by_team[int(team_id)]
    except (KeyError, TypeError, ValueError):
        return 0
    return int((hist["game_date"] < game_date).sum())
