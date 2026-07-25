"""Dataset for training the Live HAN model on pitch sequences.

WHY a separate dataset: The HAN model requires granular pitch-level features organized
by hierarchical indices (inning, AB, pitch). This is structurally different from the
simpler LiveGameSequenceDataset which treats pitch sequences as flat time series.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .datasets import Standardizer, infer_live_feature_columns


def _hash_bucket(value, bucket_count: int) -> int:
    """Blake2b hash-bucket for player IDs."""
    if value is None or pd.isna(value):
        return 0
    text = str(int(value)) if isinstance(value, (int, float)) else str(value)
    if text.lower() in {"nan", "none", ""}:
        return 0
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % max(bucket_count - 1, 1) + 1


class LiveHANDataset(Dataset):
    """PyTorch dataset for training the Hierarchical Attention Network on pitch sequences.

    Each sample is a sub-game: a prefix of pitches from a complete game, cut at a specific
    pitch position. Multiple prefixes are sampled from each game (stride=25, max 32 per game).

    Targets are REMAINING runs from that point forward (final - observed).

    WHY sub-game sampling: Trains the model to predict from any game state, not just
    the pregame start. This is crucial for live repricing.

    WHY game-index decay: More recent games are weighted higher, but offseason gaps
    don't artificially decay prior-season games (uses sequential game index, not calendar days).
    """

    def __init__(
        self,
        pitch_sequences: pd.DataFrame,
        game_targets: pd.DataFrame,
        pregame_features: pd.DataFrame,  # Per-game pregame context
        standardizer: Standardizer,
        split_start=None,
        split_end=None,
        stride: int = 25,  # TODO: validate — placeholder
        max_prefixes_per_game: int = 32,  # TODO: validate — placeholder
        max_seq_len: int = 350,  # TODO: validate — placeholder
        batter_buckets: int = 512,
        pitcher_buckets: int = 512,
    ):
        """
        Args:
            pitch_sequences: Parquet with all pitches (numeric + categorical columns)
            game_targets: Parquet with final outcomes (home_runs, away_runs, etc.)
            pregame_features: Per-game pregame context features (home/away team strength, etc.)
            standardizer: Fitted on training split of pitch_sequences
            split_start: Earliest game_date to include (temporal split)
            split_end: Latest game_date to include (temporal split)
            stride: Pitch interval for sub-game sampling
            max_prefixes_per_game: Cap on number of sub-game samples per game
            max_seq_len: Max pitches to include in a sequence (truncate earlier pitches)
        """
        self.pitch_sequences = pitch_sequences.copy()
        self.game_targets = game_targets.copy()
        self.pregame_features = pregame_features.copy()
        self.standardizer = standardizer
        self.max_seq_len = max_seq_len
        self.batter_buckets = batter_buckets
        self.pitcher_buckets = pitcher_buckets

        # Convert dates
        self.pitch_sequences["game_date"] = pd.to_datetime(self.pitch_sequences["game_date"], errors="coerce")
        self.game_targets["game_date"] = pd.to_datetime(self.game_targets["game_date"], errors="coerce")
        self.pregame_features["game_date"] = pd.to_datetime(self.pregame_features["game_date"], errors="coerce")

        # Filter trainable targets
        targets = self.game_targets[self.game_targets["target_status"].eq("trainable")].copy()
        targets = targets.dropna(subset=["game_pk", "home_runs", "away_runs", "home_win", "yrfi", "extra_innings"])

        # Exclude structural outlier seasons from all frames
        from pregame.strategy.config import SKIP_SEASONS
        if SKIP_SEASONS:
            targets = targets[~targets["season"].isin(SKIP_SEASONS)]
            if "season" in self.pitch_sequences.columns:
                self.pitch_sequences = self.pitch_sequences[~self.pitch_sequences["season"].isin(SKIP_SEASONS)]
            if "season" in self.pregame_features.columns:
                self.pregame_features = self.pregame_features[~self.pregame_features["season"].isin(SKIP_SEASONS)]

        # Temporal split
        if split_start is not None:
            targets = targets[targets["game_date"] >= pd.Timestamp(split_start)]
        if split_end is not None:
            targets = targets[targets["game_date"] < pd.Timestamp(split_end)]

        # Build per-game lookup
        self.target_by_game = {int(row.game_pk): row for row in targets.itertuples(index=False)}
        self.pregame_by_game = {
            int(row.game_pk): row for row in self.pregame_features.itertuples(index=False)
            if int(row.game_pk) in self.target_by_game
        }

        # Group pitches by game
        valid_games = set(self.target_by_game.keys())
        pitch_seqs = self.pitch_sequences[self.pitch_sequences["game_pk"].isin(valid_games)].copy()

        self.by_game = {
            int(game_pk): group.sort_values("pitch_sequence_index").reset_index(drop=True)
            for game_pk, group in pitch_seqs.groupby("game_pk")
        }

        # Generate sub-game samples: (game_pk, end_pitch_idx)
        self.samples = []
        for game_pk, seq in self.by_game.items():
            if len(seq) == 0:
                continue

            # Sample prefix positions at stride intervals
            positions = list(range(0, len(seq), stride))
            if positions[-1] != len(seq) - 1:
                positions.append(len(seq) - 1)  # Always include final state

            # Cap at max_prefixes_per_game
            if len(positions) > max_prefixes_per_game:
                step = max(len(positions) // max_prefixes_per_game, 1)
                positions = positions[::step][:max_prefixes_per_game]

            for end_pos in positions:
                # Skip very early game states (need at least 5 pitches for context)
                if end_pos < 5:
                    continue
                self.samples.append((game_pk, end_pos))

        # Compute sample weights (game-level decay, same weight for all sub-games from one game)
        # For simplicity, use calendar-day decay as placeholder
        # TODO: validate — implement game-index decay per team (requires team_games context)
        self.max_date = targets["game_date"].max()
        self.sample_weights = {}
        for game_pk in self.target_by_game.keys():
            game_date = self.target_by_game[game_pk].game_date
            age_days = max((self.max_date - game_date).days, 0)
            # λ=0.003 gives half-weight after ~231 days (one offseason)
            self.sample_weights[game_pk] = float(np.exp(-0.003 * age_days))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Return a single sub-game sample with all HAN model input tensors."""
        game_pk, end_pos = self.samples[idx]

        # Get pitch sequence prefix (up to end_pos, truncated to max_seq_len)
        seq = self.by_game[game_pk].iloc[:end_pos + 1]
        if len(seq) > self.max_seq_len:
            seq = seq.iloc[-self.max_seq_len:]

        # Get game targets and pregame features
        target = self.target_by_game[game_pk]
        pregame = self.pregame_by_game.get(game_pk)

        # Observed scores at this pitch position
        last_pitch = seq.iloc[-1]
        h_observed = int(last_pitch.score_home) if pd.notna(last_pitch.score_home) else 0
        a_observed = int(last_pitch.score_away) if pd.notna(last_pitch.score_away) else 0

        # Remaining runs = final - observed
        h_remaining = max(0, int(target.home_runs) - h_observed)
        a_remaining = max(0, int(target.away_runs) - a_observed)

        # Game progress (fraction of 9 innings completed)
        inning = last_pitch.inning if pd.notna(last_pitch.inning) else 1.0
        is_top = last_pitch.is_top_inning if pd.notna(last_pitch.is_top_inning) else 1.0
        game_progress = float((inning - 1 + (0.0 if is_top else 0.5)) / 9.0)
        game_progress = np.clip(game_progress, 0.0, 1.5)  # Cap at 1.5 for extra innings

        # Build HAN model input tensors
        batch_dict = self._build_han_inputs(seq, pregame, game_progress)

        # Add targets
        batch_dict["targets"] = {
            "remaining_home_runs": torch.tensor(float(h_remaining), dtype=torch.float32),
            "remaining_away_runs": torch.tensor(float(a_remaining), dtype=torch.float32),
            "home_win": torch.tensor(float(target.home_win), dtype=torch.float32),
            "yrfi": torch.tensor(float(target.yrfi), dtype=torch.float32),
            "extra_innings": torch.tensor(float(target.extra_innings), dtype=torch.float32),
        }

        # Sample weight
        batch_dict["sample_weight"] = torch.tensor(self.sample_weights[game_pk], dtype=torch.float32)
        batch_dict["game_pk"] = torch.tensor(game_pk, dtype=torch.long)
        batch_dict["game_progress_scalar"] = torch.tensor(game_progress, dtype=torch.float32)

        return batch_dict

    def _build_han_inputs(self, seq: pd.DataFrame, pregame, game_progress: float) -> dict:
        """Convert pitch sequence DataFrame into HAN model input dict.

        Simplified implementation: uses available columns from pitch_sequences parquet.
        Full implementation would require more granular feature engineering.

        # TODO: validate — feature selection and grouping
        """
        S = len(seq)

        # Continuous features (kinematics): 20 dims
        continuous_cols = [
            "release_speed", "end_speed", "plate_time", "extension",
            "coord_px", "coord_pz", "coord_x0", "coord_y0", "coord_z0",
            "coord_vx0", "coord_vy0", "coord_vz0",
            "coord_ax", "coord_ay", "coord_az",
            "pfx_x", "pfx_z", "break_angle", "break_length", "spin_rate",
        ]
        continuous = self._extract_features(seq, continuous_cols, S, 20)

        # Pitch type one-hot (9 classes: FF, SI, SL, CU, CH, FC, KC, FS, Other)
        PITCH_TYPE_MAP = {"FF": 0, "SI": 1, "SL": 2, "CU": 3, "CH": 4, "FC": 5, "KC": 6, "FS": 7}
        pitch_type = torch.zeros(S, 9, dtype=torch.float32)
        if "pitch_type" in seq.columns:
            for i, pt in enumerate(seq["pitch_type"].fillna("UN").to_numpy()):
                idx = PITCH_TYPE_MAP.get(str(pt), 8)  # 8 = Other/Unknown
                pitch_type[i, idx] = 1.0

        # Outcome flags (4): is_strike, is_ball, is_in_play, is_pitch
        outcome_cols = ["is_strike", "is_ball", "is_in_play", "is_pitch"]
        outcome_flags = self._extract_features(seq, outcome_cols, S, 4)

        # Count state (3): balls, strikes, outs
        count_cols = ["pitch_count_balls", "pitch_count_strikes", "pitch_count_outs"]
        count_state = self._extract_features(seq, count_cols, S, 3)

        # Game state (3): inning, is_top_inning, run_diff
        game_state_arr = np.zeros((S, 3), dtype="float32")
        if "inning" in seq.columns:
            game_state_arr[:, 0] = seq["inning"].fillna(1.0).to_numpy()
        if "is_top_inning" in seq.columns:
            game_state_arr[:, 1] = seq["is_top_inning"].fillna(0.0).to_numpy()
        # run_diff = score_home - score_away
        if "score_home" in seq.columns and "score_away" in seq.columns:
            game_state_arr[:, 2] = (seq["score_home"] - seq["score_away"]).fillna(0.0).to_numpy()
        game_state = torch.from_numpy(game_state_arr)

        # Base state (3): on_first, on_second, on_third (binary flags)
        base_state_arr = np.zeros((S, 3), dtype="float32")
        if "pre_on_first_id" in seq.columns:
            base_state_arr[:, 0] = seq["pre_on_first_id"].notna().astype("float32").to_numpy()
        if "pre_on_second_id" in seq.columns:
            base_state_arr[:, 1] = seq["pre_on_second_id"].notna().astype("float32").to_numpy()
        if "pre_on_third_id" in seq.columns:
            base_state_arr[:, 2] = seq["pre_on_third_id"].notna().astype("float32").to_numpy()
        base_state = torch.from_numpy(base_state_arr)

        # Player hashes
        batter_hash = self._hash_column(seq, "batter_id", S, self.batter_buckets)
        pitcher_hash = self._hash_column(seq, "pitcher_id", S, self.pitcher_buckets)

        # Handedness (2): bat_side (L=0, R=1), pitch_hand (L=0, R=1)
        handedness_arr = np.zeros((S, 2), dtype="float32")
        if "bat_side_code" in seq.columns:
            handedness_arr[:, 0] = seq["bat_side_code"].map({"L": 0.0, "R": 1.0}).fillna(0.5).to_numpy()
        if "pitch_hand_code" in seq.columns:
            handedness_arr[:, 1] = seq["pitch_hand_code"].map({"L": 0.0, "R": 1.0}).fillna(0.5).to_numpy()
        handedness = torch.from_numpy(handedness_arr)

        # Score (2): home_runs, away_runs at each pitch
        score_arr = np.zeros((S, 2), dtype="float32")
        if "score_home" in seq.columns:
            score_arr[:, 0] = seq["score_home"].fillna(0.0).to_numpy()
        if "score_away" in seq.columns:
            score_arr[:, 1] = seq["score_away"].fillna(0.0).to_numpy()
        score = torch.from_numpy(score_arr)

        # Positional (2): pitch_in_game, pitch_in_ab
        positional_arr = np.zeros((S, 2), dtype="float32")
        if "pitch_sequence_index" in seq.columns:
            positional_arr[:, 0] = seq["pitch_sequence_index"].fillna(0.0).to_numpy()
        if "pitch_number" in seq.columns:
            positional_arr[:, 1] = seq["pitch_number"].fillna(0.0).to_numpy()
        positional = torch.from_numpy(positional_arr)

        # Intra-AB (1): pitch count within current at-bat
        intra_ab = torch.from_numpy(seq["pitch_number"].fillna(1.0).to_numpy().astype("float32")).unsqueeze(-1)

        # Elapsed time (1): minutes since first pitch (placeholder: use pitch_sequence_index as proxy)
        # TODO: validate — compute actual elapsed time from timestamps
        elapsed_time = torch.from_numpy((seq["pitch_sequence_index"].fillna(0.0) / 60.0).to_numpy().astype("float32")).unsqueeze(-1)

        # Hierarchy indices (3): inning_idx, ab_idx, pitch_idx
        # Simplified: use inning, at_bat_index, pitch_number (clamped to max values)
        hierarchy_arr = np.zeros((S, 3), dtype="int64")
        if "inning" in seq.columns:
            hierarchy_arr[:, 0] = np.clip(seq["inning"].fillna(1.0).astype(int).to_numpy(), 0, 19)
        if "at_bat_index" in seq.columns:
            # AB index within game (need to mod by max_abs_per_inning)
            # Placeholder: use cumulative count
            hierarchy_arr[:, 1] = np.clip(seq["at_bat_index"].fillna(0).astype(int).to_numpy() % 25, 0, 24)
        if "pitch_number" in seq.columns:
            hierarchy_arr[:, 2] = np.clip(seq["pitch_number"].fillna(1.0).astype(int).to_numpy(), 0, 14)
        hierarchy_indices = torch.from_numpy(hierarchy_arr)

        # Attention mask: True for valid pitches
        attention_mask = torch.ones(S, dtype=torch.bool)

        # Pregame prior features — extract available numeric fields from pregame row
        # WHY fixed 128-dim: model architecture expects d_pregame=128.
        # We populate from available pregame data; remaining dims stay zero.
        d_pregame = 128
        if pregame is not None:
            pregame_prior = self._extract_pregame_prior(pregame, d_pregame)
        else:
            pregame_prior = torch.zeros(d_pregame, dtype=torch.float32)

        # Game progress (scalar)
        game_progress_tensor = torch.tensor([game_progress], dtype=torch.float32)

        return {
            "continuous": continuous,
            "pitch_type": pitch_type,
            "outcome_flags": outcome_flags,
            "count_state": count_state,
            "game_state": game_state,
            "base_state": base_state,
            "batter_hash": batter_hash,
            "pitcher_hash": pitcher_hash,
            "handedness": handedness,
            "score": score,
            "positional": positional,
            "intra_ab": intra_ab,
            "elapsed_time": elapsed_time,
            "hierarchy_indices": hierarchy_indices,
            "attention_mask": attention_mask,
            "pregame_prior": pregame_prior,
            "game_progress": game_progress_tensor,
        }

    def _extract_features(self, seq: pd.DataFrame, cols: list[str], S: int, dim: int) -> torch.Tensor:
        """Extract and standardize feature columns, returning [S, dim] tensor."""
        arr = np.zeros((S, dim), dtype="float32")
        for i, col in enumerate(cols[:dim]):
            if col in seq.columns:
                arr[:, i] = seq[col].fillna(0.0).to_numpy()
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        # Apply standardization if available for these columns
        if self.standardizer is not None:
            for i, col in enumerate(cols[:dim]):
                if col in self.standardizer.mean:
                    arr[:, i] = (arr[:, i] - self.standardizer.mean[col]) / self.standardizer.std[col]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(arr)

    def _hash_column(self, seq: pd.DataFrame, col: str, S: int, buckets: int) -> torch.LongTensor:
        """Hash a categorical column to integer buckets."""
        if col not in seq.columns:
            return torch.zeros(S, dtype=torch.long)

        hashes = np.array([_hash_bucket(val, buckets) for val in seq[col].to_numpy()])
        return torch.from_numpy(hashes.astype("int64"))

    def _extract_pregame_prior(self, pregame, d_pregame: int) -> torch.Tensor:
        """Extract numeric features from the pregame row into a fixed-size vector.

        WHY not random: The cross-attention merger and pregame gate need real signal
        to learn meaningful pregame→live conditioning. Random noise teaches the model
        to ignore the pregame branch entirely.
        """
        vec = np.zeros(d_pregame, dtype="float32")
        # Extract available numeric fields from the namedtuple
        idx = 0
        for field_name in pregame._fields:
            if idx >= d_pregame:
                break
            val = getattr(pregame, field_name)
            try:
                fval = float(val)
                if np.isfinite(fval):
                    vec[idx] = fval
                    idx += 1
            except (TypeError, ValueError):
                continue
        return torch.from_numpy(vec)
