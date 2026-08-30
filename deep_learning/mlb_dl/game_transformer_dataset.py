"""Dataset for the unified GameTransformer model.

Produces one sample per (game, prefix_length) pair, combining:
- Historical pitch sequences from prior SP starts and team games
- Live pitch prefix of the current game (T=0 for pregame)
- Per-player context for player props
- Game-level and player-level targets with masks

Memory-efficient: pitch sequences are stored once with an offset index;
per-game slices are loaded on demand via iloc.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .datasets import (
    Standardizer,
    SequenceSpec,
    build_team_game_index,
    compute_game_decay_weight,
    temporal_split_dates,
    _hash_bucket,
    _left_pad,
    GAME_TARGET_COLUMNS,
    PLAYER_BATTING_TARGET_COLUMNS,
    PLAYER_PITCHING_TARGET_COLUMNS,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """Controls ablation experiments for the GameTransformer.

    Each field corresponds to an experimental question about what historical
    context improves predictions beyond the live pitch prefix alone.
    """

    # Q1: Team offense context scope
    team_context_mode: str = "all_games"  # "all_games" | "lineup_overlap" | "similarity_weighted"
    lineup_overlap_threshold: int = 5  # for "lineup_overlap" mode

    # Q2: Pitcher-batter matchup
    matchup_mode: str = "none"  # "none" | "raw_sequences" | "compressed_summary"
    matchup_max_pa: int = 20  # max historical PAs to include

    # Q3: Bullpen modeling
    bullpen_mode: str = "implicit"  # "implicit" | "explicit_profile" | "reliever_sequences"
    bullpen_n_relievers: int = 5  # for "reliever_sequences" mode

    # Context sizes
    sp_history_games: int = 5
    team_history_games: int = 10
    tokens_per_game: int = 4  # Perceiver output tokens per historical game
    player_history_games: int = 15  # for player prop context


# Continuous pitch features used for encoding each pitch event
PITCH_CONTINUOUS_COLS = [
    # Pitch kinematics (17)
    "release_speed", "end_speed", "plate_time", "extension",
    "coord_px", "coord_pz", "coord_x0", "coord_y0", "coord_z0",
    "coord_vx0", "coord_vy0", "coord_vz0",
    "coord_ax", "coord_ay", "coord_az",
    "pfx_x", "pfx_z",
    # Break characteristics (4)
    "break_angle", "break_length", "break_y", "spin_rate",
    "spin_direction",
    # Strike zone and confidence (3)
    "strike_zone_top", "strike_zone_bottom", "type_confidence",
    # Zone (1)
    "zone_location",
    # Hit data (5)
    "hit_launch_speed", "hit_launch_angle", "hit_total_distance",
    "hit_coord_x", "hit_coord_y",
    # Game state — score (3)
    "score_home", "score_away", "score_diff_batting",
    # Game state — count and outs (6)
    "cum_balls", "cum_strikes", "cum_outs",
    "pitch_count_balls", "pitch_count_strikes", "pitch_count_outs",
    # Outcome flags (4)
    "is_pitch", "is_strike", "is_ball", "is_in_play",
    # Game structure (4)
    "is_top_inning", "inning", "pitch_number", "pitch_sequence_index",
    # Base runners — pre-pitch occupancy as binary (3)
    "pre_on_first", "pre_on_second", "pre_on_third",
    # Environment (1)
    "weather_temp",
]

# Event type categories for non-pitch events.
# Statcast stores pitch_event_type (or equivalent): pitch, action, no_pitch, pickoff.
# We further split "action" by at_bat_event semantics into meaningful categories.
EVENT_TYPE_VOCAB = [
    "pitch",            # 0: standard pitch thrown (~92% of rows)
    "substitution",     # 1: pitching change, defensive sub, pinch hitter
    "stolen_base",      # 2: steal attempt (success or caught)
    "wild_pitch",       # 3: wild pitch or passed ball (runner advancement)
    "balk",             # 4: balk (free base)
    "intentional_walk", # 5: automatic IBB (no pitch thrown)
    "pickoff",          # 6: pickoff attempt
    "other_action",     # 7: mound visit, replay review, delay, ejection, etc.
]
EVENT_TYPE_TO_IDX = {name: idx for idx, name in enumerate(EVENT_TYPE_VOCAB)}
NUM_EVENT_TYPES = len(EVENT_TYPE_VOCAB)

# Pitch type vocabulary (Statcast pitch_type codes).
# Index 0 reserved as padding/unknown.
PITCH_TYPE_VOCAB = [
    "<PAD>",  # 0: padding / unknown
    "FF",     # 1: Four-seam fastball
    "SI",     # 2: Sinker
    "SL",     # 3: Slider
    "CU",     # 4: Curveball
    "CH",     # 5: Changeup
    "FC",     # 6: Cutter
    "KC",     # 7: Knuckle curve
    "FS",     # 8: Splitter
    "ST",     # 9: Sweeper
    "SV",     # 10: Slurve
    "KN",     # 11: Knuckleball
    "EP",     # 12: Eephus
    "CS",     # 13: Slow curve
    "SC",     # 14: Screwball
    "UN",     # 15: Unidentified
    "FA",     # 16: Generic fastball
    "FO",     # 17: Forkball
    "PO",     # 18: Pitch out
    "IN",     # 19: Intentional ball
]
PITCH_TYPE_TO_IDX = {name: idx for idx, name in enumerate(PITCH_TYPE_VOCAB)}

# Bat side vocabulary — index 0 is padding/unknown so NaN does not map to "L"
BAT_SIDE_VOCAB = ["<PAD>", "L", "R", "S"]  # 0=unknown, 1=Left, 2=Right, 3=Switch
BAT_SIDE_TO_IDX = {name: idx for idx, name in enumerate(BAT_SIDE_VOCAB)}

# Pitch hand vocabulary — index 0 is padding/unknown so NaN does not map to "L"
PITCH_HAND_VOCAB = ["<PAD>", "L", "R"]  # 0=unknown, 1=Left, 2=Right
PITCH_HAND_TO_IDX = {name: idx for idx, name in enumerate(PITCH_HAND_VOCAB)}

# Half inning vocabulary
HALF_INNING_VOCAB = ["top", "bottom"]  # 0=top, 1=bottom
HALF_INNING_TO_IDX = {name: idx for idx, name in enumerate(HALF_INNING_VOCAB)}

# Hit trajectory vocabulary (index 0 = none/no batted ball)
HIT_TRAJECTORY_VOCAB = [
    "none",          # 0: no batted ball event (strikeout, walk, etc.)
    "ground_ball",   # 1
    "fly_ball",      # 2
    "line_drive",    # 3
    "popup",         # 4
    "bunt_ground",   # 5: bunt ground ball
    "bunt_popup",    # 6: bunt popup
]
HIT_TRAJECTORY_TO_IDX = {name: idx for idx, name in enumerate(HIT_TRAJECTORY_VOCAB)}

# Hit hardness vocabulary (index 0 = none/unknown)
HIT_HARDNESS_VOCAB = [
    "none",    # 0: no batted ball or unknown
    "soft",    # 1
    "medium",  # 2
    "hard",    # 3
]
HIT_HARDNESS_TO_IDX = {name: idx for idx, name in enumerate(HIT_HARDNESS_VOCAB)}

# Raw columns from game_meta needed for flat game-context features.
# These are pre-game-available metadata; post-game columns (attendance,
# game_duration_minutes, review challenges, no-hitter flags) are excluded
# to prevent leakage.
FLAT_CONTEXT_COLS = [
    # Venue (identity + physical)
    "venue_id", "venue_latitude", "venue_longitude",
    "venue_capacity", "venue_surface", "venue_roof_type",
    # Personnel (pre-game available)
    "umpire_hp", "probable_pitcher_home_id", "probable_pitcher_away_id",
    # Game timing (calendar features)
    "start_time", "game_date", "game_datetime_utc",
    # Scheduling
    "day_night", "game_number", "double_header", "tiebreaker",
    # Regime flags
    "rule_3batter_minimum", "rule_universal_dh", "rule_shift_ban_pitch_clock",
]

# Output dimension of _get_flat_features — must match model's flat_feature_dim.
# Only unlearnable features: things NOT recoverable from sequential inputs
# (ratings, weather temporal, pitch tokens, player history).
# Layout (30 features):
#   [0]      venue_id hash
#   [1-2]    venue_lat, venue_lon
#   [3]      venue_capacity (standardized)
#   [4]      venue_surface (turf=1)
#   [5-7]    venue_roof one-hot (open, dome, retractable)
#   [8-13]   venue dimensions (LF/CF/RF distance, LF/CF/RF wall height)
#   [14]     umpire_hp hash
#   [15-16]  probable_pitcher_home/away hash
#   [17-18]  start_hour_sin, start_hour_cos
#   [19-20]  day_of_week_sin, day_of_week_cos
#   [21-22]  month_sin, month_cos
#   [23]     day_night (night=1)
#   [24]     game_number_is_2 (doubleheader game 2)
#   [25]     double_header flag
#   [26]     tiebreaker flag
#   [27-29]  regime flags (3batter_min, universal_DH, shift_ban)
FLAT_FEATURE_DIM = 30

# Bullpen aggregate stats (for explicit_profile mode)
BULLPEN_PROFILE_DIM = 10  # ERA, FIP, K%, BB%, WHIP, recent_workload x3, avail_top1/2/3

# Matchup compressed summary dim
MATCHUP_SUMMARY_DIM = 7  # n_pa, avg, iso, k_rate, avg_vs_hand, woba_vs_hand, n_pa_bucket

# Max players per game (9 batters x 2 sides + 2 SPs)
MAX_PLAYERS_PER_GAME = 20

# Player stat line dimension (game stats + season cumulative)
PLAYER_STAT_DIM = 25  # 17 BATTING_SUM_COLUMNS + 8 BATTING_SEASON_COLUMNS


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def map_pitch_type(series: pd.Series) -> np.ndarray:
    """Map Statcast pitch_type strings to vocabulary indices.

    Unknown or NaN values map to 0 (padding).
    """
    result = np.zeros(len(series), dtype=np.int64)
    for i, val in enumerate(series):
        if pd.isna(val):
            result[i] = 0
        else:
            result[i] = PITCH_TYPE_TO_IDX.get(str(val).strip().upper(), 0)
    return result


def map_bat_side(series: pd.Series) -> np.ndarray:
    """Map bat_side_code (L/R/S) to vocabulary indices. Default=0 (L)."""
    result = np.zeros(len(series), dtype=np.int64)
    for i, val in enumerate(series):
        if pd.isna(val):
            result[i] = 0
        else:
            result[i] = BAT_SIDE_TO_IDX.get(str(val).strip().upper(), 0)
    return result


def map_pitch_hand(series: pd.Series) -> np.ndarray:
    """Map pitch_hand_code (L/R) to vocabulary indices. Default=0 (L)."""
    result = np.zeros(len(series), dtype=np.int64)
    for i, val in enumerate(series):
        if pd.isna(val):
            result[i] = 0
        else:
            result[i] = PITCH_HAND_TO_IDX.get(str(val).strip().upper(), 0)
    return result


def map_half_inning(series: pd.Series) -> np.ndarray:
    """Map is_top_inning (1=top, 0=bottom) to half-inning index."""
    result = np.zeros(len(series), dtype=np.int64)
    for i, val in enumerate(series):
        if pd.isna(val):
            result[i] = 0
        else:
            # is_top_inning=1 -> top (idx 0); is_top_inning=0 -> bottom (idx 1)
            result[i] = 0 if int(val) == 1 else 1
    return result


def map_hit_trajectory(series: pd.Series) -> np.ndarray:
    """Map hit trajectory (bb_type or launch_speed_angle category) to index.

    Statcast bb_type values: ground_ball, fly_ball, line_drive, popup.
    Extended with bunt_ground, bunt_popup from description parsing.
    NaN or missing -> 0 (none).
    """
    result = np.zeros(len(series), dtype=np.int64)
    for i, val in enumerate(series):
        if pd.isna(val):
            result[i] = 0
        else:
            key = str(val).strip().lower().replace(" ", "_").replace("-", "_")
            result[i] = HIT_TRAJECTORY_TO_IDX.get(key, 0)
    return result


def map_hit_hardness(series: pd.Series) -> np.ndarray:
    """Map hit hardness category to index.

    Derived from launch_speed: soft (<75 mph), medium (75-95), hard (>95).
    Or from Statcast launch_speed_angle classification.
    NaN or missing -> 0 (none/unknown).
    """
    result = np.zeros(len(series), dtype=np.int64)
    for i, val in enumerate(series):
        if pd.isna(val):
            result[i] = 0
        else:
            key = str(val).strip().lower()
            result[i] = HIT_HARDNESS_TO_IDX.get(key, 0)
    return result


def classify_event_type(row: pd.Series) -> int:
    """Map a Statcast pitch row to an event type index.

    Uses is_pitch flag + event_type/at_bat_event to classify non-pitch events.
    Returns integer index into EVENT_TYPE_VOCAB.
    """
    if row.get("is_pitch", 1) == 1:
        return EVENT_TYPE_TO_IDX["pitch"]

    event = str(row.get("event_type", "")).lower()
    at_bat = str(row.get("at_bat_event", "")).lower()
    pitch_call = str(row.get("pitch_call", "")).lower()

    # Substitution: pitching change, defensive sub, pinch hitter/runner
    if any(kw in pitch_call for kw in ("replaces", "substitution", "pitching change")):
        return EVENT_TYPE_TO_IDX["substitution"]
    if any(kw in at_bat for kw in ("sub", "pinch", "pitching change")):
        return EVENT_TYPE_TO_IDX["substitution"]

    # Stolen base (includes caught stealing)
    if "stolen" in at_bat or "steal" in at_bat or "stolen" in pitch_call or "steal" in pitch_call:
        return EVENT_TYPE_TO_IDX["stolen_base"]

    # Wild pitch or passed ball
    if "wild" in at_bat or "passed" in at_bat or "wild" in pitch_call or "passed" in pitch_call:
        return EVENT_TYPE_TO_IDX["wild_pitch"]

    # Balk
    if "balk" in at_bat or "balk" in pitch_call:
        return EVENT_TYPE_TO_IDX["balk"]

    # Intentional walk (automatic, no pitch thrown)
    if "intent" in at_bat or event == "no_pitch":
        return EVENT_TYPE_TO_IDX["intentional_walk"]

    # Pickoff
    if "pickoff" in at_bat or "pickoff" in event or "pickoff" in pitch_call:
        return EVENT_TYPE_TO_IDX["pickoff"]

    return EVENT_TYPE_TO_IDX["other_action"]


def classify_event_type_series(seq: pd.DataFrame) -> np.ndarray:
    """Vectorized event type classification for a sequence DataFrame."""
    result = np.zeros(len(seq), dtype=np.int64)
    is_pitch = seq.get("is_pitch", pd.Series(np.ones(len(seq))))
    pitch_mask = is_pitch == 1
    # Default: all are "pitch" (index 0)
    result[pitch_mask.to_numpy()] = EVENT_TYPE_TO_IDX["pitch"]

    if pitch_mask.all():
        return result

    # Process non-pitch rows
    non_pitch_idx = np.where(~pitch_mask.to_numpy())[0]
    for idx in non_pitch_idx:
        result[idx] = classify_event_type(seq.iloc[idx])
    return result


def hash_bucket(player_id: str, n_buckets: int = 50000) -> int:
    """Blake2b hash-bucket for player identity embedding."""
    return _hash_bucket(player_id, n_buckets)


def _category_as_object(series: pd.Series) -> pd.Series:
    """Return category-backed series as object so fillna can use new sentinels."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.astype("object")
    return series


def build_game_index(game_meta_df: pd.DataFrame) -> dict:
    """Build team->game mapping with sequential game indices.

    Returns:
        {
            "by_team": {team_id: DataFrame sorted by game_date},
            "game_to_idx": {(team_id, game_pk): sequential_index},
            "sp_by_pitcher": {pitcher_id: [game_pk, ...] sorted by date},
        }
    """
    meta = game_meta_df.copy()
    meta["game_date"] = pd.to_datetime(meta["game_date"], errors="coerce")
    meta = meta.dropna(subset=["game_date", "game_pk"])

    # Team game index: each team's games in chronological order
    by_team: dict[int, pd.DataFrame] = {}
    game_to_idx: dict[tuple[int, int], int] = {}

    # Home side
    for team_id, grp in meta.groupby("home_team_id", sort=False):
        tid = int(team_id)
        if tid not in by_team:
            by_team[tid] = []
        for _, row in grp.iterrows():
            by_team[tid].append(row)

    # Away side
    for team_id, grp in meta.groupby("away_team_id", sort=False):
        tid = int(team_id)
        if tid not in by_team:
            by_team[tid] = []
        for _, row in grp.iterrows():
            by_team[tid].append(row)

    # Sort and deduplicate per team, assign sequential indices
    for tid in by_team:
        df = pd.DataFrame(by_team[tid]).drop_duplicates("game_pk").sort_values("game_date").reset_index(drop=True)
        by_team[tid] = df
        for seq_idx, gpk in enumerate(df["game_pk"].tolist()):
            game_to_idx[(tid, int(gpk))] = seq_idx

    # SP history: pitcher -> list of game_pks they started (chronological)
    sp_by_pitcher: dict[int, list[int]] = {}
    if "probable_pitcher_home_id" in meta.columns:
        for _, row in meta.sort_values("game_date").iterrows():
            for col in ["probable_pitcher_home_id", "probable_pitcher_away_id"]:
                if col in meta.columns and pd.notna(row.get(col)):
                    pid = int(row[col])
                    if pid not in sp_by_pitcher:
                        sp_by_pitcher[pid] = []
                    sp_by_pitcher[pid].append(int(row["game_pk"]))

    return {
        "by_team": by_team,
        "game_to_idx": game_to_idx,
        "sp_by_pitcher": sp_by_pitcher,
    }


def compute_decay_weights(
    target_game_idx: int,
    history_game_indices: np.ndarray,
    seasons: np.ndarray,
    target_season: int,
    lambda_intra: float = 0.015,
    lambda_inter: float = 0.30,
) -> np.ndarray:
    """Vectorized decay weight computation.

    w_i = exp(-lambda_intra * (target_idx - i)) * exp(-lambda_inter * seasons_crossed_i)
    """
    deltas = target_game_idx - history_game_indices
    seasons_crossed = np.maximum(target_season - seasons, 0).astype(np.float32)
    weights = np.exp(-lambda_intra * deltas) * np.exp(-lambda_inter * seasons_crossed)
    return weights.astype(np.float32)


def temporal_split(
    game_meta_df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Temporal train/val/test split. Returns (train_end, val_end) dates."""
    return temporal_split_dates(game_meta_df, train_fraction=train_frac, val_fraction=val_frac)


def _compute_jaccard(lineup_a: set, lineup_b: set) -> float:
    """Jaccard similarity between two lineups (sets of batter_ids)."""
    if not lineup_a or not lineup_b:
        return 0.0
    intersection = len(lineup_a & lineup_b)
    union = len(lineup_a | lineup_b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------

class GameTransformerDataset(Dataset):
    """PyTorch Dataset for the unified GameTransformer model.

    Each sample is a (game, prefix_length) pair. Prefix_length=0 means pregame.
    Constructs historical SP/team context, live prefix, and player context.
    """

    def __init__(
        self,
        pitch_sequences: pd.DataFrame,
        game_targets: pd.DataFrame,
        game_meta: pd.DataFrame,
        team_games: pd.DataFrame,
        player_batting_history: pd.DataFrame,
        standardizer: Standardizer,
        ablation: AblationConfig | None = None,
        spec: SequenceSpec | None = None,
        split_start=None,
        split_end=None,
        include_pregame: bool = True,
        include_live: bool = True,
        game_features: Optional[pd.DataFrame] = None,
        game_features_scaler: Optional[object] = None,
        weather_features: Optional[pd.DataFrame] = None,
        weather_temporal: Optional[pd.DataFrame] = None,
        venue_dimensions: Optional[pd.DataFrame] = None,
        daily_stats: Optional[pd.DataFrame] = None,
        weather_asof: Optional[dict] = None,
        wx_hour_offsets: Optional[dict] = None,
    ):
        self.standardizer = standardizer
        self.ablation = ablation or AblationConfig()
        self.spec = spec or SequenceSpec()
        self.include_pregame = include_pregame
        self.include_live = include_live

        # --- Prepare game targets ---
        game_targets = game_targets.copy()
        game_targets["game_date"] = pd.to_datetime(game_targets["game_date"], errors="coerce")
        targets = game_targets[game_targets["target_status"].eq("trainable")].copy()
        targets = targets.dropna(subset=["game_pk", "game_date"])

        if split_start is not None:
            targets = targets[targets["game_date"] >= pd.Timestamp(split_start)]
        if split_end is not None:
            targets = targets[targets["game_date"] < pd.Timestamp(split_end)]

        self.target_by_game: dict[int, pd.Series] = {
            int(row.game_pk): row for row in targets.itertuples(index=False)
        }

        # --- Prepare game metadata ---
        game_meta = game_meta.copy()
        game_meta["game_date"] = pd.to_datetime(game_meta["game_date"], errors="coerce")
        self.game_meta = game_meta
        self.meta_by_game: dict[int, pd.Series] = {}
        for _, row in game_meta.iterrows():
            gpk = int(row["game_pk"])
            if gpk in self.target_by_game:
                self.meta_by_game[gpk] = row

        # Date lookup covering ALL game_pks (trainable + non-trainable).
        # meta_by_game only has trainable games; _get_sp_context needs dates
        # for non-trainable starts to preserve chronological SP history.
        self._game_date_by_pk: dict[int, "pd.Timestamp"] = {
            int(row["game_pk"]): row["game_date"]
            for _, row in game_meta.iterrows()
            if pd.notna(row.get("game_date"))
        }

        # --- Build game index ---
        self.game_index = build_game_index(game_meta)

        # --- Prepare team games (for bullpen/lineup data) ---
        team_games = team_games.copy()
        team_games["game_date"] = pd.to_datetime(team_games["game_date"], errors="coerce")
        self.team_games = team_games

        # Per-team game lookup for lineup extraction
        self._team_game_lookup: dict[int, pd.DataFrame] = {}
        if "team_id" in team_games.columns:
            for tid, grp in team_games.groupby("team_id", sort=False):
                self._team_game_lookup[int(tid)] = grp.sort_values("game_date").reset_index(drop=True)

        # --- Memory-efficient pitch storage with offset index ---
        valid_games = set(self.target_by_game.keys())
        pitch_seqs = pitch_sequences[pitch_sequences["game_pk"].isin(valid_games)].copy()
        # Sort by game then chronological pitch order
        sort_cols = [c for c in ["game_pk", "play_index", "pitch_sequence_index"] if c in pitch_seqs.columns]
        if sort_cols:
            pitch_seqs = pitch_seqs.sort_values(sort_cols).reset_index(drop=True)
        self._pitches = pitch_seqs

        # Build game_pk -> (start_row, end_row) offset index
        self._game_offsets: dict[int, tuple[int, int]] = {}
        if len(pitch_seqs) > 0:
            game_pks_arr = pitch_seqs["game_pk"].to_numpy()
            changes = np.where(game_pks_arr[1:] != game_pks_arr[:-1])[0] + 1
            starts = np.concatenate([[0], changes])
            ends = np.concatenate([changes, [len(game_pks_arr)]])
            for s, e in zip(starts, ends):
                gpk = int(game_pks_arr[s])
                self._game_offsets[gpk] = (int(s), int(e))

        # Pre-compute standardized continuous pitch array so _load_game_pitches
        # is a single numpy slice instead of 52 pandas column operations per call.
        # With 10M pitches and 31 game lookups per __getitem__, the pandas path
        # makes each batch take minutes; this reduces it to microseconds.
        n_cont = len(PITCH_CONTINUOUS_COLS)
        _binary_cols = {"pre_on_first", "pre_on_second", "pre_on_third",
                        "is_pitch", "is_strike", "is_ball", "is_in_play", "is_top_inning"}
        self._pitch_cont_array = np.full((len(pitch_seqs), n_cont), np.nan, dtype=np.float32)
        for i, col in enumerate(PITCH_CONTINUOUS_COLS):
            if col in ("pre_on_first", "pre_on_second", "pre_on_third"):
                id_col = col + "_id"
                if id_col in pitch_seqs.columns:
                    vals = pd.to_numeric(pitch_seqs[id_col], errors="coerce").fillna(0.0).to_numpy()
                    self._pitch_cont_array[:, i] = (vals > 0).astype(np.float32)
            elif col in pitch_seqs.columns:
                vals = pd.to_numeric(pitch_seqs[col], errors="coerce").to_numpy()
                self._pitch_cont_array[:, i] = vals
            elif col in _binary_cols:
                self._pitch_cont_array[:, i] = 0.0

        # Observability mask: 1.0 = value was measured, 0.0 = missing/NaN.
        # Built BEFORE standardization so NaN positions are correctly identified.
        self._pitch_obs_mask = np.isfinite(self._pitch_cont_array).astype(np.float32)

        for i, col in enumerate(PITCH_CONTINUOUS_COLS):
            if col not in _binary_cols and col in standardizer.mean:
                self._pitch_cont_array[:, i] = (
                    (self._pitch_cont_array[:, i] - standardizer.mean[col])
                    / standardizer.std[col]
                )
        # NaN positions (from missing data) become 0.0 AFTER standardization,
        # which is the neutral z-score (equivalent to mean-imputation).
        np.nan_to_num(self._pitch_cont_array, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Pre-compute all per-pitch categorical arrays so _get_live_prefix uses
        # only O(1) numpy slices instead of per-sample pandas column operations.
        _n = len(pitch_seqs)
        _nb = self.spec.hash_bucket_count

        def _vec_hash_col(col_name: str) -> np.ndarray:
            if col_name not in pitch_seqs.columns:
                return np.zeros(_n, dtype=np.int64)
            series = _category_as_object(pitch_seqs[col_name])
            cache = {v: _hash_bucket(v, _nb) for v in series.dropna().unique()}
            mapped = series.map(cache)
            return pd.to_numeric(mapped, errors="coerce").fillna(0).to_numpy(dtype=np.int64)

        def _vec_vocab_col(col_name: str, vocab_map: dict, transform=None) -> np.ndarray:
            if col_name not in pitch_seqs.columns:
                return np.zeros(_n, dtype=np.int64)
            series = _category_as_object(pitch_seqs[col_name])
            unique = series.dropna().unique()
            if transform:
                cache = {v: vocab_map.get(transform(str(v)) if pd.notna(v) else "", 0) for v in unique}
            else:
                cache = {v: vocab_map.get(str(v) if pd.notna(v) else "", 0) for v in unique}
            mapped = series.map(cache)
            return pd.to_numeric(mapped, errors="coerce").fillna(0).to_numpy(dtype=np.int64)

        self._batter_hash_array = _vec_hash_col("batter_id")
        self._pitcher_hash_array = _vec_hash_col("pitcher_id")
        self._catcher_hash_array = _vec_hash_col("fielder_2")

        self._pitch_type_array = _vec_vocab_col(
            "pitch_type", PITCH_TYPE_TO_IDX, transform=lambda s: s.strip().upper())
        self._bat_side_array = _vec_vocab_col(
            "bat_side_code", BAT_SIDE_TO_IDX, transform=lambda s: s.strip().upper())
        _ph_col = "pitch_hand_code" if "pitch_hand_code" in pitch_seqs.columns else "p_throws"
        self._pitch_hand_array = _vec_vocab_col(
            _ph_col, PITCH_HAND_TO_IDX, transform=lambda s: s.strip().upper())

        if "is_top_inning" in pitch_seqs.columns:
            _top = pd.to_numeric(pitch_seqs["is_top_inning"], errors="coerce").fillna(1).to_numpy()
            self._half_inning_array = (1 - np.clip(_top.astype(np.int64), 0, 1)).astype(np.int64)
        else:
            self._half_inning_array = np.zeros(_n, dtype=np.int64)

        self._hit_trajectory_array = _vec_vocab_col(
            "bb_type", HIT_TRAJECTORY_TO_IDX,
            transform=lambda s: s.strip().lower().replace(" ", "_").replace("-", "_"))

        if "hit_hardness" in pitch_seqs.columns:
            self._hit_hardness_array = _vec_vocab_col(
                "hit_hardness", HIT_HARDNESS_TO_IDX, transform=lambda s: s.strip().lower())
        elif "launch_speed" in pitch_seqs.columns:
            _ls = pd.to_numeric(pitch_seqs["launch_speed"], errors="coerce").fillna(0.0).to_numpy()
            self._hit_hardness_array = np.zeros(_n, dtype=np.int64)
            _contact = _ls > 0
            self._hit_hardness_array[_contact & (_ls < 75)] = 1
            self._hit_hardness_array[_contact & (_ls >= 75) & (_ls <= 95)] = 2
            self._hit_hardness_array[_contact & (_ls > 95)] = 3
        else:
            self._hit_hardness_array = np.zeros(_n, dtype=np.int64)

        self._event_type_array = classify_event_type_series(pitch_seqs)

        _inning = np.clip(
            pd.to_numeric(pitch_seqs["inning"], errors="coerce").fillna(1).astype(int).to_numpy()
            if "inning" in pitch_seqs.columns else np.ones(_n, dtype=int), 0, 19)
        _ab_idx = np.clip(
            pd.to_numeric(pitch_seqs["at_bat_index"], errors="coerce").fillna(0).astype(int).to_numpy() % 25
            if "at_bat_index" in pitch_seqs.columns else np.zeros(_n, dtype=int), 0, 24)
        _pitch_num = np.clip(
            pd.to_numeric(pitch_seqs["pitch_number"], errors="coerce").fillna(1).astype(int).to_numpy()
            if "pitch_number" in pitch_seqs.columns else np.ones(_n, dtype=int), 0, 14)
        self._hierarchy_array = np.stack([_inning, _ab_idx, _pitch_num], axis=1).astype(np.int64)

        self._score_home_array = (
            pd.to_numeric(pitch_seqs["score_home"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            if "score_home" in pitch_seqs.columns else np.zeros(_n, dtype=np.float32))
        self._score_away_array = (
            pd.to_numeric(pitch_seqs["score_away"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            if "score_away" in pitch_seqs.columns else np.zeros(_n, dtype=np.float32))

        # Pre-compute arrays needed by _infer_facing_pitcher and _compute_matchup_summary
        # so we can delete self._pitches and reclaim ~4GB.
        self._batter_id_array = (
            pd.to_numeric(pitch_seqs["batter_id"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
            if "batter_id" in pitch_seqs.columns else np.zeros(_n, dtype=np.int64))
        self._is_top_array = (
            pd.to_numeric(pitch_seqs["is_top_inning"], errors="coerce").fillna(-1).to_numpy(dtype=np.int8)
            if "is_top_inning" in pitch_seqs.columns else np.full(_n, -1, dtype=np.int8))
        self._at_bat_index_array = (
            pd.to_numeric(pitch_seqs["at_bat_index"], errors="coerce").fillna(-1).to_numpy(dtype=np.int32)
            if "at_bat_index" in pitch_seqs.columns else np.full(_n, -1, dtype=np.int32))
        # at_bat_event: store as category codes for memory efficiency
        if "at_bat_event" in pitch_seqs.columns:
            _events_lower = (
                _category_as_object(pitch_seqs["at_bat_event"])
                .fillna("")
                .astype(str)
                .str.lower()
                .to_numpy(dtype=object)
            )
            self._at_bat_event_array = _events_lower
        else:
            self._at_bat_event_array = np.full(_n, "", dtype=object)

        # --- Player batting history (for player context) ---
        player_batting_history = player_batting_history.copy()
        player_batting_history["game_date"] = pd.to_datetime(
            player_batting_history["game_date"], errors="coerce"
        )
        self._player_history = player_batting_history
        # Build player_id -> sorted history
        self._player_hist_by_id: dict[int, pd.DataFrame] = {}
        if "player_id" in player_batting_history.columns:
            for pid, grp in player_batting_history.groupby("player_id", sort=False):
                self._player_hist_by_id[int(pid)] = grp.sort_values("game_date").reset_index(drop=True)

        # Pre-compute player history as numpy arrays for O(log n) date search
        # and O(1) per-game stat lookup — eliminates per-sample pandas indexing.
        from .feature_store import BATTING_SUM_COLUMNS, BATTING_SEASON_COLUMNS

        self._player_hist_dates: dict[int, np.ndarray] = {}
        self._player_hist_stat_arrays: dict[int, np.ndarray] = {}
        for _pid, _hdf in self._player_hist_by_id.items():
            self._player_hist_dates[_pid] = _hdf["game_date"].to_numpy()
            _hn = len(_hdf)
            _harr = np.zeros((_hn, PLAYER_STAT_DIM), dtype=np.float32)
            for _ci, _col in enumerate(BATTING_SUM_COLUMNS):
                if _col in _hdf.columns:
                    _harr[:, _ci] = pd.to_numeric(_hdf[_col], errors="coerce").fillna(0.0).to_numpy()
            for _ci, _col in enumerate(BATTING_SEASON_COLUMNS):
                if _col in _hdf.columns:
                    _harr[:, len(BATTING_SUM_COLUMNS) + _ci] = pd.to_numeric(_hdf[_col], errors="coerce").fillna(0.0).to_numpy()
            self._player_hist_stat_arrays[_pid] = _harr

        # Pre-compute (player_id, game_pk) → stat dict for O(1) target/mask lookup.
        self._player_game_stats: dict[tuple[int, int], dict] = {}
        if "player_id" in player_batting_history.columns and "game_pk" in player_batting_history.columns:
            _scols = {"hits": "game_hits", "hr": "game_hr", "so": "game_so",
                      "hrbi": "game_hits_runs_rbi", "tb": "game_total_bases", "sb": "game_sb"}
            _garrs: dict[str, np.ndarray] = {
                _k: (pd.to_numeric(player_batting_history[_c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                     if _c in player_batting_history.columns else np.zeros(len(player_batting_history), dtype=np.float32))
                for _k, _c in _scols.items()
            }
            _statuses = (_category_as_object(player_batting_history["target_status"]).fillna("trainable").to_numpy()
                         if "target_status" in player_batting_history.columns
                         else np.full(len(player_batting_history), "trainable", dtype=object))
            _pids_arr = player_batting_history["player_id"].to_numpy()
            _gpks_arr = player_batting_history["game_pk"].to_numpy()
            for _i in range(len(player_batting_history)):
                if pd.notna(_pids_arr[_i]) and pd.notna(_gpks_arr[_i]):
                    _key = (int(_pids_arr[_i]), int(_gpks_arr[_i]))
                    self._player_game_stats[_key] = {_k: float(_garrs[_k][_i]) for _k in _scols}
                    self._player_game_stats[_key]["target_status"] = str(_statuses[_i])

        # --- Build per-game lineup sets (for Jaccard overlap) ---
        self._game_lineups: dict[int, set[int]] = {}
        self._build_game_lineups()

        # --- Build SP -> game mapping from pitch data ---
        self._sp_games: dict[int, list[int]] = {}  # pitcher_id -> [game_pk...] chronological
        self._build_sp_game_mapping()

        # Release raw DataFrame — all needed data is now in pre-computed numpy arrays.
        del self._pitches
        import gc as _gc
        _gc.collect()

        # --- Matchup index: (batter_id, pitcher_id) -> list of PA pitch slices ---
        # Built lazily to save memory
        self._matchup_cache: dict[tuple[int, int], list[tuple[int, int]]] = {}

        # --- Rating temporal sequences: (game_pk, side) -> [K, N_RATINGS] ---
        from .rating_sequences import RATING_SEQ_STEPS
        self._rating_by_game_side: dict[tuple[int, str], np.ndarray] = {}
        self._rating_dim: int = 0
        self._rating_steps: int = RATING_SEQ_STEPS
        if game_features is not None and isinstance(game_features, dict):
            self._rating_by_game_side = game_features
            if game_features:
                sample = next(iter(game_features.values()))
                self._rating_dim = sample.shape[1] if sample.ndim == 2 else 0
                self._rating_steps = sample.shape[0] if sample.ndim == 2 else RATING_SEQ_STEPS

        # --- Weather features indexed by game_pk (legacy, kept for backward compat) ---
        self._weather_by_pk: dict[int, pd.Series] = {}
        if weather_features is not None and not weather_features.empty:
            for _, row in weather_features.iterrows():
                gpk = int(row["game_pk"])
                if gpk in self.target_by_game:
                    self._weather_by_pk[gpk] = row

        # --- Weather temporal: [4, 22] per game_pk (from weather_context.py) ---
        from .weather_context import WEATHER_TOKEN_DIM, WEATHER_TEMPORAL_HOURS, WEATHER_TEMPORAL_COLUMNS
        self._weather_temporal_by_pk: dict[int, np.ndarray] = {}
        if weather_temporal is not None and not weather_temporal.empty:
            for gpk, grp in weather_temporal.groupby("game_pk"):
                gpk = int(gpk)
                if gpk not in self.target_by_game:
                    continue
                grp_sorted = grp.sort_values("hour_offset")
                feat_cols = [c for c in WEATHER_TEMPORAL_COLUMNS if c in grp_sorted.columns]
                arr = grp_sorted[feat_cols].to_numpy(dtype=np.float32, na_value=0.0)
                # Pad or truncate to exactly WEATHER_TEMPORAL_HOURS rows
                if arr.shape[0] < WEATHER_TEMPORAL_HOURS:
                    pad = np.zeros((WEATHER_TEMPORAL_HOURS - arr.shape[0], arr.shape[1]), dtype=np.float32)
                    arr = np.vstack([arr, pad])
                elif arr.shape[0] > WEATHER_TEMPORAL_HOURS:
                    arr = arr[:WEATHER_TEMPORAL_HOURS]
                self._weather_temporal_by_pk[gpk] = arr

        # --- As-of weather: game_pk -> [7, 7, 99] (standardized upstream) and
        # game_pk -> int8 per-pitch decision-hour offsets. When present, the
        # "weather_temporal" batch key becomes the [7, 99] decision row for the
        # sample's cut pitch instead of the legacy [4, 22] snapshot.
        self._weather_asof_by_pk: dict[int, np.ndarray] = {
            int(k): v for k, v in (weather_asof or {}).items()
            if int(k) in self.target_by_game
        }
        self._wx_offsets_by_pk: dict[int, np.ndarray] = {
            int(k): v for k, v in (wx_hour_offsets or {}).items()
            if int(k) in self.target_by_game
        }

        # --- Venue dimensions indexed by venue_id ---
        self._venue_dims_by_id: dict[int, pd.Series] = {}
        if venue_dimensions is not None and not venue_dimensions.empty:
            for _, row in venue_dimensions.iterrows():
                self._venue_dims_by_id[int(row["venue_id"])] = row

        # --- Daily stats (SP quality) indexed by game_pk ---
        self._daily_stats_by_pk: dict[int, pd.Series] = {}
        if daily_stats is not None and not daily_stats.empty:
            for _, row in daily_stats.iterrows():
                gpk = int(row["game_pk"])
                if gpk in self.target_by_game:
                    self._daily_stats_by_pk[gpk] = row

        # --- Generate samples: (game_pk, prefix_pitch_count) ---
        self.samples: list[tuple[int, int]] = []
        self._build_samples()

        # --- Compute sample-level decay weights ---
        self.max_date = targets["game_date"].max()
        self._sample_game_weights: dict[int, float] = {}
        self._compute_game_weights(targets)

    def _build_game_lineups(self):
        """Extract lineup (set of batter_ids) for each game from pre-computed array."""
        for gpk, (start, end) in self._game_offsets.items():
            _slice = self._batter_id_array[start:end]
            self._game_lineups[gpk] = set(int(b) for b in _slice[_slice > 0])

    def _build_sp_game_mapping(self):
        """Build starting pitcher -> games mapping from game_meta probable pitchers."""
        meta = self.game_meta
        if "probable_pitcher_home_id" not in meta.columns:
            return

        sorted_meta = meta.sort_values("game_date")
        for _, row in sorted_meta.iterrows():
            gpk = int(row["game_pk"])
            for col in ["probable_pitcher_home_id", "probable_pitcher_away_id"]:
                if col in meta.columns and pd.notna(row.get(col)):
                    pid = int(row[col])
                    if pid not in self._sp_games:
                        self._sp_games[pid] = []
                    self._sp_games[pid].append(gpk)

    def _build_samples(self):
        """Generate (game_pk, prefix_length) samples.

        T=0 for pregame; T sampled at stride=25, max 32 prefixes per game for live.
        """
        stride = self.spec.live_stride
        max_prefixes = self.spec.live_max_prefixes_per_game

        for gpk in self.target_by_game:
            if gpk not in self._game_offsets:
                continue

            # Pregame sample (prefix=0)
            if self.include_pregame:
                self.samples.append((gpk, 0))

            # Live prefix samples
            if self.include_live:
                start, end = self._game_offsets[gpk]
                n_pitches = end - start
                if n_pitches == 0:
                    continue

                positions = list(range(stride, n_pitches, stride))
                final = n_pitches - 1
                if positions and positions[-1] != final:
                    # Append the final pitch so the end-of-game state is always sampled.
                    positions.append(final)
                elif not positions and n_pitches >= stride:
                    # Games with exactly stride pitches produce an empty range() — add final.
                    # Games shorter than stride have no live prefix (only the pregame sample).
                    positions = [final]

                if len(positions) > max_prefixes:
                    step = max(len(positions) // max_prefixes, 1)
                    positions = positions[::step][:max_prefixes]

                for end_pos in positions:
                    self.samples.append((gpk, end_pos))

    def _compute_game_weights(self, targets: pd.DataFrame):
        """Compute per-game sample weights using game-index decay from present."""
        # Use the team game index structure from datasets.py
        team_game_idx = build_team_game_index(self.team_games)

        for gpk in self.target_by_game:
            target = self.target_by_game[gpk]
            game_date = target.game_date
            meta = self.meta_by_game.get(gpk)
            if meta is None:
                self._sample_game_weights[gpk] = 1.0
                continue

            # Average decay of home and away teams
            home_tid = int(meta["home_team_id"]) if pd.notna(meta.get("home_team_id")) else None
            away_tid = int(meta["away_team_id"]) if pd.notna(meta.get("away_team_id")) else None

            weights = []
            for tid in [home_tid, away_tid]:
                if tid is None:
                    weights.append(1.0)
                    continue
                entry = team_game_idx.get(tid)
                if entry is None:
                    weights.append(1.0)
                    continue
                w_list = compute_game_decay_weight(entry, game_date, self.spec)
                # Last weight is most recent prior game
                weights.append(w_list[-1] if w_list else 1.0)

            self._sample_game_weights[gpk] = float(np.mean(weights))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        game_pk, prefix_len = self.samples[idx]
        target = self.target_by_game[game_pk]
        meta = self.meta_by_game.get(game_pk)

        # --- A. Historical context ---
        sp_home_ctx = self._get_sp_context(meta, side="home", game_pk=game_pk)
        sp_away_ctx = self._get_sp_context(meta, side="away", game_pk=game_pk)
        team_home_ctx = self._get_team_context(meta, side="home", game_pk=game_pk)
        team_away_ctx = self._get_team_context(meta, side="away", game_pk=game_pk)
        flat_features = self._get_flat_features(meta)

        # --- B. Live pitch prefix ---
        prefix_data = self._get_live_prefix(game_pk, prefix_len)

        # --- C. Player context ---
        player_ctx = self._get_player_context(game_pk, meta)

        # --- D. Targets ---
        targets_dict = self._build_targets(target, game_pk, prefix_len)

        # --- E. Masks ---
        masks = self._build_masks(target, game_pk, prefix_len, player_ctx)

        # --- F. Sample weight ---
        sample_weight = self._sample_game_weights.get(game_pk, 1.0)

        return {
            # Historical SP context
            "sp_home_seqs": sp_home_ctx["sequences"],
            "sp_home_obs_mask": sp_home_ctx["obs_mask"],
            "sp_home_lengths": sp_home_ctx["lengths"],
            "sp_home_weights": sp_home_ctx["weights"],
            "sp_home_mask": sp_home_ctx["mask"],
            # Historical SP away
            "sp_away_seqs": sp_away_ctx["sequences"],
            "sp_away_obs_mask": sp_away_ctx["obs_mask"],
            "sp_away_lengths": sp_away_ctx["lengths"],
            "sp_away_weights": sp_away_ctx["weights"],
            "sp_away_mask": sp_away_ctx["mask"],
            # Team offense history
            "team_home_seqs": team_home_ctx["sequences"],
            "team_home_obs_mask": team_home_ctx["obs_mask"],
            "team_home_lengths": team_home_ctx["lengths"],
            "team_home_weights": team_home_ctx["weights"],
            "team_home_mask": team_home_ctx["mask"],
            "team_home_similarity": team_home_ctx["similarity"],
            "team_away_seqs": team_away_ctx["sequences"],
            "team_away_obs_mask": team_away_ctx["obs_mask"],
            "team_away_lengths": team_away_ctx["lengths"],
            "team_away_weights": team_away_ctx["weights"],
            "team_away_mask": team_away_ctx["mask"],
            "team_away_similarity": team_away_ctx["similarity"],
            # Flat context
            "flat_features": flat_features,
            # Weather: as-of [7, 99] decision row when the artifact is loaded,
            # legacy [4, 22] snapshot otherwise (single-keyed interface — the
            # model's weather_dim/weather_tokens config must match the source).
            "weather_temporal": (self._get_weather_asof_row(game_pk, prefix_len)
                                 if self._weather_asof_by_pk
                                 else self._get_weather_temporal(game_pk)),
            # Rating temporal context [K, N_RATINGS] per side
            "rating_home": self._get_rating_temporal(game_pk, "home"),
            "rating_away": self._get_rating_temporal(game_pk, "away"),
            # Live prefix
            "prefix_values": prefix_data["values"],
            "prefix_obs_mask": prefix_data["obs_mask"],
            "prefix_mask": prefix_data["mask"],
            "prefix_batter_hash": prefix_data["batter_hash"],
            "prefix_pitcher_hash": prefix_data["pitcher_hash"],
            "prefix_catcher_hash": prefix_data["catcher_hash"],
            "prefix_event_type": prefix_data["event_type"],
            "prefix_hierarchy": prefix_data["hierarchy"],
            "prefix_pitch_type_idx": prefix_data["pitch_type_idx"],
            "prefix_bat_side_idx": prefix_data["bat_side_idx"],
            "prefix_pitch_hand_idx": prefix_data["pitch_hand_idx"],
            "prefix_half_inning_idx": prefix_data["half_inning_idx"],
            "prefix_hit_trajectory_idx": prefix_data["hit_trajectory_idx"],
            "prefix_hit_hardness_idx": prefix_data["hit_hardness_idx"],
            "prefix_length": torch.tensor(prefix_len, dtype=torch.long),
            # Player context
            "player_hashes": player_ctx["hashes"],
            "player_history": player_ctx["history"],
            "player_history_mask": player_ctx["history_mask"],
            "player_matchup": player_ctx["matchup"],
            # Targets
            "targets": targets_dict,
            # Masks
            "yrfi_mask": masks["yrfi_mask"],
            "player_mask": masks["player_mask"],
            # Metadata
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
            "game_pk": torch.tensor(game_pk, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # A. Historical context builders
    # ------------------------------------------------------------------

    def _get_sp_context(self, meta: Optional[pd.Series], side: str, game_pk: int) -> dict:
        """Get prior starts for a starting pitcher.

        Returns pitch sequences from the SP's last N starts before this game.
        """
        n_games = self.ablation.sp_history_games
        empty = self._empty_history_context(n_games)

        if meta is None:
            return empty

        col = f"probable_pitcher_{side}_id"
        if col not in meta.index or pd.isna(meta[col]):
            return empty

        pitcher_id = int(meta[col])
        game_date = meta["game_date"]

        # Find prior starts by this pitcher
        prior_games = self._sp_games.get(pitcher_id, [])
        # Filter to games before current game_date
        prior_gpks = []
        for gpk in prior_games:
            if gpk == game_pk:
                continue
            gm_date = self._game_date_by_pk.get(gpk)
            if gm_date is not None and gm_date < game_date:
                prior_gpks.append(gpk)

        # Take most recent N
        prior_gpks = prior_gpks[-n_games:]

        if not prior_gpks:
            return empty

        return self._extract_game_sequences(prior_gpks, game_pk, n_games)

    def _get_team_context(self, meta: Optional[pd.Series], side: str, game_pk: int) -> dict:
        """Get prior offensive game sequences for a team.

        Ablation modes control which games are selected.
        """
        n_games = self.ablation.team_history_games
        empty = self._empty_history_context(n_games)
        empty["similarity"] = torch.zeros(n_games, dtype=torch.float32)

        if meta is None:
            return empty

        team_col = f"{side}_team_id"
        if team_col not in meta.index or pd.isna(meta[team_col]):
            return empty

        team_id = int(meta[team_col])
        game_date = meta["game_date"]

        # Get team's games before this date
        team_df = self.game_index["by_team"].get(team_id)
        if team_df is None or team_df.empty:
            return empty

        # Include same-date games with a smaller game_pk (doubleheader game 1 is
        # available as context for game 2; game_pk ordering is chronological within a date).
        prior_mask = (
            (team_df["game_date"] < game_date)
            | ((team_df["game_date"] == game_date) & (team_df["game_pk"] < game_pk))
        )
        prior_df = team_df[prior_mask]

        if prior_df.empty:
            return empty

        # Apply ablation mode
        mode = self.ablation.team_context_mode
        today_lineup = self._game_lineups.get(game_pk, set())

        if mode == "lineup_overlap" and today_lineup:
            selected, similarities = self._select_lineup_overlap(
                prior_df, today_lineup, n_games
            )
        elif mode == "similarity_weighted" and today_lineup:
            selected = prior_df.tail(n_games)
            similarities = np.array([
                _compute_jaccard(today_lineup, self._game_lineups.get(int(row["game_pk"]), set()))
                for _, row in selected.iterrows()
            ], dtype=np.float32)
        else:  # "all_games"
            selected = prior_df.tail(n_games)
            similarities = np.ones(len(selected), dtype=np.float32)

        prior_gpks = selected["game_pk"].astype(int).tolist()
        result = self._extract_game_sequences(prior_gpks, game_pk, n_games)

        # Pad similarity to n_games
        sim_padded = np.zeros(n_games, dtype=np.float32)
        sim_padded[-len(similarities):] = similarities[-n_games:]
        result["similarity"] = torch.from_numpy(sim_padded)

        return result

    def _select_lineup_overlap(
        self, prior_df: pd.DataFrame, today_lineup: set, n_games: int
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Select games with >= threshold lineup overlap, relaxing if needed."""
        threshold = self.ablation.lineup_overlap_threshold

        while threshold >= 1:
            mask = []
            sims = []
            for _, row in prior_df.iterrows():
                gpk = int(row["game_pk"])
                game_lineup = self._game_lineups.get(gpk, set())
                overlap = len(today_lineup & game_lineup)
                if overlap >= threshold:
                    mask.append(True)
                    sims.append(overlap / max(len(today_lineup | game_lineup), 1))
                else:
                    mask.append(False)
                    sims.append(0.0)

            filtered = prior_df[mask]
            filtered_sims = np.array([s for s, m in zip(sims, mask) if m], dtype=np.float32)

            if len(filtered) >= n_games:
                selected = filtered.tail(n_games)
                return selected, filtered_sims[-n_games:]
            elif len(filtered) >= 1:
                # Use what we have
                return filtered.tail(n_games), filtered_sims[-n_games:]

            threshold -= 1

        # Fallback: take most recent
        selected = prior_df.tail(n_games)
        sims_out = np.ones(len(selected), dtype=np.float32)
        return selected, sims_out

    def _extract_game_sequences(
        self, game_pks: list[int], target_game_pk: int, max_games: int
    ) -> dict:
        """Extract pitch sequences for a list of historical games.

        Returns padded tensors for variable-length game sequences.
        """
        n_continuous = len(PITCH_CONTINUOUS_COLS)
        sequences = []
        obs_masks = []
        lengths = []
        weights = []

        target_meta = self.meta_by_game.get(target_game_pk)
        target_date = target_meta["game_date"] if target_meta is not None else pd.Timestamp.now()
        target_season = int(target_meta.get("season", 2026)) if target_meta is not None else 2026

        for gpk in game_pks:
            seq_data, seq_obs = self._load_game_pitches(gpk, n_continuous)
            sequences.append(seq_data)
            obs_masks.append(seq_obs)
            lengths.append(seq_data.shape[0])

            # Compute decay weight for this historical game
            hist_meta = self.meta_by_game.get(gpk)
            if hist_meta is not None:
                hist_season = int(hist_meta.get("season", target_season))
                # Use date-based ordering as proxy for game index distance
                days_ago = max((target_date - hist_meta["game_date"]).days, 0)
                # Approximate game index from days (162 games / 180 days ~ 0.9 games/day)
                games_ago = int(days_ago * 0.9)
                seasons_crossed = max(target_season - hist_season, 0)
                w = math.exp(-self.spec.intra_season_lambda * games_ago) * \
                    math.exp(-self.spec.inter_season_lambda * seasons_crossed)
            else:
                w = 0.5
            weights.append(w)

        # Pad to max_games with empty sequences
        pad_count = max_games - len(sequences)
        for _ in range(pad_count):
            sequences.append(np.zeros((0, n_continuous), dtype=np.float32))
            obs_masks.append(np.zeros((0, n_continuous), dtype=np.float32))
            lengths.append(0)
            weights.append(0.0)

        # Find max sequence length for padding
        max_len = max(s.shape[0] for s in sequences) if sequences else 1
        max_len = max(max_len, 1)  # Ensure at least 1

        # Left-pad each sequence to max_len
        padded = np.zeros((max_games, max_len, n_continuous), dtype=np.float32)
        padded_obs = np.zeros((max_games, max_len, n_continuous), dtype=np.float32)
        mask = np.zeros((max_games, max_len), dtype=np.float32)

        for i, (seq, obs) in enumerate(zip(sequences[:max_games], obs_masks[:max_games])):
            if seq.shape[0] > 0:
                # Left-pad: put actual data at the end
                start_idx = max_len - seq.shape[0]
                padded[i, start_idx:, :] = seq[-max_len:]
                padded_obs[i, start_idx:, :] = obs[-max_len:]
                mask[i, start_idx:] = 1.0

        return {
            "sequences": torch.from_numpy(padded),
            "obs_mask": torch.from_numpy(padded_obs),
            "lengths": torch.tensor(lengths[:max_games], dtype=torch.long),
            "weights": torch.tensor(weights[:max_games], dtype=torch.float32),
            "mask": torch.from_numpy(mask),
        }

    def _load_game_pitches(self, game_pk: int, n_features: int) -> tuple[np.ndarray, np.ndarray]:
        """Load pitch-level continuous features and obs mask for a single game.

        Returns (values, obs_mask) tuple of (n_pitches, n_features) float32 arrays.
        """
        offsets = self._game_offsets.get(game_pk)
        if offsets is None:
            return np.zeros((0, n_features), dtype=np.float32), np.zeros((0, n_features), dtype=np.float32)
        start, end = offsets
        return self._pitch_cont_array[start:end].copy(), self._pitch_obs_mask[start:end].copy()

    def _empty_history_context(self, n_games: int) -> dict:
        """Return zero-filled history context tensors."""
        n_continuous = len(PITCH_CONTINUOUS_COLS)
        return {
            "sequences": torch.zeros(n_games, 1, n_continuous, dtype=torch.float32),
            "obs_mask": torch.zeros(n_games, 1, n_continuous, dtype=torch.float32),
            "lengths": torch.zeros(n_games, dtype=torch.long),
            "weights": torch.zeros(n_games, dtype=torch.float32),
            "mask": torch.zeros(n_games, 1, dtype=torch.float32),
        }

    # ------------------------------------------------------------------
    # Weather temporal context
    # ------------------------------------------------------------------

    def _get_weather_temporal(self, game_pk: int) -> torch.Tensor:
        """Return [4, 22] weather tensor for the 4-hour game window."""
        from .weather_context import WEATHER_TOKEN_DIM, WEATHER_TEMPORAL_HOURS
        data = self._weather_temporal_by_pk.get(game_pk)
        if data is not None:
            return torch.from_numpy(data)
        return torch.zeros(WEATHER_TEMPORAL_HOURS, WEATHER_TOKEN_DIM, dtype=torch.float32)

    def _get_wx_decision_hour(self, game_pk: int, prefix_len: int) -> int:
        """Decision hour d for a sample = the cut pitch's elapsed-hour offset.

        prefix_len == 0 is the pregame sample -> d=0 by definition. A game with
        no offsets (missing raw pitch_start_time artifact) also falls to 0 —
        the only decision row that can never look ahead."""
        if prefix_len <= 0:
            return 0
        offs = self._wx_offsets_by_pk.get(game_pk)
        if offs is None or len(offs) == 0:
            return 0
        i = min(prefix_len - 1, len(offs) - 1)
        return int(np.clip(offs[i], 0, 6))

    def _get_weather_asof_full(self, game_pk: int) -> np.ndarray:
        """[7, 7, 99] as-of tensor; zeros (= fully masked/unknown) when absent."""
        from .weather_asof import ASOF_CHANNELS, N_DECISIONS, N_TARGET_HOURS
        data = self._weather_asof_by_pk.get(game_pk)
        if data is not None:
            return data
        return np.zeros((N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS), dtype=np.float32)

    def _get_weather_asof_row(self, game_pk: int, prefix_len: int) -> torch.Tensor:
        """[7, 99] decision row for the sample's cut pitch."""
        d = self._get_wx_decision_hour(game_pk, prefix_len)
        return torch.from_numpy(np.ascontiguousarray(self._get_weather_asof_full(game_pk)[d]))

    # ------------------------------------------------------------------
    # Rating temporal context
    # ------------------------------------------------------------------

    def _get_rating_temporal(self, game_pk: int, side: str) -> torch.Tensor:
        """Return [K, N_RATINGS] rating history tensor for one side."""
        data = self._rating_by_game_side.get((game_pk, side))
        if data is not None:
            return torch.from_numpy(np.nan_to_num(data, nan=0.0, copy=True))
        return torch.zeros(self._rating_steps, max(self._rating_dim, 1), dtype=torch.float32)

    # ------------------------------------------------------------------
    # Flat features
    # ------------------------------------------------------------------

    def _get_flat_features(self, meta: Optional[pd.Series]) -> torch.Tensor:
        """Extract 30-dim flat game-context vector from pre-game metadata.

        Only features unlearnable from sequential inputs (ratings, weather temporal,
        pitch tokens, player history). See FLAT_FEATURE_DIM layout comment.
        """
        import re as _re

        flat = np.zeros(FLAT_FEATURE_DIM, dtype=np.float32)

        if meta is None:
            return torch.from_numpy(flat)

        def _safe_float(val, default=0.0):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        def _has(col):
            return col in meta.index and pd.notna(meta.get(col))

        # --- [0] Venue ID hash (normalized to [0, 1]) ---
        if _has("venue_id"):
            flat[0] = float(int(meta["venue_id"]) % 1000) / 1000.0

        # --- [1-2] Venue lat/lon ---
        if _has("venue_latitude"):
            flat[1] = (_safe_float(meta["venue_latitude"]) - 38.0) / 10.0
        if _has("venue_longitude"):
            flat[2] = (_safe_float(meta["venue_longitude"]) + 95.0) / 25.0

        # --- [3] Venue capacity (standardized) ---
        if _has("venue_capacity"):
            flat[3] = (_safe_float(meta["venue_capacity"]) - 42000.0) / 8000.0

        # --- [4] Venue surface (turf=1, grass=0) ---
        if _has("venue_surface"):
            flat[4] = 1.0 if "turf" in str(meta["venue_surface"]).lower() else 0.0

        # --- [5-7] Venue roof one-hot (open, dome, retractable) ---
        if _has("venue_roof_type"):
            roof = str(meta["venue_roof_type"]).lower()
            if "open" in roof:
                flat[5] = 1.0
            elif "dome" in roof and "retract" not in roof:
                flat[6] = 1.0
            elif "retract" in roof:
                flat[7] = 1.0

        # --- [8-13] Venue dimensions (from venue_dimensions parquet) ---
        venue_id = int(meta["venue_id"]) if _has("venue_id") else None
        vdim = self._venue_dims_by_id.get(venue_id) if venue_id else None
        if vdim is not None:
            def _vd_float(col, default=0.0):
                val = vdim.get(col)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return default
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default

            flat[8] = (_vd_float("lf_line", 330.0) - 330.0) / 15.0
            flat[9] = (_vd_float("cf_center", 400.0) - 400.0) / 15.0
            flat[10] = (_vd_float("rf_line", 330.0) - 330.0) / 15.0
            flat[11] = _vd_float("lf_wall_height", 8.0) / 20.0
            flat[12] = _vd_float("cf_wall_height", 8.0) / 20.0
            flat[13] = _vd_float("rf_wall_height", 8.0) / 20.0

        # --- [14] Umpire HP hash ---
        if _has("umpire_hp"):
            ump_str = str(meta["umpire_hp"])
            ump_hash = int(hashlib.blake2b(ump_str.encode(), digest_size=4).hexdigest(), 16)
            flat[14] = float(ump_hash % 1000) / 1000.0

        # --- [15-16] Probable pitcher hashes ---
        if _has("probable_pitcher_home_id"):
            flat[15] = float(int(meta["probable_pitcher_home_id"]) % 10000) / 10000.0
        if _has("probable_pitcher_away_id"):
            flat[16] = float(int(meta["probable_pitcher_away_id"]) % 10000) / 10000.0

        # --- [17-18] Start hour sin/cos ---
        start_hour = None
        if _has("start_time"):
            try:
                st = str(meta["start_time"])
                time_match = _re.search(r"(\d{1,2}):(\d{2})", st)
                if time_match:
                    start_hour = int(time_match.group(1)) + int(time_match.group(2)) / 60.0
            except (ValueError, TypeError):
                pass
        if _has("game_datetime_utc") and start_hour is None:
            try:
                dt = pd.Timestamp(meta["game_datetime_utc"])
                if pd.notna(dt):
                    start_hour = (dt.hour + dt.minute / 60.0 - 4.0) % 24.0
            except (ValueError, TypeError):
                pass
        if start_hour is not None:
            rad = 2.0 * math.pi * start_hour / 24.0
            flat[17] = math.sin(rad)
            flat[18] = math.cos(rad)

        # --- [19-20] Day of week sin/cos ---
        if _has("game_date"):
            try:
                gd = pd.Timestamp(meta["game_date"])
                if pd.notna(gd):
                    dow = gd.dayofweek
                    rad = 2.0 * math.pi * dow / 7.0
                    flat[19] = math.sin(rad)
                    flat[20] = math.cos(rad)
            except (ValueError, TypeError):
                pass

        # --- [21-22] Month sin/cos ---
        if _has("game_date"):
            try:
                gd = pd.Timestamp(meta["game_date"])
                if pd.notna(gd):
                    rad = 2.0 * math.pi * (gd.month - 1) / 12.0
                    flat[21] = math.sin(rad)
                    flat[22] = math.cos(rad)
            except (ValueError, TypeError):
                pass

        # --- [23] Day/night binary ---
        if _has("day_night"):
            flat[23] = 1.0 if str(meta["day_night"]).lower() == "night" else 0.0

        # --- [24-26] Scheduling flags ---
        if _has("game_number"):
            flat[24] = 1.0 if int(meta["game_number"]) >= 2 else 0.0
        if _has("double_header"):
            dh_val = meta["double_header"]
            flat[25] = 1.0 if (str(dh_val).lower() in ("y", "s", "true", "1") or dh_val == 1) else 0.0
        if _has("tiebreaker"):
            tb_val = meta["tiebreaker"]
            flat[26] = 1.0 if (str(tb_val).lower() in ("y", "true", "1") or tb_val == 1) else 0.0

        # --- [27-29] Regime flags ---
        if _has("rule_3batter_minimum"):
            flat[27] = _safe_float(meta["rule_3batter_minimum"])
        if _has("rule_universal_dh"):
            flat[28] = _safe_float(meta["rule_universal_dh"])
        if _has("rule_shift_ban_pitch_clock"):
            flat[29] = _safe_float(meta["rule_shift_ban_pitch_clock"])

        return torch.from_numpy(flat)

    # ------------------------------------------------------------------
    # B. Live pitch prefix
    # ------------------------------------------------------------------

    def _get_live_prefix(self, game_pk: int, prefix_len: int) -> dict:
        """Extract the first prefix_len pitches of the current game.

        All features come from pre-computed arrays in __init__ — no pandas ops.
        """
        n_continuous = len(PITCH_CONTINUOUS_COLS)
        max_prefix = self.spec.history_length

        _empty = {
            "values": torch.zeros(max_prefix, n_continuous, dtype=torch.float32),
            "obs_mask": torch.zeros(max_prefix, n_continuous, dtype=torch.float32),
            "mask": torch.zeros(max_prefix, dtype=torch.float32),
            "batter_hash": torch.zeros(max_prefix, dtype=torch.long),
            "pitcher_hash": torch.zeros(max_prefix, dtype=torch.long),
            "catcher_hash": torch.zeros(max_prefix, dtype=torch.long),
            "event_type": torch.zeros(max_prefix, dtype=torch.long),
            "hierarchy": torch.zeros(max_prefix, 3, dtype=torch.long),
            "pitch_type_idx": torch.zeros(max_prefix, dtype=torch.long),
            "bat_side_idx": torch.zeros(max_prefix, dtype=torch.long),
            "pitch_hand_idx": torch.zeros(max_prefix, dtype=torch.long),
            "half_inning_idx": torch.zeros(max_prefix, dtype=torch.long),
            "hit_trajectory_idx": torch.zeros(max_prefix, dtype=torch.long),
            "hit_hardness_idx": torch.zeros(max_prefix, dtype=torch.long),
        }
        if prefix_len == 0 or game_pk not in self._game_offsets:
            return _empty

        start, end = self._game_offsets[game_pk]
        actual_len = min(prefix_len, end - start)
        s, e = start, start + actual_len

        arr = self._pitch_cont_array[s:e].copy()
        obs_arr = self._pitch_obs_mask[s:e].copy()

        padded = np.zeros((max_prefix, n_continuous), dtype=np.float32)
        padded_obs = np.zeros((max_prefix, n_continuous), dtype=np.float32)
        mask_arr = np.zeros(max_prefix, dtype=np.float32)
        pad_start = max_prefix - actual_len
        if pad_start >= 0:
            padded[pad_start:] = arr
            padded_obs[pad_start:] = obs_arr
            mask_arr[pad_start:] = 1.0
        else:
            padded[:] = arr[-max_prefix:]
            padded_obs[:] = obs_arr[-max_prefix:]
            mask_arr[:] = 1.0

        def _fill1d(raw: np.ndarray) -> np.ndarray:
            out = np.zeros(max_prefix, dtype=raw.dtype)
            if pad_start >= 0:
                out[pad_start:] = raw
            else:
                out[:] = raw[-max_prefix:]
            return out

        def _fill2d(raw: np.ndarray) -> np.ndarray:
            out = np.zeros((max_prefix, raw.shape[1]), dtype=raw.dtype)
            if pad_start >= 0:
                out[pad_start:] = raw
            else:
                out[:] = raw[-max_prefix:]
            return out

        return {
            "values": torch.from_numpy(padded),
            "obs_mask": torch.from_numpy(padded_obs),
            "mask": torch.from_numpy(mask_arr),
            "batter_hash": torch.from_numpy(_fill1d(self._batter_hash_array[s:e])),
            "pitcher_hash": torch.from_numpy(_fill1d(self._pitcher_hash_array[s:e])),
            "catcher_hash": torch.from_numpy(_fill1d(self._catcher_hash_array[s:e])),
            "event_type": torch.from_numpy(_fill1d(self._event_type_array[s:e])),
            "hierarchy": torch.from_numpy(_fill2d(self._hierarchy_array[s:e])),
            "pitch_type_idx": torch.from_numpy(_fill1d(self._pitch_type_array[s:e])),
            "bat_side_idx": torch.from_numpy(_fill1d(self._bat_side_array[s:e])),
            "pitch_hand_idx": torch.from_numpy(_fill1d(self._pitch_hand_array[s:e])),
            "half_inning_idx": torch.from_numpy(_fill1d(self._half_inning_array[s:e])),
            "hit_trajectory_idx": torch.from_numpy(_fill1d(self._hit_trajectory_array[s:e])),
            "hit_hardness_idx": torch.from_numpy(_fill1d(self._hit_hardness_array[s:e])),
        }

    # ------------------------------------------------------------------
    # C. Player context
    # ------------------------------------------------------------------

    def _get_player_context(self, game_pk: int, meta: Optional[pd.Series]) -> dict:
        """Build per-player context for all players in today's game.

        Returns hashes, historical stat lines, and optional matchup data.
        """
        n_players = MAX_PLAYERS_PER_GAME
        n_history = self.ablation.player_history_games

        hashes = torch.zeros(n_players, dtype=torch.long)
        history = torch.zeros(n_players, n_history, PLAYER_STAT_DIM, dtype=torch.float32)
        history_mask = torch.zeros(n_players, n_history, dtype=torch.float32)

        # Matchup tensor shape depends on mode
        if self.ablation.matchup_mode == "compressed_summary":
            matchup = torch.zeros(n_players, MATCHUP_SUMMARY_DIM, dtype=torch.float32)
        else:
            matchup = torch.zeros(n_players, 1, dtype=torch.float32)  # placeholder

        if meta is None:
            return {"hashes": hashes, "history": history, "history_mask": history_mask, "matchup": matchup}

        game_date = meta["game_date"]

        # Get players from the game lineup
        lineup = self._game_lineups.get(game_pk, set())
        player_ids = sorted(lineup)[:n_players]

        # Get opposing SP for matchup
        home_sp = int(meta["probable_pitcher_home_id"]) if pd.notna(meta.get("probable_pitcher_home_id")) else None
        away_sp = int(meta["probable_pitcher_away_id"]) if pd.notna(meta.get("probable_pitcher_away_id")) else None

        for i, pid in enumerate(player_ids):
            if i >= n_players:
                break

            hashes[i] = _hash_bucket(pid, self.spec.hash_bucket_count)

            # Player history: O(log n) binary search into pre-computed arrays
            _dates = self._player_hist_dates.get(pid)
            _sarr = self._player_hist_stat_arrays.get(pid)
            if _dates is not None and _sarr is not None and len(_dates) > 0:
                _gd_np = np.datetime64(game_date)
                _idx = int(np.searchsorted(_dates, _gd_np, side="left"))
                _h_start = max(0, _idx - n_history)
                _prior = _sarr[_h_start:_idx]
                if len(_prior) > 0:
                    _n_actual = len(_prior)
                    _start_idx = n_history - _n_actual
                    history[i, _start_idx:, :_prior.shape[1]] = torch.from_numpy(_prior)
                    history_mask[i, _start_idx:] = 1.0

            # Matchup data (if enabled)
            if self.ablation.matchup_mode == "compressed_summary":
                # Determine opposing SP for this batter
                opposing_sp = self._get_opposing_sp(pid, game_pk, meta)
                if opposing_sp is not None:
                    matchup[i] = self._compute_matchup_summary(pid, opposing_sp, game_date)

        return {
            "hashes": hashes,
            "history": history,
            "history_mask": history_mask,
            "matchup": matchup,
        }

    def _get_opposing_sp(self, batter_id: int, game_pk: int, meta: pd.Series) -> Optional[int]:
        """Determine which SP a batter faces based on which side they bat for."""
        game_lineup_home = set()
        game_lineup_away = set()

        # Infer side from pre-computed arrays
        offsets = self._game_offsets.get(game_pk)
        if offsets is not None:
            start, end = offsets
            batter_slice = self._batter_id_array[start:end]
            mask = batter_slice == batter_id
            if mask.any():
                first_idx = np.argmax(mask)
                is_top = self._is_top_array[start + first_idx]
                if is_top >= 0:
                    if is_top == 1:
                        return int(meta["probable_pitcher_home_id"]) if pd.notna(meta.get("probable_pitcher_home_id")) else None
                    else:
                        return int(meta["probable_pitcher_away_id"]) if pd.notna(meta.get("probable_pitcher_away_id")) else None

        return None

    def _compute_matchup_summary(
        self, batter_id: int, pitcher_id: int, before_date: pd.Timestamp
    ) -> torch.Tensor:
        """Compute compressed matchup stats: [n_pa, avg, iso, k_rate, avg_vs_hand, woba_vs_hand, n_pa_bucket]."""
        summary = torch.zeros(MATCHUP_SUMMARY_DIM, dtype=torch.float32)

        # Find historical PAs of this batter vs this pitcher
        hist = self._player_hist_by_id.get(batter_id)
        if hist is None:
            return summary

        # Filter by pitcher faced (if we have that info in the pitch data)
        # Use pitch-level data for matchup
        offsets_list = []
        for gpk in self._sp_games.get(pitcher_id, []):
            gm = self.meta_by_game.get(gpk)
            if gm is not None and gm["game_date"] < before_date:
                offsets = self._game_offsets.get(gpk)
                if offsets is not None:
                    offsets_list.append((gpk, offsets))

        if not offsets_list:
            return summary

        # Collect PAs of this batter against this pitcher
        n_pa = 0
        hits = 0
        hr = 0
        so = 0
        ab = 0
        tb = 0

        for gpk, (start, end) in offsets_list[-self.ablation.matchup_max_pa:]:
            batter_slice = self._batter_id_array[start:end]
            batter_mask = batter_slice == batter_id
            if not batter_mask.any():
                continue

            ab_slice = self._at_bat_index_array[start:end][batter_mask]
            event_slice = self._at_bat_event_array[start:end][batter_mask]

            # Group by at_bat_index — find unique ABs and the last pitch of each
            unique_abs = np.unique(ab_slice[ab_slice >= 0])
            for ab_idx in unique_abs:
                ab_mask = ab_slice == ab_idx
                if not ab_mask.any():
                    continue
                last_idx = np.where(ab_mask)[0][-1]
                event = str(event_slice[last_idx])
                n_pa += 1
                ab += 1

                if event in ("single",):
                    hits += 1
                    tb += 1
                elif event in ("double",):
                    hits += 1
                    tb += 2
                elif event in ("triple",):
                    hits += 1
                    tb += 3
                elif event in ("home run", "home_run"):
                    hits += 1
                    hr += 1
                    tb += 4
                elif event in ("strikeout", "strikeout double play"):
                    so += 1

        if n_pa > 0:
            avg = hits / max(ab, 1)
            iso = (tb - hits) / max(ab, 1)
            k_rate = so / n_pa
            summary[0] = float(n_pa)
            summary[1] = avg
            summary[2] = iso
            summary[3] = k_rate
            summary[4] = avg  # avg_vs_hand (same as avg without hand split)
            summary[5] = avg * 1.2  # woba_vs_hand approximation
            summary[6] = min(float(n_pa) / 20.0, 1.0)  # n_pa_bucket normalized

        return summary

    def _extract_player_stats(self, prior_df: pd.DataFrame) -> np.ndarray:
        """Extract player stat lines as (n_games, PLAYER_STAT_DIM) array.

        First 17 dims: game-level batting stats (BATTING_SUM_COLUMNS).
        Last 8 dims: season cumulative stats (BATTING_SEASON_COLUMNS).
        """
        from .feature_store import BATTING_SUM_COLUMNS, BATTING_SEASON_COLUMNS

        n = len(prior_df)
        arr = np.zeros((n, PLAYER_STAT_DIM), dtype=np.float32)

        # Game-level stats (first 17)
        for i, col in enumerate(BATTING_SUM_COLUMNS):
            if col in prior_df.columns:
                arr[:, i] = pd.to_numeric(prior_df[col], errors="coerce").fillna(0.0).to_numpy()

        # Season cumulative stats (next 8)
        offset = len(BATTING_SUM_COLUMNS)
        for i, col in enumerate(BATTING_SEASON_COLUMNS):
            if col in prior_df.columns:
                arr[:, offset + i] = pd.to_numeric(prior_df[col], errors="coerce").fillna(0.0).to_numpy()

        return arr

    # ------------------------------------------------------------------
    # D. Targets
    # ------------------------------------------------------------------

    def _build_targets(self, target, game_pk: int, prefix_len: int) -> dict:
        """Build game-level and player-level target dict."""
        targets = {}

        def _safe_float(val, default=0.0):
            v = float(val) if val is not None else default
            return default if np.isnan(v) else v

        # Game-level targets
        targets["home_win"] = torch.tensor(_safe_float(getattr(target, "home_win", 0)), dtype=torch.float32)
        targets["yrfi"] = torch.tensor(_safe_float(getattr(target, "yrfi", 0)), dtype=torch.float32)
        targets["extra_innings"] = torch.tensor(_safe_float(getattr(target, "extra_innings", 0)), dtype=torch.float32)

        # Remaining runs: final - score_at_prefix
        final_home = _safe_float(getattr(target, "home_runs", 0))
        final_away = _safe_float(getattr(target, "away_runs", 0))

        observed_home = 0.0
        observed_away = 0.0
        if prefix_len > 0 and game_pk in self._game_offsets:
            _t_start, _ = self._game_offsets[game_pk]
            _pitch_idx = _t_start + prefix_len - 1
            if _pitch_idx < len(self._score_home_array):
                observed_home = float(self._score_home_array[_pitch_idx])
                observed_away = float(self._score_away_array[_pitch_idx])

        targets["home_runs_remaining"] = torch.tensor(
            max(0.0, final_home - observed_home), dtype=torch.float32
        )
        targets["away_runs_remaining"] = torch.tensor(
            max(0.0, final_away - observed_away), dtype=torch.float32
        )
        targets["total_runs"] = torch.tensor(
            _safe_float(getattr(target, "total_runs", 0)), dtype=torch.float32
        )

        # Player-level targets: per-batter stats for this game
        player_targets = self._build_player_targets(game_pk)
        targets["player_hits"] = player_targets["hits"]
        targets["player_hr"] = player_targets["hr"]
        targets["player_so"] = player_targets["so"]
        targets["player_hrbi"] = player_targets["hrbi"]
        targets["player_tb"] = player_targets["tb"]
        targets["player_sb"] = player_targets["sb"]

        return targets

    def _build_player_targets(self, game_pk: int) -> dict:
        """Build per-player target tensors (MAX_PLAYERS_PER_GAME,)."""
        n_players = MAX_PLAYERS_PER_GAME
        result = {
            "hits": torch.zeros(n_players, dtype=torch.float32),
            "hr": torch.zeros(n_players, dtype=torch.float32),
            "so": torch.zeros(n_players, dtype=torch.float32),
            "hrbi": torch.zeros(n_players, dtype=torch.float32),
            "tb": torch.zeros(n_players, dtype=torch.float32),
            "sb": torch.zeros(n_players, dtype=torch.float32),
        }

        lineup = sorted(self._game_lineups.get(game_pk, set()))[:n_players]

        for i, pid in enumerate(lineup):
            if i >= n_players:
                break
            _gs = self._player_game_stats.get((pid, game_pk))
            if _gs is None:
                continue
            result["hits"][i] = _gs["hits"]
            result["hr"][i] = _gs["hr"]
            result["so"][i] = _gs["so"]
            result["hrbi"][i] = _gs["hrbi"]
            result["tb"][i] = _gs["tb"]
            result["sb"][i] = _gs["sb"]

        return result

    # ------------------------------------------------------------------
    # E. Masks
    # ------------------------------------------------------------------

    def _build_masks(
        self, target, game_pk: int, prefix_len: int, player_ctx: dict
    ) -> dict:
        """Build yrfi_mask and player_mask."""
        # YRFI mask: 1.0 if still in 1st inning or pregame
        yrfi_mask = 1.0
        if prefix_len > 0 and game_pk in self._game_offsets:
            _m_start, _ = self._game_offsets[game_pk]
            _m_idx = _m_start + prefix_len - 1
            if _m_idx < len(self._hierarchy_array) and self._hierarchy_array[_m_idx, 0] > 1:
                yrfi_mask = 0.0

        # Player mask: 1 if player has valid targets (present in lineup)
        lineup = sorted(self._game_lineups.get(game_pk, set()))[:MAX_PLAYERS_PER_GAME]
        player_mask = torch.zeros(MAX_PLAYERS_PER_GAME, dtype=torch.float32)

        for i, pid in enumerate(lineup):
            if i >= MAX_PLAYERS_PER_GAME:
                break
            _gs = self._player_game_stats.get((pid, game_pk))
            if _gs is not None:
                _status = _gs.get("target_status", "trainable")
                if _status == "trainable" or _status == "nan":
                    player_mask[i] = 1.0

        return {
            "yrfi_mask": torch.tensor(yrfi_mask, dtype=torch.float32),
            "player_mask": player_mask,
        }

    # ------------------------------------------------------------------
    # Bullpen context (Q3 ablation)
    # ------------------------------------------------------------------

    def get_bullpen_context(self, meta: pd.Series, side: str, game_pk: int) -> torch.Tensor:
        """Get bullpen context tensor based on ablation mode.

        Called externally or by collate_fn if bullpen_mode != "implicit".
        """
        mode = self.ablation.bullpen_mode

        if mode == "implicit":
            return torch.zeros(1, dtype=torch.float32)

        team_col = f"{side}_team_id"
        if team_col not in meta.index or pd.isna(meta[team_col]):
            if mode == "explicit_profile":
                return torch.zeros(BULLPEN_PROFILE_DIM, dtype=torch.float32)
            else:
                return torch.zeros(self.ablation.bullpen_n_relievers, 1, len(PITCH_CONTINUOUS_COLS), dtype=torch.float32)

        team_id = int(meta[team_col])
        game_date = meta["game_date"]

        if mode == "explicit_profile":
            return self._compute_bullpen_profile(team_id, game_date)
        elif mode == "reliever_sequences":
            return self._get_reliever_sequences(team_id, game_date, game_pk)

        return torch.zeros(1, dtype=torch.float32)

    def _compute_bullpen_profile(self, team_id: int, before_date: pd.Timestamp) -> torch.Tensor:
        """Compute aggregate bullpen stats for a team.

        Returns (BULLPEN_PROFILE_DIM,) tensor with ERA, FIP, K%, BB%, WHIP,
        recent_workload (last 3/7/14 days), availability flags for top 3 arms.
        """
        profile = torch.zeros(BULLPEN_PROFILE_DIM, dtype=torch.float32)

        team_df = self._team_game_lookup.get(team_id)
        if team_df is None or team_df.empty:
            return profile

        # Use last 30 days of team games for bullpen stats
        recent = team_df[
            (team_df["game_date"] < before_date) &
            (team_df["game_date"] >= before_date - pd.Timedelta(days=30))
        ]

        if recent.empty:
            return profile

        # Aggregate pitching stats (these are team-game level)
        if "pit_game_earned_runs" in recent.columns and "pit_game_innings_pitched" in recent.columns:
            er = pd.to_numeric(recent["pit_game_earned_runs"], errors="coerce").sum()
            ip = pd.to_numeric(recent["pit_game_innings_pitched"], errors="coerce").sum()
            if ip > 0:
                profile[0] = er * 9.0 / ip  # ERA proxy
        if "pit_game_so" in recent.columns and "pit_game_pitches_thrown" in recent.columns:
            so = pd.to_numeric(recent["pit_game_so"], errors="coerce").sum()
            pitches = pd.to_numeric(recent["pit_game_pitches_thrown"], errors="coerce").sum()
            if pitches > 0:
                profile[2] = so / pitches  # K rate proxy
        if "pit_game_bb" in recent.columns:
            bb = pd.to_numeric(recent["pit_game_bb"], errors="coerce").sum()
            pitches = pd.to_numeric(recent.get("pit_game_pitches_thrown", pd.Series([0])), errors="coerce").sum()
            if pitches > 0:
                profile[3] = bb / pitches  # BB rate proxy

        # Recent workload: games in last 3, 7, 14 days
        for j, days in enumerate([3, 7, 14]):
            window = recent[recent["game_date"] >= before_date - pd.Timedelta(days=days)]
            profile[5 + j] = float(len(window)) / days

        return profile

    def _get_reliever_sequences(
        self, team_id: int, before_date: pd.Timestamp, game_pk: int
    ) -> torch.Tensor:
        """Get last outing pitch sequences for top N relievers.

        Returns (n_relievers, max_len, n_features) tensor.
        """
        n_relievers = self.ablation.bullpen_n_relievers
        n_features = len(PITCH_CONTINUOUS_COLS)
        # Placeholder: full reliever sequence extraction requires reliever
        # identification from pitch data (non-SP pitcher appearances).
        # For now return zeros; to be populated when reliever tracking is added.
        return torch.zeros(n_relievers, 1, n_features, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def game_transformer_collate_fn(batch: list[dict]) -> dict:
    """Custom collate function for GameTransformerDataset.

    Handles variable-length pitch sequences by left-padding to the max length
    in the batch. Creates proper attention masks for the prefix-LM pattern.
    """
    if not batch:
        return {}

    B = len(batch)

    # --- Helper to pad sequence tensors along the time dimension ---
    def _pad_sequences_3d(key: str, compute_attn_mask: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad (n_games, var_len, features) to (B, n_games, max_len, features).

        Returns (padded_tensor, attention_mask).
        """
        tensors = [sample[key] for sample in batch]
        n_games = tensors[0].shape[0]
        n_features = tensors[0].shape[2]
        # Cap at 200 pitches per game — extra-inning outliers (400+) otherwise
        # force the full batch to pad, blowing up [B, G, T, F] memory ~8x.
        max_len = min(max(t.shape[1] for t in tensors), 200)

        padded = torch.zeros(B, n_games, max_len, n_features, dtype=torch.float32)
        attn_mask = torch.zeros(B, n_games, max_len, dtype=torch.float32)

        for i, t in enumerate(tensors):
            seq_len = t.shape[1]
            if seq_len >= max_len:
                # Truncate: keep the most recent max_len pitches
                padded[i] = t[:, -max_len:, :]
                if compute_attn_mask:
                    attn_mask[i] = 1.0
            else:
                # Left-pad: place content at end
                start = max_len - seq_len
                padded[i, :, start:, :] = t
                if compute_attn_mask:
                    mask_key = key.replace("_seqs", "_mask")
                    if mask_key in batch[i]:
                        m = batch[i][mask_key]
                        pad_m = max_len - m.shape[1]
                        if pad_m > 0:
                            attn_mask[i, :, pad_m:] = m
                        else:
                            attn_mask[i] = m[:, -max_len:]
                    else:
                        attn_mask[i, :, start:] = 1.0

        return padded, attn_mask

    def _pad_prefix(key: str) -> torch.Tensor:
        """Stack prefix tensors (already fixed-length from dataset)."""
        return torch.stack([sample[key] for sample in batch])

    # --- Collate each field ---
    result = {}

    # Historical SP context (variable sequence lengths across batch)
    for prefix in ["sp_home", "sp_away", "team_home", "team_away"]:
        seqs_key = f"{prefix}_seqs"
        obs_key = f"{prefix}_obs_mask"
        result[seqs_key], result[f"{prefix}_attn_mask"] = _pad_sequences_3d(seqs_key)
        result[obs_key], _ = _pad_sequences_3d(obs_key, compute_attn_mask=False)
        result[f"{prefix}_lengths"] = torch.stack([s[f"{prefix}_lengths"] for s in batch])
        result[f"{prefix}_weights"] = torch.stack([s[f"{prefix}_weights"] for s in batch])
        if f"{prefix}_similarity" in batch[0]:
            result[f"{prefix}_similarity"] = torch.stack([s[f"{prefix}_similarity"] for s in batch])

    # Flat features
    result["flat_features"] = torch.stack([s["flat_features"] for s in batch])

    # Rating temporal sequences [B, K, N_RATINGS] per side
    result["rating_home"] = torch.stack([s["rating_home"] for s in batch])
    result["rating_away"] = torch.stack([s["rating_away"] for s in batch])

    # Live prefix (fixed-length from dataset)
    result["prefix_values"] = _pad_prefix("prefix_values")
    result["prefix_obs_mask"] = _pad_prefix("prefix_obs_mask")
    result["prefix_mask"] = _pad_prefix("prefix_mask")
    result["prefix_batter_hash"] = _pad_prefix("prefix_batter_hash")
    result["prefix_pitcher_hash"] = _pad_prefix("prefix_pitcher_hash")
    result["prefix_catcher_hash"] = _pad_prefix("prefix_catcher_hash")
    result["prefix_event_type"] = _pad_prefix("prefix_event_type")
    result["prefix_hierarchy"] = _pad_prefix("prefix_hierarchy")
    result["prefix_pitch_type_idx"] = _pad_prefix("prefix_pitch_type_idx")
    result["prefix_bat_side_idx"] = _pad_prefix("prefix_bat_side_idx")
    result["prefix_pitch_hand_idx"] = _pad_prefix("prefix_pitch_hand_idx")
    result["prefix_half_inning_idx"] = _pad_prefix("prefix_half_inning_idx")
    result["prefix_hit_trajectory_idx"] = _pad_prefix("prefix_hit_trajectory_idx")
    result["prefix_hit_hardness_idx"] = _pad_prefix("prefix_hit_hardness_idx")
    result["prefix_length"] = torch.stack([s["prefix_length"] for s in batch])

    # Player context (fixed-size from dataset: MAX_PLAYERS_PER_GAME)
    result["player_hashes"] = torch.stack([s["player_hashes"] for s in batch])
    result["player_history"] = torch.stack([s["player_history"] for s in batch])
    result["player_history_mask"] = torch.stack([s["player_history_mask"] for s in batch])
    result["player_matchup"] = torch.stack([s["player_matchup"] for s in batch])

    # Targets (nested dict)
    target_keys = batch[0]["targets"].keys()
    result["targets"] = {}
    for key in target_keys:
        result["targets"][key] = torch.stack([s["targets"][key] for s in batch])

    # Masks
    result["yrfi_mask"] = torch.stack([s["yrfi_mask"] for s in batch])
    result["player_mask"] = torch.stack([s["player_mask"] for s in batch])

    # Metadata
    result["sample_weight"] = torch.stack([s["sample_weight"] for s in batch])
    result["game_pk"] = torch.stack([s["game_pk"] for s in batch])

    # Causal attention mask for prefix (prefix-LM pattern):
    # Each pitch can attend to itself and all prior pitches, but not future.
    max_prefix_len = result["prefix_values"].shape[1]
    causal_mask = torch.tril(torch.ones(max_prefix_len, max_prefix_len, dtype=torch.float32))
    # Mask out padding positions
    prefix_padding = result["prefix_mask"]  # (B, max_prefix_len)
    # (B, max_prefix_len, max_prefix_len): row i can attend to col j if
    # j is valid (not padded) AND j <= i in the causal sense
    result["prefix_causal_mask"] = causal_mask.unsqueeze(0) * prefix_padding.unsqueeze(1)

    return result
