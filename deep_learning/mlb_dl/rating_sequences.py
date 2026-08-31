"""Build temporal rating sequences for the GameTransformer.

For each (game_pk, side), produces a [K, N_RATINGS] tensor representing how the
team's classical ratings evolved over their prior K games.  Features are
transformed to a team-centric perspective so that e.g. position 0 always means
"this team's Elo" regardless of whether they were home or away in the historical
game.

Usage:
    python -m mlb_dl.rating_sequences build \
        --game-features s3://mlb-265753586044-us-east-1-an/data/features/game_features.parquet \
        --output ./artifacts/feature_store/rating_sequences.npz \
        --train-end 2024-04-01
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rating feature identification
# ---------------------------------------------------------------------------

# Prefixes that identify "rating system" features in game_features.parquet.
_RATING_SYSTEM_PREFIXES = (
    "home_elo", "away_elo", "elo_diff", "elo_sum", "elo_prob",
    "home_wolfe", "away_wolfe", "wolfe_diff", "wolfe_sum", "wolfe_prob",
    "home_pythag_", "away_pythag_", "pythag_1st_", "pythag_2nd_",
    "home_srs", "away_srs", "srs_diff", "srs_sum",
    "diff_massey_", "diff_colley_",
    "log5_prob",
    "consensus_prob", "consensus_home_win_prob", "consensus_home_win_std",
    "market_massey_",
)

# How to detect interaction columns that involve at least one rating component.
_RATING_INTERACTION_COMPONENTS = (
    "elo", "massey", "colley", "wolfe", "pythag", "srs", "log5", "consensus",
)


# Features proven harmful or dead in 2026 (fold 7 sliding_3 importance analysis).
# Zeroed at load time rather than removed, preserving npz structure.
_RATING_EXCLUSIONS = frozenset({
    "wolfe_prob",              # 0/6 targets positive in 2026
    "diff_massey_full",        # 0/6 targets positive in 2026
    "diff_massey_inn6",        # 0/6 targets positive in 2026
    "diff_massey_inn7",        # 0/6 or near-zero in 2026
    "diff_massey_inn8",        # 0/6 targets positive in 2026
    "elo_prob",                # +++++++- decay pattern, negative 4/6 targets
    "elo_prob_x_same_league",  # negative, decaying
})


def identify_rating_columns(columns: list[str]) -> list[str]:
    """Select the classical rating feature columns from game_features.parquet.

    Selection logic (matches user specification):
    1. Columns matching any _RATING_SYSTEM_PREFIXES
    2. Interaction columns containing '_x_' where at least one component is a
       rating system prefix
    3. Excludes anything containing 'velo'
    """
    rating_cols = []
    for col in columns:
        if "velo" in col.lower():
            continue

        # Check direct prefix match
        if any(col.startswith(p) for p in _RATING_SYSTEM_PREFIXES):
            rating_cols.append(col)
            continue

        # Check interaction columns (_x_ separator, at least one rating component)
        if "_x_" in col:
            parts = col.split("_x_")
            for part in parts:
                if any(rp in part for rp in _RATING_INTERACTION_COMPONENTS):
                    rating_cols.append(col)
                    break

    return sorted(rating_cols)


# ---------------------------------------------------------------------------
# Perspective transformation
# ---------------------------------------------------------------------------

def _build_perspective_map(rating_cols: list[str]) -> dict[str, tuple[str, str]]:
    """Build a mapping from canonical feature name to (home_col, away_transform).

    For each rating feature, defines how to extract the team-centric value when
    the team was home vs away in a historical game.

    Returns dict: canonical_name -> (action_when_home, action_when_away)
    where action is one of:
        "same" - use the same column value
        "swap:other_col" - use other_col's value instead
        "negate" - negate the value
        "complement" - use 1.0 - value (for probabilities)
    """
    perspective_map = {}

    for col in rating_cols:
        if col.startswith("home_") and col.replace("home_", "away_", 1) in rating_cols:
            # Paired home/away absolute: swap when viewing from away perspective
            away_col = col.replace("home_", "away_", 1)
            perspective_map[col] = ("same", f"swap:{away_col}")
        elif col.startswith("away_") and col.replace("away_", "home_", 1) in rating_cols:
            # The away counterpart of a paired feature
            home_col = col.replace("away_", "home_", 1)
            perspective_map[col] = ("same", f"swap:{home_col}")
        elif "diff" in col.lower() and not "_x_" in col:
            # Differential features (home - away): negate when team was away
            perspective_map[col] = ("same", "negate")
        elif col.endswith("_prob") and "log5" not in col and "consensus" not in col:
            # Win probability from home perspective: complement when away
            perspective_map[col] = ("same", "complement")
        elif "elo_prob" in col and "_x_" not in col:
            # elo_prob specifically is P(home wins)
            perspective_map[col] = ("same", "complement")
        else:
            # Symmetric features (sums, matchup-level, interactions): no transform
            perspective_map[col] = ("same", "same")

    return perspective_map


def transform_to_team_perspective(
    row: pd.Series,
    rating_cols: list[str],
    perspective_map: dict[str, tuple[str, str]],
    team_was_home: bool,
) -> np.ndarray:
    """Extract rating features from a game row, transformed to team-centric view.

    Args:
        row: One row from game_features.parquet
        rating_cols: Ordered list of rating column names
        perspective_map: From _build_perspective_map
        team_was_home: Whether the team of interest was the home team in this game

    Returns: [N_RATINGS] float32 array
    """
    values = np.zeros(len(rating_cols), dtype=np.float32)

    for i, col in enumerate(rating_cols):
        raw_val = row.get(col, np.nan)
        if pd.isna(raw_val):
            values[i] = 0.0
            continue

        raw_val = float(raw_val)

        if team_was_home:
            values[i] = raw_val
        else:
            _, away_action = perspective_map.get(col, ("same", "same"))
            if away_action == "same":
                values[i] = raw_val
            elif away_action == "negate":
                values[i] = -raw_val
            elif away_action == "complement":
                values[i] = 1.0 - raw_val
            elif away_action.startswith("swap:"):
                swap_col = away_action[5:]
                swap_val = row.get(swap_col, np.nan)
                values[i] = float(swap_val) if not pd.isna(swap_val) else 0.0
            else:
                values[i] = raw_val

    return values


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

RATING_SEQ_STEPS = 10


def build_rating_sequences(
    game_features: pd.DataFrame,
    k_steps: int = RATING_SEQ_STEPS,
    train_end: Optional[str] = None,
) -> tuple[dict[tuple[int, str], np.ndarray], list[str], dict[str, float], dict[str, float]]:
    """Build temporal rating sequences for all games.

    Args:
        game_features: Full game_features.parquet DataFrame
        k_steps: Number of prior games to look back
        train_end: Date string for train/val split (standardization fit on train only)

    Returns:
        sequences: dict mapping (game_pk, "home"|"away") -> [k_steps, n_ratings] float32
        rating_cols: ordered list of rating column names
        means: per-feature mean (fit on training data only)
        stds: per-feature std (fit on training data only)
    """
    # Identify rating columns
    rating_cols = identify_rating_columns(list(game_features.columns))
    n_ratings = len(rating_cols)
    log.info(f"Identified {n_ratings} rating features for temporal sequences")

    # Build perspective transform map
    perspective_map = _build_perspective_map(rating_cols)

    # Ensure game_date is datetime and sorted
    gf = game_features.copy()
    gf["game_date"] = pd.to_datetime(gf["game_date"], errors="coerce")
    gf = gf.sort_values("game_date").reset_index(drop=True)

    # Build team → chronological game list
    # Each entry: (game_pk, was_home, row_index)
    team_game_index: dict[int, list[tuple[int, bool, int]]] = {}

    for idx, row in gf.iterrows():
        gpk = int(row["game_pk"])
        home_id = row.get("home_team_id")
        away_id = row.get("away_team_id")

        if pd.notna(home_id):
            tid = int(home_id)
            team_game_index.setdefault(tid, []).append((gpk, True, idx))
        if pd.notna(away_id):
            tid = int(away_id)
            team_game_index.setdefault(tid, []).append((gpk, False, idx))

    # Pre-extract rating values for all games as a matrix for vectorized lookup
    rating_matrix = gf[rating_cols].to_numpy(dtype=np.float32, na_value=0.0)

    # Build sequences
    sequences: dict[tuple[int, str], np.ndarray] = {}
    game_pk_to_idx: dict[int, int] = {int(gf.iloc[i]["game_pk"]): i for i in range(len(gf))}

    for team_id, games in team_game_index.items():
        for pos, (gpk, was_home, row_idx) in enumerate(games):
            side = "home" if was_home else "away"
            # For the target game's perspective, we want this team's prior games
            # Look back k_steps games in this team's chronological history
            start = max(0, pos - k_steps)
            prior_games = games[start:pos]  # excludes current game (no leakage)

            seq = np.zeros((k_steps, n_ratings), dtype=np.float32)

            # Fill from most recent (index k_steps-1) to oldest (index 0)
            for step_i, (hist_gpk, hist_was_home, hist_row_idx) in enumerate(
                reversed(prior_games)
            ):
                raw_row = gf.iloc[hist_row_idx]
                # Transform to team-centric perspective
                seq[k_steps - 1 - step_i] = transform_to_team_perspective(
                    raw_row, rating_cols, perspective_map, hist_was_home
                )

            # Store with the KEY being (target_game_pk, side_in_target_game)
            # We need to map back: this team was `side` in the target game
            sequences[(gpk, side)] = seq

    log.info(f"Built {len(sequences)} rating sequences ({len(sequences)//2} games × 2 sides)")

    # Fit standardization on training data only
    if train_end is not None:
        train_mask = gf["game_date"] < pd.Timestamp(train_end)
        train_gpks = set(gf.loc[train_mask, "game_pk"].astype(int).tolist())
    else:
        train_gpks = set(gf["game_pk"].astype(int).tolist())

    # Collect all training sequences into a matrix for mean/std computation
    train_seqs = []
    for (gpk, side), seq in sequences.items():
        if gpk in train_gpks:
            train_seqs.append(seq.reshape(-1, n_ratings))

    if train_seqs:
        all_train = np.concatenate(train_seqs, axis=0)
        # Exclude zero-padded rows (all zeros = no data)
        non_zero_mask = np.any(all_train != 0, axis=1)
        if non_zero_mask.any():
            all_train_valid = all_train[non_zero_mask]
            means = {col: float(all_train_valid[:, i].mean()) for i, col in enumerate(rating_cols)}
            stds = {col: float(np.maximum(all_train_valid[:, i].std(), 1e-8)) for i, col in enumerate(rating_cols)}
        else:
            means = {col: 0.0 for col in rating_cols}
            stds = {col: 1.0 for col in rating_cols}
    else:
        means = {col: 0.0 for col in rating_cols}
        stds = {col: 1.0 for col in rating_cols}

    # Apply standardization to all sequences
    mean_arr = np.array([means[col] for col in rating_cols], dtype=np.float32)
    std_arr = np.array([stds[col] for col in rating_cols], dtype=np.float32)

    for key in sequences:
        seq = sequences[key]
        # Only standardize non-zero rows (zero = padded/missing)
        non_zero = np.any(seq != 0, axis=1)
        seq[non_zero] = (seq[non_zero] - mean_arr) / std_arr
        sequences[key] = seq

    return sequences, rating_cols, means, stds


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_rating_sequences(
    sequences: dict[tuple[int, str], np.ndarray],
    rating_cols: list[str],
    means: dict[str, float],
    stds: dict[str, float],
    output_path: str,
    train_end: Optional[str] = None,
) -> None:
    """Save rating sequences to disk as .npz + metadata JSON.

    Format:
        {output_path}.npz: numpy archive with keys "home_{game_pk}" and "away_{game_pk}"
        {output_path}_meta.json: rating_cols, means, stds

    `train_end` is recorded rather than merely accepted. means/stds are only valid relative to
    the cut they were fit on, and the DL stack has three different notions of that cut: this
    builder's CLI default (2024-04-01), build_weather_asof's TRAIN_END_DATE (2024-01-01), and
    the ACTUAL split, which `datasets.temporal_split_dates` derives as an 80% quantile over
    distinct game dates and therefore MOVES whenever the population changes. Measured
    2026-08-31: the real cut was 2024-08-03, so both hardcoded dates sit earlier than it —
    conservative (they drop 1,597 and 2,119 train games from their fits) rather than leaky. But
    that ordering was unverifiable from the artifact alone, because nothing wrote the cut down.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Save sequences as npz
    arrays = {}
    for (gpk, side), seq in sequences.items():
        arrays[f"{side}_{gpk}"] = seq

    np.savez_compressed(str(output), **arrays)
    log.info(f"Saved {len(arrays)} arrays to {output}")

    # Save metadata
    meta_path = output.with_suffix(".json")
    meta = {
        "rating_cols": rating_cols,
        "n_ratings": len(rating_cols),
        "k_steps": RATING_SEQ_STEPS,
        "means": means,
        "stds": stds,
        "n_sequences": len(sequences),
        # None means "fit on every game", which is a leak. Recorded explicitly so that case is
        # visible in the artifact instead of being indistinguishable from a proper fit.
        "train_end": str(train_end) if train_end is not None else None,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Saved metadata to {meta_path}")


def load_rating_sequences(
    path: str,
) -> tuple[dict[tuple[int, str], np.ndarray], list[str], int]:
    """Load pre-built rating sequences from disk.

    Returns:
        sequences: dict (game_pk, side) -> [K, N_RATINGS] float32
        rating_cols: ordered feature names
        k_steps: number of temporal steps
    """
    npz_path = Path(path)
    meta_path = npz_path.with_suffix(".json")

    if not npz_path.exists() or not meta_path.exists():
        log.warning(f"Rating sequences not found at {path} — returning empty store")
        return {}, [], 0

    with open(meta_path) as f:
        meta = json.load(f)

    rating_cols = meta["rating_cols"]
    k_steps = meta["k_steps"]

    data = np.load(str(npz_path))
    sequences = {}
    for key in data.files:
        # key format: "{side}_{game_pk}"
        parts = key.split("_", 1)
        side = parts[0]
        gpk = int(parts[1])
        sequences[(gpk, side)] = data[key]

    log.info(f"Loaded {len(sequences)} rating sequences ({len(rating_cols)} features, K={k_steps})")

    # Zero out features proven harmful/dead in 2026 importance analysis (exact match only —
    # interaction terms containing excluded components may still carry signal)
    excluded_indices = [
        i for i, col in enumerate(rating_cols)
        if col in _RATING_EXCLUSIONS
    ]
    if excluded_indices:
        excluded_names = [rating_cols[i] for i in excluded_indices]
        for key in sequences:
            sequences[key][:, excluded_indices] = 0.0
        log.info(
            f"Zeroed {len(excluded_indices)} excluded rating features: {excluded_names}"
        )

    return sequences, rating_cols, k_steps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build rating temporal sequences")
    sub = parser.add_subparsers(dest="command")

    build_p = sub.add_parser("build", help="Build rating_sequences.npz from game_features.parquet")
    build_p.add_argument("--game-features", required=True, help="Path or S3 URI to game_features.parquet")
    build_p.add_argument("--output", required=True, help="Output path (without extension)")
    build_p.add_argument("--train-end", default="2024-04-01", help="Train split end date for standardization")
    build_p.add_argument("--k-steps", type=int, default=RATING_SEQ_STEPS, help="Lookback depth")

    info_p = sub.add_parser("info", help="Print info about a built sequences file")
    info_p.add_argument("path", help="Path to .npz file")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "build":
        log.info(f"Loading game features from {args.game_features}")
        gf_path = args.game_features
        if gf_path.startswith("s3://"):
            import s3fs
            fs = s3fs.S3FileSystem()
            with fs.open(gf_path, "rb") as f:
                game_features = pd.read_parquet(f)
        else:
            game_features = pd.read_parquet(gf_path)

        log.info(f"Loaded {len(game_features)} games × {len(game_features.columns)} columns")

        sequences, rating_cols, means, stds = build_rating_sequences(
            game_features, k_steps=args.k_steps, train_end=args.train_end
        )

        save_rating_sequences(sequences, rating_cols, means, stds, args.output,
                              train_end=args.train_end)
        log.info("Done.")

    elif args.command == "info":
        sequences, rating_cols, k_steps = load_rating_sequences(args.path)
        print(f"Sequences: {len(sequences)}")
        print(f"Rating features ({len(rating_cols)}): {rating_cols[:10]}...")
        print(f"K steps: {k_steps}")
        # Sample one sequence
        sample_key = next(iter(sequences))
        print(f"Sample shape: {sequences[sample_key].shape}")
        print(f"Sample (first 3 features, all steps):\n{sequences[sample_key][:, :3]}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
