"""Training script for the unified GameTransformer model.

Commands:
    fit-unified         Train the full model (phased: game-level → player → joint)
    learning-curves     Train at multiple data fractions to diagnose data-sufficiency
    evaluate            Evaluate a trained checkpoint against classical baseline

The learning-curve protocol runs training at [10%, 25%, 50%, 75%, 100%] of available
data and reports:
    1. Val loss vs training set size (shape diagnosis)
    2. Train/val gap at each fraction (overfit detection)
    3. Effective sequence count and state-space coverage
    4. Comparison against classical LightGBM baseline
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from .datasets import Standardizer, temporal_split_dates, SequenceSpec
from .distributions import negbin_nll
from .game_transformer import (
    ContextConfig,
    GameTransformer,
    GameTransformerLoss,
)
from .game_transformer_dataset import (
    AblationConfig,
    GameTransformerDataset,
    game_transformer_collate_fn,
)

_LOG_DIR = Path("data/logs")
_STATCAST_MIN_DATE = pd.Timestamp("2015-01-01")

# MLB gameType codes for games played under competitive rules, i.e. the games we price.
# Measured composition of the 2015+ feature store (164,438 games total, 32,193 from 2015+):
#   R 26,311 (81.7%)  S 5,248 (16.3%)  E 184  D 181  L 126  F 67  W 66  A 10
# Postseason (F wild card, D division, L league, W world series) is 440 games, ~40/season.
# It is kept deliberately: same rosters and rules as the regular season, and dropping it
# would mean the model never trains or validates on October, which is peak market volume.
# Excluded: S (spring training, minor-league rosters), E (exhibition/WBC), A (All-Star).
_COMPETITIVE_GAME_TYPES: tuple[str, ...] = ("R", "F", "D", "L", "W")

log = logging.getLogger(__name__)


def _log_memory(label: str = "") -> None:
    """Log current RAM and GPU memory usage."""
    import psutil
    proc = psutil.Process()
    rss_gb = proc.memory_info().rss / 1e9
    vm = psutil.virtual_memory()
    msg = f"[MEM {label}] RSS={rss_gb:.1f}GB, System={vm.used/1e9:.1f}/{vm.total/1e9:.1f}GB ({vm.percent}%)"
    if torch.cuda.is_available():
        gpu_used = torch.cuda.memory_allocated() / 1e9
        gpu_reserved = torch.cuda.memory_reserved() / 1e9
        msg += f", GPU_alloc={gpu_used:.1f}GB, GPU_reserved={gpu_reserved:.1f}GB"
    log.info(msg)


def _setup_logging() -> None:
    if log.handlers:
        return
    log.setLevel(logging.DEBUG)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(_LOG_DIR / "train_unified.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    log.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(sh)


# ---------------------------------------------------------------------------
# Task key groups for per-head logging
# ---------------------------------------------------------------------------

GAME_TASK_KEYS = frozenset({
    "negbin_home", "negbin_away", "bce_home_win", "bce_yrfi",
    "bce_extra_innings", "consistency",
})
PLAYER_TASK_KEYS = frozenset({
    "ce_hits", "focal_hr", "negbin_pitcher_k", "negbin_hrbi", "focal_sb",
})


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _train_one_epoch(
    model: torch.nn.Module,
    loss_fn: GameTransformerLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 5.0,
    player_context_dim: int = 512,
    scaler: Optional[torch.amp.GradScaler] = None,
    epoch: int = 0,
    phase_name: str = "",
    use_amp: bool = False,
) -> tuple[float, dict[str, float]]:
    """Single training epoch. Returns (avg_loss, avg_task_losses)."""
    model.train()
    total_loss = 0.0
    task_totals: dict[str, float] = {}
    n_batches = 0

    pbar = tqdm(loader, desc=f"Train {phase_name} E{epoch}", leave=False)
    for batch in pbar:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            model_input = _prepare_model_input(batch, player_context_dim=player_context_dim)
            predictions = model(model_input)
            targets_with_mask = {**batch["targets"], "player_mask": batch.get("player_mask")}
            loss, task_losses = loss_fn(predictions, targets_with_mask, batch.get("live_inning"))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        for k, v in task_losses.items():
            task_totals[k] = task_totals.get(k, 0.0) + v.item()
        n_batches += 1

        # Intra-epoch progress: show running loss split by game/player
        game_loss = sum(v.item() for k, v in task_losses.items() if k in GAME_TASK_KEYS)
        player_loss = sum(v.item() for k, v in task_losses.items() if k in PLAYER_TASK_KEYS)
        pbar.set_postfix(loss=f"{batch_loss:.4f}", game=f"{game_loss:.4f}", player=f"{player_loss:.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    avg_tasks = {k: v / max(n_batches, 1) for k, v in task_totals.items()}
    return avg_loss, avg_tasks


@torch.inference_mode()
def _validate(
    model: torch.nn.Module,
    loss_fn: GameTransformerLoss,
    loader: DataLoader,
    device: torch.device,
    player_context_dim: int = 512,
    use_amp: bool = False,
    epoch: int = 0,
    phase_name: str = "",
) -> tuple[float, dict[str, float]]:
    """Validation pass. Returns (avg_loss, avg_task_losses)."""
    model.eval()
    total_loss = 0.0
    task_totals: dict[str, float] = {}
    n_batches = 0

    pbar = tqdm(loader, desc=f"Val {phase_name} E{epoch}", leave=False)
    for batch in pbar:
        batch = _to_device(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            model_input = _prepare_model_input(batch, player_context_dim=player_context_dim)
            predictions = model(model_input)
            targets_with_mask = {**batch["targets"], "player_mask": batch.get("player_mask")}
            loss, task_losses = loss_fn(predictions, targets_with_mask, batch.get("live_inning"))

        total_loss += loss.item()
        for k, v in task_losses.items():
            task_totals[k] = task_totals.get(k, 0.0) + v.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    avg_tasks = {k: v / max(n_batches, 1) for k, v in task_totals.items()}
    return avg_loss, avg_tasks


def _to_device(batch: dict, device: torch.device) -> dict:
    """Recursively move batch tensors to device."""
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            result[k] = _to_device(v, device)
        else:
            result[k] = v
    return result


def _prepare_model_input(batch: dict, player_context_dim: int = 512) -> dict:
    """Map collated dataset fields to the model's expected input format.

    Collate produces flat keys (prefix_values, sp_home_seqs, etc.).
    Model expects nested structure: batch["context"], batch["live_*"], batch["player_*"].

    Args:
        player_context_dim: Expected flat dim for player_context (= player_context_tokens * d_model).
    """
    context = {}
    for prefix in ["sp_home", "sp_away", "team_home", "team_away"]:
        seqs_key = f"{prefix}_seqs"
        if seqs_key in batch:
            # attn_mask from collate is 1.0=valid, 0.0=padded
            # PerceiverResampler expects padding_mask: True=padded
            attn_mask = batch.get(f"{prefix}_attn_mask",
                torch.ones_like(batch[seqs_key][:, :, :, 0]))
            padding_mask = (attn_mask == 0)

            obs_key = f"{prefix}_obs_mask"
            context[prefix] = {
                "continuous": batch[seqs_key],
                "obs_mask": batch.get(obs_key),
                "batter_hash": batch.get(f"{prefix}_batter_hash",
                    torch.zeros_like(batch[seqs_key][:, :, :, 0]).long()),
                "pitcher_hash": batch.get(f"{prefix}_pitcher_hash",
                    torch.zeros_like(batch[seqs_key][:, :, :, 0]).long()),
                "inning_idx": batch.get(f"{prefix}_inning_idx",
                    torch.zeros_like(batch[seqs_key][:, :, :, 0]).long()),
                "ab_idx": batch.get(f"{prefix}_ab_idx",
                    torch.zeros_like(batch[seqs_key][:, :, :, 0]).long()),
                "pitch_idx": batch.get(f"{prefix}_pitch_idx",
                    torch.zeros_like(batch[seqs_key][:, :, :, 0]).long()),
                "padding_mask": padding_mask,
                "games_ago": batch.get(f"{prefix}_weights",
                    torch.zeros(batch[seqs_key].shape[0], batch[seqs_key].shape[1])),
                "seasons_crossed": torch.zeros(
                    batch[seqs_key].shape[0], batch[seqs_key].shape[1],
                    device=batch[seqs_key].device),
            }
    context["flat_features"] = batch["flat_features"]
    if "weather_temporal" in batch:
        context["weather_temporal"] = batch["weather_temporal"]
    if "rating_home" in batch:
        context["rating_home"] = batch["rating_home"]
    if "rating_away" in batch:
        context["rating_away"] = batch["rating_away"]

    model_input = {"context": context}

    # Live prefix — only include if any sample has prefix_length > 0
    has_live = batch["prefix_length"].sum() > 0
    if has_live:
        model_input["live_continuous"] = batch["prefix_values"]
        model_input["live_obs_mask"] = batch.get("prefix_obs_mask")
        model_input["live_batter_hash"] = batch["prefix_batter_hash"]
        model_input["live_pitcher_hash"] = batch["prefix_pitcher_hash"]
        model_input["live_catcher_hash"] = batch.get("prefix_catcher_hash")
        model_input["live_event_type"] = batch.get("prefix_event_type")
        model_input["live_pitch_type_idx"] = batch.get("prefix_pitch_type_idx")
        model_input["live_bat_side_idx"] = batch.get("prefix_bat_side_idx")
        model_input["live_pitch_hand_idx"] = batch.get("prefix_pitch_hand_idx")
        model_input["live_half_inning_idx"] = batch.get("prefix_half_inning_idx")
        model_input["live_hit_trajectory_idx"] = batch.get("prefix_hit_trajectory_idx")
        model_input["live_hit_hardness_idx"] = batch.get("prefix_hit_hardness_idx")
        model_input["live_inning_idx"] = batch["prefix_hierarchy"][:, :, 0]
        model_input["live_ab_idx"] = batch["prefix_hierarchy"][:, :, 1]
        model_input["live_pitch_idx"] = batch["prefix_hierarchy"][:, :, 2]

    # Player context
    model_input["player_hashes"] = batch["player_hashes"]
    if "player_history" in batch:
        B, P = batch["player_hashes"].shape
        # PlayerQueryHead expects [B, P, player_context_tokens * d_model]
        # player_history is [B, P, n_history_games, stat_dim] — flatten and
        # truncate/pad to match expected size
        expected_dim = player_context_dim
        flat_history = batch["player_history"].reshape(B, P, -1)
        actual_dim = flat_history.shape[-1]
        if actual_dim >= expected_dim:
            model_input["player_context"] = flat_history[:, :, :expected_dim]
        else:
            pad = torch.zeros(B, P, expected_dim - actual_dim,
                              device=flat_history.device, dtype=flat_history.dtype)
            model_input["player_context"] = torch.cat([flat_history, pad], dim=-1)

    return model_input


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _pitch_date_bound(arrow_type: "pa.DataType", bound: pd.Timestamp):
    """Type-match a date bound to the arrow type of pitch_sequences.game_date.

    The column has shipped as a timestamp, a date32 and a plain string across feature-store
    revisions. A pyarrow predicate whose literal type disagrees with the column either
    raises or silently matches nothing, so the bound is coerced rather than assumed.
    ISO-8601 strings are safe to compare lexicographically, which is why the string case
    can use a predicate at all.
    """
    import pyarrow as pa

    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return bound.strftime("%Y-%m-%d")
    if pa.types.is_date(arrow_type):
        return bound.date()
    return bound


def _load_feature_store(
    path: str,
    min_date: Optional[pd.Timestamp] = _STATCAST_MIN_DATE,
) -> dict[str, pd.DataFrame]:
    """Load all parquet files from the feature store directory.

    pitch_sequences is loaded with column subsetting, dtype downcasting, and
    categorical string columns to fit within commodity RAM.

    `min_date` prunes pitch_sequences AT READ TIME. The authoritative population filter is
    in `_build_datasets`, but that runs after the whole store is resident: reading all
    39.5M rows costs 11.45GB to build a frame of which only ~23% survives, and on the
    30GB/no-swap training box that pushed the build to a 26.9GB peak with 2GB free — the
    kernel then could not fork sshd for 2+ hours and neither SSH nor SSM could reach the
    instance. Pruning here is a strict SUPERSET of every split's lower bound (train is
    floored at min_date; val/test start at train_end/val_end, both later), so it cannot
    change which rows any split sees — only how many bytes are touched to get them.

    Measured on the real store (39,512,838 rows, 77 row groups, game_date timestamp[us]):
    the row groups happen to be date-ordered, so a `game_date >= 2015-01-01` predicate lets
    parquet skip 65 of 77 groups outright — 10.3M rows reach pandas instead of 39.5M, and the
    needed columns decode 4.67GB of uncompressed bytes instead of 16.02GB (70.8% less). Those
    two GB figures come from the footer's total_uncompressed_size and are NOT disk reads: the
    file is 2.53GB compressed. Confirmed on a cold-cache run of the real build via
    /proc/<pid>/read_bytes — 1.65GB read for the entire store, of which the eight non-pitch
    artifacts are 0.87GB, so ~0.78GB of pitch_sequences' 2.53GB was touched (~31%). Wall time
    for the full store load fell 26.5s -> 12.2s. Do not rely on the row-group skipping for
    correctness: it is an I/O bonus from the current write order, while the row-level filter
    is what guarantees the population.

    The game-type filter is deliberately NOT duplicated here; `_build_datasets` owns it, so
    there is one place for the population policy to live.
    """
    fs_path = Path(path)
    frames = {}

    required = ["pitch_sequences", "game_targets", "game_meta", "team_games", "player_batting_history"]
    optional = ["weather_features", "weather_temporal", "venue_dimensions", "daily_stats"]

    # Columns the GameTransformerDataset actually uses from pitch_sequences.
    # Loading only these saves ~4GB vs full 82-column read.
    _PITCH_COLS_NEEDED = [
        # Structural (sorting, indexing, date filtering)
        "game_pk", "play_index", "pitch_sequence_index", "game_date",
        # PITCH_CONTINUOUS_COLS (kinematics, game state, outcomes)
        "release_speed", "end_speed", "plate_time", "extension",
        "coord_px", "coord_pz", "coord_x0", "coord_y0", "coord_z0",
        "coord_vx0", "coord_vy0", "coord_vz0",
        "coord_ax", "coord_ay", "coord_az",
        "pfx_x", "pfx_z",
        "break_angle", "break_length", "break_y", "spin_rate", "spin_direction",
        "strike_zone_top", "strike_zone_bottom", "type_confidence",
        "zone_location",
        "hit_launch_speed", "hit_launch_angle", "hit_total_distance",
        "hit_coord_x", "hit_coord_y",
        "score_home", "score_away", "score_diff_batting",
        "cum_balls", "cum_strikes", "cum_outs",
        "pitch_count_balls", "pitch_count_strikes", "pitch_count_outs",
        "is_pitch", "is_strike", "is_ball", "is_in_play",
        "is_top_inning", "inning", "pitch_number",
        "weather_temp",
        # Pre-runner IDs (converted to binary occupancy)
        "pre_on_first_id", "pre_on_second_id", "pre_on_third_id",
        # Categorical columns (hash-bucketed or vocab-mapped)
        "batter_id", "pitcher_id", "fielder_2",
        "pitch_type", "bat_side_code", "pitch_hand_code", "p_throws",
        "bb_type", "hit_hardness", "launch_speed",
        # Event classification
        "at_bat_event", "event_type", "pitch_call", "at_bat_index",
    ]

    for name in required + optional:
        parquet_file = fs_path / f"{name}.parquet"
        if parquet_file.exists():
            log.info("Loading %s ...", parquet_file.name)
            t = time.time()
            if name == "pitch_sequences":
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(parquet_file)
                n_rows_on_disk = pf.metadata.num_rows
                schema = pf.schema_arrow
                available = set(schema.names)
                cols = [c for c in _PITCH_COLS_NEEDED if c in available]
                read_kwargs: dict = {"columns": cols}
                if min_date is not None and "game_date" in available:
                    bound = _pitch_date_bound(
                        schema.field("game_date").type, pd.Timestamp(min_date)
                    )
                    read_kwargs["filters"] = [("game_date", ">=", bound)]
                    log.info("  pruning to game_date >= %s at read time", bound)
                df = pd.read_parquet(parquet_file, **read_kwargs)
                # Downcast float64 → float32 to halve numeric memory
                f64_cols = df.select_dtypes("float64").columns
                if len(f64_cols):
                    df[f64_cols] = df[f64_cols].astype(np.float32)
                # Downcast int64 → int32 where safe (max vals < 2B)
                i64_cols = df.select_dtypes("int64").columns
                for col in i64_cols:
                    if col not in ("game_pk", "batter_id", "pitcher_id"):
                        df[col] = df[col].astype(np.int32)
                # Convert repeated string/object columns (including dates and codes)
                # after read; GameTransformerDataset normalizes category columns
                # before string fillna/map operations.
                string_cols = df.select_dtypes(include=["object", "string"]).columns
                if len(string_cols):
                    df[string_cols] = df[string_cols].astype("category")
                frames[name] = df
                mem_gb = df.memory_usage(deep=True).sum() / 1e9
                log.info("  %d rows, %d cols, %.2f GB in memory (%.1fs)",
                         len(df), len(cols), mem_gb, time.time() - t)
                # Report what the prune bought. If this ever shows 0 pruned on the real
                # store, the predicate silently failed to match and the 26.9GB peak is back.
                log.info("  pruned %d of %d rows on disk (%.1f%% kept)",
                         n_rows_on_disk - len(df), n_rows_on_disk,
                         100.0 * len(df) / max(n_rows_on_disk, 1))
            else:
                frames[name] = pd.read_parquet(parquet_file)
                log.info("  %d rows (%.1fs)", len(frames[name]), time.time() - t)
        elif name in required:
            log.warning("  %s not found — skipping", parquet_file)

    # Rating temporal sequences (pre-built .npz + .json metadata)
    rating_npz = fs_path / "rating_sequences.npz"
    if rating_npz.exists():
        from .rating_sequences import load_rating_sequences
        log.info("Loading rating_sequences ...")
        t = time.time()
        rating_seqs, rating_cols, k_steps = load_rating_sequences(str(rating_npz))
        frames["rating_sequences"] = rating_seqs
        frames["_rating_dim"] = len(rating_cols)
        log.info("  %d sequences, %d features, K=%d (%.1fs)",
                 len(rating_seqs), len(rating_cols), k_steps, time.time() - t)

    asof, wx_offsets = _load_weather_asof_artifacts(fs_path)
    if asof:
        frames["weather_asof"] = asof
        frames["wx_hour_offsets"] = wx_offsets

    return frames


def _load_weather_asof_artifacts(fs_path: Path) -> tuple[dict, dict]:
    """Load the as-of weather artifact (built by mlb_dl.build_weather_asof).

    Returns ({game_pk: [7,7,99] float32 STANDARDIZED}, {game_pk: int8 offsets}).
    The artifact stores raw values; z-scoring happens here with the train-only
    sidecar stats. Masked entries are 0 raw and stay exactly 0 after
    z*mask — standardize-then-mask (weather_asof D2).
    """
    from .weather_asof import (
        ASOF_CHANNELS, N_DECISIONS, N_TARGET_HOURS, N_DIMS, N_OBS_DIMS,
        OFF_FCST, OFF_FCST_MASK, OFF_OBS, OFF_OBS_MASK,
    )

    asof_dir = fs_path / "weather_asof"
    if not asof_dir.exists():
        return {}, {}
    norm_file = fs_path / "weather_asof_norm.json"
    stats = None
    if norm_file.exists():
        raw = json.loads(norm_file.read_text())
        stats = {k: np.asarray(raw[k], dtype=np.float32)
                 for k in ("fcst_mean", "fcst_std", "obs_mean", "obs_std")}
    else:
        log.warning("weather_asof artifact without norm sidecar — training on raw units")

    chan_cols = [f"wx_c{i:02d}" for i in range(ASOF_CHANNELS)]
    asof: dict[int, np.ndarray] = {}
    t = time.time()
    for f in sorted(asof_dir.glob("season=*.parquet")):
        df = pd.read_parquet(f).sort_values(["game_pk", "decision_hour", "target_hour"])
        # to_numpy can hand back a read-only zero-copy view; we mutate in place
        arr = np.array(df[chan_cols].to_numpy(np.float32))
        if stats is not None:
            for off, n, mean, std in ((OFF_FCST, N_DIMS, stats["fcst_mean"], stats["fcst_std"]),
                                      (OFF_OBS, N_OBS_DIMS, stats["obs_mean"], stats["obs_std"])):
                mask = arr[:, off + n:off + 2 * n] if off == OFF_FCST else arr[:, OFF_OBS_MASK:OFF_OBS_MASK + n]
                safe_std = np.where(std > 1e-8, std, 1.0)
                arr[:, off:off + n] = (arr[:, off:off + n] - mean) / safe_std * mask
        pks = df["game_pk"].to_numpy()
        per_game = N_DECISIONS * N_TARGET_HOURS
        for start in range(0, len(arr), per_game):
            asof[int(pks[start])] = arr[start:start + per_game].reshape(
                N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS)
    log.info("Loaded weather_asof: %d games (%.1fs)", len(asof), time.time() - t)

    offsets: dict[int, np.ndarray] = {}
    off_dir = fs_path / "wx_hour_offset"
    if off_dir.exists():
        for f in sorted(off_dir.glob("season=*.parquet")):
            df = pd.read_parquet(f).sort_values(["game_pk", "sequence_index"])
            for gpk, grp in df.groupby("game_pk", sort=False):
                offsets[int(gpk)] = grp["wx_hour_offset"].to_numpy(np.int8)
        log.info("Loaded wx_hour_offset: %d games", len(offsets))
    else:
        log.warning("weather_asof present but wx_hour_offset missing — "
                    "all live samples will use the pregame decision row (d=0)")
    return asof, offsets


def _build_datasets(
    frames: dict[str, pd.DataFrame],
    ablation: AblationConfig,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    min_date: Optional[pd.Timestamp] = _STATCAST_MIN_DATE,
    game_types: Optional[tuple[str, ...]] = _COMPETITIVE_GAME_TYPES,
) -> tuple[GameTransformerDataset, GameTransformerDataset, GameTransformerDataset]:
    """Build train/val/test datasets from feature store frames.

    Pre-splits pitch_sequences by date before passing to constructors so that
    the full 39.5M-row DataFrame is never held simultaneously with a filtered
    copy (which would exceed 32GB RAM).

    `min_date` is a hard floor on the TRAIN split, not merely an input to boundary
    selection. `temporal_split_dates` applies min_date only when choosing the cut points
    and returns two *upper* bounds, so a train mask of `dates < train_end` reaches back to
    the first row in the feature store. Before this floor existed the train split spanned
    1950-2024 (157,150 games, 75 seasons, 88.6% carrying an all-zero weather tensor) while
    val/test were 2024+ and ~91% weather-populated.

    `game_types` keeps only competitive MLB gameType codes — see
    `_COMPETITIVE_GAME_TYPES`. Spring training, exhibition/WBC and All-Star games field
    minor-league or novelty rosters under different run environments and are not traded, yet
    they were 16.0% of val and 16.7% of test, so they were shaping the metric used for model
    selection. Codes are logged with counts on every build: an inclusion list silently drops
    any code it does not know about, and the log is what makes that visible.
    """
    import gc

    # Fit standardizer on training data only (pitch continuous features)
    from .game_transformer_dataset import PITCH_CONTINUOUS_COLS

    # Restrict the game population BEFORE splitting, on game_targets (the frame each
    # dataset enumerates rows from) so the filter cannot be bypassed by the per-split
    # date masks below.
    allowed_pks: Optional[set] = None
    targets = frames["game_targets"]
    if game_types is not None:
        if "game_type_code" in targets.columns:
            n_before = len(targets)
            keep = targets["game_type_code"].isin(list(game_types))
            # Log what was dropped, by code. An inclusion list discards unknown codes
            # silently, so this is the only signal if the API adds one (or if postseason
            # codes ever change) — 886 rows also carry a null code in the current store.
            dropped = targets.loc[~keep, "game_type_code"].value_counts(dropna=False)
            targets = targets[keep]
            frames["game_targets"] = targets
            allowed_pks = set(targets["game_pk"])
            log.info(
                "Game-type filter %s: %d -> %d games (%d dropped)",
                tuple(game_types), n_before, len(targets), n_before - len(targets),
            )
            for code, n in dropped.items():
                log.info("  dropped game_type_code=%s: %d", code, n)
        else:
            log.warning(
                "game_type_code absent from game_targets; game-type filter SKIPPED, "
                "spring-training and exhibition games will be included"
            )

    pitch_df = frames["pitch_sequences"]
    if "game_date" in pitch_df.columns:
        dates = pitch_df["game_date"]
        # Category dtype needs special handling for comparison
        if isinstance(dates.dtype, pd.CategoricalDtype):
            dates = dates.astype(str)
        dates = pd.to_datetime(dates, errors="coerce")

        # Mirror the game population onto pitches as a mask, not a filtered copy: a
        # separate copy of the 39.5M-row frame would breach the same 32GB budget the
        # split sequencing further down exists to respect.
        in_population = pd.Series(True, index=pitch_df.index)
        if allowed_pks is not None and "game_pk" in pitch_df.columns:
            in_population = pitch_df["game_pk"].isin(allowed_pks)

        train_mask = in_population & (dates < train_end)
        if min_date is not None:
            train_mask &= dates >= pd.Timestamp(min_date)
        val_mask = in_population & (dates >= train_end) & (dates < val_end)
        test_mask = in_population & (dates >= val_end)
        log.info(
            "Population floor min_date=%s: train pitch rows %d (of %d total)",
            None if min_date is None else pd.Timestamp(min_date).date(),
            int(train_mask.sum()), len(pitch_df),
        )
        del in_population
    else:
        train_mask = pd.Series(True, index=pitch_df.index)
        val_mask = pd.Series(False, index=pitch_df.index)
        test_mask = pd.Series(False, index=pitch_df.index)

    train_pitches = pitch_df[train_mask]
    available_cols = [c for c in PITCH_CONTINUOUS_COLS if c in train_pitches.columns]
    standardizer = Standardizer.fit(train_pitches, available_cols)
    del train_pitches

    new_feature_kwargs = {
        "weather_features": frames.get("weather_features"),
        "weather_temporal": frames.get("weather_temporal"),
        "venue_dimensions": frames.get("venue_dimensions"),
        "daily_stats": frames.get("daily_stats"),
        "game_features": frames.get("rating_sequences"),
        "weather_asof": frames.get("weather_asof"),
        "wx_hour_offsets": frames.get("wx_hour_offsets"),
    }

    common_kwargs = dict(
        game_targets=frames["game_targets"],
        game_meta=frames["game_meta"],
        team_games=frames["team_games"],
        player_batting_history=frames["player_batting_history"],
        standardizer=standardizer,
        ablation=ablation,
        **new_feature_kwargs,
    )

    # Pre-split pitch_sequences by date and release the full DataFrame
    # BEFORE building any dataset. This keeps peak memory at ~16GB
    # (one split + constructor copy) instead of ~25GB (full + copy).
    log.info("Pre-splitting pitch_sequences by date...")
    pitch_train = pitch_df[train_mask].reset_index(drop=True)
    pitch_val = pitch_df[val_mask].reset_index(drop=True)
    pitch_test = pitch_df[test_mask].reset_index(drop=True)
    log.info("  train=%d, val=%d, test=%d rows",
             len(pitch_train), len(pitch_val), len(pitch_test))
    del pitch_df, frames["pitch_sequences"], train_mask, val_mask, test_mask
    gc.collect()
    _log_memory("after split + release full DF")

    log.info("Building train dataset...")
    train_ds = GameTransformerDataset(
        pitch_sequences=pitch_train,
        # split_start is what bounds game_targets below (game_transformer_dataset.py:529).
        # Filtering only pitch_sequences is insufficient: targets are filtered
        # independently inside the constructor, so train would still enumerate pre-2015
        # games and pair them with an empty pitch context.
        split_start=min_date,
        split_end=train_end,
        **common_kwargs,
    )
    del pitch_train
    gc.collect()

    log.info("Building val dataset...")
    val_ds = GameTransformerDataset(
        pitch_sequences=pitch_val,
        split_start=train_end,
        split_end=val_end,
        **common_kwargs,
    )
    del pitch_val
    gc.collect()

    log.info("Building test dataset...")
    test_ds = GameTransformerDataset(
        pitch_sequences=pitch_test,
        split_start=val_end,
        **common_kwargs,
    )
    del pitch_test
    gc.collect()

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Phased training
# ---------------------------------------------------------------------------


def _run_phased_training(
    model: GameTransformer,
    loss_fn: GameTransformerLoss,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output: Path,
    phase1_epochs: int = 50,
    phase2_epochs: int = 30,
    phase3_epochs: int = 20,
    lr1: float = 3e-4,
    lr2: float = 1e-4,
    lr3: float = 1e-5,
    weight_decay: float = 0.01,
    patience: int = 10,
    grad_clip: float = 5.0,
    writer: Optional[SummaryWriter] = None,
) -> dict:
    """Three-phase training protocol.

    Phase 1: Full model, game-level focus (player_loss_weight = 0.0)
    Phase 2: Unfreeze player heads, add player loss
    Phase 3: Joint fine-tune at low LR

    Returns training history dict.
    """
    history = {"phases": []}
    player_ctx_dim = model.d_model * 2  # player_context_tokens=2 by default

    # AMP with bfloat16 — wider dynamic range than float16, no NaN overflow
    use_amp = device.type == "cuda"
    scaler = None  # GradScaler not needed for bfloat16
    if use_amp:
        log.info("Mixed precision (AMP) enabled — bfloat16")

    # Phase 1: Game-level only
    log.info("=== Phase 1: Game-level heads (%d epochs, lr=%.0e) ===", phase1_epochs, lr1)
    log.info("  Player loss weight: 0.0 (disabled), all layers trainable")
    original_player_weight = loss_fn.PLAYER_LOSS_WEIGHT
    loss_fn.PLAYER_LOSS_WEIGHT = 0.0  # Disable player loss

    optimizer1 = torch.optim.AdamW(model.parameters(), lr=lr1, weight_decay=weight_decay)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=phase1_epochs)

    phase1_hist = _train_phase(
        model, loss_fn, train_loader, val_loader, optimizer1, scheduler1,
        device, output / "phase1", phase1_epochs, patience, grad_clip, "phase1",
        player_context_dim=player_ctx_dim,
        scaler=scaler,
        writer=writer,
        global_epoch_offset=0,
    )
    history["phases"].append({"phase": 1, "focus": "game_level", **phase1_hist})
    phase1_total_epochs = phase1_hist["epochs_trained"]

    # Phase 2: Add player heads
    log.info("=== Phase 2: + Player heads (%d epochs, lr=%.0e) ===", phase2_epochs, lr2)
    log.info("  Player loss weight: %.2f, lower 4 backbone layers FROZEN", original_player_weight)
    loss_fn.PLAYER_LOSS_WEIGHT = original_player_weight

    # Freeze lower backbone layers, train player head + top backbone layers
    _freeze_lower_layers(model, freeze=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    log.info("  Trainable: %d/%d params (%.1f%%)", n_trainable, n_total, 100 * n_trainable / n_total)
    optimizer2 = torch.optim.AdamW(trainable, lr=lr2, weight_decay=weight_decay)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=phase2_epochs)

    phase2_hist = _train_phase(
        model, loss_fn, train_loader, val_loader, optimizer2, scheduler2,
        device, output / "phase2", phase2_epochs, patience, grad_clip, "phase2",
        player_context_dim=player_ctx_dim,
        scaler=scaler,
        writer=writer,
        global_epoch_offset=phase1_total_epochs,
    )
    history["phases"].append({"phase": 2, "focus": "player_heads", **phase2_hist})
    phase2_total_epochs = phase1_total_epochs + phase2_hist["epochs_trained"]

    # Phase 3: Joint fine-tune (everything unfrozen, low LR)
    log.info("=== Phase 3: Joint fine-tune (%d epochs, lr=%.0e) ===", phase3_epochs, lr3)
    log.info("  All layers UNFROZEN, player loss weight: %.2f", loss_fn.PLAYER_LOSS_WEIGHT)
    _freeze_lower_layers(model, freeze=False)

    optimizer3 = torch.optim.AdamW(model.parameters(), lr=lr3, weight_decay=weight_decay)
    scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer3, T_max=phase3_epochs)

    phase3_hist = _train_phase(
        model, loss_fn, train_loader, val_loader, optimizer3, scheduler3,
        device, output / "phase3", phase3_epochs, patience, grad_clip, "phase3",
        player_context_dim=player_ctx_dim,
        scaler=scaler,
        writer=writer,
        global_epoch_offset=phase2_total_epochs,
    )
    history["phases"].append({"phase": 3, "focus": "joint", **phase3_hist})

    return history


def _train_phase(
    model, loss_fn, train_loader, val_loader, optimizer, scheduler,
    device, checkpoint_dir, max_epochs, patience, grad_clip, phase_name,
    player_context_dim: int = 512,
    scaler: Optional[torch.amp.GradScaler] = None,
    writer: Optional[SummaryWriter] = None,
    global_epoch_offset: int = 0,
) -> dict:
    """Train a single phase with early stopping."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    epochs_no_improve = 0
    epoch_history = []
    use_amp = device.type == "cuda"

    log.info("[%s] Starting — %d max epochs, patience=%d", phase_name, max_epochs, patience)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        global_epoch = global_epoch_offset + epoch

        train_loss, train_tasks = _train_one_epoch(
            model, loss_fn, train_loader, optimizer, device, grad_clip,
            player_context_dim=player_context_dim,
            scaler=scaler,
            epoch=epoch,
            phase_name=phase_name,
            use_amp=use_amp,
        )
        val_loss, val_tasks = _validate(
            model, loss_fn, val_loader, device,
            player_context_dim=player_context_dim,
            use_amp=use_amp,
            epoch=epoch,
            phase_name=phase_name,
        )
        scheduler.step()

        if epoch == 1:
            _log_memory(f"{phase_name} after epoch 1")

        elapsed = time.time() - t0
        gap = train_loss - val_loss

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "gap": round(gap, 5),
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": round(elapsed, 1),
            "train_tasks": {k: round(v, 5) for k, v in train_tasks.items()},
            "val_tasks": {k: round(v, 5) for k, v in val_tasks.items()},
        }
        epoch_history.append(record)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
        else:
            epochs_no_improve += 1

        mark = " *" if improved else f" ({epochs_no_improve}/{patience})"
        log.info(
            "[%s] e%d/%d (%.0fs) train=%.4f val=%.4f gap=%.4f%s",
            phase_name, epoch, max_epochs, elapsed, train_loss, val_loss, gap, mark
        )

        # Per-head loss breakdown every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            game_tasks = {k: round(v, 5) for k, v in val_tasks.items() if k in GAME_TASK_KEYS}
            player_tasks = {k: round(v, 5) for k, v in val_tasks.items() if k in PLAYER_TASK_KEYS}
            log.info("  Game heads:   %s", game_tasks)
            log.info("  Player heads: %s", player_tasks)

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar(f"loss/train_{phase_name}", train_loss, global_epoch)
            writer.add_scalar(f"loss/val_{phase_name}", val_loss, global_epoch)
            writer.add_scalar("loss/train_total", train_loss, global_epoch)
            writer.add_scalar("loss/val_total", val_loss, global_epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_epoch)

            # Per-head losses (val)
            for k, v in val_tasks.items():
                writer.add_scalar(f"val_heads/{k}", v, global_epoch)
            for k, v in train_tasks.items():
                writer.add_scalar(f"train_heads/{k}", v, global_epoch)

            # Grouped game vs player
            game_total = sum(v for k, v in val_tasks.items() if k in GAME_TASK_KEYS)
            player_total = sum(v for k, v in val_tasks.items() if k in PLAYER_TASK_KEYS)
            writer.add_scalar("loss/val_game_total", game_total, global_epoch)
            writer.add_scalar("loss/val_player_total", player_total, global_epoch)

            # Learned uncertainty weights
            if hasattr(loss_fn, "log_weights"):
                for i, w in enumerate(loss_fn.log_weights.detach().cpu()):
                    writer.add_scalar(f"log_weights/{i}", w.item(), global_epoch)

            writer.flush()

        if epochs_no_improve >= patience:
            log.info("[%s] Early stopping at epoch %d", phase_name, epoch)
            break

    # Restore best
    model.load_state_dict(torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=True))

    return {
        "best_val_loss": round(best_val, 5),
        "epochs_trained": len(epoch_history),
        "history": epoch_history,
    }


def _freeze_lower_layers(model: GameTransformer, freeze: bool = True) -> None:
    """Freeze/unfreeze context compiler and lower backbone layers (0-3)."""
    for param in model.context_compiler.parameters():
        param.requires_grad = not freeze
    for param in model.pitch_encoder.parameters():
        param.requires_grad = not freeze
    for param in model.pos_encoding.parameters():
        param.requires_grad = not freeze
    # Freeze bottom 4 of 6 backbone layers
    for i, layer in enumerate(model.backbone.layers):
        if i < 4:
            for param in layer.parameters():
                param.requires_grad = not freeze


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------


def _run_learning_curves(
    frames: dict[str, pd.DataFrame],
    ablation: AblationConfig,
    context_config: ContextConfig,
    output: Path,
    fractions: list[float],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    grad_clip: float,
    device: torch.device,
) -> dict:
    """Train at multiple data fractions and report scaling behavior."""
    output.mkdir(parents=True, exist_ok=True)

    # Temporal split
    train_end, val_end = temporal_split_dates(frames["game_targets"], min_date=_STATCAST_MIN_DATE)
    log.info("Temporal split: train < %s, val < %s, test >= %s",
             train_end.date(), val_end.date(), val_end.date())

    # Build full datasets
    log.info("Building datasets...")
    train_ds, val_ds, test_ds = _build_datasets(frames, ablation, train_end, val_end)
    log.info("Dataset sizes: train=%d, val=%d, test=%d", len(train_ds), len(val_ds), len(test_ds))

    # Report effective data characteristics
    data_stats = _compute_data_stats(train_ds, val_ds)
    log.info("Data stats: %s", json.dumps(data_stats, indent=2))

    # num_workers=2: 4 workers OOM-crashed on 30GB instance (22GB dataset × 4 CoW
    # forks exceeded free memory). persistent_workers=False prevents deadlock when
    # workers die — main process hangs in futex_wait otherwise.
    _NW = 2
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=game_transformer_collate_fn, num_workers=_NW, pin_memory=True,
    )

    results = {"fractions": [], "data_stats": data_stats}

    for frac in fractions:
        log.info("=== Learning curve: fraction=%.2f ===", frac)

        # Subsample training data (temporal: use earliest frac of training period)
        n_samples = int(len(train_ds) * frac)
        if n_samples < 1:
            log.warning("Skipping fraction %.2f (0 samples)", frac)
            continue

        # Use first n_samples (preserves temporal ordering for consistency)
        subset_indices = list(range(n_samples))
        subset = Subset(train_ds, subset_indices)

        train_loader = DataLoader(
            subset, batch_size=batch_size, shuffle=True,
            collate_fn=game_transformer_collate_fn, num_workers=_NW, pin_memory=True,
        )

        # Fresh model for each fraction (fair comparison)
        lc_d_model = 256
        model = GameTransformer(
            d_model=lc_d_model,
            flat_feature_dim=30,
            context_config=context_config,
            num_backbone_layers=4,
            num_heads=8,
            d_ff=lc_d_model * 4,
        ).to(device)
        loss_fn = GameTransformerLoss().to(device)
        lc_player_ctx_dim = lc_d_model * 2

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val = float("inf")
        best_train = float("inf")
        epochs_no_improve = 0
        curve_history = []

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, _ = _train_one_epoch(model, loss_fn, train_loader, optimizer, device, grad_clip, player_context_dim=lc_player_ctx_dim)
            val_loss, val_tasks = _validate(model, loss_fn, val_loader, device, player_context_dim=lc_player_ctx_dim)
            scheduler.step()

            if val_loss < best_val:
                best_val = val_loss
                best_train = train_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            curve_history.append({
                "epoch": epoch, "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5), "gap": round(train_loss - val_loss, 5),
            })

            if epochs_no_improve >= patience:
                break

        frac_result = {
            "fraction": frac,
            "n_samples": n_samples,
            "n_sequences": n_samples,  # each sample IS a sequence
            "best_train_loss": round(best_train, 5),
            "best_val_loss": round(best_val, 5),
            "gap_at_best": round(best_train - best_val, 5),
            "epochs_trained": len(curve_history),
            "history": curve_history,
        }
        results["fractions"].append(frac_result)

        log.info(
            "  frac=%.2f n=%d: best_val=%.4f best_train=%.4f gap=%.4f epochs=%d",
            frac, n_samples, best_val, best_train, best_train - best_val, len(curve_history)
        )

        del model, loss_fn, optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Diagnose the curve shape
    diagnosis = _diagnose_learning_curve(results["fractions"])
    results["diagnosis"] = diagnosis
    log.info("=== Diagnosis ===")
    for k, v in diagnosis.items():
        log.info("  %s: %s", k, v)

    # Save
    with open(output / "learning_curves.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def _compute_data_stats(train_ds: GameTransformerDataset, val_ds: GameTransformerDataset) -> dict:
    """Compute data sufficiency statistics."""
    n_train = len(train_ds)
    n_val = len(val_ds)

    # Count pregame vs live samples
    n_pregame = sum(1 for _, prefix_len in train_ds.samples if prefix_len == 0)
    n_live = n_train - n_pregame

    # Unique games
    unique_games = len(set(gpk for gpk, _ in train_ds.samples))

    # Parameter count for reference
    model_tmp = GameTransformer(d_model=256, flat_feature_dim=30)
    n_params = sum(p.numel() for p in model_tmp.parameters())
    del model_tmp

    # Ratio: samples per parameter (rough heuristic floor = 10-50x)
    samples_per_param = n_train / n_params

    return {
        "n_train_samples": n_train,
        "n_val_samples": n_val,
        "n_pregame_samples": n_pregame,
        "n_live_samples": n_live,
        "n_unique_games": unique_games,
        "avg_sequences_per_game": round(n_train / max(unique_games, 1), 1),
        "model_params": n_params,
        "samples_per_param_ratio": round(samples_per_param, 2),
        "heuristic_sufficient": samples_per_param >= 10,
    }


def _diagnose_learning_curve(frac_results: list[dict]) -> dict:
    """Interpret learning curve shape and train/val gap."""
    if len(frac_results) < 2:
        return {"status": "insufficient_points", "detail": "Need at least 2 fractions"}

    val_losses = [r["best_val_loss"] for r in frac_results]
    train_losses = [r["best_train_loss"] for r in frac_results]
    gaps = [r["gap_at_best"] for r in frac_results]
    fracs = [r["fraction"] for r in frac_results]

    # Slope of val loss vs fraction (linear fit in log-space)
    log_fracs = np.log(fracs)
    val_arr = np.array(val_losses)

    if len(val_arr) >= 3:
        # Fit line to last 3 points to detect plateau
        slope_last = (val_arr[-1] - val_arr[-3]) / (log_fracs[-1] - log_fracs[-3])
        # Fit line to first 3 points to detect initial improvement
        slope_first = (val_arr[2] - val_arr[0]) / (log_fracs[2] - log_fracs[0])
    else:
        slope_last = (val_arr[-1] - val_arr[0]) / (log_fracs[-1] - log_fracs[0])
        slope_first = slope_last

    # Total improvement from 10% to 100%
    total_improvement = val_arr[0] - val_arr[-1]
    relative_improvement = total_improvement / max(abs(val_arr[0]), 1e-8)

    # Gap trend
    gap_trend = gaps[-1] - gaps[0]

    # Classification
    if slope_last < -0.01 * abs(val_arr[-1]):
        shape = "still_improving"
        recommendation = "DATA_CONSTRAINED: model still improving at 100% — more data would likely help"
    elif abs(slope_last) < 0.005 * abs(val_arr[-1]):
        shape = "plateaued"
        recommendation = "SATURATED: diminishing returns from volume — gains come from architecture/features/regularization"
    else:
        shape = "mixed"
        recommendation = "MIXED: some improvement remains but rate is slowing"

    # Overfit detection: gap = train - val; overfitting means val > train (gap < 0)
    if gaps[-1] < -0.1 * abs(val_arr[-1]) and gap_trend < 0:
        overfit_status = "OVERFITTING: gap widening with more data — capacity/regularization issue"
    elif gaps[-1] < -0.1 * abs(val_arr[-1]):
        overfit_status = "MODERATE_GAP: train/val gap present but stable"
    else:
        overfit_status = "HEALTHY: train/val gap small and stable"

    return {
        "shape": shape,
        "recommendation": recommendation,
        "overfit_status": overfit_status,
        "val_loss_at_100pct": round(float(val_arr[-1]), 5),
        "val_loss_at_10pct": round(float(val_arr[0]), 5),
        "total_improvement": round(float(total_improvement), 5),
        "relative_improvement_pct": round(float(relative_improvement * 100), 2),
        "final_train_val_gap": round(float(gaps[-1]), 5),
        "gap_trend": round(float(gap_trend), 5),
        "slope_last_segment": round(float(slope_last), 6),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _hits_categorical_metrics(hits_logits: torch.Tensor,
                              hits_actual: torch.Tensor,
                              player_mask=None) -> dict:
    """CE + accuracy for the 5-way hits head, where class 4 means "4 OR MORE".

    The clamp is not defensive rounding — it is the head's definition, and it
    must match game_transformer's loss exactly. Without it the first 5-hit game
    in a split raises "Target 5 is out of bounds" and kills the whole eval pass.
    """
    B, P, C = hits_logits.shape
    logits_flat = hits_logits.reshape(-1, C)
    actual_flat = hits_actual.reshape(-1).long()
    valid = torch.from_numpy(_player_valid(hits_actual.numpy(), player_mask))
    if valid.sum() == 0:
        return {}
    target = actual_flat[valid].clamp(0, C - 1)
    pred = logits_flat[valid]
    return {
        "player_hits_ce": round(F.cross_entropy(pred, target, reduction="mean").item(), 5),
        "player_hits_accuracy": round((pred.argmax(dim=-1) == target).float().mean().item(), 4),
        "player_hits_n": int(valid.sum()),
    }


def _binary_skill(name: str, p: np.ndarray, y: np.ndarray) -> dict:
    """Brier skill against always quoting the base rate.

    A raw Brier is unreadable without its baseline: 0.207 sounds bad and 0.068
    sounds good, but the only question for a market maker is whether the number
    beats the constant quote, whose Brier is p(1-p). That identity holds ONLY for
    a binary y, so callers must binarise count targets first.
    """
    if len(y) == 0:
        return {}
    base = float(y.mean())
    const = base * (1.0 - base)
    brier = float(np.mean((p - y) ** 2))
    return {
        f"{name}_brier_constant": round(const, 5),
        f"{name}_bss": round((const - brier) / const, 4) if const > 0 else None,
    }


def _player_valid(target, player_mask) -> np.ndarray:
    """Which player slots count toward a held-out metric.

    Must match the loss exactly. GameTransformerLoss weights every player head by
    `targets["player_mask"]`, but evaluation used to filter on `y >= 0` — and
    padding slots are encoded as 0, not -1, so that filter admitted all of them.
    On the real prepared test split that is 7,076 of 54,220 slots (13.1%), and
    they carry genuine outcomes (2.5% HR positives), so scoring them both moves
    the p(1-p) baseline every skill score is measured against (HR base rate
    0.12746 masked vs 0.11415 unmasked) and scores the model on rows the loss
    deliberately excluded.

    `player_mask` may be absent on the from-frames path, in which case the
    sentinel filter is the only signal available.
    """
    y = np.asarray(target).reshape(-1)
    if player_mask is None:
        return y >= 0
    pm = np.asarray(player_mask).reshape(-1)
    if pm.shape != y.shape:
        raise ValueError(
            f"player_mask shape {pm.shape} != target shape {y.shape}; refusing to "
            "broadcast, which would silently mask the wrong slots"
        )
    return (pm > 0) & (y >= 0)


@torch.no_grad()
def _evaluate_model(
    model: GameTransformer,
    loader: DataLoader,
    device: torch.device,
    player_context_dim: int = 512,
) -> dict:
    """Compute evaluation metrics on a DataLoader."""
    model.eval()

    all_preds: dict[str, list] = {
        "mu_home": [], "alpha_home": [], "mu_away": [], "alpha_away": [],
        "home_win_logit": [], "yrfi_logit": [], "extra_innings_logit": [],
        "hr_prob": [], "hits_categorical": [],
        "pitcher_k_mu": [], "pitcher_k_alpha": [],
        "h_r_rbi_mu": [], "h_r_rbi_alpha": [],
        "stolen_bases_logit": [],
    }
    all_targets: dict[str, list] = {
        "home_runs_remaining": [], "away_runs_remaining": [],
        "home_win": [], "yrfi": [],
        "extra_innings": [], "player_hr": [], "player_hits": [],
        # The pitcher-strikeout target is "player_so" (precollate row 2, and the
        # loss at game_transformer.py:1392). Eval asked for "player_pitcher_k",
        # which the dataset never emits, so the key stayed an empty list, the
        # isinstance guard skipped the block, and the strikeout-props head shipped
        # with NO held-out metric at all. Keep metric names keyed to the target so
        # any future drift shows up in the output instead of vanishing.
        "player_so": [], "player_hrbi": [], "player_sb": [],
    }

    # player_mask rides alongside targets in the batch, not inside it (see the
    # merge at the training call sites); collect it so eval can mask like the loss.
    player_masks: list = []

    for batch in loader:
        batch = _to_device(batch, device)
        model_input = _prepare_model_input(batch, player_context_dim=player_context_dim)
        preds = model(model_input)

        for k in all_preds:
            if k in preds:
                all_preds[k].append(preds[k].cpu())
        for k in all_targets:
            if k in batch["targets"]:
                all_targets[k].append(batch["targets"][k].cpu())
        if batch.get("player_mask") is not None:
            player_masks.append(batch["player_mask"].cpu())

    # Concatenate
    for k in all_preds:
        if all_preds[k]:
            all_preds[k] = torch.cat(all_preds[k])
    for k in all_targets:
        if all_targets[k]:
            all_targets[k] = torch.cat(all_targets[k])

    pmask = torch.cat(player_masks).numpy() if player_masks else None

    metrics = {}

    # Game-level metrics
    if isinstance(all_preds.get("home_win_logit"), torch.Tensor):
        hw_prob = torch.sigmoid(all_preds["home_win_logit"]).numpy()
        hw_actual = all_targets["home_win"].numpy()
        metrics["home_win_brier"] = float(np.mean((hw_prob - hw_actual) ** 2))
        metrics["home_win_n"] = len(hw_actual)
        metrics.update(_binary_skill("home_win", hw_prob, hw_actual))

    if isinstance(all_preds.get("yrfi_logit"), torch.Tensor):
        yrfi_prob = torch.sigmoid(all_preds["yrfi_logit"]).numpy()
        yrfi_actual = all_targets["yrfi"].numpy()
        metrics["yrfi_brier"] = float(np.mean((yrfi_prob - yrfi_actual) ** 2))
        metrics["yrfi_base_rate"] = float(yrfi_actual.mean())
        metrics.update(_binary_skill("yrfi", yrfi_prob, yrfi_actual))

    # extra_innings was trained and the classical baseline block below even carries
    # an extra_innings_brier to compare against, but no metric was ever computed
    # for it — the head shipped unmeasured.
    if isinstance(all_preds.get("extra_innings_logit"), torch.Tensor) and \
            isinstance(all_targets.get("extra_innings"), torch.Tensor):
        xi_prob = torch.sigmoid(all_preds["extra_innings_logit"]).numpy()
        xi_actual = all_targets["extra_innings"].numpy()
        metrics["extra_innings_brier"] = float(np.mean((xi_prob - xi_actual) ** 2))
        metrics["extra_innings_base_rate"] = float(xi_actual.mean())
        metrics["extra_innings_pred_mean"] = float(xi_prob.mean())
        metrics.update(_binary_skill("extra_innings", xi_prob, xi_actual))

    if isinstance(all_preds.get("mu_home"), torch.Tensor):
        # Total runs MAE
        total_pred = all_preds["mu_home"].numpy() + all_preds["mu_away"].numpy()
        total_actual = (all_targets["home_runs_remaining"].numpy()
                        + all_targets["away_runs_remaining"].numpy())
        metrics["total_runs_mae"] = float(np.mean(np.abs(total_pred - total_actual)))
        metrics["total_runs_rmse"] = float(np.sqrt(np.mean((total_pred - total_actual) ** 2)))

        # NegBin NLL on held-out
        nll_home = negbin_nll(
            all_targets["home_runs_remaining"],
            all_preds["mu_home"],
            all_preds["alpha_home"],
        ).mean().item()
        nll_away = negbin_nll(
            all_targets["away_runs_remaining"],
            all_preds["mu_away"],
            all_preds["alpha_away"],
        ).mean().item()
        metrics["negbin_nll_home"] = round(nll_home, 5)
        metrics["negbin_nll_away"] = round(nll_away, 5)

    # Player-level metrics
    if isinstance(all_preds.get("hr_prob"), torch.Tensor) and isinstance(all_targets.get("player_hr"), torch.Tensor):
        hr_prob = all_preds["hr_prob"].numpy().flatten()
        hr_count = all_targets["player_hr"].numpy().flatten()
        valid = _player_valid(hr_count, pmask)
        # The head is P(1+ HR); the target is a count reaching 4. Scoring a
        # probability against a count makes Brier meaningless and its p(1-p)
        # baseline invalid, which is what made this head look worse than a
        # constant. Score the event the head actually predicts.
        hr_actual = (hr_count > 0).astype(np.float32)
        if valid.sum() > 0:
            metrics["player_hr_brier"] = float(np.mean((hr_prob[valid] - hr_actual[valid]) ** 2))
            metrics["player_hr_base_rate"] = float(hr_actual[valid].mean())
            metrics["player_hr_pred_mean"] = float(hr_prob[valid].mean())
            metrics["player_hr_n"] = int(valid.sum())
            metrics.update(_binary_skill("player_hr", hr_prob[valid], hr_actual[valid]))

    # Hits categorical (CE + accuracy)
    if isinstance(all_preds.get("hits_categorical"), torch.Tensor) and isinstance(all_targets.get("player_hits"), torch.Tensor):
        metrics.update(_hits_categorical_metrics(all_preds["hits_categorical"],
                                                 all_targets["player_hits"], pmask))

    # Pitcher K NegBin NLL
    if "pitcher_k_mu" in all_preds and "player_so" in all_targets:
        k_mu = all_preds["pitcher_k_mu"]
        k_alpha = all_preds["pitcher_k_alpha"]
        k_actual = all_targets["player_so"]
        if isinstance(k_mu, torch.Tensor) and isinstance(k_actual, torch.Tensor):
            valid = torch.from_numpy(_player_valid(k_actual.numpy(), pmask))
            if valid.sum() > 0:
                nll = negbin_nll(
                    k_actual.flatten()[valid],
                    k_mu.flatten()[valid],
                    k_alpha.flatten()[valid],
                ).mean().item()
                metrics["player_so_nll"] = round(nll, 5)
                metrics["player_so_pred_mean"] = round(k_mu.flatten()[valid].mean().item(), 3)
                metrics["player_so_actual_mean"] = round(k_actual.flatten()[valid].float().mean().item(), 3)
                metrics["player_so_n"] = int(valid.sum().item())

    # H+R+RBI NegBin NLL
    if "h_r_rbi_mu" in all_preds and "player_hrbi" in all_targets:
        hrbi_mu = all_preds["h_r_rbi_mu"]
        hrbi_alpha = all_preds["h_r_rbi_alpha"]
        hrbi_actual = all_targets["player_hrbi"]
        if isinstance(hrbi_mu, torch.Tensor) and isinstance(hrbi_actual, torch.Tensor):
            valid = torch.from_numpy(_player_valid(hrbi_actual.numpy(), pmask))
            if valid.sum() > 0:
                nll = negbin_nll(
                    hrbi_actual.flatten()[valid],
                    hrbi_mu.flatten()[valid],
                    hrbi_alpha.flatten()[valid],
                ).mean().item()
                metrics["player_hrbi_nll"] = round(nll, 5)
                metrics["player_hrbi_pred_mean"] = round(hrbi_mu.flatten()[valid].mean().item(), 3)
                metrics["player_hrbi_actual_mean"] = round(hrbi_actual.flatten()[valid].float().mean().item(), 3)

    # Stolen bases Brier
    if "stolen_bases_logit" in all_preds and "player_sb" in all_targets:
        sb_logit = all_preds["stolen_bases_logit"]
        sb_actual = all_targets["player_sb"]
        if isinstance(sb_logit, torch.Tensor) and isinstance(sb_actual, torch.Tensor):
            sb_prob = torch.sigmoid(sb_logit).numpy().flatten()
            sb_count = sb_actual.numpy().flatten()
            valid = _player_valid(sb_count, pmask)
            sb_true = (sb_count > 0).astype(np.float32)   # head is P(1+ SB)
            if valid.sum() > 0:
                metrics["player_sb_brier"] = float(np.mean((sb_prob[valid] - sb_true[valid]) ** 2))
                metrics["player_sb_base_rate"] = float(sb_true[valid].mean())
                metrics["player_sb_pred_mean"] = float(sb_prob[valid].mean())
                metrics["player_sb_n"] = int(valid.sum())
                metrics.update(_binary_skill("player_sb", sb_prob[valid], sb_true[valid]))

    return metrics


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def _resolve_weather_geometry(dataset, use_prepared: bool) -> tuple[ContextConfig, bool]:
    """Pick the weather token geometry the DATA actually carries.

    As-of weather is 7 decision-hour tokens of 99 channels; legacy is a 4x22
    snapshot. `fit-unified` and `evaluate` must agree exactly — a checkpoint
    trained on one geometry cannot be loaded into a model built for the other,
    so this stays a single function rather than two copies that can drift.

    Detection spans all three data paths: prepared manifest flag, or the
    per-game dict the cached/from-frames datasets attach. An empty dict means
    the artifact was absent, which is legacy, not as-of.
    """
    context_config = ContextConfig()
    asof_active = bool(
        (use_prepared and dataset.manifest.get("has_weather_asof"))
        or getattr(dataset, "_weather_asof_by_pk", None)
    )
    if asof_active:
        from .weather_asof import ASOF_CHANNELS, N_TARGET_HOURS
        context_config.weather_tokens = N_TARGET_HOURS
        context_config.weather_dim = ASOF_CHANNELS
        log.info("As-of weather active: weather tokens %d x %d channels",
                 context_config.weather_tokens, context_config.weather_dim)
    return context_config, asof_active


def _cmd_fit_unified(args) -> None:
    """Train the unified GameTransformer with phased protocol."""
    _setup_logging()
    t_start = time.time()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Try loading pre-collated tensors first (eliminates all per-batch assembly)
    prepared_dir = getattr(args, "prepared_dir", None)
    use_prepared = False
    if prepared_dir and Path(prepared_dir).exists() and (Path(prepared_dir) / "manifest.json").exists():
        log.info("Loading pre-collated tensors from: %s", prepared_dir)
        from .precollate import load_prepared_datasets, prepared_collate_fn
        train_ds, val_ds, test_ds = load_prepared_datasets(prepared_dir)
        _log_memory("after prepared datasets loaded")
        log.info("Dataset sizes: train=%d, val=%d, test=%d", len(train_ds), len(val_ds), len(test_ds))
        rating_dim = train_ds.manifest.get("rating_dim", 0)
        use_prepared = True
    # Try loading pre-computed cached datasets (saves ~30 min init but still has per-batch cost)
    elif (cache_dir := getattr(args, "dataset_cache", None)) and Path(cache_dir).exists() and (Path(cache_dir) / "manifest.json").exists():
        log.info("Loading pre-computed datasets from cache: %s", cache_dir)
        from .dataset_cache import load_cached_datasets
        train_ds, val_ds, test_ds = load_cached_datasets(cache_dir)
        _log_memory("after cached datasets loaded")
        log.info("Dataset sizes: train=%d, val=%d, test=%d", len(train_ds), len(val_ds), len(test_ds))
        # Rating dim from cached dataset
        rating_dim = train_ds._rating_dim if hasattr(train_ds, "_rating_dim") else 0
    else:
        # Fall back to building from feature store
        frames = _load_feature_store(args.feature_store)
        if not frames:
            log.error("No data loaded. Check --feature-store path.")
            return

        # Temporal split
        train_end, val_end = temporal_split_dates(frames["game_targets"], min_date=_STATCAST_MIN_DATE)
        log.info("Temporal split: train < %s, val < %s", train_end.date(), val_end.date())

        # Rating dimension (0 if no rating sequences available)
        rating_dim = frames.get("_rating_dim", 0)

        ablation = AblationConfig()
        train_ds, val_ds, test_ds = _build_datasets(frames, ablation, train_end, val_end)
        _log_memory("after datasets built")
        log.info("Dataset sizes: train=%d, val=%d, test=%d", len(train_ds), len(val_ds), len(test_ds))

        # Release raw frames — datasets now hold only pre-computed numpy arrays
        del frames
        gc.collect()
        _log_memory("after frames deleted")

    # Data loaders
    nw = getattr(args, "num_workers", 0)
    collate = prepared_collate_fn if use_prepared else game_transformer_collate_fn
    loader_kwargs = dict(pin_memory=True)
    if use_prepared and nw > 0:
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=nw, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=nw, **loader_kwargs,
    )

    _log_memory("after dataloaders created")

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    log.info("Device: %s", device)

    # Model
    context_config, _asof_active = _resolve_weather_geometry(train_ds, use_prepared)
    model = GameTransformer(
        d_model=args.d_model,
        rating_dim=rating_dim,
        flat_feature_dim=30,
        context_config=context_config,
        num_backbone_layers=args.n_layers,
        num_heads=args.n_heads,
        d_ff=args.d_model * 4,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    player_ctx_dim = args.d_model * 2
    log.info("Model: %d parameters (player_context_dim=%d)", n_params, player_ctx_dim)

    # Performance: TF32 matmul + cudnn autotuner + torch.compile
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        try:
            model = torch.compile(model)
            log.info("torch.compile enabled (TF32 + cudnn.benchmark + AMP)")
        except Exception as e:
            log.warning("torch.compile failed (%s), continuing without", e)

    loss_fn = GameTransformerLoss().to(device)

    # TensorBoard writer for per-head loss curves
    writer = None
    if SummaryWriter is not None:
        tb_dir = output / "runs"
        writer = SummaryWriter(log_dir=str(tb_dir))
        log.info("TensorBoard logging to %s (launch: tensorboard --logdir %s)", tb_dir, tb_dir)
    else:
        log.info("TensorBoard not installed — per-head curves in JSON only (pip install tensorboard)")

    # Phased training
    history = _run_phased_training(
        model, loss_fn, train_loader, val_loader, device, output,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase3_epochs=args.phase3_epochs,
        lr1=args.learning_rate,
        lr2=args.learning_rate * 0.33,
        lr3=args.learning_rate * 0.03,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip=args.gradient_clip,
        writer=writer,
    )
    if writer is not None:
        writer.close()

    # Evaluate on test set — guarded so a test-eval crash doesn't lose the
    # training record (training_history.json is what the diagnostic script uses
    # to decide if a step is complete; best.pt is already safe in checkpoint_dir)
    try:
        nw = getattr(args, "num_workers", 2)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=game_transformer_collate_fn, num_workers=nw,
        )
        test_metrics = _evaluate_model(model, test_loader, device, player_context_dim=player_ctx_dim)
        history["test_metrics"] = test_metrics
        log.info("=== Test Metrics ===")
        for k, v in test_metrics.items():
            log.info("  %s: %s", k, v)
    except Exception as exc:
        log.warning("Test eval failed (%s) — saving training record without test metrics", exc)
        history["test_metrics"] = {}

    # Always save — model weights were already checkpointed to best.pt during training
    with open(output / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), output / "final_model.pt")

    log.info("Total training time: %.1f minutes", (time.time() - t_start) / 60)


def _cmd_learning_curves(args) -> None:
    """Run learning curve experiment."""
    _setup_logging()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    frames = _load_feature_store(args.feature_store)
    if not frames:
        log.error("No data loaded.")
        return

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    log.info("Device: %s", device)

    ablation = AblationConfig()
    context_config = ContextConfig()

    results = _run_learning_curves(
        frames=frames,
        ablation=ablation,
        context_config=context_config,
        output=output,
        fractions=args.fractions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip=args.gradient_clip,
        device=device,
    )

    print(json.dumps(results["diagnosis"], indent=2))


def _cmd_evaluate(args) -> None:
    """Evaluate a trained model checkpoint."""
    _setup_logging()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Prefer the pre-collated tensors when they exist. Rebuilding the test split
    # from the feature store is the EBS-bound assembly path that dominated the
    # baseline run's wall clock; scoring a saved checkpoint should not pay it.
    prepared_dir = getattr(args, "prepared_dir", None)
    use_prepared = bool(
        prepared_dir and (Path(prepared_dir) / "manifest.json").exists()
    )
    if use_prepared:
        from .precollate import load_prepared_datasets, prepared_collate_fn
        log.info("Loading pre-collated tensors from: %s", prepared_dir)
        _, _, test_ds = load_prepared_datasets(prepared_dir)
        rating_dim = test_ds.manifest.get("rating_dim", 0)
        collate = prepared_collate_fn
        geometry_src = test_ds
    else:
        frames = _load_feature_store(args.feature_store)
        rating_dim = frames.get("_rating_dim", 0)
        train_end, val_end = temporal_split_dates(frames["game_targets"],
                                                  min_date=_STATCAST_MIN_DATE)
        _, _, test_ds = _build_datasets(frames, AblationConfig(), train_end, val_end)
        collate = game_transformer_collate_fn
        geometry_src = test_ds
    log.info("Test split: %d games (rating_dim=%d)", len(test_ds), rating_dim)

    context_config, _ = _resolve_weather_geometry(geometry_src, use_prepared)
    model = GameTransformer(
        d_model=args.d_model,
        rating_dim=rating_dim,
        flat_feature_dim=30,
        context_config=context_config,
        num_backbone_layers=args.n_layers,
        num_heads=args.n_heads,
        d_ff=args.d_model * 4,
        dropout=0.0,  # No dropout at eval
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    # fit-unified wraps the model in torch.compile on CUDA, which prefixes every
    # key with "_orig_mod."; loading such a checkpoint into a bare module fails
    # on every key at once, which reads like an architecture mismatch.
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=getattr(args, "num_workers", 0),
    )

    metrics = _evaluate_model(model, test_loader, device, player_context_dim=args.d_model * 2)

    # Classical baseline comparison.
    #
    # Only heads whose INFORMATION SET matches the classical model may be compared.
    # This block used to report home_win, which produced a "41.74% improvement" that
    # is pure artefact: the DL head is conditioned on live in-game state (it sees
    # runs already scored), the classical Brier is conditioned on pregame info only.
    # Subtracting them measures the information set, not the model. extra_innings is
    # the head where both sides are genuinely pregame-comparable, and there the DL
    # model is marginally WORSE -- the opposite conclusion.
    classical = _get_classical_baseline()
    metrics["classical_baseline"] = classical
    metrics["vs_classical"] = {
        "_scope": "extra_innings only; home_win/yrfi are conditioned on live in-game "
                  "state here and on pregame info in the classical model, so those "
                  "comparisons are invalid and deliberately omitted.",
    }
    if "extra_innings_brier" in metrics and "extra_innings_brier" in classical:
        dl_brier = metrics["extra_innings_brier"]
        cl_brier = classical["extra_innings_brier"]
        metrics["vs_classical"]["extra_innings_brier_improvement"] = round(cl_brier - dl_brier, 5)
        metrics["vs_classical"]["extra_innings_brier_pct_improvement"] = round(
            (cl_brier - dl_brier) / cl_brier * 100, 2
        )

    log.info("=== Evaluation Results ===")
    for k, v in metrics.items():
        if isinstance(v, dict):
            log.info("  %s:", k)
            for kk, vv in v.items():
                log.info("    %s: %s", kk, vv)
        else:
            log.info("  %s: %s", k, v)

    with open(output / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


def _cmd_precollate(args) -> None:
    """Pre-compute all training tensors from cached datasets."""
    _setup_logging()
    from .precollate import prepare_all

    log.info("Pre-collating datasets from %s → %s", args.dataset_cache, args.output)
    prepare_all(
        cache_dir=args.dataset_cache,
        output_dir=args.output,
        num_workers=args.num_workers,
    )
    log.info("Done. Use --prepared-dir %s with fit-unified.", args.output)


def _get_classical_baseline() -> dict:
    """Load classical model metrics for comparison."""
    # Look for the classical model's metrics file
    candidates = [
        Path("data/model_metrics_report.md"),
        Path("data/pregame_metrics.json"),
        Path("classical_learning/outputs/test_metrics.json"),
    ]

    for path in candidates:
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, Exception):
                continue

    # Fallback: known baseline from memory (calibration-metrics-post-fix)
    return {
        "home_win_brier": 0.2733,
        "yrfi_brier": 0.2502,
        "extra_innings_brier": 0.0677,
        "source": "memory:calibration-metrics-post-fix (2026-07-06)",
    }


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified GameTransformer training")
    sub = parser.add_subparsers(dest="command", required=True)

    # fit-unified
    fit = sub.add_parser("fit-unified", help="Train full unified model (phased)")
    fit.add_argument("--feature-store", required=True, help="Path to feature store directory")
    fit.add_argument("--output", required=True, help="Output directory for checkpoints")
    fit.add_argument("--d-model", type=int, default=384)
    fit.add_argument("--n-layers", type=int, default=8)
    fit.add_argument("--n-heads", type=int, default=12)
    fit.add_argument("--dropout", type=float, default=0.1)
    fit.add_argument("--batch-size", type=int, default=32)
    fit.add_argument("--learning-rate", type=float, default=3e-4)
    fit.add_argument("--weight-decay", type=float, default=0.01)
    fit.add_argument("--phase1-epochs", type=int, default=50)
    fit.add_argument("--phase2-epochs", type=int, default=30)
    fit.add_argument("--phase3-epochs", type=int, default=20)
    fit.add_argument("--patience", type=int, default=10)
    fit.add_argument("--gradient-clip", type=float, default=5.0)
    fit.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    fit.add_argument("--dataset-cache", default=None,
                     help="Path to pre-computed dataset cache (skips 30-min build)")
    fit.add_argument("--prepared-dir", default=None,
                     help="Path to pre-collated tensors (skips all per-batch assembly)")
    fit.set_defaults(func=_cmd_fit_unified)

    # learning-curves
    lc = sub.add_parser("learning-curves", help="Data sufficiency diagnosis via scaling curves")
    lc.add_argument("--feature-store", required=True)
    lc.add_argument("--output", required=True)
    lc.add_argument("--fractions", type=float, nargs="+", default=[0.10, 0.25, 0.50, 0.75, 1.0])
    lc.add_argument("--epochs", type=int, default=30, help="Max epochs per fraction")
    lc.add_argument("--batch-size", type=int, default=64)
    lc.add_argument("--learning-rate", type=float, default=3e-4)
    lc.add_argument("--weight-decay", type=float, default=0.01)
    lc.add_argument("--patience", type=int, default=8)
    lc.add_argument("--gradient-clip", type=float, default=5.0)
    lc.set_defaults(func=_cmd_learning_curves)

    # precollate
    pc = sub.add_parser("precollate", help="Pre-compute all training tensors (eliminates per-batch assembly)")
    pc.add_argument("--dataset-cache", required=True, help="Path to dataset cache directory")
    pc.add_argument("--output", required=True, help="Output directory for prepared tensors")
    pc.add_argument("--num-workers", type=int, default=8, help="Parallel workers for preparation")
    pc.set_defaults(func=_cmd_precollate)

    # evaluate
    ev = sub.add_parser("evaluate", help="Evaluate checkpoint against classical baseline")
    ev.add_argument("--feature-store", required=True)
    ev.add_argument("--checkpoint", required=True, help="Path to model .pt file")
    ev.add_argument("--output", required=True)
    ev.add_argument("--d-model", type=int, default=256)
    ev.add_argument("--n-layers", type=int, default=6)
    ev.add_argument("--n-heads", type=int, default=8)
    ev.add_argument("--batch-size", type=int, default=64)
    ev.add_argument("--prepared-dir", default=None,
                    help="Pre-collated tensor dir; skips feature-store rebuild")
    ev.add_argument("--num-workers", type=int, default=0)
    ev.set_defaults(func=_cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
