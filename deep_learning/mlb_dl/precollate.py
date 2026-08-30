"""Pre-compute and store fully prepared training tensors.

Eliminates the per-batch assembly bottleneck by pre-computing all expensive
operations (historical game lookups, padding, mask building) once and storing
the results as memory-mapped numpy arrays.

Architecture:
  - Per-game arrays: context sequences, player history, flat features, etc.
    (shared across all prefix_lengths for a game)
  - Per-sample arrays: live prefix data and prefix-dependent targets
    (unique per (game_pk, prefix_len) pair)

At training time, __getitem__ is pure array indexing — no computation needed.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import gc
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)

MAX_CTX_LEN = 200
SP_GAMES = 5
TEAM_GAMES = 10
TOTAL_CTX_GAMES = 2 * SP_GAMES + 2 * TEAM_GAMES  # 30
PREFIX_LEN = 20  # SequenceSpec.history_length
N_CONTINUOUS = 52  # len(PITCH_CONTINUOUS_COLS)
MAX_PLAYERS = 20
PLAYER_HIST_GAMES = 15
PLAYER_STAT_DIM = 25
FLAT_DIM = 30
WEATHER_HOURS = 4
WEATHER_DIM = 22


# ---------------------------------------------------------------------------
# Preparation: compute and save per-game + per-sample arrays
# ---------------------------------------------------------------------------


class _GameContextDataset(Dataset):
    """Thin wrapper that iterates unique game_pks and computes per-game context."""

    def __init__(self, source_dataset, game_pks: list[int]):
        self.ds = source_dataset
        self.game_pks = game_pks

    def __len__(self):
        return len(self.game_pks)

    def __getitem__(self, idx):
        game_pk = self.game_pks[idx]
        meta = self.ds.meta_by_game.get(game_pk)

        # Context sequences (the expensive part)
        sp_home = self.ds._get_sp_context(meta, side="home", game_pk=game_pk)
        sp_away = self.ds._get_sp_context(meta, side="away", game_pk=game_pk)
        team_home = self.ds._get_team_context(meta, side="home", game_pk=game_pk)
        team_away = self.ds._get_team_context(meta, side="away", game_pk=game_pk)

        # Pre-pad all sequences to MAX_CTX_LEN
        def _pad_to_max(seqs_tensor, n_games):
            """Pad (n_games, var_len, 52) -> (n_games, MAX_CTX_LEN, 52)."""
            cur_len = seqs_tensor.shape[1]
            if cur_len >= MAX_CTX_LEN:
                return seqs_tensor[:, -MAX_CTX_LEN:, :].numpy()
            out = np.zeros((n_games, MAX_CTX_LEN, N_CONTINUOUS), dtype=np.float32)
            start = MAX_CTX_LEN - cur_len
            out[:, start:, :] = seqs_tensor.numpy()
            return out

        def _pad_mask_to_max(mask_tensor, n_games):
            """Pad (n_games, var_len) -> (n_games, MAX_CTX_LEN)."""
            cur_len = mask_tensor.shape[1]
            if cur_len >= MAX_CTX_LEN:
                return mask_tensor[:, -MAX_CTX_LEN:].numpy()
            out = np.zeros((n_games, MAX_CTX_LEN), dtype=np.float32)
            start = MAX_CTX_LEN - cur_len
            out[:, start:] = mask_tensor.numpy()
            return out

        # Combine all 4 contexts into (30, MAX_CTX_LEN, 52) arrays
        ctx_seqs = np.zeros((TOTAL_CTX_GAMES, MAX_CTX_LEN, N_CONTINUOUS), dtype=np.float16)
        ctx_obs = np.zeros((TOTAL_CTX_GAMES, MAX_CTX_LEN, N_CONTINUOUS), dtype=np.uint8)
        ctx_mask = np.zeros((TOTAL_CTX_GAMES, MAX_CTX_LEN), dtype=np.float32)
        ctx_lengths = np.zeros(TOTAL_CTX_GAMES, dtype=np.int16)
        ctx_weights = np.zeros(TOTAL_CTX_GAMES, dtype=np.float32)
        ctx_similarity = np.zeros(TOTAL_CTX_GAMES, dtype=np.float32)

        offset = 0
        for ctx, n in [(sp_home, SP_GAMES), (sp_away, SP_GAMES),
                       (team_home, TEAM_GAMES), (team_away, TEAM_GAMES)]:
            ctx_seqs[offset:offset+n] = _pad_to_max(ctx["sequences"], n).astype(np.float16)
            obs_raw = _pad_to_max(ctx["obs_mask"], n)
            ctx_obs[offset:offset+n] = (obs_raw > 0.5).astype(np.uint8)
            ctx_mask[offset:offset+n] = _pad_mask_to_max(ctx["mask"], n)
            ctx_lengths[offset:offset+n] = ctx["lengths"].numpy().astype(np.int16)
            ctx_weights[offset:offset+n] = ctx["weights"].numpy()
            if "similarity" in ctx:
                ctx_similarity[offset:offset+n] = ctx["similarity"].numpy()
            offset += n

        # Player context
        player_ctx = self.ds._get_player_context(game_pk, meta)

        # Flat features
        flat_features = self.ds._get_flat_features(meta)

        # Weather (legacy snapshot + full as-of tensor; the as-of per-sample
        # d-slice happens at PreparedDataset.__getitem__ via wx_decision_hour)
        weather = self.ds._get_weather_temporal(game_pk)
        weather_asof = self.ds._get_weather_asof_full(game_pk)

        # Ratings
        rating_home = self.ds._get_rating_temporal(game_pk, "home")
        rating_away = self.ds._get_rating_temporal(game_pk, "away")

        # Game-level targets
        target = self.ds.target_by_game.get(game_pk)
        targets_dict = self.ds._build_targets(target, game_pk, 0)  # prefix=0 for game-level

        # Player targets
        player_mask_ctx = self.ds._build_masks(target, game_pk, 0, player_ctx)

        return {
            "game_pk": game_pk,
            "ctx_seqs": ctx_seqs,
            "ctx_obs": ctx_obs,
            "ctx_mask": ctx_mask,
            "ctx_lengths": ctx_lengths,
            "ctx_weights": ctx_weights,
            "ctx_similarity": ctx_similarity,
            "player_hashes": player_ctx["hashes"].numpy(),
            "player_history": player_ctx["history"].numpy(),
            "player_history_mask": player_ctx["history_mask"].numpy(),
            "player_matchup": player_ctx["matchup"].numpy(),
            "flat_features": flat_features.numpy(),
            "weather": weather.numpy(),
            "weather_asof": weather_asof,
            "rating_home": rating_home.numpy(),
            "rating_away": rating_away.numpy(),
            # Game-level targets (constant across prefixes)
            "home_win": targets_dict["home_win"].item(),
            "yrfi": targets_dict["yrfi"].item(),
            "extra_innings": targets_dict["extra_innings"].item(),
            "total_runs": targets_dict["total_runs"].item(),
            # Player targets
            "player_hits": targets_dict["player_hits"].numpy(),
            "player_hr": targets_dict["player_hr"].numpy(),
            "player_so": targets_dict["player_so"].numpy(),
            "player_hrbi": targets_dict["player_hrbi"].numpy(),
            "player_tb": targets_dict["player_tb"].numpy(),
            "player_sb": targets_dict["player_sb"].numpy(),
            # Player/yrfi masks (yrfi_mask at prefix=0 is always 1.0)
            "player_mask": player_mask_ctx["player_mask"].numpy(),
            "sample_weight": self.ds._sample_game_weights.get(game_pk, 1.0),
        }


class _PrefixDataset(Dataset):
    """Iterates samples and computes only the live prefix (cheap)."""

    def __init__(self, source_dataset):
        self.ds = source_dataset

    def __len__(self):
        return len(self.ds.samples)

    def __getitem__(self, idx):
        game_pk, prefix_len = self.ds.samples[idx]
        prefix_data = self.ds._get_live_prefix(game_pk, prefix_len)

        # Prefix-dependent targets: runs remaining
        target = self.ds.target_by_game.get(game_pk)
        home_runs_final = float(getattr(target, "home_runs", 0) or 0)
        away_runs_final = float(getattr(target, "away_runs", 0) or 0)

        observed_home = 0.0
        observed_away = 0.0
        if prefix_len > 0 and game_pk in self.ds._game_offsets:
            _t_start, _ = self.ds._game_offsets[game_pk]
            _pitch_idx = _t_start + prefix_len - 1
            if _pitch_idx < len(self.ds._score_home_array):
                observed_home = float(self.ds._score_home_array[_pitch_idx])
                observed_away = float(self.ds._score_away_array[_pitch_idx])

        home_runs_remaining = max(0.0, home_runs_final - observed_home)
        away_runs_remaining = max(0.0, away_runs_final - observed_away)

        # YRFI mask: 1.0 if still pregame or in 1st inning
        yrfi_mask = 1.0
        if prefix_len > 0 and game_pk in self.ds._game_offsets:
            _m_start, _ = self.ds._game_offsets[game_pk]
            _m_idx = _m_start + prefix_len - 1
            if _m_idx < len(self.ds._hierarchy_array) and self.ds._hierarchy_array[_m_idx, 0] > 1:
                yrfi_mask = 0.0

        return {
            "game_pk": game_pk,
            "prefix_values": prefix_data["values"].numpy().astype(np.float16),
            "prefix_obs_mask": (prefix_data["obs_mask"].numpy() > 0.5).astype(np.uint8),
            "prefix_mask": prefix_data["mask"].numpy(),
            "prefix_batter_hash": prefix_data["batter_hash"].numpy().astype(np.int32),
            "prefix_pitcher_hash": prefix_data["pitcher_hash"].numpy().astype(np.int32),
            "prefix_catcher_hash": prefix_data["catcher_hash"].numpy().astype(np.int32),
            "prefix_event_type": prefix_data["event_type"].numpy().astype(np.int16),
            "prefix_hierarchy": prefix_data["hierarchy"].numpy().astype(np.int16),
            "prefix_pitch_type_idx": prefix_data["pitch_type_idx"].numpy().astype(np.int8),
            "prefix_bat_side_idx": prefix_data["bat_side_idx"].numpy().astype(np.int8),
            "prefix_pitch_hand_idx": prefix_data["pitch_hand_idx"].numpy().astype(np.int8),
            "prefix_half_inning_idx": prefix_data["half_inning_idx"].numpy().astype(np.int8),
            "prefix_hit_trajectory_idx": prefix_data["hit_trajectory_idx"].numpy().astype(np.int8),
            "prefix_hit_hardness_idx": prefix_data["hit_hardness_idx"].numpy().astype(np.int8),
            "prefix_length": prefix_len,
            "home_runs_remaining": home_runs_remaining,
            "away_runs_remaining": away_runs_remaining,
            "yrfi_mask": yrfi_mask,
        }


def prepare_split(
    dataset,
    output_dir: Path,
    split_name: str,
    num_workers: int = 8,
) -> dict:
    """Prepare a dataset split into pre-computed numpy arrays.

    Returns manifest dict with metadata.
    """
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Identify unique games and build index ---
    log.info("[%s] Building game index...", split_name)
    game_pk_set = set()
    sample_game_pks = []
    for gpk, _ in dataset.samples:
        game_pk_set.add(gpk)
        sample_game_pks.append(gpk)

    game_pks = sorted(game_pk_set)
    game_pk_to_idx = {gpk: i for i, gpk in enumerate(game_pks)}
    n_games = len(game_pks)
    n_samples = len(dataset.samples)
    log.info("[%s] %d games, %d samples", split_name, n_games, n_samples)

    # sample_to_game mapping
    sample_to_game = np.array(
        [game_pk_to_idx[gpk] for gpk in sample_game_pks], dtype=np.int32
    )
    np.save(split_dir / "sample_to_game.npy", sample_to_game)
    np.save(split_dir / "game_pks.npy", np.array(game_pks, dtype=np.int64))

    # Per-sample as-of decision hour: which wx_asof[d] row this sample's cut
    # pitch is allowed to see. Cheap dict lookups — no loader needed.
    wx_decision_hour = np.array(
        [dataset._get_wx_decision_hour(gpk, plen) for gpk, plen in dataset.samples],
        dtype=np.int8,
    )
    np.save(split_dir / "wx_decision_hour.npy", wx_decision_hour)

    # --- Phase 2: Compute per-game context (expensive, parallelized) ---
    log.info("[%s] Computing per-game context (%d games, %d workers)...",
             split_name, n_games, num_workers)
    t0 = time.time()

    game_ds = _GameContextDataset(dataset, game_pks)

    # Freeze GC before forking — prevents cyclic GC from scanning old-gen
    # objects in workers, which is a major COW trigger via refcount writes.
    gc.freeze()
    game_loader = DataLoader(
        game_ds, batch_size=1, num_workers=num_workers,
        shuffle=False, collate_fn=lambda x: x[0],
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Determine rating dim from first game
    first_game = game_ds[0]
    rating_dim = first_game["rating_home"].shape[-1] if first_game["rating_home"].ndim > 1 else 0
    matchup_dim = first_game["player_matchup"].shape[-1]

    # Create memory-mapped files on disk (avoids 90GB+ RAM allocation)
    def _mmap_create(name, shape, dtype):
        if isinstance(shape, int):
            shape = (shape,)
        return np.lib.format.open_memmap(
            str(split_dir / name), mode="w+", shape=shape, dtype=dtype
        )

    ctx_seqs_all = _mmap_create("ctx_seqs.npy", (n_games, TOTAL_CTX_GAMES, MAX_CTX_LEN, N_CONTINUOUS), np.float16)
    ctx_obs_all = _mmap_create("ctx_obs.npy", (n_games, TOTAL_CTX_GAMES, MAX_CTX_LEN, N_CONTINUOUS), np.uint8)
    ctx_mask_all = _mmap_create("ctx_mask.npy", (n_games, TOTAL_CTX_GAMES, MAX_CTX_LEN), np.float32)
    ctx_lengths_all = _mmap_create("ctx_lengths.npy", (n_games, TOTAL_CTX_GAMES), np.int16)
    ctx_weights_all = _mmap_create("ctx_weights.npy", (n_games, TOTAL_CTX_GAMES), np.float32)
    ctx_similarity_all = _mmap_create("ctx_similarity.npy", (n_games, TOTAL_CTX_GAMES), np.float32)
    player_hashes_all = _mmap_create("player_hashes.npy", (n_games, MAX_PLAYERS), np.int64)
    player_history_all = _mmap_create("player_history.npy", (n_games, MAX_PLAYERS, PLAYER_HIST_GAMES, PLAYER_STAT_DIM), np.float32)
    player_history_mask_all = _mmap_create("player_history_mask.npy", (n_games, MAX_PLAYERS, PLAYER_HIST_GAMES), np.bool_)
    player_matchup_all = _mmap_create("player_matchup.npy", (n_games, MAX_PLAYERS, matchup_dim), np.float32)
    flat_features_all = _mmap_create("flat_features.npy", (n_games, FLAT_DIM), np.float32)
    weather_all = _mmap_create("weather.npy", (n_games, WEATHER_HOURS, WEATHER_DIM), np.float32)
    from .weather_asof import ASOF_CHANNELS, N_DECISIONS, N_TARGET_HOURS
    weather_asof_all = _mmap_create(
        "weather_asof.npy", (n_games, N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS), np.float32)
    rating_home_all = _mmap_create("rating_home.npy", (n_games, 10, max(rating_dim, 1)), np.float32)
    rating_away_all = _mmap_create("rating_away.npy", (n_games, 10, max(rating_dim, 1)), np.float32)
    targets_game_all = _mmap_create("targets_game.npy", (n_games, 4), np.float32)
    targets_player_all = _mmap_create("targets_player.npy", (n_games, 6, MAX_PLAYERS), np.float32)
    player_mask_all = _mmap_create("player_mask.npy", (n_games, MAX_PLAYERS), np.float32)
    sample_weight_all = _mmap_create("sample_weight.npy", n_games, np.float32)

    for i, game_data in enumerate(game_loader):
        if i % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_games - i) / rate if rate > 0 else 0
            msg = f"[{split_name}]   {i}/{n_games} games ({rate:.1f}/s, ETA {eta:.0f}s)"
            log.info(msg)
            print(msg, flush=True)

        ctx_seqs_all[i] = game_data["ctx_seqs"]
        ctx_obs_all[i] = game_data["ctx_obs"]
        ctx_mask_all[i] = game_data["ctx_mask"]
        ctx_lengths_all[i] = game_data["ctx_lengths"]
        ctx_weights_all[i] = game_data["ctx_weights"]
        ctx_similarity_all[i] = game_data["ctx_similarity"]
        player_hashes_all[i] = game_data["player_hashes"]
        player_history_all[i] = game_data["player_history"]
        player_history_mask_all[i] = game_data["player_history_mask"]
        player_matchup_all[i, :, :game_data["player_matchup"].shape[-1]] = game_data["player_matchup"]
        flat_features_all[i] = game_data["flat_features"]
        weather_all[i] = game_data["weather"]
        weather_asof_all[i] = game_data["weather_asof"]
        rh = game_data["rating_home"]
        ra = game_data["rating_away"]
        if rh.ndim == 2 and rh.shape[-1] > 0:
            rating_home_all[i, :rh.shape[0], :rh.shape[1]] = rh
            rating_away_all[i, :ra.shape[0], :ra.shape[1]] = ra
        targets_game_all[i] = [
            game_data["home_win"], game_data["yrfi"],
            game_data["extra_innings"], game_data["total_runs"],
        ]
        targets_player_all[i, 0] = game_data["player_hits"]
        targets_player_all[i, 1] = game_data["player_hr"]
        targets_player_all[i, 2] = game_data["player_so"]
        targets_player_all[i, 3] = game_data["player_hrbi"]
        targets_player_all[i, 4] = game_data["player_tb"]
        targets_player_all[i, 5] = game_data["player_sb"]
        player_mask_all[i] = game_data["player_mask"]
        sample_weight_all[i] = game_data["sample_weight"]

    # Flush all mmap'd files
    for arr in [ctx_seqs_all, ctx_obs_all, ctx_mask_all, ctx_lengths_all,
                ctx_weights_all, ctx_similarity_all, player_hashes_all,
                player_history_all, player_history_mask_all, player_matchup_all,
                flat_features_all, weather_all, weather_asof_all,
                rating_home_all, rating_away_all,
                targets_game_all, targets_player_all, player_mask_all, sample_weight_all]:
        arr.flush()

    log.info("[%s] Per-game context done in %.1f min", split_name, (time.time() - t0) / 60)

    del game_loader
    del ctx_seqs_all, ctx_obs_all, ctx_mask_all
    gc.collect()

    # --- Phase 3: Compute per-sample prefix (parallelized) ---
    log.info("[%s] Computing per-sample prefixes (%d samples, %d workers)...",
             split_name, n_samples, num_workers)
    t0 = time.time()

    prefix_ds = _PrefixDataset(dataset)
    prefix_loader = DataLoader(
        prefix_ds, batch_size=1, num_workers=num_workers,
        shuffle=False, collate_fn=lambda x: x[0],
        prefetch_factor=4 if num_workers > 0 else None,
    )

    # Memory-mapped per-sample arrays (written directly to disk)
    prefix_values_all = _mmap_create("prefix_values.npy", (n_samples, PREFIX_LEN, N_CONTINUOUS), np.float16)
    prefix_obs_all = _mmap_create("prefix_obs.npy", (n_samples, PREFIX_LEN, N_CONTINUOUS), np.uint8)
    prefix_mask_all = _mmap_create("prefix_mask.npy", (n_samples, PREFIX_LEN), np.float32)
    prefix_batter_hash_all = _mmap_create("prefix_batter_hash.npy", (n_samples, PREFIX_LEN), np.int32)
    prefix_pitcher_hash_all = _mmap_create("prefix_pitcher_hash.npy", (n_samples, PREFIX_LEN), np.int32)
    prefix_catcher_hash_all = _mmap_create("prefix_catcher_hash.npy", (n_samples, PREFIX_LEN), np.int32)
    prefix_event_type_all = _mmap_create("prefix_event_type.npy", (n_samples, PREFIX_LEN), np.int16)
    prefix_hierarchy_all = _mmap_create("prefix_hierarchy.npy", (n_samples, PREFIX_LEN, 3), np.int16)
    prefix_pitch_type_all = _mmap_create("prefix_pitch_type.npy", (n_samples, PREFIX_LEN), np.int8)
    prefix_bat_side_all = _mmap_create("prefix_bat_side.npy", (n_samples, PREFIX_LEN), np.int8)
    prefix_pitch_hand_all = _mmap_create("prefix_pitch_hand.npy", (n_samples, PREFIX_LEN), np.int8)
    prefix_half_inning_all = _mmap_create("prefix_half_inning.npy", (n_samples, PREFIX_LEN), np.int8)
    prefix_hit_traj_all = _mmap_create("prefix_hit_traj.npy", (n_samples, PREFIX_LEN), np.int8)
    prefix_hit_hard_all = _mmap_create("prefix_hit_hard.npy", (n_samples, PREFIX_LEN), np.int8)
    prefix_length_all = _mmap_create("prefix_length.npy", n_samples, np.int16)
    home_runs_remaining_all = _mmap_create("home_runs_remaining.npy", n_samples, np.float32)
    away_runs_remaining_all = _mmap_create("away_runs_remaining.npy", n_samples, np.float32)
    yrfi_mask_all = _mmap_create("yrfi_mask.npy", n_samples, np.float32)

    for i, pdata in enumerate(prefix_loader):
        if i % 100000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_samples - i) / rate if rate > 0 else 0
            msg = f"[{split_name}]   {i}/{n_samples} prefixes ({rate:.1f}/s, ETA {eta:.0f}s)"
            log.info(msg)
            print(msg, flush=True)

        prefix_values_all[i] = pdata["prefix_values"]
        prefix_obs_all[i] = pdata["prefix_obs_mask"]
        prefix_mask_all[i] = pdata["prefix_mask"]
        prefix_batter_hash_all[i] = pdata["prefix_batter_hash"]
        prefix_pitcher_hash_all[i] = pdata["prefix_pitcher_hash"]
        prefix_catcher_hash_all[i] = pdata["prefix_catcher_hash"]
        prefix_event_type_all[i] = pdata["prefix_event_type"]
        prefix_hierarchy_all[i] = pdata["prefix_hierarchy"]
        prefix_pitch_type_all[i] = pdata["prefix_pitch_type_idx"]
        prefix_bat_side_all[i] = pdata["prefix_bat_side_idx"]
        prefix_pitch_hand_all[i] = pdata["prefix_pitch_hand_idx"]
        prefix_half_inning_all[i] = pdata["prefix_half_inning_idx"]
        prefix_hit_traj_all[i] = pdata["prefix_hit_trajectory_idx"]
        prefix_hit_hard_all[i] = pdata["prefix_hit_hardness_idx"]
        prefix_length_all[i] = pdata["prefix_length"]
        home_runs_remaining_all[i] = pdata["home_runs_remaining"]
        away_runs_remaining_all[i] = pdata["away_runs_remaining"]
        yrfi_mask_all[i] = pdata["yrfi_mask"]

    # Flush all per-sample mmap'd files
    for arr in [prefix_values_all, prefix_obs_all, prefix_mask_all,
                prefix_batter_hash_all, prefix_pitcher_hash_all, prefix_catcher_hash_all,
                prefix_event_type_all, prefix_hierarchy_all, prefix_pitch_type_all,
                prefix_bat_side_all, prefix_pitch_hand_all, prefix_half_inning_all,
                prefix_hit_traj_all, prefix_hit_hard_all, prefix_length_all,
                home_runs_remaining_all, away_runs_remaining_all, yrfi_mask_all]:
        arr.flush()

    log.info("[%s] Per-sample prefixes done in %.1f min", split_name, (time.time() - t0) / 60)

    manifest = {
        "n_games": n_games,
        "n_samples": n_samples,
        "rating_dim": int(rating_dim),
        "matchup_dim": int(matchup_dim),
        "max_ctx_len": MAX_CTX_LEN,
        "prefix_len": PREFIX_LEN,
        "n_continuous": N_CONTINUOUS,
        # As-of weather: presence gates PreparedDataset's weather source and
        # the model's weather_tokens/weather_dim config. has_weather_asof is
        # only true when the dataset actually carried tensors — an all-zero
        # weather_asof.npy from a run without the artifact must not silently
        # train a masked-out weather channel.
        "has_weather_asof": bool(dataset._weather_asof_by_pk),
        "asof_channels": int(ASOF_CHANNELS),
        "asof_decisions": int(N_DECISIONS),
        "asof_target_hours": int(N_TARGET_HOURS),
    }
    with open(split_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("[%s] Done. Saved to %s", split_name, split_dir)
    return manifest


def prepare_all(
    cache_dir: str,
    output_dir: str,
    num_workers: int = 8,
) -> None:
    """Prepare train/val/test splits from cached datasets."""
    from .dataset_cache import load_cached_datasets

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log.info("Loading cached datasets from %s", cache_dir)
    train_ds, val_ds, test_ds = load_cached_datasets(cache_dir)

    manifests = {}
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        manifests[name] = prepare_split(ds, output_path, name, num_workers)

    # Write top-level manifest
    top_manifest = {
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_cache": str(cache_dir),
        "splits": manifests,
    }
    with open(output_path / "manifest.json", "w") as f:
        json.dump(top_manifest, f, indent=2)

    log.info("All splits prepared. Output: %s", output_path)


# ---------------------------------------------------------------------------
# Training-time: PreparedDataset + fast collate
# ---------------------------------------------------------------------------


class PreparedDataset(Dataset):
    """Memory-mapped dataset serving pre-computed tensors.

    __getitem__ is pure array indexing — no computation, no historical lookups.
    """

    def __init__(self, split_dir: str | Path):
        split_dir = Path(split_dir)
        with open(split_dir / "manifest.json") as f:
            self.manifest = json.load(f)

        self.n_samples = self.manifest["n_samples"]
        self.n_games = self.manifest["n_games"]
        self._rating_dim = self.manifest.get("rating_dim", 0)

        # Per-game arrays (memory-mapped for shared access across workers)
        self._ctx_seqs = np.load(split_dir / "ctx_seqs.npy", mmap_mode="r")
        self._ctx_obs = np.load(split_dir / "ctx_obs.npy", mmap_mode="r")
        self._ctx_mask = np.load(split_dir / "ctx_mask.npy", mmap_mode="r")
        self._ctx_lengths = np.load(split_dir / "ctx_lengths.npy", mmap_mode="r")
        self._ctx_weights = np.load(split_dir / "ctx_weights.npy", mmap_mode="r")
        self._ctx_similarity = np.load(split_dir / "ctx_similarity.npy", mmap_mode="r")
        self._player_hashes = np.load(split_dir / "player_hashes.npy", mmap_mode="r")
        self._player_history = np.load(split_dir / "player_history.npy", mmap_mode="r")
        self._player_history_mask = np.load(split_dir / "player_history_mask.npy", mmap_mode="r")
        self._player_matchup = np.load(split_dir / "player_matchup.npy", mmap_mode="r")
        self._flat_features = np.load(split_dir / "flat_features.npy", mmap_mode="r")
        self._weather = np.load(split_dir / "weather.npy", mmap_mode="r")
        # As-of weather (manifest-gated): per-game [7,7,99] + per-sample d
        self._has_weather_asof = bool(self.manifest.get("has_weather_asof"))
        if self._has_weather_asof:
            self._weather_asof = np.load(split_dir / "weather_asof.npy", mmap_mode="r")
            self._wx_decision_hour = np.load(split_dir / "wx_decision_hour.npy", mmap_mode="r")
        self._rating_home = np.load(split_dir / "rating_home.npy", mmap_mode="r")
        self._rating_away = np.load(split_dir / "rating_away.npy", mmap_mode="r")
        self._targets_game = np.load(split_dir / "targets_game.npy", mmap_mode="r")
        self._targets_player = np.load(split_dir / "targets_player.npy", mmap_mode="r")
        self._player_mask = np.load(split_dir / "player_mask.npy", mmap_mode="r")
        self._sample_weight = np.load(split_dir / "sample_weight.npy", mmap_mode="r")

        # Per-sample arrays
        self._sample_to_game = np.load(split_dir / "sample_to_game.npy", mmap_mode="r")
        self._prefix_values = np.load(split_dir / "prefix_values.npy", mmap_mode="r")
        self._prefix_obs = np.load(split_dir / "prefix_obs.npy", mmap_mode="r")
        self._prefix_mask = np.load(split_dir / "prefix_mask.npy", mmap_mode="r")
        self._prefix_batter_hash = np.load(split_dir / "prefix_batter_hash.npy", mmap_mode="r")
        self._prefix_pitcher_hash = np.load(split_dir / "prefix_pitcher_hash.npy", mmap_mode="r")
        self._prefix_catcher_hash = np.load(split_dir / "prefix_catcher_hash.npy", mmap_mode="r")
        self._prefix_event_type = np.load(split_dir / "prefix_event_type.npy", mmap_mode="r")
        self._prefix_hierarchy = np.load(split_dir / "prefix_hierarchy.npy", mmap_mode="r")
        self._prefix_pitch_type = np.load(split_dir / "prefix_pitch_type.npy", mmap_mode="r")
        self._prefix_bat_side = np.load(split_dir / "prefix_bat_side.npy", mmap_mode="r")
        self._prefix_pitch_hand = np.load(split_dir / "prefix_pitch_hand.npy", mmap_mode="r")
        self._prefix_half_inning = np.load(split_dir / "prefix_half_inning.npy", mmap_mode="r")
        self._prefix_hit_traj = np.load(split_dir / "prefix_hit_traj.npy", mmap_mode="r")
        self._prefix_hit_hard = np.load(split_dir / "prefix_hit_hard.npy", mmap_mode="r")
        self._prefix_length = np.load(split_dir / "prefix_length.npy", mmap_mode="r")
        self._home_runs_remaining = np.load(split_dir / "home_runs_remaining.npy", mmap_mode="r")
        self._away_runs_remaining = np.load(split_dir / "away_runs_remaining.npy", mmap_mode="r")
        self._yrfi_mask = np.load(split_dir / "yrfi_mask.npy", mmap_mode="r")
        self._game_pks = np.load(split_dir / "game_pks.npy", mmap_mode="r")

        log.info("PreparedDataset loaded: %d samples, %d games from %s",
                 self.n_samples, self.n_games, split_dir)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        g = int(self._sample_to_game[idx])

        # Context sequences — pre-padded to MAX_CTX_LEN, split into SP/Team
        ctx_seqs = self._ctx_seqs[g].copy()  # (30, 200, 52) float16
        ctx_obs = self._ctx_obs[g].copy()    # (30, 200, 52) uint8
        ctx_mask = self._ctx_mask[g].copy()  # (30, 200) float32

        # Sanitize all float arrays — NaN sources: rating sequences (known),
        # float16 overflow in ctx_seqs, sparse weather/player data
        _nan = dict(nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        np.nan_to_num(ctx_seqs, **_nan)
        np.nan_to_num(ctx_mask, **_nan)

        player_history = self._player_history[g].copy()
        np.nan_to_num(player_history, **_nan)
        player_matchup = self._player_matchup[g].copy()
        np.nan_to_num(player_matchup, **_nan)
        flat_features = self._flat_features[g].copy()
        np.nan_to_num(flat_features, **_nan)
        if self._has_weather_asof:
            d = int(self._wx_decision_hour[idx])
            weather = self._weather_asof[g, d].copy()   # [7, 99] decision row
        else:
            weather = self._weather[g].copy()           # legacy [4, 22]
        np.nan_to_num(weather, **_nan)
        rating_home = self._rating_home[g].copy()
        np.nan_to_num(rating_home, **_nan)
        rating_away = self._rating_away[g].copy()
        np.nan_to_num(rating_away, **_nan)
        prefix_values = self._prefix_values[idx].copy()
        np.nan_to_num(prefix_values, **_nan)

        return {
            # SP home: indices 0:5
            "sp_home_seqs": torch.from_numpy(ctx_seqs[0:SP_GAMES].astype(np.float32)),
            "sp_home_obs_mask": torch.from_numpy(ctx_obs[0:SP_GAMES].astype(np.float32)),
            "sp_home_lengths": torch.from_numpy(self._ctx_lengths[g, 0:SP_GAMES].copy().astype(np.int64)),
            "sp_home_weights": torch.from_numpy(self._ctx_weights[g, 0:SP_GAMES].copy()),
            "sp_home_mask": torch.from_numpy(ctx_mask[0:SP_GAMES]),
            # SP away: indices 5:10
            "sp_away_seqs": torch.from_numpy(ctx_seqs[SP_GAMES:2*SP_GAMES].astype(np.float32)),
            "sp_away_obs_mask": torch.from_numpy(ctx_obs[SP_GAMES:2*SP_GAMES].astype(np.float32)),
            "sp_away_lengths": torch.from_numpy(self._ctx_lengths[g, SP_GAMES:2*SP_GAMES].copy().astype(np.int64)),
            "sp_away_weights": torch.from_numpy(self._ctx_weights[g, SP_GAMES:2*SP_GAMES].copy()),
            "sp_away_mask": torch.from_numpy(ctx_mask[SP_GAMES:2*SP_GAMES]),
            # Team home: indices 10:20
            "team_home_seqs": torch.from_numpy(ctx_seqs[2*SP_GAMES:2*SP_GAMES+TEAM_GAMES].astype(np.float32)),
            "team_home_obs_mask": torch.from_numpy(ctx_obs[2*SP_GAMES:2*SP_GAMES+TEAM_GAMES].astype(np.float32)),
            "team_home_lengths": torch.from_numpy(self._ctx_lengths[g, 2*SP_GAMES:2*SP_GAMES+TEAM_GAMES].copy().astype(np.int64)),
            "team_home_weights": torch.from_numpy(self._ctx_weights[g, 2*SP_GAMES:2*SP_GAMES+TEAM_GAMES].copy()),
            "team_home_mask": torch.from_numpy(ctx_mask[2*SP_GAMES:2*SP_GAMES+TEAM_GAMES]),
            "team_home_similarity": torch.from_numpy(self._ctx_similarity[g, 2*SP_GAMES:2*SP_GAMES+TEAM_GAMES].copy()),
            # Team away: indices 20:30
            "team_away_seqs": torch.from_numpy(ctx_seqs[2*SP_GAMES+TEAM_GAMES:].astype(np.float32)),
            "team_away_obs_mask": torch.from_numpy(ctx_obs[2*SP_GAMES+TEAM_GAMES:].astype(np.float32)),
            "team_away_lengths": torch.from_numpy(self._ctx_lengths[g, 2*SP_GAMES+TEAM_GAMES:].copy().astype(np.int64)),
            "team_away_weights": torch.from_numpy(self._ctx_weights[g, 2*SP_GAMES+TEAM_GAMES:].copy()),
            "team_away_mask": torch.from_numpy(ctx_mask[2*SP_GAMES+TEAM_GAMES:]),
            "team_away_similarity": torch.from_numpy(self._ctx_similarity[g, 2*SP_GAMES+TEAM_GAMES:].copy()),
            # Flat context
            "flat_features": torch.from_numpy(flat_features),
            # Weather
            "weather_temporal": torch.from_numpy(weather),
            # Ratings
            "rating_home": torch.from_numpy(rating_home),
            "rating_away": torch.from_numpy(rating_away),
            # Live prefix
            "prefix_values": torch.from_numpy(prefix_values.astype(np.float32)),
            "prefix_obs_mask": torch.from_numpy(self._prefix_obs[idx].copy().astype(np.float32)),
            "prefix_mask": torch.from_numpy(self._prefix_mask[idx].copy()),
            "prefix_batter_hash": torch.from_numpy(self._prefix_batter_hash[idx].copy().astype(np.int64)),
            "prefix_pitcher_hash": torch.from_numpy(self._prefix_pitcher_hash[idx].copy().astype(np.int64)),
            "prefix_catcher_hash": torch.from_numpy(self._prefix_catcher_hash[idx].copy().astype(np.int64)),
            "prefix_event_type": torch.from_numpy(self._prefix_event_type[idx].copy().astype(np.int64)),
            "prefix_hierarchy": torch.from_numpy(self._prefix_hierarchy[idx].copy().astype(np.int64)),
            "prefix_pitch_type_idx": torch.from_numpy(self._prefix_pitch_type[idx].copy().astype(np.int64)),
            "prefix_bat_side_idx": torch.from_numpy(self._prefix_bat_side[idx].copy().astype(np.int64)),
            "prefix_pitch_hand_idx": torch.from_numpy(self._prefix_pitch_hand[idx].copy().astype(np.int64)),
            "prefix_half_inning_idx": torch.from_numpy(self._prefix_half_inning[idx].copy().astype(np.int64)),
            "prefix_hit_trajectory_idx": torch.from_numpy(self._prefix_hit_traj[idx].copy().astype(np.int64)),
            "prefix_hit_hardness_idx": torch.from_numpy(self._prefix_hit_hard[idx].copy().astype(np.int64)),
            "prefix_length": torch.tensor(int(self._prefix_length[idx]), dtype=torch.long),
            # Player context
            "player_hashes": torch.from_numpy(self._player_hashes[g].copy()),
            "player_history": torch.from_numpy(player_history),
            "player_history_mask": torch.from_numpy(self._player_history_mask[g].copy().astype(np.float32)),
            "player_matchup": torch.from_numpy(player_matchup),
            # Targets
            "targets": {
                "home_win": torch.tensor(float(self._targets_game[g, 0]), dtype=torch.float32),
                "yrfi": torch.tensor(float(self._targets_game[g, 1]), dtype=torch.float32),
                "extra_innings": torch.tensor(float(self._targets_game[g, 2]), dtype=torch.float32),
                "total_runs": torch.tensor(float(self._targets_game[g, 3]), dtype=torch.float32),
                "home_runs_remaining": torch.tensor(float(self._home_runs_remaining[idx]), dtype=torch.float32),
                "away_runs_remaining": torch.tensor(float(self._away_runs_remaining[idx]), dtype=torch.float32),
                "player_hits": torch.from_numpy(self._targets_player[g, 0].copy()),
                "player_hr": torch.from_numpy(self._targets_player[g, 1].copy()),
                "player_so": torch.from_numpy(self._targets_player[g, 2].copy()),
                "player_hrbi": torch.from_numpy(self._targets_player[g, 3].copy()),
                "player_tb": torch.from_numpy(self._targets_player[g, 4].copy()),
                "player_sb": torch.from_numpy(self._targets_player[g, 5].copy()),
            },
            # Masks
            "yrfi_mask": torch.tensor(float(self._yrfi_mask[idx]), dtype=torch.float32),
            "player_mask": torch.from_numpy(self._player_mask[g].copy()),
            # Metadata
            "sample_weight": torch.tensor(float(self._sample_weight[g]), dtype=torch.float32),
            "game_pk": torch.tensor(int(self._game_pks[g]), dtype=torch.long),
        }


def prepared_collate_fn(batch: list[dict]) -> dict:
    """Fast collate for PreparedDataset — all tensors are fixed-size, just stack."""
    if not batch:
        return {}

    result = {}

    # All context sequences are pre-padded to MAX_CTX_LEN — just stack
    for prefix in ["sp_home", "sp_away", "team_home", "team_away"]:
        result[f"{prefix}_seqs"] = torch.stack([s[f"{prefix}_seqs"] for s in batch])
        result[f"{prefix}_attn_mask"] = torch.stack([s[f"{prefix}_mask"] for s in batch])
        result[f"{prefix}_obs_mask"] = torch.stack([s[f"{prefix}_obs_mask"] for s in batch])
        result[f"{prefix}_lengths"] = torch.stack([s[f"{prefix}_lengths"] for s in batch])
        result[f"{prefix}_weights"] = torch.stack([s[f"{prefix}_weights"] for s in batch])
        if f"{prefix}_similarity" in batch[0]:
            result[f"{prefix}_similarity"] = torch.stack([s[f"{prefix}_similarity"] for s in batch])

    # Flat features
    result["flat_features"] = torch.stack([s["flat_features"] for s in batch])

    # Ratings
    result["rating_home"] = torch.stack([s["rating_home"] for s in batch])
    result["rating_away"] = torch.stack([s["rating_away"] for s in batch])

    # Prefix (already fixed-size)
    for key in ["prefix_values", "prefix_obs_mask", "prefix_mask",
                "prefix_batter_hash", "prefix_pitcher_hash", "prefix_catcher_hash",
                "prefix_event_type", "prefix_hierarchy",
                "prefix_pitch_type_idx", "prefix_bat_side_idx", "prefix_pitch_hand_idx",
                "prefix_half_inning_idx", "prefix_hit_trajectory_idx", "prefix_hit_hardness_idx",
                "prefix_length"]:
        result[key] = torch.stack([s[key] for s in batch])

    # Player context
    result["player_hashes"] = torch.stack([s["player_hashes"] for s in batch])
    result["player_history"] = torch.stack([s["player_history"] for s in batch])
    result["player_history_mask"] = torch.stack([s["player_history_mask"] for s in batch])
    result["player_matchup"] = torch.stack([s["player_matchup"] for s in batch])

    # Targets (nested dict)
    target_keys = batch[0]["targets"].keys()
    result["targets"] = {key: torch.stack([s["targets"][key] for s in batch]) for key in target_keys}

    # Masks
    result["yrfi_mask"] = torch.stack([s["yrfi_mask"] for s in batch])
    result["player_mask"] = torch.stack([s["player_mask"] for s in batch])

    # Metadata
    result["sample_weight"] = torch.stack([s["sample_weight"] for s in batch])
    result["game_pk"] = torch.stack([s["game_pk"] for s in batch])

    # Weather
    result["weather_temporal"] = torch.stack([s["weather_temporal"] for s in batch])

    # Causal mask for prefix
    max_prefix_len = result["prefix_values"].shape[1]
    causal_mask = torch.tril(torch.ones(max_prefix_len, max_prefix_len, dtype=torch.float32))
    prefix_padding = result["prefix_mask"]
    result["prefix_causal_mask"] = causal_mask.unsqueeze(0) * prefix_padding.unsqueeze(1)

    return result


def load_prepared_datasets(
    prepared_dir: str | Path,
) -> tuple[PreparedDataset, PreparedDataset, PreparedDataset]:
    """Load prepared train/val/test datasets."""
    prepared_path = Path(prepared_dir)
    manifest_file = prepared_path / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"No manifest.json in {prepared_path}")

    train_ds = PreparedDataset(prepared_path / "train")
    val_ds = PreparedDataset(prepared_path / "val")
    test_ds = PreparedDataset(prepared_path / "test")
    return train_ds, val_ds, test_ds
