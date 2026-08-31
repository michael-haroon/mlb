"""Serialize/deserialize GameTransformerDataset pre-computed arrays.

Build once on a big-RAM instance, save to S3, load in seconds on the GPU
training instance — skipping the 30-minute pandas-based __init__.

Usage:
    # Build + save:
    python -m mlb_dl.dataset_cache build \
        --feature-store ./artifacts/feature_store \
        --output ./precomputed_datasets

    # Upload to S3:
    aws s3 sync ./precomputed_datasets s3://mlb-.../deep_learning/precomputed_datasets/

    # In train_unified.py:
    from .dataset_cache import load_cached_datasets
    train_ds, val_ds, test_ds = load_cached_datasets("./precomputed_datasets")
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_STATCAST_MIN_DATE = pd.Timestamp("2015-01-01")


def _compute_feature_store_fingerprint(fs_path: Path) -> str:
    """SHA256 of parquet file sizes + mtimes as a change-detection fingerprint."""
    h = hashlib.sha256()
    for f in sorted(fs_path.glob("*.parquet")):
        stat = f.stat()
        h.update(f"{f.name}:{stat.st_size}:{int(stat.st_mtime)}".encode())
    npz = fs_path / "rating_sequences.npz"
    if npz.exists():
        stat = npz.stat()
        h.update(f"rating_sequences.npz:{stat.st_size}:{int(stat.st_mtime)}".encode())
    return h.hexdigest()[:16]


def save_dataset(ds, output_dir: Path, split_name: str) -> None:
    """Save a GameTransformerDataset's pre-computed state to disk."""
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # --- Large numpy arrays (the expensive-to-compute data) ---
    np.save(split_dir / "pitch_cont_array.npy", ds._pitch_cont_array)
    np.save(split_dir / "pitch_obs_mask.npy", ds._pitch_obs_mask)
    np.save(split_dir / "batter_hash_array.npy", ds._batter_hash_array)
    np.save(split_dir / "pitcher_hash_array.npy", ds._pitcher_hash_array)
    np.save(split_dir / "catcher_hash_array.npy", ds._catcher_hash_array)
    np.save(split_dir / "pitch_type_array.npy", ds._pitch_type_array)
    np.save(split_dir / "bat_side_array.npy", ds._bat_side_array)
    np.save(split_dir / "pitch_hand_array.npy", ds._pitch_hand_array)
    np.save(split_dir / "half_inning_array.npy", ds._half_inning_array)
    np.save(split_dir / "hit_trajectory_array.npy", ds._hit_trajectory_array)
    np.save(split_dir / "hit_hardness_array.npy", ds._hit_hardness_array)
    np.save(split_dir / "event_type_array.npy", ds._event_type_array)
    np.save(split_dir / "hierarchy_array.npy", ds._hierarchy_array)
    np.save(split_dir / "score_home_array.npy", ds._score_home_array)
    np.save(split_dir / "score_away_array.npy", ds._score_away_array)
    np.save(split_dir / "batter_id_array.npy", ds._batter_id_array)
    np.save(split_dir / "is_top_array.npy", ds._is_top_array)
    np.save(split_dir / "at_bat_index_array.npy", ds._at_bat_index_array)
    np.save(split_dir / "at_bat_event_array.npy", ds._at_bat_event_array)

    # --- Player history arrays (consolidated into single files) ---
    player_ids = sorted(ds._player_hist_dates.keys())
    np.save(split_dir / "player_ids.npy", np.array(player_ids, dtype=np.int64))
    # Pack all player dates and stats into single files with offset index
    all_dates = []
    all_stats = []
    offsets = []  # (pid, start, end) triples
    cursor = 0
    for pid in player_ids:
        dates = ds._player_hist_dates[pid]
        stats = ds._player_hist_stat_arrays[pid]
        n = len(dates)
        all_dates.append(dates)
        all_stats.append(stats)
        offsets.append((pid, cursor, cursor + n))
        cursor += n
    if all_dates:
        np.save(split_dir / "player_dates_packed.npy",
                np.concatenate(all_dates))
        np.save(split_dir / "player_stats_packed.npy",
                np.concatenate(all_stats).astype(np.float32))
    else:
        np.save(split_dir / "player_dates_packed.npy", np.array([], dtype="datetime64[ns]"))
        np.save(split_dir / "player_stats_packed.npy", np.zeros((0, 25), dtype=np.float32))
    with open(split_dir / "player_offsets.json", "w") as f:
        json.dump(offsets, f)

    # --- Dicts as JSON-serializable structures ---
    # game_offsets: {game_pk: (start, end)}
    offsets_serializable = {str(k): list(v) for k, v in ds._game_offsets.items()}
    with open(split_dir / "game_offsets.json", "w") as f:
        json.dump(offsets_serializable, f)

    # samples: [(game_pk, prefix_len), ...]
    with open(split_dir / "samples.json", "w") as f:
        json.dump(ds.samples, f)

    # target_by_game: game_pk -> namedtuple-like row → serialize as dict
    targets_serializable = {}
    for gpk, row in ds.target_by_game.items():
        targets_serializable[str(gpk)] = {
            field: _safe_serialize(getattr(row, field))
            for field in row._fields
        }
    with open(split_dir / "target_by_game.json", "w") as f:
        json.dump(targets_serializable, f)

    # meta_by_game: game_pk -> pd.Series → serialize as dict
    meta_serializable = {}
    for gpk, row in ds.meta_by_game.items():
        meta_serializable[str(gpk)] = {
            k: _safe_serialize(v) for k, v in row.to_dict().items()
        }
    with open(split_dir / "meta_by_game.json", "w") as f:
        json.dump(meta_serializable, f)

    # game_lineups: {game_pk: set(player_ids)}
    lineups_serializable = {str(k): sorted(v) for k, v in ds._game_lineups.items()}
    with open(split_dir / "game_lineups.json", "w") as f:
        json.dump(lineups_serializable, f)

    # sp_games: {pitcher_id: [game_pk, ...]}
    sp_serializable = {str(k): v for k, v in ds._sp_games.items()}
    with open(split_dir / "sp_games.json", "w") as f:
        json.dump(sp_serializable, f)

    # game_date_by_pk
    dates_serializable = {str(k): str(v) for k, v in ds._game_date_by_pk.items()}
    with open(split_dir / "game_date_by_pk.json", "w") as f:
        json.dump(dates_serializable, f)

    # player_game_stats: {(pid, gpk): stats_dict}
    pgs_serializable = {
        f"{pid}_{gpk}": v for (pid, gpk), v in ds._player_game_stats.items()
    }
    with open(split_dir / "player_game_stats.json", "w") as f:
        json.dump(pgs_serializable, f)

    # sample_game_weights
    with open(split_dir / "sample_game_weights.json", "w") as f:
        json.dump({str(k): v for k, v in ds._sample_game_weights.items()}, f)

    # game_index (by_team as game_pk lists, game_to_idx, sp_by_pitcher)
    gi = ds.game_index
    gi_serializable = {
        "game_to_idx": {f"{k[0]}_{k[1]}": v for k, v in gi["game_to_idx"].items()},
        "sp_by_pitcher": {str(k): v for k, v in gi["sp_by_pitcher"].items()},
    }
    with open(split_dir / "game_index.json", "w") as f:
        json.dump(gi_serializable, f)

    # Weather temporal
    if ds._weather_temporal_by_pk:
        wt_dir = split_dir / "weather_temporal"
        wt_dir.mkdir(exist_ok=True)
        for gpk, arr in ds._weather_temporal_by_pk.items():
            np.save(wt_dir / f"{gpk}.npy", arr)

    # As-of weather + per-pitch decision offsets (stacked single files — one
    # .npy per game at [7,7,99] would mean ~26k tiny files per split)
    if getattr(ds, "_weather_asof_by_pk", None):
        pks = np.array(sorted(ds._weather_asof_by_pk), dtype=np.int64)
        np.savez(split_dir / "weather_asof.npz", pks=pks,
                 tensors=np.stack([ds._weather_asof_by_pk[int(p)] for p in pks]))
    if getattr(ds, "_wx_offsets_by_pk", None):
        pks = np.array(sorted(ds._wx_offsets_by_pk), dtype=np.int64)
        np.savez(split_dir / "wx_offsets.npz", pks=pks,
                 offsets=np.array([ds._wx_offsets_by_pk[int(p)] for p in pks],
                                  dtype=object))

    # Rating temporal
    if ds._rating_by_game_side:
        rt_dir = split_dir / "rating_temporal"
        rt_dir.mkdir(exist_ok=True)
        for (gpk, side), arr in ds._rating_by_game_side.items():
            np.save(rt_dir / f"{gpk}_{side}.npy", arr)
        with open(split_dir / "rating_meta.json", "w") as f:
            json.dump({"dim": ds._rating_dim, "steps": ds._rating_steps}, f)

    # Venue dimensions
    if ds._venue_dims_by_id:
        vd_serializable = {
            str(k): {kk: _safe_serialize(vv) for kk, vv in v.to_dict().items()}
            for k, v in ds._venue_dims_by_id.items()
        }
        with open(split_dir / "venue_dims.json", "w") as f:
            json.dump(vd_serializable, f)

    # Config
    from dataclasses import asdict
    config = {
        "ablation": asdict(ds.ablation),
        "spec": asdict(ds.spec),
        "standardizer": ds.standardizer.to_dict(),
        "include_pregame": ds.include_pregame,
        "include_live": ds.include_live,
        "max_date": str(ds.max_date),
    }
    with open(split_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("Saved %s: %d samples, %d pitches", split_name, len(ds.samples),
             len(ds._pitch_cont_array))


def _npz_rows_by_pk(path: Path, values_key: str, *, allow_pickle: bool = False) -> dict:
    """Map game_pk -> one row of `values_key`, reading each archive member exactly once.

    np.load on an .npz returns a LAZY NpzFile: every subscript re-reads and re-materialises
    the whole member. The call sites used to inline
    `{int(p): z[values_key][i] for i, p in enumerate(z["pks"])}`, which re-reads the values
    member once PER GAME -- and since `arr[i]` is a view that pins its parent through
    `.base`, every row kept a private full copy of the member alive.

    Measured on the 2026-08-31 train split, where tensors is (21384, 7, 7, 99) float32 =
    414.9 MB stored uncompressed at 0.45s per read: the comprehension projected to 2.7 hours
    and 8.9 TB retained. It OOM-killed precollate twice at 27.1 GB resident plus exactly
    32.0 GiB of swap, crossing 59 GB after only ~142 games.

    Hoisting the read makes every row a view onto ONE array, so the footprint is the member
    itself (415 MB) and the cost is a single sequential read.
    """
    with np.load(path, allow_pickle=allow_pickle) as z:
        pks = z["pks"]
        values = z[values_key]
    return {int(p): values[i] for i, p in enumerate(pks)}


def load_dataset(cache_dir: Path, split_name: str):
    """Load a cached GameTransformerDataset from pre-computed files.

    Returns a lightweight object with the same interface as GameTransformerDataset
    but without any pandas-based construction.
    """
    split_dir = cache_dir / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Cache split not found: {split_dir}")

    return CachedGameTransformerDataset(split_dir)


class CachedGameTransformerDataset:
    """Drop-in replacement for GameTransformerDataset loaded from pre-computed cache.

    Has identical __getitem__ behavior but loads in seconds instead of 30 minutes.
    """

    def __init__(self, split_dir: Path):
        import torch
        from .datasets import Standardizer, SequenceSpec
        from .game_transformer_dataset import (
            AblationConfig, PITCH_CONTINUOUS_COLS, MAX_PLAYERS_PER_GAME,
            PLAYER_STAT_DIM, FLAT_FEATURE_DIM, _hash_bucket,
        )

        t0 = time.time()

        # Config
        with open(split_dir / "config.json") as f:
            config = json.load(f)
        self.ablation = AblationConfig(**config["ablation"])
        self.spec = SequenceSpec(**config["spec"])
        self.standardizer = Standardizer.from_dict(config["standardizer"])
        self.include_pregame = config["include_pregame"]
        self.include_live = config["include_live"]
        self.max_date = pd.Timestamp(config["max_date"])

        # Large arrays — memory-map for minimal RAM usage
        self._pitch_cont_array = np.load(split_dir / "pitch_cont_array.npy", mmap_mode="r")
        self._pitch_obs_mask = np.load(split_dir / "pitch_obs_mask.npy", mmap_mode="r")
        self._batter_hash_array = np.load(split_dir / "batter_hash_array.npy", mmap_mode="r")
        self._pitcher_hash_array = np.load(split_dir / "pitcher_hash_array.npy", mmap_mode="r")
        self._catcher_hash_array = np.load(split_dir / "catcher_hash_array.npy", mmap_mode="r")
        self._pitch_type_array = np.load(split_dir / "pitch_type_array.npy", mmap_mode="r")
        self._bat_side_array = np.load(split_dir / "bat_side_array.npy", mmap_mode="r")
        self._pitch_hand_array = np.load(split_dir / "pitch_hand_array.npy", mmap_mode="r")
        self._half_inning_array = np.load(split_dir / "half_inning_array.npy", mmap_mode="r")
        self._hit_trajectory_array = np.load(split_dir / "hit_trajectory_array.npy", mmap_mode="r")
        self._hit_hardness_array = np.load(split_dir / "hit_hardness_array.npy", mmap_mode="r")
        self._event_type_array = np.load(split_dir / "event_type_array.npy", mmap_mode="r")
        self._hierarchy_array = np.load(split_dir / "hierarchy_array.npy", mmap_mode="r")
        self._score_home_array = np.load(split_dir / "score_home_array.npy", mmap_mode="r")
        self._score_away_array = np.load(split_dir / "score_away_array.npy", mmap_mode="r")
        self._batter_id_array = np.load(split_dir / "batter_id_array.npy", mmap_mode="r")
        self._is_top_array = np.load(split_dir / "is_top_array.npy", mmap_mode="r")
        self._at_bat_index_array = np.load(split_dir / "at_bat_index_array.npy", mmap_mode="r")
        self._at_bat_event_array = np.load(split_dir / "at_bat_event_array.npy", allow_pickle=True)

        # Samples
        with open(split_dir / "samples.json") as f:
            self.samples = [tuple(s) for s in json.load(f)]

        # Game offsets
        with open(split_dir / "game_offsets.json") as f:
            raw = json.load(f)
            self._game_offsets = {int(k): tuple(v) for k, v in raw.items()}

        # Targets
        with open(split_dir / "target_by_game.json") as f:
            raw = json.load(f)
            from collections import namedtuple
            if raw:
                fields = list(next(iter(raw.values())).keys())
                TargetRow = namedtuple("TargetRow", fields)
                self.target_by_game = {}
                for gpk_str, vals in raw.items():
                    for k in vals:
                        if k == "game_date":
                            vals[k] = pd.Timestamp(vals[k]) if vals[k] else pd.NaT
                    self.target_by_game[int(gpk_str)] = TargetRow(**vals)
            else:
                self.target_by_game = {}

        # Meta by game
        with open(split_dir / "meta_by_game.json") as f:
            raw = json.load(f)
            self.meta_by_game = {}
            for gpk_str, vals in raw.items():
                if "game_date" in vals and vals["game_date"]:
                    vals["game_date"] = pd.Timestamp(vals["game_date"])
                self.meta_by_game[int(gpk_str)] = pd.Series(vals)

        # Game date lookup
        with open(split_dir / "game_date_by_pk.json") as f:
            raw = json.load(f)
            self._game_date_by_pk = {int(k): pd.Timestamp(v) for k, v in raw.items()}

        # Game lineups
        with open(split_dir / "game_lineups.json") as f:
            raw = json.load(f)
            self._game_lineups = {int(k): set(v) for k, v in raw.items()}

        # SP games
        with open(split_dir / "sp_games.json") as f:
            raw = json.load(f)
            self._sp_games = {int(k): v for k, v in raw.items()}

        # Player game stats
        with open(split_dir / "player_game_stats.json") as f:
            raw = json.load(f)
            self._player_game_stats = {}
            for key_str, v in raw.items():
                pid, gpk = key_str.rsplit("_", 1)
                self._player_game_stats[(int(pid), int(gpk))] = v

        # Sample game weights
        with open(split_dir / "sample_game_weights.json") as f:
            raw = json.load(f)
            self._sample_game_weights = {int(k): v for k, v in raw.items()}

        # Game index
        with open(split_dir / "game_index.json") as f:
            raw = json.load(f)
            game_to_idx = {}
            for k, v in raw["game_to_idx"].items():
                tid, gpk = k.rsplit("_", 1)
                game_to_idx[(int(tid), int(gpk))] = v
            sp_by_pitcher = {int(k): v for k, v in raw["sp_by_pitcher"].items()}
            # by_team is reconstructed minimally (only game_to_idx is needed for decay)
            self.game_index = {
                "by_team": {},
                "game_to_idx": game_to_idx,
                "sp_by_pitcher": sp_by_pitcher,
            }

        # Player history arrays (consolidated format)
        self._player_hist_dates = {}
        self._player_hist_stat_arrays = {}
        offsets_file = split_dir / "player_offsets.json"
        if offsets_file.exists():
            with open(offsets_file) as f:
                offsets = json.load(f)
            all_dates = np.load(split_dir / "player_dates_packed.npy", allow_pickle=True)
            all_stats = np.load(split_dir / "player_stats_packed.npy")
            for pid, start, end in offsets:
                self._player_hist_dates[int(pid)] = all_dates[start:end]
                self._player_hist_stat_arrays[int(pid)] = all_stats[start:end]
        elif (split_dir / "player_history").exists():
            # Legacy per-file format fallback
            player_dir = split_dir / "player_history"
            pids = np.load(player_dir / "player_ids.npy")
            for pid in pids:
                pid = int(pid)
                dates_file = player_dir / f"{pid}_dates.npy"
                stats_file = player_dir / f"{pid}_stats.npy"
                if dates_file.exists() and stats_file.exists():
                    self._player_hist_dates[pid] = np.load(
                        dates_file, allow_pickle=True)
                    self._player_hist_stat_arrays[pid] = np.load(
                        stats_file, allow_pickle=True)

        # Weather temporal
        self._weather_temporal_by_pk = {}
        wt_dir = split_dir / "weather_temporal"
        if wt_dir.exists():
            for f in wt_dir.glob("*.npy"):
                gpk = int(f.stem)
                self._weather_temporal_by_pk[gpk] = np.load(f)

        # As-of weather + per-pitch decision offsets
        self._weather_asof_by_pk = {}
        asof_file = split_dir / "weather_asof.npz"
        if asof_file.exists():
            self._weather_asof_by_pk = _npz_rows_by_pk(asof_file, "tensors")
        self._wx_offsets_by_pk = {}
        off_file = split_dir / "wx_offsets.npz"
        if off_file.exists():
            # allow_pickle because the offsets are ragged per-pitch arrays, i.e. dtype=object.
            self._wx_offsets_by_pk = _npz_rows_by_pk(
                off_file, "offsets", allow_pickle=True)

        # Rating temporal
        self._rating_by_game_side = {}
        self._rating_dim = 0
        self._rating_steps = 10
        rt_dir = split_dir / "rating_temporal"
        if rt_dir.exists():
            meta_file = split_dir / "rating_meta.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    rmeta = json.load(f)
                    self._rating_dim = rmeta["dim"]
                    self._rating_steps = rmeta["steps"]
            for f in rt_dir.glob("*.npy"):
                parts = f.stem.rsplit("_", 1)
                gpk, side = int(parts[0]), parts[1]
                self._rating_by_game_side[(gpk, side)] = np.load(f)

        # Venue dimensions
        self._venue_dims_by_id = {}
        vd_file = split_dir / "venue_dims.json"
        if vd_file.exists():
            with open(vd_file) as f:
                raw = json.load(f)
                self._venue_dims_by_id = {int(k): pd.Series(v) for k, v in raw.items()}

        # Team games (needed for by_team in game_index — rebuild from meta)
        # For _get_team_context: need game_index["by_team"][team_id] as DataFrame
        self._rebuild_team_index()

        # _player_hist_by_id not needed — we use _player_hist_dates + _player_hist_stat_arrays

        elapsed = time.time() - t0
        log.info("Loaded cached %s: %d samples, %d pitches in %.1fs",
                 split_dir.name, len(self.samples), len(self._pitch_cont_array), elapsed)

    def _rebuild_team_index(self):
        """Rebuild game_index['by_team'] from meta_by_game for team context lookups."""
        by_team: dict[int, list] = {}
        for gpk, meta in self.meta_by_game.items():
            row_data = {"game_pk": gpk, "game_date": meta.get("game_date")}
            for col in ["home_team_id", "away_team_id", "season",
                        "probable_pitcher_home_id", "probable_pitcher_away_id"]:
                row_data[col] = meta.get(col)

            for side in ["home", "away"]:
                tid_col = f"{side}_team_id"
                tid = meta.get(tid_col)
                if tid is not None and not (isinstance(tid, float) and np.isnan(tid)):
                    tid = int(tid)
                    if tid not in by_team:
                        by_team[tid] = []
                    by_team[tid].append(row_data)

        for tid in by_team:
            df = pd.DataFrame(by_team[tid]).drop_duplicates("game_pk")
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
            by_team[tid] = df.sort_values("game_date").reset_index(drop=True)

        self.game_index["by_team"] = by_team

    # --- Delegate __len__ and __getitem__ to the real dataset's methods ---
    # We import and bind the instance methods from GameTransformerDataset

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Identical logic to GameTransformerDataset.__getitem__."""
        from .game_transformer_dataset import GameTransformerDataset
        # Bind the parent class's __getitem__ to this instance
        return GameTransformerDataset.__getitem__(self, idx)


# Bind all the private helper methods from GameTransformerDataset
# so __getitem__ can call them on CachedGameTransformerDataset instances.
def _bind_methods():
    from .game_transformer_dataset import GameTransformerDataset
    methods_to_bind = [
        "_get_sp_context", "_get_team_context", "_get_live_prefix",
        "_get_player_context", "_get_flat_features", "_build_targets",
        "_build_masks", "_build_player_targets", "_extract_game_sequences",
        "_load_game_pitches", "_empty_history_context",
        "_select_lineup_overlap", "_get_opposing_sp",
        "_compute_matchup_summary", "_get_weather_temporal",
        "_get_weather_asof_full", "_get_weather_asof_row", "_get_wx_decision_hour",
        "_get_rating_temporal",
    ]
    for name in methods_to_bind:
        if hasattr(GameTransformerDataset, name):
            setattr(CachedGameTransformerDataset, name, getattr(GameTransformerDataset, name))

_bind_methods()


def load_cached_datasets(
    cache_dir: str | Path,
) -> tuple:
    """Load train/val/test cached datasets. Returns (train_ds, val_ds, test_ds)."""
    cache_path = Path(cache_dir)
    manifest_file = cache_path / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"No manifest.json in {cache_path}")

    with open(manifest_file) as f:
        manifest = json.load(f)

    log.info("Loading cached datasets (fingerprint=%s, built=%s)",
             manifest.get("fingerprint", "?"), manifest.get("built_at", "?"))

    train_ds = load_dataset(cache_path, "train")
    val_ds = load_dataset(cache_path, "val")
    test_ds = load_dataset(cache_path, "test")

    return train_ds, val_ds, test_ds


def build_and_save(
    feature_store_path: str,
    output_dir: str,
    upload_s3: Optional[str] = None,
) -> None:
    """Build datasets from feature store and save to disk (+ optional S3 upload)."""
    from .train_unified import (
        _load_feature_store, _build_datasets, _log_memory, _COMPETITIVE_GAME_TYPES,
    )
    from .datasets import temporal_split_dates, Standardizer
    from .game_transformer_dataset import AblationConfig

    _setup_logging()

    fs_path = Path(feature_store_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Fingerprint for cache invalidation
    fingerprint = _compute_feature_store_fingerprint(fs_path)
    log.info("Feature store fingerprint: %s", fingerprint)

    # Load feature store
    log.info("Loading feature store from %s ...", fs_path)
    t0 = time.time()
    frames = _load_feature_store(str(fs_path))
    log.info("Feature store loaded in %.1fs", time.time() - t0)
    _log_memory("after load")

    # Temporal split
    train_end, val_end = temporal_split_dates(
        frames["game_targets"], min_date=_STATCAST_MIN_DATE
    )
    log.info("Temporal split: train < %s, val < %s", train_end.date(), val_end.date())

    # Build datasets
    ablation = AblationConfig()
    log.info("Building datasets (this takes ~25 minutes on 37M rows)...")
    t0 = time.time()
    train_ds, val_ds, test_ds = _build_datasets(frames, ablation, train_end, val_end)
    log.info("Datasets built in %.1f minutes", (time.time() - t0) / 60)
    _log_memory("after build")

    del frames
    gc.collect()

    # Save each split
    log.info("Saving datasets to %s ...", out_path)
    t0 = time.time()
    save_dataset(train_ds, out_path, "train")
    save_dataset(val_ds, out_path, "val")
    save_dataset(test_ds, out_path, "test")

    # Write manifest
    manifest = {
        "fingerprint": fingerprint,
        "built_at": pd.Timestamp.now().isoformat(),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "train_end": str(train_end),
        "val_end": str(val_end),
        # The population, not just the boundaries. `fingerprint` covers the feature store and
        # train_end/val_end cover the cut points, so a cache built before the 2015+ floor
        # existed is byte-different but manifest-identical to one built after it — exactly how
        # a 1950-2024 train split survived unnoticed. Record what was actually kept.
        "min_date": str(_STATCAST_MIN_DATE),
        "game_types": list(_COMPETITIVE_GAME_TYPES),
        "feature_store_path": str(fs_path),
    }
    with open(out_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("Saved in %.1fs", time.time() - t0)

    # Upload to S3 if requested
    if upload_s3:
        log.info("Uploading to %s ...", upload_s3)
        import subprocess
        result = subprocess.run(
            ["aws", "s3", "sync", str(out_path), upload_s3, "--quiet"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            log.info("S3 upload complete")
        else:
            log.error("S3 upload failed: %s", result.stderr)


def _safe_serialize(val):
    """Convert a value to JSON-serializable form."""
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, pd.Timestamp):
        return str(val) if pd.notna(val) else None
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    if isinstance(val, float) and np.isnan(val):
        return None
    return val


def _setup_logging():
    # Attach to the `mlb_dl` package logger, not this module's. `build_and_save` does its real
    # work inside train_unified._build_datasets, whose logger is a sibling — handlers on
    # `mlb_dl.dataset_cache` do not catch it, so every load and population-filter line was
    # silently dropped from cache_build.log. Records go where the build can be audited.
    #
    # Derive the package from `__package__`, not from `__name__`: this module's entry point is
    # `python -m mlb_dl.dataset_cache`, where `__name__` is "__main__" and any rsplit of it
    # yields "__main__" — a logger in a different tree, so train_unified still propagated
    # nowhere. `__package__` is "mlb_dl" both under -m and on ordinary import.
    pkg_log = logging.getLogger(__package__ or "mlb_dl")
    if pkg_log.handlers:
        return
    pkg_log.setLevel(logging.DEBUG)

    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "dataset_cache.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(message)s"))
    pkg_log.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    pkg_log.addHandler(sh)


def main():
    parser = argparse.ArgumentParser(description="Build and cache GameTransformer datasets")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build datasets from feature store and save")
    build.add_argument("--feature-store", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--upload-s3", default=None,
                       help="S3 URI to upload (e.g. s3://mlb-.../precomputed_datasets/)")

    args = parser.parse_args()
    if args.command == "build":
        build_and_save(args.feature_store, args.output, args.upload_s3)


if __name__ == "__main__":
    main()
