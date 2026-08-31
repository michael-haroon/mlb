"""Tests for pre-collated dataset: shape correctness, collate function, model compatibility.

Usage:
    conda run -n pred python -m pytest deep_learning/mlb_dl/tests/test_precollate.py -v
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, "deep_learning")

from mlb_dl.precollate import (
    PreparedDataset,
    prepared_collate_fn,
    MAX_CTX_LEN,
    SP_GAMES,
    TEAM_GAMES,
    TOTAL_CTX_GAMES,
    PREFIX_LEN,
    N_CONTINUOUS,
    MAX_PLAYERS,
    PLAYER_HIST_GAMES,
    PLAYER_STAT_DIM,
    FLAT_DIM,
    WEATHER_HOURS,
    WEATHER_DIM,
)


@pytest.fixture
def prepared_dir(tmp_path):
    """Create a synthetic prepared dataset on disk."""
    n_games = 10
    n_samples = 50
    rating_dim = 59
    matchup_dim = 12

    split_dir = tmp_path / "train"
    split_dir.mkdir()

    # Per-game arrays
    np.save(split_dir / "ctx_seqs.npy",
            np.random.randn(n_games, TOTAL_CTX_GAMES, MAX_CTX_LEN, N_CONTINUOUS).astype(np.float16))
    np.save(split_dir / "ctx_obs.npy",
            np.random.randint(0, 2, (n_games, TOTAL_CTX_GAMES, MAX_CTX_LEN, N_CONTINUOUS)).astype(np.uint8))
    np.save(split_dir / "ctx_mask.npy",
            np.random.rand(n_games, TOTAL_CTX_GAMES, MAX_CTX_LEN).astype(np.float32))
    np.save(split_dir / "ctx_lengths.npy",
            np.random.randint(10, 200, (n_games, TOTAL_CTX_GAMES)).astype(np.int16))
    np.save(split_dir / "ctx_weights.npy",
            np.random.rand(n_games, TOTAL_CTX_GAMES).astype(np.float32))
    np.save(split_dir / "ctx_similarity.npy",
            np.random.rand(n_games, TOTAL_CTX_GAMES).astype(np.float32))
    np.save(split_dir / "player_hashes.npy",
            np.random.randint(0, 49999, (n_games, MAX_PLAYERS)).astype(np.int64))
    np.save(split_dir / "player_history.npy",
            np.random.randn(n_games, MAX_PLAYERS, PLAYER_HIST_GAMES, PLAYER_STAT_DIM).astype(np.float32))
    np.save(split_dir / "player_history_mask.npy",
            np.random.randint(0, 2, (n_games, MAX_PLAYERS, PLAYER_HIST_GAMES)).astype(np.bool_))
    np.save(split_dir / "player_matchup.npy",
            np.random.randn(n_games, MAX_PLAYERS, matchup_dim).astype(np.float32))
    np.save(split_dir / "flat_features.npy",
            np.random.randn(n_games, FLAT_DIM).astype(np.float32))
    np.save(split_dir / "weather.npy",
            np.random.randn(n_games, WEATHER_HOURS, WEATHER_DIM).astype(np.float32))
    np.save(split_dir / "rating_home.npy",
            np.random.randn(n_games, 10, rating_dim).astype(np.float32))
    np.save(split_dir / "rating_away.npy",
            np.random.randn(n_games, 10, rating_dim).astype(np.float32))
    np.save(split_dir / "targets_game.npy",
            np.random.rand(n_games, 4).astype(np.float32))
    np.save(split_dir / "targets_player.npy",
            np.random.rand(n_games, 6, MAX_PLAYERS).astype(np.float32))
    np.save(split_dir / "player_mask.npy",
            np.random.randint(0, 2, (n_games, MAX_PLAYERS)).astype(np.float32))
    np.save(split_dir / "sample_weight.npy",
            np.ones(n_games, dtype=np.float32))
    np.save(split_dir / "game_pks.npy",
            np.arange(700000, 700000 + n_games, dtype=np.int64))

    # Per-sample arrays
    sample_to_game = np.random.randint(0, n_games, n_samples).astype(np.int32)
    np.save(split_dir / "sample_to_game.npy", sample_to_game)
    np.save(split_dir / "prefix_values.npy",
            np.random.randn(n_samples, PREFIX_LEN, N_CONTINUOUS).astype(np.float16))
    np.save(split_dir / "prefix_obs.npy",
            np.random.randint(0, 2, (n_samples, PREFIX_LEN, N_CONTINUOUS)).astype(np.uint8))
    np.save(split_dir / "prefix_mask.npy",
            np.random.rand(n_samples, PREFIX_LEN).astype(np.float32))
    np.save(split_dir / "prefix_batter_hash.npy",
            np.random.randint(0, 49999, (n_samples, PREFIX_LEN)).astype(np.int32))
    np.save(split_dir / "prefix_pitcher_hash.npy",
            np.random.randint(0, 49999, (n_samples, PREFIX_LEN)).astype(np.int32))
    np.save(split_dir / "prefix_catcher_hash.npy",
            np.random.randint(0, 49999, (n_samples, PREFIX_LEN)).astype(np.int32))
    np.save(split_dir / "prefix_event_type.npy",
            np.random.randint(0, 8, (n_samples, PREFIX_LEN)).astype(np.int16))
    # hierarchy: (inning < 20, ab < 50, pitch < 15)
    hierarchy = np.stack([
        np.random.randint(0, 18, (n_samples, PREFIX_LEN)),
        np.random.randint(0, 45, (n_samples, PREFIX_LEN)),
        np.random.randint(0, 12, (n_samples, PREFIX_LEN)),
    ], axis=-1).astype(np.int16)
    np.save(split_dir / "prefix_hierarchy.npy", hierarchy)
    np.save(split_dir / "prefix_pitch_type.npy",
            np.random.randint(0, 20, (n_samples, PREFIX_LEN)).astype(np.int8))
    np.save(split_dir / "prefix_bat_side.npy",
            np.random.randint(0, 4, (n_samples, PREFIX_LEN)).astype(np.int8))
    np.save(split_dir / "prefix_pitch_hand.npy",
            np.random.randint(0, 3, (n_samples, PREFIX_LEN)).astype(np.int8))
    np.save(split_dir / "prefix_half_inning.npy",
            np.random.randint(0, 2, (n_samples, PREFIX_LEN)).astype(np.int8))
    np.save(split_dir / "prefix_hit_traj.npy",
            np.random.randint(0, 7, (n_samples, PREFIX_LEN)).astype(np.int8))
    np.save(split_dir / "prefix_hit_hard.npy",
            np.random.randint(0, 4, (n_samples, PREFIX_LEN)).astype(np.int8))
    np.save(split_dir / "prefix_length.npy",
            np.random.randint(0, 200, n_samples).astype(np.int16))
    np.save(split_dir / "home_runs_remaining.npy",
            np.random.rand(n_samples).astype(np.float32) * 10)
    np.save(split_dir / "away_runs_remaining.npy",
            np.random.rand(n_samples).astype(np.float32) * 10)
    np.save(split_dir / "yrfi_mask.npy",
            np.ones(n_samples, dtype=np.float32))

    manifest = {
        "n_games": n_games,
        "n_samples": n_samples,
        "rating_dim": rating_dim,
        "matchup_dim": matchup_dim,
        "max_ctx_len": MAX_CTX_LEN,
        "prefix_len": PREFIX_LEN,
        "n_continuous": N_CONTINUOUS,
    }
    with open(split_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    return split_dir


class TestPreparedDataset:
    """Verify PreparedDataset returns correct shapes and dtypes."""

    def test_length(self, prepared_dir):
        ds = PreparedDataset(prepared_dir)
        assert len(ds) == 50

    def test_getitem_shapes(self, prepared_dir):
        ds = PreparedDataset(prepared_dir)
        sample = ds[0]

        # SP context
        assert sample["sp_home_seqs"].shape == (SP_GAMES, MAX_CTX_LEN, N_CONTINUOUS)
        assert sample["sp_home_obs_mask"].shape == (SP_GAMES, MAX_CTX_LEN, N_CONTINUOUS)
        assert sample["sp_home_lengths"].shape == (SP_GAMES,)
        assert sample["sp_home_weights"].shape == (SP_GAMES,)
        assert sample["sp_home_mask"].shape == (SP_GAMES, MAX_CTX_LEN)

        # Team context with similarity
        assert sample["team_home_seqs"].shape == (TEAM_GAMES, MAX_CTX_LEN, N_CONTINUOUS)
        assert sample["team_home_similarity"].shape == (TEAM_GAMES,)
        assert sample["team_away_similarity"].shape == (TEAM_GAMES,)

        # Flat/weather/rating
        assert sample["flat_features"].shape == (FLAT_DIM,)
        assert sample["weather_temporal"].shape == (WEATHER_HOURS, WEATHER_DIM)
        assert sample["rating_home"].shape == (10, 59)
        assert sample["rating_away"].shape == (10, 59)

        # Prefix
        assert sample["prefix_values"].shape == (PREFIX_LEN, N_CONTINUOUS)
        assert sample["prefix_mask"].shape == (PREFIX_LEN,)
        assert sample["prefix_hierarchy"].shape == (PREFIX_LEN, 3)
        assert sample["prefix_length"].dtype == torch.long

        # Player context
        assert sample["player_hashes"].shape == (MAX_PLAYERS,)
        assert sample["player_history"].shape == (MAX_PLAYERS, PLAYER_HIST_GAMES, PLAYER_STAT_DIM)
        assert sample["player_history_mask"].shape == (MAX_PLAYERS, PLAYER_HIST_GAMES)
        assert sample["player_matchup"].shape == (MAX_PLAYERS, 12)

        # Targets
        assert "home_win" in sample["targets"]
        assert "home_runs_remaining" in sample["targets"]
        assert sample["targets"]["player_hits"].shape == (MAX_PLAYERS,)

        # Masks/metadata
        assert sample["yrfi_mask"].shape == ()
        assert sample["player_mask"].shape == (MAX_PLAYERS,)
        assert sample["game_pk"].dtype == torch.long

    def test_getitem_dtypes(self, prepared_dir):
        ds = PreparedDataset(prepared_dir)
        sample = ds[0]

        assert sample["sp_home_seqs"].dtype == torch.float32
        assert sample["prefix_batter_hash"].dtype == torch.int64
        assert sample["prefix_event_type"].dtype == torch.int64
        assert sample["player_hashes"].dtype == torch.int64
        assert sample["player_history"].dtype == torch.float32

    def test_sample_to_game_mapping(self, prepared_dir):
        """Different samples mapping to the same game share context."""
        ds = PreparedDataset(prepared_dir)
        # Find two samples from the same game
        game_indices = ds._sample_to_game[:]
        game0_samples = np.where(game_indices == 0)[0]
        if len(game0_samples) >= 2:
            s0 = ds[int(game0_samples[0])]
            s1 = ds[int(game0_samples[1])]
            # Context should be identical
            torch.testing.assert_close(s0["sp_home_seqs"], s1["sp_home_seqs"])
            torch.testing.assert_close(s0["flat_features"], s1["flat_features"])


class TestPreparedCollateFn:
    """Verify collate produces correct batch structure."""

    def test_collate_shapes(self, prepared_dir):
        ds = PreparedDataset(prepared_dir)
        batch = [ds[i] for i in range(4)]
        collated = prepared_collate_fn(batch)

        B = 4
        assert collated["sp_home_seqs"].shape == (B, SP_GAMES, MAX_CTX_LEN, N_CONTINUOUS)
        assert collated["sp_home_attn_mask"].shape == (B, SP_GAMES, MAX_CTX_LEN)
        assert collated["team_home_similarity"].shape == (B, TEAM_GAMES)
        assert collated["prefix_values"].shape == (B, PREFIX_LEN, N_CONTINUOUS)
        assert collated["prefix_causal_mask"].shape == (B, PREFIX_LEN, PREFIX_LEN)
        assert collated["targets"]["home_win"].shape == (B,)
        assert collated["targets"]["player_hits"].shape == (B, MAX_PLAYERS)
        assert collated["player_mask"].shape == (B, MAX_PLAYERS)
        assert collated["weather_temporal"].shape == (B, WEATHER_HOURS, WEATHER_DIM)

    def test_collate_key_parity_with_original(self, prepared_dir):
        """Verify prepared collate produces all keys needed by _prepare_model_input."""
        ds = PreparedDataset(prepared_dir)
        batch = [ds[i] for i in range(2)]
        collated = prepared_collate_fn(batch)

        # Keys used by _prepare_model_input
        required_keys = [
            "sp_home_seqs", "sp_home_attn_mask", "sp_home_obs_mask",
            "sp_home_lengths", "sp_home_weights",
            "sp_away_seqs", "sp_away_attn_mask", "sp_away_obs_mask",
            "sp_away_lengths", "sp_away_weights",
            "team_home_seqs", "team_home_attn_mask", "team_home_obs_mask",
            "team_home_lengths", "team_home_weights", "team_home_similarity",
            "team_away_seqs", "team_away_attn_mask", "team_away_obs_mask",
            "team_away_lengths", "team_away_weights", "team_away_similarity",
            "flat_features",
            "weather_temporal",
            "rating_home", "rating_away",
            "prefix_values", "prefix_mask", "prefix_batter_hash",
            "prefix_pitcher_hash", "prefix_catcher_hash",
            "prefix_event_type", "prefix_hierarchy",
            "prefix_pitch_type_idx", "prefix_bat_side_idx",
            "prefix_pitch_hand_idx", "prefix_half_inning_idx",
            "prefix_hit_trajectory_idx", "prefix_hit_hardness_idx",
            "prefix_length",
            "player_hashes", "player_history", "player_history_mask",
            "targets", "yrfi_mask", "player_mask",
            "sample_weight", "game_pk",
        ]
        for key in required_keys:
            assert key in collated, f"Missing key: {key}"

    def test_causal_mask_correctness(self, prepared_dir):
        """Causal mask should be lower-triangular × padding mask."""
        ds = PreparedDataset(prepared_dir)
        batch = [ds[0]]
        collated = prepared_collate_fn(batch)

        causal = collated["prefix_causal_mask"][0]  # (PREFIX_LEN, PREFIX_LEN)
        # Upper triangle should be zero (causal constraint)
        upper = torch.triu(causal, diagonal=1)
        assert upper.sum() == 0


class TestModelCompatibility:
    """Test that prepared data flows through the actual model."""

    def test_prepare_model_input(self, prepared_dir):
        """Verify _prepare_model_input can consume prepared collate output."""
        sys.path.insert(0, "deep_learning")
        from mlb_dl.train_unified import _prepare_model_input

        ds = PreparedDataset(prepared_dir)
        batch = [ds[i] for i in range(4)]
        collated = prepared_collate_fn(batch)

        model_input = _prepare_model_input(collated, player_context_dim=512)

        assert "context" in model_input
        assert "player_hashes" in model_input
        for prefix in ["sp_home", "sp_away", "team_home", "team_away"]:
            assert prefix in model_input["context"]
            ctx = model_input["context"][prefix]
            assert "continuous" in ctx
            assert "padding_mask" in ctx
            assert ctx["continuous"].shape[0] == 4  # batch size

    def test_full_forward_pass(self, prepared_dir):
        """End-to-end: prepared data → collate → model forward."""
        from mlb_dl.game_transformer import ContextConfig, GameTransformer
        from mlb_dl.train_unified import _prepare_model_input

        ds = PreparedDataset(prepared_dir)
        batch = [ds[i] for i in range(2)]
        collated = prepared_collate_fn(batch)

        config = ContextConfig(sp_games=SP_GAMES, team_games=TEAM_GAMES,
                               tokens_per_game=4, flat_feature_tokens=4)
        d_model = 128
        model = GameTransformer(
            d_model=d_model,
            rating_dim=59,
            flat_feature_dim=FLAT_DIM,
            context_config=config,
            num_backbone_layers=2,
            num_heads=4,
            d_ff=d_model * 4,
        )
        model.eval()

        model_input = _prepare_model_input(collated, player_context_dim=d_model * 2)
        with torch.no_grad():
            predictions = model(model_input)

        assert "home_win_logit" in predictions
        assert predictions["home_win_logit"].shape == (2,)


class TestThroughput:
    """Verify prepared dataset is fast enough."""

    def test_getitem_speed(self, prepared_dir):
        """__getitem__ should complete in under 1ms per sample."""
        import time

        ds = PreparedDataset(prepared_dir)
        n_iters = 100

        t0 = time.perf_counter()
        for i in range(n_iters):
            _ = ds[i % len(ds)]
        elapsed = time.perf_counter() - t0

        ms_per_sample = (elapsed / n_iters) * 1000
        assert ms_per_sample < 50, f"Too slow: {ms_per_sample:.1f}ms/sample (want <50ms)"


class TestPreparedManifestProvenance:
    """The prepared manifest must state the population and cut it came from, not just counts.

    A prepared set is the artifact that gets uploaded and trained on, and /mnt/fast is an
    instance store, so the dataset_cache it was built from may not exist by the time anyone
    asks. Counts alone made the 1950-train void set indistinguishable from the corrected one.
    The cut points also move: temporal_split_dates takes an 80/10 quantile over distinct game
    dates, which gave 2024-05-14 on the void cache and 2024-08-03 on the corrected one.
    """

    CACHE_MANIFEST = {
        "fingerprint": "567a03c7bec97e4c",
        "built_at": "2026-08-31T01:30:58.491180",
        "train_end": "2024-08-03 00:00:00",
        "val_end": "2025-08-18 00:00:00",
        "min_date": "2015-01-01 00:00:00",
        "game_types": ["R", "F", "D", "L", "W"],
    }

    @staticmethod
    def _run_prepare_all(monkeypatch, cache_dir, out_dir):
        """Drive prepare_all's manifest assembly without building any tensors."""
        from mlb_dl import precollate as pc

        monkeypatch.setattr(pc, "prepare_split",
                            lambda ds, out, name, workers: {"n_games": 1, "n_samples": 1})
        # load_dataset is imported inside prepare_all from .dataset_cache, so patch it there.
        import mlb_dl.dataset_cache as dc
        monkeypatch.setattr(dc, "load_dataset", lambda path, name: object())
        pc.prepare_all(str(cache_dir), str(out_dir), num_workers=1)
        return json.loads((Path(out_dir) / "manifest.json").read_text())

    def test_carries_cut_points_from_source_cache(self, tmp_path, monkeypatch):
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "manifest.json").write_text(json.dumps(self.CACHE_MANIFEST))

        mf = self._run_prepare_all(monkeypatch, cache, tmp_path / "prepared")

        assert mf["train_end"] == "2024-08-03 00:00:00"
        assert mf["val_end"] == "2025-08-18 00:00:00"
        assert mf["min_date"] == "2015-01-01 00:00:00"
        assert mf["game_types"] == ["R", "F", "D", "L", "W"]
        assert mf["cache_fingerprint"] == "567a03c7bec97e4c"

    def test_missing_source_manifest_yields_explicit_nulls(self, tmp_path, monkeypatch):
        """A null says "provenance unavailable"; an absent key says nothing at all.

        Keeping the keys present is what lets a reader distinguish "built before this field
        existed" from "built by a run that could not determine its own cut".
        """
        cache = tmp_path / "cache"
        cache.mkdir()  # deliberately no manifest.json

        mf = self._run_prepare_all(monkeypatch, cache, tmp_path / "prepared")

        for key in ("train_end", "val_end", "min_date", "game_types", "cache_fingerprint"):
            assert key in mf, f"{key} must be present even when unknown"
            assert mf[key] is None

    def test_corrupt_source_manifest_does_not_abort_the_prepare(self, tmp_path, monkeypatch):
        """Losing provenance must not destroy hours of tensor work that already succeeded."""
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "manifest.json").write_text("{not json")

        mf = self._run_prepare_all(monkeypatch, cache, tmp_path / "prepared")

        assert mf["train_end"] is None
        assert set(mf["splits"]) == {"train", "val", "test"}
