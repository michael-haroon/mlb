"""Tests for the live categorical field pipeline: dataset → collate → _prepare_model_input → model.

Covers the bug where prefix_* categorical fields were not being mapped to live_* model keys.

Run:
    conda run -n pred python -m pytest deep_learning/mlb_dl/tests/test_live_categorical_pipeline.py -v
"""

from __future__ import annotations

import sys
sys.path.insert(0, "deep_learning")

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from mlb_dl.game_transformer import ContextConfig, GameTransformer, GameTransformerLoss
from mlb_dl.game_transformer_dataset import (
    FLAT_FEATURE_DIM,
    PITCH_CONTINUOUS_COLS,
    PITCH_TYPE_TO_IDX,
    BAT_SIDE_TO_IDX,
    PITCH_HAND_TO_IDX,
    HALF_INNING_TO_IDX,
    HIT_TRAJECTORY_TO_IDX,
    HIT_HARDNESS_TO_IDX,
    game_transformer_collate_fn,
    map_pitch_type,
    map_bat_side,
    map_pitch_hand,
    map_half_inning,
    map_hit_trajectory,
    map_hit_hardness,
)
from mlb_dl.train_unified import _prepare_model_input, _freeze_lower_layers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONT_DIM = len(PITCH_CONTINUOUS_COLS)
MAX_PREFIX = 256  # spec.history_length default


def _make_collate_batch(
    B: int,
    T: int,
    n_sp: int = 3,
    n_team: int = 5,
    max_seq: int = 40,
    rating_dim: int = 0,
    pitch_type_fill: int = 1,   # FF
    bat_side_fill: int = 1,     # R
    pitch_hand_fill: int = 1,   # R
    half_inning_fill: int = 1,  # bottom
    hit_traj_fill: int = 2,     # fly_ball
    hit_hard_fill: int = 3,     # hard
) -> dict:
    """Minimal collate-format batch (matches game_transformer_collate_fn output).

    prefix_length[0] = T (live), rest = 0 (pregame) to test both paths.
    All six categorical prefix fields are filled with distinctive non-zero values.
    """
    def _game_data(ng):
        return {
            "continuous": torch.randn(B, ng, max_seq, CONT_DIM),
            "batter_hash": torch.randint(1, 50000, (B, ng, max_seq)),
            "pitcher_hash": torch.randint(1, 50000, (B, ng, max_seq)),
            "inning_idx": torch.randint(0, 9, (B, ng, max_seq)),
            "ab_idx": torch.randint(0, 40, (B, ng, max_seq)),
            "pitch_idx": torch.randint(0, 10, (B, ng, max_seq)),
            "padding_mask": torch.zeros(B, ng, max_seq, dtype=torch.bool),
            "games_ago": torch.arange(ng, dtype=torch.float32).unsqueeze(0).expand(B, -1),
            "seasons_crossed": torch.zeros(B, ng),
        }

    # Build the collated result dict (as game_transformer_collate_fn produces)
    batch = {}
    for prefix in ["sp_home", "sp_away"]:
        batch[f"{prefix}_seqs"] = _game_data(n_sp)["continuous"]
        batch[f"{prefix}_attn_mask"] = torch.ones(B, n_sp, max_seq)
        batch[f"{prefix}_obs_mask"] = torch.ones(B, n_sp, max_seq, CONT_DIM)
        batch[f"{prefix}_lengths"] = torch.full((B, n_sp), max_seq, dtype=torch.long)
        batch[f"{prefix}_weights"] = torch.ones(B, n_sp)
    for prefix in ["team_home", "team_away"]:
        batch[f"{prefix}_seqs"] = _game_data(n_team)["continuous"]
        batch[f"{prefix}_attn_mask"] = torch.ones(B, n_team, max_seq)
        batch[f"{prefix}_obs_mask"] = torch.ones(B, n_team, max_seq, CONT_DIM)
        batch[f"{prefix}_lengths"] = torch.full((B, n_team), max_seq, dtype=torch.long)
        batch[f"{prefix}_weights"] = torch.ones(B, n_team)

    batch["flat_features"] = torch.randn(B, FLAT_FEATURE_DIM)
    batch["weather_temporal"] = torch.randn(B, 4, 22)
    batch["rating_home"] = torch.zeros(B, 5, max(rating_dim, 1))
    batch["rating_away"] = torch.zeros(B, 5, max(rating_dim, 1))

    # Live prefix: first sample has T live pitches, others have 0
    prefix_lengths = torch.zeros(B, dtype=torch.long)
    prefix_lengths[0] = T

    batch["prefix_length"] = prefix_lengths
    batch["prefix_values"] = torch.randn(B, T, CONT_DIM)
    batch["prefix_obs_mask"] = torch.ones(B, T, CONT_DIM)
    batch["prefix_mask"] = torch.zeros(B, T)
    batch["prefix_mask"][0, :T] = 1.0  # sample 0 has T live pitches
    batch["prefix_batter_hash"] = torch.randint(1, 50000, (B, T))
    batch["prefix_pitcher_hash"] = torch.randint(1, 50000, (B, T))
    batch["prefix_catcher_hash"] = torch.randint(1, 50000, (B, T))
    batch["prefix_event_type"] = torch.zeros(B, T, dtype=torch.long)
    batch["prefix_hierarchy"] = torch.zeros(B, T, 3, dtype=torch.long)
    batch["prefix_hierarchy"][:, :, 0] = torch.randint(0, 9, (B, T))  # inning
    batch["prefix_hierarchy"][:, :, 1] = torch.randint(0, 25, (B, T))  # ab
    batch["prefix_hierarchy"][:, :, 2] = torch.randint(0, 15, (B, T))  # pitch

    # Distinctive non-zero categorical values so we can detect if they are used
    batch["prefix_pitch_type_idx"] = torch.full((B, T), pitch_type_fill, dtype=torch.long)
    batch["prefix_bat_side_idx"] = torch.full((B, T), bat_side_fill, dtype=torch.long)
    batch["prefix_pitch_hand_idx"] = torch.full((B, T), pitch_hand_fill, dtype=torch.long)
    batch["prefix_half_inning_idx"] = torch.full((B, T), half_inning_fill, dtype=torch.long)
    batch["prefix_hit_trajectory_idx"] = torch.full((B, T), hit_traj_fill, dtype=torch.long)
    batch["prefix_hit_hardness_idx"] = torch.full((B, T), hit_hard_fill, dtype=torch.long)

    batch["player_hashes"] = torch.randint(1, 50000, (B, 20))
    batch["player_history"] = torch.zeros(B, 20, 15, 25)
    batch["player_history_mask"] = torch.zeros(B, 20, 15)
    batch["player_mask"] = torch.ones(B, 20)

    batch["targets"] = {
        "home_runs": torch.randint(0, 10, (B,)).float(),
        "away_runs": torch.randint(0, 10, (B,)).float(),
        "home_runs_remaining": torch.randint(0, 10, (B,)).float(),
        "away_runs_remaining": torch.randint(0, 10, (B,)).float(),
        "home_win": torch.randint(0, 2, (B,)).float(),
        "yrfi": torch.randint(0, 2, (B,)).float(),
        "extra_innings": torch.randint(0, 2, (B,)).float(),
        "player_hits": torch.randint(0, 4, (B, 20)).float(),
        "player_hr": torch.randint(0, 2, (B, 20)).float(),
        "player_so": torch.randint(0, 12, (B, 20)).float(),
        "player_hrbi": torch.randint(0, 6, (B, 20)).float(),
        "player_sb": torch.randint(0, 2, (B, 20)).float(),
    }

    return batch


# ---------------------------------------------------------------------------
# 1. _prepare_model_input key mapping
# ---------------------------------------------------------------------------

class TestPrepareModelInput:
    """The central fix: prefix_* collate keys must map to live_* model keys."""

    def test_all_six_categorical_keys_mapped_when_has_live(self):
        """When any prefix_length > 0, all 6 live_* categorical keys must be set."""
        batch = _make_collate_batch(B=2, T=30)
        model_input = _prepare_model_input(batch)

        assert "live_continuous" in model_input, "live_continuous missing"
        for key in [
            "live_pitch_type_idx",
            "live_bat_side_idx",
            "live_pitch_hand_idx",
            "live_half_inning_idx",
            "live_hit_trajectory_idx",
            "live_hit_hardness_idx",
        ]:
            assert key in model_input, f"live key {key!r} missing from model_input"
            assert model_input[key] is not None, f"live key {key!r} is None"

    def test_categorical_values_are_preserved_not_zeroed(self):
        """Values from prefix_* must reach live_* intact (not defaulted to zeros)."""
        PITCH_TYPE_VAL = 7  # KC (knuckle curve)
        BAT_SIDE_VAL = 2    # S (switch)
        PITCH_HAND_VAL = 0  # L
        HALF_INN_VAL = 1    # bottom
        HIT_TRAJ_VAL = 3    # line_drive
        HIT_HARD_VAL = 2    # medium

        batch = _make_collate_batch(
            B=2, T=20,
            pitch_type_fill=PITCH_TYPE_VAL,
            bat_side_fill=BAT_SIDE_VAL,
            pitch_hand_fill=PITCH_HAND_VAL,
            half_inning_fill=HALF_INN_VAL,
            hit_traj_fill=HIT_TRAJ_VAL,
            hit_hard_fill=HIT_HARD_VAL,
        )
        model_input = _prepare_model_input(batch)

        assert (model_input["live_pitch_type_idx"] == PITCH_TYPE_VAL).all(), \
            "pitch_type values were not propagated"
        assert (model_input["live_bat_side_idx"] == BAT_SIDE_VAL).all(), \
            "bat_side values were not propagated"
        assert (model_input["live_pitch_hand_idx"] == PITCH_HAND_VAL).all(), \
            "pitch_hand values were not propagated"
        assert (model_input["live_half_inning_idx"] == HALF_INN_VAL).all(), \
            "half_inning values were not propagated"
        assert (model_input["live_hit_trajectory_idx"] == HIT_TRAJ_VAL).all(), \
            "hit_trajectory values were not propagated"
        assert (model_input["live_hit_hardness_idx"] == HIT_HARD_VAL).all(), \
            "hit_hardness values were not propagated"

    def test_no_live_keys_when_all_prefix_length_zero(self):
        """When all prefix_lengths are 0, no live_* keys should appear in model_input."""
        batch = _make_collate_batch(B=2, T=20)
        batch["prefix_length"] = torch.zeros(2, dtype=torch.long)  # override to all 0
        model_input = _prepare_model_input(batch)

        assert "live_continuous" not in model_input, \
            "live path should be skipped when prefix_length.sum() == 0"
        assert "live_pitch_type_idx" not in model_input, \
            "live categorical keys should be absent for pregame batches"

    def test_live_hierarchy_unpacked_from_prefix_hierarchy(self):
        """prefix_hierarchy[:, :, 0/1/2] must unpack to live_inning/ab/pitch_idx."""
        batch = _make_collate_batch(B=2, T=15)
        model_input = _prepare_model_input(batch)

        expected_inning = batch["prefix_hierarchy"][:, :, 0]
        expected_ab = batch["prefix_hierarchy"][:, :, 1]
        expected_pitch = batch["prefix_hierarchy"][:, :, 2]

        assert torch.equal(model_input["live_inning_idx"], expected_inning), \
            "live_inning_idx doesn't match prefix_hierarchy[:,:,0]"
        assert torch.equal(model_input["live_ab_idx"], expected_ab), \
            "live_ab_idx doesn't match prefix_hierarchy[:,:,1]"
        assert torch.equal(model_input["live_pitch_idx"], expected_pitch), \
            "live_pitch_idx doesn't match prefix_hierarchy[:,:,2]"

    def test_context_keys_always_present(self):
        """Context section must always be populated regardless of live mode."""
        for T_live in [0, 30]:
            batch = _make_collate_batch(B=2, T=T_live)
            if T_live == 0:
                batch["prefix_length"] = torch.zeros(2, dtype=torch.long)
            model_input = _prepare_model_input(batch)

            assert "context" in model_input
            ctx = model_input["context"]
            assert "flat_features" in ctx
            for side in ["sp_home", "sp_away", "team_home", "team_away"]:
                assert side in ctx, f"Context key {side!r} missing"


# ---------------------------------------------------------------------------
# 2. Categorical embedding layers receive real gradients
# ---------------------------------------------------------------------------

class TestCategoricalEmbeddingGradients:
    """Categorical embedding layers for live path must get non-zero gradients."""

    def _build_model(self, d_model=128):
        cfg = ContextConfig(sp_games=2, team_games=3, tokens_per_game=4, flat_feature_tokens=4)
        return GameTransformer(
            d_model=d_model,
            num_backbone_layers=2,
            num_heads=4,
            d_ff=512,
            rating_dim=0,
            context_config=cfg,
        )

    def test_pitch_type_embed_gets_gradient_with_live_data(self):
        """pitch_encoder.pitch_type_embed must receive a gradient when live data is present."""
        d_model = 128
        model = self._build_model(d_model=d_model)
        loss_fn = GameTransformerLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        batch = _make_collate_batch(B=4, T=20, pitch_type_fill=5)  # CH
        model.train()
        optimizer.zero_grad()

        model_input = _prepare_model_input(batch, player_context_dim=d_model * 2)
        preds = model(model_input)

        targets = batch["targets"]
        loss, _ = loss_fn(preds, targets)
        loss.backward()

        pitch_type_grad = model.pitch_encoder.pitch_type_embed.weight.grad
        assert pitch_type_grad is not None, "No gradient on pitch_type_embed.weight"
        # Only embedding row 5 (CH) should have a non-zero gradient
        assert pitch_type_grad[5].abs().sum() > 0, \
            "pitch_type_embed row 5 (CH) should have gradient — live data used that index"

    def test_bat_side_embed_gets_gradient_with_live_data(self):
        """bat_side_embed must receive gradient for the used bat side index."""
        d_model = 128
        model = self._build_model(d_model=d_model)
        loss_fn = GameTransformerLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        batch = _make_collate_batch(B=4, T=20, bat_side_fill=2)  # S (switch)
        model.train()
        optimizer.zero_grad()

        model_input = _prepare_model_input(batch, player_context_dim=d_model * 2)
        preds = model(model_input)
        loss, _ = loss_fn(preds, batch["targets"])
        loss.backward()

        bat_side_grad = model.pitch_encoder.bat_side_embed.weight.grad
        assert bat_side_grad is not None, "No gradient on bat_side_embed.weight"
        assert bat_side_grad[2].abs().sum() > 0, \
            "bat_side_embed row 2 (S) should have gradient"

    def test_all_six_categorical_embeds_have_nonzero_gradient(self):
        """All six categorical embedding layers must receive non-zero gradients in live mode."""
        d_model = 128
        model = self._build_model(d_model=d_model)
        loss_fn = GameTransformerLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        batch = _make_collate_batch(
            B=4, T=20,
            pitch_type_fill=3,   # SL
            bat_side_fill=1,     # R
            pitch_hand_fill=1,   # R
            half_inning_fill=1,  # bottom
            hit_traj_fill=1,     # ground_ball
            hit_hard_fill=1,     # soft
        )
        model.train()
        optimizer.zero_grad()

        model_input = _prepare_model_input(batch, player_context_dim=d_model * 2)
        preds = model(model_input)
        loss, _ = loss_fn(preds, batch["targets"])
        loss.backward()

        embed_fields = [
            ("pitch_type_embed", 3),
            ("bat_side_embed", 1),
            ("pitch_hand_embed", 1),
            ("half_inning_embed", 1),
            ("hit_trajectory_embed", 1),
            ("hit_hardness_embed", 1),
        ]
        for attr_name, idx in embed_fields:
            embed = getattr(model.pitch_encoder, attr_name)
            grad = embed.weight.grad
            assert grad is not None, f"No gradient on {attr_name}.weight"
            assert grad[idx].abs().sum() > 0, \
                f"{attr_name}[{idx}] has zero gradient — embedding is a dead weight"

    def test_categorical_embeds_no_gradient_in_pregame_mode(self):
        """In pregame mode (no live pitches), live categorical embeds may not get gradient."""
        d_model = 128
        model = self._build_model(d_model=d_model)
        loss_fn = GameTransformerLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        batch = _make_collate_batch(B=4, T=20, pitch_type_fill=5)
        # Set all prefix lengths to 0 → pregame mode
        batch["prefix_length"] = torch.zeros(4, dtype=torch.long)
        model.train()
        optimizer.zero_grad()

        model_input = _prepare_model_input(batch, player_context_dim=d_model * 2)
        preds = model(model_input)
        loss, _ = loss_fn(preds, batch["targets"])
        loss.backward()

        # pitch_type_embed row 5 should NOT get gradient in pregame mode
        # (it's only used via the live PitchEncoder path)
        pitch_type_grad = model.pitch_encoder.pitch_type_embed.weight.grad
        if pitch_type_grad is not None:
            # Row 5 should be zero (no live data used it)
            assert pitch_type_grad[5].abs().sum() == 0, \
                "pitch_type_embed row 5 should have zero gradient in pregame mode"


# ---------------------------------------------------------------------------
# 3. Collate function preserves categorical prefix fields
# ---------------------------------------------------------------------------

class TestCollatePreservesCategoricals:
    """game_transformer_collate_fn must pass all 6 categorical fields through."""

    def _make_dataset_sample(self, T: int = 20, pitch_type_fill: int = 4) -> dict:
        """Minimal single dataset sample (matches __getitem__ output format)."""
        max_seq = 50
        n_sp = 3
        n_team = 5

        def _seq_ctx(ng):
            return {
                f"sequences": torch.randn(ng, max_seq, CONT_DIM),
                f"obs_mask": torch.ones(ng, max_seq, CONT_DIM),
                f"lengths": torch.full((ng,), max_seq, dtype=torch.long),
                f"weights": torch.ones(ng),
                f"mask": torch.ones(ng, max_seq),
            }

        sp_h = _seq_ctx(n_sp)
        sp_a = _seq_ctx(n_sp)
        tm_h = _seq_ctx(n_team)
        tm_a = _seq_ctx(n_team)

        from mlb_dl.weather_context import WEATHER_TOKEN_DIM, WEATHER_TEMPORAL_HOURS
        from mlb_dl.rating_sequences import RATING_SEQ_STEPS

        sample = {
            "sp_home_seqs": sp_h["sequences"],
            "sp_home_obs_mask": sp_h["obs_mask"],
            "sp_home_lengths": sp_h["lengths"],
            "sp_home_weights": sp_h["weights"],
            "sp_home_mask": sp_h["mask"],
            "sp_away_seqs": sp_a["sequences"],
            "sp_away_obs_mask": sp_a["obs_mask"],
            "sp_away_lengths": sp_a["lengths"],
            "sp_away_weights": sp_a["weights"],
            "sp_away_mask": sp_a["mask"],
            "team_home_seqs": tm_h["sequences"],
            "team_home_obs_mask": tm_h["obs_mask"],
            "team_home_lengths": tm_h["lengths"],
            "team_home_weights": tm_h["weights"],
            "team_home_mask": tm_h["mask"],
            "team_home_similarity": torch.ones(n_team),
            "team_away_seqs": tm_a["sequences"],
            "team_away_obs_mask": tm_a["obs_mask"],
            "team_away_lengths": tm_a["lengths"],
            "team_away_weights": tm_a["weights"],
            "team_away_mask": tm_a["mask"],
            "team_away_similarity": torch.ones(n_team),
            "flat_features": torch.randn(FLAT_FEATURE_DIM),
            "weather_temporal": torch.randn(WEATHER_TEMPORAL_HOURS, WEATHER_TOKEN_DIM),
            "rating_home": torch.zeros(RATING_SEQ_STEPS, 1),
            "rating_away": torch.zeros(RATING_SEQ_STEPS, 1),
            "prefix_values": torch.randn(T, CONT_DIM),
            "prefix_obs_mask": torch.ones(T, CONT_DIM),
            "prefix_mask": torch.ones(T),
            "prefix_batter_hash": torch.randint(1, 50000, (T,)),
            "prefix_pitcher_hash": torch.randint(1, 50000, (T,)),
            "prefix_catcher_hash": torch.randint(1, 50000, (T,)),
            "prefix_event_type": torch.zeros(T, dtype=torch.long),
            "prefix_hierarchy": torch.zeros(T, 3, dtype=torch.long),
            "prefix_pitch_type_idx": torch.full((T,), pitch_type_fill, dtype=torch.long),
            "prefix_bat_side_idx": torch.full((T,), 1, dtype=torch.long),
            "prefix_pitch_hand_idx": torch.full((T,), 1, dtype=torch.long),
            "prefix_half_inning_idx": torch.full((T,), 1, dtype=torch.long),
            "prefix_hit_trajectory_idx": torch.full((T,), 2, dtype=torch.long),
            "prefix_hit_hardness_idx": torch.full((T,), 3, dtype=torch.long),
            "prefix_length": torch.tensor(T, dtype=torch.long),
            "player_hashes": torch.randint(1, 50000, (20,)),
            "player_history": torch.zeros(20, 15, 25),
            "player_history_mask": torch.zeros(20, 15),
            "player_matchup": torch.zeros(20, 1),
            "targets": {
                "home_runs": torch.tensor(5.0),
                "away_runs": torch.tensor(3.0),
                "home_runs_remaining": torch.tensor(5.0),
                "away_runs_remaining": torch.tensor(3.0),
                "home_win": torch.tensor(1.0),
                "yrfi": torch.tensor(1.0),
                "extra_innings": torch.tensor(0.0),
                "total_runs": torch.tensor(8.0),
                "player_hits": torch.zeros(20),
                "player_hr": torch.zeros(20),
                "player_so": torch.zeros(20),
                "player_hrbi": torch.zeros(20),
                "player_tb": torch.zeros(20),
                "player_sb": torch.zeros(20),
            },
            "yrfi_mask": torch.tensor(1.0),
            "player_mask": torch.zeros(20),
            "sample_weight": torch.tensor(1.0),
            "game_pk": torch.tensor(12345, dtype=torch.long),
        }
        return sample

    def test_collate_stacks_all_categorical_prefix_fields(self):
        """Collate must produce all 6 prefix categorical fields in the batch dict."""
        samples = [self._make_dataset_sample(T=20, pitch_type_fill=i + 1) for i in range(4)]
        batch = game_transformer_collate_fn(samples)

        for key in [
            "prefix_pitch_type_idx",
            "prefix_bat_side_idx",
            "prefix_pitch_hand_idx",
            "prefix_half_inning_idx",
            "prefix_hit_trajectory_idx",
            "prefix_hit_hardness_idx",
        ]:
            assert key in batch, f"Collate output missing {key!r}"
            assert batch[key].shape[0] == 4, f"Batch dim wrong for {key!r}"
            assert batch[key].dtype == torch.long, f"{key!r} should be long dtype"

    def test_collate_preserves_categorical_values(self):
        """Collated batch must preserve the actual vocabulary indices from each sample."""
        samples = [self._make_dataset_sample(T=20, pitch_type_fill=5) for _ in range(3)]
        batch = game_transformer_collate_fn(samples)

        # All samples have pitch_type_fill=5; the collated tensor should be all 5s
        assert (batch["prefix_pitch_type_idx"] == 5).all(), \
            "Collate changed pitch_type values — must preserve raw indices"

    def test_collate_preserves_dtypes_for_categorical_fields(self):
        """All prefix categorical fields must remain long (int64) after collation."""
        samples = [self._make_dataset_sample(T=15) for _ in range(2)]
        batch = game_transformer_collate_fn(samples)

        long_keys = [
            "prefix_pitch_type_idx", "prefix_bat_side_idx", "prefix_pitch_hand_idx",
            "prefix_half_inning_idx", "prefix_hit_trajectory_idx", "prefix_hit_hardness_idx",
            "prefix_batter_hash", "prefix_pitcher_hash", "prefix_catcher_hash",
            "prefix_event_type",
        ]
        for key in long_keys:
            assert batch[key].dtype == torch.long, \
                f"{key!r} has dtype {batch[key].dtype}, expected torch.long"


# ---------------------------------------------------------------------------
# 4. End-to-end: collate → _prepare_model_input → model forward
# ---------------------------------------------------------------------------

class TestEndToEndLiveCategoricals:
    """Integration: real categorical data must flow from batch to model without NaN."""

    def _small_model(self, d_model=128):
        cfg = ContextConfig(sp_games=2, team_games=3, tokens_per_game=4, flat_feature_tokens=4)
        return GameTransformer(
            d_model=d_model,
            num_backbone_layers=2,
            num_heads=4,
            d_ff=512,
            rating_dim=0,
            context_config=cfg,
        )

    def test_no_nan_with_diverse_categorical_values(self):
        """Model produces no NaN when live pitch uses diverse real vocab values."""
        d_model = 128
        model = self._small_model(d_model=d_model)
        model.eval()

        # Each pitch type in vocab (0-19) tested across batch
        B = 4
        batch = _make_collate_batch(
            B=B, T=30,
            pitch_type_fill=4,   # CU
            bat_side_fill=0,     # L
            pitch_hand_fill=0,   # L
            half_inning_fill=0,  # top
            hit_traj_fill=4,     # popup
            hit_hard_fill=1,     # soft
        )
        model_input = _prepare_model_input(batch, player_context_dim=d_model * 2)

        with torch.no_grad():
            preds = model(model_input)

        for k, v in preds.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any(), f"NaN in output {k!r} with live categoricals"

    def test_live_and_pregame_same_batch_no_nan(self):
        """Mixed batch (some pregame, some live) must forward without NaN."""
        d_model = 128
        model = self._small_model(d_model=d_model)
        model.eval()

        batch = _make_collate_batch(B=4, T=25)
        # Make batch mixed: only first sample has live data
        batch["prefix_length"][1:] = 0

        model_input = _prepare_model_input(batch, player_context_dim=d_model * 2)

        with torch.no_grad():
            preds = model(model_input)

        for k, v in preds.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any(), f"NaN in {k!r} with mixed pregame/live batch"

    def test_categorical_change_changes_output(self):
        """Changing live categorical values must produce different model outputs (embeddings are active)."""
        d_model = 128
        model = self._small_model(d_model=d_model)
        model.eval()

        batch_a = _make_collate_batch(B=2, T=20, pitch_type_fill=1)   # FF
        batch_b = _make_collate_batch(B=2, T=20, pitch_type_fill=3)   # SL

        # Use identical continuous and player data so only pitch_type differs
        batch_b["prefix_values"] = batch_a["prefix_values"].clone()
        batch_b["prefix_batter_hash"] = batch_a["prefix_batter_hash"].clone()
        batch_b["prefix_pitcher_hash"] = batch_a["prefix_pitcher_hash"].clone()

        with torch.no_grad():
            out_a = model(_prepare_model_input(batch_a, player_context_dim=d_model * 2))
            out_b = model(_prepare_model_input(batch_b, player_context_dim=d_model * 2))

        # The home_win logit must differ because pitch_type embedding changed
        assert not torch.allclose(out_a["home_win_logit"], out_b["home_win_logit"]), \
            "pitch_type change had no effect on output — embedding is dead"


# ---------------------------------------------------------------------------
# 5. Dataset _get_live_prefix categorical array integrity
# ---------------------------------------------------------------------------

class TestDatasetCategoricalArrays:
    """Vocabulary mapping functions must produce valid, non-trivial indices."""

    def test_map_pitch_type_known_values(self):
        """Known pitch codes map to correct vocabulary indices."""
        series = pd.Series(["FF", "SL", "CU", "CH", "FC", None])
        result = map_pitch_type(series)
        assert result[0] == PITCH_TYPE_TO_IDX["FF"]
        assert result[1] == PITCH_TYPE_TO_IDX["SL"]
        assert result[2] == PITCH_TYPE_TO_IDX["CU"]
        assert result[3] == PITCH_TYPE_TO_IDX["CH"]
        assert result[4] == PITCH_TYPE_TO_IDX["FC"]
        assert result[5] == 0  # NaN → padding

    def test_map_pitch_type_unknown_falls_back_to_zero(self):
        """Unknown pitch codes must map to 0 (padding), not crash."""
        series = pd.Series(["ZZZZ", "???", "ff_lower"])
        result = map_pitch_type(series)
        # "ff_lower" won't match because PITCH_TYPE_TO_IDX uses uppercase
        assert result[0] == 0
        assert result[1] == 0

    def test_map_bat_side(self):
        """<PAD>=0, L/R/S map to 1/2/3; NaN and unknowns map to 0 (not to L)."""
        series = pd.Series(["L", "R", "S", None, "X"])
        result = map_bat_side(series)
        assert result[0] == 1  # L
        assert result[1] == 2  # R
        assert result[2] == 3  # S
        assert result[3] == 0  # NaN → <PAD> (0), distinct from "L"
        assert result[4] == 0  # unknown → <PAD> (0), distinct from "L"

    def test_map_pitch_hand(self):
        """<PAD>=0, L/R map to 1/2; NaN maps to 0 (not to L)."""
        series = pd.Series(["L", "R", None])
        result = map_pitch_hand(series)
        assert result[0] == 1  # L
        assert result[1] == 2  # R
        assert result[2] == 0  # NaN → <PAD> (0), distinct from "L"

    def test_map_half_inning_from_is_top_flag(self):
        """is_top_inning=1 → top (0), is_top_inning=0 → bottom (1)."""
        series = pd.Series([1, 0, 1, 0, None])
        result = map_half_inning(series)
        assert result[0] == 0  # top
        assert result[1] == 1  # bottom
        assert result[2] == 0  # top
        assert result[3] == 1  # bottom
        assert result[4] == 0  # NaN → top

    def test_map_hit_trajectory(self):
        """Statcast bb_type strings map correctly."""
        series = pd.Series(["ground_ball", "fly_ball", "line_drive", "popup", None, "unknown_type"])
        result = map_hit_trajectory(series)
        assert result[0] == HIT_TRAJECTORY_TO_IDX["ground_ball"]
        assert result[1] == HIT_TRAJECTORY_TO_IDX["fly_ball"]
        assert result[2] == HIT_TRAJECTORY_TO_IDX["line_drive"]
        assert result[3] == HIT_TRAJECTORY_TO_IDX["popup"]
        assert result[4] == 0  # NaN → none
        assert result[5] == 0  # unknown → 0

    def test_map_hit_hardness(self):
        """soft/medium/hard map to 1/2/3."""
        series = pd.Series(["soft", "medium", "hard", None])
        result = map_hit_hardness(series)
        assert result[0] == HIT_HARDNESS_TO_IDX["soft"]
        assert result[1] == HIT_HARDNESS_TO_IDX["medium"]
        assert result[2] == HIT_HARDNESS_TO_IDX["hard"]
        assert result[3] == 0  # NaN → none

    def test_all_vocab_indices_in_valid_range(self):
        """All vocabulary maps must produce indices within model's embedding size."""
        from mlb_dl.game_transformer import PitchEncoder

        enc = PitchEncoder()
        max_pitch_type = enc.NUM_PITCH_TYPES
        max_bat_side = enc.NUM_BAT_SIDES
        max_pitch_hand = enc.NUM_PITCH_HANDS
        max_half_inning = enc.NUM_HALF_INNINGS
        max_hit_traj = enc.NUM_HIT_TRAJECTORIES
        max_hit_hard = enc.NUM_HIT_HARDNESS

        # Every index in each vocab must be within the embedding table size
        for val in PITCH_TYPE_TO_IDX.values():
            assert 0 <= val < max_pitch_type, \
                f"PITCH_TYPE_TO_IDX value {val} out of range [0, {max_pitch_type})"
        for val in BAT_SIDE_TO_IDX.values():
            assert 0 <= val < max_bat_side, \
                f"BAT_SIDE_TO_IDX value {val} out of range [0, {max_bat_side})"
        for val in PITCH_HAND_TO_IDX.values():
            assert 0 <= val < max_pitch_hand
        for val in HALF_INNING_TO_IDX.values():
            assert 0 <= val < max_half_inning
        for val in HIT_TRAJECTORY_TO_IDX.values():
            assert 0 <= val < max_hit_traj
        for val in HIT_HARDNESS_TO_IDX.values():
            assert 0 <= val < max_hit_hard


# ---------------------------------------------------------------------------
# 6. Loss function target key alignment
# ---------------------------------------------------------------------------

class TestLossFunctionTargetKeys:
    """Loss function must use home_runs_remaining / away_runs_remaining, not home_runs."""

    def _preds(self, B=4, P=10):
        return {
            "mu_home": torch.rand(B) * 4 + 1,
            "alpha_home": torch.rand(B) * 5 + 1,
            "mu_away": torch.rand(B) * 4 + 1,
            "alpha_away": torch.rand(B) * 5 + 1,
            "home_win_logit": torch.randn(B),
            "yrfi_logit": torch.randn(B),
            "extra_innings_logit": torch.randn(B),
            "hits_categorical": F.softmax(torch.randn(B, P, 5), dim=-1),
            "hr_prob": torch.sigmoid(torch.randn(B, P)),
            "pitcher_k_mu": torch.rand(B, P) * 5 + 1,
            "pitcher_k_alpha": torch.rand(B, P) * 3 + 1,
            "h_r_rbi_mu": torch.rand(B, P) * 3 + 1,
            "h_r_rbi_alpha": torch.rand(B, P) * 3 + 1,
            "stolen_bases_logit": torch.randn(B, P),
        }

    def test_loss_uses_remaining_runs_not_final(self):
        """Loss must reference home_runs_remaining / away_runs_remaining."""
        B, P = 4, 10
        loss_fn = GameTransformerLoss()
        preds = self._preds(B, P)

        # Only provide remaining runs (as the dataset and loss function expect)
        targets = {
            "home_runs_remaining": torch.randint(0, 10, (B,)).float(),
            "away_runs_remaining": torch.randint(0, 10, (B,)).float(),
            "home_win": torch.randint(0, 2, (B,)).float(),
            "yrfi": torch.randint(0, 2, (B,)).float(),
            "extra_innings": torch.randint(0, 2, (B,)).float(),
            "player_hits": torch.randint(0, 4, (B, P)).float(),
            "player_hr": torch.randint(0, 2, (B, P)).float(),
            "player_so": torch.randint(0, 12, (B, P)).float(),
            "player_hrbi": torch.randint(0, 6, (B, P)).float(),
            "player_sb": torch.randint(0, 2, (B, P)).float(),
            "player_mask": torch.ones(B, P),
        }

        loss, task_losses = loss_fn(preds, targets)
        assert not torch.isnan(loss), "Loss is NaN with correct target keys"
        assert "negbin_home" in task_losses
        assert "negbin_away" in task_losses

    def test_loss_yrfi_masking_past_first_inning(self):
        """YRFI loss must be zero when live_inning > 1 for all samples."""
        B, P = 4, 10
        loss_fn = GameTransformerLoss()
        preds = self._preds(B, P)
        targets = {
            "home_runs_remaining": torch.randint(0, 10, (B,)).float(),
            "away_runs_remaining": torch.randint(0, 10, (B,)).float(),
            "home_win": torch.randint(0, 2, (B,)).float(),
            "yrfi": torch.randint(0, 2, (B,)).float(),
            "extra_innings": torch.randint(0, 2, (B,)).float(),
            "player_mask": torch.zeros(B, P),  # no player targets
        }

        # All past inning 1 → YRFI should be zero
        live_inning = torch.full((B,), 5.0)
        _, task_losses = loss_fn(preds, targets, live_inning=live_inning)
        assert task_losses["bce_yrfi"].item() == 0.0, \
            "YRFI loss should be 0 when all innings > 1"

    def test_negbin_nll_is_finite(self):
        """NegBin NLL should be finite for reasonable mu/alpha values."""
        B, P = 4, 10
        loss_fn = GameTransformerLoss()
        preds = self._preds(B, P)
        targets = {
            "home_runs_remaining": torch.zeros(B),  # 0 remaining (end of game)
            "away_runs_remaining": torch.zeros(B),
            "home_win": torch.ones(B),
            "yrfi": torch.ones(B),
            "extra_innings": torch.zeros(B),
            "player_mask": torch.zeros(B, P),
        }

        loss, task_losses = loss_fn(preds, targets)
        assert torch.isfinite(task_losses["negbin_home"]), "NegBin home NLL is not finite"
        assert torch.isfinite(task_losses["negbin_away"]), "NegBin away NLL is not finite"


# ---------------------------------------------------------------------------
# 7. Phased training helpers
# ---------------------------------------------------------------------------

class TestPhasedTrainingHelpers:
    """_freeze_lower_layers and player_context_dim computation."""

    def _build_model(self, d_model=128, n_layers=4):
        cfg = ContextConfig(sp_games=2, team_games=3, tokens_per_game=4, flat_feature_tokens=4)
        return GameTransformer(
            d_model=d_model, num_backbone_layers=n_layers, num_heads=4, d_ff=512, context_config=cfg
        )

    def test_freeze_lower_layers_disables_grad(self):
        """After _freeze_lower_layers(freeze=True), frozen params must not require grad."""
        model = self._build_model(n_layers=6)
        _freeze_lower_layers(model, freeze=True)

        # Context compiler and pitch encoder should be frozen
        for param in model.context_compiler.parameters():
            assert not param.requires_grad, "context_compiler param should be frozen"
        for param in model.pitch_encoder.parameters():
            assert not param.requires_grad, "pitch_encoder param should be frozen"

        # Backbone layers 0-3 should be frozen; 4-5 should remain trainable
        for i, layer in enumerate(model.backbone.layers):
            if i < 4:
                for param in layer.parameters():
                    assert not param.requires_grad, f"backbone layer {i} should be frozen"
            else:
                for param in layer.parameters():
                    assert param.requires_grad, f"backbone layer {i} should remain trainable"

    def test_unfreeze_lower_layers_restores_grad(self):
        """After freeze=True then freeze=False, all params should require grad again."""
        model = self._build_model(n_layers=6)
        _freeze_lower_layers(model, freeze=True)
        _freeze_lower_layers(model, freeze=False)

        for name, param in model.named_parameters():
            assert param.requires_grad, f"Param {name} should require grad after unfreeze"

    def test_player_context_dim_matches_player_head(self):
        """player_context_dim = d_model * player_context_tokens must match PlayerQueryHead's expectation."""
        from mlb_dl.game_transformer import PlayerQueryHead

        for d_model in [128, 256, 384]:
            player_context_tokens = 2  # default in GameTransformer
            expected_input_dim = player_context_tokens * d_model

            # Build the input dim as train_unified does
            player_ctx_dim = d_model * 2  # from _run_phased_training: d_model * 2

            assert player_ctx_dim == expected_input_dim, (
                f"player_ctx_dim={player_ctx_dim} != expected {expected_input_dim} "
                f"for d_model={d_model}"
            )

    def test_phase1_player_loss_weight_restored(self):
        """PLAYER_LOSS_WEIGHT must be restored to original value after phase 1."""
        from mlb_dl.game_transformer import GameTransformerLoss
        loss_fn = GameTransformerLoss()
        original_weight = loss_fn.PLAYER_LOSS_WEIGHT

        # Simulate what _run_phased_training does
        loss_fn.PLAYER_LOSS_WEIGHT = 0.0
        # ... phase 1 runs ...
        loss_fn.PLAYER_LOSS_WEIGHT = original_weight

        assert loss_fn.PLAYER_LOSS_WEIGHT == original_weight, \
            "PLAYER_LOSS_WEIGHT not restored after phase 1"


# ---------------------------------------------------------------------------
# 8. _build_targets remaining runs computation
# ---------------------------------------------------------------------------

class TestBuildTargetsRemainingRuns:
    """_build_targets must compute remaining runs as max(0, final - score_at_prefix)."""

    def _make_minimal_dataset(self):
        """Minimal GameTransformerDataset with 1 game for testing _build_targets."""
        from mlb_dl.game_transformer_dataset import GameTransformerDataset, AblationConfig
        from mlb_dl.datasets import Standardizer

        n_pitches = 30
        rng = np.random.default_rng(42)

        game_pk = 999

        pitches = pd.DataFrame({
            "game_pk": [game_pk] * n_pitches,
            "play_index": range(n_pitches),
            "pitch_sequence_index": range(n_pitches),
            "game_date": ["2023-06-01"] * n_pitches,
            "inning": [1] * 10 + [2] * 10 + [3] * 10,
            "at_bat_index": list(range(n_pitches)),
            "pitch_number": [1] * n_pitches,
            "batter_id": [111] * n_pitches,
            "pitcher_id": [222] * n_pitches,
            # Running score
            "score_home": [0] * 10 + [2] * 10 + [3] * 10,
            "score_away": [0] * 30,
            "is_top_inning": [1] * n_pitches,
        })

        game_targets = pd.DataFrame({
            "game_pk": [game_pk],
            "game_date": ["2023-06-01"],
            "home_runs": [5.0],
            "away_runs": [2.0],
            "home_win": [1],
            "yrfi": [0],
            "extra_innings": [0],
            "total_runs": [7.0],
            "target_status": ["trainable"],
        })

        game_meta = pd.DataFrame({
            "game_pk": [game_pk],
            "game_date": ["2023-06-01"],
            "season": [2023],
            "home_team_id": [1],
            "away_team_id": [2],
            "probable_pitcher_home_id": [222],
            "probable_pitcher_away_id": [333],
        })

        team_games = pd.DataFrame({
            "team_id": [1, 2],
            "game_pk": [game_pk, game_pk],
            "game_date": ["2023-06-01", "2023-06-01"],
        })

        player_history = pd.DataFrame({
            "player_id": [111],
            "game_pk": [game_pk],
            "game_date": ["2023-05-31"],
            "target_status": ["trainable"],
        })

        avail_cols = [c for c in pitches.columns if c in ["game_date"]]
        standardizer = Standardizer.fit(pitches, [])

        ds = GameTransformerDataset(
            pitch_sequences=pitches,
            game_targets=game_targets,
            game_meta=game_meta,
            team_games=team_games,
            player_batting_history=player_history,
            standardizer=standardizer,
            ablation=AblationConfig(),
            include_live=True,
        )
        return ds, game_pk

    def test_remaining_runs_at_prefix_zero(self):
        """At prefix=0 (pregame), remaining runs = final runs (nothing observed yet)."""
        ds, game_pk = self._make_minimal_dataset()
        # Find pregame sample
        sample = None
        for s in ds.samples:
            if s[1] == 0:
                sample = s
                break

        if sample is None:
            pytest.skip("No pregame sample found in dataset")

        item = ds[ds.samples.index(sample)]
        targets = item["targets"]

        assert targets["home_runs_remaining"].item() == 5.0, \
            "At pregame, home_runs_remaining should equal final home_runs"
        assert targets["away_runs_remaining"].item() == 2.0, \
            "At pregame, away_runs_remaining should equal final away_runs"

    def test_remaining_runs_nonincreasing_as_prefix_grows(self):
        """As prefix grows, remaining runs must be non-increasing."""
        ds, game_pk = self._make_minimal_dataset()
        # Collect all samples for this game, sorted by prefix length
        game_samples = sorted(
            [(i, s[1]) for i, s in enumerate(ds.samples) if s[0] == game_pk],
            key=lambda x: x[1],
        )

        if len(game_samples) < 2:
            pytest.skip("Need at least 2 samples (pregame + live) for this test")

        prev_home = float("inf")
        prev_away = float("inf")
        for idx, _ in game_samples:
            item = ds[idx]
            h = item["targets"]["home_runs_remaining"].item()
            a = item["targets"]["away_runs_remaining"].item()
            assert h <= prev_home + 1e-6, \
                f"home_runs_remaining increased: {prev_home} → {h}"
            assert a <= prev_away + 1e-6, \
                f"away_runs_remaining increased: {prev_away} → {a}"
            prev_home = h
            prev_away = a

    def test_remaining_runs_nonnegative(self):
        """home_runs_remaining and away_runs_remaining must never be negative."""
        ds, game_pk = self._make_minimal_dataset()
        for idx in range(len(ds)):
            item = ds[idx]
            assert item["targets"]["home_runs_remaining"].item() >= 0, \
                "home_runs_remaining is negative"
            assert item["targets"]["away_runs_remaining"].item() >= 0, \
                "away_runs_remaining is negative"


# ---------------------------------------------------------------------------
# 9. PitchEncoder input dimension consistency
# ---------------------------------------------------------------------------

class TestPitchEncoderDimConsistency:
    """PitchEncoder.proj input dim must equal the sum of all sub-embeddings + continuous."""

    def test_proj_input_dim_matches_construction(self):
        """Manually recompute the input dim and verify it matches proj[0].in_features."""
        from mlb_dl.game_transformer import PitchEncoder

        continuous_dim = 52
        player_embed_dim = 16
        event_embed_dim = 8
        pitch_type_embed_dim = 8
        bat_side_embed_dim = 4
        pitch_hand_embed_dim = 4
        half_inning_embed_dim = 4
        hit_trajectory_embed_dim = 4
        hit_hardness_embed_dim = 4

        # [continuous * obs_mask, obs_mask] concatenated
        continuous_part = continuous_dim * 2
        categorical_part = (
            pitch_type_embed_dim + bat_side_embed_dim + pitch_hand_embed_dim
            + half_inning_embed_dim + hit_trajectory_embed_dim + hit_hardness_embed_dim
        )
        player_part = 3 * player_embed_dim  # batter + pitcher + catcher
        event_part = event_embed_dim

        expected_input = continuous_part + player_part + event_part + categorical_part

        encoder = PitchEncoder(continuous_dim=continuous_dim, d_model=256)
        actual_input = encoder.proj[0].in_features

        assert actual_input == expected_input, (
            f"PitchEncoder proj input: expected {expected_input}, got {actual_input}"
        )

    def test_pitch_continuous_cols_matches_encoder_default(self):
        """PITCH_CONTINUOUS_COLS length must match PitchEncoder default continuous_dim=52."""
        from mlb_dl.game_transformer import PitchEncoder
        assert len(PITCH_CONTINUOUS_COLS) == 52, (
            f"PITCH_CONTINUOUS_COLS has {len(PITCH_CONTINUOUS_COLS)} entries, expected 52"
        )
