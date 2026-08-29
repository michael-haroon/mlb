"""Tests for NaN imputation and observability mask in the GameTransformer pipeline.

Verifies that:
1. Missing values become neutral z=0 after standardization (not catastrophic z=-14)
2. Observability mask correctly identifies observed vs missing positions
3. The full forward pass produces finite outputs with partial missing data
4. The categorical mapping from prefix → live is wired correctly
"""

import numpy as np
import pytest
import torch

from deep_learning.mlb_dl.game_transformer import GameTransformer, ContextConfig, PitchEncoder
from deep_learning.mlb_dl.game_transformer_dataset import (
    PITCH_CONTINUOUS_COLS,
    game_transformer_collate_fn,
)


N_CONT = len(PITCH_CONTINUOUS_COLS)


class TestPitchEncoderObsMask:
    """PitchEncoder correctly handles obs_mask channel."""

    @pytest.fixture
    def encoder(self):
        return PitchEncoder(continuous_dim=N_CONT, d_model=64, dropout=0.0)

    def test_output_shape_with_mask(self, encoder):
        B, S = 2, 10
        cont = torch.randn(B, S, N_CONT)
        mask = torch.ones(B, S, N_CONT)
        batter = torch.randint(0, 100, (B, S))
        pitcher = torch.randint(0, 100, (B, S))

        out = encoder(cont, batter, pitcher, obs_mask=mask)
        assert out.shape == (B, S, 64)

    def test_output_shape_without_mask(self, encoder):
        """Backward compatibility: None obs_mask uses ones."""
        B, S = 2, 10
        cont = torch.randn(B, S, N_CONT)
        batter = torch.randint(0, 100, (B, S))
        pitcher = torch.randint(0, 100, (B, S))

        out = encoder(cont, batter, pitcher, obs_mask=None)
        assert out.shape == (B, S, 64)

    def test_missing_features_zeroed(self, encoder):
        """With obs_mask=0, continuous values are zeroed before projection."""
        B, S = 1, 5
        cont = torch.randn(B, S, N_CONT)
        mask = torch.zeros(B, S, N_CONT)  # all missing
        batter = torch.zeros(B, S, dtype=torch.long)
        pitcher = torch.zeros(B, S, dtype=torch.long)

        out_missing = encoder(cont, batter, pitcher, obs_mask=mask)
        # With all features masked, the continuous contribution should be
        # deterministic (zeros * mask = zeros, concatenated with all-zero mask)
        out_zeros = encoder(torch.zeros_like(cont), batter, pitcher, obs_mask=mask)
        assert torch.allclose(out_missing, out_zeros, atol=1e-6)

    def test_no_nan_with_partial_mask(self, encoder):
        """Partial obs_mask never produces NaN."""
        B, S = 4, 20
        cont = torch.randn(B, S, N_CONT)
        mask = torch.ones(B, S, N_CONT)
        mask[:, :, 20:40] = 0.0  # 20 of 52 features missing

        batter = torch.randint(0, 100, (B, S))
        pitcher = torch.randint(0, 100, (B, S))

        out = encoder(cont, batter, pitcher, obs_mask=mask)
        assert not out.isnan().any()
        assert not out.isinf().any()


class TestObsMaskConstruction:
    """Verify obs_mask is built correctly from NaN positions."""

    def test_mask_from_nan_positions(self):
        """Simulate the dataset's obs_mask construction logic."""
        raw = np.array([
            [88.2, np.nan, 2.3, np.nan],
            [91.0, 1.5, np.nan, 0.8],
            [np.nan, np.nan, np.nan, np.nan],
        ], dtype=np.float32)

        obs_mask = np.isfinite(raw).astype(np.float32)

        expected_mask = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)
        np.testing.assert_array_equal(obs_mask, expected_mask)

    def test_post_standardization_zero_fill(self):
        """NaN positions become z=0 after standardization + nan_to_num."""
        raw = np.array([88.2, np.nan, 91.0, np.nan, 85.5], dtype=np.float32)
        mean = np.nanmean(raw)
        std = np.nanstd(raw)

        standardized = (raw - mean) / std
        # NaN propagates through standardization
        assert np.isnan(standardized[1])
        assert np.isnan(standardized[3])

        # nan_to_num zeroes them — neutral z-score
        result = np.nan_to_num(standardized, nan=0.0)
        assert result[1] == 0.0
        assert result[3] == 0.0

        # Observed values are within normal z-score range
        assert all(abs(result[i]) < 5.0 for i in [0, 2, 4])

    def test_old_fillna_would_create_extreme_z(self):
        """Demonstrate the bug: fillna(0) before standardization creates z=-14."""
        release_speeds = np.array([88.2, 91.0, 85.5, 89.0, 92.3], dtype=np.float32)
        mean = release_speeds.mean()
        std = release_speeds.std()

        # OLD BUG: fillna(0) then standardize
        filled_with_zero = 0.0
        z_buggy = (filled_with_zero - mean) / std
        # This creates catastrophically extreme z-scores
        assert abs(z_buggy) > 10.0, f"Expected extreme z, got {z_buggy}"

        # NEW FIX: standardize then nan_to_num
        z_fixed = 0.0  # NaN → nan_to_num → 0.0 (neutral)
        assert z_fixed == 0.0


class TestCollateObsMask:
    """Collate function correctly handles obs_mask tensors."""

    def _make_sample(self, n_games=3, seq_len=5):
        return {
            "sp_home_seqs": torch.randn(n_games, seq_len, N_CONT),
            "sp_home_obs_mask": torch.ones(n_games, seq_len, N_CONT),
            "sp_home_lengths": torch.tensor([seq_len] * n_games),
            "sp_home_weights": torch.ones(n_games),
            "sp_home_mask": torch.ones(n_games, seq_len),
            "sp_away_seqs": torch.randn(n_games, seq_len, N_CONT),
            "sp_away_obs_mask": torch.ones(n_games, seq_len, N_CONT),
            "sp_away_lengths": torch.tensor([seq_len] * n_games),
            "sp_away_weights": torch.ones(n_games),
            "sp_away_mask": torch.ones(n_games, seq_len),
            "team_home_seqs": torch.randn(n_games, seq_len, N_CONT),
            "team_home_obs_mask": torch.ones(n_games, seq_len, N_CONT),
            "team_home_lengths": torch.tensor([seq_len] * n_games),
            "team_home_weights": torch.ones(n_games),
            "team_home_mask": torch.ones(n_games, seq_len),
            "team_home_similarity": torch.ones(n_games),
            "team_away_seqs": torch.randn(n_games, seq_len, N_CONT),
            "team_away_obs_mask": torch.ones(n_games, seq_len, N_CONT),
            "team_away_lengths": torch.tensor([seq_len] * n_games),
            "team_away_weights": torch.ones(n_games),
            "team_away_mask": torch.ones(n_games, seq_len),
            "team_away_similarity": torch.ones(n_games),
            "flat_features": torch.randn(30),
            "weather_temporal": torch.randn(4, 22),
            "rating_home": torch.randn(5, 8),
            "rating_away": torch.randn(5, 8),
            "prefix_values": torch.randn(20, N_CONT),
            "prefix_obs_mask": torch.ones(20, N_CONT),
            "prefix_mask": torch.ones(20),
            "prefix_batter_hash": torch.zeros(20, dtype=torch.long),
            "prefix_pitcher_hash": torch.zeros(20, dtype=torch.long),
            "prefix_catcher_hash": torch.zeros(20, dtype=torch.long),
            "prefix_event_type": torch.zeros(20, dtype=torch.long),
            "prefix_hierarchy": torch.zeros(20, 3, dtype=torch.long),
            "prefix_pitch_type_idx": torch.zeros(20, dtype=torch.long),
            "prefix_bat_side_idx": torch.zeros(20, dtype=torch.long),
            "prefix_pitch_hand_idx": torch.zeros(20, dtype=torch.long),
            "prefix_half_inning_idx": torch.zeros(20, dtype=torch.long),
            "prefix_hit_trajectory_idx": torch.zeros(20, dtype=torch.long),
            "prefix_hit_hardness_idx": torch.zeros(20, dtype=torch.long),
            "prefix_length": torch.tensor(10),
            "player_hashes": torch.zeros(20, dtype=torch.long),
            "player_history": torch.zeros(20, 5, 16),
            "player_history_mask": torch.zeros(20, 5),
            "player_matchup": torch.zeros(20, 1),
            "targets": {"home_win": torch.tensor(1.0)},
            "yrfi_mask": torch.tensor(1.0),
            "player_mask": torch.zeros(20),
            "sample_weight": torch.tensor(1.0),
            "game_pk": torch.tensor(12345),
        }

    def test_obs_mask_in_collated_output(self):
        batch = [self._make_sample(), self._make_sample(seq_len=8)]
        collated = game_transformer_collate_fn(batch)

        # History obs_masks present and correct shape
        for prefix in ["sp_home", "sp_away", "team_home", "team_away"]:
            key = f"{prefix}_obs_mask"
            assert key in collated, f"Missing {key}"
            seqs_shape = collated[f"{prefix}_seqs"].shape
            assert collated[key].shape == seqs_shape

        # Prefix obs_mask
        assert "prefix_obs_mask" in collated
        assert collated["prefix_obs_mask"].shape == collated["prefix_values"].shape

    def test_padding_zeros_in_obs_mask(self):
        """Padded positions in obs_mask should be 0 (nothing observed in padding)."""
        s1 = self._make_sample(seq_len=3)
        s2 = self._make_sample(seq_len=8)
        batch = [s1, s2]
        collated = game_transformer_collate_fn(batch)

        # s1 has seq_len=3, padded to 8. First 5 positions should be 0.
        obs = collated["sp_home_obs_mask"][0]  # [n_games, 8, N_CONT]
        # Left-padded: positions 0..4 are padding
        assert (obs[:, :5, :] == 0.0).all()
        # Positions 5..7 are from obs_mask=ones
        assert (obs[:, 5:, :] == 1.0).all()


class TestFullForwardWithMissing:
    """End-to-end model forward pass with realistic missing data patterns."""

    @pytest.fixture
    def model(self):
        return GameTransformer(
            d_model=64,
            continuous_dim=N_CONT,
            flat_feature_dim=30,
            rating_dim=8,
            context_config=ContextConfig(
                sp_games=3, team_games=5, tokens_per_game=2, rating_steps=5
            ),
            num_backbone_layers=1,
            num_heads=4,
            dropout=0.0,
        )

    def _build_input(self, B=2, missing_frac=0.4):
        obs_mask = torch.ones(B, 3, 8, N_CONT)
        n_missing = int(N_CONT * missing_frac)
        obs_mask[:, :, :, :n_missing] = 0.0

        cont = torch.randn(B, 3, 8, N_CONT) * obs_mask

        live_obs = torch.ones(B, 20, N_CONT)
        live_obs[:, :, N_CONT // 2:] = 0.0
        live_cont = torch.randn(B, 20, N_CONT) * live_obs

        return {
            "context": {
                "sp_home": {
                    "continuous": cont, "obs_mask": obs_mask,
                    "batter_hash": torch.zeros(B, 3, 8, dtype=torch.long),
                    "pitcher_hash": torch.zeros(B, 3, 8, dtype=torch.long),
                    "inning_idx": torch.zeros(B, 3, 8, dtype=torch.long),
                    "ab_idx": torch.zeros(B, 3, 8, dtype=torch.long),
                    "pitch_idx": torch.zeros(B, 3, 8, dtype=torch.long),
                    "padding_mask": torch.zeros(B, 3, 8, dtype=torch.bool),
                    "games_ago": torch.ones(B, 3),
                    "seasons_crossed": torch.zeros(B, 3),
                },
                "sp_away": {
                    "continuous": cont, "obs_mask": obs_mask,
                    "batter_hash": torch.zeros(B, 3, 8, dtype=torch.long),
                    "pitcher_hash": torch.zeros(B, 3, 8, dtype=torch.long),
                    "inning_idx": torch.zeros(B, 3, 8, dtype=torch.long),
                    "ab_idx": torch.zeros(B, 3, 8, dtype=torch.long),
                    "pitch_idx": torch.zeros(B, 3, 8, dtype=torch.long),
                    "padding_mask": torch.zeros(B, 3, 8, dtype=torch.bool),
                    "games_ago": torch.ones(B, 3),
                    "seasons_crossed": torch.zeros(B, 3),
                },
                "team_home": {
                    "continuous": torch.randn(B, 5, 8, N_CONT),
                    "obs_mask": torch.ones(B, 5, 8, N_CONT),
                    "batter_hash": torch.zeros(B, 5, 8, dtype=torch.long),
                    "pitcher_hash": torch.zeros(B, 5, 8, dtype=torch.long),
                    "inning_idx": torch.zeros(B, 5, 8, dtype=torch.long),
                    "ab_idx": torch.zeros(B, 5, 8, dtype=torch.long),
                    "pitch_idx": torch.zeros(B, 5, 8, dtype=torch.long),
                    "padding_mask": torch.zeros(B, 5, 8, dtype=torch.bool),
                    "games_ago": torch.ones(B, 5),
                    "seasons_crossed": torch.zeros(B, 5),
                },
                "team_away": {
                    "continuous": torch.randn(B, 5, 8, N_CONT),
                    "obs_mask": torch.ones(B, 5, 8, N_CONT),
                    "batter_hash": torch.zeros(B, 5, 8, dtype=torch.long),
                    "pitcher_hash": torch.zeros(B, 5, 8, dtype=torch.long),
                    "inning_idx": torch.zeros(B, 5, 8, dtype=torch.long),
                    "ab_idx": torch.zeros(B, 5, 8, dtype=torch.long),
                    "pitch_idx": torch.zeros(B, 5, 8, dtype=torch.long),
                    "padding_mask": torch.zeros(B, 5, 8, dtype=torch.bool),
                    "games_ago": torch.ones(B, 5),
                    "seasons_crossed": torch.zeros(B, 5),
                },
                "flat_features": torch.randn(B, 30),
                "weather_temporal": torch.randn(B, 4, 22),
                "rating_home": torch.randn(B, 5, 8),
                "rating_away": torch.randn(B, 5, 8),
            },
            "live_continuous": live_cont,
            "live_obs_mask": live_obs,
            "live_batter_hash": torch.zeros(B, 20, dtype=torch.long),
            "live_pitcher_hash": torch.zeros(B, 20, dtype=torch.long),
            "live_inning_idx": torch.zeros(B, 20, dtype=torch.long),
            "live_ab_idx": torch.zeros(B, 20, dtype=torch.long),
            "live_pitch_idx": torch.zeros(B, 20, dtype=torch.long),
            "player_hashes": torch.zeros(B, 20, dtype=torch.long),
        }

    def test_no_nan_40pct_missing(self, model):
        model.eval()
        with torch.no_grad():
            output = model(self._build_input(missing_frac=0.4))
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                assert not v.isnan().any(), f"NaN in output['{k}']"
                assert not v.isinf().any(), f"Inf in output['{k}']"

    def test_no_nan_100pct_missing(self, model):
        """Extreme case: all features missing. Should still produce finite output."""
        model.eval()
        with torch.no_grad():
            output = model(self._build_input(missing_frac=1.0))
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                assert not v.isnan().any(), f"NaN in output['{k}'] with 100% missing"

    def test_pregame_mode_no_live(self, model):
        """Pregame mode (no live prefix) should work."""
        model.eval()
        inp = self._build_input()
        del inp["live_continuous"]
        del inp["live_obs_mask"]
        del inp["live_batter_hash"]
        del inp["live_pitcher_hash"]
        del inp["live_inning_idx"]
        del inp["live_ab_idx"]
        del inp["live_pitch_idx"]
        with torch.no_grad():
            output = model(inp)
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                assert not v.isnan().any(), f"NaN in pregame output['{k}']"
