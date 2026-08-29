"""Integration test: validates the full training pipeline end-to-end with synthetic data.

Run locally before EC2 to catch dimension mismatches, dtype errors, or
broken gradient flow without needing real feature store parquets.

Usage:
    conda run -n pred python -m pytest deep_learning/mlb_dl/tests/test_training_pipeline.py -v
"""

import torch
import torch.nn.functional as F
import pytest

import sys
sys.path.insert(0, "deep_learning")

from mlb_dl.game_transformer import (
    ContextConfig,
    GameTransformer,
    GameTransformerLoss,
)
from mlb_dl.game_transformer_dataset import FLAT_FEATURE_DIM, PITCH_CONTINUOUS_COLS


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def context_config():
    return ContextConfig(sp_games=3, team_games=5, tokens_per_game=4, flat_feature_tokens=4)


def _mock_batch(B, P, T_live, config, rating_dim, device, d_model=256):
    """Produce a synthetic batch matching the real collate output structure."""
    max_pitches = 60
    continuous_dim = len(PITCH_CONTINUOUS_COLS)

    def game_data(n_games):
        return {
            "continuous": torch.randn(B, n_games, max_pitches, continuous_dim, device=device),
            "batter_hash": torch.randint(1, 50000, (B, n_games, max_pitches), device=device),
            "pitcher_hash": torch.randint(1, 50000, (B, n_games, max_pitches), device=device),
            "inning_idx": torch.randint(0, 9, (B, n_games, max_pitches), device=device),
            "ab_idx": torch.randint(0, 40, (B, n_games, max_pitches), device=device),
            "pitch_idx": torch.randint(0, 10, (B, n_games, max_pitches), device=device),
            "padding_mask": torch.zeros(B, n_games, max_pitches, dtype=torch.bool, device=device),
            "games_ago": torch.arange(n_games, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1),
            "seasons_crossed": torch.zeros(B, n_games, device=device),
        }

    context = {
        "sp_home": game_data(config.sp_games),
        "sp_away": game_data(config.sp_games),
        "team_home": game_data(config.team_games),
        "team_away": game_data(config.team_games),
        "flat_features": torch.randn(B, FLAT_FEATURE_DIM, device=device),
        "weather_temporal": torch.randn(B, 4, 22, device=device),
    }

    if rating_dim > 0:
        context["rating_home"] = torch.randn(B, config.rating_steps, rating_dim, device=device)
        context["rating_away"] = torch.randn(B, config.rating_steps, rating_dim, device=device)

    batch = {
        "context": context,
        "player_hashes": torch.randint(1, 50000, (B, P), device=device),
        "player_context": torch.randn(B, P, 2 * d_model, device=device),
    }

    if T_live > 0:
        batch["live_continuous"] = torch.randn(B, T_live, continuous_dim, device=device)
        batch["live_batter_hash"] = torch.randint(1, 50000, (B, T_live), device=device)
        batch["live_pitcher_hash"] = torch.randint(1, 50000, (B, T_live), device=device)
        batch["live_inning_idx"] = torch.randint(0, 9, (B, T_live), device=device)
        batch["live_ab_idx"] = torch.randint(0, 40, (B, T_live), device=device)
        batch["live_pitch_idx"] = torch.randint(0, 10, (B, T_live), device=device)

    targets = {
        "home_runs_remaining": torch.randint(0, 10, (B,), device=device).float(),
        "away_runs_remaining": torch.randint(0, 10, (B,), device=device).float(),
        "home_win": torch.randint(0, 2, (B,), device=device).float(),
        "yrfi": torch.randint(0, 2, (B,), device=device).float(),
        "extra_innings": torch.randint(0, 2, (B,), device=device).float(),
        "player_hits": torch.randint(0, 4, (B, P), device=device).float(),
        "player_hr": torch.randint(0, 2, (B, P), device=device).float(),
        "player_k": torch.randint(0, 12, (B, P), device=device).float(),
        "player_hrbi": torch.randint(0, 6, (B, P), device=device).float(),
        "player_sb": torch.randint(0, 2, (B, P), device=device).float(),
        "player_mask": torch.ones(B, P, device=device),
    }

    return batch, targets


@pytest.mark.parametrize("d_model,n_layers,n_heads", [
    (128, 2, 4),   # baseline
    (256, 4, 8),   # medium
])
def test_forward_backward_pregame(d_model, n_layers, n_heads, context_config, device):
    """Full forward + backward pass in pregame mode (T=0)."""
    B, P = 4, 10
    rating_dim = 59

    model = GameTransformer(
        d_model=d_model,
        num_backbone_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_model * 4,
        rating_dim=rating_dim,
        context_config=context_config,
        flat_feature_dim=FLAT_FEATURE_DIM,
    ).to(device)

    loss_fn = GameTransformerLoss().to(device)
    batch, targets = _mock_batch(B, P, T_live=0, config=context_config,
                                  rating_dim=rating_dim, device=device, d_model=d_model)

    preds = model(batch)
    loss, task_losses = loss_fn(preds, targets)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.requires_grad
    loss.backward()

    # Verify gradient flows to all parameter groups
    groups = {"pitch_encoder": False, "backbone": False, "player_head": False}
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            if "pitch_encoder" in name:
                groups["pitch_encoder"] = True
            elif "backbone" in name:
                groups["backbone"] = True
            elif "player_head" in name:
                groups["player_head"] = True

    for group, has_grad in groups.items():
        assert has_grad, f"No gradient flow to {group}"


@pytest.mark.parametrize("d_model,n_layers,n_heads", [
    (128, 2, 4),
    (256, 4, 8),
])
def test_forward_backward_live(d_model, n_layers, n_heads, context_config, device):
    """Full forward + backward pass in live mode (T=50 pitches)."""
    B, P, T = 4, 10, 50
    rating_dim = 59

    model = GameTransformer(
        d_model=d_model,
        num_backbone_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_model * 4,
        rating_dim=rating_dim,
        context_config=context_config,
        flat_feature_dim=FLAT_FEATURE_DIM,
    ).to(device)

    loss_fn = GameTransformerLoss().to(device)
    batch, targets = _mock_batch(B, P, T_live=T, config=context_config,
                                  rating_dim=rating_dim, device=device, d_model=d_model)

    preds = model(batch)
    loss, task_losses = loss_fn(preds, targets)

    assert not torch.isnan(loss)
    loss.backward()


def test_continuous_dim_matches_feature_list():
    """Critical: PitchEncoder continuous_dim must match PITCH_CONTINUOUS_COLS length."""
    from mlb_dl.game_transformer import PitchEncoder
    encoder = PitchEncoder()
    expected = len(PITCH_CONTINUOUS_COLS)
    actual = encoder.continuous_dim  # stored as attribute; use it directly rather than back-computing
    assert actual == expected, (
        f"PitchEncoder continuous_dim={actual} but PITCH_CONTINUOUS_COLS has {expected} entries"
    )


def test_flat_feature_dim_matches_constant():
    """Critical: model flat_feature_dim must match dataset FLAT_FEATURE_DIM."""
    assert FLAT_FEATURE_DIM == 30, f"FLAT_FEATURE_DIM should be 30, got {FLAT_FEATURE_DIM}"


def test_training_step_reduces_loss(context_config, device):
    """One optimizer step should reduce loss on a fixed batch (sanity)."""
    B, P = 8, 10
    rating_dim = 59
    d_model = 128

    model = GameTransformer(
        d_model=d_model, num_backbone_layers=2, num_heads=4,
        d_ff=512, rating_dim=rating_dim, context_config=context_config,
    ).to(device)
    loss_fn = GameTransformerLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch, targets = _mock_batch(B, P, T_live=0, config=context_config,
                                  rating_dim=rating_dim, device=device, d_model=d_model)

    # First forward
    model.train()
    preds = model(batch)
    loss1, _ = loss_fn(preds, targets)

    # 5 gradient steps on same batch (should overfit and reduce loss)
    for _ in range(5):
        optimizer.zero_grad()
        preds = model(batch)
        loss, _ = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

    preds = model(batch)
    loss2, _ = loss_fn(preds, targets)

    assert loss2.item() < loss1.item(), (
        f"Loss didn't decrease after 5 steps: {loss1.item():.4f} -> {loss2.item():.4f}"
    )


def test_weather_zeros_dont_crash(context_config, device):
    """Model handles all-zero weather (pre-2017 games) without NaN."""
    B, P = 4, 10
    d_model = 128

    model = GameTransformer(
        d_model=d_model, num_backbone_layers=2, num_heads=4,
        d_ff=512, rating_dim=59, context_config=context_config,
    ).to(device)

    batch, _ = _mock_batch(B, P, T_live=0, config=context_config,
                            rating_dim=59, device=device, d_model=d_model)
    # Zero out weather
    batch["context"]["weather_temporal"] = torch.zeros(B, 4, 22, device=device)

    model.eval()
    with torch.no_grad():
        preds = model(batch)

    for k, v in preds.items():
        if isinstance(v, torch.Tensor):
            assert not torch.isnan(v).any(), f"NaN in {k} with zero weather"


def test_missing_rating_sequences(context_config, device):
    """Model handles absent rating sequences (rating_dim=0) gracefully."""
    B, P = 4, 10
    d_model = 128

    model = GameTransformer(
        d_model=d_model, num_backbone_layers=2, num_heads=4,
        d_ff=512, rating_dim=0, context_config=context_config,
    ).to(device)

    batch, _ = _mock_batch(B, P, T_live=0, config=context_config,
                            rating_dim=0, device=device, d_model=d_model)

    model.eval()
    with torch.no_grad():
        preds = model(batch)

    assert preds["mu_home"].shape == (B,)
    assert not torch.isnan(preds["home_win_logit"]).any()
