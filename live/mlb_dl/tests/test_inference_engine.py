"""
Unit tests for the LiveInferenceEngine.

These tests validate:
1. Game registration and state management
2. Pitch event encoding and tensor construction
3. Inference latency (must be <100ms on CPU)
4. Thread safety of the TradingBridge
5. Hierarchy tracking (inning/AB/pitch indices)

Run with:
    conda run -n pred pytest live/mlb_dl/tests/test_inference_engine.py -v
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

from live.mlb_dl.inference_engine import (
    LiveInferenceEngine,
    PregamePrior,
    PitchEvent,
    GameInferenceState,
    TradingBridge,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a minimal model checkpoint for testing."""
    checkpoint_path = tmp_path / "test_model.pt"

    # Create a minimal LiveGameModel checkpoint
    config = {
        "feature_dim": 40,
        "hidden_dim": 128,
        "batter_buckets": 50000,
        "pitcher_buckets": 50000,
        "pitch_type_buckets": 256,
        "embed_dim": 16,
    }

    # Create a minimal model state dict (just the keys, no weights needed for structure tests)
    from live.mlb_dl.models import LiveGameModel
    model = LiveGameModel(
        feature_dim=40,
        hidden_dim=128,
        dropout=0.0,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "feature_mean": {
            "release_speed": 92.0,
            "spin_rate": 2300.0,
            "coord_px": 0.0,
            "coord_pz": 2.5,
        },
        "feature_std": {
            "release_speed": 4.5,
            "spin_rate": 400.0,
            "coord_px": 1.0,
            "coord_pz": 0.8,
        },
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def engine(mock_model_path):
    """Create a LiveInferenceEngine instance."""
    return LiveInferenceEngine(
        model_path=str(mock_model_path),
        device="cpu",
    )


@pytest.fixture
def pregame_prior():
    """Create a default PregamePrior for testing."""
    return PregamePrior(
        game_pk=12345,
        home_win_prob=0.55,
        yrfi_prob=0.48,
        mu_total_runs=9.0,
        mu_home_run_diff=0.5,
        scale_total_runs=3.0,
        scale_home_run_diff=2.8,
    )


@pytest.fixture
def sample_pitch():
    """Create a sample PitchEvent."""
    return PitchEvent(
        game_pk=12345,
        inning=1,
        is_top_inning=True,
        at_bat_index=1,
        pitch_number=1,
        batter_id=547180,  # Mike Trout
        pitcher_id=543037,  # Gerrit Cole
        pitch_type="FF",
        balls=0,
        strikes=0,
        outs=0,
        on_first=False,
        on_second=False,
        on_third=False,
        pitch_call="StrikeCalled",
        is_scoring_play=False,
        rbi_count=0,
        score_home=0,
        score_away=0,
        release_speed=98.5,
        spin_rate=2450,
        coord_px=0.1,
        coord_pz=2.5,
    )


# ── Tests ────────────────────────────────────────────────────────────────────


def test_pregame_prior_to_tensor(pregame_prior):
    """Test that PregamePrior correctly serializes to a tensor."""
    tensor = pregame_prior.to_tensor()

    assert tensor.shape == (19,)
    assert tensor.dtype == torch.float32
    assert 0.0 <= tensor[0].item() <= 1.0  # home_win_prob
    assert 0.0 <= tensor[1].item() <= 1.0  # yrfi_prob


def test_register_game(engine, pregame_prior):
    """Test game registration."""
    engine.register_game(12345, pregame_prior)

    assert 12345 in engine._games
    state = engine._games[12345]
    assert state.game_pk == 12345
    assert state.pregame_prior.home_win_prob == 0.55
    assert state.pitch_count == 0
    assert len(state.pitch_tensors) == 0


def test_unregister_game(engine, pregame_prior):
    """Test game cleanup."""
    engine.register_game(12345, pregame_prior)
    assert 12345 in engine._games

    engine.unregister_game(12345)
    assert 12345 not in engine._games


def test_encode_pitch(engine, pregame_prior, sample_pitch):
    """Test pitch event encoding."""
    engine.register_game(12345, pregame_prior)
    state = engine._games[12345]

    pitch_tensor, batter_hash, pitcher_hash, ptype_hash = engine._encode_pitch(
        sample_pitch, state
    )

    # Check tensor shape and dtype
    assert pitch_tensor.shape == (40,)
    assert pitch_tensor.dtype == torch.float32

    # Check that hashes are non-zero
    assert batter_hash > 0
    assert pitcher_hash > 0
    assert ptype_hash > 0

    # Check that standardization happened (values should be normalized)
    assert torch.isfinite(pitch_tensor).all()


def test_on_pitch_event(engine, pregame_prior, sample_pitch):
    """Test processing a pitch event."""
    engine.register_game(12345, pregame_prior)

    prices = engine.on_pitch_event(12345, sample_pitch)

    # Check that prices were returned
    assert prices is not None
    assert "home_win" in prices
    assert "yrfi" in prices
    assert "total_runs_mu" in prices

    # Check that probabilities are valid
    assert 0.0 <= prices["home_win"] <= 1.0
    assert 0.0 <= prices["yrfi"] <= 1.0
    assert prices["home_win"] + prices["away_win"] == pytest.approx(1.0, abs=1e-5)

    # Check that state was updated
    state = engine._games[12345]
    assert state.pitch_count == 1
    assert len(state.pitch_tensors) == 1


def test_multiple_pitches(engine, pregame_prior):
    """Test processing a sequence of pitches."""
    engine.register_game(12345, pregame_prior)

    # Process 10 pitches
    for i in range(10):
        pitch = PitchEvent(
            game_pk=12345,
            inning=1,
            is_top_inning=True,
            at_bat_index=1 + (i // 5),  # 2 ABs
            pitch_number=(i % 5) + 1,
            batter_id=547180,
            pitcher_id=543037,
            pitch_type="FF",
            balls=i % 4,
            strikes=i % 3,
            outs=0,
            on_first=False,
            on_second=False,
            on_third=False,
            pitch_call="StrikeCalled",
            is_scoring_play=False,
            rbi_count=0,
            score_home=0,
            score_away=0,
        )
        engine.on_pitch_event(12345, pitch)

    state = engine._games[12345]
    assert state.pitch_count == 10
    assert len(state.pitch_tensors) == 10


def test_hierarchy_tracking(engine, pregame_prior):
    """Test that inning/AB/pitch tracking works correctly."""
    engine.register_game(12345, pregame_prior)
    state = engine._games[12345]

    # First pitch of first AB
    pitch1 = PitchEvent(
        game_pk=12345,
        inning=1,
        is_top_inning=True,
        at_bat_index=1,
        pitch_number=1,
        batter_id=547180,
        pitcher_id=543037,
        pitch_type="FF",
        balls=0, strikes=0, outs=0,
        on_first=False, on_second=False, on_third=False,
        pitch_call="Ball",
        is_scoring_play=False,
        rbi_count=0,
        score_home=0,
        score_away=0,
    )
    engine.on_pitch_event(12345, pitch1)

    assert state.current_inning == 1
    assert state.ab_index_global == 1
    assert state.current_pitch_in_ab == 1

    # Second pitch of same AB
    pitch2 = PitchEvent(
        game_pk=12345,
        inning=1,
        is_top_inning=True,
        at_bat_index=1,
        pitch_number=2,
        batter_id=547180,
        pitcher_id=543037,
        pitch_type="FF",
        balls=1, strikes=0, outs=0,
        on_first=False, on_second=False, on_third=False,
        pitch_call="Strike",
        is_scoring_play=False,
        rbi_count=0,
        score_home=0,
        score_away=0,
    )
    engine.on_pitch_event(12345, pitch2)

    assert state.ab_index_global == 1
    assert state.current_pitch_in_ab == 2

    # First pitch of new AB (pitch_number resets to 1)
    pitch3 = PitchEvent(
        game_pk=12345,
        inning=1,
        is_top_inning=True,
        at_bat_index=2,
        pitch_number=1,  # Reset
        batter_id=518692,  # New batter
        pitcher_id=543037,
        pitch_type="FF",
        balls=0, strikes=0, outs=1,
        on_first=False, on_second=False, on_third=False,
        pitch_call="Ball",
        is_scoring_play=False,
        rbi_count=0,
        score_home=0,
        score_away=0,
    )
    engine.on_pitch_event(12345, pitch3)

    assert state.ab_index_global == 2
    assert state.current_pitch_in_ab == 1


def test_get_latest_prices(engine, pregame_prior, sample_pitch):
    """Test retrieving cached prices."""
    engine.register_game(12345, pregame_prior)

    # Before any pitches
    prices = engine.get_latest_prices(12345)
    assert prices is None

    # After one pitch
    engine.on_pitch_event(12345, sample_pitch)
    prices = engine.get_latest_prices(12345)
    assert prices is not None
    assert "home_win" in prices


def test_inference_latency(engine, pregame_prior):
    """Test that inference completes within latency budget (<100ms)."""
    engine.register_game(12345, pregame_prior)

    # Process 20 pitches and measure average latency
    latencies = []
    for i in range(20):
        pitch = PitchEvent(
            game_pk=12345,
            inning=1,
            is_top_inning=True,
            at_bat_index=i + 1,
            pitch_number=1,
            batter_id=547180 + i,
            pitcher_id=543037,
            pitch_type="FF",
            balls=0, strikes=0, outs=0,
            on_first=False, on_second=False, on_third=False,
            pitch_call="Strike",
            is_scoring_play=False,
            rbi_count=0,
            score_home=0,
            score_away=0,
        )

        start = time.time()
        engine.on_pitch_event(12345, pitch)
        elapsed = time.time() - start

        latencies.append(elapsed * 1000)  # Convert to ms

    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)

    print(f"\nInference latency stats:")
    print(f"  Average: {avg_latency:.1f}ms")
    print(f"  Max:     {max_latency:.1f}ms")

    # Latency budget: <100ms on CPU
    assert avg_latency < 100.0, f"Average latency {avg_latency:.1f}ms exceeds 100ms budget"
    assert max_latency < 200.0, f"Max latency {max_latency:.1f}ms exceeds 200ms ceiling"


def test_standardization(engine, pregame_prior):
    """Test that feature standardization handles None values correctly."""
    engine.register_game(12345, pregame_prior)
    state = engine._games[12345]

    # Create pitch with missing Statcast data
    pitch = PitchEvent(
        game_pk=12345,
        inning=1,
        is_top_inning=True,
        at_bat_index=1,
        pitch_number=1,
        batter_id=547180,
        pitcher_id=543037,
        pitch_type="FF",
        balls=0, strikes=0, outs=0,
        on_first=False, on_second=False, on_third=False,
        pitch_call="Ball",
        is_scoring_play=False,
        rbi_count=0,
        score_home=0,
        score_away=0,
        release_speed=None,  # Missing
        spin_rate=None,      # Missing
        coord_px=None,       # Missing
        coord_pz=None,       # Missing
    )

    tensor, _, _, _ = engine._encode_pitch(pitch, state)

    # Should not contain NaN despite missing values
    assert torch.isfinite(tensor).all()


def test_hash_consistency(engine):
    """Test that player hashing is consistent."""
    # Same player ID should always hash to same bucket
    hash1 = engine._hash_player_id(547180)
    hash2 = engine._hash_player_id(547180)
    assert hash1 == hash2

    # Different players should (usually) hash to different buckets
    hash3 = engine._hash_player_id(543037)
    assert hash1 != hash3  # Collision unlikely with 512 buckets

    # Zero should map to zero (padding)
    assert engine._hash_player_id(0) == 0


def test_build_batch_padding(engine, pregame_prior):
    """Test that batch construction handles padding correctly."""
    engine.register_game(12345, pregame_prior)

    # Add a few pitches (less than max_seq_len)
    for i in range(5):
        pitch = PitchEvent(
            game_pk=12345,
            inning=1,
            is_top_inning=True,
            at_bat_index=1,
            pitch_number=i + 1,
            batter_id=547180,
            pitcher_id=543037,
            pitch_type="FF",
            balls=0, strikes=0, outs=0,
            on_first=False, on_second=False, on_third=False,
            pitch_call="Strike",
            is_scoring_play=False,
            rbi_count=0,
            score_home=0,
            score_away=0,
        )
        engine.on_pitch_event(12345, pitch)

    state = engine._games[12345]
    batch = engine._build_batch(state)

    # Check batch structure
    assert "values" in batch
    assert "padding" in batch
    assert "batter_hashes" in batch

    # Check shapes (batch_size=1, max_seq_len=350)
    assert batch["values"].shape == (1, 350, 40)
    assert batch["padding"].shape == (1, 350)

    # Check that padding mask is correct (first 5 are real, rest are padding)
    padding = batch["padding"].cpu()
    assert padding[0, :5].sum() == 5  # First 5 are real
    assert padding[0, 5:].sum() == 0  # Rest are padding


def test_on_reprice_callback(mock_model_path, pregame_prior, sample_pitch):
    """Test that the on_reprice callback is fired."""
    callback_fired = []

    def callback(game_pk, prices):
        callback_fired.append((game_pk, prices))

    engine = LiveInferenceEngine(
        model_path=str(mock_model_path),
        device="cpu",
        on_reprice=callback,
    )

    engine.register_game(12345, pregame_prior)
    engine.on_pitch_event(12345, sample_pitch)

    assert len(callback_fired) == 1
    assert callback_fired[0][0] == 12345
    assert "home_win" in callback_fired[0][1]


# ── TradingBridge Tests ──────────────────────────────────────────────────────


def test_trading_bridge_start_stop(mock_model_path):
    """Test TradingBridge lifecycle."""
    bridge = TradingBridge(model_path=str(mock_model_path), device="cpu")

    bridge.start()
    time.sleep(0.5)  # Wait for initialization

    assert bridge.engine is not None
    assert bridge._running

    bridge.stop()
    assert not bridge._running


def test_trading_bridge_thread_safety(mock_model_path, pregame_prior):
    """Test that TradingBridge is thread-safe."""
    bridge = TradingBridge(model_path=str(mock_model_path), device="cpu")
    bridge.start()
    time.sleep(0.5)

    # Register game from main thread
    bridge.register_game(12345, pregame_prior)
    time.sleep(0.2)  # Let registration propagate

    # Query from main thread (should be thread-safe)
    active = bridge.get_active_games()
    assert 12345 in active

    prices = bridge.get_live_prices(12345)
    # prices will be None until a pitch is processed

    bridge.stop()


def test_trading_bridge_multiple_games(mock_model_path):
    """Test tracking multiple games simultaneously."""
    bridge = TradingBridge(model_path=str(mock_model_path), device="cpu")
    bridge.start()
    time.sleep(0.5)

    # Register 3 games
    for game_pk in [12345, 12346, 12347]:
        prior = PregamePrior(game_pk=game_pk)
        bridge.register_game(game_pk, prior)

    time.sleep(0.2)

    active = bridge.get_active_games()
    assert len(active) == 3
    assert 12345 in active
    assert 12346 in active
    assert 12347 in active

    bridge.stop()


# ── Edge Cases ───────────────────────────────────────────────────────────────


def test_unregistered_game_pitch(engine, sample_pitch):
    """Test handling pitch for unregistered game."""
    # Don't register game first
    prices = engine.on_pitch_event(12345, sample_pitch)

    # Should return None and log warning
    assert prices is None


def test_empty_sequence_inference(engine, pregame_prior):
    """Test inference with zero pitches (edge case)."""
    engine.register_game(12345, pregame_prior)
    state = engine._games[12345]

    # Try to run inference with no pitches
    # This shouldn't crash, but return default prices
    prices = engine._run_inference(state)

    assert prices is not None
    assert "home_win" in prices


def test_max_sequence_truncation(engine, pregame_prior):
    """Test that sequences longer than max_seq_len are truncated."""
    engine.register_game(12345, pregame_prior)

    # Add 400 pitches (more than max_seq_len=350)
    for i in range(400):
        pitch = PitchEvent(
            game_pk=12345,
            inning=(i // 20) + 1,
            is_top_inning=(i % 2 == 0),
            at_bat_index=i + 1,
            pitch_number=1,
            batter_id=547180 + (i % 100),
            pitcher_id=543037,
            pitch_type="FF",
            balls=0, strikes=0, outs=0,
            on_first=False, on_second=False, on_third=False,
            pitch_call="Strike",
            is_scoring_play=False,
            rbi_count=0,
            score_home=0,
            score_away=0,
        )
        engine.on_pitch_event(12345, pitch)

    state = engine._games[12345]
    batch = engine._build_batch(state)

    # Batch should still be max_seq_len, not 400
    assert batch["values"].shape[1] == 350


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
