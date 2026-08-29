"""Numerical stability tests for game_transformer.py.

Tests valid-input → NaN paths: operations that receive finite (often all-zero)
inputs but produce NaN due to degenerate arithmetic, not corrupted data.

Run:
    conda run -n pred python -m pytest deep_learning/mlb_dl/tests/test_numerical_stability.py -v
"""

import sys
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, "deep_learning")

from mlb_dl.game_transformer import (
    PerceiverResampler,
    _PerceiverCrossAttentionLayer,
    GameTransformer,
    GameTransformerLoss,
    ContextConfig,
)
from mlb_dl.game_transformer_dataset import FLAT_FEATURE_DIM, PITCH_CONTINUOUS_COLS
from mlb_dl.distributions import negbin_nll


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_context():
    return ContextConfig(sp_games=2, team_games=3, tokens_per_game=2, flat_feature_tokens=2)


@pytest.fixture
def small_model(small_context):
    return GameTransformer(
        d_model=64,
        num_backbone_layers=2,
        num_heads=4,
        d_ff=256,
        rating_dim=0,
        context_config=small_context,
        flat_feature_dim=FLAT_FEATURE_DIM,
    ).eval()


def _make_context_batch(B, config, d_model=64, rating_dim=0, device="cpu"):
    """Build a synthetic context_batch matching ContextCompiler.forward() expectations."""
    S = 15
    cont_dim = len(PITCH_CONTINUOUS_COLS)

    def _game_block(n_games, all_padded=False):
        blk = {
            "continuous": torch.randn(B, n_games, S, cont_dim, device=device),
            "batter_hash": torch.randint(1, 50000, (B, n_games, S), device=device),
            "pitcher_hash": torch.randint(1, 50000, (B, n_games, S), device=device),
            "inning_idx": torch.randint(0, 9, (B, n_games, S), device=device),
            "ab_idx": torch.randint(0, 40, (B, n_games, S), device=device),
            "pitch_idx": torch.randint(0, 10, (B, n_games, S), device=device),
            "padding_mask": torch.ones(B, n_games, S, dtype=torch.bool, device=device)
            if all_padded
            else torch.zeros(B, n_games, S, dtype=torch.bool, device=device),
            "games_ago": torch.arange(n_games, dtype=torch.float32, device=device)
            .unsqueeze(0).expand(B, -1),
            "seasons_crossed": torch.zeros(B, n_games, device=device),
        }
        return blk

    ctx = {
        "sp_home": _game_block(config.sp_games),
        "sp_away": _game_block(config.sp_games),
        "team_home": _game_block(config.team_games),
        "team_away": _game_block(config.team_games),
        "flat_features": torch.randn(B, FLAT_FEATURE_DIM, device=device),
        "weather_temporal": torch.randn(B, 4, 22, device=device),
    }
    return ctx


def _make_full_batch(B, P, T_live, config, d_model=64, rating_dim=0, device="cpu",
                     sp_all_padded=False):
    """Full batch dict matching GameTransformer.forward() input spec."""
    S = 15
    cont_dim = len(PITCH_CONTINUOUS_COLS)

    def _game_block(n_games, all_padded=False):
        return {
            "continuous": torch.zeros(B, n_games, S, cont_dim, device=device),
            "batter_hash": torch.zeros(B, n_games, S, dtype=torch.long, device=device),
            "pitcher_hash": torch.zeros(B, n_games, S, dtype=torch.long, device=device),
            "inning_idx": torch.zeros(B, n_games, S, dtype=torch.long, device=device),
            "ab_idx": torch.zeros(B, n_games, S, dtype=torch.long, device=device),
            "pitch_idx": torch.zeros(B, n_games, S, dtype=torch.long, device=device),
            "padding_mask": torch.ones(B, n_games, S, dtype=torch.bool, device=device)
            if all_padded
            else torch.zeros(B, n_games, S, dtype=torch.bool, device=device),
            "games_ago": torch.arange(n_games, dtype=torch.float32, device=device)
            .unsqueeze(0).expand(B, -1),
            "seasons_crossed": torch.zeros(B, n_games, device=device),
        }

    ctx = {
        "sp_home": _game_block(config.sp_games, all_padded=sp_all_padded),
        "sp_away": _game_block(config.sp_games, all_padded=sp_all_padded),
        "team_home": _game_block(config.team_games, all_padded=False),
        "team_away": _game_block(config.team_games, all_padded=False),
        "flat_features": torch.zeros(B, FLAT_FEATURE_DIM, device=device),
        "weather_temporal": torch.zeros(B, 4, 22, device=device),
    }

    batch = {
        "context": ctx,
        "player_hashes": torch.randint(1, 50000, (B, P), device=device),
        "player_context": torch.zeros(B, P, 2 * d_model, device=device),
    }

    if T_live > 0:
        batch["live_continuous"] = torch.zeros(B, T_live, cont_dim, device=device)
        batch["live_batter_hash"] = torch.zeros(B, T_live, dtype=torch.long, device=device)
        batch["live_pitcher_hash"] = torch.zeros(B, T_live, dtype=torch.long, device=device)
        batch["live_inning_idx"] = torch.zeros(B, T_live, dtype=torch.long, device=device)
        batch["live_ab_idx"] = torch.zeros(B, T_live, dtype=torch.long, device=device)
        batch["live_pitch_idx"] = torch.zeros(B, T_live, dtype=torch.long, device=device)

    return batch


# ============================================================================
# A. PerceiverResampler: all-padding → softmax NaN
# ============================================================================


class TestPerceiverAllPaddingNaN:
    """
    CRITICAL BUG: When key_padding_mask is all-True (new pitcher with no prior
    starts), the float kpm is all-(-inf). PyTorch's MHA softmax over all-(-inf)
    produces 0/0 = NaN, which propagates to every downstream tensor.

    Fix: torch.nan_to_num(attn_out, nan=0.0) after the cross_attn call, OR
    detect all-padding rows before building kpm and skip/zero them.
    """

    def test_all_padding_produces_nan_without_fix(self):
        """Reproduces the all-padding NaN: this is the failing state before fix."""
        d_model, num_heads = 32, 4
        layer = _PerceiverCrossAttentionLayer(d_model=d_model, num_heads=num_heads, dropout=0.0)
        layer.eval()

        B, K, S = 2, 4, 10
        q = torch.randn(B, K, d_model)
        kv = torch.randn(B, S, d_model)
        mask = torch.ones(B, S, dtype=torch.bool)  # ALL padded

        # Convert mask the same way the production code does
        kpm = torch.zeros_like(mask, dtype=q.dtype).masked_fill(mask, float("-inf"))
        with torch.no_grad():
            out, _ = layer.cross_attn(
                layer.norm_q(q), layer.norm_kv(kv), layer.norm_kv(kv),
                key_padding_mask=kpm,
            )

        # This IS expected to be NaN — we're reproducing the bug
        assert torch.isnan(out).any(), (
            "Expected NaN from all-padding softmax — if this fails the bug was "
            "fixed in PyTorch or the repro changed"
        )

    def test_all_padding_mask_perceiver_nan(self):
        """CRITICAL: PerceiverResampler.forward must not NaN with all-padding key_mask."""
        resampler = PerceiverResampler(d_model=32, num_queries=4, num_layers=2, num_heads=4)
        resampler.eval()

        B, S = 3, 12
        x = torch.randn(B, S, 32)
        mask = torch.ones(B, S, dtype=torch.bool)  # ALL padded

        with torch.no_grad():
            out = resampler(x, key_padding_mask=mask)

        assert not torch.isnan(out).any(), (
            "PerceiverResampler NaN with all-padding — new SP cold-start will "
            "corrupt all predictions. Fix: nan_to_num or all-padding gate."
        )

    def test_partial_padding_no_nan(self):
        """Partial padding (normal case) must remain NaN-free."""
        resampler = PerceiverResampler(d_model=32, num_queries=4, num_layers=2, num_heads=4)
        resampler.eval()

        B, S = 3, 12
        x = torch.randn(B, S, 32)
        # Only last 6 positions padded
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[:, 6:] = True

        with torch.no_grad():
            out = resampler(x, key_padding_mask=mask)

        assert not torch.isnan(out).any()

    def test_no_padding_mask_no_nan(self):
        """No mask at all (full game) must be NaN-free."""
        resampler = PerceiverResampler(d_model=32, num_queries=4, num_layers=2, num_heads=4)
        resampler.eval()

        B, S = 3, 12
        x = torch.randn(B, S, 32)

        with torch.no_grad():
            out = resampler(x, key_padding_mask=None)

        assert not torch.isnan(out).any()

    def test_mixed_batch_some_all_padded(self):
        """Batch where some items are fully padded, others are not — NaN must not leak."""
        resampler = PerceiverResampler(d_model=32, num_queries=4, num_layers=2, num_heads=4)
        resampler.eval()

        B, S = 4, 12
        x = torch.randn(B, S, 32)
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[0] = True   # item 0: fully padded (new pitcher)
        mask[2] = True   # item 2: fully padded

        with torch.no_grad():
            out = resampler(x, key_padding_mask=mask)

        assert not torch.isnan(out).any(), (
            "NaN leaked from the fully-padded items (0, 2) into normally-padded items"
        )


# ============================================================================
# B. Cold-start SP context: all-zeros all-padding
# ============================================================================


class TestColdStartSP:
    """
    Full end-to-end test: new SP with no prior starts produces all-zero context
    AND all-padded masks → should not produce NaN anywhere in the model output.

    This combines the LayerNorm-on-zeros path (safe, PyTorch eps handles it)
    with the all-padding softmax path (unsafe without fix).
    """

    def test_cold_start_sp_pregame_no_nan(self, small_model, small_context):
        """New SP (all-zero all-padded context) must not produce NaN at pregame."""
        B, P = 2, 5
        batch = _make_full_batch(B, P, T_live=0, config=small_context,
                                 d_model=64, sp_all_padded=True)

        with torch.no_grad():
            preds = small_model(batch)

        for k, v in preds.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any(), f"NaN in {k} (cold-start SP, pregame)"
                assert not torch.isinf(v).any(), f"Inf in {k} (cold-start SP, pregame)"

    def test_cold_start_sp_live_no_nan(self, small_model, small_context):
        """New SP (all-zero all-padded context) must not produce NaN with live pitches."""
        B, P, T = 2, 5, 8
        batch = _make_full_batch(B, P, T_live=T, config=small_context,
                                 d_model=64, sp_all_padded=True)

        with torch.no_grad():
            preds = small_model(batch)

        for k, v in preds.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any(), f"NaN in {k} (cold-start SP, live)"

    def test_all_zero_flat_features_no_nan(self, small_model, small_context):
        """All-zero flat features (park/weather missing) must not NaN."""
        B, P = 2, 5
        batch = _make_full_batch(B, P, T_live=0, config=small_context, d_model=64)

        with torch.no_grad():
            preds = small_model(batch)

        for k, v in preds.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any(), f"NaN in {k} (zero flat features)"

    def test_all_zero_weather_no_nan(self, small_model, small_context):
        """All-zero weather (pre-2017 game) must not NaN."""
        B, P = 2, 5
        batch = _make_full_batch(B, P, T_live=0, config=small_context, d_model=64)
        batch["context"]["weather_temporal"] = torch.zeros(B, 4, 22)

        with torch.no_grad():
            preds = small_model(batch)

        for k, v in preds.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any(), f"NaN in {k} (zero weather)"


# ============================================================================
# C. LayerNorm on all-zero input
# ============================================================================


class TestLayerNormAllZero:
    """
    LayerNorm uses (x - mean) / sqrt(var + eps). When x=0 everywhere, mean=0,
    var=0, result = 0/sqrt(eps) = 0. PyTorch's eps=1e-5 prevents NaN.
    This is SAFE — document with a test so it's never regressed.
    """

    def test_layernorm_all_zero_is_zero_not_nan(self):
        norm = nn.LayerNorm(64)
        x = torch.zeros(4, 10, 64)
        out = norm(x)
        assert not torch.isnan(out).any()
        assert (out == 0).all(), "LayerNorm(zeros) should return zeros (mean=0, std→eps)"

    def test_layernorm_all_zero_single_element(self):
        norm = nn.LayerNorm(1)
        x = torch.zeros(4, 1)
        out = norm(x)
        assert not torch.isnan(out).any()

    def test_layernorm_constant_non_zero_is_zero(self):
        """LayerNorm of a constant tensor (all same value) normalizes to zero."""
        norm = nn.LayerNorm(64)
        x = torch.ones(4, 10, 64) * 5.0
        out = norm(x)
        assert not torch.isnan(out).any()
        assert out.abs().max() < 1e-5, "LayerNorm(constant) should be near zero"


# ============================================================================
# D. NegBin NLL numerical boundary cases
# ============================================================================


class TestNegBinNLLBoundary:
    """
    negbin_nll has mu.clamp_min(1e-6) and alpha.clamp_min(1e-3).
    These guards make the loss finite for all valid (non-negative integer) targets.
    Tests confirm no NaN/Inf for the boundary inputs that could otherwise cause
    log(0), lgamma(0), or 0*log(0)=NaN.
    """

    def test_y_zero_is_finite(self):
        """y_true=0 with any valid (mu, alpha) must be finite."""
        y = torch.zeros(4)
        mu = torch.tensor([0.01, 0.1, 1.0, 5.0])
        alpha = torch.tensor([0.001, 0.01, 0.5, 2.0])
        nll = negbin_nll(y, mu, alpha)
        assert torch.isfinite(nll).all(), f"NaN/Inf for y=0: {nll}"

    def test_y_large_is_finite(self):
        """y_true=large (e.g. 30 runs — extreme but possible) must be finite."""
        y = torch.tensor([10.0, 20.0, 30.0])
        mu = torch.tensor([5.0, 10.0, 15.0])
        alpha = torch.tensor([0.5, 1.0, 2.0])
        nll = negbin_nll(y, mu, alpha)
        assert torch.isfinite(nll).all(), f"NaN/Inf for y=large: {nll}"

    def test_alpha_at_floor_is_finite(self):
        """alpha at clamp floor (1e-3) must be finite."""
        y = torch.tensor([0.0, 3.0, 7.0])
        mu = torch.tensor([2.0, 2.0, 2.0])
        alpha = torch.tensor([1e-3, 1e-3, 1e-3])
        nll = negbin_nll(y, mu, alpha)
        assert torch.isfinite(nll).all(), f"NaN/Inf at alpha=1e-3: {nll}"

    def test_mu_at_floor_is_finite(self):
        """mu at clamp floor (1e-6) must be finite."""
        y = torch.tensor([0.0, 0.0, 0.0])
        mu = torch.tensor([1e-6, 1e-6, 1e-6])
        alpha = torch.tensor([0.5, 0.5, 0.5])
        nll = negbin_nll(y, mu, alpha)
        assert torch.isfinite(nll).all(), f"NaN/Inf at mu→0: {nll}"

    def test_mu_zero_before_clamp_is_finite(self):
        """mu=0 before clamp (raw model output of -inf softplus) is clamped to 1e-6."""
        y = torch.tensor([0.0, 2.0])
        mu = torch.tensor([0.0, 0.0])  # will be clamped
        alpha = torch.tensor([1.0, 1.0])
        nll = negbin_nll(y, mu, alpha)
        assert torch.isfinite(nll).all()

    def test_backward_is_finite(self):
        """Loss gradient must be finite for typical inputs."""
        y = torch.tensor([0.0, 2.0, 5.0])
        mu = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
        alpha = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
        nll = negbin_nll(y, mu, alpha).mean()
        nll.backward()
        assert torch.isfinite(mu.grad).all(), "Non-finite gradient on mu"
        assert torch.isfinite(alpha.grad).all(), "Non-finite gradient on alpha"

    def test_y_zero_mu_tiny_gradient_not_nan(self):
        """
        y=0 with mu→0: the term -y*log(mu)=0*(-inf) should be 0 (not NaN).
        This is safe because 0 * anything = 0 in float arithmetic,
        but the gradient path can differ — verify backward is finite.
        """
        y = torch.tensor([0.0])
        mu = torch.tensor([1e-7], requires_grad=True)  # below clamp, will be raised to 1e-6
        alpha = torch.tensor([1.0], requires_grad=True)
        nll = negbin_nll(y, mu, alpha)
        assert torch.isfinite(nll).all()
        nll.backward()
        assert torch.isfinite(mu.grad).all()


# ============================================================================
# E. Loss function with degenerate target patterns
# ============================================================================


class TestLossDegenerate:
    """Tests for GameTransformerLoss under edge-case target patterns."""

    def _make_targets(self, B, P, device="cpu"):
        # GameTransformerLoss expects *_remaining keys (runs still to score),
        # not final totals. At pregame T=0, remaining == total.
        return {
            "home_runs_remaining": torch.randint(0, 10, (B,), device=device).float(),
            "away_runs_remaining": torch.randint(0, 10, (B,), device=device).float(),
            "home_win": torch.randint(0, 2, (B,), device=device).float(),
            "yrfi": torch.randint(0, 2, (B,), device=device).float(),
            "extra_innings": torch.randint(0, 2, (B,), device=device).float(),
            "player_hits": torch.randint(0, 3, (B, P), device=device).float(),
            "player_hr": torch.zeros(B, P, device=device),
            "player_k": torch.randint(0, 10, (B, P), device=device).float(),
            "player_hrbi": torch.randint(0, 4, (B, P), device=device).float(),
            "player_sb": torch.zeros(B, P, device=device),
            "player_mask": torch.ones(B, P, device=device),
        }

    def test_all_zero_runs_no_nan(self, small_model, small_context):
        """All targets = 0 runs (extreme shutout game) must not NaN the loss."""
        B, P = 2, 5
        loss_fn = GameTransformerLoss()
        batch = _make_full_batch(B, P, T_live=0, config=small_context, d_model=64)

        with torch.no_grad():
            preds = small_model(batch)

        targets = self._make_targets(B, P)
        targets["home_runs_remaining"] = torch.zeros(B)
        targets["away_runs_remaining"] = torch.zeros(B)

        loss, _ = loss_fn(preds, targets)
        assert torch.isfinite(loss), f"Loss NaN with all-zero runs: {loss}"

    def test_all_zero_players_no_nan(self, small_model, small_context):
        """All player stats = 0 (0-fer performance) must not NaN the loss."""
        B, P = 2, 5
        loss_fn = GameTransformerLoss()
        batch = _make_full_batch(B, P, T_live=0, config=small_context, d_model=64)

        with torch.no_grad():
            preds = small_model(batch)

        targets = self._make_targets(B, P)
        targets["player_hits"] = torch.zeros(B, P)
        targets["player_hr"] = torch.zeros(B, P)
        targets["player_k"] = torch.zeros(B, P)
        targets["player_hrbi"] = torch.zeros(B, P)
        targets["player_sb"] = torch.zeros(B, P)

        loss, _ = loss_fn(preds, targets)
        assert torch.isfinite(loss), f"Loss NaN with all-zero players: {loss}"

    def test_large_run_targets_no_nan(self, small_model, small_context):
        """Large run totals (25+) — extreme but valid — must not NaN."""
        B, P = 2, 5
        loss_fn = GameTransformerLoss()
        batch = _make_full_batch(B, P, T_live=0, config=small_context, d_model=64)

        with torch.no_grad():
            preds = small_model(batch)

        targets = self._make_targets(B, P)
        targets["home_runs_remaining"] = torch.full((B,), 25.0)
        targets["away_runs_remaining"] = torch.full((B,), 25.0)

        loss, _ = loss_fn(preds, targets)
        assert torch.isfinite(loss), f"Loss NaN with large runs: {loss}"


# ============================================================================
# F. Attention mask correctness for edge-case T values
# ============================================================================


class TestAttentionMaskEdgeCases:
    """Tests for _build_prefix_lm_mask under extreme live-token counts."""

    def test_prefix_lm_mask_pregame(self, small_model):
        """Pregame (num_live=0): all context tokens attend each other bidirectionally."""
        num_context, num_live = 20, 0
        mask = small_model.backbone._build_prefix_lm_mask(num_context, num_live,
                                                           device=torch.device("cpu"))
        # Context-to-context block should be all 0 (no masking)
        assert (mask[:num_context, :num_context] == 0).all()
        assert mask.shape == (num_context, num_context)

    def test_prefix_lm_mask_single_live_token(self, small_model):
        """Single live token: sees all context, attends causally (itself only)."""
        num_context, num_live = 20, 1
        mask = small_model.backbone._build_prefix_lm_mask(num_context, num_live,
                                                           device=torch.device("cpu"))
        total = num_context + num_live
        assert mask.shape == (total, total)
        # Live token (row 20) attending live tokens: should be 0 for col 20 (itself), -inf otherwise
        assert mask[20, 20] == 0
        # Live token CAN attend all context tokens
        assert (mask[20, :num_context] == 0).all()

    def test_prefix_lm_mask_context_cannot_see_live(self, small_model):
        """Context tokens must NOT attend live tokens (prefix-LM invariant)."""
        num_context, num_live = 20, 5
        mask = small_model.backbone._build_prefix_lm_mask(num_context, num_live,
                                                           device=torch.device("cpu"))
        # Context rows (0..19) attending live columns (20..24) must be -inf
        context_to_live = mask[:num_context, num_context:]
        assert (context_to_live == float("-inf")).all(), (
            "Context tokens must not attend live tokens"
        )

    def test_prefix_lm_mask_no_nan(self, small_model):
        """The mask itself must not contain NaN."""
        mask = small_model.backbone._build_prefix_lm_mask(16, 8,
                                                            device=torch.device("cpu"))
        assert not torch.isnan(mask).any()
        # Only 0 and -inf are valid values in this mask
        is_zero = mask == 0
        is_neginf = mask == float("-inf")
        assert (is_zero | is_neginf).all()


# ============================================================================
# G. _decode_negbin with extreme raw values (Item 6)
# ============================================================================


class TestDecodeNegBinExtremeRaw:
    """
    GameTransformer._decode_negbin: F.softplus(raw).clamp_min(floor).
    softplus(-100) → 0 (underflow), clamped to 0.01/0.1 — safe.
    softplus(+100) → 100 (PyTorch threshold guard) — safe.
    Verifies no NaN for any extreme raw value a randomly-initialized model could produce.
    """

    def test_very_negative_raw(self):
        """raw = -100: softplus underflows to 0, clamp_min catches it."""
        model = GameTransformer(d_model=64, num_backbone_layers=2, num_heads=4, d_ff=128)
        raw = torch.tensor([[-100.0, -100.0], [-1000.0, -1000.0]])
        mu, alpha = model._decode_negbin(raw)
        assert not torch.isnan(mu).any(), f"NaN in mu from very negative raw: {mu}"
        assert not torch.isnan(alpha).any(), f"NaN in alpha from very negative raw: {alpha}"
        assert (mu >= 0.01).all(), f"mu below floor: {mu}"
        assert (alpha >= 0.1).all(), f"alpha below floor: {alpha}"

    def test_very_positive_raw(self):
        """raw = +100: PyTorch softplus threshold returns x directly, no overflow."""
        model = GameTransformer(d_model=64, num_backbone_layers=2, num_heads=4, d_ff=128)
        raw = torch.tensor([[100.0, 100.0], [50.0, 50.0]])
        mu, alpha = model._decode_negbin(raw)
        assert not torch.isnan(mu).any(), f"NaN in mu from positive raw: {mu}"
        assert not torch.isinf(mu).any(), f"Inf in mu from positive raw: {mu}"
        assert torch.allclose(mu, torch.tensor([100.0, 50.0]), atol=0.1)

    def test_zero_raw(self):
        """raw = 0: softplus(0) = ln(2) ≈ 0.693, above both floors."""
        model = GameTransformer(d_model=64, num_backbone_layers=2, num_heads=4, d_ff=128)
        raw = torch.tensor([[0.0, 0.0]])
        mu, alpha = model._decode_negbin(raw)
        expected = torch.tensor([0.6931])  # ln(2)
        assert torch.allclose(mu, expected, atol=0.01)
        assert not torch.isnan(mu).any()


# ============================================================================
# H. PlayerQueryHead._derive_prop_probs edge cases (Item 7)
# ============================================================================


class TestDerivePlayerProps:
    """
    _derive_prop_probs: softmax probs → pow() computations.
    Key safety property: softmax outputs are always > 0, so pow(x, n) > 0 for any n.
    pa_floor is clamped to [1, 7], so exponents are always >= 0.
    pow(0, 0) = 1 in IEEE 754, but softmax never gives exactly 0.
    """

    @pytest.fixture
    def player_head(self):
        from mlb_dl.game_transformer import PlayerQueryHead
        return PlayerQueryHead(d_model=64, hash_buckets=1000, player_embed_dim=8)

    def test_extreme_k_batter(self, player_head):
        """Batter almost always strikes out (p_K ≈ 0.95)."""
        import torch.nn.functional as F
        # Manually set pa_probs: mostly K
        pa_probs = torch.tensor([[0.95, 0.03, 0.01, 0.005, 0.003, 0.001, 0.001]])
        pa_probs = pa_probs / pa_probs.sum(dim=-1, keepdim=True)
        pa_count = torch.tensor([4.0])
        result = player_head._derive_prop_probs(pa_probs, pa_count)
        for key, val in result.items():
            assert not torch.isnan(val).any(), f"NaN in {key} (high-K batter)"
            assert not torch.isinf(val).any(), f"Inf in {key} (high-K batter)"

    def test_minimum_pa_floor(self, player_head):
        """pa_count = 2.5 (offset from softplus) → floor=2, clamped to max(1,2)=2.
        (pa_floor-1).clamp_min(0) = 1, so pow(x, 1) — no edge case."""
        import torch.nn.functional as F
        pa_probs = F.softmax(torch.randn(4, 7), dim=-1)
        # Minimum pa_count from model: softplus(0) + 2.5 = 0.693 + 2.5 = 3.193
        # But to test the floor=1 edge case directly:
        pa_count = torch.tensor([1.0, 1.0, 1.0, 1.0])  # floor=1
        result = player_head._derive_prop_probs(pa_probs, pa_count)
        for key, val in result.items():
            assert not torch.isnan(val).any(), f"NaN in {key} (pa_floor=1)"

    def test_pa_floor_equals_one_pow_zero(self, player_head):
        """When pa_floor=1: (pa_floor-1).clamp_min(0) = 0 → pow(x, 0) = 1.
        This is the IEEE 754 convention. Verify it holds for all softmax outputs."""
        import torch.nn.functional as F
        pa_probs = F.softmax(torch.randn(8, 7), dim=-1)
        pa_count = torch.ones(8)  # floor → 1
        result = player_head._derive_prop_probs(pa_probs, pa_count)
        # p_1_hit_lo uses pow((pa_floor-1).clamp_min(0)) = pow(0) = 1
        assert not torch.isnan(result["hits_categorical"]).any()
        # hits_categorical should sum to 1
        sums = result["hits_categorical"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_near_zero_hit_probability(self, player_head):
        """p_hit_single ≈ 1e-7: pow(1e-7, 7) underflows to 0 (not NaN)."""
        pa_probs = torch.tensor([[0.45, 0.45, 0.0999999, 1e-7, 1e-8, 1e-8, 1e-8]])
        pa_probs = pa_probs / pa_probs.sum(dim=-1, keepdim=True)
        pa_count = torch.tensor([5.0])
        result = player_head._derive_prop_probs(pa_probs, pa_count)
        for key, val in result.items():
            assert not torch.isnan(val).any(), f"NaN in {key} (near-zero hit prob)"
        # Almost all mass should be on 0 hits
        assert result["hits_categorical"][0, 0] > 0.99

    def test_high_hr_rate(self, player_head):
        """Extreme HR hitter (pa_probs[6] ≈ 0.5): p_1plus_hr should be high."""
        pa_probs = torch.tensor([[0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.5]])
        pa_probs = pa_probs / pa_probs.sum(dim=-1, keepdim=True)
        pa_count = torch.tensor([4.5])
        result = player_head._derive_prop_probs(pa_probs, pa_count)
        assert not torch.isnan(result["hr_prob"]).any()
        # With 50% HR rate and ~4.5 PA, P(1+HR) should be very high
        assert result["hr_prob"].item() > 0.9


# ============================================================================
# I. KV cache shape consistency (Item 8)
# ============================================================================


class TestKVCacheConsistency:
    """
    KV cache correctness: incremental live-token decoding must produce numerically
    identical outputs to a full-sequence forward pass.

    The fix passes attn_mask=None in the kv_cache branch of _TransformerBlock to
    avoid the [T,T] mask shape mismatch with [T, cached+T] key length.
    """

    def test_kv_cache_with_prefix_lm_mask_no_error(self):
        """After fix: KV cache + prefix-LM mask must not raise and must not NaN."""
        from mlb_dl.game_transformer import _TransformerBlock, GameTransformerBackbone
        d_model = 64
        block = _TransformerBlock(d_model=d_model, num_heads=4, d_ff=128, dropout=0.0)
        block.eval()

        B = 2
        num_context, num_live = 8, 4
        total = num_context + num_live
        x = torch.randn(B, total, d_model)

        backbone = GameTransformerBackbone(d_model=d_model, num_heads=4, num_layers=1, d_ff=128)
        attn_mask = backbone._build_prefix_lm_mask(num_context, num_live, x.device)

        with torch.no_grad():
            _, cache = block(x, attn_mask=attn_mask)
            out, _ = block(x, attn_mask=attn_mask, kv_cache=cache)

        assert not torch.isnan(out).any(), "KV cache forward produced NaN"
        assert out.shape == (B, total, d_model)

    def test_kv_cache_without_mask_no_nan(self):
        """KV cache without attn_mask should work (no shape constraint)."""
        from mlb_dl.game_transformer import _TransformerBlock
        d_model = 64
        block = _TransformerBlock(d_model=d_model, num_heads=4, d_ff=128, dropout=0.0)
        block.eval()

        B, T = 2, 6
        x = torch.randn(B, T, d_model)

        with torch.no_grad():
            _, cache = block(x, attn_mask=None)
            x_new = torch.randn(B, 2, d_model)
            out, _ = block(x_new, attn_mask=None, kv_cache=cache)

        assert not torch.isnan(out).any(), "KV cache (no mask) produced NaN"
        assert out.shape == (B, 2, d_model)

    def test_incremental_decode_matches_full_sequence(self):
        """Incremental live-token decoding must match full-sequence inference numerically.

        Production invariant: encode context once, then decode each new live pitch
        one token at a time with the accumulated KV cache. The live-token outputs
        must be bit-close to what the full-sequence pass produces for those positions.

        Proof of correctness (by induction):
        - Context representations are identical in both passes: prefix-LM blocks
          context from attending to live tokens, so context reps are unaffected
          by live token presence.
        - For each live token t, the cached K/V from prior tokens 0..t-1 are the
          same as what the full-sequence pass would use (by inductive hypothesis),
          so live_t attends to the same keys/values and produces the same output.
        """
        from mlb_dl.game_transformer import GameTransformerBackbone

        torch.manual_seed(42)
        d_model = 64
        num_heads = 4
        num_layers = 2
        num_context = 8
        num_live = 4
        B = 2

        backbone = GameTransformerBackbone(
            d_model=d_model, num_heads=num_heads, num_layers=num_layers, d_ff=128, dropout=0.0
        )
        backbone.eval()

        context = torch.randn(B, num_context, d_model)
        live_tokens = torch.randn(B, num_live, d_model)
        x_full = torch.cat([context, live_tokens], dim=1)

        with torch.no_grad():
            # Full-sequence reference pass
            full_out, _ = backbone(x_full, num_context=num_context)
            live_ref = full_out[:, num_context:, :]  # [B, num_live, d_model]

            # Incremental pass: encode context, then decode one live token at a time
            _, cache = backbone(context, num_context=num_context)

            incremental_outs = []
            for t in range(num_live):
                token = live_tokens[:, t : t + 1, :]  # [B, 1, d_model]
                out, cache = backbone(token, num_context=0, kv_cache=cache)
                incremental_outs.append(out)  # [B, 1, d_model]

            live_incremental = torch.cat(incremental_outs, dim=1)  # [B, num_live, d_model]

        torch.testing.assert_close(
            live_incremental, live_ref, rtol=1e-4, atol=1e-4,
            msg="Incremental KV-cache decode diverges from full-sequence inference",
        )
