"""Hierarchical Attention Network for live MLB game state prediction.

Architecture exploits baseball's natural hierarchy: pitches → at-bats → half-innings → game.
Chosen over alternatives because:
1. Explicit hierarchical structure matches sport (Yang et al., 2016)
2. Intermediate representations map to market targets (YRFI = first half-inning; F5 = first 10)
3. Max sequence length 350 pitches → O(S²) attention feasible (~122K ops/layer)
4. Mamba/SSM lacks interpretable intermediate structure needed for inning-level markets

KNOWN LIMITATIONS (production TODOs):
1. Hierarchical segmentation: Current implementation pools entire pitch sequence at each level
   rather than properly segmenting by AB/inning boundaries. Production version needs to
   implement ragged-batch processing to respect hierarchy.
2. Pregame/live blending: Gate currently operates on already-conditioned representation.
   Production version should maintain separate pregame/live paths and blend outputs.
3. All hyperparameters marked with "TODO: validate" need empirical ablation studies.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PitchEncoder(nn.Module):
    """Raw pitch features → d_model embedding.

    Handles:
    - Continuous kinematics (20): velocity, movement, spin
    - Categorical (9): pitch type one-hot
    - Binary flags (4): outcome (strike/ball/in_play/foul)
    - Count/game state (11): balls, strikes, outs, inning, run_diff, bases, scores
    - Player identity (2): batter/pitcher hash → embedding lookup
    - Handedness (2): bat_side, pitch_hand
    - Positional/temporal (4): pitch numbers, elapsed time

    Total: 20 + 9 + 4 + 11 + 2 + 4 = 50 continuous dims + 2 hash lookups
    """

    def __init__(
        self,
        d_model: int = 128,  # TODO: validate — placeholder (ablation 64/128/256)
        batter_buckets: int = 512,  # blake2b hash-bucket
        pitcher_buckets: int = 512,  # blake2b hash-bucket
        player_embed_dim: int = 16,  # TODO: validate — placeholder
        dropout: float = 0.1,  # TODO: validate — placeholder
    ):
        super().__init__()
        self.batter_embed = nn.Embedding(batter_buckets, player_embed_dim, padding_idx=0)
        self.pitcher_embed = nn.Embedding(pitcher_buckets, player_embed_dim, padding_idx=0)

        # Continuous features breakdown:
        # 20 (kinematics) + 9 (pitch_type) + 4 (outcome_flags) + 3 (count) +
        # 3 (game) + 3 (base) + 2 (handedness) + 2 (score) + 2 (positional) +
        # 1 (intra_ab) + 1 (elapsed) = 50 dims
        # After embedding concat: 50 + 2*player_embed
        continuous_dim = 50
        input_dim = continuous_dim + 2 * player_embed_dim

        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        continuous: torch.Tensor,  # [B, S, 20] kinematics
        pitch_type_onehot: torch.Tensor,  # [B, S, 9]
        outcome_flags: torch.Tensor,  # [B, S, 4]
        count_state: torch.Tensor,  # [B, S, 3]
        game_state: torch.Tensor,  # [B, S, 3]
        base_state: torch.Tensor,  # [B, S, 3]
        batter_hash: torch.LongTensor,  # [B, S]
        pitcher_hash: torch.LongTensor,  # [B, S]
        handedness: torch.Tensor,  # [B, S, 2]
        score: torch.Tensor,  # [B, S, 2]
        positional: torch.Tensor,  # [B, S, 2]
        intra_ab: torch.Tensor,  # [B, S, 1]
        elapsed_time: torch.Tensor,  # [B, S, 1]
    ) -> torch.Tensor:
        """Returns [B, S, d_model]."""
        batter_emb = self.batter_embed(batter_hash)  # [B, S, player_embed_dim]
        pitcher_emb = self.pitcher_embed(pitcher_hash)  # [B, S, player_embed_dim]

        # Concatenate all continuous features
        features = torch.cat([
            continuous,
            pitch_type_onehot,
            outcome_flags,
            count_state,
            game_state,
            base_state,
            handedness,
            score,
            positional,
            intra_ab,
            elapsed_time,
        ], dim=-1)  # [B, S, 52]

        # Add player embeddings
        x = torch.cat([features, batter_emb, pitcher_emb], dim=-1)
        return self.proj(x)  # [B, S, d_model]


class HierarchicalPositionalEncoding(nn.Module):
    """Learned positional encoding: inning_emb + ab_in_inning_emb + pitch_in_ab_emb.

    Baseball has natural hierarchical indices that attention alone can't infer.
    Learned rather than sinusoidal because: (1) max values are small (18 innings,
    ~20 ABs/inning, ~10 pitches/AB); (2) irregular hierarchy depth (extra innings).
    """

    def __init__(
        self,
        d_model: int = 128,
        max_innings: int = 20,  # Handles 18-inning game + buffer
        max_abs_per_inning: int = 25,  # TODO: validate — placeholder (check data)
        max_pitches_per_ab: int = 15,  # TODO: validate — placeholder (check data)
    ):
        super().__init__()
        self.inning_emb = nn.Embedding(max_innings, d_model)
        self.ab_emb = nn.Embedding(max_abs_per_inning, d_model)
        self.pitch_emb = nn.Embedding(max_pitches_per_ab, d_model)

    def forward(
        self,
        inning_idx: torch.LongTensor,  # [B, S]
        ab_idx: torch.LongTensor,  # [B, S]
        pitch_idx: torch.LongTensor,  # [B, S]
    ) -> torch.Tensor:
        """Returns [B, S, d_model]."""
        return (
            self.inning_emb(inning_idx) +
            self.ab_emb(ab_idx) +
            self.pitch_emb(pitch_idx)
        )


class AttentionPool(nn.Module):
    """Attention-weighted pooling of variable-length sequences into fixed representation.

    Q = learned parameter (single query vector), K/V = sequence elements.
    Produces weighted average where weights = softmax(Q^T K).
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) / math.sqrt(d_model))
        self.scale = math.sqrt(d_model)

    def forward(
        self,
        x: torch.Tensor,  # [B, S, d_model]
        mask: torch.BoolTensor,  # [B, S] — True for valid positions
    ) -> torch.Tensor:
        """Returns [B, d_model]."""
        # Q: [1, 1, d_model] broadcast to [B, 1, d_model]
        # K: [B, S, d_model]
        q = self.query.expand(x.size(0), -1, -1)
        attn_logits = torch.bmm(q, x.transpose(1, 2)) / self.scale  # [B, 1, S]

        # Mask invalid positions
        attn_logits = attn_logits.masked_fill(~mask.unsqueeze(1), float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, 1, S]

        pooled = torch.bmm(attn_weights, x)  # [B, 1, d_model]
        return pooled.squeeze(1)  # [B, d_model]


class PitchLevelEncoder(nn.Module):
    """Transformer encoder over pitches within an at-bat.

    2 layers, 4 heads, standard pre-norm architecture.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,  # TODO: validate — placeholder
        n_layers: int = 2,  # TODO: validate — placeholder
        dim_feedforward: int = 512,  # TODO: validate — placeholder (4*d_model standard)
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm (Xiong et al., 2020)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        x: torch.Tensor,  # [B, S, d_model]
        mask: torch.BoolTensor,  # [B, S] — True for valid
    ) -> torch.Tensor:
        """Returns [B, S, d_model]."""
        # TransformerEncoder expects src_key_padding_mask: True = ignore
        padding_mask = ~mask
        return self.encoder(x, src_key_padding_mask=padding_mask)


class ABLevelEncoder(nn.Module):
    """Transformer encoder over at-bats within a half-inning.

    2 layers, 4 heads, standard pre-norm architecture.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        x: torch.Tensor,  # [B, S, d_model]
        mask: torch.BoolTensor,  # [B, S]
    ) -> torch.Tensor:
        """Returns [B, S, d_model]."""
        padding_mask = ~mask
        return self.encoder(x, src_key_padding_mask=padding_mask)


class GameLevelEncoder(nn.Module):
    """Transformer encoder over half-innings played so far.

    2 layers, 4 heads, standard pre-norm architecture.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        x: torch.Tensor,  # [B, S, d_model]
        mask: torch.BoolTensor,  # [B, S]
    ) -> torch.Tensor:
        """Returns [B, S, d_model]."""
        padding_mask = ~mask
        return self.encoder(x, src_key_padding_mask=padding_mask)


class CrossAttentionMerger(nn.Module):
    """Conditions live game representation on pregame prior via cross-attention.

    Q = live tokens, K/V = pregame prior (single token).
    Inspired by FiLM conditioning (Perez et al., 2018) but using attention mechanism.

    The pregame prior captures team strength, pitcher matchup, park effects, etc.
    Cross-attention allows live model to selectively weight pregame components
    based on observed in-game state.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_pregame: int = 128,  # TODO: validate — placeholder (pregame feature dim)
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Project pregame features to d_model if needed
        self.pregame_proj = nn.Linear(d_pregame, d_model)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        live_repr: torch.Tensor,  # [B, d_model] — current live game state
        pregame_prior: torch.Tensor,  # [B, d_pregame] — pregame features
    ) -> torch.Tensor:
        """Returns [B, d_model]."""
        # Project pregame to d_model and expand to [B, 1, d_model]
        kv = self.pregame_proj(pregame_prior).unsqueeze(1)

        # Query: live state [B, 1, d_model]
        q = live_repr.unsqueeze(1)

        # Cross-attention: Q from live, K/V from pregame
        attn_out, _ = self.cross_attn(q, kv, kv)  # [B, 1, d_model]

        # Residual connection + norm
        out = self.norm(q + self.dropout(attn_out))
        return out.squeeze(1)  # [B, d_model]


class NegBinOutputHead(nn.Module):
    """Predicts Negative Binomial parameters (mu, alpha) for remaining runs.

    NegBin chosen over Poisson because:
    1. Baseball run scoring exhibits overdispersion (variance > mean)
    2. Empirically validated in pregame module (see CLAUDE.md Track NegBin)
    3. Flexible dispersion parameter α captures varying uncertainty

    Softplus ensures positivity: mu, alpha > 0.
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int = 64,  # TODO: validate — placeholder
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # [mu, alpha]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, alpha) both [B]."""
        params = self.net(x)  # [B, 2]
        mu = F.softplus(params[:, 0]) + 1e-4  # Numerical stability
        alpha = F.softplus(params[:, 1]) + 1e-4
        return mu, alpha


class ClassificationHead(nn.Module):
    """Binary classification head with optional game-progress masking.

    For targets that become deterministic during game:
    - YRFI: settled after 1st inning
    - Extra innings: settled after 9th inning regulation

    Masking not applied at inference (return logit regardless), but can be used
    during training to prevent gradient flow for settled outcomes.
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logit [B]."""
        return self.net(x).squeeze(-1)


class LiveHANModel(nn.Module):
    """Full hierarchical attention model for live MLB game state prediction.

    Pipeline:
      1. PitchEncoder: raw features → d_model embeddings
      2. HierarchicalPositionalEncoding: add structural position
      3. PitchLevelEncoder: attention over pitches within each AB
      4. AttentionPool: pool pitches → AB representation
      5. ABLevelEncoder: attention over ABs within each half-inning
      6. AttentionPool: pool ABs → half-inning representation
      7. GameLevelEncoder: attention over half-innings
      8. AttentionPool: pool half-innings → game state
      9. CrossAttentionMerger: condition on pregame prior
      10. PregameGate: learned weighting of live vs pregame as game progresses
      11. Output heads: NegBin (remaining runs) + classification (win, YRFI, extra innings)

    Key invariant: hierarchy_indices must correctly segment the pitch sequence.
    The model assumes pitches are pre-sorted by (game, inning, AB, pitch) and that
    indices encode segment boundaries.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_pregame: int = 128,
        batter_buckets: int = 512,
        pitcher_buckets: int = 512,
        player_embed_dim: int = 16,
        n_heads: int = 4,
        n_layers_per_level: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_innings: int = 20,
        max_abs_per_inning: int = 25,
        max_pitches_per_ab: int = 15,
    ):
        super().__init__()

        # Stage 1: Pitch encoding
        self.pitch_encoder = PitchEncoder(
            d_model=d_model,
            batter_buckets=batter_buckets,
            pitcher_buckets=pitcher_buckets,
            player_embed_dim=player_embed_dim,
            dropout=dropout,
        )
        self.pos_encoding = HierarchicalPositionalEncoding(
            d_model=d_model,
            max_innings=max_innings,
            max_abs_per_inning=max_abs_per_inning,
            max_pitches_per_ab=max_pitches_per_ab,
        )

        # Stage 2: Pitch-level processing
        self.pitch_level_encoder = PitchLevelEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers_per_level,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.pitch_pool = AttentionPool(d_model)

        # Stage 3: AB-level processing
        self.ab_level_encoder = ABLevelEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers_per_level,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.ab_pool = AttentionPool(d_model)

        # Stage 4: Game-level processing
        self.game_level_encoder = GameLevelEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers_per_level,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.game_pool = AttentionPool(d_model)

        # Stage 5: Pregame conditioning
        self.cross_attn_merger = CrossAttentionMerger(
            d_model=d_model,
            d_pregame=d_pregame,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Stage 6: Pregame-to-live gating
        # WHY separate paths: at t=0 (no pitches), model should output pure pregame.
        # As game progresses, live signal dominates. Gate learns this transition.
        self.pregame_only_proj = nn.Linear(d_pregame, d_model)
        self.pregame_gate = nn.Sequential(
            nn.Linear(d_model + 1, 64),  # +1 for game_progress scalar
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Stage 7: Output heads
        self.negbin_home = NegBinOutputHead(d_model, dropout=dropout)
        self.negbin_away = NegBinOutputHead(d_model, dropout=dropout)
        self.head_home_win = ClassificationHead(d_model, dropout=dropout)
        self.head_extra_innings = ClassificationHead(d_model, dropout=dropout)
        self.head_yrfi = ClassificationHead(d_model, dropout=dropout)

    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch: Dictionary with keys:
                - continuous: [B, S, 20] - kinematic features
                - pitch_type: [B, S, 9] - one-hot
                - outcome_flags: [B, S, 4] - binary flags
                - count_state: [B, S, 3] - balls/strikes/outs
                - game_state: [B, S, 3] - inning/is_top/run_diff
                - base_state: [B, S, 3] - runner flags
                - batter_hash: [B, S] - LongTensor
                - pitcher_hash: [B, S] - LongTensor
                - handedness: [B, S, 2] - binary
                - score: [B, S, 2] - home_runs, away_runs
                - positional: [B, S, 2] - pitch_in_game, pitch_in_AB
                - intra_ab: [B, S, 1] - pitches_seen_this_AB
                - elapsed_time: [B, S, 1] - minutes since first pitch
                - hierarchy_indices: [B, S, 3] - (inning_idx, ab_idx, pitch_idx)
                - attention_mask: [B, S] - BoolTensor, True for valid
                - pregame_prior: [B, d_pregame] - pregame feature vector
                - game_progress: [B] - fraction of game elapsed (0-1)

        Returns:
            Dictionary with keys:
                - mu_home_remaining: [B]
                - alpha_home_remaining: [B]
                - mu_away_remaining: [B]
                - alpha_away_remaining: [B]
                - home_win_logit: [B]
                - extra_innings_logit: [B]
                - yrfi_logit: [B]
        """
        # Stage 1: Encode pitches
        pitch_emb = self.pitch_encoder(
            continuous=batch["continuous"],
            pitch_type_onehot=batch["pitch_type"],
            outcome_flags=batch["outcome_flags"],
            count_state=batch["count_state"],
            game_state=batch["game_state"],
            base_state=batch["base_state"],
            batter_hash=batch["batter_hash"],
            pitcher_hash=batch["pitcher_hash"],
            handedness=batch["handedness"],
            score=batch["score"],
            positional=batch["positional"],
            intra_ab=batch["intra_ab"],
            elapsed_time=batch["elapsed_time"],
        )  # [B, S, d_model]

        # Add hierarchical positional encoding
        hierarchy = batch["hierarchy_indices"]  # [B, S, 3]
        pos_emb = self.pos_encoding(
            inning_idx=hierarchy[:, :, 0],
            ab_idx=hierarchy[:, :, 1],
            pitch_idx=hierarchy[:, :, 2],
        )  # [B, S, d_model]
        x = pitch_emb + pos_emb

        mask = batch["attention_mask"]  # [B, S]

        # Stage 2: Pitch-level encoding
        # NOTE: Full implementation would need to segment pitches by AB and process separately.
        # For now, we process the full sequence at each level and rely on attention masking.
        # A production version should implement hierarchical batching to truly respect
        # the pitch→AB→inning boundaries. This is a simplification.
        # TODO: validate — implement proper hierarchical segmentation

        x = self.pitch_level_encoder(x, mask)  # [B, S, d_model]

        # Pool pitches to AB level (simplified: pools entire sequence)
        ab_repr = self.pitch_pool(x, mask)  # [B, d_model]

        # Stage 3: AB-level encoding
        # In full implementation, would have [B, num_abs, d_model] tensor
        # For now, expand pooled AB back to sequence for demonstration
        # TODO: validate — implement proper hierarchical segmentation
        ab_seq = ab_repr.unsqueeze(1)  # [B, 1, d_model]
        ab_mask = torch.ones(ab_seq.size(0), 1, dtype=torch.bool, device=ab_seq.device)

        ab_encoded = self.ab_level_encoder(ab_seq, ab_mask)  # [B, 1, d_model]

        # Pool ABs to half-inning level
        inning_repr = self.ab_pool(ab_encoded, ab_mask)  # [B, d_model]

        # Stage 4: Game-level encoding
        # Similar simplification
        inning_seq = inning_repr.unsqueeze(1)  # [B, 1, d_model]
        inning_mask = torch.ones(inning_seq.size(0), 1, dtype=torch.bool, device=inning_seq.device)

        game_encoded = self.game_level_encoder(inning_seq, inning_mask)  # [B, 1, d_model]
        game_repr = self.game_pool(game_encoded, inning_mask)  # [B, d_model]

        # Stage 5: Cross-attention with pregame prior
        conditioned = self.cross_attn_merger(game_repr, batch["pregame_prior"])  # [B, d_model]

        # Stage 6: Pregame-to-live gating
        # Gate increases weight on live state as game progresses
        game_progress = batch["game_progress"].unsqueeze(-1)  # [B, 1]
        gate_input = torch.cat([conditioned, game_progress], dim=-1)
        live_weight = self.pregame_gate(gate_input)  # [B, 1]

        # Blend live-conditioned representation with pregame-only representation
        pregame_repr = self.pregame_only_proj(batch["pregame_prior"])  # [B, d_model]
        final_repr = live_weight * conditioned + (1 - live_weight) * pregame_repr

        # Stage 7: Output heads
        mu_home, alpha_home = self.negbin_home(final_repr)
        mu_away, alpha_away = self.negbin_away(final_repr)

        return {
            "mu_home_remaining": mu_home,
            "alpha_home_remaining": alpha_home,
            "mu_away_remaining": mu_away,
            "alpha_away_remaining": alpha_away,
            "home_win_logit": self.head_home_win(final_repr),
            "extra_innings_logit": self.head_extra_innings(final_repr),
            "yrfi_logit": self.head_yrfi(final_repr),
        }


def _test_forward_pass():
    """Smoke test: verify shapes and no NaN outputs."""
    B, S = 4, 100  # 4 games, 100 pitches
    d_pregame = 128

    model = LiveHANModel(
        d_model=128,
        d_pregame=d_pregame,
        batter_buckets=512,
        pitcher_buckets=512,
    )

    # Mock batch
    batch = {
        "continuous": torch.randn(B, S, 20),
        "pitch_type": F.one_hot(torch.randint(0, 9, (B, S)), num_classes=9).float(),
        "outcome_flags": torch.randint(0, 2, (B, S, 4)).float(),
        "count_state": torch.randint(0, 4, (B, S, 3)).float(),
        "game_state": torch.randn(B, S, 3),
        "base_state": torch.randint(0, 2, (B, S, 3)).float(),
        "batter_hash": torch.randint(0, 512, (B, S)),
        "pitcher_hash": torch.randint(0, 512, (B, S)),
        "handedness": torch.randint(0, 2, (B, S, 2)).float(),
        "score": torch.randint(0, 10, (B, S, 2)).float(),
        "positional": torch.randn(B, S, 2),
        "intra_ab": torch.randint(1, 10, (B, S, 1)).float(),
        "elapsed_time": torch.randn(B, S, 1).abs(),
        "hierarchy_indices": torch.stack([
            torch.randint(0, 10, (B, S)),  # inning
            torch.randint(0, 20, (B, S)),  # ab
            torch.randint(0, 10, (B, S)),  # pitch
        ], dim=-1),
        "attention_mask": torch.ones(B, S, dtype=torch.bool),
        "pregame_prior": torch.randn(B, d_pregame),
        "game_progress": torch.rand(B),
    }

    model.eval()
    with torch.no_grad():
        out = model(batch)

    # Verify shapes
    assert out["mu_home_remaining"].shape == (B,)
    assert out["alpha_home_remaining"].shape == (B,)
    assert out["mu_away_remaining"].shape == (B,)
    assert out["alpha_away_remaining"].shape == (B,)
    assert out["home_win_logit"].shape == (B,)
    assert out["extra_innings_logit"].shape == (B,)
    assert out["yrfi_logit"].shape == (B,)

    # Verify no NaN
    for key, val in out.items():
        assert not torch.isnan(val).any(), f"NaN detected in {key}"

    # Verify positivity constraints for NegBin params
    assert (out["mu_home_remaining"] > 0).all()
    assert (out["alpha_home_remaining"] > 0).all()
    assert (out["mu_away_remaining"] > 0).all()
    assert (out["alpha_away_remaining"] > 0).all()

    print("✓ Forward pass smoke test passed")


if __name__ == "__main__":
    _test_forward_pass()
