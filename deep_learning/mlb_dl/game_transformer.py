"""Unified pitch-by-pitch Transformer for pregame + live MLB market pricing.

Architecture: Perceiver-style context compression + prefix-LM Transformer backbone.
Prices all markets (team totals, win, YRFI, player props) from a single forward pass.

Key design decisions:
1. PerceiverResampler compresses variable-length game histories into fixed K=4 tokens,
   enabling O(1) inference cost per historical game regardless of pitch count.
2. Prefix-LM attention mask: context tokens attend bidirectionally; live tokens are
   causal. This allows pregame pricing (T=0) and autoregressive live updates.
3. Game-index decay (not calendar decay) avoids penalizing offseason gaps.
4. Shared PitchEncoder for both context compilation and live encoding — parameter
   efficiency and consistent representation space.

Total parameters: ~8-10M (fits 16GB VRAM at batch_size=64).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import negbin_nll
from .game_transformer_dataset import FLAT_FEATURE_DIM
from .rating_sequences import RATING_SEQ_STEPS


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ContextConfig:
    """Ablation hooks for context assembly strategies."""

    team_context_mode: str = "all_games"  # "all_games" | "lineup_overlap" | "similarity_weighted"
    matchup_mode: str = "none"  # "none" | "raw_sequences" | "compressed_summary"
    bullpen_mode: str = "implicit"  # "implicit" | "explicit_profile" | "reliever_sequences"

    # Context budget (tokens per category)
    sp_games: int = 5
    team_games: int = 10
    tokens_per_game: int = 4
    flat_feature_tokens: int = 4
    weather_tokens: int = 4  # one per hour in the game window (7 for as-of weather)
    # Per-token weather width: 22 for the legacy [4,22] snapshot; 99 for the
    # as-of tensor's [7,99] decision row (weather_asof.ASOF_CHANNELS).
    weather_dim: int = 22
    rating_steps: int = RATING_SEQ_STEPS  # temporal steps per side for rating history

    @property
    def sp_tokens(self) -> int:
        return self.sp_games * self.tokens_per_game

    @property
    def team_tokens(self) -> int:
        return self.team_games * self.tokens_per_game

    @property
    def rating_tokens(self) -> int:
        """Rating temporal tokens: K steps per side × 2 sides."""
        return 2 * self.rating_steps

    @property
    def total_context_tokens(self) -> int:
        """SP_home + SP_away + Team_home + Team_away + flat + weather + ratings."""
        return (
            2 * self.sp_tokens + 2 * self.team_tokens
            + self.flat_feature_tokens + self.weather_tokens
            + self.rating_tokens
        )


# ============================================================================
# PitchEncoder
# ============================================================================


class PitchEncoder(nn.Module):
    """Raw pitch features + player identity + event context -> d_model embedding.

    Shared encoder for both historical context compilation and live pitch encoding.
    Encodes three player identities (batter, pitcher, catcher) and event type to
    handle non-pitch events (substitutions, stolen bases, balks) as meaningful tokens.
    """

    # 8 event types: pitch, substitution, stolen_base, wild_pitch, balk,
    # intentional_walk, pickoff, other_action
    NUM_EVENT_TYPES = 8

    # Categorical vocabulary sizes for pitch-level features
    NUM_PITCH_TYPES = 20       # FF, SI, SL, CU, CH, FC, KC, FS, ST, SV, KN, EP, CS, SC, UN + padding
    NUM_BAT_SIDES = 4          # <PAD>, L, R, S (index 0 = unknown/padding)
    NUM_PITCH_HANDS = 3        # <PAD>, L, R (index 0 = unknown/padding)
    NUM_HALF_INNINGS = 2       # top, bottom
    NUM_HIT_TRAJECTORIES = 7   # none/padding + ground_ball, fly_ball, line_drive, popup, bunt_ground, bunt_popup
    NUM_HIT_HARDNESS = 4       # none/unknown, soft, medium, hard

    def __init__(
        self,
        continuous_dim: int = 52,
        d_model: int = 256,
        hash_buckets: int = 50_000,
        player_embed_dim: int = 16,
        event_embed_dim: int = 8,
        pitch_type_embed_dim: int = 8,
        bat_side_embed_dim: int = 4,
        pitch_hand_embed_dim: int = 4,
        half_inning_embed_dim: int = 4,
        hit_trajectory_embed_dim: int = 4,
        hit_hardness_embed_dim: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.batter_embed = nn.Embedding(hash_buckets, player_embed_dim, padding_idx=0)
        self.pitcher_embed = nn.Embedding(hash_buckets, player_embed_dim, padding_idx=0)
        self.catcher_embed = nn.Embedding(hash_buckets, player_embed_dim, padding_idx=0)
        self.event_type_embed = nn.Embedding(self.NUM_EVENT_TYPES, event_embed_dim)

        # Pitch-level categorical embeddings
        self.pitch_type_embed = nn.Embedding(
            self.NUM_PITCH_TYPES, pitch_type_embed_dim, padding_idx=0
        )
        self.bat_side_embed = nn.Embedding(self.NUM_BAT_SIDES, bat_side_embed_dim)
        self.pitch_hand_embed = nn.Embedding(self.NUM_PITCH_HANDS, pitch_hand_embed_dim)
        self.half_inning_embed = nn.Embedding(self.NUM_HALF_INNINGS, half_inning_embed_dim)
        self.hit_trajectory_embed = nn.Embedding(
            self.NUM_HIT_TRAJECTORIES, hit_trajectory_embed_dim, padding_idx=0
        )
        self.hit_hardness_embed = nn.Embedding(
            self.NUM_HIT_HARDNESS, hit_hardness_embed_dim, padding_idx=0
        )

        categorical_embed_dim = (
            pitch_type_embed_dim + bat_side_embed_dim + pitch_hand_embed_dim
            + half_inning_embed_dim + hit_trajectory_embed_dim + hit_hardness_embed_dim
        )
        self.continuous_dim = continuous_dim
        # Input: [continuous * obs_mask, obs_mask] = 2 * continuous_dim
        input_dim = continuous_dim * 2 + 3 * player_embed_dim + event_embed_dim + categorical_embed_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        continuous: torch.Tensor,
        batter_hash: torch.LongTensor,
        pitcher_hash: torch.LongTensor,
        catcher_hash: Optional[torch.LongTensor] = None,
        event_type: Optional[torch.LongTensor] = None,
        pitch_type_idx: Optional[torch.LongTensor] = None,
        bat_side_idx: Optional[torch.LongTensor] = None,
        pitch_hand_idx: Optional[torch.LongTensor] = None,
        half_inning_idx: Optional[torch.LongTensor] = None,
        hit_trajectory_idx: Optional[torch.LongTensor] = None,
        hit_hardness_idx: Optional[torch.LongTensor] = None,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            continuous: [B, S, continuous_dim] all continuous pitch features
            batter_hash: [B, S] blake2b hash bucket indices
            pitcher_hash: [B, S] blake2b hash bucket indices
            catcher_hash: [B, S] blake2b hash bucket indices (0 if unknown)
            event_type: [B, S] index into EVENT_TYPE_VOCAB (0=pitch if absent)
            pitch_type_idx: [B, S] index into PITCH_TYPE_VOCAB (0=padding if absent)
            bat_side_idx: [B, S] index into BAT_SIDE_VOCAB (0=L if absent)
            pitch_hand_idx: [B, S] index into PITCH_HAND_VOCAB (0=L if absent)
            half_inning_idx: [B, S] index into half-inning (0=top, 1=bottom)
            hit_trajectory_idx: [B, S] index into HIT_TRAJECTORY_VOCAB (0=none if absent)
            hit_hardness_idx: [B, S] index into HIT_HARDNESS_VOCAB (0=none if absent)

        Returns: [B, S, d_model]
        """
        B, S = continuous.shape[:2]
        device = continuous.device

        batter_emb = self.batter_embed(batter_hash)
        pitcher_emb = self.pitcher_embed(pitcher_hash)

        if catcher_hash is None:
            catcher_hash = torch.zeros(B, S, dtype=torch.long, device=device)
        catcher_emb = self.catcher_embed(catcher_hash)

        if event_type is None:
            event_type = torch.zeros(B, S, dtype=torch.long, device=device)
        event_emb = self.event_type_embed(event_type)

        # Pitch-level categorical embeddings (default to zeros if absent)
        if pitch_type_idx is None:
            pitch_type_idx = torch.zeros(B, S, dtype=torch.long, device=device)
        pitch_type_emb = self.pitch_type_embed(pitch_type_idx)

        if bat_side_idx is None:
            bat_side_idx = torch.zeros(B, S, dtype=torch.long, device=device)
        bat_side_emb = self.bat_side_embed(bat_side_idx)

        if pitch_hand_idx is None:
            pitch_hand_idx = torch.zeros(B, S, dtype=torch.long, device=device)
        pitch_hand_emb = self.pitch_hand_embed(pitch_hand_idx)

        if half_inning_idx is None:
            half_inning_idx = torch.zeros(B, S, dtype=torch.long, device=device)
        half_inning_emb = self.half_inning_embed(half_inning_idx)

        if hit_trajectory_idx is None:
            hit_trajectory_idx = torch.zeros(B, S, dtype=torch.long, device=device)
        hit_trajectory_emb = self.hit_trajectory_embed(hit_trajectory_idx)

        if hit_hardness_idx is None:
            hit_hardness_idx = torch.zeros(B, S, dtype=torch.long, device=device)
        hit_hardness_emb = self.hit_hardness_embed(hit_hardness_idx)

        if obs_mask is None:
            obs_mask = torch.ones_like(continuous)
        # [continuous * obs_mask, obs_mask]: model sees masked values + which are observed
        continuous_with_mask = torch.cat([continuous * obs_mask, obs_mask], dim=-1)

        x = torch.cat([
            continuous_with_mask, batter_emb, pitcher_emb, catcher_emb, event_emb,
            pitch_type_emb, bat_side_emb, pitch_hand_emb,
            half_inning_emb, hit_trajectory_emb, hit_hardness_emb,
        ], dim=-1)
        return self.proj(x)


# ============================================================================
# HierarchicalPositionalEncoding
# ============================================================================


class HierarchicalPositionalEncoding(nn.Module):
    """Learned positional encoding exploiting baseball's natural hierarchy.

    Additive composition: inning + at_bat + pitch_within_ab.
    Learned (not sinusoidal) because max values are small and spacing is irregular.
    """

    def __init__(
        self,
        d_model: int = 256,
        max_innings: int = 20,
        max_at_bats: int = 50,
        max_pitches_per_ab: int = 15,
    ):
        super().__init__()
        self.inning_emb = nn.Embedding(max_innings, d_model)
        self.ab_emb = nn.Embedding(max_at_bats, d_model)
        self.pitch_emb = nn.Embedding(max_pitches_per_ab, d_model)

    def forward(
        self,
        inning_idx: torch.LongTensor,
        ab_idx: torch.LongTensor,
        pitch_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            inning_idx: [B, S] inning number (0-indexed)
            ab_idx: [B, S] at-bat number within game (0-indexed)
            pitch_idx: [B, S] pitch number within at-bat (0-indexed)

        Returns: [B, S, d_model]
        """
        return self.inning_emb(inning_idx) + self.ab_emb(ab_idx) + self.pitch_emb(pitch_idx)


# ============================================================================
# PerceiverResampler
# ============================================================================


class PerceiverResampler(nn.Module):
    """Compresses variable-length pitch sequences into K fixed tokens via cross-attention.

    Learned query tokens cross-attend into the pitch sequence. Supports additive
    attention bias for game-index decay weighting (recent games weighted higher).
    """

    def __init__(
        self,
        d_model: int = 256,
        num_queries: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        self.layers = nn.ModuleList([
            _PerceiverCrossAttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.BoolTensor] = None,
        decay_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, d_model] encoded pitch sequence
            key_padding_mask: [B, S] True for padded positions
            decay_weight: [B] scalar decay weight per game (applied as additive bias)

        Returns: [B, K, d_model] compressed representation
        """
        B = x.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, d_model]

        for layer in self.layers:
            q = layer(q, x, key_padding_mask=key_padding_mask, decay_bias=decay_weight)

        return self.norm(q)


class _PerceiverCrossAttentionLayer(nn.Module):
    """Single cross-attention layer with pre-norm and FFN."""

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: Optional[torch.BoolTensor] = None,
        decay_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q: [B, K, d_model]
            kv: [B, S, d_model]
            key_padding_mask: [B, S] True where padded
            decay_bias: [B] additive scalar applied uniformly to all attention logits
                        (scales overall attention magnitude by game recency)
        """
        q_normed = self.norm_q(q)
        kv_normed = self.norm_kv(kv)

        # Compute attention with optional decay bias via attn_mask
        attn_mask = None
        if decay_bias is not None:
            K, S = q.size(1), kv.size(1)
            # log(weight) as additive bias preserves softmax proportionality
            log_bias = torch.log(decay_bias.clamp_min(1e-8))  # [B]
            attn_mask = log_bias[:, None, None].expand(-1, K, S)  # [B, K, S]
            num_heads = self.cross_attn.num_heads
            attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)  # [B*H, K, S]

        # Convert key_padding_mask to float to match attn_mask type.
        # All-padded sequences (new pitcher, no prior starts) would make softmax
        # receive all -inf → NaN output with undefined backward gradient.
        # Fix: unmask first position so softmax always has one valid target.
        kpm = None
        if key_padding_mask is not None:
            all_masked = key_padding_mask.all(dim=-1)  # [B]
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False
            kpm = torch.zeros_like(key_padding_mask, dtype=q.dtype)
            kpm = kpm.masked_fill(key_padding_mask, float("-inf"))

        attn_out, _ = self.cross_attn(
            q_normed, kv_normed, kv_normed,
            key_padding_mask=kpm,
            attn_mask=attn_mask,
        )
        q = q + attn_out

        # FFN with pre-norm
        q = q + self.ffn(self.norm_ff(q))
        return q


# ============================================================================
# ContextCompiler
# ============================================================================


class ContextCompiler(nn.Module):
    """Assembles historical context from multiple game histories into fixed-size token sequence.

    Pipeline per historical game:
        raw pitches -> PitchEncoder -> HierPosEnc -> PerceiverResampler -> K tokens
    Then concatenates all compressed games + flat feature projection.

    PitchEncoder and HierPosEnc are passed in (shared with live encoder for parameter efficiency).
    """

    def __init__(
        self,
        d_model: int = 256,
        context_config: Optional[ContextConfig] = None,
        pitch_encoder: Optional[PitchEncoder] = None,
        pos_encoding: Optional[HierarchicalPositionalEncoding] = None,
        flat_feature_dim: int = FLAT_FEATURE_DIM,
        rating_dim: int = 0,
        dropout: float = 0.1,
        lambda_intra: float = 0.015,
        lambda_inter: float = 0.30,
    ):
        super().__init__()
        self.config = context_config or ContextConfig()
        self.d_model = d_model
        self.lambda_intra = lambda_intra
        self.lambda_inter = lambda_inter
        self.rating_dim = rating_dim

        # Shared encoder (passed in from parent to share weights with live path)
        self.pitch_encoder = pitch_encoder or PitchEncoder(d_model=d_model, dropout=dropout)
        self.pos_encoding = pos_encoding or HierarchicalPositionalEncoding(d_model=d_model)
        self.resampler = PerceiverResampler(
            d_model=d_model,
            num_queries=self.config.tokens_per_game,
            num_layers=2,
            num_heads=4,
            dropout=dropout,
        )

        # Project flat matchup/venue/weather features to tokens
        self.flat_proj = nn.Sequential(
            nn.Linear(flat_feature_dim, d_model * self.config.flat_feature_tokens),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Weather temporal tokens: weather_dim features per hour → d_model per
        # token (22 legacy snapshot / 99 as-of channels)
        self.weather_proj = nn.Sequential(
            nn.Linear(self.config.weather_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.weather_hour_embed = nn.Embedding(self.config.weather_tokens, d_model)

        # Rating temporal encoder: [B, K, rating_dim] → [B, K, d_model] per side
        if rating_dim > 0:
            self.rating_proj = nn.Sequential(
                nn.Linear(rating_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.rating_step_embed = nn.Embedding(self.config.rating_steps, d_model)

        # Segment type embeddings:
        # sp_home(0), sp_away(1), team_home(2), team_away(3), flat(4), weather(5),
        # rating_home(6), rating_away(7)
        self.segment_embed = nn.Embedding(8, d_model)

    def compute_decay_weight(self, games_ago: torch.Tensor, seasons_crossed: torch.Tensor) -> torch.Tensor:
        """Game-index decay: w = exp(-lambda_intra * games_ago) * exp(-lambda_inter * seasons_crossed)."""
        return torch.exp(-self.lambda_intra * games_ago) * torch.exp(-self.lambda_inter * seasons_crossed)

    def _encode_game_batch(
        self,
        continuous: torch.Tensor,
        batter_hash: torch.LongTensor,
        pitcher_hash: torch.LongTensor,
        inning_idx: torch.LongTensor,
        ab_idx: torch.LongTensor,
        pitch_idx: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        decay_weights: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a batch of games and compress each to K tokens.

        Args:
            continuous: [B, num_games, max_pitches, continuous_dim]
            batter_hash: [B, num_games, max_pitches]
            pitcher_hash: [B, num_games, max_pitches]
            inning_idx: [B, num_games, max_pitches]
            ab_idx: [B, num_games, max_pitches]
            pitch_idx: [B, num_games, max_pitches]
            padding_mask: [B, num_games, max_pitches] True for padded
            decay_weights: [B, num_games] per-game decay scalar
            obs_mask: [B, num_games, max_pitches, continuous_dim] observability mask

        Returns: [B, num_games * K, d_model]
        """
        B, G, S = continuous.shape[:3]
        K = self.config.tokens_per_game

        # Flatten batch and game dims for efficient encoding
        cont_flat = continuous.reshape(B * G, S, -1)
        bat_flat = batter_hash.reshape(B * G, S)
        pit_flat = pitcher_hash.reshape(B * G, S)
        inn_flat = inning_idx.reshape(B * G, S)
        ab_flat = ab_idx.reshape(B * G, S)
        pch_flat = pitch_idx.reshape(B * G, S)
        pad_flat = padding_mask.reshape(B * G, S)
        mask_flat = obs_mask.reshape(B * G, S, -1) if obs_mask is not None else None

        # Encode pitches
        encoded = self.pitch_encoder(cont_flat, bat_flat, pit_flat, obs_mask=mask_flat)
        pos = self.pos_encoding(inn_flat, ab_flat, pch_flat)
        encoded = encoded + pos

        # Decay weights: [B, G] -> [B*G]
        decay_flat = decay_weights.reshape(B * G)

        # Compress each game to K tokens
        compressed = self.resampler(encoded, key_padding_mask=pad_flat, decay_weight=decay_flat)  # [B*G, K, d_model]

        # Reshape back to [B, G*K, d_model]
        return compressed.reshape(B, G * K, self.d_model)

    def forward(self, context_batch: dict) -> torch.Tensor:
        """
        Args:
            context_batch: Dictionary with keys for each context category.
                Each category has: continuous, batter_hash, pitcher_hash,
                inning_idx, ab_idx, pitch_idx, padding_mask, games_ago, seasons_crossed.
                Plus "flat_features": [B, flat_feature_dim].

        Returns: [B, total_context_tokens, d_model]
        """
        segments = []
        segment_ids = []

        # Process each context category: sp_home(0), sp_away(1), team_home(2), team_away(3)
        for seg_idx, key in enumerate(["sp_home", "sp_away", "team_home", "team_away"]):
            if key not in context_batch:
                continue

            cat = context_batch[key]
            decay = self.compute_decay_weight(cat["games_ago"], cat["seasons_crossed"])

            tokens = self._encode_game_batch(
                continuous=cat["continuous"],
                batter_hash=cat["batter_hash"],
                pitcher_hash=cat["pitcher_hash"],
                inning_idx=cat["inning_idx"],
                ab_idx=cat["ab_idx"],
                pitch_idx=cat["pitch_idx"],
                padding_mask=cat["padding_mask"],
                decay_weights=decay,
                obs_mask=cat.get("obs_mask"),
            )

            segments.append(tokens)
            num_tokens = tokens.size(1)
            segment_ids.append(torch.full((tokens.size(0), num_tokens), seg_idx, device=tokens.device))

        # Flat features -> tokens
        flat = context_batch["flat_features"]  # [B, flat_feature_dim]
        flat_tokens = self.flat_proj(flat).reshape(flat.size(0), self.config.flat_feature_tokens, self.d_model)
        segments.append(flat_tokens)
        segment_ids.append(torch.full(
            (flat.size(0), self.config.flat_feature_tokens), 4, device=flat.device
        ))

        # Weather temporal tokens -> 4 tokens with learned hour offsets
        if "weather_temporal" in context_batch:
            wx = context_batch["weather_temporal"]  # [B, 4, 22]
            wx_tokens = self.weather_proj(wx)  # [B, 4, d_model]
            hour_offsets = torch.arange(
                wx.size(1), device=wx.device
            ).unsqueeze(0).expand(wx.size(0), -1)
            wx_tokens = wx_tokens + self.weather_hour_embed(hour_offsets)
            segments.append(wx_tokens)
            segment_ids.append(torch.full(
                (wx.size(0), wx.size(1)), 5, device=wx.device
            ))
        else:
            # Graceful degradation: zero-filled weather tokens
            B_size = flat.size(0)
            wx_tokens = torch.zeros(
                B_size, self.config.weather_tokens, self.d_model, device=flat.device
            )
            segments.append(wx_tokens)
            segment_ids.append(torch.full(
                (B_size, self.config.weather_tokens), 5, device=flat.device
            ))

        # Rating temporal tokens: [B, K, rating_dim] per side → [B, K, d_model]
        if self.rating_dim > 0:
            B_size = flat.size(0)
            K = self.config.rating_steps
            step_indices = torch.arange(K, device=flat.device).unsqueeze(0).expand(B_size, -1)
            # Game-index decay: step 0 = oldest (highest decay), step K-1 = most recent
            decay = torch.exp(-self.lambda_intra * (K - 1 - step_indices).float())

            for side_key, seg_id in [("rating_home", 6), ("rating_away", 7)]:
                if side_key in context_batch:
                    rating_seq = context_batch[side_key]  # [B, K, rating_dim]
                else:
                    rating_seq = torch.zeros(B_size, K, self.rating_dim, device=flat.device)

                rating_tokens = self.rating_proj(rating_seq)  # [B, K, d_model]
                rating_tokens = rating_tokens + self.rating_step_embed(step_indices)
                # Apply temporal decay weighting
                rating_tokens = rating_tokens * decay.unsqueeze(-1)

                segments.append(rating_tokens)
                segment_ids.append(torch.full(
                    (B_size, K), seg_id, device=flat.device
                ))

        # Concatenate all context tokens
        context_tokens = torch.cat(segments, dim=1)  # [B, C, d_model]
        seg_id_tensor = torch.cat(segment_ids, dim=1).long()  # [B, C]

        # Add segment embeddings
        context_tokens = context_tokens + self.segment_embed(seg_id_tensor)

        return context_tokens


# ============================================================================
# GameTransformerBackbone
# ============================================================================


class GameTransformerBackbone(nn.Module):
    """6-layer Transformer with prefix-LM attention pattern.

    Context tokens (prefix): full bidirectional self-attention.
    Live tokens (suffix): causal attention among themselves, full attention to context.
    This enables KV caching — context is encoded once and reused for each new pitch.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.layers = nn.ModuleList([
            _TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def _build_prefix_lm_mask(
        self,
        num_context: int,
        num_live: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build attention mask implementing prefix-LM pattern.

        Returns float mask where -inf blocks attention and 0.0 allows it.
        Shape: [num_context + num_live, num_context + num_live]
        """
        total = num_context + num_live

        # Start with all blocked
        mask = torch.full((total, total), float("-inf"), device=device)

        # Context tokens attend to all context tokens (bidirectional)
        mask[:num_context, :num_context] = 0.0

        # Live tokens attend to all context tokens
        if num_live > 0:
            mask[num_context:, :num_context] = 0.0

            # Live tokens: causal among themselves
            live_causal = torch.triu(
                torch.full((num_live, num_live), float("-inf"), device=device),
                diagonal=1,
            )
            mask[num_context:, num_context:] = live_causal

        # Context tokens CANNOT attend to live tokens (already -inf from initialization)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        num_context: int,
        kv_cache: Optional[list[dict[str, torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, Optional[list[dict[str, torch.Tensor]]]]:
        """
        Args:
            x: [B, num_context + num_live, d_model]
            num_context: number of prefix context tokens
            kv_cache: optional cached KV from prior forward pass (for live inference)

        Returns:
            output: [B, num_context + num_live, d_model]
            new_kv_cache: updated KV cache (list of dicts per layer)
        """
        total_len = x.size(1)
        num_live = total_len - num_context

        attn_mask = self._build_prefix_lm_mask(num_context, num_live, x.device)

        new_cache: list[dict[str, torch.Tensor]] = []

        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, cache_out = layer(x, attn_mask=attn_mask, kv_cache=layer_cache)
            new_cache.append(cache_out)

        x = self.final_norm(x)
        return x, new_cache


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer block with KV caching support."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Pre-norm self-attention
        x_normed = self.norm1(x)

        # KV caching: prepend cached keys/values for incremental decoding
        if kv_cache is not None:
            cached_k = kv_cache["k"]
            cached_v = kv_cache["v"]
            # Only the new token's query attends to all prior KVs + itself
            kv_input = torch.cat([cached_k, x_normed], dim=1)
            new_k = kv_input
            new_v = torch.cat([cached_v, x_normed], dim=1)
            # Full-sequence attn_mask shape [T,T] doesn't match key dim
            # [cached_len+T] in incremental decode. Under prefix-LM, new live
            # tokens attend to all prior context unconditionally — None is correct.
            attn_out, _ = self.attn(x_normed, new_k, new_v, attn_mask=None)
        else:
            attn_out, _ = self.attn(x_normed, x_normed, x_normed, attn_mask=attn_mask)
            new_k = x_normed
            new_v = x_normed

        x = x + attn_out

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))

        cache_out = {"k": new_k.detach(), "v": new_v.detach()}
        return x, cache_out


# ============================================================================
# PlayerQueryHead
# ============================================================================


class PlayerQueryHead(nn.Module):
    """Per-player prop prediction via PA-outcome multinomial.

    Predicts a per-PA outcome distribution [K, non_K_out, BB_HBP, 1B, 2B, 3B, HR]
    plus expected PA count. All prop markets (hits, HR, TB) are derived analytically,
    guaranteeing logical consistency (e.g., P(1+HR) <= P(1+H) always).

    Pitcher K and SB are separate heads because they depend on factors outside the
    batter's PA outcome distribution (innings pitched, baserunning opportunities).
    H+R+RBI is NegBin because R and RBI are context-dependent (runners on base).
    """

    # PA outcome classes and their stat contributions
    PA_CLASSES = 7  # [K, non_K_out, BB_HBP, 1B, 2B, 3B, HR]
    # Hits per PA outcome type
    HITS_PER_OUTCOME = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)
    # Total bases per PA outcome type
    TB_PER_OUTCOME = torch.tensor([0, 0, 0, 1, 2, 3, 4], dtype=torch.float32)

    def __init__(
        self,
        d_model: int = 256,
        hash_buckets: int = 50_000,
        player_embed_dim: int = 16,
        player_context_tokens: int = 2,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.player_embed_dim = player_embed_dim
        self.player_context_tokens = player_context_tokens
        self.d_model = d_model

        self.player_embed = nn.Embedding(hash_buckets, player_embed_dim, padding_idx=0)

        input_dim = player_embed_dim + player_context_tokens * d_model + d_model
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

        # PA-outcome multinomial: 7-class logits per PA
        self.head_pa_outcome = nn.Linear(output_dim, self.PA_CLASSES)
        # Expected PA count (log-space, softplus decoded): batters typically get 3-5 PA
        self.head_pa_count = nn.Linear(output_dim, 1)

        # Pitcher K: NegBin (depends on innings pitched, not derivable from batter PA)
        self.head_pitcher_k = nn.Linear(output_dim, 2)
        # H+R+RBI: NegBin (R, RBI depend on baserunner context beyond hit type)
        self.head_h_r_rbi = nn.Linear(output_dim, 2)
        # SB: Bernoulli (baserunning event, independent of PA outcome)
        self.head_stolen_bases = nn.Linear(output_dim, 1)

    def _negbin_params(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert raw linear output to valid NegBin parameters."""
        mu = F.softplus(raw[..., 0]).clamp_min(0.01)
        alpha = F.softplus(raw[..., 1]).clamp_min(0.1)
        return mu, alpha

    def _derive_prop_probs(
        self,
        pa_probs: torch.Tensor,
        pa_count: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Analytically derive prop market probabilities from PA-outcome distribution.

        Given per-PA outcome probs and expected PA count, computes:
        - P(1+ hits), P(2+ hits), P(0 hits) via categorical(5)
        - P(1+ HR)
        - E[TB], P(1+ TB), P(2+ TB)

        All derived quantities are differentiable w.r.t. pa_probs and pa_count.
        """
        device = pa_probs.device
        hits_per = self.HITS_PER_OUTCOME.to(device)
        tb_per = self.TB_PER_OUTCOME.to(device)

        # P(no hit in single PA) = P(K) + P(non_K_out) + P(BB/HBP)
        p_no_hit_single = pa_probs[:, 0] + pa_probs[:, 1] + pa_probs[:, 2]
        # P(no HR in single PA) = 1 - P(HR)
        p_no_hr_single = 1.0 - pa_probs[:, 6]

        # Round PA count to integer for probability computation
        # Use soft approximation: weighted sum over possible PA counts (3, 4, 5, 6)
        # pa_count is continuous; we compute a weighted mixture
        pa_floor = pa_count.floor().long().clamp(1, 7)
        pa_frac = pa_count - pa_count.floor()

        # P(0 hits in game) = P(no hit)^n_PA
        p_0_hits_lo = p_no_hit_single.pow(pa_floor.float())
        p_0_hits_hi = p_no_hit_single.pow((pa_floor + 1).float())
        p_0_hits = (1 - pa_frac) * p_0_hits_lo + pa_frac * p_0_hits_hi

        # P(exactly 1 hit) = n * P(hit) * P(no_hit)^(n-1)
        p_hit_single = 1.0 - p_no_hit_single
        n_lo = pa_floor.float()
        n_hi = n_lo + 1
        p_1_hit_lo = n_lo * p_hit_single * p_no_hit_single.pow((pa_floor - 1).float().clamp_min(0))
        p_1_hit_hi = n_hi * p_hit_single * p_no_hit_single.pow(pa_floor.float())
        p_1_hit = (1 - pa_frac) * p_1_hit_lo + pa_frac * p_1_hit_hi

        # Hits categorical: [P(0), P(1), P(2), P(3), P(4+)]
        p_2plus_hits = (1.0 - p_0_hits - p_1_hit).clamp_min(0)

        # Binomial for 2 and 3 hits (approximate partition of 2+)
        p_2_hit_lo = 0.5 * n_lo * (n_lo - 1) * p_hit_single.pow(2) * p_no_hit_single.pow((pa_floor - 2).float().clamp_min(0))
        p_2_hit_hi = 0.5 * n_hi * n_lo * p_hit_single.pow(2) * p_no_hit_single.pow((pa_floor - 1).float().clamp_min(0))
        p_2_hit = (1 - pa_frac) * p_2_hit_lo + pa_frac * p_2_hit_hi
        p_2_hit = torch.min(p_2_hit.clamp_min(0), p_2plus_hits)

        p_3_hit = (p_2plus_hits - p_2_hit).clamp_min(0) * 0.7  # most of remainder is 3-hit
        p_4plus_hit = (p_2plus_hits - p_2_hit - p_3_hit).clamp_min(0)

        hits_categorical = torch.stack([p_0_hits, p_1_hit, p_2_hit, p_3_hit, p_4plus_hit], dim=-1)
        # Re-normalize to ensure valid distribution (numerical safety)
        hits_categorical = hits_categorical / hits_categorical.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # P(1+ HR in game) = 1 - P(no HR)^n_PA
        p_0_hr_lo = p_no_hr_single.pow(pa_floor.float())
        p_0_hr_hi = p_no_hr_single.pow((pa_floor + 1).float())
        p_0_hr = (1 - pa_frac) * p_0_hr_lo + pa_frac * p_0_hr_hi
        p_1plus_hr = 1.0 - p_0_hr

        # E[TB per game] = n_PA * sum(P(outcome_i) * TB_i)
        e_tb_per_pa = (pa_probs * tb_per.unsqueeze(0)).sum(dim=-1)
        e_tb = pa_count * e_tb_per_pa

        return {
            "hits_categorical": hits_categorical,  # [B*P, 5]
            "hr_prob": p_1plus_hr,  # [B*P]
            "e_tb": e_tb,  # [B*P]
            "pa_probs": pa_probs,  # [B*P, 7] raw multinomial (for loss)
            "pa_count": pa_count,  # [B*P]
        }

    def forward(
        self,
        player_hashes: torch.LongTensor,
        player_context: Optional[torch.Tensor],
        game_repr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            player_hashes: [B, P] hash bucket indices for P players
            player_context: [B, P, player_context_tokens * d_model] or None
            game_repr: [B, d_model] game-level representation

        Returns: Dict with per-player predictions.
        """
        B, P = player_hashes.shape

        p_emb = self.player_embed(player_hashes)

        if player_context is not None:
            p_ctx = player_context
        else:
            p_ctx = torch.zeros(
                B, P, self.player_context_tokens * self.d_model,
                device=player_hashes.device, dtype=game_repr.dtype,
            )

        g_repr = game_repr.unsqueeze(1).expand(-1, P, -1)
        x = torch.cat([p_emb, p_ctx, g_repr], dim=-1)
        x = x.reshape(B * P, -1)

        h = self.mlp(x)

        # PA-outcome multinomial
        pa_outcome_logits = self.head_pa_outcome(h)  # [B*P, 7]
        pa_probs = F.softmax(pa_outcome_logits, dim=-1)
        # PA count (typical range 3-5, softplus ensures positivity)
        pa_count = F.softplus(self.head_pa_count(h).squeeze(-1)) + 2.5  # floor ~2.5 PA

        # Derive hit/HR/TB props from the multinomial
        derived = self._derive_prop_probs(pa_probs, pa_count)

        # Pitcher K (NegBin)
        k_raw = self.head_pitcher_k(h)
        k_mu, k_alpha = self._negbin_params(k_raw)

        # H+R+RBI (NegBin)
        hrbi_raw = self.head_h_r_rbi(h)
        hrbi_mu, hrbi_alpha = self._negbin_params(hrbi_raw)

        # SB (Bernoulli)
        sb_logit = self.head_stolen_bases(h).squeeze(-1)

        return {
            # PA-outcome multinomial (raw logits for CE loss)
            "pa_outcome_logits": pa_outcome_logits.reshape(B, P, self.PA_CLASSES),
            "pa_probs": derived["pa_probs"].reshape(B, P, self.PA_CLASSES),
            "pa_count": derived["pa_count"].reshape(B, P),
            # Derived props (guaranteed consistent)
            "hits_categorical": derived["hits_categorical"].reshape(B, P, 5),
            "hr_prob": derived["hr_prob"].reshape(B, P),
            "e_tb": derived["e_tb"].reshape(B, P),
            # Independent heads
            "pitcher_k_mu": k_mu.reshape(B, P),
            "pitcher_k_alpha": k_alpha.reshape(B, P),
            "h_r_rbi_mu": hrbi_mu.reshape(B, P),
            "h_r_rbi_alpha": hrbi_alpha.reshape(B, P),
            "stolen_bases_logit": sb_logit.reshape(B, P),
        }


# ============================================================================
# GameTransformer (main model)
# ============================================================================


class GameTransformer(nn.Module):
    """Unified pregame + live MLB market pricing model.

    Composes: ContextCompiler -> PitchEncoder -> Backbone -> Team heads + Player heads.
    Pregame mode (T=0): prices from context tokens only.
    Live mode (T>0): prices from last live token (autoregressive).
    """

    def __init__(
        self,
        d_model: int = 256,
        continuous_dim: int = 52,
        hash_buckets: int = 50_000,
        player_embed_dim: int = 16,
        flat_feature_dim: int = FLAT_FEATURE_DIM,
        rating_dim: int = 0,
        context_config: Optional[ContextConfig] = None,
        num_backbone_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_players: int = 20,
        player_context_tokens: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.context_config = context_config or ContextConfig()

        # Shared pitch encoder + positional encoding (same weights for context and live)
        self.pitch_encoder = PitchEncoder(
            continuous_dim=continuous_dim,
            d_model=d_model,
            hash_buckets=hash_buckets,
            player_embed_dim=player_embed_dim,
            event_embed_dim=8,
            dropout=dropout,
        )
        self.pos_encoding = HierarchicalPositionalEncoding(d_model=d_model)

        # Context compilation (uses shared encoder)
        self.context_compiler = ContextCompiler(
            d_model=d_model,
            context_config=self.context_config,
            pitch_encoder=self.pitch_encoder,
            pos_encoding=self.pos_encoding,
            flat_feature_dim=flat_feature_dim,
            rating_dim=rating_dim,
            dropout=dropout,
        )

        # Backbone
        self.backbone = GameTransformerBackbone(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_backbone_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Team-level output heads
        self.head_negbin_home = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        self.head_negbin_away = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        self.head_home_win = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.head_yrfi = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.head_extra_innings = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Player-level output head
        self.player_head = PlayerQueryHead(
            d_model=d_model,
            hash_buckets=hash_buckets,
            player_embed_dim=player_embed_dim,
            player_context_tokens=player_context_tokens,
            dropout=dropout,
        )

    def _team_readout(
        self,
        backbone_out: torch.Tensor,
        num_context: int,
        live_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract team-level representations for output heads.

        Returns (game_repr, game_repr_no_rating):
            game_repr: full context pool (for game heads that benefit from ratings)
            game_repr_no_rating: excludes rating tokens (for player heads where
                team-strength ratings are noise/harmful)

        Pregame (0 real pitches): mean-pool context tokens.
        Live: last live token (rating info already mixed via attention).

        The pregame/live choice is PER ROW, keyed on `live_lengths`. It used to be keyed on
        `num_live = total_len - num_context`, a tensor shape shared by the whole batch, so a
        prefix_length=0 row read `backbone_out[:, -1, :]` — pure prefix padding — whenever any
        batchmate had pitches. At a 6.55% pregame rate a shuffled batch of 64 is never all
        pregame, so the mean-pool branch got essentially zero gradient over a whole run while
        every real pregame serving request (one game, batch of 1) landed on exactly that
        untrained branch. Pinned by
        tests/test_pregame_readout_invariance.py::test_pregame_row_prices_identically_alone_and_beside_a_live_row.

        Rows WITH pitches are unchanged bit-for-bit: prefixes are left-padded, so position -1 is
        their last real pitch. The context pool is safe to mix in per row because the prefix-LM
        mask forbids context->live attention, making context token outputs independent of
        whether live tokens are present at all.
        """
        total_len = backbone_out.size(1)
        num_live = total_len - num_context
        num_rating = self.context_config.rating_tokens  # 20 (10 steps × 2 sides)

        ctx_pool = backbone_out[:, :num_context, :].mean(dim=1)
        ctx_pool_no_rating = backbone_out[:, :num_context - num_rating, :].mean(dim=1)

        if num_live == 0:
            return ctx_pool, ctx_pool_no_rating

        last_live = backbone_out[:, -1, :]
        if live_lengths is None:
            # A caller that attaches live tensors without saying how many are real is asserting
            # every row is live. Preserves the legacy path for hand-built batches (smoke tests,
            # kv-cache decode) rather than silently guessing from padding.
            return last_live, last_live

        is_pregame = (live_lengths.reshape(-1) == 0).unsqueeze(-1)  # [B, 1]
        game_repr = torch.where(is_pregame, ctx_pool, last_live)
        game_repr_no_rating = torch.where(is_pregame, ctx_pool_no_rating, last_live)
        return game_repr, game_repr_no_rating

    def _decode_negbin(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Softplus both params with floors for numerical stability."""
        mu = F.softplus(raw[:, 0]).clamp_min(0.01)
        alpha = F.softplus(raw[:, 1]).clamp_min(0.1)
        return mu, alpha

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """
        Args:
            batch: Dictionary with keys:
                Context data (for ContextCompiler):
                    "context": dict passed to ContextCompiler.forward()

                Live pitch data (may be absent for pregame):
                    "live_continuous": [B, T, 50] or None
                    "live_batter_hash": [B, T]
                    "live_pitcher_hash": [B, T]
                    "live_inning_idx": [B, T]
                    "live_ab_idx": [B, T]
                    "live_pitch_idx": [B, T]

                Player data:
                    "player_hashes": [B, P] up to 20 players
                    "player_context": [B, P, player_context_tokens * d_model] or None

        Returns: Dict with team-level and player-level predictions.
        """
        # Step 1: Compile context
        context_tokens = self.context_compiler(batch["context"])  # [B, C, d_model]
        num_context = context_tokens.size(1)

        # Step 2: Encode live pitches (if any)
        if "live_continuous" in batch and batch["live_continuous"] is not None:
            live_encoded = self.pitch_encoder(
                batch["live_continuous"],
                batch["live_batter_hash"],
                batch["live_pitcher_hash"],
                catcher_hash=batch.get("live_catcher_hash"),
                event_type=batch.get("live_event_type"),
                pitch_type_idx=batch.get("live_pitch_type_idx"),
                bat_side_idx=batch.get("live_bat_side_idx"),
                pitch_hand_idx=batch.get("live_pitch_hand_idx"),
                half_inning_idx=batch.get("live_half_inning_idx"),
                hit_trajectory_idx=batch.get("live_hit_trajectory_idx"),
                hit_hardness_idx=batch.get("live_hit_hardness_idx"),
                obs_mask=batch.get("live_obs_mask"),
            )
            live_pos = self.pos_encoding(
                batch["live_inning_idx"],
                batch["live_ab_idx"],
                batch["live_pitch_idx"],
            )
            live_tokens = live_encoded + live_pos  # [B, T, d_model]

            # Step 3: Concatenate [context | live]
            x = torch.cat([context_tokens, live_tokens], dim=1)
        else:
            x = context_tokens

        # Step 4: Backbone with prefix-LM mask
        backbone_out, kv_cache = self.backbone(x, num_context=num_context)

        # Step 5: Team readout (dual — full for game heads, no-rating for player heads)
        game_repr, game_repr_no_rating = self._team_readout(
            backbone_out, num_context, live_lengths=batch.get("live_lengths")
        )

        # Step 6: Team heads (use full game_repr including ratings)
        home_raw = self.head_negbin_home(game_repr)
        away_raw = self.head_negbin_away(game_repr)
        mu_home, alpha_home = self._decode_negbin(home_raw)
        mu_away, alpha_away = self._decode_negbin(away_raw)

        home_win_logit = self.head_home_win(game_repr).squeeze(-1)
        yrfi_logit = self.head_yrfi(game_repr).squeeze(-1)
        extra_innings_logit = self.head_extra_innings(game_repr).squeeze(-1)

        outputs = {
            "mu_home": mu_home,
            "alpha_home": alpha_home,
            "mu_away": mu_away,
            "alpha_away": alpha_away,
            "home_win_logit": home_win_logit,
            "yrfi_logit": yrfi_logit,
            "extra_innings_logit": extra_innings_logit,
        }

        # Step 7: Player heads (use game_repr_no_rating — ratings excluded)
        if "player_hashes" in batch:
            player_out = self.player_head(
                player_hashes=batch["player_hashes"],
                player_context=batch.get("player_context"),
                game_repr=game_repr_no_rating,
            )
            outputs.update(player_out)

        return outputs


# ============================================================================
# Loss Functions
# ============================================================================


class GameTransformerLoss(nn.Module):
    """Multi-task loss with explicit team/player loss weighting.

    Architecture:
        Team block (weight 1.0):
            - NegBin NLL for home/away remaining runs
            - BCE for home_win, YRFI, extra_innings
            - Consistency: NegBin-derived win prob vs direct head
        Player block (weight 0.2 total):
            - CE for hits categorical (from PA-outcome multinomial)
            - Focal BCE for HR (derived from multinomial, rare event ~3%)
            - NegBin for pitcher K
            - NegBin for H+R+RBI
            - Focal BCE for SB (rare event ~2-5%)

    Player block capped at 20% of team loss magnitude to prevent noisy per-player
    gradients from destabilizing the well-calibrated game-level distribution.
    Focal loss (Lin et al. 2017, gamma=2.0) on rare binary events prevents the
    model from collapsing to predicting the base rate.
    """

    PLAYER_LOSS_WEIGHT = 0.2
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA_POS = 0.75

    def __init__(self, num_team_tasks: int = 6, num_player_tasks: int = 5):
        super().__init__()
        self.num_team_tasks = num_team_tasks
        self.num_player_tasks = num_player_tasks
        total = num_team_tasks + num_player_tasks
        self.log_weights = nn.Parameter(torch.zeros(total))

        self.team_task_names = [
            "negbin_home", "negbin_away",
            "bce_home_win", "bce_yrfi", "bce_extra_innings",
            "consistency",
        ]
        self.player_task_names = [
            "ce_hits", "focal_hr", "negbin_pitcher_k",
            "negbin_hrbi", "focal_sb",
        ]

    def _focal_bce(
        self, logit: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Focal loss for rare binary events (Lin et al. 2017).

        gamma=2.0 down-weights easy negatives; alpha=0.75 up-weights positives.
        """
        p = torch.sigmoid(logit)
        ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        # p_t = p if target=1, else 1-p
        p_t = p * target + (1 - p) * (1 - target)
        focal_weight = (1 - p_t).pow(self.FOCAL_GAMMA)
        alpha_t = self.FOCAL_ALPHA_POS * target + (1 - self.FOCAL_ALPHA_POS) * (1 - target)
        return alpha_t * focal_weight * ce

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        live_inning: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            predictions: model output dict
            targets: ground truth dict with keys:
                home_runs, away_runs, home_win, yrfi, extra_innings,
                player_hits (int, 0-4), player_hr (binary), player_k (count),
                player_hrbi (count), player_sb (binary),
                player_mask (1 where player present, 0 for padding)
            live_inning: [B] current inning (for YRFI masking). None = pregame.

        Returns:
            total_loss: scalar
            task_losses: dict of per-task losses for logging
        """
        weights = torch.exp(self.log_weights)
        losses = {}
        device = predictions["mu_home"].device

        # === Team block ===

        losses["negbin_home"] = negbin_nll(
            targets["home_runs_remaining"], predictions["mu_home"], predictions["alpha_home"]
        ).mean()
        losses["negbin_away"] = negbin_nll(
            targets["away_runs_remaining"], predictions["mu_away"], predictions["alpha_away"]
        ).mean()

        losses["bce_home_win"] = F.binary_cross_entropy_with_logits(
            predictions["home_win_logit"], targets["home_win"].float()
        )

        # YRFI: mask when live prefix extends past 1st inning
        yrfi_logit = predictions["yrfi_logit"]
        yrfi_target = targets["yrfi"].float()
        if live_inning is not None:
            yrfi_mask = (live_inning <= 1).float()
            if yrfi_mask.sum() > 0:
                yrfi_loss = F.binary_cross_entropy_with_logits(
                    yrfi_logit, yrfi_target, reduction="none"
                )
                losses["bce_yrfi"] = (yrfi_loss * yrfi_mask).sum() / yrfi_mask.sum()
            else:
                losses["bce_yrfi"] = torch.tensor(0.0, device=device)
        else:
            losses["bce_yrfi"] = F.binary_cross_entropy_with_logits(yrfi_logit, yrfi_target)

        losses["bce_extra_innings"] = F.binary_cross_entropy_with_logits(
            predictions["extra_innings_logit"], targets["extra_innings"].float()
        )

        # Consistency: NegBin-derived win prob vs direct head
        negbin_win_logit = predictions["mu_home"] - predictions["mu_away"]
        direct_win_prob = torch.sigmoid(predictions["home_win_logit"])
        negbin_win_prob = torch.sigmoid(negbin_win_logit)
        losses["consistency"] = F.mse_loss(direct_win_prob, negbin_win_prob.detach())

        # === Player block ===

        player_mask = targets.get("player_mask")
        if player_mask is not None and player_mask.sum() > 0:
            pm = player_mask.float()
            n_valid = pm.sum().clamp_min(1.0)

            # Hits: cross-entropy on categorical(5) derived from PA-outcome multinomial
            if "player_hits" in targets and "hits_categorical" in predictions:
                # Target is integer hit count [0, 1, 2, 3, 4+] -> class index
                hit_target = targets["player_hits"].long().clamp(0, 4)  # [B, P]
                hit_probs = predictions["hits_categorical"]  # [B, P, 5]
                # CE loss per player
                log_probs = torch.log(hit_probs.clamp_min(1e-8))
                B_h, P_h = hit_target.shape
                ce = -log_probs[
                    torch.arange(B_h, device=device).unsqueeze(1).expand_as(hit_target),
                    torch.arange(P_h, device=device).unsqueeze(0).expand_as(hit_target),
                    hit_target,
                ]
                losses["ce_hits"] = (ce * pm).sum() / n_valid

            # HR: focal BCE on derived P(1+ HR)
            if "player_hr" in targets and "hr_prob" in predictions:
                # The head is P(1+ HR) (see `p_1plus_hr`), but the target is a
                # COUNT (0-4 in the real splits, 0.80% of slots > 1). BCE with a
                # target outside [0,1] is not a probability loss: for t=2 the
                # gradient -t/p - (1-t)/(1-p) is negative for EVERY p, so those
                # slots push the probability to 1 without bound, and at p=0.3
                # they carry 7x the loss of a 1-HR slot.
                hr_target = (targets["player_hr"] > 0).float()
                # Convert prob to logit for focal loss
                hr_prob_clamped = predictions["hr_prob"].clamp(1e-6, 1 - 1e-6)
                hr_logit = torch.log(hr_prob_clamped / (1 - hr_prob_clamped))
                focal = self._focal_bce(hr_logit, hr_target)
                losses["focal_hr"] = (focal * pm).sum() / n_valid

            # Pitcher K: NegBin (dataset key is player_so)
            if "player_so" in targets and "pitcher_k_mu" in predictions:
                nll = negbin_nll(
                    targets["player_so"], predictions["pitcher_k_mu"], predictions["pitcher_k_alpha"]
                )
                losses["negbin_pitcher_k"] = (nll * pm).sum() / n_valid

            # H+R+RBI: NegBin
            if "player_hrbi" in targets and "h_r_rbi_mu" in predictions:
                nll = negbin_nll(
                    targets["player_hrbi"], predictions["h_r_rbi_mu"], predictions["h_r_rbi_alpha"]
                )
                losses["negbin_hrbi"] = (nll * pm).sum() / n_valid

            # SB: focal BCE
            if "player_sb" in targets and "stolen_bases_logit" in predictions:
                # Same count-vs-event mismatch as HR: player_sb reaches 3-4, and
                # a target of 3 costs 24x a target of 1 while still pushing p up.
                focal = self._focal_bce(
                    predictions["stolen_bases_logit"], (targets["player_sb"] > 0).float()
                )
                losses["focal_sb"] = (focal * pm).sum() / n_valid
        else:
            for key in self.player_task_names:
                losses[key] = torch.tensor(0.0, device=device)

        # === Weighted combination with team/player split ===

        team_loss = torch.tensor(0.0, device=device)
        for i, name in enumerate(self.team_task_names):
            if name in losses:
                team_loss = team_loss + weights[i] * losses[name]

        player_loss = torch.tensor(0.0, device=device)
        for i, name in enumerate(self.player_task_names):
            if name in losses:
                player_loss = player_loss + weights[self.num_team_tasks + i] * losses[name]

        total_loss = team_loss + self.PLAYER_LOSS_WEIGHT * player_loss

        # Regularization: prevent weights from collapsing
        total_loss = total_loss + 0.01 * torch.sum(torch.exp(-self.log_weights))

        return total_loss, {k: v.detach() for k, v in losses.items()}


# ============================================================================
# Smoke test
# ============================================================================


def _test_forward_pass():
    """Verify shapes, no NaN, parameter count in expected range."""
    B, T, P = 4, 50, 10  # batch, live pitches, players
    d_model = 256
    rating_dim = 59

    config = ContextConfig(sp_games=2, team_games=3, tokens_per_game=4, flat_feature_tokens=4)
    model = GameTransformer(
        d_model=d_model,
        rating_dim=rating_dim,
        context_config=config,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    max_pitches_ctx = 80
    sp_games = config.sp_games
    team_games = config.team_games

    def mock_game_data(num_games: int) -> dict:
        return {
            "continuous": torch.randn(B, num_games, max_pitches_ctx, 52),
            "batter_hash": torch.randint(1, 50000, (B, num_games, max_pitches_ctx)),
            "pitcher_hash": torch.randint(1, 50000, (B, num_games, max_pitches_ctx)),
            "inning_idx": torch.randint(0, 9, (B, num_games, max_pitches_ctx)),
            "ab_idx": torch.randint(0, 40, (B, num_games, max_pitches_ctx)),
            "pitch_idx": torch.randint(0, 10, (B, num_games, max_pitches_ctx)),
            "padding_mask": torch.zeros(B, num_games, max_pitches_ctx, dtype=torch.bool),
            "games_ago": torch.arange(num_games).float().unsqueeze(0).expand(B, -1),
            "seasons_crossed": torch.zeros(B, num_games),
        }

    context_batch = {
        "sp_home": mock_game_data(sp_games),
        "sp_away": mock_game_data(sp_games),
        "team_home": mock_game_data(team_games),
        "team_away": mock_game_data(team_games),
        "flat_features": torch.randn(B, FLAT_FEATURE_DIM),
        "rating_home": torch.randn(B, config.rating_steps, rating_dim),
        "rating_away": torch.randn(B, config.rating_steps, rating_dim),
    }

    batch = {
        "context": context_batch,
        "live_continuous": torch.randn(B, T, 52),
        "live_batter_hash": torch.randint(1, 50000, (B, T)),
        "live_pitcher_hash": torch.randint(1, 50000, (B, T)),
        "live_catcher_hash": torch.randint(1, 50000, (B, T)),
        "live_event_type": torch.randint(0, 8, (B, T)),
        "live_pitch_type_idx": torch.randint(0, 20, (B, T)),
        "live_bat_side_idx": torch.randint(0, 3, (B, T)),
        "live_pitch_hand_idx": torch.randint(0, 2, (B, T)),
        "live_half_inning_idx": torch.randint(0, 2, (B, T)),
        "live_hit_trajectory_idx": torch.randint(0, 7, (B, T)),
        "live_hit_hardness_idx": torch.randint(0, 4, (B, T)),
        "live_inning_idx": torch.randint(0, 9, (B, T)),
        "live_ab_idx": torch.randint(0, 40, (B, T)),
        "live_pitch_idx": torch.randint(0, 10, (B, T)),
        "player_hashes": torch.randint(1, 50000, (B, P)),
        "player_context": torch.randn(B, P, 2 * d_model),
    }

    model.eval()
    with torch.no_grad():
        out = model(batch)

    # Verify team outputs
    assert out["mu_home"].shape == (B,), f"Expected ({B},), got {out['mu_home'].shape}"
    assert out["alpha_home"].shape == (B,)
    assert out["mu_away"].shape == (B,)
    assert out["alpha_away"].shape == (B,)
    assert out["home_win_logit"].shape == (B,)
    assert out["yrfi_logit"].shape == (B,)
    assert out["extra_innings_logit"].shape == (B,)

    # Verify player outputs from PA-outcome multinomial
    assert out["pa_outcome_logits"].shape == (B, P, PlayerQueryHead.PA_CLASSES)
    assert out["hits_categorical"].shape == (B, P, 5)
    assert out["hr_prob"].shape == (B, P)
    assert out["pa_count"].shape == (B, P)
    assert out["e_tb"].shape == (B, P)
    assert out["pitcher_k_mu"].shape == (B, P)
    assert out["pitcher_k_alpha"].shape == (B, P)
    assert out["h_r_rbi_mu"].shape == (B, P)
    assert out["h_r_rbi_alpha"].shape == (B, P)
    assert out["stolen_bases_logit"].shape == (B, P)

    # Verify no NaN
    for key, val in out.items():
        assert not torch.isnan(val).any(), f"NaN in {key}"

    # Verify NegBin positivity
    assert (out["mu_home"] > 0).all()
    assert (out["alpha_home"] > 0).all()
    assert (out["pitcher_k_mu"] > 0).all()
    assert (out["pitcher_k_alpha"] > 0).all()
    assert (out["h_r_rbi_mu"] > 0).all()

    # Verify hits categorical is a valid distribution
    hits_cat = out["hits_categorical"]
    assert (hits_cat >= 0).all(), "Negative probability in hits_categorical"
    cat_sums = hits_cat.sum(dim=-1)
    assert torch.allclose(cat_sums, torch.ones_like(cat_sums), atol=1e-5), \
        f"hits_categorical doesn't sum to 1: {cat_sums}"

    # Verify consistency: P(1+HR) <= P(1+H) always (structural guarantee)
    p_1plus_h = 1.0 - hits_cat[:, :, 0]  # 1 - P(0 hits)
    p_1plus_hr = out["hr_prob"]
    violations = (p_1plus_hr > p_1plus_h + 1e-5).sum()
    assert violations == 0, f"Consistency violation: P(1+HR) > P(1+H) in {violations} cases"

    # Verify PA count is reasonable (>= 2.5 from softplus + offset)
    assert (out["pa_count"] >= 2.5).all(), f"PA count below 2.5: {out['pa_count'].min()}"

    # Test pregame mode (no live pitches)
    batch_pregame = {
        "context": context_batch,
        "live_continuous": None,
        "player_hashes": torch.randint(1, 50000, (B, P)),
        "player_context": torch.randn(B, P, 2 * d_model),
    }
    with torch.no_grad():
        out_pregame = model(batch_pregame)
    assert out_pregame["mu_home"].shape == (B,)
    assert out_pregame["hits_categorical"].shape == (B, P, 5)
    assert not torch.isnan(out_pregame["mu_home"]).any()

    print("Forward pass smoke test PASSED (live + pregame modes)")
    print(f"  Hits categorical valid distributions: YES")
    print(f"  P(1+HR) <= P(1+H) consistency: GUARANTEED")
    print(f"  PA count range: [{out['pa_count'].min():.2f}, {out['pa_count'].max():.2f}]")


def _test_loss():
    """Verify loss computation with new player prop structure."""
    B, P = 4, 10
    device = torch.device("cpu")

    loss_fn = GameTransformerLoss()

    predictions = {
        "mu_home": torch.rand(B) * 4 + 1,
        "alpha_home": torch.rand(B) * 5 + 1,
        "mu_away": torch.rand(B) * 4 + 1,
        "alpha_away": torch.rand(B) * 5 + 1,
        "home_win_logit": torch.randn(B),
        "yrfi_logit": torch.randn(B),
        "extra_innings_logit": torch.randn(B),
        # Player heads
        "hits_categorical": F.softmax(torch.randn(B, P, 5), dim=-1),
        "hr_prob": torch.sigmoid(torch.randn(B, P)),
        "pitcher_k_mu": torch.rand(B, P) * 5 + 1,
        "pitcher_k_alpha": torch.rand(B, P) * 3 + 1,
        "h_r_rbi_mu": torch.rand(B, P) * 3 + 1,
        "h_r_rbi_alpha": torch.rand(B, P) * 3 + 1,
        "stolen_bases_logit": torch.randn(B, P),
    }

    targets = {
        "home_runs": torch.randint(0, 10, (B,)).float(),
        "away_runs": torch.randint(0, 10, (B,)).float(),
        "home_win": torch.randint(0, 2, (B,)).float(),
        "yrfi": torch.randint(0, 2, (B,)).float(),
        "extra_innings": torch.randint(0, 2, (B,)).float(),
        "player_hits": torch.randint(0, 4, (B, P)).float(),
        "player_hr": torch.randint(0, 2, (B, P)).float(),
        "player_k": torch.randint(0, 12, (B, P)).float(),
        "player_hrbi": torch.randint(0, 6, (B, P)).float(),
        "player_sb": torch.randint(0, 2, (B, P)).float(),
        "player_mask": torch.ones(B, P),
    }

    total_loss, task_losses = loss_fn(predictions, targets)

    assert not torch.isnan(total_loss), "Total loss is NaN"
    assert total_loss > 0, "Total loss should be positive"

    # Verify all expected task losses are present
    expected_tasks = {"negbin_home", "negbin_away", "bce_home_win", "bce_yrfi",
                      "bce_extra_innings", "consistency",
                      "ce_hits", "focal_hr", "negbin_pitcher_k", "negbin_hrbi", "focal_sb"}
    for key in expected_tasks:
        assert key in task_losses, f"Missing task loss: {key}"
        assert not torch.isnan(task_losses[key]), f"NaN in {key} loss"

    # Verify focal loss is strictly positive when there are positive labels
    assert task_losses["focal_hr"] > 0, "Focal HR loss should be positive"

    # Verify gradient flows through the loss
    total_loss.backward()
    assert loss_fn.log_weights.grad is not None, "No gradient on log_weights"

    # Test with YRFI masking (live, past 1st inning)
    loss_fn.zero_grad()
    live_inning = torch.tensor([3, 1, 5, 1], dtype=torch.float32)
    total_loss_masked, task_losses_masked = loss_fn(predictions, targets, live_inning=live_inning)
    assert not torch.isnan(total_loss_masked)
    # With 2/4 samples masked, YRFI loss should differ from unmasked
    assert task_losses_masked["bce_yrfi"] != task_losses["bce_yrfi"] or \
           task_losses["bce_yrfi"] == 0, "YRFI masking should change loss"

    # Test with no players (empty player mask)
    targets_no_players = {**targets, "player_mask": torch.zeros(B, P)}
    total_loss_np, task_losses_np = loss_fn(predictions, targets_no_players)
    assert not torch.isnan(total_loss_np)
    assert task_losses_np["ce_hits"] == 0, "Player loss should be zero with empty mask"

    print("Loss function test PASSED")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Team tasks: {sum(task_losses[k].item() for k in ['negbin_home','negbin_away','bce_home_win','bce_yrfi','bce_extra_innings','consistency']):.4f}")
    print(f"  Player tasks: {sum(task_losses[k].item() for k in ['ce_hits','focal_hr','negbin_pitcher_k','negbin_hrbi','focal_sb']):.4f}")


def _test_consistency_guarantee():
    """Adversarial test: verify P(1+HR) <= P(1+H) under extreme inputs."""
    B, P = 100, 20
    d_model = 256

    config = ContextConfig(sp_games=2, team_games=3, tokens_per_game=4, flat_feature_tokens=4)
    model = GameTransformer(
        d_model=d_model,
        context_config=config,
    )

    max_pitches_ctx = 80

    def mock_game_data(num_games: int) -> dict:
        return {
            "continuous": torch.randn(B, num_games, max_pitches_ctx, 52),
            "batter_hash": torch.randint(1, 50000, (B, num_games, max_pitches_ctx)),
            "pitcher_hash": torch.randint(1, 50000, (B, num_games, max_pitches_ctx)),
            "inning_idx": torch.randint(0, 9, (B, num_games, max_pitches_ctx)),
            "ab_idx": torch.randint(0, 40, (B, num_games, max_pitches_ctx)),
            "pitch_idx": torch.randint(0, 10, (B, num_games, max_pitches_ctx)),
            "padding_mask": torch.zeros(B, num_games, max_pitches_ctx, dtype=torch.bool),
            "games_ago": torch.arange(num_games).float().unsqueeze(0).expand(B, -1),
            "seasons_crossed": torch.zeros(B, num_games),
        }

    # Run with random weights (adversarial — untrained model may produce extreme values)
    batch = {
        "context": {
            "sp_home": mock_game_data(config.sp_games),
            "sp_away": mock_game_data(config.sp_games),
            "team_home": mock_game_data(config.team_games),
            "team_away": mock_game_data(config.team_games),
            "flat_features": torch.randn(B, FLAT_FEATURE_DIM),
        },
        "live_continuous": torch.randn(B, 30, 52),
        "live_batter_hash": torch.randint(1, 50000, (B, 30)),
        "live_pitcher_hash": torch.randint(1, 50000, (B, 30)),
        "live_pitch_type_idx": torch.randint(0, 20, (B, 30)),
        "live_bat_side_idx": torch.randint(0, 3, (B, 30)),
        "live_pitch_hand_idx": torch.randint(0, 2, (B, 30)),
        "live_half_inning_idx": torch.randint(0, 2, (B, 30)),
        "live_hit_trajectory_idx": torch.randint(0, 7, (B, 30)),
        "live_hit_hardness_idx": torch.randint(0, 4, (B, 30)),
        "live_inning_idx": torch.randint(0, 9, (B, 30)),
        "live_ab_idx": torch.randint(0, 40, (B, 30)),
        "live_pitch_idx": torch.randint(0, 10, (B, 30)),
        "player_hashes": torch.randint(1, 50000, (B, P)),
        "player_context": torch.randn(B, P, 2 * d_model),
    }

    model.eval()
    with torch.no_grad():
        out = model(batch)

    # Check consistency across 100*20 = 2000 player predictions
    p_1plus_h = 1.0 - out["hits_categorical"][:, :, 0]
    p_1plus_hr = out["hr_prob"]

    violations = (p_1plus_hr > p_1plus_h + 1e-5)
    n_violations = violations.sum().item()
    total_predictions = B * P

    assert n_violations == 0, \
        f"CONSISTENCY VIOLATION: {n_violations}/{total_predictions} cases where P(1+HR) > P(1+H)"

    # Additional: verify hits_categorical sums to 1
    sums = out["hits_categorical"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), \
        f"Hits categorical doesn't sum to 1: range [{sums.min():.6f}, {sums.max():.6f}]"

    print(f"Consistency guarantee test PASSED ({total_predictions} predictions, 0 violations)")
    print(f"  P(1+HR) range: [{p_1plus_hr.min():.4f}, {p_1plus_hr.max():.4f}]")
    print(f"  P(1+H) range: [{p_1plus_h.min():.4f}, {p_1plus_h.max():.4f}]")


if __name__ == "__main__":
    _test_forward_pass()
    print()
    _test_loss()
    print()
    _test_consistency_guarantee()
