"""A pregame sample must price the same alone as it does beside a live sample.

WHY this is a serving question, not just a training one. `_build_samples` emits one
prefix_len=0 sample per game and up to 32 live prefixes, so pregame rows are a small
minority that essentially always share a shuffled batch with live rows. But
`_prepare_model_input` gates the live branch on `batch["prefix_length"].sum() > 0` —
a batch-level scalar — and `GameTransformer._team_readout` then switches on
`num_live = total_len - num_context`, also batch-level. So a prefix_len=0 row takes
the mean-pooled-context path only when EVERY row in its batch is pregame, and
otherwise reads out from `backbone_out[:, -1, :]`, which for that row is pure prefix
padding.

At serving time the pregame market (Scheduled / Pre-Game / Warmup, before any pitch)
is priced with prefix_length=0. Whether that request hits the trained path or the
untrained one depends on who else is in the batch, which is a property of the request
queue, not of the game. These tests pin the invariance that makes the pregame price
well-defined.
"""

from __future__ import annotations

import pytest
import torch

from deep_learning.mlb_dl.game_transformer import (
    FLAT_FEATURE_DIM,
    ContextConfig,
    GameTransformer,
)
from deep_learning.mlb_dl.train_unified import _prepare_model_input

D_MODEL = 64
RATING_DIM = 8
N_PLAYERS = 4
PREFIX_LEN = 12
CTX_PITCHES = 6


def _model() -> GameTransformer:
    torch.manual_seed(0)
    cfg = ContextConfig(sp_games=2, team_games=2, tokens_per_game=2, flat_feature_tokens=2)
    m = GameTransformer(d_model=D_MODEL, rating_dim=RATING_DIM, context_config=cfg,
                        num_backbone_layers=2, num_heads=4, d_ff=64,
                        max_players=N_PLAYERS, player_context_tokens=2)
    m.eval()
    return m


def _collated(prefix_lengths: list[int], seed: int = 1) -> dict:
    """A collate-shaped batch. Row i has prefix_lengths[i] real pitches, left-padded.

    Mirrors game_transformer_dataset._get_live_prefix: content at the END of the
    window, mask 1.0 there and 0.0 in the leading pad.
    """
    g = torch.Generator().manual_seed(seed)
    B = len(prefix_lengths)
    cfg = ContextConfig(sp_games=2, team_games=2, tokens_per_game=2, flat_feature_tokens=2)

    out: dict = {}
    for name, n_games in (("sp_home", cfg.sp_games), ("sp_away", cfg.sp_games),
                          ("team_home", cfg.team_games), ("team_away", cfg.team_games)):
        out[f"{name}_seqs"] = torch.randn(B, n_games, CTX_PITCHES, 52, generator=g)
        out[f"{name}_obs_mask"] = torch.ones(B, n_games, CTX_PITCHES, 52)
        out[f"{name}_attn_mask"] = torch.ones(B, n_games, CTX_PITCHES)
        out[f"{name}_lengths"] = torch.full((B, n_games), CTX_PITCHES, dtype=torch.long)
        out[f"{name}_weights"] = torch.zeros(B, n_games)

    out["flat_features"] = torch.randn(B, FLAT_FEATURE_DIM, generator=g)
    out["rating_home"] = torch.randn(B, cfg.rating_steps, RATING_DIM, generator=g)
    out["rating_away"] = torch.randn(B, cfg.rating_steps, RATING_DIM, generator=g)

    out["prefix_values"] = torch.zeros(B, PREFIX_LEN, 52)
    out["prefix_obs_mask"] = torch.zeros(B, PREFIX_LEN, 52)
    out["prefix_mask"] = torch.zeros(B, PREFIX_LEN)
    for key, hi in (("prefix_batter_hash", 50_000), ("prefix_pitcher_hash", 50_000),
                    ("prefix_catcher_hash", 50_000), ("prefix_event_type", 8),
                    ("prefix_pitch_type_idx", 20), ("prefix_bat_side_idx", 4),
                    ("prefix_pitch_hand_idx", 3), ("prefix_half_inning_idx", 2),
                    ("prefix_hit_trajectory_idx", 7), ("prefix_hit_hardness_idx", 4)):
        out[key] = torch.zeros(B, PREFIX_LEN, dtype=torch.long)
    out["prefix_hierarchy"] = torch.zeros(B, PREFIX_LEN, 3, dtype=torch.long)

    for i, n in enumerate(prefix_lengths):
        if n <= 0:
            continue
        n = min(n, PREFIX_LEN)
        s = PREFIX_LEN - n  # left-pad: real pitches occupy the tail
        out["prefix_values"][i, s:] = torch.randn(n, 52, generator=g)
        out["prefix_obs_mask"][i, s:] = 1.0
        out["prefix_mask"][i, s:] = 1.0
        out["prefix_batter_hash"][i, s:] = torch.randint(1, 50_000, (n,), generator=g)
        out["prefix_pitcher_hash"][i, s:] = torch.randint(1, 50_000, (n,), generator=g)
        out["prefix_catcher_hash"][i, s:] = torch.randint(1, 50_000, (n,), generator=g)
        out["prefix_event_type"][i, s:] = torch.randint(0, 8, (n,), generator=g)
        out["prefix_pitch_type_idx"][i, s:] = torch.randint(1, 20, (n,), generator=g)
        out["prefix_hierarchy"][i, s:, 0] = torch.randint(1, 9, (n,), generator=g)

    out["prefix_length"] = torch.tensor([min(n, PREFIX_LEN) for n in prefix_lengths],
                                        dtype=torch.long)
    out["player_hashes"] = torch.randint(1, 50_000, (B, N_PLAYERS), generator=g)
    out["player_history"] = torch.randn(B, N_PLAYERS, 15, 25, generator=g)
    return out


def _price(model: GameTransformer, collated: dict, row: int) -> dict[str, float]:
    with torch.no_grad():
        out = model(_prepare_model_input(collated, player_context_dim=2 * D_MODEL))
    return {k: float(v[row].reshape(-1)[0]) for k, v in out.items()}


def test_pregame_row_prices_identically_alone_and_beside_a_live_row():
    """The same pregame game must get one price, whoever else is in the batch."""
    model = _model()

    # Row 0 is pregame in both batches and byte-identical across them; only the
    # companion row differs, so any change in row 0's price comes from the
    # batch-level live/pregame switch rather than from its own inputs.
    both_pregame = _collated([0, 0], seed=7)
    mixed = _collated([0, 0], seed=7)
    live_rows = _collated([0, 9], seed=7)
    for key, val in live_rows.items():
        if isinstance(val, torch.Tensor):
            mixed[key][1] = val[1]

    alone = _price(model, both_pregame, row=0)
    beside_live = _price(model, mixed, row=0)

    drift = {k: abs(alone[k] - beside_live[k]) for k in alone}
    worst = max(drift, key=drift.get)
    assert drift[worst] < 1e-5, (
        f"pregame price depends on batch composition: {worst} moved "
        f"{alone[worst]:.4f} -> {beside_live[worst]:.4f} (delta {drift[worst]:.4f}); "
        f"all deltas: { {k: round(v, 4) for k, v in drift.items()} }"
    )


def test_live_row_ignores_prefix_padding_of_its_batchmates():
    """A live row's price must not move when a batchmate's prefix gets longer.

    Collate pads the prefix window to a common length; without a per-row key-padding
    mask, one row's real pitches change what another row's padded positions can be
    attended to only if the mask is shared, which is exactly what to rule out here.
    """
    model = _model()

    short = _collated([9, 3], seed=11)
    longer = _collated([9, 3], seed=11)
    grown = _collated([9, PREFIX_LEN], seed=11)
    for key, val in grown.items():
        if isinstance(val, torch.Tensor):
            longer[key][1] = val[1]

    a = _price(model, short, row=0)
    b = _price(model, longer, row=0)
    drift = {k: abs(a[k] - b[k]) for k in a}
    worst = max(drift, key=drift.get)
    assert drift[worst] < 1e-5, (
        f"live price leaks across batch rows: {worst} moved {a[worst]:.4f} -> "
        f"{b[worst]:.4f} (delta {drift[worst]:.4f})"
    )
