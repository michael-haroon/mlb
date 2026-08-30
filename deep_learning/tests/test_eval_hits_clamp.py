"""Eval-path parity with the training loss on the truncated hits head.

The hits head is 5-way where class 4 means "4 OR MORE" — game_transformer's loss
encodes that with `.clamp(0, 4)`. The eval path omitted the clamp, so the first
5-hit game in a split crashed test evaluation ("Target 5 is out of bounds") and
the 2026-08-30 baseline run finished with no held-out metrics at all.
"""

import torch

from mlb_dl.train_unified import _hits_categorical_metrics


def test_five_hit_game_does_not_crash_eval():
    logits = torch.randn(2, 3, 5)
    actual = torch.tensor([[0, 1, 5], [4, -1, 2]], dtype=torch.float32)
    m = _hits_categorical_metrics(logits, actual)
    assert "player_hits_ce" in m
    # -1 is the missing-player sentinel and must stay excluded; the 5 is folded
    # into class 4, so 5 of 6 slots count.
    assert m["player_hits_n"] == 5


def test_clamp_matches_training_loss_semantics():
    """A 5-hit and a 4-hit target must score identically — same class."""
    logits = torch.zeros(1, 2, 5)
    logits[0, :, 4] = 10.0
    m = _hits_categorical_metrics(logits, torch.tensor([[4, 5]], dtype=torch.float32))
    assert m["player_hits_accuracy"] == 1.0


def test_all_missing_returns_no_metrics():
    m = _hits_categorical_metrics(torch.randn(1, 2, 5),
                                 torch.tensor([[-1, -1]], dtype=torch.float32))
    assert m == {}
