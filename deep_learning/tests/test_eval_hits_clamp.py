"""Eval-path parity with the training loss on the truncated hits head.

The hits head is 5-way where class 4 means "4 OR MORE" — game_transformer's loss
encodes that with `.clamp(0, 4)`. The eval path omitted the clamp, so the first
5-hit game in a split crashed test evaluation ("Target 5 is out of bounds") and
the 2026-08-30 baseline run finished with no held-out metrics at all.
"""

import pytest
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


def test_ce_matches_the_training_loss_on_probability_input():
    """The head emits PROBABILITIES, not logits — `hits_categorical` is normalized to
    sum to 1 (game_transformer, `hits_categorical / sum`), and game_transformer's
    `ce_hits` scores it as -log(p[target]). The eval metric has to agree, because the
    A/B compares the two runs on this number.

    Feeding probabilities to F.cross_entropy instead re-applies log_softmax to them.
    Probabilities live in [0, 1], so a softmax over five of them is nearly uniform and
    the score collapses toward log(5) = 1.609 whatever the model predicted — a perfect
    head and a random one report almost the same value.
    """
    probs = torch.tensor([[[0.9, 0.05, 0.03, 0.01, 0.01],
                           [0.1, 0.7, 0.1, 0.05, 0.05]]])
    actual = torch.tensor([[0, 1]], dtype=torch.float32)
    m = _hits_categorical_metrics(probs, actual)
    expected = float(-(torch.log(torch.tensor([0.9, 0.7]))).mean())  # 0.22579
    assert m["player_hits_ce"] == pytest.approx(expected, abs=1e-4), (
        f"got {m['player_hits_ce']}, expected {expected:.5f}; a value near "
        f"{float(torch.log(torch.tensor(5.0))):.3f} means the probabilities were softmaxed again")


def test_ce_actually_separates_a_good_head_from_a_bad_one():
    """The regression guard with teeth: whatever the formula, a confidently-correct head
    must score far better than a confidently-wrong one. Re-softmaxing probabilities makes
    these two nearly equal, which is what let the defect survive the other tests here."""
    actual = torch.tensor([[0, 0]], dtype=torch.float32)
    good = torch.tensor([[[0.96, 0.01, 0.01, 0.01, 0.01]] * 2])
    bad = torch.tensor([[[0.01, 0.96, 0.01, 0.01, 0.01]] * 2])
    ce_good = _hits_categorical_metrics(good, actual)["player_hits_ce"]
    ce_bad = _hits_categorical_metrics(bad, actual)["player_hits_ce"]
    assert ce_bad - ce_good > 3.0, (
        f"good {ce_good} vs bad {ce_bad}: the metric barely distinguishes them, so it "
        f"cannot support an A/B decision")
