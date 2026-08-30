"""HR and SB heads predict P(1+ event); their loss targets must be binary.

`GameTransformer` returns `"hr_prob": p_1plus_hr` (game_transformer.py:931) and
`stolen_bases_logit` for the same 1-or-more event. But the loss fed them the raw
targets, which are COUNTS:

    prepared test split, player_mask=1 slots
      player_hr: min 0, max 4, 0.80% of slots > 1
      player_sb: min 0, max 3, 0.49% of slots > 1

BCE with a target t outside [0,1] is not a probability loss. For t = 2:

    -[t*log p + (1-t)*log(1-p)] = -2*log p + log(1-p)

the second term has a NEGATIVE coefficient, so it *rewards* p -> 1 without bound.
Those slots therefore drag the head upward with no opposing force, which is part
of why hr_prob came out 3.5x its base rate. The fix is to binarise the target to
match what the head means.
"""

import torch

from mlb_dl.game_transformer import GameTransformerLoss


def _loss_for_targets(target_value: float, key: str, loss_key: str) -> float:
    """Run one player head's loss with every valid slot set to `target_value`."""
    B, P = 2, 4
    crit = GameTransformerLoss()
    preds = _preds(B, P)
    targets = _targets(B, P)
    targets[key] = torch.full((B, P), float(target_value))
    _total, losses = crit(preds, targets)
    return float(losses[loss_key])


def _preds(B, P):
    """The loss reads every head, so a partial dict raises before reaching the
    player block."""
    torch.manual_seed(0)
    return {
        "mu_home": torch.rand(B) * 4 + 1, "alpha_home": torch.rand(B) * 5 + 1,
        "mu_away": torch.rand(B) * 4 + 1, "alpha_away": torch.rand(B) * 5 + 1,
        "home_win_logit": torch.randn(B), "yrfi_logit": torch.randn(B),
        "extra_innings_logit": torch.randn(B),
        "hits_categorical": torch.softmax(torch.randn(B, P, 5), dim=-1),
        "hr_prob": torch.full((B, P), 0.3),
        "pitcher_k_mu": torch.rand(B, P) * 5 + 1,
        "pitcher_k_alpha": torch.rand(B, P) * 3 + 1,
        "h_r_rbi_mu": torch.rand(B, P) * 3 + 1,
        "h_r_rbi_alpha": torch.rand(B, P) * 3 + 1,
        "stolen_bases_logit": torch.full((B, P), -0.8),
    }


def _targets(B, P):
    return {
        "home_runs_remaining": torch.full((B,), 4.0),
        "away_runs_remaining": torch.full((B,), 3.0),
        "home_win": torch.ones(B), "yrfi": torch.ones(B),
        "extra_innings": torch.zeros(B),
        "player_hits": torch.ones(B, P),
        "player_hr": torch.zeros(B, P),
        "player_so": torch.ones(B, P),
        "player_hrbi": torch.ones(B, P),
        "player_sb": torch.zeros(B, P),
        "player_mask": torch.ones(B, P),
    }


def test_hr_count_target_two_scores_same_as_one():
    """A 2-HR game and a 1-HR game are the same event for a P(1+ HR) head."""
    assert _loss_for_targets(2.0, "player_hr", "focal_hr") == \
           _loss_for_targets(1.0, "player_hr", "focal_hr")


def test_sb_count_target_two_scores_same_as_one():
    assert _loss_for_targets(3.0, "player_sb", "focal_sb") == \
           _loss_for_targets(1.0, "player_sb", "focal_sb")


def test_multi_event_target_does_not_produce_a_lower_loss_than_a_single_event():
    """The pre-fix failure mode in one assertion: with a raw count target, a
    confidently-wrong-scale prediction scored BETTER on 2-HR slots than on 1-HR
    slots, because the (1-t) term flipped sign and paid the model to say p=1."""
    for key, lk in (("player_hr", "focal_hr"), ("player_sb", "focal_sb")):
        l1 = _loss_for_targets(1.0, key, lk)
        l2 = _loss_for_targets(2.0, key, lk)
        assert l2 >= l1 - 1e-6, f"{lk}: count target 2 gave lower loss ({l2}) than 1 ({l1})"


def test_zero_target_still_penalises_a_high_prediction():
    """Sanity: binarising must not flatten the negative class."""
    for key, lk in (("player_hr", "focal_hr"), ("player_sb", "focal_sb")):
        assert _loss_for_targets(0.0, key, lk) > 0.0
        assert _loss_for_targets(0.0, key, lk) != _loss_for_targets(1.0, key, lk)


def test_binary_targets_are_unchanged_by_the_fix():
    """0/1 targets are already the event indicator, so the fix is a no-op there
    — the change must not move any already-correct row."""
    B, P = 3, 5
    crit = GameTransformerLoss()
    preds = _preds(B, P)
    y = (torch.rand(B, P) > 0.5).float()
    t1 = _targets(B, P); t1["player_hr"] = y; t1["player_sb"] = y
    t2 = _targets(B, P); t2["player_hr"] = (y > 0).float(); t2["player_sb"] = (y > 0).float()
    _a, out = crit(preds, t1)
    _b, out2 = crit(preds, t2)
    assert torch.equal(out["focal_hr"], out2["focal_hr"])
    assert torch.equal(out["focal_sb"], out2["focal_sb"])
