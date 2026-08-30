"""Every trained head must produce a held-out metric, keyed to a real target.

Two silent gaps found on 2026-08-30 by reading a completed eval JSON:

  * `extra_innings_logit` is trained and `classical_baseline` even carries an
    `extra_innings_brier` to compare against, but `_evaluate_model` never computed
    one — so the head shipped unmeasured.
  * the pitcher-strikeout block reads `all_targets["player_pitcher_k"]`, while the
    dataset emits `player_so` (precollate row 2). The key never matched, the
    `if` was never entered, and no error was raised: the strikeout-props head had
    NO held-out number at all.

Both are the same failure mode — a metric that is absent rather than wrong, which
no assertion on values can catch. These tests assert on the key sets instead.
"""

import ast
import inspect

import pytest

from mlb_dl import precollate, train_unified


def _dict_literal_keys(func, varname: str) -> set[str]:
    """Keys of the first `varname = {...}` literal assigned inside `func`."""
    tree = ast.parse(inspect.getsource(func).lstrip())
    for node in ast.walk(tree):
        if (isinstance(node, ast.AnnAssign | ast.Assign)
                and isinstance(node.value, ast.Dict)):
            tgt = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id == varname:
                return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    raise AssertionError(f"no dict literal named {varname} in {func.__name__}")


def _prepared_target_keys() -> set[str]:
    """Target keys PreparedDataset.__getitem__ actually emits."""
    src = inspect.getsource(precollate.PreparedDataset.__getitem__)
    tree = ast.parse(src.lstrip())
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys |= {k.value for k in node.keys
                     if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return keys


def test_every_target_eval_reads_is_actually_produced():
    """A key the dataset never emits makes its whole metric block dead code."""
    wanted = _dict_literal_keys(train_unified._evaluate_model, "all_targets")
    produced = _prepared_target_keys()
    missing = sorted(wanted - produced)
    assert not missing, (
        f"_evaluate_model reads target keys the dataset never produces: {missing}. "
        "The metric block is silently skipped rather than failing."
    )


@pytest.mark.parametrize("head_metric", [
    "home_win_brier",
    "yrfi_brier",
    "extra_innings_brier",
    "player_hr_brier",
    "player_sb_brier",
    "player_hits_ce",
    "player_hrbi_nll",
    "player_so_nll",
])
def test_metric_is_emitted_for_every_trained_head(head_metric):
    """Grep the source for the literal metric key. Crude, but it is exactly the
    check that would have caught both gaps, and it needs no GPU or fixtures."""
    # `player_hits_ce` is emitted by the _hits_categorical_metrics helper rather
    # than inline, so search the helpers _evaluate_model delegates to as well.
    src = "".join(inspect.getsource(f) for f in (
        train_unified._evaluate_model,
        train_unified._hits_categorical_metrics,
        train_unified._binary_skill,
    ))
    assert f'"{head_metric}"' in src, f"{head_metric} is never computed"


def test_binary_heads_report_their_constant_baseline():
    """A Brier without its p(1-p) baseline is unreadable, and comparing the two by
    hand is how the HR head was misjudged as unusable."""
    src = inspect.getsource(train_unified._evaluate_model)
    for head in ("home_win", "yrfi", "extra_innings", "player_hr", "player_sb"):
        assert f'_binary_skill("{head}"' in src, f"{head} has no skill-score line"


def test_home_win_is_not_compared_against_the_pregame_classical_brier():
    """The DL home_win head is conditioned on LIVE in-game state; the classical
    home_win Brier is conditioned on pregame information only. Subtracting them
    measures the information set, not the model, and the resulting "41.74%
    improvement" is the kind of number that justifies a bad go/no-go call.

    extra_innings is the one head where both sides are pregame-comparable, and
    there the DL model is marginally WORSE (0.06812 vs 0.0677) -- the opposite
    conclusion to the one the home_win comparison invited.
    """
    src = inspect.getsource(train_unified._cmd_evaluate)
    assert "home_win_brier_pct_improvement" not in src, (
        "home_win is being compared across different information sets "
        "(in-game state vs pregame). This comparison is invalid."
    )
