"""`_pregame_metrics` must be able to tell "no skill" apart from "good-looking Brier".

The whole reason this metric exists is that a collapsed head — one emitting essentially the
same probability for every game — posts a perfectly respectable Brier score by simply matching
the base rate. That is what the A/B control checkpoint did on all three pregame classification
heads while the pooled val loss looked like it was improving. So the tests here are less about
arithmetic and more about the two signals that distinguish the collapse: BSS against the
slice's own base rate, and the standard deviation of the emitted probabilities.
"""

from __future__ import annotations

import math

import torch

from deep_learning.mlb_dl.train_unified import _pregame_metrics

BASE_HEADS = ("home_win", "yrfi", "extra_innings")


def _chunks(p_by_head: dict[str, list[float]], y_by_head: dict[str, list[float]],
            mu_home: list[float], mu_away: list[float],
            y_home: list[float], y_away: list[float]) -> dict[str, list[torch.Tensor]]:
    out: dict[str, list[torch.Tensor]] = {}
    for head in BASE_HEADS:
        out[f"p_{head}"] = [torch.tensor(p_by_head[head])]
        out[f"y_{head}"] = [torch.tensor(y_by_head[head])]
    out["mu_home"] = [torch.tensor(mu_home)]
    out["mu_away"] = [torch.tensor(mu_away)]
    out["y_home_runs"] = [torch.tensor(y_home)]
    out["y_away_runs"] = [torch.tensor(y_away)]
    return out


def _uniform(p: list[float], y: list[float]) -> dict[str, list[torch.Tensor]]:
    """Same p/y on every classification head, with runs held at a trivial fixed value."""
    n = len(y)
    return _chunks(
        {h: p for h in BASE_HEADS}, {h: y for h in BASE_HEADS},
        mu_home=[2.0] * n, mu_away=[2.0] * n,
        y_home=[2.0] * n, y_away=[2.0] * n,
    )


def test_empty_input_returns_no_metrics():
    """A split with no pregame rows must yield nothing, not a division by zero."""
    assert _pregame_metrics({}) == {}


def test_constant_predictor_at_the_base_rate_scores_zero_skill():
    """THE COLLAPSE SIGNATURE: Brier 0.25 looks unremarkable, BSS says it is worthless."""
    y = [1.0, 0.0, 1.0, 0.0]
    m = _pregame_metrics(_uniform([0.5] * 4, y))

    assert m["pregame/n"] == 4
    for head in BASE_HEADS:
        assert math.isclose(m[f"pregame/{head}_brier"], 0.25, abs_tol=1e-9)
        assert math.isclose(m[f"pregame/{head}_bss"], 0.0, abs_tol=1e-9), (
            f"{head} BSS should be exactly 0 for a base-rate constant"
        )
        assert math.isclose(m[f"pregame/{head}_logloss"], math.log(2), abs_tol=1e-6)
        assert math.isclose(m[f"pregame/{head}_pstd"], 0.0, abs_tol=1e-9), (
            f"{head} pstd must be 0 for a constant head — this is the collapse detector"
        )


def test_a_skewed_base_rate_cannot_fake_skill():
    """extra_innings sits near 8%, where a constant 0.08 gets a Brier of ~0.074.

    Absolute Brier would read as excellent. BSS must still be 0, because the reference is the
    slice's own base rate rather than 0.5.
    """
    y = [0.0] * 92 + [1.0] * 8
    m = _pregame_metrics(_uniform([0.08] * 100, y))

    assert m["pregame/extra_innings_brier"] < 0.08, "sanity: skewed Brier looks good"
    assert math.isclose(m["pregame/extra_innings_bss"], 0.0, abs_tol=1e-9)


def test_perfect_predictor_scores_bss_one_and_wide_spread():
    y = [1.0, 0.0, 1.0, 0.0]
    m = _pregame_metrics(_uniform(y, y))
    for head in BASE_HEADS:
        assert m[f"pregame/{head}_bss"] > 0.999
        assert m[f"pregame/{head}_pstd"] > 0.5, "a discriminating head must spread its output"


def test_anticorrelated_predictor_scores_negative_skill():
    """Worse than a constant must read as negative, not clip at zero."""
    y = [1.0, 0.0, 1.0, 0.0]
    m = _pregame_metrics(_uniform([0.1, 0.9, 0.1, 0.9], y))
    for head in BASE_HEADS:
        assert m[f"pregame/{head}_bss"] < -1.0


def test_degenerate_single_class_slice_does_not_divide_by_zero():
    """If every game in the slice has the same outcome the base-rate Brier is 0."""
    m = _pregame_metrics(_uniform([0.7, 0.7, 0.7], [1.0, 1.0, 1.0]))
    for head in BASE_HEADS:
        assert m[f"pregame/{head}_bss"] == 0.0
        assert math.isfinite(m[f"pregame/{head}_brier"])


def test_total_runs_mae_is_measured_against_the_slice_mean():
    """The control checkpoint lost to this baseline, so the baseline must be reported too."""
    y_home, y_away = [2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0]
    # totals 4, 6, 8, 10 -> mean 7 -> mean |y - 7| = (3+1+1+3)/4 = 2.0
    exact = _pregame_metrics(_chunks(
        {h: [0.5] * 4 for h in BASE_HEADS}, {h: [1.0, 0.0, 1.0, 0.0] for h in BASE_HEADS},
        mu_home=y_home, mu_away=y_away, y_home=y_home, y_away=y_away,
    ))
    assert math.isclose(exact["pregame/total_runs_mae"], 0.0, abs_tol=1e-9)
    assert math.isclose(exact["pregame/total_runs_mae_base"], 2.0, abs_tol=1e-9)

    # A constant predictor at the slice mean must exactly tie the baseline.
    tied = _pregame_metrics(_chunks(
        {h: [0.5] * 4 for h in BASE_HEADS}, {h: [1.0, 0.0, 1.0, 0.0] for h in BASE_HEADS},
        mu_home=[3.5] * 4, mu_away=[3.5] * 4, y_home=y_home, y_away=y_away,
    ))
    assert math.isclose(
        tied["pregame/total_runs_mae"], tied["pregame/total_runs_mae_base"], abs_tol=1e-9
    )


def test_metrics_accumulate_across_batches():
    """Chunks arrive one per validation batch; concatenation order must not change the result."""
    split = {
        k: [t[0][:2].clone(), t[0][2:].clone()]
        for k, t in _uniform([0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 1.0, 0.0]).items()
    }
    whole = _uniform([0.5] * 4, [1.0, 0.0, 1.0, 0.0])
    a, b = _pregame_metrics(split), _pregame_metrics(whole)
    assert a.keys() == b.keys()
    for k in a:
        assert math.isclose(a[k], b[k], abs_tol=1e-9), k


def test_validate_harvests_only_pregame_rows_end_to_end():
    """Integration: `_validate` must emit pregame/* keys and count ONLY prefix_length==0 rows.

    Guards the wiring, not the arithmetic — a batch-level rather than per-row selection, or a
    missing target key, would sail past the unit tests above.
    """
    import torch as _t

    from deep_learning.mlb_dl.game_transformer import GameTransformerLoss
    from deep_learning.mlb_dl.train_unified import _validate
    from test_pregame_readout_invariance import D_MODEL, _collated, _model

    prefix_lengths = [0, 5, 0, 9]  # exactly 2 pregame rows
    batch = _collated(prefix_lengths, seed=5)
    n = len(prefix_lengths)
    batch["targets"] = {
        "home_runs_remaining": _t.tensor([3.0, 2.0, 5.0, 1.0]),
        "away_runs_remaining": _t.tensor([4.0, 1.0, 2.0, 6.0]),
        "home_win": _t.tensor([1.0, 0.0, 1.0, 0.0]),
        "yrfi": _t.tensor([0.0, 1.0, 1.0, 0.0]),
        "extra_innings": _t.tensor([0.0, 0.0, 1.0, 0.0]),
    }
    batch["player_mask"] = _t.zeros(n, 4)

    model = _model()
    loss, tasks = _validate(
        model, GameTransformerLoss(), [batch], _t.device("cpu"),
        player_context_dim=2 * D_MODEL,
    )

    assert math.isfinite(loss)
    assert tasks["pregame/n"] == 2, (
        f"harvested {tasks['pregame/n']} rows from prefix_lengths={prefix_lengths}; expected the "
        "2 rows with no pitches"
    )
    for head in BASE_HEADS:
        for suffix in ("brier", "logloss", "bss", "pstd"):
            assert math.isfinite(tasks[f"pregame/{head}_{suffix}"]), f"{head}_{suffix}"
    assert math.isfinite(tasks["pregame/total_runs_mae"])
    assert math.isfinite(tasks["pregame/total_runs_mae_base"])


def test_validate_emits_no_pregame_keys_when_every_row_is_live():
    """A live-only split must not fabricate a pregame metric out of zero rows."""
    import torch as _t

    from deep_learning.mlb_dl.game_transformer import GameTransformerLoss
    from deep_learning.mlb_dl.train_unified import _validate
    from test_pregame_readout_invariance import D_MODEL, _collated, _model

    batch = _collated([4, 6], seed=5)
    batch["targets"] = {
        "home_runs_remaining": _t.tensor([3.0, 2.0]),
        "away_runs_remaining": _t.tensor([4.0, 1.0]),
        "home_win": _t.tensor([1.0, 0.0]),
        "yrfi": _t.tensor([0.0, 1.0]),
        "extra_innings": _t.tensor([0.0, 0.0]),
    }
    batch["player_mask"] = _t.zeros(2, 4)

    _, tasks = _validate(
        _model(), GameTransformerLoss(), [batch], _t.device("cpu"),
        player_context_dim=2 * D_MODEL,
    )
    assert not [k for k in tasks if k.startswith("pregame/")]
