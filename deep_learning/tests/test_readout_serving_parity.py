"""Adversarial coverage for the per-row `_team_readout` switch.

`test_pregame_readout_invariance.py` pins the core property (a pregame row prices the same
beside a live row as beside another pregame row). These are the cases that property does not
reach, and each one maps to a way the fix could have been wrong:

  1. SERVING PARITY. Production prices one pregame game as a batch of 1, where
     `_prepare_model_input` omits the live tensors entirely (`prefix_length.sum() == 0`), so the
     model takes the `num_live == 0` branch — a *different* branch from the per-row one a mixed
     training batch takes. Both must agree, or the number the book trades is not the number the
     model was trained to produce.
  2. LIVE REGRESSION. Rows with pitches must be bit-exact against the legacy readout. Prefixes
     are left-padded so position -1 is already the last real pitch; if this drifts at all, every
     live price from every existing checkpoint has silently moved and the fix is not
     backward-compatible.
  3. NEIGHBOUR IMMUNITY. Flipping row 0 from pregame to live must not perturb any other row.
     Catches a `torch.where` that broadcasts along the wrong axis — which would still satisfy
     (1) while quietly corrupting live rows.
"""

from __future__ import annotations

import torch

from deep_learning.mlb_dl.train_unified import _prepare_model_input


# Plain (non-relative) import: deep_learning/tests has no __init__.py, so pytest puts this
# directory on sys.path rather than importing it as a package.
from test_pregame_readout_invariance import D_MODEL, _collated, _model  # noqa: E402

BATCH = 64
# Row 0 is the pregame row under test; the rest carry pitches. Lengths vary so the batch is not
# accidentally uniform in a way that hides a broadcast bug.
LIVE_LENGTHS = [7 + (i % 5) for i in range(BATCH - 1)]


def _price(model, collated) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        return model(_prepare_model_input(collated, player_context_dim=2 * D_MODEL))


def _row(out: dict, i: int) -> dict[str, torch.Tensor]:
    return {k: v[i].reshape(-1).clone() for k, v in out.items()}


def _worst(a: dict, b: dict) -> tuple[str, float]:
    deltas = {k: (a[k] - b[k]).abs().max().item() for k in a}
    key = max(deltas, key=deltas.get)
    return key, deltas[key]


def _mixed_batch() -> dict:
    return _collated([0] + LIVE_LENGTHS, seed=3)


def test_pregame_prices_the_same_as_a_batch_of_one():
    """THE SERVING CASE: batch-of-1 hits `num_live == 0`, batch-of-64 hits the per-row path."""
    model = _model()
    mixed = _mixed_batch()

    solo = _collated([0], seed=99)
    for key, val in mixed.items():
        if isinstance(val, torch.Tensor):
            solo[key] = val[0:1].clone()

    key, delta = _worst(_row(_price(model, mixed), 0), _row(_price(model, solo), 0))
    assert delta < 1e-5, (
        f"pregame price depends on batch size: {key} differs by {delta:.3e} between a batch of "
        "64 and the batch of 1 that serving actually uses"
    )


def test_live_rows_are_bit_exact_against_the_legacy_readout():
    """Legacy behaviour is reproduced by withholding `live_lengths`, so compare against that."""
    model = _model()
    mixed = _mixed_batch()

    new_input = _prepare_model_input(mixed, player_context_dim=2 * D_MODEL)
    legacy_input = dict(new_input)
    # Tolerant pop: if the per-row fix is ever reverted the key is simply absent, and these
    # tests should then fail on their own assertion rather than on a KeyError.
    legacy_input.pop("live_lengths", None)
    with torch.no_grad():
        new_out = model(new_input)
        legacy_out = model(legacy_input)

    live_idx = [i for i, n in enumerate(mixed["prefix_length"].tolist()) if n > 0]
    assert live_idx, "fixture produced no live rows"
    for i in live_idx:
        key, delta = _worst(_row(new_out, i), _row(legacy_out, i))
        assert delta == 0.0, (
            f"live row {i} moved: {key} by {delta:.3e}. Live readout must be untouched or every "
            "existing checkpoint's live prices have silently shifted"
        )


def test_the_pregame_row_does_change_so_the_fix_demonstrably_engages():
    """Guards against the whole suite passing because the per-row branch is never taken."""
    model = _model()
    mixed = _mixed_batch()

    new_input = _prepare_model_input(mixed, player_context_dim=2 * D_MODEL)
    legacy_input = dict(new_input)
    # Tolerant pop: if the per-row fix is ever reverted the key is simply absent, and these
    # tests should then fail on their own assertion rather than on a KeyError.
    legacy_input.pop("live_lengths", None)
    with torch.no_grad():
        new_out = model(new_input)
        legacy_out = model(legacy_input)

    key, delta = _worst(_row(new_out, 0), _row(legacy_out, 0))
    assert delta > 1e-6, (
        "pregame row is identical to the legacy padding readout, so the per-row branch never "
        f"engaged (largest move was {key} at {delta:.3e})"
    )


def test_flipping_row_zero_to_live_leaves_every_other_row_untouched():
    """A wrong broadcast axis in the torch.where would pass the parity test but fail here."""
    model = _model()
    mixed = _mixed_batch()

    flipped = _collated([11] + LIVE_LENGTHS, seed=3)
    for key, val in mixed.items():
        if isinstance(val, torch.Tensor):
            flipped[key][1:] = val[1:]

    baseline = _price(model, mixed)
    after = _price(model, flipped)

    live_idx = [i for i, n in enumerate(mixed["prefix_length"].tolist()) if n > 0]
    for i in live_idx:
        key, delta = _worst(_row(after, i), _row(baseline, i))
        assert delta < 1e-6, (
            f"row {i} moved by {delta:.3e} ({key}) when row 0 flipped pregame->live; the "
            "readout is leaking across the batch dimension"
        )
