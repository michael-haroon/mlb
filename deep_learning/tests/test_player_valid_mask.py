"""Held-out player metrics must use the SAME validity mask training used.

Training weights every player head by `targets["player_mask"]`
(GameTransformerLoss, `(focal * pm).sum() / n_valid`). Evaluation filtered on
`y >= 0` instead. Padding slots are encoded as 0, not -1, so that filter is a
no-op: it admits every slot, including the ones the loss deliberately excluded.

Measured on the real prepared test split (2,711 games x 20 slots):
  player_mask valid : 47,144
  y >= 0 valid      : 54,220   <- 7,076 padding slots scored anyway
  padding slots carry genuine outcomes (2.5% HR positives, max 2 HR)
  HR base rate      : 0.12746 masked vs 0.11415 unmasked
  SB base rate      : 0.06881 masked vs 0.06215 unmasked

That is not a rounding difference. It shifts the constant-predictor baseline
every skill score is measured against, and it scores the model on rows it was
never trained to fit.
"""

import numpy as np
import pytest

from mlb_dl.train_unified import _player_valid


def test_padding_slots_encoded_as_zero_are_excluded():
    """The bug in one assertion: y >= 0 keeps all 4, player_mask keeps 2."""
    y = np.array([1.0, 0.0, 0.0, 0.0])
    pm = np.array([1.0, 1.0, 0.0, 0.0])
    v = _player_valid(y, pm)
    assert v.tolist() == [True, True, False, False]
    assert (y >= 0).sum() == 4          # what the old filter admitted


def test_masked_base_rate_differs_from_unmasked():
    """Padding slots dilute the base rate, which moves the p(1-p) baseline."""
    y = np.concatenate([np.ones(3), np.zeros(7), np.zeros(10)])
    pm = np.concatenate([np.ones(10), np.zeros(10)])
    v = _player_valid(y, pm)
    assert y[v].mean() == pytest.approx(0.3)
    assert y[y >= 0].mean() == pytest.approx(0.15)


def test_padding_slots_with_nonzero_targets_are_still_excluded():
    """Real data has HR=2 on masked slots. Excluding by target value instead of
    by the mask would silently keep exactly those rows."""
    y = np.array([0.0, 2.0, 1.0])
    pm = np.array([1.0, 0.0, 0.0])
    assert _player_valid(y, pm).tolist() == [True, False, False]


def test_negative_sentinel_still_excluded_when_mask_present():
    """Belt and braces: a -1 must not slip through just because its mask is 1."""
    y = np.array([-1.0, 1.0])
    pm = np.array([1.0, 1.0])
    assert _player_valid(y, pm).tolist() == [False, True]


def test_falls_back_to_sentinel_filter_when_mask_absent():
    """The from-frames path may not carry player_mask; the metric must still be
    computable rather than crashing or silently scoring everything."""
    y = np.array([-1.0, 0.0, 1.0])
    assert _player_valid(y, None).tolist() == [False, True, True]


def test_boolean_and_float_masks_both_accepted():
    y = np.zeros(3)
    assert _player_valid(y, np.array([1.0, 0.0, 1.0])).tolist() == [True, False, True]
    assert _player_valid(y, np.array([True, False, True])).tolist() == [True, False, True]


def test_shape_mismatch_is_an_error_not_a_silent_broadcast():
    """A (B,P) mask flattened against a (B*P,) target must line up exactly;
    numpy would happily broadcast a length-1 mask over everything."""
    with pytest.raises(ValueError):
        _player_valid(np.zeros(20), np.ones(7))
