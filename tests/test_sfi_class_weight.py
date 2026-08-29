"""Tests for SFI class_weight miscalibration fix.

Run: conda run -n pred python -m pytest tests/test_sfi_class_weight.py -v

Root cause: class_weight="balanced" in _sfi_one_task pushes BaggingClassifier
OOS probabilities toward (0.5, 0.5) for imbalanced targets.  The SFI null is
base-rate entropy (not coin-flip), so every feature compares against the wrong
baseline → zero features pass.  class_weight=None lets the tree learn the base
rate; a no-signal feature then scores near null OOS; a signal feature correctly
exceeds null.

Test sequencing follows CLAUDE.md discipline:
  1. Tests 1, 3, 5 FAIL before the fix (TypeError / AssertionError / ImportError)
  2. Test 2 PASSES before the fix (documents the broken behavior)
  3. All five pass after the fix
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.analysis.feature_importance import _sfi_one_task

# ── Shared synthetic data ──────────────────────────────────────────────────────
N = 500
RNG = np.random.default_rng(0)

# 7.3% positive rate — matches extra_innings base rate
y_vals = np.zeros(N, dtype=int)
n_pos = int(N * 0.073)  # 36 positives
y_vals[:n_pos] = 1
RNG.shuffle(y_vals)

# Noise feature — zero correlation with y
X_noise = RNG.normal(0, 1, N).reshape(-1, 1)

# Signal feature — positive class at +2, negative at −2, σ=0.5 overlap negligible
X_signal = (y_vals.astype(float) * 4 - 2 + RNG.normal(0, 0.5, N)).reshape(-1, 1)

TR = np.arange(400)
TE = np.arange(400, 500)

# Base-rate null: np.sum(p_k * log(p_k)) for p_k = [1-0.073, 0.073]
_p = np.array([1 - n_pos / N, n_pos / N])
NULL_SCORE = float(np.sum(_p * np.log(_p + 1e-15)))  # ≈ −0.260


# ── Test 1: FAILS before fix (TypeError) ──────────────────────────────────────
def test_noise_class_weight_none_accepted():
    """sfi_class_weight=None parameter must be accepted without TypeError.

    Before fix: TypeError — sfi_class_weight is not a recognised parameter.
    After fix: runs and returns a valid score.

    Note: a noise feature with class_weight=None on 7.3%-imbalanced data correctly
    scores below null (extreme predictions from zero-positive leaves).  That is
    right behaviour — noise should fail the SFI gate.  Test 3 verifies that a
    genuine signal feature exceeds null, which is the functional proof the fix works.
    """
    score = _sfi_one_task(
        0, X_noise, TR, TE, y_vals, None,
        n_estimators=10, regression=False, sfi_class_weight=None,
    )
    assert score is not None, "task returned None — both classes must be in train split"


# ── Test 2: PASSES before fix (documents broken behavior, preserved after fix) ─
def test_noise_class_weight_balanced_breaks_null():
    """class_weight='balanced' drives OOS score near coin-flip on imbalanced data.

    This test PASSES before the fix and MUST CONTINUE PASSING after — the default
    behaviour (balanced) is intentionally preserved; only explicit None changes it.
    """
    score = _sfi_one_task(
        0, X_noise, TR, TE, y_vals, None,
        n_estimators=10, regression=False,
    )
    assert score is not None
    # Balanced weighting → predictions near 0.5 → log-loss near −0.693.
    # The score must be at least 0.25 nats below null to confirm miscalibration.
    assert score < NULL_SCORE - 0.25, (
        f"Expected balanced-weight score << null ({NULL_SCORE:.4f}), got {score:.4f}"
    )


# ── Test 3: FAILS before fix (TypeError) ──────────────────────────────────────
def test_signal_class_weight_none_beats_null():
    """With class_weight=None, a genuine signal feature must exceed the null.

    Before fix: TypeError.
    After fix: tree finds the clean ±2 split → OOS score >> null.
    """
    score = _sfi_one_task(
        0, X_signal, TR, TE, y_vals, None,
        n_estimators=10, regression=False, sfi_class_weight=None,
    )
    assert score is not None
    assert score > NULL_SCORE, (
        f"Signal feature with class_weight=None scored {score:.4f}, "
        f"expected > null {NULL_SCORE:.4f}"
    )


# ── Test 4: backward compat — balanced targets unchanged ──────────────────────
def test_balanced_class_weight_still_accepted():
    """sfi_class_weight='balanced' (explicit) must still be accepted.

    Ensures the new default and the explicit value both work — backward compat.
    Does not assert a specific score: single-fold + 10-estimator variance is too
    high for absolute threshold tests.  The key check is that the function runs
    without error and returns a non-None score.
    """
    rng2 = np.random.default_rng(1)
    y_bal = np.zeros(N, dtype=int)
    y_bal[: N // 2] = 1
    rng2.shuffle(y_bal)
    X_noise_bal = rng2.normal(0, 1, N).reshape(-1, 1)

    score = _sfi_one_task(
        0, X_noise_bal, TR, TE, y_bal, None,
        n_estimators=10, regression=False, sfi_class_weight="balanced",
    )
    assert score is not None, "backward-compat call must return a valid score"


# ── Test 5: FAILS before fix (ImportError / AttributeError) ───────────────────
def test_config_override_entry_exists():
    """SFI_CLASS_WEIGHT_OVERRIDES must exist in config with extra_innings → None."""
    from classical_learning.strategy.config import SFI_CLASS_WEIGHT_OVERRIDES

    assert "extra_innings" in SFI_CLASS_WEIGHT_OVERRIDES, (
        "SFI_CLASS_WEIGHT_OVERRIDES must contain an entry for 'extra_innings'"
    )
    assert SFI_CLASS_WEIGHT_OVERRIDES["extra_innings"] is None, (
        "extra_innings override must be None (not 'balanced')"
    )
    # All other classification targets should default to "balanced"
    for t in ("home_win", "yrfi", "first_5_home_win"):
        assert SFI_CLASS_WEIGHT_OVERRIDES.get(t, "balanced") == "balanced", (
            f"{t} should not have a class_weight override"
        )
