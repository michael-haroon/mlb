"""The calibration diagnosis decides whether a mispriced head needs a post-hoc
map or an architecture change, so its own math has to be right first."""

import numpy as np
import pytest

from diagnose_head_calibration import _auc, _platt, _to_logit, analyse


def test_auc_hand_computable():
    """2 positives, 2 negatives, one discordant pair out of four -> 0.75."""
    y = np.array([0.0, 1.0, 0.0, 1.0])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    assert _auc(y, s) == pytest.approx(0.75)


def test_auc_perfect_and_inverted():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert _auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert _auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)


def test_auc_all_ties_is_one_half():
    """A constant score discriminates nothing; average ranks must give 0.5, not 1.0."""
    y = np.array([0.0, 1.0, 0.0, 1.0])
    assert _auc(y, np.full(4, 0.42)) == pytest.approx(0.5)


def test_auc_undefined_with_one_class():
    assert np.isnan(_auc(np.zeros(5), np.arange(5.0)))


def test_auc_is_invariant_to_monotone_rescaling():
    """The property the whole diagnosis rests on: inflating every score cannot
    change AUC, so AUC isolates signal from scale."""
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.2).astype(float)
    z = rng.normal(size=500) + 2 * y
    p = 1 / (1 + np.exp(-z))
    inflated = 1 / (1 + np.exp(-(3.0 * z + 1.5)))
    assert _auc(y, p) == pytest.approx(_auc(y, inflated))


def test_platt_recovers_a_known_inflation():
    """Scores built from a true probability then pushed up by a fixed logit
    offset must come back to the true mean after the map."""
    rng = np.random.default_rng(1)
    n = 20000
    x = rng.normal(size=n)
    p_true = 1 / (1 + np.exp(-(0.8 * x - 1.5)))
    y = (rng.random(n) < p_true).astype(float)
    z_inflated = 0.8 * x - 1.5 + 2.0          # 2.0 logit units too high
    p_cal = _platt(z_inflated, y, z_inflated)
    assert abs(p_cal.mean() - y.mean()) < 0.01
    # and it must beat the uncalibrated score on Brier
    p_bad = 1 / (1 + np.exp(-z_inflated))
    assert np.mean((p_cal - y) ** 2) < np.mean((p_bad - y) ** 2)


def test_platt_is_identity_on_already_calibrated_scores():
    rng = np.random.default_rng(2)
    n = 20000
    z = rng.normal(size=n) - 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    p_cal = _platt(z, y, z)
    p_raw = 1 / (1 + np.exp(-z))
    assert np.mean((p_cal - y) ** 2) == pytest.approx(np.mean((p_raw - y) ** 2), abs=2e-4)


def _fake_split(y, s, pkey="stolen_bases_logit", tkey="player_sb"):
    return {f"p::{pkey}": s, f"t::{tkey}": y}


def test_verdict_miscalibrated_only_for_inflated_but_ranking_head():
    """The HR/SB signature: strong ranking, Brier worse than a constant, and a
    val-fit map restores positive skill."""
    rng = np.random.default_rng(3)
    n = 40000
    xa, xb = rng.normal(size=n), rng.normal(size=n)
    z_a, z_b = 1.2 * xa - 2.2, 1.2 * xb - 2.2
    y_a = (rng.random(n) < 1 / (1 + np.exp(-z_a))).astype(float)
    y_b = (rng.random(n) < 1 / (1 + np.exp(-z_b))).astype(float)
    # focal-style inflation: same ranking, far too high in level
    rep = analyse(_fake_split(y_a, z_a + 3.0), _fake_split(y_b, z_b + 3.0))["player_sb"]
    assert rep["pred_mean_over_base"] > 2.0
    assert rep["bss_vs_constant"] < 0            # worse than quoting the base rate
    assert rep["auc"] > 0.7                      # but it ranks well
    assert rep["bss_after_platt"] > 0            # and the map recovers skill
    assert rep["verdict"].startswith("MISCALIBRATED ONLY")


def test_verdict_no_signal_for_noise_head():
    rng = np.random.default_rng(4)
    n = 20000
    y = (rng.random(n) < 0.1).astype(float)
    rep = analyse(_fake_split((rng.random(n) < 0.1).astype(float), rng.normal(size=n)),
                  _fake_split(y, rng.normal(size=n)))["player_sb"]
    assert rep["auc"] < 0.55
    assert rep["verdict"].startswith("NO SIGNAL")


def test_empty_player_slots_are_excluded_not_scored_as_negatives():
    """A -1 slot means "no player here". Counting it as a negative would both
    invent outcomes and dilute the base rate the constant baseline uses."""
    y = np.array([-1.0, 1.0, 1.0, 0.0, -1.0, 0.0])
    s = np.array([0.0, 3.0, 2.0, -3.0, 0.0, -2.0])
    rep = analyse(_fake_split(y, s), _fake_split(y, s))["player_sb"]
    assert rep["n"] == 4
    assert rep["base_rate"] == pytest.approx(0.5)


def test_probability_head_is_logit_transformed_before_mapping():
    """hr_prob arrives as a probability; feeding it to Platt raw would fit a
    logistic on [0,1] and silently weaken the correction."""
    p = np.array([0.01, 0.5, 0.99])
    z = _to_logit(p)
    assert z[1] == pytest.approx(0.0)
    assert z[0] < -4 and z[2] > 4
    # clipping keeps 0 and 1 finite
    assert np.isfinite(_to_logit(np.array([0.0, 1.0]))).all()


# ── Isotonic calibration ──────────────────────────────────────────────────────
from diagnose_head_calibration import _isotonic


def test_isotonic_is_monotone_non_decreasing():
    rng = np.random.default_rng(10)
    n = 5000
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    grid = np.linspace(-4, 4, 200)
    p = _isotonic(z, y, grid)
    assert np.all(np.diff(p) >= -1e-12)


def test_isotonic_recovers_a_nonlinear_distortion_platt_cannot():
    """The decisive property. A score distorted NON-linearly in logit space (what
    focal loss does) defeats Platt but not the monotone optimum."""
    rng = np.random.default_rng(11)
    n = 60000
    x = rng.normal(size=n)
    z_true = 1.5 * x - 2.0
    y = (rng.random(n) < 1 / (1 + np.exp(-z_true))).astype(float)
    # monotone but strongly non-linear reparameterisation of the true logit
    z_warped = np.sign(z_true) * np.abs(z_true) ** 3 + 4.0
    b_const = y.mean() * (1 - y.mean())
    b_platt = np.mean((_platt(z_warped, y, z_warped) - y) ** 2)
    b_iso = np.mean((_isotonic(z_warped, y, z_warped) - y) ** 2)
    assert b_iso < b_platt
    assert b_iso < b_const          # monotone optimum must beat the constant


def test_isotonic_beats_constant_whenever_auc_exceeds_half():
    """Brier calibration-refinement decomposition: any discrimination at all is
    enough once the map is monotone-optimal. This is the assumption the verdict
    logic leans on, so it is asserted rather than assumed."""
    rng = np.random.default_rng(12)
    for base in (0.05, 0.12, 0.4):
        n = 80000
        x = rng.normal(size=n)
        z = 0.7 * x + np.log(base / (1 - base))
        y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
        s_inflated = 9.0 * z + 20.0        # ranks identically, absurd scale
        b_const = y.mean() * (1 - y.mean())
        b_iso = np.mean((_isotonic(s_inflated, y, s_inflated) - y) ** 2)
        assert _auc(y, s_inflated) > 0.55
        assert b_iso < b_const, f"base={base}: iso {b_iso} !< const {b_const}"


def test_isotonic_on_pure_noise_collapses_to_the_base_rate():
    """No signal -> the monotone optimum is a single block at the base rate, so
    it ties the constant instead of beating it."""
    rng = np.random.default_rng(13)
    n = 40000
    y = (rng.random(n) < 0.1).astype(float)
    p = _isotonic(rng.normal(size=n), y, rng.normal(size=n))
    assert abs(p.mean() - 0.1) < 0.02
    b_const = y.mean() * (1 - y.mean())
    assert np.mean((p - y) ** 2) <= b_const + 1e-3


def test_isotonic_handles_constant_score_and_single_class_blocks():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    p = _isotonic(np.full(4, 2.0), y, np.array([-5.0, 2.0, 9.0]))
    assert np.allclose(p, 0.5)
