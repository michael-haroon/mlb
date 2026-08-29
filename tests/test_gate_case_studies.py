"""Gate case studies — document what the current filter_features logic does.

Each test specifies a sequence of fold importances, pre-computes the relevant
gate inputs (mean, ci_hi, global rho, recent_mean), then asserts BOTH the
intermediate values AND the expected tier so we can see exactly which gates
fire and which don't.

Run:
    conda run -n pred python -m pytest tests/test_gate_case_studies.py -v

No production code is modified here. This is read-only exploration.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_ci(vals, n_boot=2000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, size=len(vals), replace=True).mean()
             for _ in range(n_boot)]
    alpha = 1 - ci
    return (vals.mean(),
            np.percentile(means, 100 * alpha / 2),
            np.percentile(means, 100 * (1 - alpha / 2)))


def _gate_summary(vals, null=0.0, recency=3):
    """Return a dict of every gate input so tests can assert on each."""
    mean, ci_lo, ci_hi = _bootstrap_ci(vals)
    rho, _ = spearmanr(np.arange(len(vals)), vals)
    recent_mean = vals[-recency:].mean()
    peak = vals.max()

    initial_pass = (mean > null) and (ci_hi > null)
    demotion = initial_pass and (rho < -0.6) and (recent_mean <= null)
    rescue    = (not initial_pass) and (recent_mean > null) and (rho > 0.5)

    if demotion:
        predicted_tier = "REJECTED"
    elif rescue:
        predicted_tier = "NEEDS SPECIFICATION"
    elif initial_pass:
        predicted_tier = "ACCEPTED"
    else:
        predicted_tier = "REJECTED"

    return dict(mean=mean, ci_lo=ci_lo, ci_hi=ci_hi,
                rho=rho, recent_mean=recent_mean, peak=peak,
                initial_pass=initial_pass, demotion=demotion, rescue=rescue,
                predicted_tier=predicted_tier)


def _run_filter(vals_dict, null=0.0):
    """Call filter_features with identical arrays for all methods."""
    from classical_learning.analysis.feature_importance import filter_features

    names = list(vals_dict.keys())
    mdi_raw      = pd.DataFrame({n: np.maximum(v, 0.0) for n, v in vals_dict.items()})
    sfi_raw      = pd.DataFrame(vals_dict)
    desub_raw    = pd.DataFrame(vals_dict)
    pca_raw      = pd.DataFrame(vals_dict)
    resid_raw    = pd.DataFrame(vals_dict)
    cfi_raw      = pd.DataFrame({i: vals_dict[n] for i, n in enumerate(names)})
    clusters     = {i: [n] for i, n in enumerate(names)}

    return filter_features(
        mdi_raw=mdi_raw, cfi_mda_raw=cfi_raw, clusters=clusters,
        sfi_raw=sfi_raw, sfi_null=null,
        desub_mda_raw=desub_raw, pca_mda_raw=pca_raw, resid_mda_raw=resid_raw,
    )


ANCHOR = np.array([0.8, 0.9, 0.7, 0.85, 0.95, 0.88, 0.92, 0.87])


# ─────────────────────────────────────────────────────────────────────────────
#  Case studies
# ─────────────────────────────────────────────────────────────────────────────

class TestGateCaseStudies:

    # ── 1. User's new example ────────────────────────────────────────────────

    def test_case_01_user_new_example(self):
        """[-1.2, -1, -.5, -.8, -.2, .01, -0.1, .05]

        Overall upward trend (rho ≈ +0.95) but mean is heavily negative
        and recent 3-fold mean is also slightly negative (-0.013).

        Gates:
          mean (-0.47) > 0          → FAILS — mean gate kills it
          ci_hi                     → likely negative (dominated by large negatives)
          initial_pass              → False
          rescue (recent > 0?)      → False (recent_mean ≈ -0.013 < 0)
          → REJECTED, NO rescue despite upward global trend
        """
        vals = np.array([-1.2, -1.0, -0.5, -0.8, -0.2, 0.01, -0.1, 0.05])
        g = _gate_summary(vals)

        assert g["mean"] < 0,        f"mean={g['mean']:.4f}: should be negative"
        assert g["ci_hi"] < 0,       f"ci_hi={g['ci_hi']:.4f}: should be negative (dominated by large negatives)"
        assert g["rho"] > 0.8,       f"rho={g['rho']:.3f}: overall upward trend should be strong"
        assert g["recent_mean"] < 0, f"recent_mean={g['recent_mean']:.4f}: last 3 still net negative"
        assert not g["initial_pass"]
        assert not g["rescue"],      "rescue blocked: recent_mean < 0 even though rho > 0.5"
        assert g["predicted_tier"] == "REJECTED"

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        assert report.loc["feature", "tier"] == "REJECTED"

    # ── 2. Noisy-stable — all positive, oscillating ─────────────────────────

    def test_case_02_noisy_stable(self):
        """[0.03, 0.05, 0.02, 0.04, 0.03, 0.05, 0.02, 0.04]

        All folds positive, oscillating with no trend.

        Gates:
          mean > 0                  → PASSES
          ci_hi > 0                 → PASSES (entire CI positive)
          initial_pass              → True
          rho ≈ 0                   → demotion blocked (|rho| << 0.6)
          → ACCEPTED/NEEDS SPEC, let models weigh it
        """
        vals = np.array([0.03, 0.05, 0.02, 0.04, 0.03, 0.05, 0.02, 0.04])
        g = _gate_summary(vals)

        assert g["mean"] > 0
        assert g["ci_hi"] > 0
        assert g["initial_pass"]
        assert g["rho"] > -0.6,  f"rho={g['rho']:.3f}: oscillating, should not trigger demotion"
        assert not g["demotion"]
        assert g["predicted_tier"] in ("ACCEPTED", "NEEDS SPECIFICATION")

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        assert report.loc["feature", "tier"] in ("ACCEPTED", "NEEDS SPECIFICATION")

    # ── 3. Truly decaying into negatives ────────────────────────────────────

    def test_case_03_truly_decaying_negative(self):
        """[0.08, 0.07, 0.05, 0.03, 0.01, -0.01, -0.02, -0.03]

        Was clearly signal, now clearly noise.

        Gates:
          mean (0.025) > 0          → PASSES mean gate
          ci_hi > 0                 → PASSES (historical mean was real)
          initial_pass              → True
          rho ≈ -1.0                → demotion check triggered
          recent_mean (-0.02) <= 0  → demotion fires
          → REJECTED by trend demotion
        """
        vals = np.array([0.08, 0.07, 0.05, 0.03, 0.01, -0.01, -0.02, -0.03])
        g = _gate_summary(vals)

        assert g["mean"] > 0
        assert g["ci_hi"] > 0
        assert g["initial_pass"]
        assert g["rho"] < -0.6,         f"rho={g['rho']:.3f}: should be strongly negative"
        assert g["recent_mean"] <= 0,   f"recent_mean={g['recent_mean']:.4f}: last 3 should be negative"
        assert g["demotion"],           "demotion should fire: rho < -0.6 AND recent <= 0"
        assert g["predicted_tier"] == "REJECTED"

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        assert report.loc["feature", "tier"] == "REJECTED"

    # ── 4. Near-null decaying — the current gap ──────────────────────────────

    def test_case_04_near_null_decaying_CURRENT_PASSES(self):
        """[0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01, 0.005]

        Monotonically declining from 0.10 to 0.005. Feature has given back
        95% of its historical signal and is still falling.

        Gates:
          mean (0.043) > 0          → PASSES
          ci_hi > 0                 → PASSES (all positive)
          initial_pass              → True
          rho ≈ -1.0                → demotion check triggered
          recent_mean (0.012) > 0   → demotion BLOCKED ← the gap
          → PASSES (current code considers this acceptable)

          IT SHOULD FAIL! because it degenerated to noise in recent years.

        This test documents the current behavior, NOT necessarily the desired behavior.
        The feature has 95% peak drawdown and is still declining — is current behavior right?
        """
        vals = np.array([0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01, 0.005])
        g = _gate_summary(vals)

        assert g["mean"] > 0
        assert g["ci_hi"] > 0
        assert g["initial_pass"]
        assert g["rho"] < -0.6,          f"rho={g['rho']:.3f}: should be strongly negative"
        assert g["recent_mean"] > 0,     f"recent_mean={g['recent_mean']:.4f}: blocks demotion cond A"
        assert not g["demotion"],        "current code: demotion blocked because recent_mean > 0"
        assert g["predicted_tier"] == "ACCEPTED"

        # Documenting current code output — not asserting this is correct
        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        actual_tier = report.loc["feature", "tier"]
        # Current code passes this. Change this assertion when logic is updated.
        assert actual_tier in ("REJECTED"), (
            f"Current code passes near_null_decaying as {actual_tier}. "
            f"peak={g['peak']:.3f}, last={vals[-1]:.3f}, rho={g['rho']:.3f}"
        )

    # ── 5. Recovering feature — was bad, now clearly improving ───────────────

    def test_case_05_recovering_trend_rescued(self):
        """[-0.5, -0.4, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3]

        Historical mean is negative but the feature is on a clear upward
        trajectory and recent folds are solidly positive.

        Gates:
          mean (-0.0875) > 0        → FAILS mean gate
          ci_hi                     → spans zero, likely positive (trend pulls it up)
          initial_pass              → False (mean gate fails)
          rho ≈ +1.0                → rescue check triggered
          recent_mean (0.2) > 0     → rescue fires
          → NEEDS SPECIFICATION via rescue, meaning it passes
        """
        vals = np.array([-0.5, -0.4, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3])
        g = _gate_summary(vals)

        assert g["mean"] < 0,         f"mean={g['mean']:.4f}: should be negative"
        assert g["rho"] > 0.9,        f"rho={g['rho']:.3f}: should be strongly positive"
        assert g["recent_mean"] > 0,  f"recent_mean={g['recent_mean']:.4f}: should be positive"
        assert not g["initial_pass"]
        assert g["rescue"],           "rescue should fire: recent > 0 AND rho > 0.5"
        assert g["predicted_tier"] == "NEEDS SPECIFICATION"

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        assert report.loc["feature", "tier"] == "NEEDS SPECIFICATION"

    # ── 6. Sudden drop then plateau ──────────────────────────────────────────

    def test_case_06_sudden_drop_then_plateau(self):
        """[0.10, 0.02, 0.015, 0.018, 0.012, 0.016, 0.014, 0.013]

        One big initial drop, then bounces around a stable low level.

        Gates:
          mean (0.026) > 0          → PASSES
          ci_hi > 0                 → PASSES (all positive)
          initial_pass              → True
          rho: initial drop dominates, likely < -0.6
          recent_mean (0.014) > 0   → demotion BLOCKED (same gap as case 4)
          → PASSES (plateau is treated as stable, not decaying)

        Different from case 4: this is a one-time regime shift, not a trend.
        Keeping it is probably correct. Contrast with case 4 where the decline
        is continuous with no sign of stabilization.
        """
        vals = np.array([0.10, 0.02, 0.015, 0.018, 0.012, 0.016, 0.014, 0.013])
        g = _gate_summary(vals)

        assert g["mean"] > 0
        assert g["ci_hi"] > 0
        assert g["initial_pass"]
        assert g["rho"] < -0.6,       f"rho={g['rho']:.3f}: initial drop drives negative rho"
        assert g["recent_mean"] > 0,  "recent plateau keeps demotion blocked"
        assert not g["demotion"]
        assert g["predicted_tier"] in ("ACCEPTED", "NEEDS SPECIFICATION")

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        assert report.loc["feature", "tier"] in ("ACCEPTED", "NEEDS SPECIFICATION")

    # ── 7. Single outlier blip inflating mean ────────────────────────────────

    def test_case_07_single_outlier_inflates_mean(self):
        """[-0.1, -0.15, -0.05, 0.8, -0.12, -0.08, -0.09, -0.11]

        One huge positive fold (0.8) pulls mean positive. Everything else negative.
        Bootstrap CI should NOT be positive — the outlier distorts the mean but
        the CI sees through it (the other 7 folds are all negative).

        Gates:
          mean (0.0125) > 0         → PASSES (outlier inflated it)
          ci_hi                     → spans zero or negative? CI based on resampling
                                       will often exclude the outlier — likely negative
          → Depends on bootstrap behavior
        """
        vals = np.array([-0.1, -0.15, -0.05, 0.8, -0.12, -0.08, -0.09, -0.11])
        g = _gate_summary(vals)

        # Document actual values — do not assert ci direction (it's the interesting question)
        print(f"\n  mean={g['mean']:.4f}, ci=[{g['ci_lo']:.4f}, {g['ci_hi']:.4f}], "
              f"rho={g['rho']:.3f}, recent_mean={g['recent_mean']:.4f}")
        print(f"  initial_pass={g['initial_pass']}, demotion={g['demotion']}, "
              f"rescue={g['rescue']}, predicted={g['predicted_tier']}")

        # Mean IS positive due to outlier
        assert g["mean"] > 0, f"mean={g['mean']:.4f}: outlier pushes mean positive"
        # Recent 3 folds are all negative — no rescue possible
        assert g["recent_mean"] < 0, f"recent_mean={g['recent_mean']:.4f}: last 3 are negative"

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        actual = report.loc["feature", "tier"]
        # Just document — the outlier case is the interesting question for ci_hi behavior
        print(f"  actual_tier={actual}")

    # ── 8. Flat at tiny positive ─────────────────────────────────────────────

    def test_case_08_flat_tiny_positive(self):
        """[0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]

        Perfectly constant tiny positive. Zero variance — bootstrap CI degenerates
        to a point mass. Passes all gates easily despite being tiny.

        Gates:
          mean (0.001) > 0          → PASSES
          ci_hi = 0.001 > 0         → PASSES
          rho = undefined (constant) → no demotion
          → ACCEPTED
        """
        vals = np.array([0.001] * 8)
        g = _gate_summary(vals)

        assert g["mean"] > 0
        assert g["ci_hi"] > 0
        assert g["initial_pass"]
        assert g["predicted_tier"] in ("ACCEPTED", "NEEDS SPECIFICATION")

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        assert report.loc["feature", "tier"] in ("ACCEPTED", "NEEDS SPECIFICATION")

    # ── 9. Zeros then positives — emerging signal ────────────────────────────

    def test_case_09_zeros_then_positives(self):
        """[0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.06, 0.07]

        Feature dormant then suddenly active. Mean is low (0.022) but positive.
        Rho ≈ +1 (monotonic). Recent folds positive.

        Gates:
          mean (0.022) > 0          → PASSES
          ci_hi > 0                 → PASSES (recent positives)
          initial_pass              → True
          rho ≈ +1                  → no demotion (rho > 0)
          → ACCEPTED/NEEDS SPEC
        """
        vals = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.06, 0.07])
        g = _gate_summary(vals)

        assert g["mean"] > 0
        # Four leading zeros create tied ranks, deflating Spearman slightly below 1.0
        assert g["rho"] > 0.8,   f"rho={g['rho']:.3f}: generally increasing (ties deflate rho)"
        assert g["initial_pass"]
        assert not g["demotion"]
        assert g["predicted_tier"] in ("ACCEPTED", "NEEDS SPECIFICATION")

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        assert report.loc["feature", "tier"] in ("ACCEPTED", "NEEDS SPECIFICATION")

    # ── 10. Negative then slowly creeping toward zero, not yet rescued ───────

    def test_case_10_creeping_toward_zero_not_rescued(self):
        """[-0.4, -0.35, -0.3, -0.25, -0.2, -0.1, -0.05, -0.01]

        Upward trend, rho ≈ +1. But even the most recent fold is negative.
        Rescue requires recent_mean > 0 — not met yet.

        Gates:
          mean (-0.21) > 0          → FAILS
          ci_hi                     → negative (all values negative)
          initial_pass              → False
          rho ≈ +1                  → rescue check triggered
          recent_mean (-0.053) > 0  → FAILS — rescue blocked
          → REJECTED, no rescue
        """
        vals = np.array([-0.4, -0.35, -0.3, -0.25, -0.2, -0.1, -0.05, -0.01])
        g = _gate_summary(vals)

        assert g["mean"] < 0
        assert g["ci_hi"] < 0,       f"ci_hi={g['ci_hi']:.4f}: all negative, CI should be negative"
        assert g["rho"] > 0.9,       f"rho={g['rho']:.3f}: should be strongly positive"
        assert g["recent_mean"] < 0, f"recent_mean={g['recent_mean']:.4f}: last 3 still negative"
        assert not g["rescue"],      "rescue blocked: recent_mean still < 0"
        assert g["predicted_tier"] == "REJECTED"

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        assert report.loc["feature", "tier"] == "REJECTED"

    # ── 11. Zigzag with positive mean — no trend either way ─────────────────

    def test_case_11_zigzag_positive_mean(self):
        """[0.1, -0.05, 0.12, -0.03, 0.09, -0.04, 0.11, -0.02]

        Alternating high/low. Mean is positive (0.035). CI likely spans zero.
        Rho ≈ 0 (no trend). Mean > 0, ci_hi > 0 (pulled by positives) → PASSES.

        But this one needs specificication. Should be flagged for checking
        """
        vals = np.array([0.1, -0.05, 0.12, -0.03, 0.09, -0.04, 0.11, -0.02])
        g = _gate_summary(vals)

        assert g["mean"] > 0,      f"mean={g['mean']:.4f}: alternating but net positive"
        assert abs(g["rho"]) < 0.6, f"rho={g['rho']:.3f}: no systematic trend"
        assert g["initial_pass"]
        assert not g["demotion"]
        assert g["predicted_tier"] in ("ACCEPTED", "NEEDS SPECIFICATION")

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        assert report.loc["feature", "tier"] in ("ACCEPTED", "NEEDS SPECIFICATION")

    # ── 12. Slow positive trend, all positive ───────────────────────────────

    def test_case_12_slow_growth_all_positive(self):
        """[0.02, 0.03, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

        Growing signal, always positive.

        Gates:
          mean > 0                  → PASSES
          ci_hi > 0                 → PASSES
          rho ≈ +1                  → no demotion
          → ACCEPTED
        """
        vals = np.array([0.02, 0.03, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
        g = _gate_summary(vals)

        assert g["mean"] > 0
        assert g["ci_hi"] > 0
        assert g["rho"] > 0.9,   f"rho={g['rho']:.3f}"
        assert g["initial_pass"]
        assert not g["demotion"]
        assert g["predicted_tier"] in ("ACCEPTED", "NEEDS SPECIFICATION")

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        assert report.loc["feature", "tier"] in ("ACCEPTED", "NEEDS SPECIFICATION")

    # ── 13. W-shaped — dip then recovery, net positive ───────────────────────

    def test_case_13_w_shaped_dip_recovery(self):
        """[0.08, 0.05, 0.02, -0.01, -0.02, 0.03, 0.07, 0.09]

        Feature dipped into negative then strongly recovered.

        Gates:
          mean (0.038) > 0          → PASSES
          ci_hi > 0                 → PASSES
          initial_pass              → True
          rho: U-shaped means rho ≈ 0 (not monotonic either way)
          → no demotion, PASSES
        """
        vals = np.array([0.08, 0.05, 0.02, -0.01, -0.02, 0.03, 0.07, 0.09])
        g = _gate_summary(vals)

        assert g["mean"] > 0
        assert g["initial_pass"]
        assert abs(g["rho"]) < 0.6, f"rho={g['rho']:.3f}: U-shaped should have low |rho|"
        assert not g["demotion"]
        assert g["predicted_tier"] in ("ACCEPTED", "NEEDS SPECIFICATION")

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        assert report.loc["feature", "tier"] in ("ACCEPTED", "NEEDS SPECIFICATION")

    # ── 14. Strong signal fading to noise but mean still clearly positive ────

    def test_case_14_fading_signal_mean_still_positive(self):
        """[0.30, 0.25, 0.20, 0.10, 0.05, 0.04, 0.03, 0.02]

        Larger magnitude case. Mean = 0.124 (positive). Recent mean = 0.03 (positive).
        rho ≈ -1. Demotion blocked because recent_mean > 0.

        This is the same structural gap as case 4 — larger magnitude version.
        """
        vals = np.array([0.30, 0.25, 0.20, 0.10, 0.05, 0.04, 0.03, 0.02])
        g = _gate_summary(vals)

        assert g["mean"] > 0
        assert g["rho"] < -0.6,       f"rho={g['rho']:.3f}"
        assert g["recent_mean"] > 0,  f"recent_mean={g['recent_mean']:.4f}: blocks demotion"
        assert not g["demotion"],     "same gap as case 4, larger magnitude"
        assert g["predicted_tier"] in ("ACCEPTED", "NEEDS SPECIFICATION")

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        assert report.loc["feature", "tier"] in ("ACCEPTED", "NEEDS SPECIFICATION")

    # ── 15a. Decaying — last fold touches zero, demotion just misses ─────────

    def test_case_15a_decaying_plateau_at_zero(self):
        """[0.10, 0.08, 0.06, 0.04, 0.02, 0.01, 0.005, 0.0]

        Last fold exactly hits null. recent_mean = (0.01+0.005+0.0)/3 = 0.005 > 0.
        Demotion requires recent_mean <= 0 — blocked by 0.005.
        One more declining fold would flip it. Documents the boundary of condition A.
        Current code: PASSES.
        """
        vals = np.array([0.10, 0.08, 0.06, 0.04, 0.02, 0.01, 0.005, 0.0])
        g = _gate_summary(vals)

        assert g["rho"] < -0.6
        assert g["recent_mean"] > 0, (
            f"recent_mean={g['recent_mean']:.4f}: barely positive, demotion blocked"
        )
        assert not g["demotion"]

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        assert report.loc["feature", "tier"] in ("ACCEPTED", "NEEDS SPECIFICATION")

    # ── 15b. Decaying — last fold just crosses zero ───────────────────────────

    def test_case_15b_decaying_just_below_zero(self):
        """[0.12, 0.10, 0.08, 0.04, 0.02, 0.01, 0.005, -0.001]

        Monotonically decreasing. recent_mean = (0.01+0.005-0.001)/3 = +0.0047 > 0.
        Same structural gap as cases 4 and 14 — recent_mean stays positive even
        though the feature is clearly converging toward (and crossing) null.
        Expected: REJECTED. Current code: PASSES.
        """
        vals = np.array([0.12, 0.10, 0.08, 0.04, 0.02, 0.01, 0.005, -0.001])
        g = _gate_summary(vals)

        assert g["rho"] < -0.6
        assert g["recent_mean"] > 0, (
            f"recent_mean={g['recent_mean']:.4f}: still just positive, demotion blocked"
        )
        assert not g["demotion"]

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        assert report.loc["feature", "tier"] == "REJECTED", (
            f"Monotonically declining feature crossing null should be REJECTED, "
            f"got {report.loc['feature', 'tier']}"
        )

    # ── 16. Single-outlier fragility — demotion fires on noise, not decay ────

    def test_case_16_single_outlier_fragility(self):
        """[0.30, 0.28, 0.32, 0.29, 0.31, 0.05, -0.50, 0.10]

        Six folds strongly positive (0.28–0.32), one outlier fold at -0.50
        (bad game, noise, regime shock), then recovery to 0.10.
        The -0.50 is not decay — it's a single bad fold with immediate recovery.

        Gates (current):
          mean = +0.144, ci_hi = +0.283   → initial_pass = True
          rho = -0.619 (just below -0.6)  → demotion threshold crossed
          recent_mean = -0.117 (driven by the -0.50 outlier) → demotion fires
          => REJECTED(demotion)

        Expected: NEEDS SPECIFICATION — the outlier should not doom a
        feature that was cleanly positive for 6/8 folds and recovered.
        The demotion rule cannot distinguish a single noise outlier from
        genuine terminal decay. This test FAILS against current code.
        """
        vals = np.array([0.30, 0.28, 0.32, 0.29, 0.31, 0.05, -0.50, 0.10])
        g = _gate_summary(vals)

        # Preconditions: verify this is the outlier-fragility scenario
        assert g["mean"] > 0,       f"mean={g['mean']:.4f}: bulk signal is positive"
        assert g["ci_hi"] > 0,      f"ci_hi={g['ci_hi']:.4f}"
        assert g["initial_pass"]
        assert g["rho"] < -0.6,     f"rho={g['rho']:.4f}: outlier drags rho negative"
        assert g["recent_mean"] < 0, f"recent_mean={g['recent_mean']:.4f}: dominated by -0.50"
        assert g["demotion"],       "current code: demotion fires due to outlier"

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        tier = report.loc["feature", "tier"]
        # FAILS against current code — documents the single-outlier fragility bug
        assert tier in ("ACCEPTED", "NEEDS SPECIFICATION"), (
            f"Feature with 6/8 positive folds and one noise outlier should not be "
            f"REJECTED, got {tier}. The demotion rule cannot distinguish outlier from decay."
        )

    # ── 17. Early outlier inflates mean — should be REJECTED ─────────────────

    def test_case_17_early_outlier_inflates_mean(self):
        """[0.50, -0.20, -0.15, 0.15, -0.10, 0.10, -0.05, 0.15]

        One large positive fold early (0.50). The remaining 7 folds oscillate
        tightly around zero (mean of remaining = 0.0 - roughly). The 0.50 is
        an early-season outlier, not representative of the feature's true signal.

        Gates (current):
          mean = +0.050, ci = [-0.081, +0.206]  → mean > 0 AND ci_hi > 0 → initial_pass
          rho = +0.12 (no trend detected)       → no demotion
          => ACCEPTED/NEEDS SPEC

        Expected: REJECTED — the overall mean is only positive because of one
        early outlier. The feature itself is noise (7 folds near zero, no trend).
        The two-part gate (mean > 0 AND ci_hi > 0) admits it because the outlier
        pulls both mean and CI upper into positive territory. This test FAILS.
        """
        vals = np.array([0.50, -0.20, -0.15, 0.15, -0.10, 0.10, -0.05, 0.15])
        g = _gate_summary(vals)

        # Preconditions: verify the outlier-inflated-mean scenario
        assert vals[0] > 0.4,          "first fold is the outlier"
        assert vals[1:].mean() < 0.05, (
            f"remaining 7 folds mean={vals[1:].mean():.4f}: nearly zero without outlier"
        )
        assert g["mean"] > 0,          f"mean={g['mean']:.4f}: outlier inflates it"
        assert g["ci_hi"] > 0,         f"ci_hi={g['ci_hi']:.4f}: outlier pulls CI up"
        assert g["initial_pass"],      "current code: gate admits because mean > 0 and ci_hi > 0"
        assert abs(g["rho"]) < 0.3,    f"rho={g['rho']:.4f}: no systematic trend detected"
        assert not g["demotion"],      "no demotion: rho not strongly negative"

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        tier = report.loc["feature", "tier"]
        # FAILS against current code — documents the noise-inflated-mean bug
        assert tier == "REJECTED", (
            f"Feature driven positive only by one early outlier should be REJECTED, "
            f"got {tier}. 7/8 folds are near-zero noise."
        )

    # ── 18. All-negative monotonic rise — REJECTED despite upward trend ───────

    def test_case_18_all_negative_monotonic_rise(self):
        """[-2.0, -1.8, -1.5, -1.3, -1.0, -0.8, -0.5, -0.3]

        Strongly negative throughout, rho = +1.0 (perfectly monotonic increase).
        Even the most recent fold is -0.3. Rescue requires recent_mean > 0 — blocked.
        Mean gate fails outright.

        Gates:
          mean = -1.15, ci entirely negative  → initial_pass = False
          rho = +1.0                          → rescue check triggered
          recent_mean = -0.533               → rescue blocked (must be > 0)
          => REJECTED(gate)

        Expected: REJECTED. The trend is encouraging but the feature has never
        been positive — not enough evidence to rescue it yet.
        """
        vals = np.array([-2.0, -1.8, -1.5, -1.3, -1.0, -0.8, -0.5, -0.3])
        g = _gate_summary(vals)

        assert g["mean"] < 0
        assert g["ci_hi"] < 0,       f"ci_hi={g['ci_hi']:.4f}: entirely negative"
        assert not g["initial_pass"]
        assert g["rho"] > 0.99,      f"rho={g['rho']:.4f}: perfect monotonic rise"
        assert g["recent_mean"] < 0, f"recent_mean={g['recent_mean']:.4f}: still negative"
        assert not g["rescue"],      "rescue blocked: recent_mean < 0"

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        assert report.loc["feature", "tier"] == "REJECTED"

    # ── 19. Low rho, strong recent mean — should NOT be rejected ─────────────

    def test_case_19_low_rho_strong_recent_not_rejected(self):
        """[0.60, -0.10, 0.55, -0.45, -0.10, 0.36, 0.45, 0.65]

        Highly volatile — big swings up and down. No systematic trend (rho ≈ 0.18).
        But the overall mean is +0.245 and recent 3 folds are strong (+0.487).
        The volatility is noise around a genuine positive signal.

        Gates:
          mean = +0.245, ci = [-0.043, +0.494]  → mean > 0 AND ci_hi > 0 → initial_pass
          rho = +0.18                            → no demotion (not < -0.6)
          => ACCEPTED/NEEDS SPEC

        Expected: NOT REJECTED — volatile but net positive, no decay trend.
        """
        vals = np.array([0.60, -0.10, 0.55, -0.45, -0.10, 0.36, 0.45, 0.65])
        g = _gate_summary(vals)

        assert g["mean"] > 0,          f"mean={g['mean']:.4f}"
        assert g["recent_mean"] > 0.4, f"recent_mean={g['recent_mean']:.4f}: strongly positive"
        assert abs(g["rho"]) < 0.3,    f"rho={g['rho']:.4f}: no systematic trend"
        assert g["initial_pass"]
        assert not g["demotion"]

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        assert report.loc["feature", "tier"] != "REJECTED", (
            f"Volatile but net-positive feature should not be REJECTED, "
            f"got {report.loc['feature', 'tier']}"
        )

    # ── 20. Dip then recovery — recent mean barely negative, should NOT reject

    def test_case_20_dip_then_recovery_not_rejected(self):
        """[0.8, 0.6, 0.4, -0.2, -0.1, -1.3, 0.5, 0.7]

        Feature dips sharply at fold 6 (-1.3), then recovers strongly (0.5, 0.7).
        Overall mean = +0.175. rho = -0.238 (not strongly negative — U-shaped).
        recent_mean = (-1.3+0.5+0.7)/3 = -0.033 (barely negative due to the dip).

        Gates:
          mean = +0.175, ci_hi = +0.563  → initial_pass = True
          rho = -0.238                   → demotion not triggered (not < -0.6)
          => ACCEPTED/NEEDS SPEC (recovery is real, dip was transient)

        Expected: NOT REJECTED — the strong recovery at folds 7–8 shows the
        feature rebounded. The low rho correctly reflects the V-shape, not decay.
        """
        vals = np.array([0.8, 0.6, 0.4, -0.2, -0.1, -1.3, 0.5, 0.7])
        g = _gate_summary(vals)

        assert g["mean"] > 0,         f"mean={g['mean']:.4f}"
        assert g["initial_pass"]
        assert abs(g["rho"]) < 0.6,   f"rho={g['rho']:.4f}: V-shape prevents strong neg rho"
        assert not g["demotion"],     "rho not negative enough to trigger demotion"

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        assert report.loc["feature", "tier"] != "REJECTED", (
            f"V-shaped feature with strong recovery should not be REJECTED, "
            f"got {report.loc['feature', 'tier']}"
        )

    # ── 21. Noisy with overall negative mean — REJECTED ───────────────────────

    def test_case_21_noisy_negative_mean_rejected(self):
        """[0.10, -0.30, 0.35, -0.10, 0.25, -0.35, 0.05, -0.32]

        Alternating positive/negative with no trend. Mean = -0.040 (negative).
        ci_hi = +0.141 (CI spans zero, upper is positive) but mean < 0.
        The two-part gate requires BOTH mean > 0 AND ci_hi > 0 — mean gate kills it.

        Gates:
          mean = -0.040                → initial_pass = False (mean < 0)
          rho = -0.429                 → no rescue triggered (rho < 0.5)
          recent_mean = -0.207        → no rescue (recent not positive)
          => REJECTED(gate)

        Expected: REJECTED — mean is negative, no upward trend, recent negative.
        """
        vals = np.array([0.10, -0.30, 0.35, -0.10, 0.25, -0.35, 0.05, -0.32])
        g = _gate_summary(vals)

        assert g["mean"] < 0,          f"mean={g['mean']:.4f}: negative overall"
        assert g["ci_hi"] > 0,         f"ci_hi={g['ci_hi']:.4f}: CI spans zero (upper positive)"
        assert not g["initial_pass"],  "mean gate fails even though ci_hi > 0"
        assert g["recent_mean"] < 0,   f"recent_mean={g['recent_mean']:.4f}: recent also negative"
        assert not g["rescue"]

        report = _run_filter({"feature": vals, "anchor": ANCHOR})
        assert report.loc["feature", "tier"] == "REJECTED"

