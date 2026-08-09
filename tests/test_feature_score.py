"""Test oracle for feature_score() — eight vectors with independently-determined outcomes.

Run: conda run -n pred python -m pytest tests/test_feature_score.py -v

These test vectors exercise the scoring logic against known pathological patterns.
Expected outcomes are derived from the statistical properties of each vector,
NOT from the implementation's output.
"""
import numpy as np
import pytest
from scipy.stats import kendalltau

from pregame.analysis.feature_importance import (
    feature_score,
    feature_score_resid_sometimes_zero,
    EB_PRIORS,
)


# Use desub_MDA prior (d0=5.69, s0_sq=6.94e-06) for test vectors —
# moderate shrinkage, well-validated.
D0 = EB_PRIORS["desub_mda"]["d0"]
S0_SQ = EB_PRIORS["desub_mda"]["s0_sq"]
NULL = 0.0
TREND_ALPHA = 0.05
CI_ALPHA = 0.10


class TestCleanDecay:
    """clean_decay: monotonically declining, mean still above null.

    With level=mean(all)=0.095>0, the CI is entirely above null despite
    tau=-1.0. This hits the 'contradicts' branch: significant negative
    trend BUT evidence (CI) still positive → ACCEPT with NEEDS_SPEC flag.
    The feature IS declining but hasn't become harmful yet.
    """

    vals = np.array([0.20, 0.17, 0.14, 0.11, 0.08, 0.05, 0.02, -0.01])

    def test_tau_is_minus_one(self):
        tau, p = kendalltau(np.arange(8), self.vals, method='exact')
        assert tau == pytest.approx(-1.0, abs=1e-9)
        assert p < 0.01

    def test_contradicts_branch_accepts_with_flag(self):
        result = feature_score(self.vals, "desub_mda", null=NULL, d0=D0, s0_sq=S0_SQ,
                          trend_alpha=TREND_ALPHA, ci_alpha=CI_ALPHA)
        assert result["decision"] == "ACCEPT"
        assert result["flag"] == "NEEDS_SPECIFICATION"
        assert result["trend_tau"] == pytest.approx(-1.0, abs=1e-9)
        assert result["ci_lo"] > NULL

    def test_level_is_sample_mean(self):
        result = feature_score(self.vals, "desub_mda", null=NULL, d0=D0, s0_sq=S0_SQ,
                          trend_alpha=TREND_ALPHA, ci_alpha=CI_ALPHA)
        assert result["level"] == pytest.approx(np.mean(self.vals))


class TestCase01RegressionGuard:
    """Regression guard: level is sample mean (coherent with SE = sqrt(mod_var/n))."""

    vals = np.array([-1.2, -1.0, -0.5, -0.8, -0.2, 0.01, -0.1, 0.05])

    def test_level_is_sample_mean(self):
        result = feature_score(self.vals, "desub_mda", null=NULL, d0=D0, s0_sq=S0_SQ,
                          trend_alpha=TREND_ALPHA, ci_alpha=CI_ALPHA)
        # mean(-1.2, -1.0, -0.5, -0.8, -0.2, 0.01, -0.1, 0.05) = -0.4675
        assert result["level"] == pytest.approx(np.mean(self.vals))

    def test_level_is_bounded_by_data(self):
        result = feature_score(self.vals, "desub_mda", null=NULL, d0=D0, s0_sq=S0_SQ,
                          trend_alpha=TREND_ALPHA, ci_alpha=CI_ALPHA)
        assert result["level"] >= self.vals.min()
        assert result["level"] <= self.vals.max()


class TestUShape:
    """U-shape: tau~0, non-monotonic — must flag NEEDS_SPECIFICATION."""

    vals = np.array([0.20, 0.10, 0.02, -0.05, -0.05, 0.02, 0.10, 0.20])

    def test_tau_is_zero(self):
        # Ties in y: use default (asymptotic) method
        tau, p = kendalltau(np.arange(8), self.vals)
        assert abs(tau) < 0.01

    def test_flags_needs_specification(self):
        result = feature_score(self.vals, "desub_mda", null=NULL, d0=D0, s0_sq=S0_SQ,
                          trend_alpha=TREND_ALPHA, ci_alpha=CI_ALPHA)
        # tau=0, not significant -> must NOT silently reject
        assert result["decision"] != "REJECT"
        # Either decision or flag should indicate needs specification
        assert (result["decision"] == "NEEDS_SPECIFICATION" or
                result["flag"] == "NEEDS_SPECIFICATION" or
                result["decision"] == "ACCEPT")


class TestInvertedU:
    """Inverted U: tau~0, non-monotonic — must flag NEEDS_SPECIFICATION."""

    vals = np.array([-0.05, 0.05, 0.15, 0.20, 0.20, 0.15, 0.05, -0.05])

    def test_tau_is_zero(self):
        tau, p = kendalltau(np.arange(8), self.vals)
        assert abs(tau) < 0.01

    def test_flags_needs_specification(self):
        result = feature_score(self.vals, "desub_mda", null=NULL, d0=D0, s0_sq=S0_SQ,
                          trend_alpha=TREND_ALPHA, ci_alpha=CI_ALPHA)
        assert result["decision"] != "REJECT"
        assert (result["decision"] == "NEEDS_SPECIFICATION" or
                result["flag"] == "NEEDS_SPECIFICATION" or
                result["decision"] == "ACCEPT")


class TestStepChange:
    """Step change: significant positive trend but mean=0 (exactly at null).

    With level=mean(all)=0.0, the CI spans null despite tau>0+significant.
    This hits the 'contradicts' branch: positive trend BUT level not > null.
    NEEDS_SPECIFICATION — the feature recently became positive but overall
    evidence across all 8 folds is inconclusive.
    """

    vals = np.array([-0.10, -0.10, -0.10, -0.10, 0.10, 0.10, 0.10, 0.10])

    def test_tau_and_significance(self):
        tau, p = kendalltau(np.arange(8), self.vals)
        assert tau == pytest.approx(0.756, abs=0.01)
        assert p < 0.05

    def test_contradicts_branch_needs_spec(self):
        result = feature_score(self.vals, "desub_mda", null=NULL, d0=D0, s0_sq=S0_SQ,
                          trend_alpha=TREND_ALPHA, ci_alpha=CI_ALPHA)
        assert result["decision"] == "NEEDS_SPECIFICATION"
        assert result["flag"] == "NEEDS_SPECIFICATION"
        assert result["level"] == pytest.approx(0.0)


class TestVarianceRegimeShift:
    """Variance regime shift: stable first half, explosive second half."""

    vals = np.array([0.05, 0.05, 0.05, 0.05, 0.30, -0.20, 0.25, -0.15])

    def test_tau_near_zero(self):
        tau, p = kendalltau(np.arange(8), self.vals)
        assert abs(tau) < 0.15
        assert p > 0.05

    def test_flags_instability(self):
        result = feature_score(self.vals, "desub_mda", null=NULL, d0=D0, s0_sq=S0_SQ,
                          trend_alpha=TREND_ALPHA, ci_alpha=CI_ALPHA)
        # Must not decide on trend alone — flag instability, override to NEEDS_SPECIFICATION
        assert result["flag"] == "INSTABILITY"
        assert result["decision"] == "NEEDS_SPECIFICATION"


class TestOscillationNetPositive:
    """Oscillation with net-positive level: tau insignificant, decision falls to CI."""

    vals = np.array([0.05, -0.03, 0.06, -0.02, 0.07, -0.01, 0.08, 0.01])

    def test_tau_insignificant(self):
        tau, p = kendalltau(np.arange(8), self.vals, method='exact')
        assert abs(tau) < 0.35
        assert p > 0.20

    def test_decision_from_level_not_trend(self):
        result = feature_score(self.vals, "desub_mda", null=NULL, d0=D0, s0_sq=S0_SQ,
                          trend_alpha=TREND_ALPHA, ci_alpha=CI_ALPHA)
        # level = mean(all) = mean(0.05, -0.03, 0.06, -0.02, 0.07, -0.01, 0.08, 0.01)
        assert result["level"] == pytest.approx(np.mean(self.vals))
        # With insignificant trend, decision comes from moderated-t CI
        # Key: NOT rejected purely because trend is weak
        assert result["decision"] != "REJECT"


class TestMarginalTrend:
    """Marginal trend: tau=0.500, p=0.109 — NOT significant at trend_alpha=0.05.

    Must fall through to CI-based path, not the hard trend-driven ACCEPT branch.
    This is the vector that tests whether trend_alpha actually gates the trend
    decision — if magnitude thresholds were substituted for the p-value, this
    would incorrectly fire the trend-ACCEPT branch (tau=0.5 is large).
    """

    vals = np.array([0.08, 0.01, 0.06, 0.04, 0.09, 0.07, 0.12, 0.10])

    def test_tau_and_p_value(self):
        tau, p = kendalltau(np.arange(8), self.vals, method='exact')
        assert tau == pytest.approx(0.500, abs=0.001)
        assert p == pytest.approx(0.1087, abs=0.001)
        # NOT significant at 0.05
        assert p > 0.05

    def test_falls_through_to_ci_path(self):
        result = feature_score(self.vals, "desub_mda", null=NULL, d0=D0, s0_sq=S0_SQ,
                          trend_alpha=TREND_ALPHA, ci_alpha=CI_ALPHA)
        # p=0.109 > trend_alpha=0.05 → trend is NOT significant
        # Falls to CI path: level=mean(all)=0.07125, CI based on EB-moderated variance
        assert result["level"] == pytest.approx(np.mean(self.vals))
        # With positive mean and EB shrinkage, CI_lo should be > null → ACCEPT via CI
        assert result["decision"] == "ACCEPT"
        assert result["flag"] == "NEEDS_SPECIFICATION"

    def test_would_differ_at_lower_trend_alpha(self):
        """If trend_alpha were 0.15, the same vector would hit the hard ACCEPT branch
        (no flag). This proves trend_alpha is the actual gate."""
        result_loose = feature_score(self.vals, "desub_mda", null=NULL, d0=D0, s0_sq=S0_SQ,
                                     trend_alpha=0.15, ci_alpha=CI_ALPHA)
        assert result_loose["decision"] == "ACCEPT"
        assert result_loose["flag"] is None  # hard trend path, no flag


class TestSignificantDeclineStillPositive:
    """Significant decline but level still above null — contradicts branch.

    tau=-1.0 (p<0.001) means trend IS significant and negative, but ci_lo is
    still above null (level=0.26, CI entirely positive). This hits the
    "significant trend contradicts level/CI" else branch. Must carry
    flag=NEEDS_SPECIFICATION to reduce confidence — the decline is real evidence
    even though null hasn't been crossed yet.
    """

    vals = np.array([0.50, 0.45, 0.40, 0.35, 0.30, 0.28, 0.26, 0.25])

    def test_tau_and_significance(self):
        tau, p = kendalltau(np.arange(8), self.vals, method='exact')
        assert tau == pytest.approx(-1.0, abs=1e-9)
        assert p < 0.001

    def test_contradicts_branch_flags_needs_spec(self):
        result = feature_score(self.vals, "desub_mda", null=NULL, d0=D0, s0_sq=S0_SQ,
                          trend_alpha=TREND_ALPHA, ci_alpha=CI_ALPHA)
        # Trend is significant (p<0.05) and negative, but ci_lo > null
        # -> hits the "contradicts" else branch
        assert result["decision"] == "ACCEPT"
        assert result["flag"] == "NEEDS_SPECIFICATION"
        # Level = mean(all) = mean(0.50,...,0.25) = 0.34875
        assert result["level"] == pytest.approx(np.mean(self.vals))
        # CI should be entirely above null
        assert result["ci_lo"] > NULL


class TestCombinerUnanimousAccept:
    """All 4 EB tests ACCEPT with full confidence → ACCEPTED."""

    def test_unanimous_accept(self):
        from pregame.analysis.feature_importance import combine_test_scores
        scores = {
            "sfi":       {"decision": "ACCEPT", "flag": None},
            "desub_mda": {"decision": "ACCEPT", "flag": None},
            "pca_mda":   {"decision": "ACCEPT", "flag": None},
            "resid_mda": {"decision": "ACCEPT", "flag": None},
        }
        result = combine_test_scores(scores)
        assert result["tier"] == "ACCEPTED"
        assert result["accept_frac"] == 1.0
        assert result["total_available"] == 4.0


class TestCombinerUnanimousReject:
    """All tests REJECT → REJECTED."""

    def test_unanimous_reject(self):
        from pregame.analysis.feature_importance import combine_test_scores
        scores = {
            "sfi":       {"decision": "REJECT", "flag": None},
            "desub_mda": {"decision": "REJECT", "flag": None},
            "pca_mda":   {"decision": "REJECT", "flag": None},
            "resid_mda": {"decision": "REJECT", "flag": None},
            "mdi":       {"decision": "REJECT", "flag": None},
            "cfi_mda":   {"decision": "REJECT", "flag": None},
        }
        result = combine_test_scores(scores)
        assert result["tier"] == "REJECTED"
        assert result["accept_votes"] == 0.0


class TestCombinerFlagReducesWeight:
    """flag=NEEDS_SPEC halves effective weight — 3 full ACCEPT + 1 flagged REJECT
    should still ACCEPT (3.0 accept vs 0.5 reject out of 3.5 available)."""

    def test_flagged_reject_underweighted(self):
        from pregame.analysis.feature_importance import combine_test_scores
        scores = {
            "sfi":       {"decision": "ACCEPT", "flag": None},       # +1.0
            "desub_mda": {"decision": "ACCEPT", "flag": None},       # +1.0
            "pca_mda":   {"decision": "ACCEPT", "flag": None},       # +1.0
            "resid_mda": {"decision": "REJECT", "flag": "NEEDS_SPECIFICATION"},  # -0.5
        }
        result = combine_test_scores(scores)
        assert result["tier"] == "ACCEPTED"
        assert result["accept_votes"] == pytest.approx(3.0)
        assert result["reject_votes"] == pytest.approx(0.5)
        assert result["total_available"] == pytest.approx(3.5)


class TestCombinerInstabilityAbstains:
    """flag=INSTABILITY means the test abstains entirely — doesn't count toward
    available votes."""

    def test_instability_abstains(self):
        from pregame.analysis.feature_importance import combine_test_scores
        scores = {
            "sfi":       {"decision": "ACCEPT", "flag": None},        # +1.0
            "desub_mda": {"decision": "REJECT", "flag": None},        # -1.0
            "pca_mda":   {"decision": "ACCEPT", "flag": "INSTABILITY"},  # abstain
            "resid_mda": {"decision": "ACCEPT", "flag": None},        # +1.0
        }
        result = combine_test_scores(scores)
        # pca_mda abstains: available = 1+1+1 = 3, accept = 2, reject = 1
        assert result["total_available"] == pytest.approx(3.0)
        assert result["accept_votes"] == pytest.approx(2.0)
        assert result["abstain_votes"] == pytest.approx(1.0)
        assert result["tier"] == "ACCEPTED"  # 2/3 > 0.5


class TestCombinerCLTTestsHalfWeight:
    """MDI/CFI_MDA carry base weight 0.5 — calibrated z-test but near-zero selectivity."""

    def test_clt_tests_half_weight(self):
        from pregame.analysis.feature_importance import combine_test_scores
        # 2 EB reject (2.0) vs 2 CLT accept (1.0) → reject wins
        scores = {
            "sfi":       {"decision": "REJECT", "flag": None},   # -1.0
            "desub_mda": {"decision": "REJECT", "flag": None},   # -1.0
            "mdi":       {"decision": "ACCEPT", "flag": None},   # +0.5
            "cfi_mda":   {"decision": "ACCEPT", "flag": None},   # +0.5
        }
        result = combine_test_scores(scores)
        assert result["reject_votes"] == pytest.approx(2.0)
        assert result["accept_votes"] == pytest.approx(1.0)
        assert result["total_available"] == pytest.approx(3.0)
        assert result["tier"] == "REJECTED"

    def test_mdi_half_weight_value(self):
        """MDI vote contributes weight 0.5."""
        from pregame.analysis.feature_importance import combine_test_scores
        scores = {
            "mdi": {"decision": "ACCEPT", "flag": None},
        }
        result = combine_test_scores(scores)
        assert result["accept_votes"] == pytest.approx(0.5)
        assert result["total_available"] == pytest.approx(0.5)


class TestCombinerSplitVoteNeedsSpec:
    """Even split between accept and reject → NEEDS SPECIFICATION."""

    def test_even_split(self):
        from pregame.analysis.feature_importance import combine_test_scores
        scores = {
            "sfi":       {"decision": "ACCEPT", "flag": None},   # +1.0
            "desub_mda": {"decision": "REJECT", "flag": None},   # -1.0
            "pca_mda":   {"decision": "NEEDS_SPECIFICATION", "flag": "NEEDS_SPECIFICATION"},  # 0
        }
        result = combine_test_scores(scores)
        # accept=1.0, reject=1.0, needs_spec=0.5 (counted in available but not in either)
        # accept_frac = 1.0/2.5 = 0.4 < 0.5; reject_frac = 1.0/2.5 = 0.4 < 0.5
        assert result["tier"] == "NEEDS SPECIFICATION"


class TestCombinerResidSometimesZeroAbstains:
    """resid_MDA sometimes-zero features (decision=None) abstain from voting."""

    def test_sometimes_zero_abstains(self):
        from pregame.analysis.feature_importance import combine_test_scores
        scores = {
            "sfi":       {"decision": "ACCEPT", "flag": None},
            "desub_mda": {"decision": "ACCEPT", "flag": None},
            "resid_mda": {"decision": None, "fold_nonzero_frac": 0.375, "flag": None},
        }
        result = combine_test_scores(scores)
        assert result["total_available"] == pytest.approx(2.0)
        assert result["abstain_votes"] == pytest.approx(1.0)
        assert result["tier"] == "ACCEPTED"


class TestCombinerIsolatedAcceptDiluted:
    """1 ACCEPT (full weight) + 3 NEEDS_SPECIFICATION (no opposition).
    Intended: ACCEPTED — uncertainty without opposition shouldn't block."""

    def test_unopposed_accept_passes(self):
        from pregame.analysis.feature_importance import combine_test_scores
        scores = {
            "sfi":       {"decision": "ACCEPT", "flag": None},                    # +1.0
            "desub_mda": {"decision": "NEEDS_SPECIFICATION", "flag": "NEEDS_SPECIFICATION"},  # 0
            "pca_mda":   {"decision": "NEEDS_SPECIFICATION", "flag": "NEEDS_SPECIFICATION"},  # 0
            "resid_mda": {"decision": "NEEDS_SPECIFICATION", "flag": "NEEDS_SPECIFICATION"},  # 0
        }
        result = combine_test_scores(scores)
        # accept=1.0, reject=0.0 → unopposed accept → ACCEPTED
        assert result["tier"] == "ACCEPTED"
        assert result["accept_votes"] == pytest.approx(1.0)
        assert result["reject_votes"] == pytest.approx(0.0)

    def test_isolated_reject_symmetric(self):
        """Symmetric case: 1 REJECT + 3 NEEDS_SPEC → REJECTED."""
        from pregame.analysis.feature_importance import combine_test_scores
        scores = {
            "sfi":       {"decision": "REJECT", "flag": None},
            "desub_mda": {"decision": "NEEDS_SPECIFICATION", "flag": "NEEDS_SPECIFICATION"},
            "pca_mda":   {"decision": "NEEDS_SPECIFICATION", "flag": "NEEDS_SPECIFICATION"},
            "resid_mda": {"decision": "NEEDS_SPECIFICATION", "flag": "NEEDS_SPECIFICATION"},
        }
        result = combine_test_scores(scores)
        assert result["tier"] == "REJECTED"


class TestCombinerAllAbstain:
    """If all tests abstain, tier = UNKNOWN."""

    def test_all_abstain(self):
        from pregame.analysis.feature_importance import combine_test_scores
        scores = {
            "sfi":       {"decision": "ACCEPT", "flag": "INSTABILITY"},
            "desub_mda": {"decision": None, "flag": None},
        }
        result = combine_test_scores(scores)
        assert result["tier"] == "UNKNOWN"
        assert result["total_available"] == 0.0


class TestFeatureScoreCLT:
    """Oracle tests for CLT-based MDI/CFI_MDA scoring (de Prado z-test)."""

    def test_clearly_above_null_accepts(self):
        """Feature with mean well above empirical null → ACCEPT."""
        from pregame.analysis.feature_importance import feature_score_clt
        rng = np.random.default_rng(42)
        # 80 trees, mean~0.05 (vs empirical null~0.029), std~0.01
        vals = rng.normal(0.05, 0.01, size=80)
        result = feature_score_clt(vals, null=0.029, alpha=0.05)
        assert result["decision"] == "ACCEPT"
        assert result["flag"] is None
        assert result["z_stat"] > 2.0
        assert result["p_value"] < 0.05

    def test_at_null_needs_spec(self):
        """Feature centered exactly at empirical null → NEEDS_SPECIFICATION."""
        from pregame.analysis.feature_importance import feature_score_clt
        rng = np.random.default_rng(99)
        null = 0.029
        # Mean exactly at null, moderate variance
        vals = rng.normal(null, 0.015, size=70)
        result = feature_score_clt(vals, null=null, alpha=0.05)
        # With mean≈null, z≈0 → not significant
        assert result["decision"] == "NEEDS_SPECIFICATION"
        assert result["flag"] == "NEEDS_SPECIFICATION"

    def test_below_null_rejects(self):
        """Feature significantly below empirical null → REJECT."""
        from pregame.analysis.feature_importance import feature_score_clt
        rng = np.random.default_rng(7)
        null = 0.029
        # Mean at 0.010, well below null
        vals = rng.normal(0.010, 0.005, size=90)
        result = feature_score_clt(vals, null=null, alpha=0.05)
        assert result["decision"] == "REJECT"
        assert result["flag"] is None
        assert result["z_stat"] < -2.0

    def test_insufficient_trees_needs_spec(self):
        """Fewer than 10 valid trees → insufficient data, flag."""
        from pregame.analysis.feature_importance import feature_score_clt
        vals = np.array([0.01, 0.02, np.nan, np.nan, np.nan, np.nan, np.nan, 0.015])
        result = feature_score_clt(vals, null=0.029, alpha=0.05)
        assert result["decision"] == "NEEDS_SPECIFICATION"
        assert result["n"] == 3

    def test_n_counts_only_valid(self):
        """N should count only non-NaN values."""
        from pregame.analysis.feature_importance import feature_score_clt
        rng = np.random.default_rng(55)
        vals = np.full(100, np.nan)
        vals[:60] = rng.normal(0.05, 0.01, size=60)
        result = feature_score_clt(vals, null=0.029, alpha=0.05)
        assert result["n"] == 60

    def test_ci_width_scales_with_sqrt_n(self):
        """CI should narrow as N increases (basic sanity)."""
        from pregame.analysis.feature_importance import feature_score_clt
        rng = np.random.default_rng(12)
        vals_small = rng.normal(0.03, 0.01, size=30)
        vals_large = rng.normal(0.03, 0.01, size=200)
        r_small = feature_score_clt(vals_small, null=0.029, alpha=0.05)
        r_large = feature_score_clt(vals_large, null=0.029, alpha=0.05)
        ci_width_small = r_small["ci_hi"] - r_small["ci_lo"]
        ci_width_large = r_large["ci_hi"] - r_large["ci_lo"]
        assert ci_width_large < ci_width_small

    def test_alpha_controls_rejection(self):
        """Marginal feature: significant at alpha=0.10, not at alpha=0.01."""
        from pregame.analysis.feature_importance import feature_score_clt
        rng = np.random.default_rng(33)
        null = 0.029
        # Mean slightly above null, such that p is between 0.01 and 0.10
        vals = rng.normal(null + 0.003, 0.015, size=60)
        r_strict = feature_score_clt(vals, null=null, alpha=0.01)
        r_loose = feature_score_clt(vals, null=null, alpha=0.10)
        # At alpha=0.01 should not accept, at alpha=0.10 should accept
        assert r_strict["decision"] == "NEEDS_SPECIFICATION"
        assert r_loose["decision"] == "ACCEPT"


class TestResidMdaDispatch:
    """Oracle tests for resid_mda_dispatch routing."""

    def test_always_zero(self):
        from pregame.analysis.feature_importance import resid_mda_dispatch
        vals = np.zeros(8)
        result = resid_mda_dispatch(vals)
        assert result["path"] == "always_zero"
        assert result["decision"] == "NO_UNIQUE_SIGNAL"

    def test_always_nonzero(self):
        from pregame.analysis.feature_importance import resid_mda_dispatch
        vals = np.array([0.001, 0.002, 0.0015, 0.0018, 0.0012, 0.0022, 0.0019, 0.0025])
        result = resid_mda_dispatch(vals)
        assert result["path"] == "always_nonzero"
        assert result["decision"] in ("ACCEPT", "REJECT", "NEEDS_SPECIFICATION")
        assert "mod_t" in result  # EB path produces moderated t
        assert result["mod_t"] is not None

    def test_sometimes_zero(self):
        from pregame.analysis.feature_importance import resid_mda_dispatch
        vals = np.array([0.0, 0.0, 0.001, 0.0, 0.002, 0.0, 0.0015, 0.0])
        result = resid_mda_dispatch(vals)
        assert result["path"] == "sometimes_zero"
        assert result["fold_nonzero_frac"] == 3 / 8
        assert result["level"] == pytest.approx(0.0015)  # median of nonzero
        assert result["decision"] is None  # no hard decision from this path

    def test_single_nonzero_fold(self):
        from pregame.analysis.feature_importance import resid_mda_dispatch
        vals = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5e-7])
        result = resid_mda_dispatch(vals)
        assert result["path"] == "sometimes_zero"
        assert result["fold_nonzero_frac"] == 1 / 8
        assert result["n_nonzero_folds"] == 1


class TestResidSometimesZero:
    """Test the sometimes-zero scoring path."""

    def test_basic(self):
        vals = np.array([0.0, 0.0, 0.0, 0.001, 0.002, 0.0, 0.0015, 0.0])
        result = feature_score_resid_sometimes_zero(vals)
        assert result["fold_nonzero_frac"] == 3 / 8
        assert result["n_nonzero_folds"] == 3
        # median of nonzero: median(0.001, 0.002, 0.0015) = 0.0015
        assert result["level"] == pytest.approx(0.0015)

    def test_all_zero(self):
        vals = np.zeros(8)
        result = feature_score_resid_sometimes_zero(vals)
        assert result["fold_nonzero_frac"] == 0.0
        assert result["decision"] == "NO_UNIQUE_SIGNAL"

    def test_single_nonzero(self):
        vals = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.005])
        result = feature_score_resid_sometimes_zero(vals)
        assert result["fold_nonzero_frac"] == 1 / 8
        assert result["level"] == pytest.approx(0.005)
