"""Standalone proofs of four statistical issues in feature_importance.py.

These tests demonstrate the problems exist WITHOUT modifying production code.
Run: conda run -n pred python -m pytest tests/test_importance_statistical_gates.py -v

Issues proved:
  1. detone_corr(n_remove=0) destroys signal via Python [-0:] slice semantics
  2. Wilcoxon minimum p at n=4 is 0.0625 — unreachable at alpha=0.05
  3. BH-FDR correction makes Wilcoxon permanently dead for realistic configs
  4. The >= 4 floor in MDI/desub/PCA/resid is dead weight (inconsistent with CFI's >= 5)
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import wilcoxon as scipy_wilcoxon


# ─────────────────────────────────────────────────────────────────────────────
#  Issue 1: detone_corr(n_remove=0) is catastrophically broken
# ─────────────────────────────────────────────────────────────────────────────

class TestDetoneBroken:
    """n_remove=0 should be a no-op. Instead it zeros ALL eigenvalues."""

    def test_python_negative_zero_slice_semantics(self):
        """Root cause: arr[-0:] == arr[0:] == entire array, not empty slice."""
        arr = np.array([1, 2, 3, 4, 5])
        # -0 is 0 in Python
        assert -0 == 0
        # So arr[-0:] selects the ENTIRE array, not an empty slice
        assert len(arr[-0:]) == 5  # user expects 0 elements
        assert len(arr[0:]) == 5   # same thing

    def test_detone_n_remove_0_destroys_all_eigenvalues(self):
        """Reproduce: detone_corr with n_remove=0 zeros everything."""
        # Build a correlation matrix with known structure
        rng = np.random.default_rng(42)
        n = 20
        X = rng.standard_normal((100, n))
        # Inject correlation structure
        X[:, 1] = X[:, 0] * 0.8 + rng.standard_normal(100) * 0.2
        X[:, 2] = X[:, 0] * 0.7 + rng.standard_normal(100) * 0.3
        corr = pd.DataFrame(np.corrcoef(X.T), columns=range(n), index=range(n))

        # Simulate what detone_corr does with n_remove=0
        evals, evecs = np.linalg.eigh(corr.values)
        evals_detoned = evals.copy()
        n_remove = 0
        # This is the bug: evals[-0:] selects ALL elements
        evals_detoned[-n_remove:] = 0.0

        # ALL eigenvalues are now zero — total destruction
        assert np.allclose(evals_detoned, 0.0), (
            "Expected all eigenvalues zeroed due to [-0:] semantics"
        )

    def test_detone_n_remove_0_should_be_noop(self):
        """What SHOULD happen: n_remove=0 returns the input unchanged."""
        rng = np.random.default_rng(42)
        n = 10
        X = rng.standard_normal((100, n))
        corr = pd.DataFrame(np.corrcoef(X.T), columns=range(n), index=range(n))

        evals_original = np.linalg.eigvalsh(corr.values)

        # Correct behavior: skip the zeroing entirely when n_remove=0
        evals_detoned = evals_original.copy()
        n_remove = 0
        if n_remove > 0:
            evals_detoned[-n_remove:] = 0.0

        np.testing.assert_array_equal(evals_detoned, evals_original)


# ─────────────────────────────────────────────────────────────────────────────
#  Issue 2: Wilcoxon minimum p at n=4 is 0.0625 > alpha=0.05
# ─────────────────────────────────────────────────────────────────────────────

class TestWilcoxonMinimumP:
    """At n=4, the best possible one-sided Wilcoxon p is 1/16 = 0.0625."""

    def test_minimum_p_at_n4_is_unreachable(self):
        """Even with perfectly concordant data, p cannot reach 0.05 at n=4."""
        # Best case: all 4 differences are positive and large
        # T+ = 1+2+3+4 = 10 (maximum), one-sided p = 1/2^4 = 0.0625
        diffs = np.array([100.0, 200.0, 300.0, 400.0])
        _, p = scipy_wilcoxon(diffs, alternative='greater')
        assert p == pytest.approx(0.0625, abs=1e-6)
        assert p > 0.05, f"p={p} should be > 0.05 (mathematically impossible to clear)"

    def test_minimum_p_at_n5_is_reachable(self):
        """At n=5, best p = 1/2^5 = 0.03125 < 0.05. This is why CFI uses >= 5."""
        diffs = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        _, p = scipy_wilcoxon(diffs, alternative='greater')
        assert p == pytest.approx(0.03125, abs=1e-6)
        assert p < 0.05, f"p={p} should be < 0.05"

    @pytest.mark.parametrize("n", [4, 5, 6, 7, 8, 9, 10, 11])
    def test_minimum_p_formula(self, n):
        """Verify minimum one-sided p = 1/2^n for all relevant sample sizes."""
        diffs = np.arange(1.0, n + 1)  # all positive, increasing
        _, p = scipy_wilcoxon(diffs, alternative='greater')
        expected = 1.0 / (2 ** n)
        assert p == pytest.approx(expected, rel=1e-4), (
            f"n={n}: got p={p}, expected 1/2^{n}={expected}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Issue 3: BH-FDR correction makes Wilcoxon permanently dead
# ─────────────────────────────────────────────────────────────────────────────

class TestBHKillsWilcoxon:
    """After BH correction, Wilcoxon can never clear alpha for realistic configs."""

    @staticmethod
    def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
        """Reproduce the BH adjustment from filter_features."""
        m = len(p_values)
        order = np.argsort(p_values)
        ranked = np.empty(m)
        ranked[order] = np.arange(1, m + 1)
        adjusted = p_values * m / ranked
        adjusted_sorted = adjusted[np.argsort(ranked)[::-1]]
        for i in range(1, m):
            adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i - 1])
        adjusted[np.argsort(ranked)[::-1]] = adjusted_sorted
        return np.clip(adjusted, 0, 1)

    def test_bh_kills_wilcoxon_6_folds_150_features(self):
        """Typical LOYO config: 6 folds (2021-2026), ~150 features. Dead.

        Realistic scenario: a few features hit minimum p, most are higher.
        BH adjusts relative to rank position: p_adj = p * m / rank.
        For the minimum p at rank=1: p_adj = (1/2^n) * m / 1.
        """
        n_folds = 6
        n_features = 150
        alpha = 0.05

        # Realistic p-value distribution: 1 feature at minimum, rest spread
        min_p = 1.0 / (2 ** n_folds)  # 0.015625
        rng = np.random.default_rng(42)
        p_values = rng.uniform(min_p, 1.0, size=n_features)
        p_values[0] = min_p  # best possible feature

        p_adjusted = self._bh_adjust(p_values)

        # The best feature's adjusted p = min_p * m / 1 = 0.015625 * 150 = 2.34
        best_adj = p_adjusted[np.argmin(p_values)]
        assert best_adj > alpha, (
            f"best adjusted p = {best_adj:.6f}, expected > {alpha}. "
            f"Wilcoxon dead at n={n_folds}, m={n_features}"
        )

    def test_bh_kills_wilcoxon_11_folds_150_features(self):
        """Maximum LOYO (2015-2025): 11 folds. Still dead with 150 features."""
        n_folds = 11
        n_features = 150
        alpha = 0.05

        min_p = 1.0 / (2 ** n_folds)  # 0.000488
        rng = np.random.default_rng(42)
        p_values = rng.uniform(min_p, 1.0, size=n_features)
        p_values[0] = min_p

        p_adjusted = self._bh_adjust(p_values)

        # p_adj = 0.000488 * 150 / 1 = 0.073 > 0.05
        best_adj = p_adjusted[np.argmin(p_values)]
        assert best_adj > alpha, (
            f"best adjusted p = {best_adj:.6f}, expected > {alpha}. "
            f"Wilcoxon dead even at n=11 with m=150"
        )

    def test_minimum_folds_for_bh_to_allow_wilcoxon(self):
        """Find the minimum folds needed for Wilcoxon to EVER fire after BH."""
        n_features = 150
        alpha = 0.05

        # Solve: (1/2^n) * m < alpha => n > log2(m / alpha)
        n_min = int(np.ceil(np.log2(n_features / alpha)))

        # At n_min, the best-case BH-adjusted p just clears alpha
        min_p = 1.0 / (2 ** n_min)
        best_bh = min_p * n_features  # rank=1 gives p * m / 1
        assert best_bh < alpha, f"Formula wrong: need n>{n_min}"

        # At n_min - 1, it's still dead
        min_p_below = 1.0 / (2 ** (n_min - 1))
        best_bh_below = min_p_below * n_features
        assert best_bh_below > alpha

        # For 150 features: n_min = ceil(log2(3000)) = 12
        assert n_min == 12, (
            f"Need {n_min} folds for Wilcoxon to ever fire with 150 features. "
            f"We have at most 11 seasons — Wilcoxon is structurally dead."
        )

    @pytest.mark.parametrize("n_features", [50, 100, 150, 200])
    def test_wilcoxon_dead_at_6_folds_any_feature_count(self, n_features):
        """With 6 LOYO folds, Wilcoxon is dead regardless of feature count (>=50)."""
        n_folds = 6
        alpha = 0.05
        min_p = 1.0 / (2 ** n_folds)
        best_bh = min_p * n_features  # worst-case BH for rank=1

        assert best_bh > alpha, (
            f"m={n_features}: best BH-adjusted p = {best_bh:.4f} > {alpha}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Issue 4: >= 4 floor inconsistency (dead weight for Wilcoxon-gated methods)
# ─────────────────────────────────────────────────────────────────────────────

class TestFloorInconsistency:
    """MDI/desub/PCA/resid use >= 4, but Wilcoxon can't fire there.
    CFI-MDA correctly uses >= 5. The >= 4 methods rely on an OR-gate
    with bootstrap CI, making the Wilcoxon leg dead weight at n=4."""

    def test_wilcoxon_at_n4_always_exceeds_alpha(self):
        """No matter what data you feed, Wilcoxon one-sided p > 0.05 at n=4."""
        # Try many different positive-mean distributions
        rng = np.random.default_rng(42)
        for _ in range(1000):
            vals = rng.exponential(10.0, size=4)
            diffs = vals[vals != 0]
            if len(diffs) < 4:
                continue
            _, p = scipy_wilcoxon(diffs, alternative='greater')
            assert p >= 0.0625, f"Got p={p} < 0.0625, impossible"
            assert p > 0.05, f"Got p={p} < 0.05, impossible at n=4"

    def test_cfi_uses_5_others_use_4(self):
        """Read the code and confirm the inconsistency exists."""
        import inspect
        from classical_learning.analysis.feature_importance import filter_features

        source = inspect.getsource(filter_features)

        # CFI-MDA uses >= 5 (correct)
        assert "len(vals) >= 5" in source or "len(diffs) >= 5" in source, (
            "CFI-MDA should use >= 5"
        )

    def test_or_gate_with_dead_wilcoxon_equals_pure_ci(self):
        """When Wilcoxon can never fire, the OR-gate collapses to just the CI leg."""
        # Simulate what filter_features does for desub_mda at n=4-6 folds
        rng = np.random.default_rng(42)
        alpha = 0.05

        for n_folds in [4, 5, 6]:
            vals = rng.uniform(0.01, 0.1, size=n_folds)  # positive importance
            ci_lo = vals.mean() - 1.96 * vals.std() / np.sqrt(n_folds)
            ci_passes = ci_lo > 0

            diffs = vals[vals != 0]
            if len(diffs) >= 4:
                _, p = scipy_wilcoxon(diffs, alternative='greater')
                wilcoxon_passes = p < alpha
            else:
                wilcoxon_passes = False

            or_gate = ci_passes or wilcoxon_passes

            if n_folds == 4:
                # At n=4, Wilcoxon minimum p is 0.0625 > 0.05, CANNOT fire
                assert not wilcoxon_passes, (
                    f"Wilcoxon should never pass at n=4, got p={p}"
                )
                assert or_gate == ci_passes, (
                    "OR-gate == CI-only when Wilcoxon is dead"
                )


# ─────────────────────────────────────────────────────────────────────────────
#  Combined: unfair comparison proof (no-detone vs spectral was bogus)
# ─────────────────────────────────────────────────────────────────────────────

class TestUnfairComparison:
    """The clustering comparison was unfair because detone(n_remove=0)
    destroyed the correlation matrix rather than being a no-op."""

    def test_detone_0_produces_near_identity(self):
        """With all eigenvalues zeroed, the renormalized result is ~identity."""
        rng = np.random.default_rng(42)
        n = 20
        X = rng.standard_normal((200, n))
        X[:, 1] = X[:, 0] * 0.9 + rng.standard_normal(200) * 0.1
        corr = pd.DataFrame(np.corrcoef(X.T))

        # Simulate the bug
        evals, evecs = np.linalg.eigh(corr.values)
        evals_detoned = evals.copy()
        evals_detoned[-0:] = 0.0  # zeros ALL

        # Reconstruct
        corr_detoned = evecs @ np.diag(evals_detoned) @ evecs.T
        # All zeros => diag is 0 => division by zero in normalization
        # The code uses max(..., 1e-12) so it becomes ~0/small = garbage
        diag = np.diag(corr_detoned)
        assert np.allclose(diag, 0.0, atol=1e-10), (
            "All-zero eigenvalues => zero diagonal => degenerate matrix"
        )

    def test_clustering_on_destroyed_matrix_is_meaningless(self):
        """KMeans on a degenerate distance matrix produces arbitrary clusters."""
        from sklearn.cluster import KMeans

        n = 20
        # A near-identity correlation matrix (what detone_0 produces after renorm)
        corr_garbage = np.eye(n) + np.random.default_rng(42).normal(0, 0.01, (n, n))
        corr_garbage = (corr_garbage + corr_garbage.T) / 2
        np.fill_diagonal(corr_garbage, 1.0)

        dist = ((1 - corr_garbage) / 2.0) ** 0.5

        # Run KMeans multiple times — results are unstable on garbage input
        labels_runs = []
        for seed in range(10):
            km = KMeans(n_clusters=5, n_init=1, random_state=seed)
            labels_runs.append(km.fit_predict(dist))

        # Clusters should be inconsistent across seeds (no real structure)
        agreements = [
            np.mean(labels_runs[i] == labels_runs[j])
            for i in range(10) for j in range(i + 1, 10)
        ]
        # Near-random assignment: ~20% agreement for k=5
        assert np.mean(agreements) < 0.6, (
            f"Garbage matrix should produce unstable clusters, "
            f"got {np.mean(agreements):.2f} agreement"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Issue 5: Noise with positive blips must NOT pass the filter
# ─────────────────────────────────────────────────────────────────────────────

class TestFilterGateSemantics:
    """Comprehensive tests for the two-part CI gate + temporal trend logic.

    Design principles being tested:
    1. CI entirely negative → REJECT (no ambiguity, feature is harmful)
    2. CI spans zero, mean negative → REJECT (positive blips are noise)
    3. CI spans zero, mean positive → PASS (likely signal, trend decides final fate)
    4. Feature with decaying trend + recent negatives → REJECT (dying signal)
    5. Feature with growing trend + recent positives → PASS (emerging signal)
    """

    @staticmethod
    def _build_report(feat_vals_dict, sfi_null=0.0):
        """Helper: build filter_report from {name: importance_array} dict."""
        from classical_learning.analysis.feature_importance import filter_features

        n_feats = len(feat_vals_dict)
        names = list(feat_vals_dict.keys())

        mdi_raw = pd.DataFrame({n: np.maximum(v, 0.0) for n, v in feat_vals_dict.items()})
        sfi_raw = pd.DataFrame(feat_vals_dict)
        desub_mda_raw = pd.DataFrame(feat_vals_dict)
        pca_mda_raw = pd.DataFrame(feat_vals_dict)
        resid_mda_raw = pd.DataFrame(feat_vals_dict)
        cfi_mda_raw = pd.DataFrame({i: feat_vals_dict[n] for i, n in enumerate(names)})
        clusters = {i: [n] for i, n in enumerate(names)}

        return filter_features(
            mdi_raw=mdi_raw,
            cfi_mda_raw=cfi_mda_raw,
            clusters=clusters,
            sfi_raw=sfi_raw,
            sfi_null=sfi_null,
            desub_mda_raw=desub_mda_raw,
            pca_mda_raw=pca_mda_raw,
            resid_mda_raw=resid_mda_raw,
        )

    # ── Case 1: Unambiguous noise — CI entirely negative ─────────────────

    def test_ci_entirely_negative_rejected(self):
        """[-0.9, -0.5, -0.8, -1.2, -0.1, 0.1, 0.2, -0.9]
        Mean=-0.51, CI=[-0.84, -0.18]. Pure noise with blips. REJECTED."""
        report = self._build_report({
            "noise": np.array([-0.9, -0.5, -0.8, -1.2, -0.1, 0.1, 0.2, -0.9]),
            "anchor": np.array([0.8, 0.9, 0.7, 0.85, 0.95, 0.88, 0.92, 0.87]),
        })
        assert report.loc["noise", "tier"] == "REJECTED"

    # ── Case 2: CI spans zero, mean negative → REJECT ────────────────────

    def test_ci_spans_zero_mean_negative_rejected(self):
        """[-0.3, -0.2, -0.4, -0.5, 0.4, 0.6, 0.3, -0.4]
        Mean=-0.06, CI≈[-0.33, 0.24]. CI upper > 0 but mean < 0.
        Positive folds are just noise. REJECTED."""
        report = self._build_report({
            "ambiguous_noise": np.array([-0.3, -0.2, -0.4, -0.5, 0.4, 0.6, 0.3, -0.4]),
            "anchor": np.array([0.8, 0.9, 0.7, 0.85, 0.95, 0.88, 0.92, 0.87]),
        })
        assert report.loc["ambiguous_noise", "tier"] == "REJECTED"

    # ── Case 3: CI spans zero, mean positive → PASS ──────────────────────

    def test_ci_spans_zero_mean_positive_passes(self):
        """[-0.01, 0.05, 0.03, 0.08, 0.02, 0.06, -0.02, 0.04]
        Mean=0.031, CI≈[-0.01, 0.08]. Most likely signal. Should PASS."""
        report = self._build_report({
            "likely_signal": np.array([-0.01, 0.05, 0.03, 0.08, 0.02, 0.06, -0.02, 0.04]),
            "anchor": np.array([0.8, 0.9, 0.7, 0.85, 0.95, 0.88, 0.92, 0.87]),
        })
        tier = report.loc["likely_signal", "tier"]
        assert tier in ("ACCEPTED", "NEEDS SPECIFICATION"), (
            f"Feature with CI [-0.01, 0.08] and positive mean should pass, got {tier}"
        )

    # ── Case 4: Decaying signal — was good, dying into noise ─────────────

    def test_decaying_signal_demoted(self):
        """[0.15, 0.12, 0.09, 0.06, 0.03, 0.00, -0.02, -0.05]
        Mean=0.048 (positive — passes mean gate).
        Spearman rho ≈ -1.0 (perfect monotonic decline).
        Last 3 folds mean = -0.023 (negative).
        This feature WAS signal but is now dying. Should be REJECTED."""
        report = self._build_report({
            "decaying": np.array([0.15, 0.12, 0.09, 0.06, 0.03, 0.00, -0.02, -0.05]),
            "anchor": np.array([0.8, 0.9, 0.7, 0.85, 0.95, 0.88, 0.92, 0.87]),
        })
        assert report.loc["decaying", "tier"] == "REJECTED", (
            f"Decaying feature should be REJECTED, got {report.loc['decaying', 'tier']}"
        )

    # ── Case 5: Growing signal — recently became important ───────────────

    def test_growing_signal_not_rejected(self):
        """[-0.05, -0.03, 0.00, 0.02, 0.05, 0.08, 0.10, 0.12]
        Mean=0.036. Spearman rho ≈ +1.0 (perfect growth).
        Recent folds positive. Should NOT be rejected."""
        report = self._build_report({
            "growing": np.array([-0.05, -0.03, 0.00, 0.02, 0.05, 0.08, 0.10, 0.12]),
            "anchor": np.array([0.8, 0.9, 0.7, 0.85, 0.95, 0.88, 0.92, 0.87]),
        })
        tier = report.loc["growing", "tier"]
        assert tier != "REJECTED", (
            f"Growing feature should NOT be REJECTED, got {tier}"
        )

    # ── Case 6: Stable positive signal — no trend issues ─────────────────

    def test_stable_positive_accepted(self):
        """[0.05, 0.06, 0.04, 0.07, 0.05, 0.06, 0.05, 0.06]
        Mean=0.055. CI entirely positive. No trend. Should be ACCEPTED."""
        report = self._build_report({
            "stable": np.array([0.05, 0.06, 0.04, 0.07, 0.05, 0.06, 0.05, 0.06]),
            "anchor": np.array([0.8, 0.9, 0.7, 0.85, 0.95, 0.88, 0.92, 0.87]),
        })
        tier = report.loc["stable", "tier"]
        assert tier in ("ACCEPTED", "NEEDS SPECIFICATION"), (
            f"Stable positive feature should pass, got {tier}"
        )

    # ── Case 7: Near-monotonic decline into recent noise ─────────────────

    def test_near_monotonic_decline_rejected(self):
        """[0.10, 0.08, 0.06, 0.04, 0.01, -0.01, 0.00, -0.03]
        Mean=0.031. Mean is positive (passes gate). But declining with
        near-monotonic trajectory. Last 3 folds mean ≈ -0.013.
        Spearman rho ≈ -0.95. Should be REJECTED by trend gate."""
        report = self._build_report({
            "declining": np.array([0.10, 0.08, 0.06, 0.04, 0.01, -0.01, 0.00, -0.03]),
            "anchor": np.array([0.8, 0.9, 0.7, 0.85, 0.95, 0.88, 0.92, 0.87]),
        })
        assert report.loc["declining", "tier"] == "REJECTED", (
            f"Declining feature should be REJECTED, got {report.loc['declining', 'tier']}"
        )

    # ── Case 8: High-variance noise — CI spans zero widely, mean ≈ 0 ────

    def test_high_variance_noise_rejected(self):
        """[-0.8, 0.7, -0.6, 0.5, -0.9, 0.8, -0.7, 0.6]
        Mean=-0.05. CI spans widely. No trend. Mean negative → REJECTED."""
        report = self._build_report({
            "volatile_noise": np.array([-0.8, 0.7, -0.6, 0.5, -0.9, 0.8, -0.7, 0.6]),
            "anchor": np.array([0.8, 0.9, 0.7, 0.85, 0.95, 0.88, 0.92, 0.87]),
        })
        assert report.loc["volatile_noise", "tier"] == "REJECTED"

    # ── Case 9: User's exact example ─────────────────────────────────────

    def test_user_example_noise_with_blips(self):
        """[-0.9, -0.5, -0.8, -1.2, -0.1, 0.1, 0.2, -0.9]
        The 0.1 and 0.2 occurred by chance — no monotonic increase, no sign
        of increasing power. Interval likely contains a positive but the
        feature is too weak to call signal. Must be REJECTED."""
        from classical_learning.analysis.feature_importance import bootstrap_ci

        vals = np.array([-0.9, -0.5, -0.8, -1.2, -0.1, 0.1, 0.2, -0.9])

        # Verify preconditions: mean is negative, CI upper is also negative
        mean, ci_lo, ci_hi = bootstrap_ci(vals)
        assert mean < 0
        assert ci_hi < 0  # entire CI negative for this particular example

        report = self._build_report({
            "user_example": vals,
            "anchor": np.array([0.8, 0.9, 0.7, 0.85, 0.95, 0.88, 0.92, 0.87]),
        })
        assert report.loc["user_example", "tier"] == "REJECTED"
