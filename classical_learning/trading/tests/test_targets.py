"""Tests for pregame.trading.targets sub-package."""
from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from classical_learning.trading.targets import TOTAL_RUNS_CONFIG, load_total_runs_config
from classical_learning.trading.targets.base import TargetConfig
from classical_learning.trading.targets.total_runs import _negbin_half_spread


# ---------------------------------------------------------------------------
# Unit tests: _negbin_half_spread
# ---------------------------------------------------------------------------

class TestNegbinHalfSpread:
    """Verify NegBin-derived spread width across mu range."""

    @pytest.mark.parametrize("mu,expected", [
        (7.0, 3),
        (8.0, 3),
        (9.0, 3),
        (10.0, 3),
        (11.0, 3),
        (12.0, 3),
        (13.0, 3),
    ])
    def test_expected_values_across_mu_range(self, mu: float, expected: int):
        """Verify expected spread values for mu in [7, 13]."""
        result = _negbin_half_spread(mu, alpha=6.732)
        assert result == expected, f"mu={mu}: expected {expected}, got {result}"

    def test_bounded_between_2_and_5(self):
        """Output must always be in [2, 5] regardless of input."""
        for mu in np.linspace(0.1, 50.0, 200):
            result = _negbin_half_spread(mu, alpha=6.732)
            assert 2 <= result <= 5, f"mu={mu}: got {result}, outside [2, 5]"

    def test_mu_zero_edge_case(self):
        """mu=0 produces sqrt(0) = 0, raw = 1 + 0 = 1, clamped to 2."""
        result = _negbin_half_spread(0.0, alpha=6.732)
        assert result == 2

    def test_mu_very_large(self):
        """Very large mu should be clamped to 5."""
        result = _negbin_half_spread(30.0, alpha=6.732)
        assert result == 5

    def test_returns_int(self):
        """Output type must be int for use as cents."""
        result = _negbin_half_spread(9.0, alpha=6.732)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# Unit tests: TargetConfig.get_half_spread
# ---------------------------------------------------------------------------

class TestTargetConfigGetHalfSpread:
    """Verify delegation logic in get_half_spread."""

    def test_delegates_to_callable_when_set(self):
        """When compute_half_spread is set, get_half_spread delegates to it."""
        mock_spread_fn = lambda mu, alpha: 4
        cfg = TargetConfig(
            name="test",
            distribution_type="negbin",
            model_error_std=0.5,
            half_spread_base_cents=2,
            price_floor=0.10,
            price_ceiling=0.90,
            cluster_max_contracts=5,
            standard_lines=(8.5, 9.5),
            compute_half_spread=mock_spread_fn,
        )
        assert cfg.get_half_spread(9.0, 6.73) == 4

    def test_falls_back_to_base_cents_when_none(self):
        """When compute_half_spread is None, return half_spread_base_cents."""
        cfg = TargetConfig(
            name="test",
            distribution_type="negbin",
            model_error_std=0.5,
            half_spread_base_cents=3,
            price_floor=0.10,
            price_ceiling=0.90,
            cluster_max_contracts=5,
            standard_lines=(8.5, 9.5),
            compute_half_spread=None,
        )
        assert cfg.get_half_spread(9.0, 6.73) == 3

    def test_callable_receives_correct_args(self):
        """Verify mu and alpha are passed through correctly."""
        received = {}

        def capture_fn(mu, alpha):
            received['mu'] = mu
            received['alpha'] = alpha
            return 3

        cfg = TargetConfig(
            name="test",
            distribution_type="negbin",
            model_error_std=0.5,
            half_spread_base_cents=2,
            price_floor=0.10,
            price_ceiling=0.90,
            cluster_max_contracts=5,
            standard_lines=(8.5, 9.5),
            compute_half_spread=capture_fn,
        )
        cfg.get_half_spread(11.5, 7.0)
        assert received['mu'] == 11.5
        assert received['alpha'] == 7.0


# ---------------------------------------------------------------------------
# Unit tests: load_total_runs_config
# ---------------------------------------------------------------------------

class TestLoadTotalRunsConfig:
    """Test loading config from calibration bundles."""

    def test_extracts_alpha_and_model_error_std(self):
        """Verify attributes are pulled from calibration bundle."""
        bundle = SimpleNamespace(negbin_alpha=5.5, model_error_std=0.82)
        cfg = load_total_runs_config(bundle)
        assert cfg.negbin_alpha == 5.5
        assert cfg.model_error_std == 0.82

    def test_fallback_when_no_model_error_std(self):
        """Old pickles may lack model_error_std — should fall back to 0.795."""
        bundle = SimpleNamespace(negbin_alpha=6.0)
        cfg = load_total_runs_config(bundle)
        assert cfg.model_error_std == 0.795

    def test_fallback_when_no_negbin_alpha(self):
        """If alpha is missing, fall back to 6.732."""
        bundle = SimpleNamespace(model_error_std=0.8)
        cfg = load_total_runs_config(bundle)
        assert cfg.negbin_alpha == 6.732

    def test_fallback_when_model_error_std_is_none(self):
        """If model_error_std is explicitly None, fall back to 0.795."""
        bundle = SimpleNamespace(negbin_alpha=6.0, model_error_std=None)
        cfg = load_total_runs_config(bundle)
        assert cfg.model_error_std == 0.795

    def test_all_fields_populated(self):
        """Verify all required fields are set on returned config."""
        bundle = SimpleNamespace(negbin_alpha=6.732, model_error_std=0.795)
        cfg = load_total_runs_config(bundle)
        assert cfg.name == "total_runs"
        assert cfg.distribution_type == "negbin"
        assert cfg.half_spread_base_cents == 3
        assert cfg.price_floor == 0.12
        assert cfg.price_ceiling == 0.88
        assert cfg.cluster_max_contracts == 10
        assert cfg.standard_lines == (5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5)
        assert cfg.compute_half_spread is not None

    def test_with_real_pickle(self):
        """Load from actual ensemble pickle if available."""
        pkl_path = Path("pregame/artifacts/models/ensemble_total_runs_A.pkl")
        if not pkl_path.exists():
            pytest.skip("Pickle artifact not available in test environment")
        with open(pkl_path, 'rb') as f:
            bundle = pickle.load(f)
        cfg = load_total_runs_config(bundle['calibration'])
        assert cfg.negbin_alpha > 0
        assert cfg.model_error_std > 0
        assert cfg.get_half_spread(9.0, cfg.negbin_alpha) in range(2, 6)


# ---------------------------------------------------------------------------
# Integration test: import and default config
# ---------------------------------------------------------------------------

class TestIntegration:
    """Verify the package imports and TOTAL_RUNS_CONFIG has expected values."""

    def test_import_total_runs_config(self):
        """Verify TOTAL_RUNS_CONFIG is importable and has correct name."""
        assert TOTAL_RUNS_CONFIG.name == "total_runs"

    def test_config_field_values(self):
        """Verify default field values match specification."""
        assert TOTAL_RUNS_CONFIG.distribution_type == "negbin"
        assert TOTAL_RUNS_CONFIG.model_error_std == 0.795
        assert TOTAL_RUNS_CONFIG.half_spread_base_cents == 3
        assert TOTAL_RUNS_CONFIG.price_floor == 0.12
        assert TOTAL_RUNS_CONFIG.price_ceiling == 0.88
        assert TOTAL_RUNS_CONFIG.cluster_max_contracts == 10
        assert TOTAL_RUNS_CONFIG.negbin_alpha == 6.732
        assert TOTAL_RUNS_CONFIG.standard_lines == (5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5)

    def test_config_get_half_spread_uses_negbin(self):
        """Default config should use the NegBin spread function."""
        result = TOTAL_RUNS_CONFIG.get_half_spread(9.0, 6.732)
        assert isinstance(result, int)
        assert 2 <= result <= 5


# ---------------------------------------------------------------------------
# Stress tests: extreme mu values
# ---------------------------------------------------------------------------

class TestStressExtremeValues:
    """Verify _negbin_half_spread handles extreme inputs gracefully."""

    @pytest.mark.parametrize("mu", [0.1, 0.5, 1.0, 2.0, 3.0])
    def test_very_low_mu(self, mu: float):
        """Very low mu should not crash and should return >= 2."""
        result = _negbin_half_spread(mu, alpha=6.732)
        assert isinstance(result, int)
        assert result >= 2

    @pytest.mark.parametrize("mu", [20.0, 30.0, 40.0, 50.0])
    def test_very_high_mu(self, mu: float):
        """Very high mu should not crash and should return <= 5."""
        result = _negbin_half_spread(mu, alpha=6.732)
        assert isinstance(result, int)
        assert result <= 5

    def test_alpha_very_small(self):
        """Very small alpha (high overdispersion) should still be bounded."""
        result = _negbin_half_spread(9.0, alpha=0.5)
        assert 2 <= result <= 5

    def test_alpha_very_large(self):
        """Very large alpha (approaching Poisson) should still be bounded."""
        result = _negbin_half_spread(9.0, alpha=1000.0)
        assert 2 <= result <= 5

    def test_monotonicity_in_typical_range(self):
        """Spread should be non-decreasing for mu in [5, 15] (typical MLB range)."""
        spreads = [_negbin_half_spread(mu, alpha=6.732) for mu in range(5, 16)]
        for i in range(len(spreads) - 1):
            assert spreads[i] <= spreads[i + 1], (
                f"Non-monotonic: mu={5+i} -> {spreads[i]}, mu={6+i} -> {spreads[i+1]}"
            )
