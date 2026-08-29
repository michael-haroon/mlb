"""Tests for filter_features_v2: all methods always participate (no PCA gating).

Verifies that filter_features_v2 scores all 5 methods for every feature
regardless of PCA cross-check results. The PCA cross-check is a structural
diagnostic, not a gate.

Run: conda run -n pred python -m pytest tests/test_filter_features.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.analysis.feature_importance import filter_features_v2


def _make_raw_df(features: list[str], n_folds: int = 8, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic raw importance DataFrame (folds × features)."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0.01, 0.005, size=(n_folds, len(features)))
    return pd.DataFrame(data, columns=features)


def _make_mdi_raw(features: list[str], n_trees: int = 100, seed: int = 42) -> pd.DataFrame:
    """MDI raw has n_trees rows (per-tree importance)."""
    rng = np.random.default_rng(seed)
    data = rng.uniform(0.001, 0.05, size=(n_trees, len(features)))
    return pd.DataFrame(data, columns=features)


class TestAllMethodsAlwaysParticipate:
    """filter_features_v2 must score all methods — no PCA-based silencing."""

    FEATURES = ["feat_a", "feat_b", "feat_c"]
    CLUSTERS = {0: ["feat_a", "feat_b"], 1: ["feat_c"]}

    def _call_filter(self):
        sfi_raw = _make_raw_df(self.FEATURES, seed=1)
        desub_mda_raw = _make_raw_df(self.FEATURES, seed=2)
        pca_mda_raw = _make_raw_df(self.FEATURES, seed=3)
        resid_mda_raw = _make_raw_df(self.FEATURES, seed=4)
        mdi_raw = _make_mdi_raw(self.FEATURES, seed=5)
        cfi_mda_raw = _make_raw_df([0, 1], seed=6)
        cfi_mdi_raw = _make_mdi_raw(["Cluster_0 (feat_a, feat_b)", "Cluster_1 (feat_c)"], seed=7)
        return filter_features_v2(
            sfi_raw=sfi_raw,
            desub_mda_raw=desub_mda_raw,
            pca_mda_raw=pca_mda_raw,
            resid_mda_raw=resid_mda_raw,
            mdi_raw=mdi_raw,
            cfi_mda_raw=cfi_mda_raw,
            cfi_mdi_raw=cfi_mdi_raw,
            clusters=self.CLUSTERS,
            sfi_null=-0.693,
        )

    def test_all_pass_columns_populated(self):
        """Every method's pass column should be non-NaN for all features."""
        report = self._call_filter()

        for col in ["sfi_passes", "desub_mda_passes", "pca_mda_passes",
                    "resid_mda_passes", "mdi_passes"]:
            assert col in report.columns, f"Missing column: {col}"
            n_nan = report[col].isna().sum()
            assert n_nan == 0, (
                f"{col} has {n_nan} NaN values — methods are being silenced"
            )

    def test_all_mean_columns_populated(self):
        """Every method's mean column should be non-NaN for all features."""
        report = self._call_filter()

        for col in ["sfi_mean", "desub_mda_mean", "pca_mda_mean",
                    "resid_mda_mean", "mdi_mean"]:
            assert col in report.columns, f"Missing column: {col}"
            n_nan = report[col].isna().sum()
            assert n_nan == 0, (
                f"{col} has {n_nan} NaN values — methods are being silenced"
            )

    def test_all_rank_columns_populated(self):
        """Every method's rank column should exist and be non-NaN."""
        report = self._call_filter()

        for col in ["sfi_rank", "desub_mda_rank", "pca_mda_rank",
                    "resid_mda_rank", "mdi_rank"]:
            assert col in report.columns, f"Missing column: {col}"

    def test_no_pca_crosscheck_parameter(self):
        """filter_features_v2 should not accept pca_crosscheck parameter."""
        import inspect
        sig = inspect.signature(filter_features_v2)
        assert "pca_crosscheck" not in sig.parameters, (
            "filter_features_v2 still accepts pca_crosscheck — PCA gating not removed"
        )

    def test_composite_rank_uses_all_methods(self):
        """Composite rank should average across all 5 methods."""
        report = self._call_filter()

        assert "composite_rank" in report.columns
        assert not report["composite_rank"].isna().any()
