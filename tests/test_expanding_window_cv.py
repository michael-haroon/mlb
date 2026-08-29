"""Behavioral tests for ExpandingWindowYearCV and forward-only importance pipeline.

Verifies:
  1. Fold structure — train years < test year, expanding sizes, skip/min logic
  2. Importance functions produce correct output shapes with new CV
  3. Edge cases — insufficient years, all skipped, boundary conditions
"""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def standard_seasons():
    """11 seasons (2015-2026, 2020 excluded) with 100 rows each."""
    years = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
    return pd.Series(np.repeat(years, 100))


@pytest.fixture
def synthetic_data(standard_seasons):
    """Small synthetic dataset for importance function tests."""
    rng = np.random.default_rng(42)
    n = len(standard_seasons)
    X = pd.DataFrame(
        rng.standard_normal((n, 5)),
        columns=["signal_a", "signal_b", "noise_c", "noise_d", "noise_e"],
    )
    # Inject signal into first two features
    X["signal_a"] += (standard_seasons.values > 2020).astype(float) * 0.8
    y = pd.Series((X["signal_a"] + rng.normal(0, 0.5, n) > 0).astype(int))
    return X, y, standard_seasons


# ---------------------------------------------------------------------------
# 1. Fold structure tests
# ---------------------------------------------------------------------------

class TestExpandingWindowYearCV:

    def test_fold_count(self, standard_seasons):
        from classical_learning.analysis.feature_importance import ExpandingWindowYearCV

        cv = ExpandingWindowYearCV(standard_seasons)
        # min_train_seasons=3: first valid test year is 2018 (trains on 2015,2016,2017)
        # Test years: 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026 = 8
        assert cv.get_n_splits() == 8

    def test_train_years_strictly_before_test(self, standard_seasons):
        from classical_learning.analysis.feature_importance import ExpandingWindowYearCV

        cv = ExpandingWindowYearCV(standard_seasons)
        X = np.zeros((len(standard_seasons), 3))

        for train_idx, test_idx in cv.split(X, groups=standard_seasons.values):
            train_years = set(standard_seasons.iloc[train_idx].unique())
            test_years = set(standard_seasons.iloc[test_idx].unique())
            assert len(test_years) == 1
            test_year = test_years.pop()
            assert all(ty < test_year for ty in train_years), (
                f"Train years {train_years} not all < test year {test_year}"
            )

    def test_2020_never_in_train_or_test(self, standard_seasons):
        from classical_learning.analysis.feature_importance import ExpandingWindowYearCV

        cv = ExpandingWindowYearCV(standard_seasons)
        X = np.zeros((len(standard_seasons), 3))

        for train_idx, test_idx in cv.split(X, groups=standard_seasons.values):
            train_years = set(standard_seasons.iloc[train_idx].unique())
            test_years = set(standard_seasons.iloc[test_idx].unique())
            assert 2020 not in train_years
            assert 2020 not in test_years

    def test_training_sizes_monotonically_expand(self, standard_seasons):
        from classical_learning.analysis.feature_importance import ExpandingWindowYearCV

        cv = ExpandingWindowYearCV(standard_seasons)
        X = np.zeros((len(standard_seasons), 3))
        train_sizes = [len(tr) for tr, _ in cv.split(X, groups=standard_seasons.values)]
        assert train_sizes == sorted(train_sizes), (
            f"Training sizes not monotonically increasing: {train_sizes}"
        )

    def test_correct_test_year_order(self, standard_seasons):
        from classical_learning.analysis.feature_importance import ExpandingWindowYearCV

        cv = ExpandingWindowYearCV(standard_seasons)
        X = np.zeros((len(standard_seasons), 3))

        test_years = [
            standard_seasons.iloc[te].unique()[0]
            for _, te in cv.split(X, groups=standard_seasons.values)
        ]
        assert test_years == [2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]

    def test_first_fold_trains_on_exactly_3_years(self, standard_seasons):
        from classical_learning.analysis.feature_importance import ExpandingWindowYearCV

        cv = ExpandingWindowYearCV(standard_seasons)
        X = np.zeros((len(standard_seasons), 3))
        folds = list(cv.split(X, groups=standard_seasons.values))
        first_train_years = sorted(standard_seasons.iloc[folds[0][0]].unique())
        assert first_train_years == [2015, 2016, 2017]

    def test_no_groups_raises(self, standard_seasons):
        from classical_learning.analysis.feature_importance import ExpandingWindowYearCV

        cv = ExpandingWindowYearCV(standard_seasons)
        X = np.zeros((len(standard_seasons), 3))
        with pytest.raises(ValueError, match="groups"):
            list(cv.split(X))

    def test_min_train_seasons_respected(self):
        """With only 2 years, min_train_seasons=3 produces no folds."""
        from classical_learning.analysis.feature_importance import ExpandingWindowYearCV

        years = pd.Series(np.repeat([2015, 2016, 2017], 50))
        cv = ExpandingWindowYearCV(years, skip_seasons=[], min_train_seasons=3)
        X = np.zeros((len(years), 3))
        folds = list(cv.split(X, groups=years.values))
        assert len(folds) == 0

    def test_custom_skip_seasons(self):
        from classical_learning.analysis.feature_importance import ExpandingWindowYearCV

        years = pd.Series(np.repeat([2015, 2016, 2017, 2018, 2019], 50))
        cv = ExpandingWindowYearCV(years, skip_seasons=[2017], min_train_seasons=2)
        X = np.zeros((len(years), 3))

        folds = list(cv.split(X, groups=years.values))
        test_years = [years.iloc[te].unique()[0] for _, te in folds]
        # 2017 skipped from both train and test
        assert 2017 not in test_years
        for tr, _ in folds:
            assert 2017 not in set(years.iloc[tr].unique())

    def test_matches_training_pipeline_splits(self):
        """Forward-only CV should produce same splits as generate_loyo_splits."""
        from classical_learning.analysis.feature_importance import ExpandingWindowYearCV
        from classical_learning.strategy.data import generate_loyo_splits

        years = pd.Series(np.repeat(
            [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026], 80
        ))
        cv = ExpandingWindowYearCV(years)
        X = np.zeros((len(years), 3))
        cv_folds = list(cv.split(X, groups=years.values))

        loyo_splits = generate_loyo_splits(years)

        assert len(cv_folds) == len(loyo_splits)
        for (cv_train, cv_test), loyo_split in zip(cv_folds, loyo_splits):
            assert set(cv_train) == set(loyo_split.train_idx)
            assert set(cv_test) == set(loyo_split.val_idx)


# ---------------------------------------------------------------------------
# 2. Importance functions produce correct output with new CV
# ---------------------------------------------------------------------------

class TestImportanceFunctionsWithExpandingCV:

    def test_mda_runs_and_returns_8_folds(self, synthetic_data):
        from classical_learning.analysis.feature_importance import feat_imp_mda, build_rf

        X, y, years = synthetic_data
        clf = build_rf(n_estimators=20, n_jobs=1)
        summary, raw = feat_imp_mda(clf, X, y, years)

        assert isinstance(summary, pd.DataFrame)
        assert list(summary.columns) == ["mean", "std"]
        assert len(summary) == 5  # 5 features
        assert raw.shape[0] == 8  # 8 folds

    def test_sfi_runs_and_returns_8_folds(self, synthetic_data):
        from classical_learning.analysis.feature_importance import feat_imp_sfi, build_rf

        X, y, years = synthetic_data
        clf = build_rf(n_estimators=20, n_jobs=1)
        summary, raw = feat_imp_sfi(clf, X, y, years)

        assert isinstance(summary, pd.DataFrame)
        assert "mean" in summary.columns
        assert len(summary) == 5
        assert raw.shape[0] == 8

    def test_desub_mda_runs_and_returns_8_folds(self, synthetic_data):
        from classical_learning.analysis.feature_importance import feat_imp_desub_mda

        X, y, years = synthetic_data
        clusters = {0: ["signal_a", "signal_b"], 1: ["noise_c", "noise_d", "noise_e"]}
        summary, raw = feat_imp_desub_mda(
            X, y, years, clusters, n_estimators=20,
        )

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 5
        # raw is a dict of lists, each of length 8
        for feat, scores in raw.items():
            assert len(scores) == 8

    def test_cfi_mda_runs_and_returns_8_folds(self, synthetic_data):
        from classical_learning.analysis.feature_importance import feat_imp_cfi_mda, build_rf

        X, y, years = synthetic_data
        clusters = {0: ["signal_a", "signal_b"], 1: ["noise_c", "noise_d", "noise_e"]}
        clf = build_rf(n_estimators=20, n_jobs=1)
        summary, raw = feat_imp_cfi_mda(clf, X, y, years, clusters)

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2  # 2 clusters
        assert isinstance(raw, pd.DataFrame)
        assert raw.shape[0] == 8

    def test_pca_mda_runs(self, synthetic_data):
        from classical_learning.analysis.feature_importance import feat_imp_pca_mda

        X, y, years = synthetic_data
        summary, raw, pc_summary = feat_imp_pca_mda(
            X, y, years, n_estimators=20,
        )

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 5

    def test_residual_mda_runs(self, synthetic_data):
        from classical_learning.analysis.feature_importance import feat_imp_residual_mda

        X, y, years = synthetic_data
        clusters = {0: ["signal_a", "signal_b"], 1: ["noise_c", "noise_d", "noise_e"]}
        summary, raw = feat_imp_residual_mda(
            X, y, years, clusters, n_estimators=20,
        )

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 5
        assert raw.shape[0] == 8


# ---------------------------------------------------------------------------
# 3. Signal detection — forward-only should still rank signal above noise
# ---------------------------------------------------------------------------

class TestSignalDetection:

    def test_mda_ranks_signal_above_noise(self, synthetic_data):
        """Strong signal features should rank above pure noise."""
        from classical_learning.analysis.feature_importance import feat_imp_mda, build_rf

        X, y, years = synthetic_data
        clf = build_rf(n_estimators=50, n_jobs=1)
        summary, _ = feat_imp_mda(clf, X, y, years)

        signal_mean = summary.loc["signal_a", "mean"]
        noise_mean = summary.loc[["noise_c", "noise_d", "noise_e"], "mean"].mean()
        assert signal_mean > noise_mean
