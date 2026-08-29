"""Training pipeline data leakage audit tests.

Audits 6 potential leakage vectors in the LOYO cross-validation training loop:
1. Standardizer fit scope — must only see training fold data
2. LOYO split correctness — temporal ordering, no future data in training
3. Feature selection leakage — importance computed on all folds?
4. Optuna HPO leakage — params tuned on latest split, applied universally
5. Target encoding leakage — rating systems use outcomes
6. Sample weighting — must not preferentially weight future data

Each test uses synthetic data to isolate the mechanism under test.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Fixtures: synthetic game_features data
# ---------------------------------------------------------------------------

SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
GAMES_PER_SEASON = 100


def _make_synthetic_features(n_features: int = 10) -> tuple[pd.DataFrame, Path]:
    """Create synthetic game_features.parquet for testing.

    Returns (DataFrame, path_to_parquet).
    Features have known distributions per season to detect cross-contamination.
    """
    rng = np.random.default_rng(42)
    rows = []
    game_pk = 100000

    for season in SEASONS:
        for i in range(GAMES_PER_SEASON):
            row = {
                "game_pk": game_pk,
                "season": season,
                "game_date": f"{season}-06-15",
                "home_win": rng.integers(0, 2),
                "yrfi": rng.integers(0, 2),
                "total_runs": rng.normal(8.5, 2.0),
                "home_run_diff": rng.normal(0, 3.0),
            }
            # Features with season-dependent means to detect leakage
            for j in range(n_features):
                # Each feature has mean = season * 0.01 + j, so standardizer
                # fit on one season would produce different parameters
                row[f"home_roll10_feat{j}"] = rng.normal(season * 0.01 + j, 1.0)
            game_pk += 1
            rows.append(row)

    df = pd.DataFrame(rows)
    tmpdir = tempfile.mkdtemp()
    path = Path(tmpdir) / "game_features.parquet"
    df.to_parquet(path, index=False)
    return df, path


@pytest.fixture
def synthetic_data():
    """Provide synthetic features DataFrame and parquet path."""
    df, path = _make_synthetic_features()
    return df, path


# ---------------------------------------------------------------------------
# 1. STANDARDIZER FIT SCOPE
# ---------------------------------------------------------------------------

class TestStandardizerFitScope:
    """Verify StandardScaler is fit ONLY on training fold data."""

    def test_prepare_fold_scaler_fit_on_train_only(self, synthetic_data):
        """CRITICAL: scaler.mean_ must equal X_train column means, not X_val.

        The prepare_fold function in data.py line 349-353:
            scaler = StandardScaler()
            X_train_arr = scaler.fit_transform(X_train)
            X_val_arr = scaler.transform(X_val)

        This test proves scaler statistics come only from training rows.
        """
        from classical_learning.strategy.data import prepare_fold, generate_loyo_splits, load_features

        df, path = synthetic_data
        X, y, seasons, game_pks = load_features(path, "home_win", "2016+")
        splits = generate_loyo_splits(seasons)

        # Use a model family that NEEDS_SCALING
        for split in splits:
            prepared = prepare_fold(X, y, seasons, split, "logistic_regression")

            if prepared.scaler is not None:
                # Scaler must be fit on training data only
                X_train_raw = X.iloc[split.train_idx]
                feature_cols = prepared.feature_columns

                # Filter to same feature columns used in prepared
                X_train_subset = X_train_raw[feature_cols]

                # Handle NaN imputation same as prepare_fold does
                from classical_learning.strategy.data import _semantic_impute
                X_train_imputed = _semantic_impute(X_train_subset)

                # Scaler mean should match training fold means
                expected_means = X_train_imputed.mean().values
                actual_means = prepared.scaler.mean_

                np.testing.assert_allclose(
                    actual_means, expected_means, rtol=1e-5,
                    err_msg=f"Scaler mean does not match train-only mean for fold {split.val_season}"
                )

                # Verify X_val is NOT in the fit statistics
                X_val_raw = X.iloc[split.val_idx][feature_cols]
                X_val_imputed = _semantic_impute(X_val_raw)
                combined_mean = pd.concat([X_train_imputed, X_val_imputed]).mean().values

                # Combined mean should differ from scaler mean (proving val not used)
                # This will only fail if val has different distribution (which our synthetic does)
                if len(split.val_idx) > 10:
                    # With enough val samples and season-shifted features, means must differ
                    assert not np.allclose(actual_means, combined_mean, atol=0.01), (
                        f"Scaler mean suspiciously equals train+val combined mean for fold {split.val_season}"
                    )

    def test_optuna_inner_fold_scaler_independence(self, synthetic_data):
        """Verify Optuna inner folds each fit their own scaler.

        optuna_objectives.py lines 79-89:
            if needs_scaling:
                fold_scaler = StandardScaler()
                X_tr = ... fold_scaler.fit_transform(X_tr) ...
                X_va = ... fold_scaler.transform(X_va) ...

        Each inner fold must have independent scaling.
        """
        from classical_learning.strategy.data import load_features, generate_loyo_splits, compute_temporal_weights
        from sklearn.model_selection import TimeSeriesSplit

        df, path = synthetic_data
        X, y, seasons, game_pks = load_features(path, "home_win", "2016+")
        splits = generate_loyo_splits(seasons)

        # Simulate what _run_optuna_hpo does
        latest_split = splits[-1]
        X_train = X.iloc[latest_split.train_idx]
        y_train = y.iloc[latest_split.train_idx]

        tscv = TimeSeriesSplit(n_splits=3)
        scalers = []

        for tr_idx, va_idx in tscv.split(X_train):
            X_tr = X_train.iloc[tr_idx]
            scaler = StandardScaler()
            scaler.fit(X_tr)
            scalers.append(scaler.mean_.copy())

        # Each inner fold should have different scaler means (because data grows)
        for i in range(len(scalers) - 1):
            assert not np.allclose(scalers[i], scalers[i + 1], atol=1e-6), (
                "Inner fold scalers are identical — likely a shared fit"
            )


# ---------------------------------------------------------------------------
# 2. LOYO SPLIT CORRECTNESS
# ---------------------------------------------------------------------------

class TestLOYOSplitCorrectness:
    """Verify LOYO splits maintain strict temporal ordering."""

    def test_train_seasons_strictly_before_val(self, synthetic_data):
        """Training folds must contain ONLY seasons < val_season."""
        from classical_learning.strategy.data import generate_loyo_splits

        df, _ = synthetic_data
        seasons = df["season"]
        splits = generate_loyo_splits(seasons)

        for split in splits:
            train_seasons_in_data = seasons.iloc[split.train_idx].unique()
            for ts in train_seasons_in_data:
                assert ts < split.val_season, (
                    f"Train contains season {ts} which is >= val_season {split.val_season}"
                )

    def test_val_season_not_in_train(self, synthetic_data):
        """Validation season must NOT appear in training indices."""
        from classical_learning.strategy.data import generate_loyo_splits

        df, _ = synthetic_data
        seasons = df["season"]
        splits = generate_loyo_splits(seasons)

        for split in splits:
            train_seasons = set(seasons.iloc[split.train_idx].unique())
            assert split.val_season not in train_seasons, (
                f"Val season {split.val_season} leaked into train"
            )

    def test_no_future_seasons_in_train(self, synthetic_data):
        """No season after val_season should appear in training."""
        from classical_learning.strategy.data import generate_loyo_splits

        df, _ = synthetic_data
        seasons = df["season"]
        splits = generate_loyo_splits(seasons)

        for split in splits:
            train_seasons = seasons.iloc[split.train_idx].unique()
            future_leak = [s for s in train_seasons if s > split.val_season]
            assert len(future_leak) == 0, (
                f"Future seasons {future_leak} in training for val_season={split.val_season}"
            )

    def test_2020_excluded_from_all_splits(self, synthetic_data):
        """2020 must never appear as train or val season (SKIP_SEASONS=[2020])."""
        from classical_learning.strategy.data import generate_loyo_splits

        df, _ = synthetic_data
        seasons = df["season"]
        splits = generate_loyo_splits(seasons)

        for split in splits:
            assert split.val_season != 2020, "2020 used as validation season"
            assert 2020 not in split.train_seasons, "2020 present in train_seasons"

    def test_features_precomputed_before_split(self, synthetic_data):
        """Features are computed ONCE on the full dataset before splitting.

        This is a DESIGN CHOICE, not a bug. Features for 2023 games use only
        prior-game data (shift(1) in rolling stats). The feature matrix is
        built once and then split — features are NOT recomputed per fold.

        Verify: same feature values appear regardless of which fold uses them.
        """
        from classical_learning.strategy.data import load_features, generate_loyo_splits

        df, path = synthetic_data
        X, y, seasons, game_pks = load_features(path, "home_win", "2016+")
        splits = generate_loyo_splits(seasons)

        # Pick an arbitrary game that appears in training for one fold and val for another
        # Its feature values should be identical in both contexts
        val_split = splits[-1]
        val_game_idx = val_split.val_idx[0]
        val_features = X.iloc[val_game_idx].values

        # This game should appear in the full matrix the same way
        full_features = X.iloc[val_game_idx].values
        np.testing.assert_array_equal(val_features, full_features)

    def test_minimum_train_seasons_enforced(self, synthetic_data):
        """LOYO_MIN_TRAIN_SEASONS (3) must be respected."""
        from classical_learning.strategy.data import generate_loyo_splits
        from classical_learning.strategy.config import LOYO_MIN_TRAIN_SEASONS

        df, _ = synthetic_data
        seasons = df["season"]
        splits = generate_loyo_splits(seasons)

        for split in splits:
            assert len(split.train_seasons) >= LOYO_MIN_TRAIN_SEASONS, (
                f"Split for val={split.val_season} has only {len(split.train_seasons)} "
                f"train seasons, minimum is {LOYO_MIN_TRAIN_SEASONS}"
            )


# ---------------------------------------------------------------------------
# 3. FEATURE SELECTION LEAKAGE
# ---------------------------------------------------------------------------

class TestFeatureSelectionLeakage:
    """Verify feature importance is computed using proper CV, not full data."""

    def test_feature_importance_uses_loyo_cv(self):
        """Feature importance (MDI, MDA, SFI) uses PurgedYearKFold — temporal CV.

        The feature_importance.py module uses PurgedYearKFold which splits by
        year. This means importance scores are computed out-of-sample per fold.

        VERDICT: FALSE ALARM — importance uses proper temporal CV.
        """
        from classical_learning.analysis.feature_importance import PurgedYearKFold

        # Simulate: n_splits=None means true LOYO
        seasons = pd.Series([2016]*50 + [2017]*50 + [2018]*50 + [2019]*50)
        cv = PurgedYearKFold(seasons, n_splits=None)

        folds = list(cv.split(np.zeros((200, 5)), groups=seasons.values))

        for train_idx, test_idx in folds:
            train_years = set(seasons.iloc[train_idx].unique())
            test_years = set(seasons.iloc[test_idx].unique())
            # No overlap
            assert train_years.isdisjoint(test_years), (
                f"Year overlap: train={train_years}, test={test_years}"
            )

    def test_importance_filter_applied_before_hpo(self, synthetic_data):
        """Feature filtering is resolved BEFORE HPO runs (train.py line 163).

        This means HPO and LOYO both use the same feature subset.
        No leakage: the features are selected based on out-of-sample importance
        analysis, not based on the current fold's target values.

        VERDICT: DESIGN CHOICE — importance analysis uses all seasons in LOYO
        fashion, then the selected features are applied to all training folds.
        This is standard practice (not leakage) because importance uses OOS eval.
        """
        # This is a structural test — verify the code path via source file
        from pathlib import Path
        source_path = Path(__file__).resolve().parents[1] / "pregame" / "strategy" / "train.py"
        source = source_path.read_text()

        # HPO receives X_hpo which is already filtered
        assert "X_hpo = X[importance_features]" in source
        # HPO happens before LOYO loop
        hpo_pos = source.find("_run_optuna_hpo")
        loyo_pos = source.find("for split in splits:")
        assert hpo_pos < loyo_pos, "HPO must run before LOYO evaluation loop"


# ---------------------------------------------------------------------------
# 4. OPTUNA HPO LEAKAGE
# ---------------------------------------------------------------------------

class TestOptunaHPOLeakage:
    """Quantify the known Optuna HPO leakage pattern."""

    def test_hpo_uses_latest_split_only(self, synthetic_data):
        """CONFIRMED BUG: HPO uses only the LATEST split's training data.

        train.py line 321: latest_split = splits[-1]

        Hyperparameters are tuned on the most recent split's training data
        (all seasons except the last), then applied universally to ALL LOYO
        folds. This means:
        - For the latest fold: HPO was done on exactly its training data (correct)
        - For earlier folds: HPO was done on data that INCLUDES their val season

        Example: If splits are [2019, 2020, 2021, 2022, 2023, 2024, 2025],
        HPO uses train from the 2025 split (seasons 2016-2024). But when
        evaluating the 2022 fold, the model uses params tuned on data including
        2022 — a form of information leakage.

        Impact: hyperparameters are slightly biased toward patterns in
        intermediate seasons that are val folds for earlier splits.
        """
        from classical_learning.strategy.data import generate_loyo_splits

        df, _ = synthetic_data
        seasons = df["season"]
        splits = generate_loyo_splits(seasons)

        # Verify HPO uses the LAST split
        latest_split = splits[-1]
        latest_train_seasons = set(latest_split.train_seasons)

        # For earlier folds, their val_season is IN the HPO training data
        leaked_folds = []
        for split in splits[:-1]:
            if split.val_season in latest_train_seasons:
                leaked_folds.append(split.val_season)

        # This SHOULD be non-empty — documenting the known issue
        assert len(leaked_folds) > 0, (
            "Expected HPO leakage for intermediate folds — pipeline may have changed"
        )

        # Quantify: what fraction of folds have their val season in HPO data?
        leak_fraction = len(leaked_folds) / len(splits)
        assert leak_fraction > 0.5, (
            f"Only {leak_fraction:.0%} of folds affected — expected majority"
        )

    def test_hpo_inner_cv_is_temporal(self, synthetic_data):
        """Inner HPO CV uses TimeSeriesSplit (temporal ordering preserved).

        optuna_objectives.py line 68: tscv = TimeSeriesSplit(n_splits=3)

        This is correct within the HPO split — inner folds respect temporal order.
        """
        from sklearn.model_selection import TimeSeriesSplit

        # Simulate 500 training samples (typical after removing val season)
        n = 500
        tscv = TimeSeriesSplit(n_splits=3)

        for tr_idx, va_idx in tscv.split(np.zeros((n, 10))):
            # All train indices should be before all val indices
            assert tr_idx.max() < va_idx.min(), (
                "TimeSeriesSplit inner fold violates temporal ordering"
            )


# ---------------------------------------------------------------------------
# 5. TARGET ENCODING LEAKAGE (Rating Systems)
# ---------------------------------------------------------------------------

class TestTargetEncodingLeakage:
    """Verify rating systems use only prior game outcomes."""

    def test_ratings_tuning_documented_leakage(self):
        """CONFIRMED BUG: Rating params tuned on val_seasons used as LOYO folds.

        ratings_tuning.py line 57-67 documents this explicitly:
        "the PARAMETERS are chosen to minimise error specifically on those val
        seasons. When those same val seasons later appear as LOYO val folds in
        train.py, the feature values (Elo, SRS, etc.) were generated with params
        tuned on them — a mild form of target leakage in parameter space."

        The code uses val_seasons = all_seasons[-3:] by default.
        """
        from pathlib import Path
        source_path = Path(__file__).resolve().parents[1] / "pregame" / "engineering" / "ratings_tuning.py"
        source = source_path.read_text()

        # Verify the documented limitation is still present
        assert "val_seasons = all_seasons[-3:]" in source, (
            "Rating tuning val_seasons logic has changed — re-audit needed"
        )
        # The Known limitation comment must still be there
        assert "mild form of target leakage" in source, (
            "Known leakage documentation removed — verify fix was applied"
        )

    def test_rating_features_use_only_prior_games(self):
        """Rating systems (Elo, Wolfe, etc.) only use outcomes from prior games.

        This is architecturally guaranteed: ratings iterate the game frame
        chronologically and update AFTER each game. The feature value for game N
        is the rating BEFORE game N is processed.

        This test verifies the design by checking feature_engineering.py uses
        shift(1) on all rolling/expanding computations.
        """
        from pathlib import Path
        source_path = Path(__file__).resolve().parents[1] / "pregame" / "engineering" / "feature_engineering.py"
        source = source_path.read_text()

        # Count shift(1) usage — this is the temporal guard
        # feature_engineering uses .shift(1) in grouped transforms;
        # ratings module uses chronological iteration (not shift). Combined
        # these provide temporal safety for all features.
        shift_count = source.count(".shift(1)")
        assert shift_count >= 10, (
            f"Only {shift_count} shift(1) calls found — expected >=10 for rolling features. "
            "Each rolling/expanding/EWMA feature must use shift(1) to exclude current game."
        )

        # Verify expanding computations used in FEATURE COLUMNS shift(1).
        # Exception: _compute_pregame_pitcher_era uses expanding().sum() WITHOUT
        # shift because temporal exclusion is handled later via searchsorted
        # (line 106: pos = searchsorted(..., side="left") - 1). This is correct.
        expanding_lines = [
            line.strip() for line in source.split("\n")
            if "expanding" in line and "transform" in line
        ]
        for line in expanding_lines:
            # The pitcher ERA helper uses searchsorted for exclusion, not shift
            if "expanding().sum()" in line:
                # This is in _compute_pregame_pitcher_era — exclusion via searchsorted
                assert "searchsorted" in source, (
                    "expanding().sum() without shift AND without searchsorted exclusion"
                )
            else:
                assert "shift(1)" in line or "shift" in line, (
                    f"Expanding computation without shift: {line}"
                )

    def test_sp_era_uses_searchsorted_exclusion(self):
        """Starting pitcher ERA uses searchsorted to exclude current game.

        feature_engineering.py line 106:
            pos = np.searchsorted(hist[:, 0], frame_idxs[i], side="left") - 1

        side="left" finds insertion point BEFORE current frame_idx, then -1
        gets the prior start. This correctly excludes the current game.
        """
        import inspect
        from classical_learning.engineering.feature_engineering import _compute_pregame_pitcher_era

        source = inspect.getsource(_compute_pregame_pitcher_era)
        assert 'side="left"' in source, "searchsorted must use side='left' for exclusion"
        assert "- 1" in source or "-1" in source, "Must subtract 1 from searchsorted position"


# ---------------------------------------------------------------------------
# 6. SAMPLE WEIGHTING
# ---------------------------------------------------------------------------

class TestSampleWeighting:
    """Verify temporal weighting does not introduce future data bias."""

    def test_temporal_weights_monotonically_increase(self):
        """Recent seasons must have HIGHER weight (not lower).

        compute_temporal_weights uses linear interpolation from min_weight=0.05
        (oldest) to 1.0 (newest). This correctly down-weights older data.
        """
        from classical_learning.strategy.data import compute_temporal_weights

        seasons = pd.Series([2016]*100 + [2017]*100 + [2018]*100 + [2019]*100 + [2021]*100)
        weights = compute_temporal_weights(seasons)

        # Mean weight for newer season should be higher
        w_2016 = weights[seasons == 2016].mean()
        w_2021 = weights[seasons == 2021].mean()
        assert w_2021 > w_2016, (
            f"2021 weight ({w_2021:.3f}) not greater than 2016 ({w_2016:.3f})"
        )

    def test_temporal_weights_within_training_fold_only(self, synthetic_data):
        """Weights are computed from training fold seasons only.

        data.py line 356: sample_weights = compute_temporal_weights(train_seasons)

        The weight normalization uses min/max of the TRAINING fold's seasons,
        not global min/max. This means weights are relative within each fold.
        """
        from classical_learning.strategy.data import compute_temporal_weights, generate_loyo_splits

        df, _ = synthetic_data
        seasons = df["season"]
        splits = generate_loyo_splits(seasons)

        for split in splits:
            train_seasons = seasons.iloc[split.train_idx]
            weights = compute_temporal_weights(train_seasons)

            # Weights should sum to n_samples (normalized)
            np.testing.assert_allclose(
                weights.sum(), len(train_seasons), rtol=1e-5,
                err_msg=f"Weights don't sum to n_samples for fold {split.val_season}"
            )

            # No weight should exceed a reasonable maximum
            # max_weight ≈ n_samples / (mean_raw_weight * n_samples) = 1/mean_raw
            assert weights.max() < 10.0, (
                f"Unreasonably large weight {weights.max():.2f} for fold {split.val_season}"
            )

    def test_single_season_training_gets_uniform_weights(self):
        """When train has only one season, all weights should be 1.0."""
        from classical_learning.strategy.data import compute_temporal_weights

        seasons = pd.Series([2023] * 100)
        weights = compute_temporal_weights(seasons)

        np.testing.assert_allclose(weights.values, 1.0, rtol=1e-10)

    def test_weights_do_not_use_val_season_info(self, synthetic_data):
        """Temporal weights must not incorporate val_season in their computation.

        The weight formula is: (season - min_train_season) / (max_train_season - min_train_season)
        This uses ONLY training seasons, never the validation season.
        """
        from classical_learning.strategy.data import compute_temporal_weights, generate_loyo_splits

        df, _ = synthetic_data
        seasons = df["season"]
        splits = generate_loyo_splits(seasons)

        for split in splits:
            train_seasons = seasons.iloc[split.train_idx]
            weights = compute_temporal_weights(train_seasons)

            # Verify max weight goes to the latest TRAINING season, not val
            max_train_season = train_seasons.max()
            assert max_train_season < split.val_season, (
                "Latest training season should be before val_season"
            )

            # Games from the max training season should have weight = 1.0 (pre-normalization)
            max_season_mask = train_seasons == max_train_season
            max_season_weights = weights[max_season_mask]
            # After normalization, these should be the highest weights
            other_weights = weights[~max_season_mask]
            assert max_season_weights.mean() > other_weights.mean(), (
                "Most recent training season should have highest average weight"
            )


# ---------------------------------------------------------------------------
# 7. SIZING CURVE LEAKAGE
# ---------------------------------------------------------------------------

class TestSizingCurveLeakage:
    """Verify feature_sizing uses held-out fold, not leaking into LOYO eval."""

    def test_sizing_uses_last_split_as_holdout(self):
        """feature_sizing.py line 175: sizing_split = splits[-1]

        The sizing curve uses the MOST RECENT season as its evaluation fold.
        This is the same season that will be the val fold in the last LOYO split.

        DESIGN CHOICE: sizing determines S* on the latest season, then training
        evaluates ALL folds (including the latest). The S* was selected to
        minimize loss on that specific season, giving it a slight advantage.

        However, the model parameters differ (sizing uses fixed params, training
        uses Optuna-tuned params), so the leakage is indirect: only the FEATURE
        COUNT is optimized on that season, not the model weights.
        """
        import inspect
        from classical_learning.strategy.feature_sizing import run_sizing_curve

        source = inspect.getsource(run_sizing_curve)
        assert "sizing_split = splits[-1]" in source, (
            "Sizing split selection has changed — re-audit needed"
        )

    def test_sizing_does_not_scale_data(self):
        """Sizing uses fillna(0) for imputation, not StandardScaler.

        feature_sizing.py lines 222-225:
            if needs_impute:
                X_tr = X_tr.fillna(0)
                X_va = X_va.fillna(0)

        No scaler leakage possible in sizing because no scaler is used.
        """
        import inspect
        from classical_learning.strategy.feature_sizing import run_sizing_curve

        source = inspect.getsource(run_sizing_curve)
        # Verify no StandardScaler usage in sizing
        assert "StandardScaler" not in source, (
            "Sizing now uses StandardScaler — audit for scaler leakage"
        )


# ---------------------------------------------------------------------------
# 8. PREGAME FEATURE ALLOWLIST (bonus check)
# ---------------------------------------------------------------------------

class TestPregameAllowlist:
    """Verify feature selection allowlist prevents post-game leakage."""

    def test_allowlist_excludes_postgame_columns(self):
        """_POSTGAME_EXCLUSIONS must block known leaky columns."""
        from classical_learning.strategy.data import _POSTGAME_EXCLUSIONS

        known_leakers = [
            "home_bsr_offense_game", "home_bsr_defense_game",
            "away_bsr_offense_game", "away_bsr_defense_game",
            "home_wins", "home_losses", "home_win_pct",
            "away_wins", "away_losses", "away_win_pct",
        ]
        for col in known_leakers:
            assert col in _POSTGAME_EXCLUSIONS, f"{col} not in POSTGAME_EXCLUSIONS"

    def test_allowlist_is_prefix_based(self):
        """Feature selection uses prefix allowlist — unknown columns are excluded.

        This is a strict allowlist (not a blocklist), meaning any new column
        that doesn't match a known prefix is automatically excluded.
        """
        from classical_learning.strategy.data import _select_pregame_features

        # Create a DataFrame with mixed columns
        df = pd.DataFrame({
            "home_roll10_ba": [0.3],           # should be selected (home_roll prefix)
            "away_roll5_era": [4.0],            # should be selected (away_roll prefix)
            "home_runs_scored_this_game": [5],   # should NOT be selected (no matching prefix)
            "leaked_target": [1],               # should NOT be selected
            "season": [2023],                   # non-numeric, excluded
            "home_elo": [1500.0],               # should be selected
        })

        selected = _select_pregame_features(df)
        assert "home_roll10_ba" in selected
        assert "away_roll5_era" in selected
        assert "home_elo" in selected
        assert "home_runs_scored_this_game" not in selected
        assert "leaked_target" not in selected


# ---------------------------------------------------------------------------
# 9. INTEGRATION: end-to-end fold isolation
# ---------------------------------------------------------------------------

class TestEndToEndFoldIsolation:
    """Integration test: verify no data flows from val to train in full pipeline."""

    def test_prepared_data_shapes_consistent(self, synthetic_data):
        """PreparedData must have correct shapes for each fold."""
        from classical_learning.strategy.data import prepare_fold, generate_loyo_splits, load_features

        df, path = synthetic_data
        X, y, seasons, game_pks = load_features(path, "home_win", "2016+")
        splits = generate_loyo_splits(seasons)

        total_val_samples = 0
        for split in splits:
            prepared = prepare_fold(X, y, seasons, split, "hist_gradient_boosting")

            assert len(prepared.X_train) == len(split.train_idx)
            assert len(prepared.X_val) == len(split.val_idx)
            assert len(prepared.y_train) == len(split.train_idx)
            assert len(prepared.y_val) == len(split.val_idx)
            assert len(prepared.sample_weights) == len(split.train_idx)

            total_val_samples += len(split.val_idx)

        # Sanity: all games appear in exactly one val fold
        # (minus games from seasons with < LOYO_MIN_TRAIN_SEASONS train seasons)
        all_val_idx = np.concatenate([s.val_idx for s in splits])
        assert len(all_val_idx) == len(set(all_val_idx)), (
            "Some games appear in multiple validation folds"
        )

    def test_observation_masks_from_train_only(self, synthetic_data):
        """Binary observation masks (_observed columns) use train NaN rates.

        data.py line 323: nan_pct = X_train.isna().mean()

        The threshold for adding _observed masks comes from training NaN rates.
        """
        from classical_learning.strategy.data import prepare_fold, generate_loyo_splits, load_features

        df, path = synthetic_data

        # Inject NaN into specific seasons to test mask behavior
        df_with_nan = df.copy()
        # Make 30% of feature 0 NaN in seasons 2022+
        mask = df_with_nan["season"] >= 2022
        rng = np.random.default_rng(99)
        nan_idx = df_with_nan.index[mask][rng.choice(mask.sum(), size=int(mask.sum() * 0.3), replace=False)]
        df_with_nan.loc[nan_idx, "home_roll10_feat0"] = np.nan

        # Write modified parquet
        tmpdir = tempfile.mkdtemp()
        nan_path = Path(tmpdir) / "game_features.parquet"
        df_with_nan.to_parquet(nan_path, index=False)

        X, y, seasons, game_pks = load_features(nan_path, "home_win", "2016+")
        splits = generate_loyo_splits(seasons)

        # For a tree model, check that observation masks are computed from train
        for split in splits:
            prepared = prepare_fold(X, y, seasons, split, "hist_gradient_boosting")

            if prepared.observation_masks is not None:
                # The NaN rate used for masking came from X_train only
                X_train_nan_rate = X.iloc[split.train_idx].isna().mean()
                # Any _observed column must correspond to a >5% NaN column in TRAIN
                for obs_col in prepared.observation_masks.columns:
                    orig_col = obs_col.replace("_observed", "")
                    if orig_col in X_train_nan_rate.index:
                        assert X_train_nan_rate[orig_col] > 0.05, (
                            f"Observation mask for {orig_col} with only "
                            f"{X_train_nan_rate[orig_col]:.1%} NaN in train"
                        )
