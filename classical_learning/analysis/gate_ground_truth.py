"""Ground truth labels for feature importance gating backtest.

For each split point, computes:
  - Gate folds (1..s): the folds available to make the gating decision
  - Held-out folds (s+1..8): future data to evaluate whether the gate was right

Ground truth per feature is the MEDIAN held-out importance (robust to outlier folds).
Multiple split points are tested for stability — conclusions must hold across them.

The labels produced here are independent of the gating decision function. They will
later be used to evaluate combine_test_scores() and to fit a proper weight vector
from labeled data, replacing the hand-tuned combiner.

Fold structure (ExpandingWindowCV, skip 2020, min_train=3):
  Fold 1: test=2018, train=[2015-2017]
  Fold 2: test=2019, train=[2015-2018]
  Fold 3: test=2021, train=[2015-2019]
  Fold 4: test=2022, train=[2015-2021]
  Fold 5: test=2023, train=[2015-2022]
  Fold 6: test=2024, train=[2015-2023]
  Fold 7: test=2025, train=[2015-2024]
  Fold 8: test=2026, train=[2015-2025]
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compute_ground_truth_labels(
    importance_raw: pd.DataFrame,
    split_points: list[int] = None,
    null: float = 0.0,
) -> dict[int, pd.DataFrame]:
    """Compute per-feature ground truth labels from held-out folds.

    Parameters
    ----------
    importance_raw : DataFrame with shape (n_folds, n_features).
        Fold values ordered chronologically (fold 0 = earliest test year).
    split_points : which fold indices to split on (gate uses folds 0..s-1,
        ground truth uses folds s..n). Default [4, 5, 6] gives 4/3/2 held-out
        folds respectively.
    null : null value for the importance test (0.0 for MDA tests, ln(0.5) for SFI).

    Returns
    -------
    dict mapping split_point -> DataFrame with columns:
        - held_out_median: median importance across held-out folds
        - held_out_mean: mean importance across held-out folds
        - held_out_std: std across held-out folds
        - n_held_out: number of held-out folds
        - signal: held_out_median - null (positive = feature has signal)
        - label: binary (1 = feature has signal in held-out, 0 = does not)
    """
    if split_points is None:
        split_points = [4, 5, 6]

    n_folds = importance_raw.shape[0]
    features = list(importance_raw.columns)
    results = {}

    for s in split_points:
        if s >= n_folds or s < 2:
            continue

        held_out = importance_raw.iloc[s:]
        n_ho = held_out.shape[0]

        ho_median = held_out.median(axis=0)
        ho_mean = held_out.mean(axis=0)
        ho_std = held_out.std(axis=0, ddof=1)
        signal = ho_median - null

        label = (signal > 0).astype(int)

        df = pd.DataFrame({
            "held_out_median": ho_median,
            "held_out_mean": ho_mean,
            "held_out_std": ho_std,
            "n_held_out": n_ho,
            "signal": signal,
            "label": label,
        }, index=features)
        results[s] = df

    return results


def compute_cv_lift_labels(
    X: pd.DataFrame,
    y: np.ndarray,
    years: pd.Series,
    split_points: list[int] = None,
    n_estimators: int = 300,
    n_jobs: int = -1,
) -> dict[int, pd.DataFrame]:
    """Compute ground truth via actual CV lift (drop-one-feature Brier/log-loss change).

    For each split point:
      - Train a forest on gate folds, evaluate on held-out folds
      - For each feature: retrain WITHOUT that feature, measure held-out loss change
      - Positive lift = feature helps; negative = feature hurts or is noise

    This is the gold standard but expensive: O(n_features * n_split_points) model fits.
    With 513 features and 3 split points: ~1539 model fits.

    Parameters
    ----------
    X : full feature DataFrame
    y : target array
    years : season series (for fold assignment)
    split_points : same as compute_ground_truth_labels
    n_estimators : trees per forest (lower = faster, 300 is sufficient for ranking)
    n_jobs : parallelism

    Returns
    -------
    dict mapping split_point -> DataFrame with columns:
        - base_loss: log-loss with all features
        - drop_loss: log-loss without this feature
        - lift: drop_loss - base_loss (positive = feature helps)
        - label: binary (1 = feature helps, 0 = does not)
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.metrics import log_loss

    if split_points is None:
        split_points = [4, 5, 6]

    from classical_learning.strategy.config import SKIP_SEASONS, LOYO_MIN_TRAIN_SEASONS
    all_years = sorted(years.unique())
    valid_test_years = []
    for test_year in all_years:
        if test_year in SKIP_SEASONS:
            continue
        train_years = [s for s in all_years if s < test_year and s not in SKIP_SEASONS]
        if len(train_years) >= LOYO_MIN_TRAIN_SEASONS:
            valid_test_years.append(test_year)

    features = list(X.columns)
    results = {}

    for s in split_points:
        if s >= len(valid_test_years) or s < 2:
            continue

        # Gate years = test years for folds 0..s-1
        # Held-out years = test years for folds s..end
        gate_test_years = valid_test_years[:s]
        held_out_test_years = valid_test_years[s:]

        # Train set: all years before the first held-out year
        first_ho_year = held_out_test_years[0]
        train_years = [yr for yr in all_years if yr < first_ho_year and yr not in SKIP_SEASONS]
        train_mask = years.isin(train_years)
        test_mask = years.isin(held_out_test_years)

        X_train = X.loc[train_mask].fillna(X.loc[train_mask].median())
        y_train = y[train_mask.values]
        X_test = X.loc[test_mask].fillna(X.loc[train_mask].median())
        y_test = y[test_mask.values]

        # Base model: all features
        base_clf = BaggingClassifier(
            estimator=DecisionTreeClassifier(
                criterion="entropy", max_features=1,
                class_weight="balanced", min_weight_fraction_leaf=0.02,
            ),
            n_estimators=n_estimators,
            max_features=1.0, max_samples=1.0, oob_score=False,
            n_jobs=n_jobs, random_state=42,
        )
        base_clf.fit(X_train.values, y_train)
        base_proba = base_clf.predict_proba(X_test.values)
        base_ll = log_loss(y_test, base_proba)

        # Drop-one-feature
        lifts = []
        for i, feat in enumerate(features):
            cols_mask = [j for j in range(len(features)) if j != i]
            clf_drop = BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    criterion="entropy", max_features=1,
                    class_weight="balanced", min_weight_fraction_leaf=0.02,
                ),
                n_estimators=n_estimators,
                max_features=1.0, max_samples=1.0, oob_score=False,
                n_jobs=1, random_state=42,
            )
            clf_drop.fit(X_train.values[:, cols_mask], y_train)
            drop_proba = clf_drop.predict_proba(X_test.values[:, cols_mask])
            drop_ll = log_loss(y_test, drop_proba)
            lifts.append({
                "feature": feat,
                "base_loss": base_ll,
                "drop_loss": drop_ll,
                "lift": drop_ll - base_ll,
            })

        df = pd.DataFrame(lifts).set_index("feature")
        df["label"] = (df["lift"] > 0).astype(int)
        results[s] = df

    return results


def build_ground_truth_from_s3(
    base_path: str = "/tmp/importance_s3/home_win",
) -> dict:
    """Build ground truth labels from all available raw importance CSVs.

    Returns dict with structure:
      {test_name: {split_point: DataFrame}}
    """
    tests = {
        "sfi": {"file": "importance_sfi_raw.csv", "null": np.log(0.5)},
        "desub_mda": {"file": "importance_desub_mda_raw.csv", "null": 0.0},
        "pca_mda": {"file": "importance_pca_mda_raw.csv", "null": 0.0},
        "resid_mda": {"file": "importance_resid_mda_raw.csv", "null": 0.0},
    }

    all_labels = {}
    for test_name, cfg in tests.items():
        fpath = Path(base_path) / cfg["file"]
        if not fpath.exists():
            continue
        raw = pd.read_csv(fpath, index_col=0)
        labels = compute_ground_truth_labels(raw, null=cfg["null"])
        all_labels[test_name] = labels

    return all_labels


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    print("Computing ground truth labels from held-out fold medians...")
    all_labels = build_ground_truth_from_s3()

    print(f"\n{'='*70}")
    print(f"  GROUND TRUTH LABELS (held-out fold median > null)")
    print(f"{'='*70}")

    for test_name, splits in all_labels.items():
        print(f"\n  {test_name.upper()}:")
        for s, df in sorted(splits.items()):
            n_pos = df["label"].sum()
            n_neg = (1 - df["label"]).sum()
            n_total = len(df)
            print(f"    Split {s} (folds 1..{s} gate, {8-s} held-out): "
                  f"SIGNAL={n_pos} ({n_pos/n_total*100:.1f}%), "
                  f"NOISE={n_neg} ({n_neg/n_total*100:.1f}%)")

    # Cross-split stability
    print(f"\n{'='*70}")
    print(f"  CROSS-SPLIT STABILITY (agreement across split points)")
    print(f"{'='*70}")
    for test_name, splits in all_labels.items():
        split_keys = sorted(splits.keys())
        if len(split_keys) < 2:
            continue
        labels_matrix = pd.DataFrame({
            f"split_{s}": splits[s]["label"] for s in split_keys
        })
        # Fraction of features that have the same label across ALL splits
        all_agree = (labels_matrix.nunique(axis=1) == 1).mean()
        # Pairwise agreement
        agreements = []
        for i in range(len(split_keys)):
            for j in range(i+1, len(split_keys)):
                a = labels_matrix.iloc[:, i]
                b = labels_matrix.iloc[:, j]
                agreements.append(float((a == b).mean()))
        print(f"  {test_name:<12}: all-agree={all_agree:.3f}, "
              f"pairwise-agree={np.mean(agreements):.3f} "
              f"(n_splits={len(split_keys)})")
