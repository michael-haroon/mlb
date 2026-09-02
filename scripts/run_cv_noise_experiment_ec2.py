"""CV noise experiment: empirically compare single-fold vs sliding-window vs expanding.

Measures permutation importance stability under 4 CV training regimes, holding
the test year constant at 2026. For each regime, trains one RF and runs 50
independent permutation repeats per feature.

Outputs per (target × mode):
  - Full repeat matrix: (n_repeats × n_features) — raw per-repeat importance
  - Split-half Spearman: rank correlation between repeat halves (reliability)
  - Summary stats: within-feature SD, top-K overlap across halves

Cross-mode outputs:
  - Top-50 Jaccard overlap between modes (same target)
  - Rank correlation of mean importance between modes

Usage (EC2):
    python3.11 scripts/run_cv_noise_experiment_ec2.py

Instance: c8g.8xlarge (32 vCPU ARM64)
Expected runtime: ~20-30 min
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import boto3
from joblib import Parallel, delayed
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.strategy.config import SKIP_SEASONS, TARGETS_CLASSIFICATION
from classical_learning.strategy.data import load_features, compute_temporal_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BUCKET = "mlb-265753586044-us-east-1-an"
FEATURES_KEY = "data/features/game_features.parquet"
OUTPUT_PREFIX = "classical_learning/artifacts/importance/cv_noise_experiment"

TARGETS = ["home_win", "total_runs"]
TEST_YEAR = 2026
N_REPEATS = 200
N_ESTIMATORS = 300

CV_MODES = {
    "single_fold": {
        "description": "Train on test_year-1 only",
        "train_years_fn": lambda test_year, all_years: [test_year - 1],
    },
    "sliding_2": {
        "description": "Train on 2 most recent years before test",
        "train_years_fn": lambda test_year, all_years: sorted(
            [y for y in all_years if y < test_year and y not in SKIP_SEASONS]
        )[-2:],
    },
    "sliding_3": {
        "description": "Train on 3 most recent years before test",
        "train_years_fn": lambda test_year, all_years: sorted(
            [y for y in all_years if y < test_year and y not in SKIP_SEASONS]
        )[-3:],
    },
    "expanding": {
        "description": "Train on all prior years (current pipeline)",
        "train_years_fn": lambda test_year, all_years: sorted(
            [y for y in all_years if y < test_year and y not in SKIP_SEASONS]
        ),
    },
}


def get_n_workers() -> int:
    return max(1, os.cpu_count() - 2)


def build_and_fit_rf(X_train, y_train, sample_weight, regression: bool):
    """Build and fit a de Prado-style bagged RF.

    Uses n_jobs=-1 for training (fast), but predictions in the permutation
    loop are single-threaded per worker — joblib parallelizes over repeats.
    """
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import BaggingClassifier, BaggingRegressor

    if regression:
        base = DecisionTreeRegressor(max_features=1, min_weight_fraction_leaf=0.02)
        clf = BaggingRegressor(
            estimator=base, n_estimators=N_ESTIMATORS,
            max_features=1.0, max_samples=1.0, oob_score=False,
            n_jobs=-1, random_state=42,
        )
    else:
        base = DecisionTreeClassifier(
            criterion="entropy", max_features=1,
            class_weight="balanced", min_weight_fraction_leaf=0.02,
        )
        clf = BaggingClassifier(
            estimator=base, n_estimators=N_ESTIMATORS,
            max_features=1.0, max_samples=1.0, oob_score=False,
            n_jobs=-1, random_state=42,
        )

    clf.fit(X_train, y_train, sample_weight=sample_weight)
    # Switch to single-threaded predict for parallelization over repeats
    clf.n_jobs = 1
    return clf


def score_model(clf, X_test, y_test, scoring: str, all_labels):
    """Score a fitted model on test data."""
    if scoring == "r2":
        from sklearn.metrics import r2_score
        return r2_score(y_test, clf.predict(X_test))
    else:
        from sklearn.metrics import log_loss
        prob = clf.predict_proba(X_test)
        # Align columns to all_labels
        if hasattr(clf, 'classes_'):
            classes = clf.classes_
        else:
            classes = clf.estimators_[0].classes_
        if not np.array_equal(classes, all_labels):
            aligned = np.zeros((prob.shape[0], len(all_labels)))
            for i, c in enumerate(classes):
                idx = np.where(all_labels == c)[0][0]
                aligned[:, idx] = prob[:, i]
            prob = aligned
        return -log_loss(y_test, prob, labels=all_labels)


def _one_repeat(clf, X_test_arr, y_test_arr, col_names, scoring, all_labels,
                base_score, rep_seed):
    """One permutation repeat: shuffle each feature, score, return importance row."""
    from sklearn.metrics import log_loss, r2_score
    n_features = len(col_names)
    row = np.empty(n_features)
    rng = np.random.default_rng(rep_seed)

    is_regression = (scoring == "r2")

    for fi in range(n_features):
        X_perm = X_test_arr.copy()
        X_perm[:, fi] = rng.permutation(X_perm[:, fi])
        if is_regression:
            pred = clf.predict(X_perm)
            perm_score = r2_score(y_test_arr, pred)
        else:
            prob = clf.predict_proba(X_perm)
            if hasattr(clf, 'classes_') and not np.array_equal(clf.classes_, all_labels):
                aligned = np.zeros((prob.shape[0], len(all_labels)))
                for ci, c in enumerate(clf.classes_):
                    idx = np.where(all_labels == c)[0][0]
                    aligned[:, idx] = prob[:, ci]
                prob = aligned
            perm_score = -log_loss(y_test_arr, prob, labels=all_labels)
        row[fi] = base_score - perm_score

    return row


def run_mda_full_repeats(clf, X_test, y_test, scoring: str, all_labels,
                         n_repeats: int, col_names: list[str]) -> np.ndarray:
    """Run MDA with full per-repeat output, parallelized over repeats.

    Returns: importance matrix of shape (n_repeats, n_features)
        Each entry = base_score - permuted_score for that (repeat, feature).
    """
    base_score = score_model(clf, X_test, y_test, scoring, all_labels)
    log.info(f"  Base score ({scoring}): {base_score:.6f}")

    X_test_arr = X_test.values if hasattr(X_test, 'values') else X_test
    y_test_arr = y_test.values if hasattr(y_test, 'values') else y_test

    n_workers = get_n_workers()
    log.info(f"  Parallelizing {n_repeats} repeats across {n_workers} workers...")

    rows = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_one_repeat)(
            clf, X_test_arr, y_test_arr, col_names, scoring, all_labels,
            base_score, rep * 7919
        )
        for rep in range(n_repeats)
    )

    importance_matrix = np.vstack(rows)
    return importance_matrix


def compute_diagnostics(importance_matrix: np.ndarray, col_names: list[str]) -> dict:
    """Compute split-half stability and noise diagnostics from repeat matrix."""
    n_repeats = importance_matrix.shape[0]
    half = n_repeats // 2

    # Split-half means
    mean_a = importance_matrix[:half].mean(axis=0)
    mean_b = importance_matrix[half:2*half].mean(axis=0)
    mean_full = importance_matrix.mean(axis=0)

    # Split-half Spearman (all features)
    rho_all, p_all = spearmanr(mean_a, mean_b)

    # Split-half Spearman (top-100 by full mean)
    top100_idx = np.argsort(mean_full)[-100:]
    rho_top100, p_top100 = spearmanr(mean_a[top100_idx], mean_b[top100_idx])

    # Top-50 overlap between halves
    top50_a = set(np.argsort(mean_a)[-50:])
    top50_b = set(np.argsort(mean_b)[-50:])
    top50_jaccard = len(top50_a & top50_b) / 50

    # Top-25 overlap
    top25_a = set(np.argsort(mean_a)[-25:])
    top25_b = set(np.argsort(mean_b)[-25:])
    top25_jaccard = len(top25_a & top25_b) / 25

    # Within-feature SD (permutation noise)
    within_sd = importance_matrix.std(axis=0, ddof=1)
    # Signal-to-noise: |mean| / SD
    snr = np.abs(mean_full) / np.where(within_sd > 0, within_sd, 1e-10)

    # Top-50 features by mean importance
    top50_idx = np.argsort(mean_full)[-50:][::-1]
    top50_features = [col_names[i] for i in top50_idx]
    top50_means = mean_full[top50_idx]
    top50_sds = within_sd[top50_idx]
    top50_snrs = snr[top50_idx]

    return {
        "split_half_rho_all": float(rho_all),
        "split_half_p_all": float(p_all),
        "split_half_rho_top100": float(rho_top100),
        "split_half_p_top100": float(p_top100),
        "top50_jaccard": float(top50_jaccard),
        "top25_jaccard": float(top25_jaccard),
        "mean_within_sd": float(within_sd.mean()),
        "median_within_sd": float(np.median(within_sd)),
        "mean_snr_top50": float(snr[top50_idx].mean()),
        "median_snr_top50": float(np.median(snr[top50_idx])),
        "n_features_positive": int((mean_full > 0).sum()),
        "n_features_significantly_positive": int(
            ((mean_full > 0) & (mean_full > 2 * within_sd / np.sqrt(n_repeats))).sum()
        ),
        "base_score": float(mean_full.sum()),  # not meaningful per se, but for reference
        "top50_features": top50_features,
        "top50_means": top50_means.tolist(),
        "top50_sds": top50_sds.tolist(),
        "top50_snrs": top50_snrs.tolist(),
    }


def compute_cross_mode_diagnostics(results: dict, target: str) -> dict:
    """Compare importance rankings across CV modes for one target."""
    modes = list(CV_MODES.keys())
    cross = {}

    for i, mode_a in enumerate(modes):
        key_a = f"{target}_{mode_a}"
        if key_a not in results:
            continue
        mean_a = results[key_a]["importance_matrix"].mean(axis=0)

        for mode_b in modes[i+1:]:
            key_b = f"{target}_{mode_b}"
            if key_b not in results:
                continue
            mean_b = results[key_b]["importance_matrix"].mean(axis=0)

            pair = f"{mode_a}_vs_{mode_b}"

            # Full Spearman
            rho, p = spearmanr(mean_a, mean_b)
            cross[f"{pair}_rho_all"] = float(rho)

            # Top-100 Spearman
            union_top100 = set(np.argsort(mean_a)[-100:]) | set(np.argsort(mean_b)[-100:])
            union_idx = list(union_top100)
            rho_top, _ = spearmanr(mean_a[union_idx], mean_b[union_idx])
            cross[f"{pair}_rho_top100_union"] = float(rho_top)

            # Top-50 Jaccard
            top50_a = set(np.argsort(mean_a)[-50:])
            top50_b = set(np.argsort(mean_b)[-50:])
            cross[f"{pair}_top50_jaccard"] = len(top50_a & top50_b) / 50

            # Top-25 Jaccard
            top25_a = set(np.argsort(mean_a)[-25:])
            top25_b = set(np.argsort(mean_b)[-25:])
            cross[f"{pair}_top25_jaccard"] = len(top25_a & top25_b) / 25

    return cross


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--mode", required=True, choices=list(CV_MODES.keys()))
    args = parser.parse_args()

    target = args.target
    mode_name = args.mode
    mode_cfg = CV_MODES[mode_name]

    task = "classification" if target in TARGETS_CLASSIFICATION else "regression"
    regression = (task == "regression")
    scoring = "r2" if regression else "log_loss"

    t_start = time.time()

    log.info(f"{'='*70}")
    log.info(f"TARGET: {target} ({task}) | MODE: {mode_name} ({mode_cfg['description']})")
    log.info(f"{'='*70}")

    s3 = boto3.client("s3")

    # Download features
    log.info(f"Downloading s3://{BUCKET}/{FEATURES_KEY} ...")
    obj = s3.get_object(Bucket=BUCKET, Key=FEATURES_KEY)
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.write(obj["Body"].read())
    tmp.close()
    features_path = Path(tmp.name)

    X, y, seasons, _ = load_features(features_path, target, "2015+")

    # Drop >95% NaN columns
    nan_pct = X.isna().mean()
    valid_cols = nan_pct[nan_pct < 0.95].index.tolist()
    X = X[valid_cols]
    col_names = list(X.columns)
    log.info(f"Features: {len(col_names)}, Samples: {len(X):,}")

    all_years = sorted(seasons.unique())

    # Test set
    test_mask = seasons == TEST_YEAR
    X_test = X[test_mask].copy()
    y_test = y[test_mask].copy()
    log.info(f"Test set (year={TEST_YEAR}): {len(X_test):,} games")

    all_labels = np.unique(y.values) if not regression else None

    # Train set
    train_years = mode_cfg["train_years_fn"](TEST_YEAR, all_years)
    log.info(f"Train years: {train_years}")

    train_mask = seasons.isin(train_years)
    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()
    seasons_train = seasons[train_mask]
    sample_weight = compute_temporal_weights(seasons_train)

    log.info(f"Train: {len(X_train):,} games across {len(train_years)} seasons")

    # NaN fill: median from train
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test_filled = X_test.fillna(medians)

    # Train RF
    t0 = time.time()
    clf = build_and_fit_rf(X_train, y_train, sample_weight, regression)
    log.info(f"RF trained in {time.time() - t0:.1f}s")

    # Run full-repeat MDA
    t0 = time.time()
    importance_matrix = run_mda_full_repeats(
        clf, X_test_filled, y_test, scoring, all_labels,
        N_REPEATS, col_names)
    elapsed = time.time() - t0
    log.info(f"MDA ({N_REPEATS} repeats × {len(col_names)} features) in {elapsed:.1f}s")

    # Compute diagnostics
    diag = compute_diagnostics(importance_matrix, col_names)
    diag["mode"] = mode_name
    diag["target"] = target
    diag["n_train"] = len(X_train)
    diag["n_test"] = len(X_test)
    diag["train_years"] = train_years
    diag["elapsed_s"] = round(elapsed, 1)

    log.info(f"Split-half ρ (all features): {diag['split_half_rho_all']:.4f}")
    log.info(f"Split-half ρ (top-100): {diag['split_half_rho_top100']:.4f}")
    log.info(f"Top-50 Jaccard (half A vs B): {diag['top50_jaccard']:.3f}")
    log.info(f"Top-25 Jaccard (half A vs B): {diag['top25_jaccard']:.3f}")
    log.info(f"Mean SNR (top-50): {diag['mean_snr_top50']:.3f}")
    log.info(f"Significantly positive features: {diag['n_features_significantly_positive']}")

    # Upload results to S3
    key = f"{target}_{mode_name}"
    log.info(f"Uploading results to s3://{BUCKET}/{OUTPUT_PREFIX}/{key}/ ...")

    # Repeat matrix
    matrix_df = pd.DataFrame(importance_matrix, columns=col_names)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        matrix_df.to_csv(f.name, index=False)
        s3.upload_file(f.name, BUCKET, f"{OUTPUT_PREFIX}/{key}_repeat_matrix.csv")

    # Diagnostics
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(diag, f, indent=2, default=str)
        s3.upload_file(f.name, BUCKET, f"{OUTPUT_PREFIX}/{key}_diagnostics.json")

    elapsed_total = time.time() - t_start
    log.info(f"DONE in {elapsed_total/60:.1f} minutes")


if __name__ == "__main__":
    main()
