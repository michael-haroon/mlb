"""Label Integrity Diagnostics for feature importance pipeline.

Measures how often the importance tests produce wrong pass/fail labels by:
  1a. Synthetic validation: inject known-informative and noise features into the
      real feature matrix, run all 5 primary tests, report FPR and FNR per test.
  1c. PCA-MDA vs Resid-MDA agreement analysis: load the feature report (with
      margins from 1b) and identify borderline disagreements.

Usage:
    conda run -n pred python -m pregame.analysis.label_integrity
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
#  Logging setup
# ─────────────────────────────────────────────────────────────────────────────

log = logging.getLogger(__name__)


def _setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # stdout handler at INFO
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                      datefmt="%H:%M:%S"))
    root.addHandler(sh)

    # File handler at DEBUG
    log_path = Path("pregame/analysis/label_integrity.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(fh)


# ─────────────────────────────────────────────────────────────────────────────
#  1a. Synthetic validation across all 5 primary tests
# ─────────────────────────────────────────────────────────────────────────────

def run_synthetic_validation(
    features_path: Path = None,
    n_informative: int = 5,
    n_noise: int = 5,
    n_real_features: int = 30,
    random_state: int = 42,
) -> dict:
    """Inject known-informative and noise features into the real feature matrix.

    Runs MDI, SFI, desub-MDA, PCA-MDA, and resid-MDA on the combined matrix
    and reports per-test False Positive Rate (noise that passes) and False
    Negative Rate (informative that fails).

    Uses 12 LOYO folds matching the production pipeline.
    """
    from .feature_importance import (
        PurgedYearKFold,
        bootstrap_ci,
        build_rf,
        compute_shared_clustering,
        feat_imp_desub_mda,
        feat_imp_mda,
        feat_imp_mdi,
        feat_imp_pca_mda,
        feat_imp_residual_mda,
        feat_imp_sfi,
        filter_features,
    )
    from ..strategy.data import compute_temporal_weights, load_features

    # Resolve features path relative to this file's location in the repo tree
    if features_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        features_path = repo_root / "pregame" / "artifacts" / "features" / "game_features.parquet"
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(
            f"game_features.parquet not found at {features_path}. "
            "Pass features_path explicitly or run from the repo root."
        )

    log.info("=" * 70)
    log.info("LABEL INTEGRITY: Synthetic Validation")
    log.info("=" * 70)

    # ── Load real features ───────────────────────────────────────────────
    log.info("Loading real features...")
    X_real, y, seasons, _game_pks = load_features(features_path, "home_win", "2015+")
    sample_weight = compute_temporal_weights(seasons)

    # Drop >95% NaN columns (matching production)
    nan_pct = X_real.isna().mean()
    valid_cols = nan_pct[nan_pct < 0.95].index.tolist()
    X_real = X_real[valid_cols]

    # Subsample real features to keep runtime tractable.
    # Select top features by variance (captures real distributional diversity
    # without needing the full 288-feature matrix for the diagnostic).
    if n_real_features is not None and X_real.shape[1] > n_real_features:
        variances = X_real.var().sort_values(ascending=False)
        top_cols = variances.head(n_real_features).index.tolist()
        X_real = X_real[top_cols]
        log.info(f"Subsampled to top-{n_real_features} real features by variance")

    log.info(f"Real features: {X_real.shape[1]} cols, {len(X_real):,} samples, "
             f"{seasons.nunique()} seasons")

    # ── Generate synthetic features ──────────────────────────────────────
    rng = np.random.default_rng(random_state)
    n_samples = len(X_real)

    # Calibrate informative signal strength to match real top features.
    # consensus_home_win_prob typically has Pearson r ~ 0.15-0.20 with target.
    # We use weight=0.6 + noise(std=1.8) to achieve similar magnitude.
    y_vals = y.values.astype(float)

    informative_cols = {}
    for i in range(n_informative):
        # weight calibrated so feature has moderate predictive power
        weight = 0.5 + 0.2 * i  # range 0.5 to 1.3
        noise_std = 1.8
        feat = y_vals * weight + rng.normal(0, noise_std, n_samples)
        informative_cols[f"SYNTH_INFO_{i}"] = feat

    noise_cols = {}
    for i in range(n_noise):
        noise_cols[f"SYNTH_NOISE_{i}"] = rng.normal(0, 1, n_samples)

    X_synth_info = pd.DataFrame(informative_cols, index=X_real.index)
    X_synth_noise = pd.DataFrame(noise_cols, index=X_real.index)

    # Combine: real features + synthetic informative + synthetic noise
    X_combined = pd.concat([X_real, X_synth_info, X_synth_noise], axis=1)
    log.info(f"Combined matrix: {X_combined.shape[1]} features "
             f"({X_real.shape[1]} real + {n_informative} informative + {n_noise} noise)")

    # Verify signal injection
    for col in X_synth_info.columns:
        r = np.corrcoef(X_combined[col].values, y_vals)[0, 1]
        log.debug(f"  {col}: Pearson r = {r:.4f}")

    # ── Cluster the combined features ────────────────────────────────────
    log.info("Computing ONC clustering on combined matrix...")
    X_for_clustering = X_combined.fillna(X_combined.median())
    clustering = compute_shared_clustering(X_for_clustering)
    clusters = clustering["clusters"]
    log.info(f"  {len(clusters)} clusters found")

    # ── Run all importance methods ───────────────────────────────────────
    # Use 12 LOYO folds (matching production with 2015-2026 data)
    n_jobs_full = -1

    # 1. MDI (in-sample, needs a fitted RF)
    log.info("Running MDI...")
    X_filled = X_combined.fillna(X_combined.median())
    clf = build_rf(n_estimators=1000, n_jobs=n_jobs_full)
    clf.fit(X_filled, y, sample_weight=sample_weight)
    mdi_summary, mdi_raw = feat_imp_mdi(clf, list(X_combined.columns))

    # 2. SFI
    log.info("Running SFI...")
    sfi_summary, sfi_raw = feat_imp_sfi(
        build_rf(n_estimators=300, n_jobs=1),
        X_combined, y, seasons, sample_weight,
    )
    null_col = "null_log_loss"
    sfi_null = sfi_summary[null_col].iloc[0] if null_col in sfi_summary.columns else 0.0

    # 3. De-substituted MDA
    log.info("Running desub-MDA...")
    desub_summary, desub_raw = feat_imp_desub_mda(
        X_combined, y, seasons, clusters,
        sample_weight=sample_weight,
        scoring="log_loss",
        n_estimators=300,
    )

    # 4. PCA-MDA
    log.info("Running PCA-MDA...")
    pca_mda_summary, pca_mda_raw, _ = feat_imp_pca_mda(
        X_combined, y, seasons,
        sample_weight=sample_weight,
        scoring="log_loss",
        n_estimators=300,
    )

    # 5. Residualized MDA
    log.info("Running resid-MDA...")
    resid_summary, resid_raw = feat_imp_residual_mda(
        X_combined, y, seasons, clusters,
        sample_weight=sample_weight,
        scoring="log_loss",
        n_estimators=300,
    )

    # ── Apply the same filter logic as production ────────────────────────
    log.info("Running filter_features (production logic)...")
    from .feature_importance import filter_features as _ff
    # We need the CFI-MDA raw too for filter_features, but it's cluster-level.
    # For synthetic validation we only care about the 5 individual tests,
    # so pass cfi_mda_raw=None (CFI-MDA is cluster-level, not used for
    # individual feature pass/fail).
    report = _ff(
        mdi_raw=mdi_raw,
        cfi_mda_raw=None,
        clusters=clusters,
        sfi_raw=sfi_raw,
        sfi_null=sfi_null,
        desub_mda_raw=desub_raw,
        pca_mda_raw=pca_mda_raw,
        resid_mda_raw=resid_raw,
    )

    # ── Evaluate FPR / FNR per test ──────────────────────────────────────
    info_feats = list(X_synth_info.columns)
    noise_feats = list(X_synth_noise.columns)

    test_cols = {
        "MDI": "mdi_passes",
        "SFI": "sfi_passes",
        "desub_MDA": "desub_mda_passes",
        "PCA_MDA": "pca_mda_passes",
        "resid_MDA": "resid_mda_passes",
    }

    results = {}
    log.info("")
    log.info("=" * 50)
    log.info("RESULTS: Per-test FPR and FNR")
    log.info("=" * 50)

    for test_name, col in test_cols.items():
        if col not in report.columns:
            log.warning(f"  {test_name}: column {col} not in report, skipping")
            results[test_name] = {"fpr": np.nan, "fnr": np.nan}
            continue

        # FPR: fraction of noise features that pass
        noise_passes = report.loc[
            report.index.isin(noise_feats), col
        ].fillna(False).astype(bool)
        fpr = noise_passes.sum() / len(noise_feats) if len(noise_feats) > 0 else 0.0

        # FNR: fraction of informative features that fail
        info_passes = report.loc[
            report.index.isin(info_feats), col
        ].fillna(False).astype(bool)
        fnr = (~info_passes).sum() / len(info_feats) if len(info_feats) > 0 else 0.0

        results[test_name] = {"fpr": float(fpr), "fnr": float(fnr)}
        log.info(f"  {test_name:12s}  FPR={fpr:.2%} ({noise_passes.sum()}/{len(noise_feats)} noise pass)  "
                 f"FNR={fnr:.2%} ({(~info_passes).sum()}/{len(info_feats)} info fail)")

    log.info("")
    return {
        "per_test": results,
        "report": report,
        "info_feats": info_feats,
        "noise_feats": noise_feats,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  1c. PCA-MDA vs Resid-MDA agreement analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_agreement_analysis(
    report_path: Path = None,
) -> dict:
    """Analyze PCA-MDA vs Resid-MDA agreement from the feature report.

    Loads the feature report (with margin columns from 1b), computes agreement
    rate, lists disagreeing features, and shows both margins for disagreements.

    Returns dict with:
        agreement_rate: float
        n_agree: int
        n_disagree: int
        disagreements: DataFrame with both pass columns and margins
    """
    # Resolve default path relative to repo root
    if report_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        report_path = repo_root / "pregame" / "artifacts" / "importance" / "v10_canary" / "home_win" / "filtered" / "feature_report.csv"
    report_path = Path(report_path)

    log.info("=" * 70)
    log.info("PCA-MDA vs Resid-MDA Agreement Analysis")
    log.info("=" * 70)

    report = pd.read_csv(report_path, index_col="feature")

    # Check required columns exist
    required = ["pca_mda_passes", "resid_mda_passes"]
    for col in required:
        if col not in report.columns:
            raise ValueError(f"Column {col!r} not in feature report. Run importance pipeline first.")

    # Filter to features where both tests have results (not NaN)
    both_available = report[
        report["pca_mda_passes"].notna() & report["resid_mda_passes"].notna()
    ].copy()
    n_total = len(both_available)

    if n_total == 0:
        log.warning("No features have both PCA-MDA and resid-MDA results.")
        return {"agreement_rate": np.nan, "n_agree": 0, "n_disagree": 0,
                "disagreements": pd.DataFrame()}

    # Convert to boolean
    both_available["pca_mda_passes"] = both_available["pca_mda_passes"].astype(bool)
    both_available["resid_mda_passes"] = both_available["resid_mda_passes"].astype(bool)

    agree_mask = both_available["pca_mda_passes"] == both_available["resid_mda_passes"]
    n_agree = agree_mask.sum()
    n_disagree = n_total - n_agree
    agreement_rate = n_agree / n_total

    log.info(f"  Total features with both tests: {n_total}")
    log.info(f"  Agreement: {n_agree} ({agreement_rate:.1%})")
    log.info(f"  Disagreement: {n_disagree} ({1 - agreement_rate:.1%})")

    # Build disagreement table
    disagree_feats = both_available[~agree_mask].copy()

    # Include margin columns if available
    margin_cols = ["pca_mda_margin", "resid_mda_margin"]
    display_cols = ["pca_mda_passes", "resid_mda_passes",
                    "pca_mda_mean", "resid_mda_mean"]
    for mc in margin_cols:
        if mc in disagree_feats.columns:
            display_cols.append(mc)

    disagree_display = disagree_feats[
        [c for c in display_cols if c in disagree_feats.columns]
    ].copy()

    if len(disagree_display) > 0:
        log.info("")
        log.info("  Disagreeing features (one passes, other fails):")
        log.info("  " + "-" * 60)

        # Classify disagreement type
        for feat in disagree_display.index:
            pca_pass = disagree_display.loc[feat, "pca_mda_passes"]
            resid_pass = disagree_display.loc[feat, "resid_mda_passes"]
            pca_margin = disagree_display.loc[feat, "pca_mda_margin"] if "pca_mda_margin" in disagree_display.columns else np.nan
            resid_margin = disagree_display.loc[feat, "resid_mda_margin"] if "resid_mda_margin" in disagree_display.columns else np.nan

            # Determine if disagreement is noise (near-zero margin on failing side)
            failing_margin = resid_margin if pca_pass else pca_margin
            borderline = abs(failing_margin) < 0.01 if not np.isnan(failing_margin) else False
            tag = " [BORDERLINE]" if borderline else ""

            direction = "PCA passes, resid fails" if pca_pass else "resid passes, PCA fails"
            log.info(f"    {feat}: {direction}  "
                     f"(pca_margin={pca_margin:.6f}, resid_margin={resid_margin:.6f}){tag}")

    result = {
        "agreement_rate": float(agreement_rate),
        "n_agree": int(n_agree),
        "n_disagree": int(n_disagree),
        "n_total": int(n_total),
        "disagreements": disagree_display,
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _setup_logging()

    log.info("Label Integrity Diagnostics")
    log.info("=" * 70)

    # ── 1a: Synthetic validation ─────────────────────────────────────────
    synth_results = run_synthetic_validation()

    log.info("")
    log.info("SUMMARY — Synthetic Validation:")
    for test, metrics in synth_results["per_test"].items():
        log.info(f"  {test:12s}  FPR={metrics['fpr']:.2%}  FNR={metrics['fnr']:.2%}")

    # ── 1c: Agreement analysis ───────────────────────────────────────────
    # Try v10_canary first, fall back to home_win
    repo_root = Path(__file__).resolve().parents[2]
    report_paths = [
        repo_root / "pregame" / "artifacts" / "importance" / "v10_canary" / "home_win" / "filtered" / "feature_report.csv",
        repo_root / "pregame" / "artifacts" / "importance" / "home_win" / "filtered" / "feature_report.csv",
    ]
    agreement_result = None
    for rp in report_paths:
        if rp.exists():
            try:
                agreement_result = run_agreement_analysis(rp)
                break
            except (ValueError, KeyError) as e:
                log.warning(f"Could not run agreement analysis on {rp}: {e}")

    if agreement_result is None:
        log.warning("No feature report with margins found. "
                    "Run the importance pipeline with margin columns (1b) first.")

    # ── Final summary ────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("FINAL SUMMARY")
    log.info("=" * 70)
    log.info("Synthetic FPR/FNR:")
    for test, metrics in synth_results["per_test"].items():
        log.info(f"  {test:12s}  FPR={metrics['fpr']:.2%}  FNR={metrics['fnr']:.2%}")
    if agreement_result:
        log.info(f"PCA vs Resid agreement: {agreement_result['agreement_rate']:.1%} "
                 f"({agreement_result['n_agree']}/{agreement_result['n_total']})")
        log.info(f"  Disagreements: {agreement_result['n_disagree']}")


if __name__ == "__main__":
    main()
