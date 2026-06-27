"""Orchestrate the full de Prado feature importance pipeline.

Runs all importance methods (MDI, MDA, SFI, CFI, de-substituted MDA,
PCA-MDA, residualized MDA), performs ONC clustering, statistical
significance testing, and evidence-based feature routing.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..strategy.config import TARGETS_CLASSIFICATION
from ..strategy.data import compute_temporal_weights, load_features
from .feature_importance import (
    compute_shared_clustering,
    plot_cfi_mda_distributions,
    run_all_importance,
    synthetic_validation,
)
from .feature_routing import route_features, write_routing_report

log = logging.getLogger(__name__)


def run_importance_analysis(
    features_path: Path,
    output_dir: Path,
    target: str,
    data_mode: str = "2015+",
    run_sfi: bool = True,
    run_desub_mda: bool = True,
    run_pca_mda: bool = True,
    run_residual_mda: bool = True,
) -> dict:
    """Run the full de Prado feature importance pipeline for one target.

    Pipeline:
      1. Load features + drop >95% NaN columns
      2. Synthetic validation (sanity check that MDI recovers known signal)
      3. Target-independent ONC clustering (denoise + detone + cluster)
      4. Run all importance methods (MDI, SFI, desub-MDA, PCA-MDA, resid-MDA, CFI)
      5. Algorithmic filtering (ACCEPTED / NEEDS SPECIFICATION / REJECTED)
      6. Evidence-based routing to model families
      7. Save all artifacts

    Parameters
    ----------
    features_path : Path
        Path to game_features.parquet.
    output_dir : Path
        Root output directory. Artifacts go into output_dir/{target}/.
    target : str
        Target column name.
    data_mode : str
        "2015+" or "all".
    run_sfi, run_desub_mda, run_pca_mda, run_residual_mda : bool
        Toggle individual methods on/off.

    Returns
    -------
    dict with summary statistics.
    """
    target_dir = Path(output_dir) / target
    target_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir = target_dir / "filtered"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    task = "classification" if target in TARGETS_CLASSIFICATION else "regression"
    regression = (task == "regression")
    log.info(f"=" * 70)
    log.info(f"Feature importance pipeline: target={target}, task={task}, mode={data_mode}")
    log.info(f"  Methods: SFI={run_sfi}, desub_MDA={run_desub_mda}, "
             f"PCA_MDA={run_pca_mda}, resid_MDA={run_residual_mda}")
    log.info(f"=" * 70)

    t0 = time.time()

    # ── Step 1: Load features ────────────────────────────────────────────
    log.info("Loading features...")
    X, y, seasons = load_features(features_path, target, data_mode)
    sample_weight = compute_temporal_weights(seasons)

    # Drop columns with >95% NaN
    nan_pct = X.isna().mean()
    valid_cols = nan_pct[nan_pct < 0.95].index.tolist()
    dropped = X.shape[1] - len(valid_cols)
    X = X[valid_cols]
    log.info(f"Features: {X.shape[1]} columns, {len(X):,} samples "
             f"(dropped {dropped} cols with >95% NaN)")

    # Fill NaN with median for importance analysis
    # (de Prado's RF uses sklearn BaggingClassifier which cannot handle NaN)
    X_filled = X.fillna(X.median())

    # ── Step 2: Synthetic validation ─────────────────────────────────────
    log.info("Running synthetic validation...")
    synth = synthetic_validation()
    if not synth["mdi_pass"]:
        log.warning("SYNTHETIC VALIDATION FAILED: MDI cannot recover known signal. "
                    "Results may be unreliable.")
    else:
        log.info("Synthetic validation passed: MDI correctly ranks informative > noise")

    # ── Step 3: Target-independent clustering ────────────────────────────
    log.info("Computing target-independent ONC clustering...")
    clustering = compute_shared_clustering(X_filled)
    clusters = clustering["clusters"]
    denoising_info = clustering["denoising_info"]

    # Save clustering artifacts
    cluster_map = {str(k): v for k, v in clusters.items()}
    with open(target_dir / "cluster_map.json", "w") as f:
        json.dump(cluster_map, f, indent=2)
    with open(target_dir / "denoising_report.json", "w") as f:
        json.dump(denoising_info, f, indent=2)

    # ── Step 4-10: Run all importance methods ────────────────────────────
    results = run_all_importance(
        X=X_filled,
        y=y,
        years=seasons,
        sample_weight=sample_weight,
        run_sfi=run_sfi,
        run_desub_mda=run_desub_mda,
        run_pca_mda=run_pca_mda,
        run_residual_mda=run_residual_mda,
        regression=regression,
        precomputed=clustering,
    )

    # ── Save all raw artifacts ───────────────────────────────────────────
    log.info("Saving artifacts...")

    results["summary"].to_csv(target_dir / "importance_summary.csv")
    results["mdi_raw"].to_csv(target_dir / "importance_mdi_raw.csv")

    if results["sfi_raw"] is not None:
        results["sfi_raw"].to_csv(target_dir / "importance_sfi_raw.csv")
    if results["desub_mda_raw"] is not None:
        results["desub_mda_raw"].to_csv(target_dir / "importance_desub_mda_raw.csv")
    if results["pca_mda_raw"] is not None:
        results["pca_mda_raw"].to_csv(target_dir / "importance_pca_mda_raw.csv")
    if results["resid_mda_raw"] is not None:
        results["resid_mda_raw"].to_csv(target_dir / "importance_resid_mda_raw.csv")
    if results["cfi_mda_raw"] is not None:
        results["cfi_mda_raw"].to_csv(target_dir / "importance_cfi_mda_raw.csv")

    # PCA cross-check
    if results["pca_info"] is not None:
        results["pca_info"].to_csv(target_dir / "pca_cross_check.csv")
    if results["tau_results"] is not None:
        with open(target_dir / "kendall_tau.json", "w") as f:
            json.dump(results["tau_results"], f, indent=2)

    # CFI-MDA distribution plot
    if results["cfi_mda_raw"] is not None:
        try:
            plot_cfi_mda_distributions(
                results["cfi_mda_raw"], clusters,
                output_path=str(target_dir / "cfi_mda_distributions.png"),
            )
        except Exception as e:
            log.warning(f"Could not generate CFI-MDA distribution plot: {e}")

    # ── Save filter report ───────────────────────────────────────────────
    filter_report = results["filter_report"]
    filter_report.to_csv(filtered_dir / "feature_report.csv")

    # ── Evidence-based routing ───────────────────────────────────────────
    log.info("Running evidence-based feature routing...")
    write_routing_report(filtered_dir, filter_report)

    elapsed = time.time() - t0
    tier_counts = filter_report["tier"].value_counts().to_dict()

    summary = {
        "target": target,
        "task": task,
        "n_features_input": X.shape[1],
        "n_samples": len(X),
        "n_seasons": int(seasons.nunique()),
        "n_clusters": len(clusters),
        "tier_counts": tier_counts,
        "n_survivors": len(results["survivors"]),
        "synthetic_validation_pass": synth["mdi_pass"],
        "elapsed_secs": round(elapsed, 1),
    }

    with open(target_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Pipeline complete in {elapsed:.1f}s")
    log.info(f"  Tiers: {tier_counts}")
    log.info(f"  Survivors: {len(results['survivors'])} features")
    log.info(f"  Artifacts: {target_dir}")

    return summary
