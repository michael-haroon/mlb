"""Orchestrate the full de Prado feature importance pipeline.

Runs all importance methods (MDI, MDA, SFI, CFI, de-substituted MDA,
PCA-MDA, residualized MDA) in parallel across targets × tests,
performs ONC clustering, statistical significance testing, and
evidence-based feature routing.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..strategy.config import SFI_CLASS_WEIGHT_OVERRIDES, TARGETS_CLASSIFICATION
from ..strategy.data import compute_temporal_weights, load_features
from .compute import get_n_jobs, set_parallel_mode, blas_limit
from .feature_importance import (
    build_rf,
    compute_shared_clustering,
    feat_imp_cfi_mda,
    feat_imp_cfi_mdi_deprado,
    feat_imp_desub_mda,
    feat_imp_mda,
    feat_imp_mdi,
    feat_imp_pca_mda,
    feat_imp_residual_mda,
    feat_imp_sfi,
    filter_features_v2,
    pca_cross_check,
    plot_cfi_mda_distributions,
    compute_pvalues,
    synthetic_validation,
)
from .feature_routing import write_routing_report

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-test worker functions (each runs in a loky-spawned process)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_mdi_cfi_mdi(features_path, target, data_mode, valid_cols, clusters, regression):
    """MDI + CFI-MDI: fit one RF, extract per-tree importances + cluster aggregation."""
    set_parallel_mode(True)
    X, y, seasons, _ = load_features(Path(features_path), target, data_mode)
    X = X[valid_cols]
    sample_weight = compute_temporal_weights(seasons)

    X_filled = X.fillna(X.median())
    clf = build_rf(n_estimators=1000, n_jobs=1, regression=regression)
    clf.fit(X_filled, y, sample_weight=sample_weight)

    mdi, mdi_raw = feat_imp_mdi(clf, list(X.columns))
    cfi_mdi, cfi_mdi_per_feat, cfi_mdi_raw = feat_imp_cfi_mdi_deprado(
        clf, list(X.columns), clusters)

    return {
        "test": "mdi_cfi_mdi",
        "target": target,
        "mdi": mdi,
        "mdi_raw": mdi_raw,
        "cfi_mdi": cfi_mdi,
        "cfi_mdi_per_feat": cfi_mdi_per_feat,
        "cfi_mdi_raw": cfi_mdi_raw,
    }


def _worker_cfi_mda(features_path, target, data_mode, valid_cols, clusters, regression):
    """CFI-MDA: cluster-level permutation importance via fold-based CV."""
    set_parallel_mode(True)
    X, y, seasons, _ = load_features(Path(features_path), target, data_mode)
    X = X[valid_cols]
    sample_weight = compute_temporal_weights(seasons)

    scoring = "r2" if regression else "log_loss"
    clf = build_rf(n_estimators=300, n_jobs=1, regression=regression)
    cfi_mda, cfi_mda_raw = feat_imp_cfi_mda(
        clf, X, y, seasons, clusters, sample_weight, scoring=scoring)

    return {
        "test": "cfi_mda",
        "target": target,
        "cfi_mda": cfi_mda,
        "cfi_mda_raw": cfi_mda_raw,
    }


def _worker_sfi(features_path, target, data_mode, valid_cols, clusters, regression):
    """SFI: single feature importance (standalone OOS)."""
    set_parallel_mode(True)
    X, y, seasons, _ = load_features(Path(features_path), target, data_mode)
    X = X[valid_cols]
    sample_weight = compute_temporal_weights(seasons)

    sfi_class_weight = SFI_CLASS_WEIGHT_OVERRIDES.get(target, "balanced")
    clf = build_rf(n_estimators=300, n_jobs=1, regression=regression)
    sfi, sfi_raw = feat_imp_sfi(
        clf, X, y, seasons, sample_weight,
        regression=regression, sfi_class_weight=sfi_class_weight)

    return {
        "test": "sfi",
        "target": target,
        "sfi": sfi,
        "sfi_raw": sfi_raw,
    }


def _worker_desub_mda(features_path, target, data_mode, valid_cols, clusters, regression,
                      n_repeats=1):
    """De-substituted MDA: within-cluster ranking, substitution-free."""
    set_parallel_mode(True)
    X, y, seasons, _ = load_features(Path(features_path), target, data_mode)
    X = X[valid_cols]
    sample_weight = compute_temporal_weights(seasons)

    scoring = "r2" if regression else "log_loss"
    return_repeat_sd = n_repeats > 1
    result = feat_imp_desub_mda(
        X, y, seasons, clusters,
        sample_weight=sample_weight, scoring=scoring,
        n_estimators=300, regression=regression,
        n_repeats=n_repeats, return_repeat_sd=return_repeat_sd)

    out = {"test": "desub_mda", "target": target}
    if return_repeat_sd:
        out["desub_mda"], out["desub_mda_raw"], out["desub_mda_repeat_sd"] = result
    else:
        out["desub_mda"], out["desub_mda_raw"] = result
    return out


def _worker_pca_mda(features_path, target, data_mode, valid_cols, clusters, regression):
    """PCA-MDA: orthogonal basis importance."""
    set_parallel_mode(True)
    X, y, seasons, _ = load_features(Path(features_path), target, data_mode)
    X = X[valid_cols]
    sample_weight = compute_temporal_weights(seasons)

    scoring = "r2" if regression else "log_loss"
    pca_mda, pca_mda_raw, pca_mda_pc_summary, pca_evr = feat_imp_pca_mda(
        X, y, seasons,
        sample_weight=sample_weight, scoring=scoring,
        n_estimators=300, regression=regression)

    return {
        "test": "pca_mda",
        "target": target,
        "pca_mda": pca_mda,
        "pca_mda_raw": pca_mda_raw,
        "pca_mda_pc_summary": pca_mda_pc_summary,
        "pca_explained_variance_ratio": pca_evr,
    }


def _worker_resid_mda(features_path, target, data_mode, valid_cols, clusters, regression):
    """Residualized MDA: cross-cluster orthogonalization."""
    set_parallel_mode(True)
    X, y, seasons, _ = load_features(Path(features_path), target, data_mode)
    X = X[valid_cols]
    sample_weight = compute_temporal_weights(seasons)

    scoring = "r2" if regression else "log_loss"
    resid_mda, resid_mda_raw = feat_imp_residual_mda(
        X, y, seasons, clusters,
        sample_weight=sample_weight, scoring=scoring,
        n_estimators=300, regression=regression)

    return {
        "test": "resid_mda",
        "target": target,
        "resid_mda": resid_mda,
        "resid_mda_raw": resid_mda_raw,
    }


_TEST_WORKERS = {
    "mdi_cfi_mdi": _worker_mdi_cfi_mdi,
    "cfi_mda": _worker_cfi_mda,
    "sfi": _worker_sfi,
    "desub_mda": _worker_desub_mda,
    "pca_mda": _worker_pca_mda,
    "resid_mda": _worker_resid_mda,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Parallel orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def _run_worker(features_path, target, test_name, data_mode, valid_cols, clusters, regression):
    """Dispatch to the correct worker. Top-level function for joblib pickling."""
    worker_fn = _TEST_WORKERS[test_name]
    return worker_fn(features_path, target, data_mode, valid_cols, clusters, regression)


def run_importance_parallel(
    features_path: Path,
    output_dir: Path,
    targets: list[str] | None = None,
    data_mode: str = "2015+",
    run_sfi: bool = True,
    run_desub_mda: bool = True,
    run_pca_mda: bool = True,
    run_residual_mda: bool = True,
) -> dict:
    """Run the full feature importance pipeline for all targets in parallel.

    Architecture:
      Phase 1 (sequential): Per-target data loading + shared ONC clustering
      Phase 2 (parallel):   All (target × test) pairs run concurrently
      Phase 3 (sequential): Per-target aggregation, PCA cross-check, filtering, routing

    Returns dict mapping target → summary.
    """
    from ..strategy.config import ALL_TARGETS

    if targets is None:
        targets = ALL_TARGETS

    features_path = Path(features_path)
    output_dir = Path(output_dir)

    t0_total = time.time()
    log.info(f"{'='*70}")
    log.info(f"Feature importance pipeline: {len(targets)} targets, parallel mode")
    log.info(f"{'='*70}")

    # ── Phase 1: Per-target setup (sequential, fast) ────────────────────
    log.info("Phase 1: Loading data + computing clustering per target...")
    target_configs = {}

    for target in targets:
        task = "classification" if target in TARGETS_CLASSIFICATION else "regression"
        regression = (task == "regression")

        X, y, seasons, _ = load_features(features_path, target, data_mode)
        nan_pct = X.isna().mean()
        valid_cols = nan_pct[nan_pct < 0.95].index.tolist()
        X = X[valid_cols]

        X_for_clustering = X.fillna(X.median())
        clustering = compute_shared_clustering(X_for_clustering)

        target_configs[target] = {
            "valid_cols": valid_cols,
            "clusters": clustering["clusters"],
            "denoising_info": clustering["denoising_info"],
            "regression": regression,
            "task": task,
            "n_features": X.shape[1],
            "n_samples": len(X),
            "n_seasons": int(seasons.nunique()),
        }
        log.info(f"  {target}: {X.shape[1]} features, {len(X):,} samples, "
                 f"{len(clustering['clusters'])} clusters")

    # Synthetic validation (once, not per-target)
    synth = synthetic_validation()
    if not synth["mdi_pass"]:
        log.warning("SYNTHETIC VALIDATION FAILED: MDI cannot recover known signal")
    else:
        log.info("Synthetic validation passed")

    # ── Phase 2: Build task list and run in parallel ────────────────────
    tests_to_run = ["mdi_cfi_mdi", "cfi_mda"]
    if run_sfi:
        tests_to_run.append("sfi")
    if run_desub_mda:
        tests_to_run.append("desub_mda")
    if run_pca_mda:
        tests_to_run.append("pca_mda")
    if run_residual_mda:
        tests_to_run.append("resid_mda")

    tasks = []
    for target in targets:
        cfg = target_configs[target]
        for test_name in tests_to_run:
            tasks.append((
                str(features_path), target, test_name, data_mode,
                cfg["valid_cols"], cfg["clusters"], cfg["regression"],
            ))

    n_tasks = len(tasks)
    n_workers = min(n_tasks, get_n_jobs())
    log.info(f"Phase 2: Running {n_tasks} tasks ({len(targets)} targets × "
             f"{len(tests_to_run)} tests) with {n_workers} workers...")

    t1 = time.time()
    with blas_limit(1):
        results_list = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_run_worker)(*task) for task in tasks
        )
    log.info(f"Phase 2 complete in {time.time() - t1:.1f}s")

    # ── Phase 3: Per-target aggregation ─────────────────────────────────
    log.info("Phase 3: Aggregating results, filtering, routing...")
    all_summaries = {}

    for target in targets:
        cfg = target_configs[target]
        target_dir = output_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)
        filtered_dir = target_dir / "filtered"
        filtered_dir.mkdir(parents=True, exist_ok=True)

        # Gather results for this target
        target_results = {}
        for r in results_list:
            if r["target"] == target:
                target_results[r["test"]] = r

        # Extract per-test outputs
        mdi_r = target_results.get("mdi_cfi_mdi", {})
        cfi_mda_r = target_results.get("cfi_mda", {})
        sfi_r = target_results.get("sfi", {})
        desub_r = target_results.get("desub_mda", {})
        pca_r = target_results.get("pca_mda", {})
        resid_r = target_results.get("resid_mda", {})

        mdi_raw = mdi_r.get("mdi_raw")
        cfi_mdi_raw = mdi_r.get("cfi_mdi_raw")
        cfi_mda_raw = cfi_mda_r.get("cfi_mda_raw")
        sfi_raw = sfi_r.get("sfi_raw")
        desub_mda_raw = desub_r.get("desub_mda_raw")
        pca_mda_raw = pca_r.get("pca_mda_raw")
        resid_mda_raw = resid_r.get("resid_mda_raw")

        clusters = cfg["clusters"]

        # Save clustering artifacts
        cluster_map = {str(k): v for k, v in clusters.items()}
        with open(target_dir / "cluster_map.json", "w") as f:
            json.dump(cluster_map, f, indent=2)
        with open(target_dir / "denoising_report.json", "w") as f:
            json.dump(cfg["denoising_info"], f, indent=2)

        # Save raw importance artifacts
        if mdi_raw is not None:
            mdi_raw.to_csv(target_dir / "importance_mdi_raw.csv")
        if sfi_raw is not None:
            sfi_raw.to_csv(target_dir / "importance_sfi_raw.csv")
        if desub_mda_raw is not None:
            desub_mda_raw.to_csv(target_dir / "importance_desub_mda_raw.csv")
        if pca_mda_raw is not None:
            pca_mda_raw.to_csv(target_dir / "importance_pca_mda_raw.csv")
        if resid_mda_raw is not None:
            resid_mda_raw.to_csv(target_dir / "importance_resid_mda_raw.csv")
        if cfi_mda_raw is not None:
            cfi_mda_raw.to_csv(target_dir / "importance_cfi_mda_raw.csv")

        # Build summary DataFrame
        mdi = mdi_r.get("mdi")
        sfi = sfi_r.get("sfi")
        if mdi is not None:
            summary = mdi[["mean"]].rename(columns={"mean": "MDI"})
            mdi_pvals = compute_pvalues(mdi_raw, null_mean=1.0 / cfg["n_features"])
            summary = summary.join(mdi_pvals.rename("p_MDI"), how="left")
        else:
            summary = pd.DataFrame()

        summary.to_csv(target_dir / "importance_summary.csv")

        # PCA cross-check (diagnostic only — runs sequentially, relatively fast)
        log.info(f"  [{target}] PCA cross-check (diagnostic)...")
        X, y, seasons, _ = load_features(features_path, target, data_mode)
        X = X[cfg["valid_cols"]]
        sample_weight = compute_temporal_weights(seasons)
        pca_info, tau_results = pca_cross_check(
            X, y, seasons, sample_weight, regression=cfg["regression"])
        if pca_info is not None:
            pca_info.to_csv(target_dir / "pca_cross_check.csv")
        if tau_results is not None:
            with open(target_dir / "kendall_tau.json", "w") as f:
                json.dump(tau_results, f, indent=2)

        # CFI-MDA distribution plot
        if cfi_mda_raw is not None:
            try:
                plot_cfi_mda_distributions(
                    cfi_mda_raw, clusters,
                    output_path=str(target_dir / "cfi_mda_distributions.png"))
            except Exception as e:
                log.warning(f"  [{target}] Could not generate CFI-MDA plot: {e}")

        # Algorithmic filtering
        null_score = 0.0
        if sfi is not None:
            null_col = "null_r2" if cfg["regression"] else "null_log_loss"
            null_score = sfi[null_col].iloc[0] if null_col in sfi.columns else 0.0

        filter_report = filter_features_v2(
            sfi_raw=sfi_raw,
            desub_mda_raw=desub_mda_raw,
            pca_mda_raw=pca_mda_raw,
            resid_mda_raw=resid_mda_raw,
            mdi_raw=mdi_raw,
            cfi_mda_raw=cfi_mda_raw if cfi_mda_raw is not None else pd.DataFrame(),
            cfi_mdi_raw=cfi_mdi_raw if cfi_mdi_raw is not None else pd.DataFrame(),
            clusters=clusters,
            sfi_null=null_score,
        )
        filter_report.to_csv(filtered_dir / "feature_report.csv")

        # Evidence-based routing
        write_routing_report(filtered_dir, filter_report)

        survivors = filter_report[
            filter_report["tier"].isin(["ACCEPTED", "NEEDS SPECIFICATION"])]
        tier_counts = {k: int(v) for k, v in
                       filter_report["tier"].value_counts().to_dict().items()}

        target_summary = {
            "target": target,
            "task": cfg["task"],
            "n_features_input": cfg["n_features"],
            "n_samples": cfg["n_samples"],
            "n_seasons": cfg["n_seasons"],
            "n_clusters": len(clusters),
            "tier_counts": tier_counts,
            "n_survivors": len(survivors),
            "synthetic_validation_pass": bool(synth["mdi_pass"]),
        }
        with open(target_dir / "pipeline_summary.json", "w") as f:
            json.dump(target_summary, f, indent=2)

        all_summaries[target] = target_summary
        log.info(f"  [{target}] {tier_counts}, {len(survivors)} survivors")

    elapsed = time.time() - t0_total
    log.info(f"Pipeline complete in {elapsed:.1f}s for {len(targets)} targets")
    return all_summaries


# ─────────────────────────────────────────────────────────────────────────────
#  Single-target convenience wrapper (backward compat)
# ─────────────────────────────────────────────────────────────────────────────

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
    """Run the full feature importance pipeline for one target.

    Delegates to run_importance_parallel with a single target.
    """
    results = run_importance_parallel(
        features_path=features_path,
        output_dir=output_dir,
        targets=[target],
        data_mode=data_mode,
        run_sfi=run_sfi,
        run_desub_mda=run_desub_mda,
        run_pca_mda=run_pca_mda,
        run_residual_mda=run_residual_mda,
    )
    return results.get(target, {})
