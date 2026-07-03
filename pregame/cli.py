#!/usr/bin/env python3
"""CLI for the MLB pregame prediction pipeline.

Subcommands:
  eda             — Run raw EDA on S3/local data
  build-features  — Engineer features → game_features.parquet
  run-importance  — Feature importance analysis (De Prado methods)
  train           — LOYO CV training with Optuna HPO
  evaluate        — Evaluation metrics, QQ plots, calibration diagnostics
  predict         — Inference on new game features
"""

import argparse
import json
import logging
import sys
from pathlib import Path

LOG_DIR = Path("data/logs")


def _configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    if not root.handlers:
        fh = logging.FileHandler(LOG_DIR / "pregame_cli.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(message)s"))
        root.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
        root.addHandler(sh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLB pregame prediction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage")

    # --- EDA (existing) ---
    eda_parser = subparsers.add_parser("eda", help="Raw EDA analysis")
    eda_parser.add_argument("--source", required=True)
    eda_parser.add_argument("--output", required=True)
    eda_parser.add_argument("--season-start", type=int, default=2015)
    eda_parser.add_argument("--season-end", type=int, default=2026)

    # --- Build Features ---
    build_parser = subparsers.add_parser("build-features", help="Engineer features")
    build_parser.add_argument("--source", help="S3 URI or local path to raw data",
                              default="s3://mlb-265753586044-us-east-1-an/data")
    build_parser.add_argument("--output", help="Output directory for artifacts",
                              default="pregame/artifacts/features")
    build_parser.add_argument("--season-start", type=int, default=2015)
    build_parser.add_argument("--season-end", type=int, default=None)
    build_parser.add_argument("--tune-ratings", action="store_true", default=True,
                              help="Run Optuna HPO on rating system parameters")
    build_parser.add_argument("--no-tune-ratings", action="store_false", dest="tune_ratings")
    build_parser.add_argument("--n-trials", type=int, default=100,
                              help="Optuna trials per rating system")
    build_parser.add_argument("--ratings-params", type=str, default=None,
                              help="Path to pre-tuned ratings_params.json")

    # --- Feature Importance ---
    imp_parser = subparsers.add_parser("run-importance", help="Feature importance analysis")
    imp_parser.add_argument("--features", default="pregame/artifacts/features/game_features.parquet",
                             help="Path to game_features.parquet")
    imp_parser.add_argument("--output", default="pregame/artifacts/importance")
    imp_parser.add_argument("--target", default=None, help="Target column (default: all targets)")
    imp_parser.add_argument("--data-mode", default="2015+", choices=["2015+", "all"])
    imp_parser.add_argument("--run-sfi", action="store_true", default=True)
    imp_parser.add_argument("--no-sfi", action="store_false", dest="run_sfi")
    imp_parser.add_argument("--run-desub-mda", action="store_true", default=True)
    imp_parser.add_argument("--no-desub-mda", action="store_false", dest="run_desub_mda")
    imp_parser.add_argument("--run-pca-mda", action="store_true", default=True)
    imp_parser.add_argument("--no-pca-mda", action="store_false", dest="run_pca_mda")
    imp_parser.add_argument("--run-residual-mda", action="store_true", default=True)
    imp_parser.add_argument("--no-residual-mda", action="store_false", dest="run_residual_mda")

    # --- Train ---
    train_parser = subparsers.add_parser("train", help="LOYO CV training with Optuna")
    train_parser.add_argument("--features", default="pregame/artifacts/features/game_features.parquet",
                               help="Path to game_features.parquet")
    train_parser.add_argument("--output", default="pregame/artifacts/models")
    train_parser.add_argument("--target", default=None, help="Target column (default: all targets)")
    train_parser.add_argument("--data-mode", default="2015+", choices=["2015+", "all"])
    train_parser.add_argument("--families", nargs="*", default=None,
                              help="Model families to train (default: all)")
    train_parser.add_argument("--n-trials", type=int, default=100)
    train_parser.add_argument("--tier", default="A", choices=["A", "B", "C"])

    # --- Ensemble ---
    ens_parser = subparsers.add_parser("ensemble", help="Build ensemble from trained OOF predictions")
    ens_parser.add_argument("--models", required=True, help="Path to models directory (contains OOF .npy and params .json)")
    ens_parser.add_argument("--features", default="pregame/artifacts/features/game_features.parquet",
                             help="Path to game_features.parquet")
    ens_parser.add_argument("--output", default=None, help="Output directory for ensemble .pkl (default: same as --models)")
    ens_parser.add_argument("--target", default=None, help="Target column (default: all targets)")
    ens_parser.add_argument("--tier", default="A", choices=["A", "B", "C"])
    ens_parser.add_argument("--data-mode", default="2015+", choices=["2015+", "all"])

    # --- Compare Ensemble ---
    cmp_parser = subparsers.add_parser(
        "compare-ensemble",
        help="Compare calibration × ensemble method combinations (diagnostic, no pickle changes)",
    )
    cmp_parser.add_argument("--models", required=True, help="Path to models directory (contains OOF .npy and params .json)")
    cmp_parser.add_argument("--features", default="pregame/artifacts/features/game_features.parquet",
                             help="Path to game_features.parquet")
    cmp_parser.add_argument("--output", default=None, help="Output directory for comparison JSON (default: same as --models)")
    cmp_parser.add_argument("--target", default=None, help="Target column (default: all targets)")
    cmp_parser.add_argument("--tier", default="A", choices=["A", "B", "C"])
    cmp_parser.add_argument("--data-mode", default="2015+", choices=["2015+", "all"])

    # --- Evaluate ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluation and diagnostics")
    eval_parser.add_argument("--models", required=True, help="Path to models directory")
    eval_parser.add_argument("--output", default="pregame/artifacts/evaluation")
    eval_parser.add_argument("--target", default=None, help="Target column (default: all targets)")
    eval_parser.add_argument("--tier", default="A", choices=["A", "B", "C"])

    # --- Predict ---
    pred_parser = subparsers.add_parser("predict", help="Inference on new features")
    pred_parser.add_argument("--ensemble", required=True, help="Path to ensemble .pkl")
    pred_parser.add_argument("--features", required=True, help="Path to features parquet/csv")
    pred_parser.add_argument("--target", default=None, help="Target column (default: all targets)")

    args = parser.parse_args()
    _configure_logging()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "eda":
        _run_eda(args)
    elif args.command == "build-features":
        _run_build_features(args)
    elif args.command == "run-importance":
        _run_importance(args)
    elif args.command == "train":
        _run_train(args)
    elif args.command == "ensemble":
        _run_ensemble(args)
    elif args.command == "compare-ensemble":
        _run_compare_ensemble(args)
    elif args.command == "evaluate":
        _run_evaluate(args)
    elif args.command == "predict":
        _run_predict(args)


def _run_eda(args):
    from .raw_eda import run_raw_eda

    seasons = list(range(args.season_start, args.season_end + 1))
    outputs = run_raw_eda(source_uri=args.source, output_dir=args.output, seasons=seasons)
    print(json.dumps(outputs, indent=2))


def _run_build_features(args):
    from .engineering.build import build_features

    ratings_params = None
    if args.ratings_params:
        with open(args.ratings_params) as f:
            ratings_params = json.load(f)

    out_path = build_features(
        source=args.source,
        output=Path(args.output),
        season_start=args.season_start,
        season_end=args.season_end,
        tune_ratings=args.tune_ratings,
        n_trials=args.n_trials,
        ratings_params=ratings_params,
    )
    print(f"Features written to: {out_path}")


def _run_importance(args):
    from .analysis.run import run_importance_analysis
    from .strategy.config import ALL_TARGETS

    targets = [args.target] if args.target else ALL_TARGETS
    results = {}
    for target in targets:
        results[target] = run_importance_analysis(
            features_path=Path(args.features),
            output_dir=Path(args.output),
            target=target,
            data_mode=args.data_mode,
            run_sfi=args.run_sfi,
            run_desub_mda=args.run_desub_mda,
            run_pca_mda=args.run_pca_mda,
            run_residual_mda=args.run_residual_mda,
        )
    print(json.dumps(results, indent=2))


def _run_train(args):
    import gc

    from .strategy.config import ALL_TARGETS
    from .strategy.train import train_target

    targets = [args.target] if args.target else ALL_TARGETS
    all_results = {}
    for target in targets:
        result = train_target(
            features_path=Path(args.features),
            target=target,
            output_dir=Path(args.output),
            data_mode=args.data_mode,
            families=args.families,
            n_trials=args.n_trials,
            tier=args.tier,
        )
        all_results[target] = {k: v.get("status", "unknown") for k, v in result.items()}
        # Release per-target result dict (fold metrics, OOF arrays, etc.) before
        # loading the next target's feature matrix
        del result
        gc.collect()
    print(json.dumps(all_results, indent=2))


def _run_ensemble(args):
    import gc

    import numpy as np
    import pandas as pd

    from .strategy.calibration import calibrate_classification, calibrate_regression
    from .strategy.config import ALL_TARGETS, TARGETS_CLASSIFICATION
    from .strategy.ensemble import (
        build_ensemble,
        fit_and_save_ensemble,
        load_ensemble_oof,
        predict_ensemble,
    )
    from .strategy.evaluate import compute_metrics

    models_dir = Path(args.models)
    output_dir = Path(args.output) if args.output else models_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = Path(args.features)

    targets = [args.target] if args.target else ALL_TARGETS
    df = pd.read_parquet(features_path)

    results = {}
    for target in targets:
        task = "classification" if target in TARGETS_CLASSIFICATION else "regression"

        if target not in df.columns:
            print(f"[{target}] Not found in features, skipping")
            continue

        oof_matrix = load_ensemble_oof(models_dir, target, args.tier)
        if not oof_matrix:
            print(f"[{target}] No OOF files found, skipping")
            continue

        y_series = df[target].dropna()
        y_true = y_series.values

        # Build per-family aggregate metrics from training_summary for quality filtering
        summary_path = models_dir / f"training_summary_{target}_{args.tier}.json"
        family_metrics = {}
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            primary = "log_loss" if task == "classification" else "mae"
            for fam, v in summary.items():
                if v.get("status") == "success":
                    agg = v.get("aggregate_metrics", {})
                    val = agg.get(primary)
                    if val is not None:
                        family_metrics[fam] = {primary: val}
                        if task == "regression":
                            r2 = agg.get("r2")
                            if r2 is not None:
                                family_metrics[fam]["r2"] = r2

        # Align OOF arrays to y_true length (OOF may be shorter if early seasons skipped)
        n = len(y_true)
        aligned_oof = {}
        for fam, arr in oof_matrix.items():
            aligned_oof[fam] = arr[:n] if len(arr) >= n else np.pad(arr, (0, n - len(arr)), constant_values=np.nan)

        # 2020 was a 60-game shortened season played under pandemic protocols: no fans,
        # neutral-site bubble games, universal DH for the first time, and dramatically
        # compressed schedule. Every distributional property (run environment, rest
        # patterns, win rates) is a structural outlier — including it in ensemble weight
        # optimization or calibration contaminates both. Exclude permanently and silently.
        #
        # OOF arrays are positionally aligned to the non-null target rows of the full df
        # (same row order as y_series). The no2020_mask is therefore built over those
        # same non-null rows so the boolean index is compatible with both y_true and
        # each OOF array.
        no2020_mask = (df.loc[y_series.index, "season"] != 2020).values
        y_true = y_true[no2020_mask]
        aligned_oof = {fam: arr[no2020_mask] for fam, arr in aligned_oof.items()}

        ens_result = build_ensemble(
            oof_matrix=aligned_oof,
            y_true=y_true,
            metrics=family_metrics,
            task=task,
        )

        members = ens_result.get("members", [])
        weights = ens_result.get("weights", [])

        if not members:
            print(f"[{target}] No ensemble members selected: {ens_result.get('error')}")
            results[target] = ens_result
            continue

        nonzero_members = [(m, w) for m, w in zip(members, weights) if w >= 0.01]
        print(f"[{target}] Selected {len(members)} members, {len(nonzero_members)} with non-zero weight: {[m for m,_ in nonzero_members]}")
        print(f"[{target}] Weights: {[round(w, 3) for _, w in nonzero_members]}")
        print(f"[{target}] Ensemble metrics: {ens_result.get('ensemble_metrics')}")

        # Refit on full data and save pickle.
        # Pass aligned_oof so fit_and_save_ensemble can fit per-model isotonic calibrators
        # that match the calibration applied during SLSQP weight optimization.
        pkl_path = fit_and_save_ensemble(
            features_path=features_path,
            models_dir=models_dir,
            target=target,
            tier=args.tier,
            members=members,
            weights=weights,
            data_mode=args.data_mode,
            output_path=output_dir / f"ensemble_{target}_{args.tier}.pkl",
            oof_matrix=aligned_oof,
        )

        # Calibrate using ensemble OOF blend.
        # For classification: apply each member's per-model isotonic calibrator
        # before blending so the CalibrationBundle is fitted on the same
        # distribution it will receive at inference time (calibrated blend,
        # not raw blend). Without this, the bundle corrects the wrong distribution.
        valid_mask = ~np.isnan(y_true)
        for fam in members:
            valid_mask &= ~np.isnan(aligned_oof[fam])

        import pickle as _pickle
        with open(pkl_path, "rb") as _f:
            _saved_bundle = _pickle.load(_f)
        _member_map = {mb["family"]: mb for mb in _saved_bundle["member_bundles"]}

        if task == "classification":
            cal_cols = []
            for fam in members:
                col = aligned_oof[fam][valid_mask]
                iso = _member_map.get(fam, {}).get("isotonic_calibrator")
                if iso is not None:
                    col = iso.predict(col)
                cal_cols.append(col)
            selected_preds = np.column_stack(cal_cols)
        else:
            selected_preds = np.column_stack([aligned_oof[m][valid_mask] for m in members])

        oof_blend = selected_preds @ np.array(weights)
        oof_std = selected_preds.std(axis=1)
        yt = y_true[valid_mask]

        if task == "classification":
            cal_bundle = calibrate_classification(yt, oof_blend, oof_std)
        else:
            cal_bundle = calibrate_regression(yt, oof_blend, oof_std)

        # Attach calibration to pickle (reuse _saved_bundle already loaded above)
        _saved_bundle["calibration"] = cal_bundle
        with open(pkl_path, "wb") as f:
            _pickle.dump(_saved_bundle, f, protocol=_pickle.HIGHEST_PROTOCOL)

        results[target] = {
            "status": "success",
            "pkl": str(pkl_path),
            "members": members,
            "weights": weights,
            "ensemble_metrics": ens_result.get("ensemble_metrics"),
        }
        gc.collect()

    print(json.dumps(results, indent=2, default=str))


def _run_compare_ensemble(args):
    import gc

    import numpy as np
    import pandas as pd

    from .strategy.config import ALL_TARGETS, TARGETS_CLASSIFICATION
    from .strategy.ensemble import compare_ensemble_methods, load_ensemble_oof

    models_dir = Path(args.models)
    output_dir = Path(args.output) if args.output else models_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = Path(args.features)

    targets = [args.target] if args.target else ALL_TARGETS
    df = pd.read_parquet(features_path)

    for target in targets:
        task = "classification" if target in TARGETS_CLASSIFICATION else "regression"

        if target not in df.columns:
            print(f"[{target}] Not found in features, skipping")
            continue

        oof_matrix = load_ensemble_oof(models_dir, target, args.tier)
        if not oof_matrix:
            print(f"[{target}] No OOF files found, skipping")
            continue

        y_series = df[target].dropna()
        y_true = y_series.values

        # Build per-family aggregate metrics from training_summary for quality filtering
        summary_path = models_dir / f"training_summary_{target}_{args.tier}.json"
        family_metrics = {}
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            primary = "log_loss" if task == "classification" else "mae"
            for fam, v in summary.items():
                if v.get("status") == "success":
                    agg = v.get("aggregate_metrics", {})
                    val = agg.get(primary)
                    if val is not None:
                        family_metrics[fam] = {primary: val}
                        if task == "regression":
                            r2 = agg.get("r2")
                            if r2 is not None:
                                family_metrics[fam]["r2"] = r2

        # Align OOF arrays to y_true length (OOF may be shorter if early seasons skipped)
        n = len(y_true)
        aligned_oof = {}
        for fam, arr in oof_matrix.items():
            aligned_oof[fam] = arr[:n] if len(arr) >= n else np.pad(arr, (0, n - len(arr)), constant_values=np.nan)

        # Exclude 2020: structural outlier (60-game pandemic season) — same logic as _run_ensemble.
        # OOF arrays are positionally aligned to the non-null target rows of the full df.
        no2020_mask = (df.loc[y_series.index, "season"] != 2020).values
        y_true = y_true[no2020_mask]
        aligned_oof = {fam: arr[no2020_mask] for fam, arr in aligned_oof.items()}

        print(f"\n[{target}] Running calibration × ensemble comparison ({task})")
        comparison = compare_ensemble_methods(
            oof_matrix=aligned_oof,
            y_true=y_true,
            task=task,
            metrics=family_metrics,
        )

        if "error" in comparison:
            print(f"[{target}] Error: {comparison['error']}")
            continue

        # Print results table sorted by primary metric
        primary = "log_loss" if task == "classification" else "mae"
        if task == "classification":
            col_keys = ["log_loss", "auc_roc", "brier_score", "ece"]
        else:
            col_keys = ["mae", "rmse", "r2"]

        # Build rows: (primary_val, key, metrics_dict)
        rows = []
        for key, v in comparison.items():
            if "error" in v:
                continue
            em = v.get("ensemble_metrics", {})
            pval = em.get(primary, float("inf"))
            rows.append((pval, key, em))
        rows.sort()

        # Header
        col_w = 10
        header = f"  {'Method':<30}"
        for c in col_keys:
            header += f"  {c.upper():<{col_w}}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for pval, key, em in rows:
            line = f"  {key:<30}"
            for c in col_keys:
                val = em.get(c, float("nan"))
                line += f"  {val:<{col_w}.4f}"
            print(line)

        # Save full results JSON
        out_path = output_dir / f"comparison_{target}_{args.tier}.json"
        with open(out_path, "w") as f:
            json.dump(comparison, f, indent=2, default=str)
        print(f"[{target}] Full results saved to {out_path}")

        gc.collect()


def _run_evaluate(args):
    import numpy as np
    import pandas as pd

    from .strategy.config import ALL_TARGETS, TARGETS_CLASSIFICATION
    from .strategy.distributions import fit_best_distribution, generate_qq_plots
    from .strategy.evaluate import (
        calibration_curve_data,
        ensemble_diagnostics,
        print_model_comparison,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models)
    tier = getattr(args, "tier", "A")

    targets = [args.target] if args.target else ALL_TARGETS
    features_path = models_dir.parent / "features" / "game_features.parquet"
    df = pd.read_parquet(features_path) if features_path.exists() else None

    for target in targets:
        task = "classification" if target in TARGETS_CLASSIFICATION else "regression"

        # --- 1. Per-family ranking table ---
        summary_path = models_dir / f"training_summary_{target}_{tier}.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            print(f"\n=== {target} ({task}) — model comparison ===")
            print_model_comparison(summary, task)
        else:
            print(f"[{target}] No training summary found at {summary_path}")

        if df is None or target not in df.columns:
            continue

        y_true = df[target].dropna().values

        # --- 2. QQ plots and residual distributions for regression ---
        if target not in TARGETS_CLASSIFICATION:
            oof_files = list(models_dir.glob(f"oof_{target}_*_{tier}.npy"))
            for oof_file in oof_files:
                oof = np.load(oof_file)
                valid = ~np.isnan(oof) & ~np.isnan(y_true[:len(oof)])
                if valid.sum() > 30:
                    residuals = y_true[:len(oof)][valid] - oof[valid]
                    generate_qq_plots(residuals, output_dir, target)
                    fit_result = fit_best_distribution(residuals)
                    print(json.dumps({target: fit_result}, indent=2, default=str))

        # --- 3. Ensemble-level diagnostics if pickle exists ---
        import pickle
        pkl_path = models_dir / f"ensemble_{target}_{tier}.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                bundle = pickle.load(f)
            members = bundle.get("members", [])
            weights = np.array(bundle.get("weights", []))

            # Reconstruct ensemble OOF blend for diagnostics
            n = len(y_true)
            member_oofs = []
            for m in members:
                arr = np.load(models_dir / f"oof_{target}_{m}_{tier}.npy")
                member_oofs.append(arr[:n] if len(arr) >= n else np.pad(arr, (0, n - len(arr)), constant_values=np.nan))

            if member_oofs:
                valid_mask = ~np.isnan(y_true)
                for arr in member_oofs:
                    valid_mask &= ~np.isnan(arr)

                preds = np.column_stack([a[valid_mask] for a in member_oofs])
                oof_blend = preds @ weights
                oof_std = preds.std(axis=1)
                yt = y_true[valid_mask]

                diag = ensemble_diagnostics(bundle, yt, oof_blend, oof_std)
                print(f"\n=== {target} — ensemble diagnostics ===")
                print(json.dumps(diag, indent=2, default=str))

                # Save diagnostics JSON
                diag_path = output_dir / f"diagnostics_{target}_{tier}.json"
                with open(diag_path, "w") as f:
                    json.dump(diag, f, indent=2, default=str)

    print(f"\nEvaluation artifacts written to: {output_dir}")


def _run_predict(args):
    import pandas as pd

    from .strategy.config import ALL_TARGETS
    from .strategy.predict import predict_game

    features = pd.read_parquet(args.features)
    targets = [args.target] if args.target else ALL_TARGETS
    all_results = {}
    for target in targets:
        all_results[target] = predict_game(features, Path(args.ensemble), target)
    print(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
