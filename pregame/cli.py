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

    # --- Evaluate ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluation and diagnostics")
    eval_parser.add_argument("--models", required=True, help="Path to models directory")
    eval_parser.add_argument("--output", required=True)
    eval_parser.add_argument("--target", default=None, help="Target column (default: all targets)")

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


def _run_evaluate(args):
    import numpy as np
    import pandas as pd

    from .strategy.config import ALL_TARGETS, TARGETS_CLASSIFICATION
    from .strategy.distributions import fit_best_distribution, generate_qq_plots
    from .strategy.evaluate import calibration_curve_data

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models)

    targets = [args.target] if args.target else ALL_TARGETS
    features_path = models_dir.parent / "features" / "game_features.parquet"
    df = pd.read_parquet(features_path) if features_path.exists() else None

    for target in targets:
        oof_files = list(models_dir.glob(f"oof_{target}_*.npy"))
        if not oof_files:
            print(f"No OOF files found for target {target} in {models_dir}, skipping")
            continue

        if df is not None and target in df.columns:
            y_true = df[target].dropna().values
            if target not in TARGETS_CLASSIFICATION:
                for oof_file in oof_files:
                    oof = np.load(oof_file)
                    valid = ~np.isnan(oof) & ~np.isnan(y_true[:len(oof)])
                    if valid.sum() > 30:
                        residuals = y_true[:len(oof)][valid] - oof[valid]
                        generate_qq_plots(residuals, output_dir, target)
                        fit_result = fit_best_distribution(residuals)
                        print(json.dumps({target: fit_result}, indent=2, default=str))

    print(f"Evaluation artifacts written to: {output_dir}")


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
