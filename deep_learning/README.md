# MLB Deep Learning Trading System

This package is the deep-learning stack of the MLB modeling system: live
in-game models (GameTransformer over pitch sequences) trained via
`mlb_dl/train_unified.py` and served via `mlb_dl/inference_engine.py`.
Classical pregame market making lives in `classical_learning/`. The staged
plan below is the original roadmap; stage 4 (live pitch-sequence models) is
now implemented.

## Decision

Deep learning is worth using here, but not as a giant model that swallows every
raw table at once. With about 5 GB of MLB data, the robust path is:

1. Build official, auditable targets from linescore and boxscore tables.
2. Train a pre-game sequence model on prior team/player histories.
3. Calibrate market probabilities and compare against simple baselines.
4. Add live pitch-sequence models once the target and calibration layer is stable.

This lets neural nets learn recent-form representations while preserving the
settlement rules and temporal boundaries that matter for trading.

## Install

From the repository root:

```bash
conda run -n pred python -m pip install -r deep_learning/requirements-deep-learning.txt
```

## Build The Feature Store

Local parquet data:

```bash
PYTHONPATH=deep_learning conda run -n pred python -m mlb_dl.train build-feature-store \
  --source data \
  --output deep_learning/artifacts/feature_store \
  --season-start 2015
```

S3 parquet data:

```bash
PYTHONPATH=deep_learning conda run -n pred python -m mlb_dl.train build-feature-store \
  --source s3://mlb-265753586044-us-east-1-an/data \
  --output deep_learning/artifacts/feature_store \
  --season-start 2015
```

The output contains:

- `game_targets.parquet`: game-level settlement labels.
- `player_batting_targets.parquet`: batter prop labels including total bases.
- `player_pitching_targets.parquet`: pitcher prop labels.
- `team_games.parquet`: per-team completed-game history rows used as model input.

## Train Pre-Game Model

```bash
PYTHONPATH=deep_learning conda run -n pred python -m mlb_dl.train fit-pregame \
  --feature-store deep_learning/artifacts/feature_store \
  --output deep_learning/artifacts/pregame_cnn \
  --history-length 20 \
  --epochs 20
```

The model predicts:

- `home_win`: Bernoulli probability.
- `yrfi`: Bernoulli probability.
- `total_runs`: Gaussian predictive distribution.
- `home_run_diff`: Gaussian predictive distribution.

Spreads and totals are priced by integrating the predicted distribution over a
line, not by training one head for every possible line.

## Rule Notes

Kalshi's baseball entity-stat terms make three details important:

- Extra innings count unless the contract title says otherwise.
- Sub-game periods only count statistics in that period.
- Scratched/non-starting player markets may settle to last fair price, so those
  rows should not be trained as ordinary zero outcomes.

The target builders keep `target_status` columns so training can include only
normal, trainable settlements and retain excluded rows for audits.
