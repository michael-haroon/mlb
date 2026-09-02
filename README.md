# MLB Prediction-Market Trading System

A pipeline for pricing and trading MLB markets on Kalshi. It ingests MLB game
data (MLB Stats API / GUMBO feed), engineers features, trains models to
predict game and player outcomes, prices Kalshi contracts from those
predictions, and can place/manage orders via the Kalshi API.

This project is no longer actively developed. It is published for reference.

## Layout

- `classical_learning/` — the primary pregame modeling stack: feature
  engineering, feature-importance analysis (de Prado-style MDI/MDA/SFI +
  clustering), LOYO-CV training with Optuna HPO, evaluation/calibration, and
  inference. Entry point: `classical_learning/cli.py` (`eda`, `build-features`,
  `run-importance`, `train`, `evaluate`, `predict` subcommands).
- `deep_learning/` — a second modeling stack (`GameTransformer`) that
  consumes live in-game pitch-sequence state rather than pregame aggregates.
  Training is implemented (`mlb_dl/train_unified.py`); real-time serving of
  this stack is not wired up. See `deep_learning/README.md` and
  `deep_learning/ARCHITECTURE.md`.
- `trading/` — Kalshi integration: REST client with RSA-PSS request signing
  (`kalshi_client.py`), a websocket client for order/fill/orderbook updates
  (`ws.py`), position sizing (`sizing.py`), risk limits (`risk.py`), and the
  main trading loop (`runner.py`, run via `python -m trading`).
- `data_curation/` — scripts that pull and cache raw MLB game/player data and
  weather data used by feature engineering.
- `research/` — exploratory analysis (e.g. feature clustering) that fed into
  modeling decisions but isn't part of the production pipeline.
- `data/` — local working directory for engineered features, logs, and model
  artifacts. Bulk data (parquet, CSVs, feature stores) is not committed;
  see `.gitignore` — the canonical copies live on S3.
- `scripts/` — one-off analysis, backtest, and EC2 launch/orchestration
  scripts used during development.
- `tests/` — test suite for `classical_learning`; each package also carries
  its own `tests/` (e.g. `trading/tests/`, `deep_learning/tests/`).

## Key docs

- `CLAUDE.md` — architecture summary and engineering conventions used while
  building this repo.
- `SCHEMA.md` — data schema reference.
- `DATA_AVAILABILITY.md` — timing of when each MLB data source becomes
  available relative to game time.
- `TARGETS.md` — the Kalshi market families this system targets (moneyline,
  totals, spreads, first-5-innings, YRFI/NRFI, player props, extra innings).
- `MODELS.md` — exploratory architecture notes (not a status report).

## Environment

Developed against a conda environment named `pred`
(`conda run -n pred python ...`). Kalshi credentials and other secrets are
supplied via a local `.env` file (see `trading/kalshi_client.py` for the
expected variable names); no credentials are committed to this repository.
