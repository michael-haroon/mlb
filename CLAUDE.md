# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

The conda environment is `pred`. Prefix all Python commands with `conda run -n pred`.

## Commands

### Install dependencies
```bash
conda run -n pred python -m pip install -r deep_learning/requirements-deep-learning.txt
```

### Data ingestion
```bash
# Download historical games to S3 (default) or local
conda run -n pred python data_curation/scripts/download_history.py --season-start 2015 --season-end 2024
conda run -n pred python data_curation/scripts/download_history.py --season-start 2024 --local

# Real-time live game polling daemon
conda run -n pred python data_curation/scripts/live_daemon.py

# Fix S3 Parquet schema inconsistencies (safe to re-run, idempotent)
conda run -n pred python data_curation/scripts/standardize_parquets.py [--dry-run] [--table pitches] [--season 2024]
```

### Feature store and training
```bash
# Build feature store from local data
PYTHONPATH=deep_learning conda run -n pred python -m mlb_dl.train build-feature-store \
  --source data \
  --output deep_learning/artifacts/feature_store \
  --season-start 2015

# Build feature store from S3
PYTHONPATH=deep_learning conda run -n pred python -m mlb_dl.train build-feature-store \
  --source s3://mlb-265753586044-us-east-1-an/data \
  --output deep_learning/artifacts/feature_store \
  --season-start 2015

# Train pre-game CNN model
PYTHONPATH=deep_learning conda run -n pred python -m mlb_dl.train fit-pregame \
  --feature-store deep_learning/artifacts/feature_store \
  --output deep_learning/artifacts/pregame_cnn \
  --history-length 20 \
  --epochs 20
```

### Tests
```bash
cd deep_learning && conda run -n pred python -m pytest tests/smoke_test.py -v
```

## Architecture

### Data flow
```
MLB statsapi.mlb.com
    ↓ download_history.py (80-worker ThreadPoolExecutor, 10 req/sec rate limit)
S3: mlb-265753586044-us-east-1-an/data/season={YYYY}/table_batch_*.parquet
    ↓ ParquetCatalog (data_sources.py) — abstracts S3 vs. local paths
feature_store.py — aggregates 14 output artifacts
    ↓
deep_learning/artifacts/feature_store/
  team_games.parquet        ← primary model input (~500 numeric features per team-game)
  game_targets.parquet
  player_batting_targets.parquet
  player_pitching_targets.parquet
  pitch_sequences.parquet   ← reserved for future live models
    ↓ PregameSequenceDataset (datasets.py) — temporal windowing + time-decay weights
PyTorch DataLoader → PregameMultiTaskModel (models.py)
    ↓
deep_learning/artifacts/pregame_cnn/
  model.pt, history.json, eval_val/, eval_test/
```

### Key design decisions

**Leakage policy**: Pregame samples use only rows with `game_date < target_game_date`. This boundary is enforced in `feature_store.py` and documented in `manifest.json`.

**Distribution heads, not per-line heads**: `PregameMultiTaskModel` outputs Bernoulli logits (home_win, yrfi) and Gaussian parameters (total_runs, home_run_diff). Any spread or total line is priced at inference by integrating the predicted distribution — not by training a separate head per line. This covers all 21 market families in `targets.market_specs()`.

**Target status**: Every target row carries `target_status ∈ {trainable, settles_last_fair, no_appearance}`. Training only uses `trainable` rows; the others are retained for audits. Scratched/non-starting players settle to last fair price and must never be trained as ordinary zero outcomes.

**Time decay**: Sample weights use `exp(-lambda * age_days)` (default λ=0.003 → half-weight after ~231 days) to upweight recent games in the CNN training objective.

**Sequence modeling**: Each team's prior 20 games become input sequences. Sequences shorter than `min_history` are dropped. Shorter-than-full sequences are left-padded with zeros; a boolean mask prevents the padded positions from contributing to the CNN's pooling step.

**Live state**: `live_daemon.py` writes JSON snapshots to `data/live_state/{game_pk}.json`. On game `Final`, it delegates full persistence to `download_history.py`. `pitch_sequences.parquet` in the feature store is the future input for in-game repricing.

### Module responsibilities

| Module | Role |
|--------|------|
| `data_curation/scripts/download_history.py` | MLB API → Parquet batches |
| `data_curation/scripts/live_daemon.py` | Real-time game state polling |
| `data_curation/scripts/standardize_parquets.py` | One-time S3 schema repair |
| `mlb_dl/data_sources.py` | `ParquetCatalog`: discover + read Parquet from S3 or local |
| `mlb_dl/targets.py` | Raw tables → settlement labels + `market_specs()` |
| `mlb_dl/feature_store.py` | Labels + raw tables → 14 model-ready artifacts |
| `mlb_dl/datasets.py` | `PregameSequenceDataset`, `Standardizer`, temporal splits |
| `mlb_dl/models.py` | `PregameMultiTaskModel` (CNN), `PregamePlayerModel`, `LiveGameModel` (stub) |
| `mlb_dl/distributions.py` | `gaussian_nll`, `weighted_mean`, `suggest_distribution` |
| `mlb_dl/evaluation.py` | `BinaryMetrics`, `GaussianMetrics`, `SeasonReport` |
| `mlb_dl/train.py` | CLI entry point for `build-feature-store` and `fit-pregame` |

## Logging conventions

All scripts use two handlers:

1. **File handler** — `DEBUG` level, granular. Captures every step, SQL-like query, row counts, API call details, retry attempts.
2. **Stdout handler** — `INFO` level, summarized. Shows high-level progress milestones. Warnings and errors always appear regardless of level.

### Setup pattern
```python
import logging, sys
from pathlib import Path

LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# File: granular
_fh = logging.FileHandler(LOG_DIR / "script_name.log")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(threadName)s %(message)s"))
log.addHandler(_fh)

# Stdout: summarized progress
_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.INFO)
_sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
log.addHandler(_sh)
```

Existing scripts:
- `download_history.py` → `data/logs/gumbo_ingest.log`
- `live_daemon.py` → `data/logs/gumbo_live.log`
- `feature_store.py` → stdout only (no file handler yet — add when modifying)

When adding new scripts, pick a descriptive log file name under `data/logs/` and follow the two-handler pattern above.
