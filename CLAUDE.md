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

# Run EDA on an already-built feature store
PYTHONPATH=deep_learning conda run -n pred python -m mlb_dl.train run-eda \
  --feature-store deep_learning/artifacts/feature_store \
  --output deep_learning/artifacts/eda

# Run EDA directly from S3 (builds feature store internally first)
PYTHONPATH=deep_learning conda run -n pred python -m mlb_dl.train run-eda \
  --source s3://mlb-265753586044-us-east-1-an/data \
  --output deep_learning/artifacts/eda \
  --season-start 2022
```

### Tests
```bash
cd deep_learning && conda run -n pred python -m pytest tests/smoke_test.py -v
# Run a single test function
cd deep_learning && conda run -n pred python -m pytest tests/smoke_test.py::test_smoke -v
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
    ↓ PregameSequenceDataset (datasets.py) — temporal windowing + game-index decay weights
PyTorch DataLoader → PregameMultiTaskModel (models.py)
    ↓
deep_learning/artifacts/pregame_cnn/
  model.pt                  ← best checkpoint (keyed: model_state, standardizer, feature_columns, config)
  checkpoint_epoch{NNN}.pt  ← per-epoch checkpoints with same structure
  history.json              ← per-epoch train/val loss
  eval_val/, eval_test/     ← SeasonReport JSON artifacts
```

### Key design decisions

**Leakage policy**: Pregame samples use only rows with `game_date < target_game_date`. This boundary is enforced in `feature_store.py` and documented in `manifest.json`. Standardizer is also fit only on training rows to prevent leakage through normalization statistics.

**Distribution heads, not per-line heads**: `PregameMultiTaskModel` outputs Bernoulli logits (`home_win`, `yrfi`) and Gaussian parameters (`total_runs`, `home_run_diff`). Any spread or total line is priced at inference by integrating the predicted distribution — not by training a separate head per line. This covers all 21 market families in `targets.market_specs()`.

**Loss weighting**: The training objective weights tasks as `home_win + 0.5·yrfi + 0.25·total_runs + 0.25·run_diff`. The reduced weight on continuous targets prevents the Gaussian NLL from dominating the binary cross-entropy terms during early training. See `train._loss()`.

**Target status**: Every target row carries `target_status ∈ {trainable, settles_last_fair, no_appearance}`. Training only uses `trainable` rows; the others are retained for audits. Scratched/non-starting players settle to last fair price and must never be trained as ordinary zero outcomes.

**Game-index decay (not calendar-day decay)**: Sample weights use sequential game-index distances instead of calendar days. The original calendar-day `exp(-λ·days)` unfairly penalised prior-season games by 5 months of offseason dead time. The replacement uses two lambdas: `λ_intra=0.015` (per game elapsed within a span) and `λ_inter=0.30` (applied once per season boundary crossed). See `SequenceSpec` and `compute_game_decay_weight()` in `datasets.py`.

**Sequence modeling**: Each team's prior 20 games become input sequences. Sequences shorter than `min_history` are dropped. Shorter-than-full sequences are left-padded with zeros; a boolean mask prevents the padded positions from contributing to the CNN's pooling step. Masks carry two roles: NaN imputation (feature-level) and padding (timestep-level) — both are floats fed through the network separately.

**Structural break features**: `feature_store.py` defines `MLB_REGIME_CHANGES` — binary flags for the 3-batter minimum (2020), universal DH (2022), and shift ban/pitch clock (2023). These are appended as numeric features so the model can learn distribution shifts at rule-change boundaries rather than confounding them with team form.

**Batch contract**: All models use dict-in / dict-out. The input dict always contains `{side}_values`, `{side}_mask`, `{side}_padding` tensors. Output dicts use consistent key names (`home_win_logit`, `total_runs_mu`, `total_runs_sigma`, etc.) that are referenced by name in `train._loss()` and `evaluation.py`.

**Player identity**: `PregamePlayerModel` and `LiveGameModel` use hash-bucket embeddings (`blake2b mod bucket_count`) rather than learned integer-ID embeddings. This handles unseen player IDs at inference without an OOV token.

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
| `mlb_dl/train.py` | CLI entry point for `build-feature-store`, `fit-pregame`, `run-eda` |
| `mlb_dl/eda.py` | Distribution analysis on feature store artifacts |

## Scientific rigour

Every modelling decision — hyperparameter values, feature engineering choices, architecture variants, loss weightings, decay schedules — must be grounded in one of the following:

1. **Empirical validation**: measure the effect on held-out val/test metrics before committing. A change that "seems reasonable" is not sufficient — run the experiment and record the delta.
2. **Published research or domain literature**: cite the paper, book, or established baseball-analytics finding that motivates the choice. A comment with no citation is not justification.
3. **First-principles derivation**: show the mathematical or statistical argument in a code comment or accompanying note. If you cannot derive it, it needs empirical backing instead.

**Heuristics and assumptions are not acceptable without validation.** This includes:
- Decay lambda values (λ_intra, λ_inter, time_decay_lambda) — each must reference why that value was chosen, not just what it produces
- Loss task weights (the `1 + 0.5 + 0.25 + 0.25` balance) — verify that reweighting does not hurt calibration
- `min_history` thresholds — back with an analysis of how sample count and model accuracy trade off at different cutoffs
- Regime-change dates — confirm the distributional shift in the data, not just that a rule change occurred

When you introduce a new constant, threshold, or architectural choice, the comment must state the validation evidence or citation, not just describe the effect. If validation has not yet been run, mark it explicitly with `# TODO: validate — current value is a placeholder` so it is visible and trackable.

## Code commenting philosophy

**Always comment the WHY, not the WHAT.** When writing new code, add inline comments that explain architectural choices — hidden constraints, non-obvious invariants, tradeoffs made. Well-named identifiers explain what; comments explain why.

Good examples already in the codebase:
- `SequenceSpec`: decay lambda values include numerical reasoning (`half-weight after ~46 games`)
- `MLB_REGIME_CHANGES`: each entry names the measurable effect on the data distribution
- `_last_prior_game_weight`: explains that `Δ=0` for the most recent game and what that implies

Add comments like these whenever you introduce or modify a decision that a future reader couldn't derive from the code alone. Skip comments that restate the function name or describe mechanics already clear from the code.

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

Existing log files:
- `download_history.py` → `data/logs/gumbo_ingest.log`
- `live_daemon.py` → `data/logs/gumbo_live.log`
- `train.py` → `data/logs/train.log`
- `feature_store.py` → stdout only (no file handler yet — add when modifying)

When adding new scripts, pick a descriptive log file name under `data/logs/` and follow the two-handler pattern above. Never remove or weaken existing log statements.
