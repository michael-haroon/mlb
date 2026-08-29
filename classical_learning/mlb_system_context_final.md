# MLB Pregame-to-Live Prediction System — Architecture & Design

For: Gemini / System Architecture Review

---

## System Overview

**Mission:** Price 21 market families (home win, YRFI, totals, run differential, etc.) via a two-phase ML stack:
- **Pregame (Pre-First-Pitch):** Classical ensemble ML on team/pitcher/context features
- **Live (In-Game):** Deep learning on pitch-by-pitch state for real-time repricing

**Core Pipeline:**
```
Raw Data (MLB API, Statcast, PITCHf/x)
    ↓
Feature Engineering (14 pregame artifact families + live pitch sequences)
    ↓
Stage 1: Pregame Pricing (sklearn ensemble, inference ~100ms)
    ↓
Stage 2: Live In-Game Repricing (deep learning, streaming pitch state, latency <50ms)
    ↓
Market Pricing & Two-Sided Quotes
    ↓
Execution & Position Management
```

---

## Pregame Architecture (Complete)

### Data Tiers
- **Tier A** (2015–present): Full Statcast coverage; 144+ advanced metrics
- **Tier B** (2008–2014): PITCHf/x only; pitcher movement features
- **Tier C** (all history): Game-level aggregates; team strength only

### Feature Design — 14 Core Artifact Families

| Family | Scope | Dimensionality | Key Signals | Availability |
|--------|-------|-----------------|-------------|--------------|
| **Team Ratings** | Team strength derived from history | 4–6 features | Elo, SRS, Wolfe, Log5, Pythagenpat | Every game (historical baseline) |
| **SP Quality** | Pitcher-specific metrics | 8–12 features | FIP, ERA, velocity, K/9, BB/9 momentum | 92% (TBD SP ~8% of games) |
| **Bullpen Quality** | Relief ERA, workload, recent usage | 4–6 features | Bullpen ERA rolling, avg appearances, stress indicator | 100% (team aggregate, always available) |
| **Offense** | Batting stats (rolling windows) | 10–15 features | wRC+, ISO, K%, BB%, BABIP, exit velo | 100% (team-level rolling) |
| **Defense** | Fielding & positioning | 3–5 features | UZR/150, shift%, positional quality | 60% (availability varies by tier) |
| **Recency / Momentum** | Short-term form | 6–10 features | Runs scored/allowed (5g, 10g), streak W/L | 100% (game-by-game) |
| **Matchup** | SP vs batter dynamics | 6–8 features | LHH% vs RHP, splits, handedness factors | 92% (tied to SP availability) |
| **Rest & Context** | Game logistics | 4–6 features | Days rest, day/night, altitude, temp, ballpark factor | 100% |
| **Efficiency** | Process metrics | 4–6 features | BABIP normalized, LOB%, soft/medium/hard contact % | 85% (Statcast era only) |
| **Season Trends** | Macro form | 3–4 features | Season W%, run diff/game, last-30 performance | 100% |
| **Player Absence** | Scratched/injured lineup impact | 2–3 features | Lineup WAR delta, key positional absence | 30% (data quality varies) |
| **Availability Signals** | Missing data indicators | 2–3 binary features | `sp_era_observed`, `bullpen_quality_observed` | 100% (synthetic) |
| **Pitcher Adjustment Signals** | Game-day pitching changes | 2 features | Days since last appearance, piggyback indicator | 100% |
| **Acquisition / Roster** | Mid-season trades, transactions | 1–2 features | Deadline acquisitions, youth rating | 50% |

**Total Pregame Features:** ~80–120 after feature selection and importance filtering

### Missing Data Strategy

**Starting Pitcher Problem (8% of games):**
- SP metrics filled with **0.0** (neutral matchup proxy)
- Binary indicator `sp_era_observed = 0` signals to linear/MLP models: "this is likely bullpen game, high variance"
- Tree models (XGBoost, LightGBM) use native NaN handling; observation masks automatically weight down TBD-SP pitches

**Tier-Based Filtering:**
- Tier A: All 144 Statcast features available
- Tier B: Exclude pitch-tracking velocity features; use PITCHf/x proxies
- Tier C: Team aggregates only; no pitcher granularity

**Imputation Hierarchy:**
1. **Tree models** (XGBoost, LightGBM, CatBoost, RandomForest, ExtraTrees, HistGradientBoosting): Native NaN support via observation masks ✅
2. **Linear models** (LogReg, Ridge, Lasso, MLP, KNN): Require imputation before training
3. **AdaBoost + RandomForest + ExtraTrees:** Require imputation (sklearn DecisionTree-based ensemble implementations do NOT handle NaN natively)

**When Imputation Happens:**
- **Location:** `data.py:325-327` in `prepare_fold()`, BEFORE model training
- **Trigger:** `if model_family in NEEDS_IMPUTATION: X_train = _semantic_impute(X_train); X_val = _semantic_impute(X_val)`
- **Strategy:** `_semantic_impute()` fills NaN with domain-correct static priors (not data-dependent means/medians to avoid lookahead bias)

**Semantic Fill Rules (Priority Order):**
| Feature Pattern | Fill Value | Rationale |
|---|---|---|
| `winrate`, `log5_prob`, `consensus_home_win_prob` | **0.5** | No-edge prior for probability features |
| `park_factor` | **1.0** | League-average multiplier by definition |
| `days_rest` | **7.0** | Offseason proxy; first game of season gets ~1 week rest |
| `sp_era_diff`, `sp_whip_diff` | **0.0** | No-edge prior when both SPs unknown; E[away_ERA - home_ERA] ≈ 0 |
| `season_era` | **4.50** | Replacement-level pitcher (2015-2024 MLB avg: ~4.2-4.5) |
| `season_whip` | **1.30** | Replacement-level pitcher |
| `venue_latitude`, `venue_longitude` | **0.0** | Non-informative for neutral-site games |
| **Everything else** (rolling stats, streaks, differentials, sums) | **0.0** | Absence of signal; zero-centered stats (SRS, run_diff) default to league average |

**Example:** A game with TBD starting pitchers gets:
- `sp_era_diff=0.0` (neutral), `sp_whip_diff=0.0` (neutral)
- `season_era=4.50` (replacement level), `season_whip=1.30` (replacement level)
- Linear/MLP models see these consistent, semantically meaningful fills
- Tree models with NaN support ignore them entirely; observation masks downweight to zero signal

---

## Live Architecture (Planned Deep Learning Stack)

### Data Pipeline: WebSocket → State Accumulation → Inference

The live module consumes real-time game data via **MLB StatsAPI WebSocket + Diff polling**:

```
┌─────────────────────────────────────────────────────────────┐
│  MLB StatsAPI WebSocket (wss://ws.statsapi.mlb.com/...)     │
│  • Heartbeat: Gameday5 every ~10s                           │
│  • Push frames: gameEvents (hit_into_play, field_out, etc.) │
│  • updateId + timeStamp in each event                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────┐
         │  Local Game State Accumulator       │
         │  • Linescore (score, inning, outs) │
         │  • Baserunners (occupancy)         │
         │  • Batter/Pitcher identity         │
         │  • Play history (all events so far)│
         └────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        Every Pitch      Timeout (>2s)   Connection Drop
              │               │               │
              ▼               ▼               ▼
        ┌─────────────────────────────────────────┐
        │  Diff Endpoint (https://...game/{pk}...) │
        │  ?timestamp={lastUpdateId}              │
        │  • Returns full GUMBO state as of ts   │
        │  • Patch local state if diverged       │
        └─────────────────────────────────────────┘
```

**WebSocket Endpoints (Tested & Functional):**
- **Event stream:** `wss://ws.statsapi.mlb.com/api/v1`
- **Game subscriptions:** `subscribe` to `game-{gamePk}` topics
- **Push latency:** ~100-200ms behind live game
- **Reliability:** Heartbeat prevents disconnection

**Fallback (Diff Polling):**
- **Endpoint:** `https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live?timestamp={lastUpdateId}`
- **Response:** Full GUMBO (gameData + liveData) as immutable diff
- **Usage:** Patch local state after connection gap or validation

### Pitch-Level Feature Engineering

```
Per-pitch vector (16–24 features):
  - Pitch attributes: type, velocity, movement (spin rate, axis), location (px, pz)
  - Game state: score, inning, bases, outs, balls/strikes, batter count
  - Pitcher state: pitches thrown (in PA, in game), velocity trend (drift?), error rate
  - Batter state: contact%, exit velo trend, strikeout tendency, plate discipline
  - Context: temperature, wind, ballpark geometry, home run factor
  - Time encoding: seconds since last pitch (captures asymmetry, delays)
```

**State Accumulation Strategy:**
1. **Per-Plate-Appearance:** Collect all pitches; compute PA-level features (length, contact rate)
2. **Per-Inning:** Aggregate all PAs in inning; momentum features (runs scored, outs)
3. **Per-Game:** Rolling statistics (fatigue decay, velocity regression)

### Proposed Deep Learning Models (from MODELS.md)

| Model | Architecture | Input Tensor Shape | Key Advantage | Use Case |
|-------|--------------|-------------------|---------------|----------|
| **Hierarchical Attention Network (HAN)** | Pitch → PA → Inning → Game (4D encoding) | `(B, I, A, P, F)` | Structural alignment with baseball rules | YRFI, First-5, inning-level targets |
| **Selective State Space (Mamba)** | Linear-time sequence encoder | `(B, S, F)` with data-dependent forget gates | O(S) scaling vs Transformer's O(S²); selective memory of critical pitches | Full-game repricing (efficient production) |
| **Neural ODE** | Continuous-time state evolution | `(B, S, F)` + Δt per pitch | Handles asynchronous gaps (rain delays, pitching changes) | Long-game stability |
| **Cross-Attention Merger** | Static (pregame) + Live (pitch sequences) towers | Static: `(B, F_static)`, Live: `(B, S, F_live)` | Conditions pitch evaluation on team quality priors | Hybrid: pregame baseline + live updates |
| **Transformer Encoder** | Multi-head attention over flat sequence | `(B, S, F)` | Proven strong baseline; standard infrastructure | Comparison / production fallback |

### Live Model Outputs
- Per-pitch score: Updated probability for each of 21 market families
- Confidence interval (epistemic + aleatoric uncertainty)
- Repricing trigger: If confidence shift > threshold, generate new quote

### Latency Constraints
- **WebSocket push latency:** ~100-200ms (MLB API → browser/server)
- **Pitch-to-feature computation:** ~20ms (accumulate state in local tensor)
- **Inference latency budget:** <30ms for model forward pass
- **Quote generation + transmission:** <20ms
- **Total:** <100ms to new quote in market (achievable)

---

## Cross-Module Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                      Raw Feature Store                           │
│  (S3: team_games.parquet, pitch_sequences.parquet, context)      │
└──────────────────────────────────────────────────────────────────┘
       │                                      │
       ├──────────────────────────┬──────────┤
       ▼                          ▼          ▼
┌────────────────────┐  ┌──────────────────────────────┐
│  Pregame Pipeline  │  │  Live Pipeline (In-Game)     │
│ (sklearn ensemble) │  │ (Deep Learning Models)       │
│                    │  │                              │
│ - Train on 2015-Y  │  │ - WebSocket: pitch events   │
│ - OOF calibration  │  │ - State accumulation (PA)   │
│ - Ensemble weights │  │ - Real-time repricing      │
│                    │  │                              │
│ Output:            │  │ Output:                      │
│ P(outcome) per     │  │ P(outcome | game state)      │
│ market at T-0      │  │ per market every pitch       │
└────────────────────┘  └──────────────────────────────┘
       │                           │
       └──────────────┬────────────┘
                      ▼
         ┌────────────────────────┐
         │  Quote Generation      │
         │  (blended odds)        │
         │  & Sizing (Kelly)      │
         └────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  Order Execution       │
         │  & Position Management │
         └────────────────────────┘
```

---

## Cross-Validation & Calibration Strategy

### Pregame Training: Leave-One-Year-Out (LOYO)
```python
for val_season in [2015, 2016, ..., 2026]:
    train_seasons = [s for s in all_seasons if s < val_season]  # Temporal integrity
    ensemble.fit(data[data.season.isin(train_seasons)])
    oof_pred[val_season] = ensemble.predict(data[data.season == val_season])
```

**Why LOYO:**
1. **Temporal integrity:** No future data in training
2. **Covariate shift:** League conditions evolve (dead-ball → live-ball → analytics era)
3. **Production alignment:** Training on 2015-Y data to predict 2027+ mirrors reality

**Calibration Pipeline (Classification Targets):**
1. Fit per-model isotonic on OOF predictions (fold-specific)
2. Ensemble via SLSQP weight optimization (minimize log-loss)
3. Blend per-model isotonic outputs (no secondary overfit)

### Live Validation: Walk-Forward Testing
- Train on games 1–100 of season
- Validate on games 101–162
- Measure repricing edge (% of times live model's quote is more accurate than pregame)

---

## Ensemble Architecture

### Pregame Stage 1: Model Diversity
- **Linear:** LogisticRegression, Ridge, Lasso, ElasticNet, KNN, MLP
- **Tree-based:** XGBoost, LightGBM, CatBoost, RandomForest, ExtraTrees, HistGradientBoosting
- **Other:** SGD, Bagging, Gaussian NB, LDA, QDA

**Selection:** Train all ~15 families, use top 8–12 by validation performance after correlation filtering (`max_correlation = 0.95`)

### Pregame Stage 2: Weight Optimization
- **Objective:** Minimize log-loss on OOF predictions
- **Algorithm:** Sequential Least Squares Programming (SLSQP)
- **Constraint:** Weights sum to 1.0, non-negative

### Pregame Stage 3: Calibration
- **Per-model isotonic:** Monotonic transform on [0, 1]
- **Ensemble blend:** Weighted average (using Stage 2 weights)
- **Result:** Final probability per market

---

## Feature Fortification Priorities

### High Priority (System-Level)

1. **Bullpen Quality Indicator**
   - **Signal:** Team's bullpen ERA rolling window (always available vs TBD SP)
   - **Impact:** Fills structural gap when SP is TBD (~8% of games)
   - **Integration:** Add to Bullpen family; linear/MLP models learn the MNAR signal

2. **SP Observation Marker**
   - **Design:** Binary `sp_era_observed` (1 if SP stats loaded, 0 if TBD/unknown)
   - **Impact:** Linear models distinguish "average matchup fill" from "genuine unknown"
   - **Data:** Synthetic; 1 for 92% of games, 0 for 8%

3. **Pitcher Fatigue Signal**
   - **Features:** Velocity drop (current inning vs season avg), pitch count acceleration, error rate
   - **Scope:** Live model only (requires pitch-level data)
   - **Impact:** Identifies innings 6–9 when control degrades

### Medium Priority (Feature Engineering)

4. **Exit Velocity Distribution (live model)**
   - Contact quality tracking per inning
   - Signals offensive momentum shifts

5. **Momentum Decay Weights**
   - Game-index decay (sequential distance) vs calendar days
   - Avoids penalizing offseason gaps; recency bias natural

6. **Ballpark Adjustment Evolution**
   - Year-by-year tuning (balls play differently with rule changes, humidor)
   - Context-specific HR factors

### Lower Priority (Nice-to-Have)

7. Platoon splits (LHH vs RHP, RHH vs LHP)
8. Bullpen sequencing models (who gets called, impact on game state)
9. Weather interaction terms (temperature × ballpark altitude effect on fly balls)

---

## Known Data Issues & Status

### ✅ RESOLVED
- **OOF/Parquet Misalignment (July 6 2026):** Tail-aligned OOF array to regenerated parquet; per-model isotonic now fits on clean data

### 🔴 HIGH PRIORITY (Rating Leakage)
- **Issue:** Rating hyperparameters (Elo K-factors, SRS iterations) tuned on same seasons used as LOYO validation folds
- **Impact:** ~1–3% optimistic Brier score
- **Fix:** Retune ratings on seasons strictly before validation seasons
- **Timeline:** Before next retrain cycle

### 🟡 MEDIUM PRIORITY (HPO Distribution Mismatch)
- **Issue:** Optuna HPO on latest fold's data; hyperparams applied universally
- **Impact:** Distribution mismatch for older validation years
- **Fix:** Nest HPO inside each LOYO fold
- **Timeline:** Medium term (lower impact than rating leakage)

---

## Current Metrics & Baseline

### Pregame Classification Targets (Stage1 blend, n=7327)
| Target | ECE | Brier | Notes |
|--------|-----|-------|-------|
| home_win | 0.1213 | 0.2733 | Baseline: ~0.25 (random) |
| yrfi | 0.0079 | 0.2502 | Well-calibrated |
| extra_innings | 0.0081 | 0.0677 | Well-calibrated |
| first_5_home_win | 0.0461 | 0.2514 | Good calibration |

**Interpretation:**
- YRFI and extra_innings: High-quality predictions, low miscalibration
- Home_win: Structurally challenging (high entropy); moderate calibration
- All above random baseline → models have real signal

---

## Production Deployment & Latency Budget

### Pregame Inference (<100ms per game)
1. Load 8–12 model binaries: ~20ms
2. Feature fetch (S3 cache): ~20ms
3. Model inference (ensemble): ~40ms
4. Calibration (isotonic): ~15ms
5. Quote generation (Kelly): ~5ms
**Total:** ~100ms (ample for pre-first-pitch)

### Live Inference (<100ms per pitch)
1. WebSocket push received: ~100-200ms (MLB API latency)
2. Pitch feature assembly: ~20ms
3. Live model forward pass (Mamba or HAN): ~20ms
4. Calibration (lightweight): ~10ms
5. Quote generation: ~5ms
6. Network latency to exchange: ~5ms
**Total:** ~150ms wall-clock (within acceptable bounds for intra-game repricing)

---

## Trading Integration

### Pregame Phase (Pre-First Pitch)
- Generate two-sided quotes at tight spreads
- Size positions using Kelly criterion (1.5% max per order)
- Fight for top-of-book via reprice loop (every 60 sec)
- Monitor until first pitch

### Live Phase (In-Game)
- Real-time repricing on every pitch (~80–100 pitches per game = 80–100 quote updates)
- Update probabilities for each market based on game state (via WebSocket)
- Exit weak positions for profit if threshold met
- Hold strong positions to settlement

### Risk Controls
- Bankroll-based Kelly sizing
- Reprice cooldown (10 sec minimum between updates)
- Max reprices per order (8 times, then hold)
- Dry-run mode for testing new strategies

---

## System Strengths & Architectural Advantages

1. **Two-Phase Design:** Pregame model is low-latency, fully interpretable (sklearn). Live model optimized for speed & depth (DL).
2. **Temporal Rigor:** LOYO respects causality; no future-data leakage in training evaluation.
3. **Feature Richness:** 14 artifact families across multiple timescales (single-game, rolling 5/10/20g, season-long).
4. **Calibration Pipeline:** Per-model isotonic + ensemble blend avoids secondary overfitting (unlike CalibrationBundle approach).
5. **Adaptability:** Tier-based feature filtering allows retroactive models (2008+) and robust degradation (Tier C fallback).
6. **Real-Time Data Ingestion:** WebSocket + diff-polling provides sub-200ms game state updates; stateless local accumulator ensures resilience.

---

## Next System Goals

### Phase 1: Live Model Deployment
- Implement Hierarchical Attention Network (HAN) for inning-level targets (YRFI, First-5)
- Validate walk-forward repricing edge on 2026 season
- Optimize latency via ONNX export (if needed for sub-50ms inference)

### Phase 2: Feature Fortification
- Add bullpen quality indicator + SP observation marker
- Implement pitcher fatigue signal (velocity, pitch count trends) in live feature pipeline
- Backtest repricing lift from live model vs pregame-only

### Phase 3: Production Hardening
- A/B test pregame-only vs pregame+live pricing
- Measure Sharpe ratio, max drawdown, Kelly alignment
- Deploy with safety nets (Kelly override, position limits, WebSocket fallback)

---

## Related Documentation
- **CLAUDE.md:** Codebase instructions & code philosophy
- **MODELS.md:** Deep learning architectures (HAN, Mamba, Neural ODE, Cross-Attention)
- **Predictive MLB Team Ratings.md:** Rating system mathematics & implementations
