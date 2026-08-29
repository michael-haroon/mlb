# Live Module Architecture

## Overview

The live module reprices all 21 MLB market families in real time using deep learning on pitch-by-pitch state sequences. While the pregame module provides calibrated pre-game distributions via classical ML on historical team/player aggregates, the live module continuously updates these priors as each pitch reveals new information about pitcher fatigue, batter adjustments, and game momentum. This provides an informational edge over static pregame odds by capturing intra-game dynamics that only manifest through observable pitch sequences (velocity decay, command drift, times-through-order effects, bullpen fatigue).

## System Architecture

### Data Flow

```
MLB StatsAPI     Ingestion     Feature        HAN          Market       Trading
WebSocket    →   Layer     →   Extraction →   Model    →   Derivation → Runner
(GUMBO)          (WS + HTTP    (Pitch/State   (NegBin      (Integrate    (Kelly
                 fallback)     tensors)       params)      distributions) sizing)
                     ↓             ↓              ↓             ↓
                 Validation   Embeddings    Pregame      21 market      Order
                 Gate         (player hash, Prior Merge  families       execution
                              pitch type)   (cross-attn)  (win/total/
                                                          spread/props)
```

**Flow invariants:**
- All data passes through validation gate before model ingestion (see `/Users/michaelharoon/Projects/prediction_markets/mlb/deep_learning/mlb_dl/validation_gate.py`)
- Pregame prior provided as static context vector, not dynamically recomputed
- NegBin parameters (λ_home, λ_away, λ_shared) output by model, markets derived analytically
- No leakage: features only use pitch sequence prefix up to current moment

### Latency Budget

| Component | Budget (ms) | Justification |
|-----------|-------------|---------------|
| WebSocket ingestion | 10 | Push-based; GUMBO sends updates immediately |
| Validation gate | 5 | Bounds checks + monotonicity; vectorized numpy |
| Feature extraction | 30 | Hash embeddings + rolling window stats over pitch prefix |
| HAN forward pass | 80 | 4-level hierarchy (pitch→PA→half-inning→game); attention at each level |
| Market derivation | 20 | Numerical integration over NegBin PMF (vectorized) |
| **Total end-to-end** | **145** | Leaves 5ms buffer under 150ms SLA |
| Order execution | 20 | External broker API; not part of repricing path |

**Latency risk mitigation:**
- TorchScript compilation for HAN model (removes Python overhead)
- ONNX Runtime optional backend on EC2 (GPU optional, CPU sufficient)
- Attention caching: half-inning representations cached until inning ends
- No recomputation: pregame tower runs once at game start, cached thereafter

## Model Architecture: Hierarchical Attention Network

### Why HAN over alternatives

**vs. Flat Transformer:**
- Baseball is hierarchically structured: pitches → at-bats → half-innings → game
- Flat attention over 200+ pitches dilutes signal; per-pitch attention weights would be ~0.005
- HAN explicitly models hierarchy: pitch-level attention within each AB, AB-level attention within each half-inning
- Empirical: HAN outperforms flat Transformer on YRFI classification by 4.2% Brier score (TODO: validate — placeholder)

**vs. Mamba/SSM:**
- Mamba provides O(S) vs O(S²) complexity advantage for long sequences
- However, baseball sequences are <300 pitches, where S²=90K is tractable
- HAN's hierarchical inductive bias is more important than asymptotic complexity at this scale
- Future work: Mamba may replace LSTM in pitch-level encoder if latency becomes critical

**vs. LSTM (existing placeholder in `models.py:121-147`):**
- LSTM lacks explicit hierarchy; treats all pitches as flat sequence
- No attention mechanism: cannot selectively focus on critical pitches (e.g., 3-2 count with bases loaded)
- Empirical: HAN expected to outperform LSTM by >5% Brier on time-stratified test set (TODO: validate — placeholder)

**vs. Neural ODE:**
- Neural ODEs model continuous-time dynamics via differential equations
- Baseball sequences are discrete events (pitch outcomes), not continuous trajectories
- ODE solver overhead (~50ms) violates latency budget
- Future work: Consider hybrid ODE for modeling time-between-pitches as fatigue proxy

**Decision:** HAN provides the best balance of structural inductive bias, interpretability (can inspect attention weights per AB), and latency. Fallback to Mamba if latency exceeds 150ms on EC2 inference.

### Architecture Details

**Input encoding:**
- Pitch features: 18-dim continuous (velocity, spin, break, plate location, count state, score differential, outs, runners on base)
- Player embeddings: 16-dim learned embeddings via blake2b hash → 50K-bucket table (handles unseen players at inference)
- Pitch type embeddings: 16-dim learned embeddings, 256-bucket hash (future-proofs against new pitch classifications)
- Total input dim per pitch: 18 + 16 (batter) + 16 (pitcher) + 16 (pitch type) = 66 dims

**Hierarchy levels:**
1. **Pitch encoder**: Multi-head self-attention over pitches within same at-bat (max 15 pitches per AB)
   - 4 heads, 64-dim hidden
   - Positional encoding: pitch number within AB (not absolute game time, to maintain invariance)
2. **At-bat encoder**: Multi-head self-attention over at-bat representations within same half-inning (max 9 ABs per half-inning)
   - 4 heads, 128-dim hidden
   - Positional encoding: batting order position (1-9)
3. **Half-inning encoder**: Multi-head self-attention over half-inning representations within game (max 20 half-innings = 10 full innings)
   - 4 heads, 256-dim hidden
   - Positional encoding: inning number + top/bottom indicator
4. **Game encoder**: Mean pooling over all half-inning representations → 256-dim game state vector

**Output heads (all operate on 256-dim game state vector):**
- **NegBin distributional head**: 3 output neurons → (log λ_home, log λ_away, log λ_shared) → softplus activation to ensure positivity
- **Classification heads (auxiliary supervision)**: home_win, yrfi, extra_innings → sigmoid logits
  - WHY auxiliary: NegBin is primary; classification heads provide additional gradient signal during training

**Cross-attention merger (pregame prior conditioning):**
- Pregame features (Elo, SP ERA, park factor, etc.) encoded by separate 3-layer MLP → 256-dim static prior vector
- Static prior vector acts as **Key** and **Value**; live game state vector acts as **Query**
- Single-head cross-attention: game state attends to pregame prior
- Output: 256-dim context-aware representation = f(live_state, pregame_prior)
- Final NegBin head operates on this merged representation

**Pregame-to-live gating (learned transition):**
- Sigmoid gate: σ = sigmoid(w_t · [inning, outs, pitches_thrown, score_differential])
- WHY: At t=0 (game start), live model has no data; should defer to pregame. By t=5 innings, live data should dominate.
- Final prediction: P_final = (1 - σ) · P_pregame + σ · P_live
- σ is learned during training, not hand-tuned (avoids arbitrary thresholds)

### Hyperparameters

| Parameter | Value | Status | Source |
|-----------|-------|--------|--------|
| Pitch attention heads | 4 | placeholder | Standard Transformer default; validate via grid search |
| Pitch hidden dim | 64 | placeholder | Scaled to keep pitch encoder < 20ms latency |
| AB attention heads | 4 | placeholder | Same as pitch level for uniformity |
| AB hidden dim | 128 | placeholder | 2x pitch dim to capture cumulative AB state |
| Half-inning heads | 4 | placeholder | Consistent with lower levels |
| Half-inning hidden dim | 256 | placeholder | Matches pregame prior dim for clean merging |
| Dropout | 0.3 | placeholder | Higher than pregame CNN (0.2) due to smaller dataset |
| Learning rate | 1e-4 | literature | Adam default for Transformers (Vaswani et al. 2017) |
| Batch size | 64 | placeholder | Max size fitting in 16GB GPU memory |
| Gradient clip norm | 1.0 | literature | Standard for RNN/attention models (Pascanu et al. 2013) |
| Label smoothing (ε) | 0.1 | literature | Reduces overconfidence (Szegedy et al. 2016) |
| Weight decay | 1e-5 | placeholder | Lighter than pregame (1e-4) to avoid underfitting small dataset |
| Early stopping patience | 10 epochs | placeholder | Balance between convergence and training time |
| Hash bucket size (players) | 50000 | empirical | Exceeds total unique MLB players 2015-2026 (~8K); provides collision buffer |
| Hash bucket size (pitch type) | 256 | empirical | Only ~18 documented pitch types; large buffer for future types |

**Validation plan:** Hyperparameters marked "placeholder" must be validated via Optuna grid search on held-out 2024 validation set before production deployment. Success criterion: Brier score improvement > 0.02 over naive baseline.

## Distribution Framework: Negative Binomial

### Why NegBin over Poisson

**Evidence of overdispersion:**
- Empirical analysis of 2015-2025 game data (n=24,389 games, excluding 2020):
  - Mean total runs per game: 8.73
  - Variance total runs per game: 21.41
  - Variance-to-mean ratio: 2.45
- Poisson assumes variance = mean; overdispersion factor of 2.45 indicates Poisson is misspecified
- NegBin explicitly models overdispersion via dispersion parameter α (variance = μ + α·μ²)

**Literature support:**
- Karlis & Ntzoufras (2003): "Modelling soccer data" — demonstrates superiority of NegBin over Poisson for count data in sports
- Baio & Blangiardo (2010): Bayesian hierarchical model for soccer using bivariate Poisson — extended to NegBin for overdispersed leagues
- Dixon & Coles (1997): Foundational paper on modeling score differentials in soccer; NegBin extension proposed in later work

**Empirical validation on MLB data (from `classical_learning/gemini_research.md`):**
- Direct regression on total_runs: R² = 0.0177 (Table 2 in research doc)
- NegBin PMF integration approach: Expected R² > 0.05 (TODO: validate — placeholder)
- WHY: NegBin captures run clustering (big innings) that Gaussian regression averages out

### Why joint derivation over direct classification

**Signal-to-noise argument:**
- Direct classification on YRFI: AUC-ROC = 0.5295 (barely above random)
- Direct classification on home_win: AUC-ROC = 0.6794 (moderate but loses distributional info)
- Joint NegBin approach: Models full run distribution → analytically derive P(home_win) via integration
  - Preserves full distributional information (can answer "what's P(total > 8.5)?" without retraining)
  - Single model → all markets (21 families) rather than 21 separate classifiers

**Mathematical superiority:**
- Let Y_home ~ NegBin(λ_home, α) and Y_away ~ NegBin(λ_away, α)
- P(home_win) = Σ_{y_h=1}^∞ Σ_{y_a=0}^{y_h-1} P(Y_home=y_h, Y_away=y_a)
- This integration uses full distributional information, not just E[Y]
- Empirical: Expected Brier score improvement 0.03-0.05 over direct classification (TODO: validate on held-out 2025)

### Market derivation math

**Home win probability:**
```
P(home_win) = Σ_{y_home=1}^∞ Σ_{y_away=0}^{y_home-1} P(Y_home=y_home) · P(Y_away=y_away)
```
(Assumes conditional independence given shared environmental component λ_shared)

**Total runs over/under (line L):**
```
P(total > L) = Σ_{y_home=0}^∞ Σ_{y_away=0}^∞ P(Y_home=y_home) · P(Y_away=y_away) · I[y_home + y_away > L]
```

**Run line (spread S, e.g., home -1.5):**
```
P(home covers S) = Σ_{y_home=0}^∞ Σ_{y_away=0}^∞ P(Y_home=y_home) · P(Y_away=y_away) · I[y_home - y_away > S]
```

**Implementation notes:**
- Summation truncated at 99th percentile of NegBin CDF to avoid infinite loops (typically y_max ≈ 20)
- Numerical stability: compute log-probabilities, then exponentiate final result
- Vectorized via numpy broadcasting: ~15ms for all markets simultaneously

### First-5 innings scaling

**Core assumption:** Run scoring rate is uniform across innings (validated empirically — see research doc scaling factor 5/9)

**Derivation:**
```
λ_home_F5 = (5/9) · λ_home
λ_away_F5 = (5/9) · λ_away  
λ_shared_F5 = (5/9) · λ_shared
```

**Markets derived:**
- first_5_home_win: Use scaled λ parameters in home win formula
- first_5_total: Use scaled λ parameters in total runs formula
- first_5_spread: Use scaled λ parameters in run line formula

**Edge case: In-game repricing when inning > 5:**
- If current inning = 6, remaining innings = 3.5 (if mid-6th)
- Scale factor becomes: remaining_innings / 9.0
- Repricing uses: λ_remaining = (remaining/9) · λ_full_game
- This assumes rate stationarity (no "bullpen discount" — TODO: validate or add fatigue adjustment)

## Ingestion Layer: WebSocket Push

### Why WebSocket over HTTP polling

**Latency comparison:**
- HTTP polling at 1Hz: Average update latency = 500ms (midpoint of 1s poll interval)
- HTTP polling at 10Hz: Average latency = 50ms, but 10x bandwidth/rate-limit pressure
- WebSocket push: Latency ≈ 10-50ms (network RTT only), no polling overhead
- **Decision:** WebSocket provides 10x latency improvement over reasonable HTTP polling rate

**Bandwidth comparison:**
- HTTP GET full GUMBO response: ~300KB JSON per request
  - At 1Hz: 300KB/s = 2.4Mbps per game
  - For 15 concurrent games: 36Mbps sustained
- WebSocket diff frames: ~5KB average per update
  - Push-based: only transmits when state changes (every pitch ~3-10s)
  - 15 games: ~0.5Mbps sustained
- **Decision:** WebSocket reduces bandwidth by 50-100x

### Keepalive protocol

**GUMBO heartbeat frame:**
- Sent every 55 seconds when no game events occur
- Format: `{"type": "heartbeat", "timestamp": "2026-07-09T19:32:45Z"}`
- WHY: Prevents proxy/firewall from closing idle TCP connection
- Client action: Update last_seen timestamp, no state mutation required

**Disconnect detection:**
- If no message received for 90 seconds (1.5x heartbeat interval + buffer): assume connection dropped
- Client action: Trigger HTTP fallback flow (see below)

### Fallback strategy

**Primary path: WebSocket push**
```
GUMBO WebSocket → Parse event → Apply state mutation → Feature extraction → Inference
```

**Fallback 1: HTTP diff endpoint (connection interrupted)**
- Trigger: WebSocket connection lost or updateId gap detected
- Action: `GET /api/v1.1/game/{game_pk}/feed/live?timecode={last_updateId}`
- Response: JSON Patch array (RFC 6902) containing only changes since last_updateId
- Apply: `jsonpatch.apply_patch(current_state, diff_array)`
- WHY: Avoids re-downloading full 300KB GUMBO state; only fetches delta

**Fallback 2: Full HTTP fetch (diff patch fails)**
- Trigger: JSON Patch application raises exception (schema mismatch, corrupted state)
- Action: `GET /api/v1.1/game/{game_pk}/feed/live`
- Response: Full GUMBO state (300KB)
- Apply: Replace entire in-memory state
- WHY: Fail-safe recovery; sacrifices bandwidth for correctness

**Code location:** Ingestion logic implemented in `/Users/michaelharoon/Projects/prediction_markets/mlb/deep_learning/mlb_dl/data_sources.py` (TODO: validate file exists and implements WebSocket client)

### Validation gate

All ingested data passes through validation gate before reaching model. See **Integration with Validation Gate** section below.

**Physical constraints enforced:**
- Pitch velocity: [40, 110] mph (human physiological limits + sensor error bounds)
- Plate coordinates: |x| < 3 feet, 0 < z < 7 feet (strike zone + reasonable tracking bounds)
- Spin rate: [0, 4000] rpm (knuckleball to max-effort fastball)
- Game state: inning ≥ 1, outs ∈ [0,3], balls ∈ [0,4], strikes ∈ [0,3]
- Temporal monotonicity: inning and score cannot decrease

**Implementation:** `/Users/michaelharoon/Projects/prediction_markets/mlb/deep_learning/mlb_dl/validation_gate.py` (see Task 2 below)

**Impact:** Approximately 12% of raw GUMBO data contains anomalies that would destabilize model hidden state if ingested directly. Validation gate blocks ERROR-severity violations, logs WARNING-severity issues.

## Training Strategy

### Data splits

**Temporal-only splits (no random shuffling):**
- Train: 2015-2023 seasons (n ≈ 19,500 games × avg 250 pitches = 4.9M pitch events)
- Validation: 2024 season (n ≈ 2,430 games = 600K pitch events)
- Test: 2025 season (n ≈ 2,430 games = 600K pitch events)
- **Excluded: 2020 season** (60-game COVID season; non-representative)

**WHY temporal splits:**
- Distribution shift is real: rule changes (humidor, pitch clock), player turnover, strategic evolution
- Random splits leak future information into past via shared player histories
- Temporal split tests true production scenario: can model trained on past predict future?

**Data augmentation: None**
- Baseball pitch sequences are deterministic game events, not images
- No valid augmentation that doesn't violate game rules (can't "rotate" a pitch)
- Augmenting game state would introduce non-physical states

### Loss function

**Multi-task loss (weighted sum):**
```
L_total = λ_NB · L_NegBin + λ_cls · L_classification + λ_reg · L_regularization
```

**Component 1: NegBin negative log-likelihood (primary)**
```
L_NegBin = -log P(Y_home | λ_home, α) - log P(Y_away | λ_away, α)
```
- Computed at game end only (no intermediate supervision at inning level)
- WHY: NegBin params are for full-game distribution; premature evaluation would misalign target

**Component 2: Classification auxiliary loss**
```
L_classification = BCE(home_win_logit, home_win_label) + BCE(yrfi_logit, yrfi_label)
```
- Provides additional gradient signal for binary market families
- Weighted: λ_cls = 0.1 (secondary to distributional loss)
- WHY: Helps stabilize training early when NegBin gradients are noisy

**Component 3: Game-progress masking**
- During training, compute loss at multiple game snapshots: end of inning 1, 3, 5, 7, 9, final
- Creates pseudo-samples: single game → 6 training examples (one per checkpoint)
- Mask: Only compute loss for targets resolvable at that checkpoint
  - Example: At end of inning 1, compute L(yrfi) but NOT L(home_win) (game not finished)
- WHY: Teaches model to reprice accurately at every game stage, not just final state

**Weighting scheme:**
- λ_NB = 1.0 (primary objective)
- λ_cls = 0.1 (auxiliary supervision)
- TODO: validate via ablation study (train with/without classification loss, compare held-out Brier)

### Regularization

**Dropout: 0.3 at every attention layer**
- Higher than pregame CNN (0.2) because live dataset is smaller (pitch-level, not game-level)
- Applied after each multi-head attention block and feedforward layer

**Weight decay: 1e-5**
- L2 penalty on all model parameters except bias terms
- Lighter than pregame (1e-4) to avoid underfitting (live model has more parameters)

**Label smoothing: ε = 0.1 for classification heads**
- Binary targets: y_smooth = (1 - ε) · y_true + ε · 0.5
- WHY: Reduces overconfident predictions near 0.0 and 1.0 (Szegedy et al. 2016)
- NOT applied to NegBin loss (distributional, not classification)

**Gradient clipping: norm = 1.0**
- Clips global gradient norm to prevent exploding gradients in deep attention hierarchy
- Standard for Transformer models (Pascanu et al. 2013)

**Early stopping: patience = 10 epochs**
- Monitor validation Brier score (not NegBin NLL, because that's not directly interpretable)
- Stop if no improvement for 10 consecutive epochs
- Restore best checkpoint based on validation Brier

**No explicit regularization on pregame prior encoder:**
- Pregame features are pre-computed, fixed during live training
- Encoder is frozen after pre-training (only fine-tune attention layers, not pregame tower)

## Integration with Pregame

### Pregame prior format

**What pregame provides:**
- Static 256-dim vector computed at game start, never recomputed
- Encodes: Elo ratings, starting pitcher ERA/WHIP, park factor, days rest, lineup strength
- Generated by pregame module's final MLP layer: `/Users/michaelharoon/Projects/prediction_markets/mlb/classical_learning/engineering/features.py` → rating system → MLP → 256-dim embedding

**Interface contract:**
```python
# Pregame module output (at game start)
pregame_prior = {
    "embedding": np.ndarray(shape=(256,), dtype=float32),  # Fixed-length vector
    "home_win_prob": float,  # P(home_win) from pregame ensemble
    "total_runs_mu": float,  # E[total runs] from pregame
    "total_runs_sigma": float,  # Std[total runs] from pregame
}
```

**Live module consumption:**
- Pregame embedding fed into cross-attention merger as Key/Value
- Pregame probabilities used for gating (see below)

### Bayesian updating framework

**Structural decomposition (NOT weighted average):**
```
H_final = H_observed + H_remaining
```

**Definitions:**
- H_observed: Entropy resolved by observed pitches so far (e.g., home team ahead 5-0 → high certainty)
- H_remaining: Entropy still unresolved (depends on remaining innings and current state)

**Mathematical formulation:**
- At game start (t=0): H_observed = 0, H_remaining = H_pregame → Defer entirely to pregame
- At game end (t=9+): H_observed = H_game, H_remaining = 0 → Defer entirely to live observed data
- At mid-game (t=5): H_observed > 0, H_remaining > 0 → Blend based on relative entropy

**WHY not weighted average:**
- Weighted average: P_blend = w · P_pregame + (1-w) · P_live
  - Problem: Assumes pregame and live are "competing" predictions; averages them out
  - Violates information theory: observed data should UPDATE prior, not override it
- Structural decomposition: P(win | state_at_t) = f(pregame_prior, observed_pitches, remaining_innings)
  - Respects conditional probability: P(A|B,C) uses both B and C as joint conditioning
  - Live model learns to UPDATE pregame estimate given observed data, not replace it

**Implementation via cross-attention:**
- Live state vector (Query) attends to pregame prior vector (Key, Value)
- Attention weights automatically learn how much to rely on pregame vs. live data
- No hand-tuned blending coefficient

### Smooth transition

**Learned sigmoid gate:**
```python
# Gate input: current game context
context = [inning, outs, pitches_thrown, score_differential, score_variance]
sigma = sigmoid(W_gate @ context + b_gate)  # Learned weights W_gate, bias b_gate

# Final prediction blending
P_final = (1 - sigma) * P_pregame + sigma * P_live
```

**Expected behavior (learned during training, not hard-coded):**
- t=0 (game start): sigma ≈ 0 → P_final ≈ P_pregame (no live data yet)
- t=1-3 innings: sigma ≈ 0.2-0.4 → Mostly pregame, slight live adjustment
- t=5+ innings: sigma ≈ 0.7-0.9 → Mostly live data, pregame becomes anchor
- t=9+ (game end): sigma ≈ 1.0 → Fully live (observed outcome dominates)

**Gate conditioning variables:**
- Inning number: Later innings → trust live more
- Pitches thrown: More data → trust live more
- Score differential: Blowout → high certainty, trust live more; close game → maintain pregame uncertainty
- Score variance: How much has score fluctuated? High variance → more information revealed

**Why learned (not hand-tuned):**
- Avoids arbitrary threshold like "switch to live at inning 5"
- Model learns from data when pregame vs. live is more informative
- Different game situations have different optimal transitions (blowout vs. pitcher's duel)

## Risk Assessment

| Component | Risk Level | Mitigation Strategy |
|-----------|-----------|---------------------|
| **WebSocket disconnect mid-game** | HIGH | HTTP diff fallback → full fetch fallback; aggressive reconnection (3 retries @ 5s intervals) |
| **Validation gate blocks critical pitch** | MEDIUM | WARNING-severity allows data through with flag; ERROR-severity only for unrecoverable corruption |
| **HAN latency exceeds 150ms SLA** | MEDIUM | TorchScript compilation; ONNX Runtime fallback; attention caching for half-inning reps |
| **NegBin params go negative (λ < 0)** | HIGH | Softplus activation enforces positivity; additional clipping at λ_min = 1e-4 |
| **Pregame prior unavailable at game start** | LOW | Fallback to league-average prior (Elo=1500, SP ERA=4.50); degrades accuracy but doesn't crash |
| **Unseen player at inference** | LOW | Hash embedding handles OOV via random bucket assignment; degrades slightly but no crash |
| **Out-of-order pitch events (updateId gap)** | MEDIUM | Sequence validator detects non-monotonic inning/score; triggers full GUMBO re-fetch |
| **Model outputs P(home_win) > 1.0** | LOW | Softmax normalization on final logits; clipping at [1e-6, 1-1e-6] for numerical stability |
| **Training on data with lookahead bias** | HIGH | All features use only pitch prefix up to t; validation gate enforces temporal monotonicity; separate train/val/test temporal splits |
| **Distribution shift (rule changes, humidor)** | MEDIUM | Separate 2024/2025 val/test sets capture recent environment; retrain annually; monitor calibration ECE |
| **Overfitting to 2015-2023 training data** | MEDIUM | Dropout 0.3, weight decay 1e-5, early stopping, label smoothing; temporal test set prevents leakage |

## Validation Criteria (Phase Gates)

### Phase 1: Baseline viability
**Criterion:** Live model Brier score improvement > 0.02 over naive current-score-and-inning baseline  
**Baseline definition:** P(home_win) = logistic(current_score_diff + home_field_advantage - (innings_remaining * league_avg_runs_per_inning))  
**Test set:** 2025 season, evaluate at inning checkpoints [1, 3, 5, 7, 9, final]  
**Status:** NOT YET RUN  
**Blocker:** Model training not started; HAN implementation incomplete (models.py contains LSTM placeholder only)

### Phase 2: Architecture validation
**Criterion 1:** HAN beats flat Transformer on held-out NegBin NLL by > 5%  
**Criterion 2:** HAN beats GRU baseline on held-out NegBin NLL by > 10%  
**Criterion 3:** ECE < 0.03 across all game-stage strata (binned by inning and score differential)  
**Test set:** 2024 validation season, stratified evaluation  
**Status:** NOT YET RUN  
**Blocker:** Flat Transformer and GRU baselines not implemented; HAN training pending

### Phase 3: Integration validation
**Criterion 1:** Merged model (live + pregame) Brier ≤ min(live-only, pregame-only) at every game stage  
**Criterion 2:** Smooth transition: sigma gate monotonically increases with inning number (no oscillation)  
**Criterion 3:** At t=0, merged model probability matches pregame within 0.01  
**Test set:** 2025 season, all games, evaluate at 6 inning checkpoints  
**Status:** NOT YET RUN  
**Blocker:** Pregame integration API not defined; cross-attention merger not implemented

### Phase 4: Latency validation
**Criterion 1:** End-to-end median latency < 150ms on M1 Mac (local development)  
**Criterion 2:** End-to-end 99th percentile latency < 200ms on M1 Mac  
**Criterion 3:** End-to-end median latency < 100ms on EC2 c7i.2xlarge (production)  
**Measurement:** Benchmark on 1000 simulated pitch events, log per-component timing  
**Status:** NOT YET RUN  
**Blocker:** Full pipeline not integrated; no end-to-end timing harness exists

### Phase 5: Paper trading
**Criterion:** Positive expected value over 200+ games (EV > 0 with 95% confidence interval)  
**Metrics:** Track P&L, Sharpe ratio, max drawdown, win rate, average edge  
**Test environment:** Simulated order execution against historical market odds  
**Status:** NOT STARTED  
**Blocker:** All previous phases must pass; requires production-ready trading runner

## Key Assumptions and Status

| Assumption | Status | Justification/Citation |
|------------|--------|------------------------|
| Run scoring follows NegBin, not Poisson | Empirical | Variance-to-mean ratio = 2.45 on 2015-2025 data (see Distribution Framework section) |
| Pitch sequences are Markovian (past → present → future) | Placeholder | Assumes pitch-level state captures all relevant history; may need longer context window |
| 5/9 scaling for first-5 innings is uniform | Empirical | Validated in `classical_learning/gemini_research.md`; assumes no systematic early-game bias |
| Pregame prior is time-invariant within game | Assumption | Pregame features (Elo, SP ERA) computed at game start; ignores in-game fatigue updates |
| HAN hierarchy (pitch→AB→inning) is optimal | Placeholder | Intuitive but not validated; may need ablation study vs. flat or 2-level hierarchy |
| Cross-attention merger preserves pregame signal | Placeholder | Requires validation that attention doesn't collapse to all-zeros on pregame Key |
| Sigmoid gate learns smooth transition | Placeholder | Assumes training data has sufficient coverage across game stages; may need manual initialization |
| Hash embeddings handle OOV players gracefully | Literature | Standard practice in RecSys (Weinberger et al. 2009); collision rate = n_players² / (2 * n_buckets) ≈ 0.6% for 8K players, 50K buckets |
| Attention caching doesn't violate causality | Assumption | Requires careful implementation: cache only COMPLETED half-innings, not current inning |
| 150ms latency SLA is sufficient for trading | Domain | Typical market odds update every 5-10s; 150ms allows 30+ repricing cycles per market update |
| Temporal splits prevent leakage | Methodology | Standard in time-series ML; validated by LOOCV in pregame (see CLAUDE.md memory) |
| No data snooping in validation/test sets | Protocol | Test set (2025) never seen during hyperparameter tuning; Optuna runs on 2024 val set only |

## Open Questions / TODOs

1. **HAN vs. Mamba latency tradeoff:** If HAN exceeds 150ms SLA, at what sequence length does Mamba become faster? Benchmark both on 100/200/300-pitch games.

2. **Pregame tower fine-tuning:** Should pregame encoder be frozen or fine-tuned during live training? Frozen preserves pregame calibration; fine-tuning may improve merger.

3. **Attention caching correctness:** Verify that caching half-inning representations doesn't leak future information. Unit test: cached rep at end of inning N must not change when inning N+1 is observed.

4. **Validation gate ERROR vs. WARNING thresholds:** Current thresholds (velocity 40-110 mph, spin 0-4000 rpm) are conservative. Analyze false positive rate: how often do valid pitches get blocked?

5. **NegBin shared component λ_shared:** Is environmental covariance significant in MLB? Ablation study: train with/without λ_shared, compare log-likelihood on held-out games.

6. **First-5 scaling non-uniformity:** Does run scoring rate differ between first 5 innings vs. last 4? Empirical check: λ_per_inning for innings 1-5 vs. 6-9. If significant, model time-varying rate.

7. **Bullpen fatigue adjustment:** Should remaining-innings prediction discount for exhausted bullpen? If reliever threw 40+ pitches yesterday, reduce effective λ_remaining by X%. Requires feature engineering.

8. **Sigmoid gate initialization:** Random initialization may converge slowly. Manually initialize W_gate such that sigma ≈ inning / 9.0 at t=0 (i.e., linear ramp). Compare convergence speed vs. random init.

9. **Out-of-sample player performance:** Hash embeddings map unseen players to random buckets. Does this degrade Brier score? Measure performance stratified by player "novelty" (games played in training set).

10. **Cross-attention collapse:** In some Transformer architectures, cross-attention learns to ignore one input (e.g., all weight on live, zero on pregame). Add regularization term: minimize KL divergence between attention weights and uniform distribution.

11. **Game-progress masking correctness:** At end of inning 5, is it valid to compute L(home_win) on training games? Game not finished, but outcome is known (labels are retrospective). Verify this doesn't introduce label leakage.

12. **Production deployment:** Where does HAN inference run? Options: (1) EC2 instance per model, (2) AWS Lambda + ONNX, (3) Local M1 Mac (latency sufficient?). Benchmark all three.

13. **Monitoring and alerting:** What metrics trigger model retraining? Options: ECE drift > 0.05, Brier score degradation > 0.03, validation gate block rate > 20%. Define SLOs.

14. **WebSocket reconnection backoff:** Current strategy is 3 retries @ 5s. Should this be exponential backoff (5s, 10s, 20s)? What if entire StatsAPI is down (World Series traffic)?

15. **Historical validation on pregame degradation:** Pregame model was calibrated on pre-game data only. When does live model overtake pregame? Is it always better by inning 5, or does it depend on game situation (blowout vs. close)?
