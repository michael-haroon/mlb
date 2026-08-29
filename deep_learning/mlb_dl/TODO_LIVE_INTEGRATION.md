# Live Inference Integration — TODO Checklist

**Status**: Infrastructure complete (2026-07-09); model training implemented and run (`train_unified.py` fit-unified, 2026-08; first sweep results void — contaminated population, retrained 2026-08-29)  
**Next**: Runner integration — nothing calls `register_game` in production yet

---

## ✅ Completed (2026-07-09)

- [x] Implement `LiveInferenceEngine` core class
- [x] Implement `PregamePrior` dataclass
- [x] Implement `PitchEvent` dataclass  
- [x] Implement `GameInferenceState` per-game tracking
- [x] Implement `TradingBridge` async/sync bridge
- [x] Write pitch encoding logic (40-dim feature vector)
- [x] Write batch construction with padding
- [x] Write player ID hashing (blake2b, 512 buckets)
- [x] Write hierarchy tracking (inning/AB/pitch)
- [x] Write unit test suite (20+ tests, 609 lines)
- [x] Write standalone example (`example_inference.py`)
- [x] Write integration guide (`runner_integration_guide.md`)
- [x] Write comprehensive README (`INFERENCE_ENGINE_README.md`)
- [x] Syntax validation (all files compile)

---

## 🔥 Critical Path (Required for Production)

### 1. Model Training

**File**: `deep_learning/mlb_dl/train.py`

**Tasks**:
- [ ] Generate training data from parquet
  - [ ] Load PITCHES table (2015-2025)
  - [ ] Build pitch sequences (350-pitch windows)
  - [ ] Extract game outcomes (home_win, yrfi, total_runs, etc.)
  - [ ] Split into train/val/test (LOYO by season)
- [ ] Implement training loop
  - [ ] Loss: BCE for classification + Gaussian NLL for regression
  - [ ] Optimizer: AdamW with weight decay
  - [ ] LR schedule: Cosine annealing
  - [ ] Early stopping on val Brier score
- [ ] Validate latency on CPU
  - [ ] Forward pass must be <30ms on CPU
  - [ ] Profile hotspots if latency exceeds budget
- [ ] Save best checkpoint to `checkpoints/best_model.pt`

**Estimated Time**: 2-3 days (data prep 1 day, training 1-2 days)

### 2. Ingestion Layer

**File**: `deep_learning/mlb_dl/ingestion.py`

**Tasks**:
- [ ] Implement MLB GUMBO WebSocket client
  - [ ] Connect to `wss://ws.statsapi.mlb.com/api/v1/game/{game_pk}/feed/live`
  - [ ] Parse play-by-play messages into `PitchEvent` objects
  - [ ] Handle reconnection with exponential backoff
  - [ ] Log all pitch events to disk for replay/debugging
- [ ] Implement pitch event filtering
  - [ ] Skip non-pitches (pickoffs, mound visits, etc.)
  - [ ] Skip pitches with missing critical data
- [ ] Implement multi-game subscription
  - [ ] Subscribe to 15 games simultaneously
  - [ ] Manage WebSocket connections per game
  - [ ] Graceful shutdown on game final
- [ ] Integration with `LiveInferenceEngine`
  - [ ] Call `engine.on_pitch_event(game_pk, pitch)` on each pitch
  - [ ] Handle errors without crashing the feed

**Estimated Time**: 2-3 days (WebSocket 1 day, parsing 1 day, testing 1 day)

### 3. Runner Integration

**File**: `classical_learning/trading/runner.py`

**Tasks**:
- [ ] Add `_live_bridge` field to `TradingRunner.__init__`
- [ ] Start bridge in `TradingRunner.start()`
  - [ ] Load model checkpoint from `deep_learning/mlb_dl/checkpoints/best_model.pt`
  - [ ] Initialize `TradingBridge(model_path, device="cpu")`
  - [ ] Call `bridge.start()`
- [ ] Hook into `_handle_game_start(ticker)`
  - [ ] Implement `_resolve_game_pk(ticker)` (ticker → game_pk lookup)
  - [ ] Implement `_build_pregame_prior(game_pk)` (wire to ensemble store)
  - [ ] Call `bridge.register_game(game_pk, pregame_prior)`
- [ ] Add live repricing to main loop
  - [ ] Query `bridge.get_live_prices(game_pk)` for active games
  - [ ] Implement `_handle_live_reprice(game_pk, prices)` decision logic
  - [ ] Only reprice EXIT positions initially (not HOLD)
- [ ] Add cleanup to `_handle_settlement(ticker)`
  - [ ] Call `bridge.engine.unregister_game(game_pk)` on settlement

**Estimated Time**: 1-2 days (integration 0.5 day, testing 1 day)

---

## 🚀 Deployment (Dry-Run → Production)

### 4. Dry-Run Testing (EC2)

**Environment**: EC2 instance with conda env `pred`

**Tasks**:
- [ ] Deploy code to EC2
  - [ ] Push to git, pull on EC2
  - [ ] Ensure model checkpoint is synced
- [ ] Run in dry-run mode
  - [ ] `conda run -n pred python -m pregame.trading.runner --dry-run`
  - [ ] Subscribe to 2-3 live games
  - [ ] Monitor logs for errors
- [ ] Validate metrics
  - [ ] Inference latency stays <60ms
  - [ ] No memory leaks (check RSS over 4-hour session)
  - [ ] Live prices are sensible (no hallucinations)
- [ ] Compare live vs pregame prices
  - [ ] Log both to `logs/live_vs_pregame.csv`
  - [ ] Analyze price divergence (should be <0.3 typically)

**Duration**: 1 week (3-4 live game sessions)

### 5. Production Rollout (Cautious)

**Phase 5a: EXIT positions only**
- [ ] Enable live repricing for EXIT positions
  - [ ] If live price allows profitable exit, take it
  - [ ] Log all exit decisions + P&L attribution
- [ ] Monitor for 1 week
  - [ ] Track P&L from live exits vs holding to settlement
  - [ ] Validate no excessive churn (max 3 reprices per position)

**Phase 5b: HOLD positions (if 5a profitable)**
- [ ] Enable live repricing for HOLD positions
  - [ ] Rare: only if live model is very confident AND disagrees with pregame
  - [ ] Require |live_prob - pregame_prob| > 0.2 threshold
- [ ] Monitor for 1 week

**Phase 5c: Live entry (VERY RISKY, optional)**
- [ ] Consider entering new positions based on live prices
  - [ ] Only if live edge > 10 cents (much higher than pregame threshold)
  - [ ] Requires tight risk controls (position sizing, stop-loss)
- [ ] Monitor for 2 weeks before scaling

---

## 🎯 Enhancements (Post-Launch)

### 6. Market Derivation (integrate `distributions.py`)

**File**: `deep_learning/mlb_dl/distributions.py` (already exists)

**Tasks**:
- [ ] Import market derivation functions
  - [ ] `derive_totals_market(mu, sigma, line)`
  - [ ] `derive_spread_market(mu_diff, sigma_diff, line)`
  - [ ] `derive_team_total_market(mu, sigma, line)`
- [ ] Update `_run_inference()` to return full market family
  - [ ] Currently: only classification probs + raw regression params
  - [ ] Target: all 21 market families (like pregame module)
- [ ] Validate derived markets vs pregame
  - [ ] Should be close at game start, diverge during play

**Estimated Time**: 1 day

### 7. Selective Repricing Logic

**File**: `deep_learning/mlb_dl/inference_engine.py`

**Tasks**:
- [ ] Add repricing trigger conditions to `on_pitch_event()`
  - [ ] Skip inference if no critical state change
  - [ ] Critical changes: score, 2-out RISP, 3-2 count, pitcher change
- [ ] Add leverage calculation
  - [ ] Use Win Probability Added (WPA) as proxy
  - [ ] Only reprice if leverage > 1.5 (late & close)
- [ ] Add blowout detection
  - [ ] Skip inference if |score_diff| > 5 and inning > 6

**Estimated Time**: 0.5 day

### 8. Confidence Bounds

**File**: `deep_learning/mlb_dl/inference_engine.py`

**Tasks**:
- [ ] Add sanity checks to `_run_inference()`
  - [ ] Require |live_prob - pregame_prob| < 0.3 (prevent hallucination)
  - [ ] If violated, log warning and return pregame price
- [ ] Add minimum pitch count threshold
  - [ ] Require ≥50 pitches before trusting live model
  - [ ] Before that, return pregame prices unchanged
- [ ] Add confidence tier from pregame prior
  - [ ] If pregame confidence is LOW, widen bounds to ±0.4
  - [ ] If pregame confidence is HIGH, tighten bounds to ±0.2

**Estimated Time**: 0.5 day

### 9. Multi-Game Batching

**File**: `deep_learning/mlb_dl/inference_engine.py`

**Tasks**:
- [ ] Refactor `_run_inference()` to accept list of game_pks
  - [ ] Build batch with shape (num_games, 350, 40)
  - [ ] Single forward pass for all games
  - [ ] 5-10x throughput improvement
- [ ] Update `TradingBridge` to batch inference calls
  - [ ] Accumulate pitch events for 100ms
  - [ ] Batch inference for all games with pending updates
- [ ] Benchmark latency
  - [ ] Target: <100ms for 15 games (6-7ms per game)

**Estimated Time**: 1 day

---

## 🔬 Research (Long-Term)

### 10. HAN Architecture Upgrade

**File**: `deep_learning/mlb_dl/models.py`

**Tasks**:
- [ ] Implement Hierarchical Attention Network
  - [ ] Pitch-level attention (within AB)
  - [ ] AB-level attention (within inning)
  - [ ] Inning-level attention (full game)
- [ ] Add positional encoding for hierarchy
  - [ ] Use inning/AB/pitch indices from `GameInferenceState`
- [ ] Train on 2015-2025 data
- [ ] Validate improvement over LSTM baseline
  - [ ] Target: 2-3% Brier improvement

**Estimated Time**: 1-2 weeks (implementation 3 days, training 2 days, validation 2 days)

### 11. Pregame-Live Gate Learning

**File**: `deep_learning/mlb_dl/models.py`

**Tasks**:
- [ ] Add learned gate to `LiveGameModel`
  - [ ] `gate = sigmoid(w_pitch_count * pitch_count + w_score * score_diff + b)`
  - [ ] Blend: `final = gate * live_pred + (1 - gate) * pregame_pred`
- [ ] Add gate loss term
  - [ ] Penalize gate for deviating from optimal blend (measured on validation set)
- [ ] Visualize gate behavior
  - [ ] Plot gate value over pitch count for different game states

**Estimated Time**: 3-4 days (implementation 1 day, training 1 day, analysis 1 day)

### 12. Real-Time Feature Store

**File**: `deep_learning/mlb_dl/feature_cache.py` (new)

**Tasks**:
- [ ] Implement in-memory feature cache
  - [ ] Per-player rolling stats (last 10 games)
  - [ ] Per-matchup stats (batter vs pitcher history)
  - [ ] Per-park stats (last 30 days)
- [ ] Update cache on each pitch
  - [ ] Increment batter's AB count, hits, etc.
  - [ ] Update pitcher's pitch count, strikes, etc.
- [ ] Integrate with `_encode_pitch()`
  - [ ] Add 10-20 rolling features to the 40-dim vector
- [ ] Benchmark impact on Brier score
  - [ ] Target: 1-2% improvement

**Estimated Time**: 1 week (implementation 2 days, integration 1 day, validation 2 days)

---

## 📊 Monitoring & Metrics

### 13. Live Inference Dashboard

**File**: `deep_learning/mlb_dl/dashboard.py` (new, optional)

**Tasks**:
- [ ] Build Streamlit dashboard
  - [ ] Real-time pitch feed for active games
  - [ ] Live vs pregame price comparison
  - [ ] Inference latency histogram
  - [ ] Memory usage graph
- [ ] Deploy to EC2
  - [ ] Accessible via SSH tunnel or public IP

**Estimated Time**: 2 days

### 14. P&L Attribution

**File**: `classical_learning/trading/analytics/pnl_attribution.py`

**Tasks**:
- [ ] Track P&L by source
  - [ ] Pregame alpha (entry at pregame prices)
  - [ ] Live alpha (repricing based on live model)
  - [ ] Settlement luck (random noise)
- [ ] Compute Sharpe by source
  - [ ] Is live repricing actually profitable?
  - [ ] Or is it just churn that bleeds edge?
- [ ] Log to CSV for analysis

**Estimated Time**: 1 day

---

## 🐛 Known Issues / Technical Debt

- [ ] `_resolve_game_pk(ticker)` not yet implemented (TODO in integration guide)
- [ ] `_build_pregame_prior(game_pk)` not yet wired to ensemble store (TODO)
- [ ] Market derivation only returns classification + raw regression (TODO: integrate distributions.py)
- [ ] Feature set is placeholder (40 dims, marked as `# TODO: validate`)
- [ ] Max sequence length (350) is placeholder (marked as `# TODO: validate`)
- [ ] Standardization uses placeholder mean/std until model is trained

---

## 📝 Documentation

- [x] `INFERENCE_ENGINE_README.md` — Overview + architecture
- [x] `runner_integration_guide.md` — Step-by-step integration
- [x] `example_inference.py` — Standalone example
- [x] `tests/test_inference_engine.py` — Unit test suite
- [ ] `INGESTION_LAYER_SPEC.md` — MLB GUMBO API spec (future)
- [ ] `MODEL_TRAINING_GUIDE.md` — Training data pipeline (future)
- [ ] `LIVE_REPRICING_DECISION_LOGIC.md` — When/how to reprice (future)

---

## 🎓 Learning Resources

**MLB GUMBO API**:
- WebSocket endpoint: `wss://ws.statsapi.mlb.com/api/v1/game/{game_pk}/feed/live`
- REST fallback: `https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live`
- Documentation: https://github.com/toddrob99/MLB-StatsAPI

**Hierarchical Attention Networks**:
- Original paper: Yang et al. (2016) — "Hierarchical Attention Networks for Document Classification"
- Applied to sports: Shah et al. (2020) — "Predicting Shot Outcomes in Basketball with Hierarchical Models"

**Real-Time Inference**:
- Latency budgets: https://blog.acolyer.org/2019/10/07/serving-dnns-in-the-cloud/
- CPU vs GPU for inference: https://blog.deci.ai/cpu-inference/

---

## 📞 Contacts / Questions

**For questions on**:
- Model architecture → Check `deep_learning/mlb_dl/models.py` + `INFERENCE_ENGINE_README.md`
- Integration with runner → Check `runner_integration_guide.md`
- Testing → Check `tests/test_inference_engine.py`
- Performance → Check latency benchmarks in README

**Blockers**:
- Model training requires historical PITCHES parquet (already in S3)
- Ingestion layer requires MLB GUMBO WebSocket access (publicly available)
- Runner integration requires understanding of `TradingRunner` lifecycle (documented in runner.py)

---

**Last Updated**: 2026-07-09  
**Status**: Infrastructure complete, ready for model training + ingestion layer + runner integration
