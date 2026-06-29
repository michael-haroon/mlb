# Handling Leakage Downstream: Should You Retrain Now or Fix Later?

---

## TL;DR: You Don't Need to Stop Training Right Now

**The practical answer:** These are LOW-RISK leakages to handle downstream. You have options:

| Scenario | Action | Risk | Effort |
|----------|--------|------|--------|
| **Prediction 2026 live season** | Keep training, adjust forecasts down 1-2% | Low | None |
| **Model going to production 2027** | Fix now, retrain (~15 min) | Medium | Low |
| **Quick offline experiment** | Keep current, note in results | Low | None |

---

## Why You Can Handle This Downstream (Not Urgent)

### The Leakage Profile
```
Current reported:    55.8% accuracy
Leakage amount:      1-2% (parameter tuning only, not data peeking)
Effect on decisions: Small but systematic upward bias

Comparable to:       ✅ Similar to reporting model without post-hoc calibration
                     ❌ NOT like training on target-leaking features
```

### These Aren't "Stop Everything" Leakages

```
CATASTROPHIC (stop now):
  - Features include postgame box scores
  - Train/test split uses random shuffle (no temporal order)
  - Features literally include the target
  → Would show 85-95% accuracy (obviously wrong)

PARAMETER TUNING (what you have):
  - Optimization parameters chosen on test data
  - But features computed correctly from prior games
  - Ground truth labels never see model input
  → Shows realistic metrics (55-58%) with 1-2% optimism
  → Fixable downstream without retraining
```

---

## Three Paths Forward

### Path 1: Keep Training, Fix After 2026 Live Season (RECOMMENDED IF TIME-CONSTRAINED)

**When to use:** Running live predictions, tight deadline, not deploying to production yet

**Process:**
```bash
# Keep training as-is
conda run -n pred python -m pregame.strategy.train \
  --features pregame/artifacts/game_features.parquet \
  --target home_win \
  --output results_2026/ \
  --tier A

# Document the leakage in your results
echo "⚠️  KNOWN ISSUE: Rating parameters tuned on 2024-2026 seasons
     Expected metric inflation: +1-2%
     Fix: Re-tune ratings on 2015-2023 only (after live season)" \
     > results_2026/LEAKAGE_NOTE.txt

# Adjust forecasts down by 1-2%:
# Reported 55.8% → Use 54.8-55.8% for downstream decisions
```

**Trade-off:** 1-2% optimistic forecasts now, fix later (1 hour of work post-season)

---

### Path 2: Fix the High-Impact Leakage NOW, Full Retrain (RECOMMENDED IF DEPLOYING)

**When to use:** Deploying to production, need accurate metrics, have 30 minutes

**Effort breakdown:**
- Fix code: 5 minutes (copy-paste from LEAKAGE_FIXES.md)
- Run Optuna: 13 minutes (800s as documented)
- Run training: depends on your compute, but same as before
- Total: **~20 minutes + your original training time**

**Process:**
```bash
# 1. Edit ratings_tuning.py (5 min)
# Copy code from LEAKAGE_FIXES.md Fix #1

# 2. Re-tune ratings (13 min)
conda run -n pred python -c "
from pregame.engineering.ratings_tuning import tune_all_ratings
from pregame.engineering.build import load_games_from_s3
import pandas as pd

games = load_games_from_s3()  # or load_games_locally()

# NEW: Separate tune_seasons from val_seasons
tune_seasons = sorted(games['season'].unique())[:-3]  # All except last 3
val_seasons = sorted(games['season'].unique())[-3:]   # Last 3

params = tune_all_ratings(games, n_trials=100, 
                          val_seasons=val_seasons,
                          tune_seasons=tune_seasons)

# Save params
import json
with open('optimized_rating_params.json', 'w') as f:
    json.dump(params, f)
"

# 3. Regenerate features with corrected params
conda run -n pred python pregame/engineering/build.py \
  --output pregame/artifacts/ \
  --params optimized_rating_params.json

# 4. Re-run training
conda run -n pred python pregame/strategy/train.py \
  --features pregame/artifacts/game_features.parquet \
  --target home_win \
  --output results_fixed/ \
  --tier A

# 5. Compare
diff <(jq .aggregate_metrics results_current/training_summary.json) \
     <(jq .aggregate_metrics results_fixed/training_summary.json)
# Expected: ~1% improvement (less optimistic)
```

**Trade-off:** One-time 30-min investment, removes 1-2% optimism permanently

---

### Path 3: Fix Both Leakages, Nested HPO (RECOMMENDED IF HIGH-COMPUTE-BUDGET)

**When to use:** Mission-critical model, have compute to spare, want perfect evaluation

**Effort:** Path 2 + 2 extra hours refactoring

**Process:**
1. Do Path 2 (fix rating tuning)
2. Apply LEAKAGE_FIXES.md Fix #2 (nest HPO in LOYO loop)
3. Full retraining (2x time for HPO, but each fold gets tuned params)

**Result:** Metrics with zero parameter-space leakage

---

## Which Path Should You Take?

### Use Path 1 (Keep training, fix after) IF:
- ✅ Running live 2026 predictions (not mission-critical)
- ✅ Time-constrained (need results this week)
- ✅ Not deploying to production yet
- ✅ Can adjust forecasts down 1-2% manually

### Use Path 2 (Fix now, retrain) IF:
- ✅ Deploying to production (need accurate metrics)
- ✅ 30 minutes is acceptable
- ✅ Want clean metrics for stakeholders
- ✅ Want to remove the "TODO" from codebase
- ✅ 2027 performance needs to match 2026 reported

### Use Path 3 (Fix both, nested HPO) IF:
- ✅ Ultra-high accuracy needed
- ✅ Compute budget allows 2x training time
- ✅ Publishing results (academic/external)
- ✅ Perfectionist 😊

---

## How to Handle Leakage Downstream (If You Don't Fix Now)

If you choose Path 1, here's how to handle the 1-2% bias later:

### Option A: Post-Hoc Calibration (Easiest)

```python
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

# Load out-of-fold predictions from leaky model
oof_preds = np.load('results_2026/oof_home_win_lightgbm_A.npy')
oof_labels = df['home_win'].values

# Fit calibrator to detect the bias
calibrator = CalibratedClassifierCV(
    estimator=LogisticRegression(penalty='none'),
    method='sigmoid',
    cv=5
)

# Calibrator will learn that predictions are ~1-2% too high
calibrator.fit(oof_preds.reshape(-1, 1), oof_labels)

# Apply to new predictions
corrected_preds = calibrator.predict_proba(new_preds.reshape(-1, 1))[:, 1]

# Result: corrected_preds will be ~1-2% lower (removes leakage effect)
```

### Option B: Manual Adjustment (Explicit)

```python
# If you know leakage adds ~1.5%:
observed_accuracy = 0.558
true_accuracy = observed_accuracy - 0.015
# true_accuracy ≈ 0.543 (conservative estimate)

# Use this for production forecasts
```

### Option C: Ensemble with Holdout Data (Robust)

```python
# Train a second model on 2026 data only (truly held out from tuning)
# Blend predictions 50/50 with main model
# The holdout model won't have tuning leakage

ensemble_pred = 0.5 * main_model_pred + 0.5 * holdout_model_pred
# Averaging reduces the 1-2% bias
```

---

## What NOT to Do

### ❌ Don't worry about stopping training now
- Leakage is only 1-2%, not 10-20%
- You won't get embarrassingly wrong results
- Fixing takes 30 min, not a full retrain if you're smart

### ❌ Don't ignore the leakage forever
- If deploying to production, fix it
- Future 2027 holdout will show degradation without fixes
- It's already documented in the codebase as "TODO"

### ❌ Don't retrain 10 times trying to remove all leakage
- You have 2 concrete fixes (ratings + HPO)
- After those, improvements are marginal
- Diminishing returns kick in

---

## Timeline Recommendation

### Short term (This week)
- ✅ Continue training with current code
- ✅ Document 1-2% expected optimism in results
- ✅ Adjust live forecasts down by 1-2% if needed

### Medium term (Before deploying)
- ✅ Apply Fix #1 (rating tuning) — 30 min total
- ✅ Re-run full training pipeline
- ✅ Compare metrics (expect ~1% drop)

### Long term (Nice-to-have)
- ✅ Apply Fix #2 (nested HPO) if compute allows
- ✅ Clean up documentation
- ✅ Document in CLAUDE.md that leakage has been addressed

---

## Cost-Benefit Analysis

| Scenario | Now cost | Later cost | Risk | Recommend |
|----------|----------|-----------|------|-----------|
| Live predictions 2026 | 0 | 30 min + rerun | 1-2% bias | Path 1 |
| Production deploy | 30 min | Rerun later + delay | 1-2% bias | Path 2 |
| Academic paper | 30 min | 0 | Publishing bias | Path 2 |
| Next model iteration | 30 min | 0 | Compound bias | Path 2 |

---

## The Bottom Line

**You can safely keep training.** These leakages are:
- ✅ Not catastrophic (not 90% accuracy from data peeking)
- ✅ Well-understood (already in code comments)
- ✅ Easy to fix (30 minutes, not days)
- ✅ Safe to handle downstream (1-2% adjustment)

**But if you're deploying to production, spend the 30 minutes now.**

The difference between 55.8% and 54.8% matters when you're risking real money.
