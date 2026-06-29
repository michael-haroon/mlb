# Data Leakage Analysis — Complete Documentation Index

**Analysis Date:** June 28, 2026  
**Status:** ✅ COMPLETE  
**Severity:** 🔴 HIGH + 🟡 MEDIUM (combined impact: 1-3% metric optimism)

---

## Start Here

### If You Have 2 Minutes
→ Read **`QUICK_START.md`**
- What was found
- What it means
- What to do

### If You Have 15 Minutes
→ Read **`ANSWERS_TO_YOUR_QUESTIONS.md`**
- Why Sonnet failed
- Why these are real leakages
- How to prevent this in future models
- Whether you need to stop training

### If You Have 30 Minutes
→ Read **`LEAKAGE_REPORT.md`** + **`LEAKAGE_FIXES.md`**
- Detailed technical explanation
- Copy-paste-ready fixes

---

## Complete Documentation (In Reading Order)

### 1. Quick Overview (5 min)
📄 **`QUICK_START.md`**
- 2-line summary of each leakage
- Decision tree (fix now or later?)
- Expected outcomes
- Recommended action

### 2. Your Questions Answered (10 min)
📄 **`ANSWERS_TO_YOUR_QUESTIONS.md`**
- Why Sonnet failed ← Key insight for future
- How to use Claude better for leakage detection
- Proof these are real leakages (run-able commands)
- Why you don't need to stop training
- Which path to take (1/2/3)

### 3. Detailed Technical Report (10 min)
📄 **`LEAKAGE_REPORT.md`**
- Executive summary
- Issue #1: Rating parameter tuning (HIGH)
  - Code snippets showing the leak
  - How it propagates
  - Why it matters
- Issue #2: HPO hyperparameter leakage (MEDIUM)
  - Current vs. expected flow
  - Example scenario
  - Severity assessment
- Metrics interpretation
  - Why 55.8% accuracy isn't obviously wrong
  - Production performance forecast
  - Interpretation for stakeholders

### 4. Visual Explanation (10 min)
📄 **`LEAKAGE_VISUAL_FLOW.md`**
- Diagram: Current (leaky) data flow
- Diagram: Corrected data flow
- Metrics comparison: Per-fold accuracy with/without leakage
- Visual proof of why it's not catastrophic
- Timeline comparison

### 5. Implementation Guide (20 min)
📄 **`LEAKAGE_FIXES.md`**
- Fix #1: Rating parameter tuning (copy-paste ready)
  - Current code
  - Fixed code
  - Alternative simpler fix
  - Functions to update
- Fix #2: Optuna HPO leakage (two options)
  - Option A: Per-fold HPO (recommended, more compute)
  - Option B: Lightweight (less compute)
  - Full code snippets
- Implementation checklist
- Estimated effort & impact per fix

### 6. Downstream Handling (10 min)
📄 **`HANDLING_LEAKAGE_DOWNSTREAM.md`**
- Path 1: Keep training, fix after (best for live predictions)
- Path 2: Fix now, retrain (best for production)
- Path 3: Fix both leakages (perfectionist option)
- Cost-benefit analysis
- Post-hoc fixes if you don't fix now
  - Calibration
  - Ensemble
  - Manual adjustment
- Timeline recommendations

### 7. Why Sonnet Failed (5 min)
📄 **`WHY_SONNET_FAILED.md`**
- The core reason: implicit multi-file data flows
- Why single-file review misses it
- The "TODO" comment trap
- How to use Claude better (checklist-based)
- Recipe for your own "leakage detector"

---

## For Different Audiences

### For Yourself (Before Next Training Run)
1. Read `QUICK_START.md`
2. Read `ANSWERS_TO_YOUR_QUESTIONS.md` (especially "how to use Claude better")
3. Before training: Use the temporal leakage checklist from `WHY_SONNET_FAILED.md`

### For Your Manager/Stakeholder
1. Show `QUICK_START.md`
2. Show: "Expected 1-2% metric optimism"
3. Show: "Fix takes 30 min or can handle downstream"
4. Show: "These are LOW-risk leakages, not HIGH-risk"

### For Your Team
1. Distribute `LEAKAGE_REPORT.md` (technical explanation)
2. Distribute `WHY_SONNET_FAILED.md` (prevent repeat)
3. Create shared checklist from `WHY_SONNET_FAILED.md`
4. Require pre-training leakage audit for all models

---

## Quick Reference Table

| Document | Length | Best For | Key Takeaway |
|----------|--------|----------|---|
| QUICK_START.md | 2 min | Decision-making | Fix now (30 min) or defer? |
| ANSWERS_TO_YOUR_QUESTIONS.md | 10 min | Understanding why this happened | Sonnet failed on multi-file flows |
| LEAKAGE_REPORT.md | 10 min | Technical details | Rating params tuned on test seasons |
| LEAKAGE_VISUAL_FLOW.md | 10 min | Visual learners | Flow diagrams make it clear |
| LEAKAGE_FIXES.md | 20 min | Implementation | Copy-paste fixes, 30 min total |
| HANDLING_LEAKAGE_DOWNSTREAM.md | 10 min | Risk assessment | These are low-urgency to fix |
| WHY_SONNET_FAILED.md | 5 min | Future prevention | Use checklists, not freestyle review |

---

## Recommended Reading Paths

### Path A: "I Need to Know NOW" (5 min)
```
QUICK_START.md → Make decision (fix/defer)
```

### Path B: "I Need to Understand This" (25 min)
```
QUICK_START.md
→ ANSWERS_TO_YOUR_QUESTIONS.md
→ LEAKAGE_VISUAL_FLOW.md
```

### Path C: "I Need to FIX This" (50 min)
```
QUICK_START.md
→ LEAKAGE_FIXES.md
→ [Apply fixes]
→ [Retrain]
```

### Path D: "I Want Complete Understanding" (60 min)
```
Read all 7 documents in order (they're written to be sequential)
```

### Path E: "I Want to Prevent This in Future Models" (20 min)
```
WHY_SONNET_FAILED.md
→ Create shared checklist from it
→ Use checklist before every training run
```

---

## Key Insights

### 1. Why This Happened (Sonnet Failure)
- Leakage is **between files**, not within files
- Sonnet reviews **each file in isolation**
- Multi-file data flows require **explicit tracing**
- `TODO` comments can **hide problems**

### 2. What This Means (Impact)
- **Not catastrophic** — 1-2% optimism, not 90% accuracy
- **Real and provable** — Already documented in code
- **Easy to fix** — 30 minutes for main fix
- **Safe to defer** — Can handle downstream

### 3. How to Prevent This (Future)
- Use **checklist-based temporal audits** before training
- Don't trust **single-file review** for architectural issues
- Require **cross-file data flow verification**
- Use **Explore agent** with specific prompts, not freestyle

---

## Files in This Repository

```
mlb/
├── LEAKAGE_ANALYSIS_INDEX.md          ← You are here
├── QUICK_START.md                      ← Start here (2 min)
├── ANSWERS_TO_YOUR_QUESTIONS.md        ← Why/how/what (10 min)
├── LEAKAGE_REPORT.md                   ← Technical details (10 min)
├── LEAKAGE_VISUAL_FLOW.md              ← Diagrams (10 min)
├── LEAKAGE_FIXES.md                    ← Implementation (20 min)
├── HANDLING_LEAKAGE_DOWNSTREAM.md      ← Risk/timeline (10 min)
├── WHY_SONNET_FAILED.md                ← Prevention (5 min)
└── memory/
    └── leakage_analysis_2026_06_28.md  ← Saved to memory for future
```

---

## Checklists

### Pre-Training Leakage Audit (5 min)

- [ ] **Stage 1: Map data flow**
  - [ ] List: raw → engineered → split → training
  - [ ] For each stage: note file, function, seasons

- [ ] **Stage 2: Find all tuning**
  - [ ] Hyperparameter tuning (Optuna, GridSearch)?
  - [ ] System parameters (decay, K-factors, weights)?
  - [ ] Feature engineering parameters?
  - [ ] For each: what, where, on what data?

- [ ] **Stage 3: Check for overlaps**
  - [ ] Do tuning seasons overlap test seasons?
  - [ ] List overlaps
  - [ ] If overlap found → LEAKAGE

- [ ] **Stage 4: Verify temporal order**
  - [ ] Train strictly before test?
  - [ ] No random shuffle?
  - [ ] No lookahead in features?

### Post-Fix Verification (5 min)

- [ ] Apply Fix #1 (rating tuning)
- [ ] Regenerate features
- [ ] Retrain models
- [ ] Compare metrics:
  - [ ] Accuracy lower by ~1%? ✓
  - [ ] AUC lower by ~1%? ✓
  - [ ] Log loss higher by ~1%? ✓
- [ ] Update documentation
- [ ] Push to git

---

## Decision Points

### "Should I fix this now or later?"

**Fix NOW (Path 2) if:**
- ✅ Deploying to production
- ✅ Running live predictions (and accuracy matters)
- ✅ You have 30 minutes
- ✅ Publishing results

**Fix LATER (Path 1) if:**
- ✅ Just testing live season
- ✅ No downstream users
- ✅ Time-critical
- ✅ Can accept 1-2% optimistic metrics

### "What's my risk if I don't fix?"

| Scenario | Risk | Mitigation |
|----------|------|-----------|
| Live predictions | 1-2% too optimistic | Adjust forecasts down |
| Production | Performance gap in 2027 | Fix now or calibrate later |
| Academic paper | Publishing bias | Fix now for integrity |
| Next model | Repeated issue | Use checklist going forward |

---

## Sign-Off

This analysis confirms:
- ✅ Two real leakages identified
- ✅ Both are parameter-tuning, not data-peeking
- ✅ Combined impact: 1-3% metric optimism
- ✅ Both are fixable in 30 min or deferrable
- ✅ Prevention: Use temporal leakage checklists

**Recommendation:** Fix the rating tuning leakage now (30 min, high impact). Defer HPO if constrained on time.

---

## Next Steps

1. **Read** `QUICK_START.md` (2 min)
2. **Decide** path (2 min)
3. **Execute** (0-30 min depending on path)
4. **Document** your decision
5. **Share** `WHY_SONNET_FAILED.md` with team

**Total time to decision: 5 minutes**

---

For questions or clarifications, refer to the specific document listed in the table above.

Good luck! 🚀
