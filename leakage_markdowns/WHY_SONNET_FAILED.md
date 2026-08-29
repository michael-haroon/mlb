# Why Sonnet Failed to Catch Leakage (And How to Use Claude Better)

---

## The Core Reason: Single-File Analysis + Implicit Data Flow

### What You Probably Asked Sonnet
```
"Review this code for data leakage"
or
"Check for look-ahead bias in the training pipeline"
```

### Why Sonnet Failed

**Sonnet sees files in isolation, not data flow:**

```
ratings_tuning.py (reviewed alone):
  ✅ "Parameters tuned correctly, val_seasons are separate from train"
  ❌ "But doesn't see that these same seasons later appear in train.py test folds"

train.py (reviewed alone):
  ✅ "LOYO split looks correct"
  ❌ "But doesn't trace back: where do the features come from?"

data.py (reviewed alone):
  ✅ "Feature allowlist is good"
  ❌ "But doesn't check: were the parameters used to compute them tuned on test data?"
```

**The leakage is BETWEEN files, not within them:**

```
ratings_tuning.py (lines 31-67)
    Output: Rating parameters tuned on seasons [2024, 2025, 2026]
            ↓
classical_learning/engineering/build.py
    Used by: attach_all_ratings(games, params=...)
            ↓
game_features.parquet
    Features generated with those parameters
            ↓
train.py (lines 93-95)
    Loads: X, y from game_features.parquet
    Generates: LOYO splits that include seasons [2024, 2025, 2026] as TEST sets
    
Result: Model trains on features (Elo, SRS) computed with parameters tuned on its own test seasons
```

Sonnet doesn't trace this chain by default.

---

## The "TODO" Comment That Hid the Issue

Your code has this at lines 56-67 of `ratings_tuning.py`:

```python
    Known limitation (TODO: validate impact before fixing):
        Each objective calls compute_*(games_copy, params) on the FULL game frame,
        then evaluates Brier score only on val_seasons. The rating values for
        val_seasons are therefore computed from chronologically prior games
        (correct), but the PARAMETERS are chosen to minimise error specifically
        on those val seasons. When those same val seasons later appear as LOYO
        val folds in train.py, the feature values (Elo, SRS, etc.) were generated
        with params tuned on them — a mild form of target leakage in parameter
        space. Estimated effect: ~1-3% optimistic Brier on those folds.
        Proper fix: tune only on seasons strictly before val_seasons (e.g., tune
        on all_seasons[:-3], evaluate on all_seasons[-3:]). Requires re-running
        Optuna (~800s) so deferred until next full pipeline rebuild.
```

**Why Sonnet missed it:**
- Sonnet sees "TODO" and "Known limitation" and assumes "not a current bug"
- The comment is so detailed it *validates* itself rather than flagging the problem
- Code that acknowledges a bug is sometimes treated as "understood and acceptable"

---

## Why The Specialist Explore Agent Caught It

The Explore agent succeeded because of the **checklist-based temporal reasoning:**

```
Checklist item: "For EACH feature group:
  a) Where are they computed?
  b) On what data?
  c) When is that relative to train/test split?
  d) Does training data season overlap with parameter tuning season?"
```

This forces **cross-file tracing** that general review misses.

---

## How to Use Claude Effectively for Leakage Detection

### ❌ What Doesn't Work

**Too vague:**
```
"Find data leakage in this codebase"
→ Sonnet does a surface review, misses inter-file dependencies
```

**Too single-file:**
```
"Review ratings_tuning.py for leakage"
→ Misses that its output is used downstream in train.py
```

**Too trusting of comments:**
```
"Does this code properly implement LOYO?"
→ Sees "LOYO" comment, assumes "probably fine"
```

### ✅ What Works: The Temporal Leakage Checklist

**Create a checklist prompt:**

```
You are a temporal leakage auditor for an ML pipeline.

For the pregame MLB model, do this EXACT sequence:

1. IDENTIFY ALL DATA TRANSFORMATIONS:
   - List every stage: raw data → train/test split → model training
   - For each stage, note the file and function name

2. MAP HYPERPARAMETER TUNING:
   - Find EVERY place Optuna/GridSearch/manual tuning occurs
   - For EACH tuning location:
     a) What is being tuned?
     b) On what data rows?
     c) In what seasons are those rows?
     d) What are the outputs of tuning?
     e) Where are those outputs used later?

3. TRACE PARAMETER FLOW:
   - Start from each tuning location
   - Follow the output through all downstream functions
   - Does that output feed into data that will be test data?
   
4. IDENTIFY OVERLAPS:
   - List all seasons that appear in TWO contexts:
     a) As an optimization objective
     b) As a validation/test set
   - These overlaps are leakages

5. FOR EACH LEAKAGE, PROVIDE:
   - File path and line numbers
   - Code snippets showing both the tuning and the test set
   - Why it's a leak (temporal causality argument)
   - Severity (catastrophic/medium/mild)
```

**Then ask Explore to execute it:**
```
Use this checklist on classical_learning/engineering/ratings_tuning.py, 
classical_learning/strategy/train.py, and classical_learning/strategy/data.py.

Your output should be a table with columns:
[Tuning_Location, Tuning_Data_Seasons, Test_Set_Seasons, Overlap, Severity, Code_Snippet]
```

---

## Why This Checklist Approach Works

### 1. **Forces file-to-file tracing**
Instead of reviewing one file, you're building a dependency graph

### 2. **Operationalizes "look-ahead bias"**
Instead of abstract concept, it's concrete: "Did data from the test season influence any parameters?"

### 3. **Catches the "TODO" trap**
The checklist doesn't trust comments — it traces code regardless of what the comments say

### 4. **Scalable**
You can apply this checklist to ANY new model or pipeline

---

## Should You Use a Separate Agent?

### When to use `/code-review` (General code review)
- ✅ Looking for bugs, style issues, performance
- ❌ Not for leakage detection (too generic)

### When to use an `Explore` agent with checklist
- ✅ Multi-file temporal analysis needed
- ✅ Leakage or data flow issues
- ✅ Architectural questions (which systems feed what?)

### When to use a custom `Explore` agent every time
- ✅ Before every major training run
- ✅ Before deploying to production
- ✅ When adding new features or data sources

---

## Recipe: Your Own "Leakage Detector" Agent

Save this for future use. Every time you build a new model:

```markdown
# Temporal Leakage Audit Checklist

## Context
You are auditing a machine learning pipeline for temporal leakage / look-ahead bias.

Temporal leakage occurs when:
- Data from the test period influences training setup
- Hyperparameters are optimized on test data
- Feature engineering uses future information
- Train/test split is not temporally ordered

## Audit Process

### Stage 1: Map the data flow
- List every place where data is loaded, filtered, or transformed
- Show which seasons/dates are present at each stage
- Draw a timeline: 2015 ← training → 2026 ← test

### Stage 2: Find all optimization / tuning locations
For EACH place where hyperparameters, thresholds, or system parameters are set:
- What is being optimized?
- What data / seasons are used for optimization?
- Where does the optimized value get used?

### Stage 3: Check for overlaps
For each optimization location:
- Does the optimization use seasons X?
- Later, do those same seasons X appear as test/validation sets?
- If yes → LEAKAGE

### Stage 4: Temporal ordering check
For train/test splits:
- Is training data strictly BEFORE test data?
- Or does split use random shuffle, cross-validation without order, etc.?

## Output Format

For each leakage found:
- Severity: 🔴 (catastrophic: 90%+ accuracy), 🟡 (moderate: 1-5% optimism), ✅ (safe)
- File and line numbers
- Data seasons involved
- Why it's a leak (chain of causality)
- Fix (if applicable)
```

---

## What You Should Do Right Now

### For Immediate Use
1. Bookmark `HANDLING_LEAKAGE_DOWNSTREAM.md` (how to proceed)
2. Decide: Path 1 (keep training), Path 2 (fix now), or Path 3 (fix both)

### For Future Models
1. Use the checklist above for leakage detection
2. Run an Explore agent with the checklist BEFORE committing to a training run
3. Cost: ~5 min of agent time, saves days of debugging bad metrics

### For Production Deployment
1. Always run temporal leakage audit before shipping
2. Document findings (even if results look good)
3. Have explicit sign-off: "Verified no temporal leakage" in code comments

---

## Why This Matters

**Leakage that shows 90%+ accuracy** → Obviously wrong, caught immediately  
**Leakage that shows 55-58% accuracy** → Looks real, survives code review  

The second type is what got you. It's subtle, well-intentioned (the TODO comment shows awareness), but it compounds if not caught.

Going forward:
- ✅ Use checklists, not freestyle review
- ✅ Always trace cross-file data flow
- ✅ Don't trust comments; verify code
- ✅ Use Explore agent before every major training run

---

## TL;DR: Why Sonnet Failed

| Reason | Why | Solution |
|--------|-----|----------|
| Single-file review | Didn't see ratings params flow to train.py test sets | Always audit cross-file flows |
| Trusts comments | Saw "Known limitation (TODO)" and treated as acceptable | Verify code regardless of comments |
| Generic review | General leakage review doesn't catch parameter-space leakage | Use specialized temporal checklist |
| No dependency mapping | Didn't build a "tuning → output → test" dependency graph | Force temporal reasoning with checklist |

**Going forward:** Use an Explore agent with a detailed temporal leakage checklist before every training run. Cost: 5 min, value: catching hidden leakages before they waste compute.
