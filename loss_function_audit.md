# Loss Function Audit for All Models

## Confirmation: ✅ All models for the same target use the SAME loss function

### Key Finding
**All models across all families for a given target use identical loss functions during HPO and evaluation.**

---

## Loss Functions by Task Type

### Classification Targets
All classification models optimize: **`log_loss`** (binary cross-entropy)

**Location**: `classical_learning/strategy/optuna_objectives.py`, line 116
```python
if task == "classification":
    proba = model.predict_proba(X_va)[:, 1]
    score = log_loss(y_va, proba.clip(0.01, 0.99))
```

**Affected targets**: `home_win`, `yrfi`, etc.

**Affected model families** (18 total):
- lightgbm
- xgboost
- catboost
- random_forest
- extra_trees
- hist_gradient_boosting (loss="log_loss", line 164)
- adaboost
- logistic_regression
- ridge
- lasso
- elasticnet
- sgd (loss="log_loss", line 250)
- knn
- lda
- qda
- gaussian_nb
- mlp
- bagging_logreg
- ydf_oblique_gbt

---

### Regression Targets
All regression models optimize: **`mean_squared_error` (MSE)**

**Location**: `classical_learning/strategy/optuna_objectives.py`, line 119
```python
else:
    preds = model.predict(X_va)
    score = mean_squared_error(y_va, preds)
```

**Affected targets**: `total_runs`, `home_run_diff`, etc.

**Affected model families** (19 total):
- lightgbm
- xgboost
- catboost
- random_forest
- extra_trees
- hist_gradient_boosting (loss="squared_error", line 166)
- adaboost
- logistic_regression → Ridge fallback
- ridge
- lasso
- elasticnet
- sgd (loss="squared_error", line 252)
- knn
- lda → Ridge fallback (line 283)
- qda → Ridge fallback (line 296)
- gaussian_nb → Ridge fallback (line 307)
- mlp
- bagging_logreg
- ydf_oblique_gbt

---

## How to Verify

The loss function is determined **by task type, not by model family**:

1. **Task determination** (`classical_learning/strategy/train.py`, line 93):
   ```python
   task = "classification" if target in TARGETS_CLASSIFICATION else "regression"
   ```

2. **Unified objective function** (`classical_learning/strategy/optuna_objectives.py`, lines 62-134):
   - Single `create_objective()` function for all families
   - Receives `task` parameter
   - Applies task-specific loss on line 110-119
   - Used for all 19 model families

3. **All training paths converge**:
   - HPO: Uses `create_objective(family, task, ...)` → lines 116/119
   - LOYO evaluation: Uses same `compute_metrics()` → `classical_learning/strategy/evaluate.py`
   - Inference metrics: Uses same evaluation functions

---

## Invariant: Task Type Controls Loss Function

**Because:**
- All families receive the same `task` parameter from `train_target()`
- The loss function is determined solely by `task == "classification"`
- There is no per-family loss function override

**Therefore:**
- `home_win` (classification) + LightGBM = log_loss
- `home_win` (classification) + XGBoost = log_loss ✅
- `total_runs` (regression) + LightGBM = MSE
- `total_runs` (regression) + XGBoost = MSE ✅

---

## No Surprises Found

- ✅ No model family uses a different loss for the same target
- ✅ No hardcoded per-family loss function overrides
- ✅ All regression models converge on MSE (not MAE, Huber, or quantile loss)
- ✅ All classification models converge on log_loss (not Brier score or others)
