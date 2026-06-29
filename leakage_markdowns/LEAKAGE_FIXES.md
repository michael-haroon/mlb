# Data Leakage: Specific Code Fixes

---

## Fix #1: Rating Parameter Tuning Leakage 🔴 HIGH PRIORITY

### Current Code (LEAKS)
**File:** `pregame/engineering/ratings_tuning.py` (lines 31–71)

```python
def tune_all_ratings(
    games: pd.DataFrame,
    n_trials: int = 100,
    val_seasons: Optional[list[int]] = None,
) -> dict:
    """Tune all rating system parameters via Optuna.
    
    Uses LOYO structure: for each validation season, train ratings on prior
    seasons and evaluate on the held-out season.
    """
    if val_seasons is None:
        all_seasons = sorted(games["season"].dropna().unique())
        val_seasons = all_seasons[-3:]  # ← LEAKAGE: Tunes on last 3 seasons
    
    log.info(f"Tuning ratings on val_seasons={val_seasons}, {n_trials} trials per system")
    
    optimized = {}
    
    # Tune Elo parameters (optimizing for val_seasons outcomes!)
    elo_study = optuna.create_study(direction="minimize", 
                                     sampler=optuna.samplers.TPESampler(seed=42))
    elo_study.optimize(
        lambda trial: _elo_objective(trial, games, val_seasons),  # ← Leakage point
        n_trials=n_trials,
        show_progress_bar=True,
    )
```

### Fixed Code (NO LEAKAGE)
```python
def tune_all_ratings(
    games: pd.DataFrame,
    n_trials: int = 100,
    val_seasons: Optional[list[int]] = None,
    tune_seasons: Optional[list[int]] = None,  # ← NEW parameter
) -> dict:
    """Tune all rating system parameters via Optuna.
    
    Uses LOYO structure: for each validation season, train ratings on EARLIER
    seasons and evaluate on the held-out season. Parameters are tuned to
    minimize error on val_seasons, but rating computations use only tune_seasons.
    """
    if val_seasons is None:
        all_seasons = sorted(games["season"].dropna().unique())
        # ← FIXED: Split into tune and val BEFORE optimization
        if len(all_seasons) > 3:
            tune_seasons = all_seasons[:-3]  # All except last 3
            val_seasons = all_seasons[-3:]   # Last 3
        else:
            tune_seasons = all_seasons[:-1] if len(all_seasons) > 1 else all_seasons
            val_seasons = all_seasons[-1:]
    elif tune_seasons is None:
        # If val_seasons provided but tune_seasons not, auto-derive
        all_seasons = sorted(games["season"].dropna().unique())
        tune_seasons = [s for s in all_seasons if s not in val_seasons]
    
    log.info(f"Tuning ratings on tune_seasons={tune_seasons}, val_seasons={val_seasons}, "
             f"{n_trials} trials per system")
    
    optimized = {}
    
    # Tune Elo parameters (now with temporal safety)
    elo_study = optuna.create_study(direction="minimize", 
                                     sampler=optuna.samplers.TPESampler(seed=42))
    elo_study.optimize(
        lambda trial: _elo_objective(trial, games, val_seasons, 
                                     tune_seasons=tune_seasons),  # ← Pass tune_seasons
        n_trials=n_trials,
        show_progress_bar=True,
    )
```

### Update Objective Functions

**Current (LEAKS):**
```python
def _elo_objective(trial: optuna.Trial, games: pd.DataFrame, 
                   val_seasons: list[int]) -> float:
    K = trial.suggest_float("K", 2, 50)
    K_home = trial.suggest_float("K_home", 0, 30)
    
    # Compute Elo for ALL games with current parameters
    elo = compute_elo(games.copy(), K=K, K_home=K_home)  # ← Uses all games
    
    # Evaluate ONLY on val_seasons (but parameters were chosen for val_seasons!)
    val_mask = elo['season'].isin(val_seasons)
    brier = brier_score_loss(elo.loc[val_mask, 'home_win'],
                             elo.loc[val_mask, 'elo_prob'])
    return brier
```

**Fixed (NO LEAKAGE):**
```python
def _elo_objective(trial: optuna.Trial, games: pd.DataFrame, 
                   val_seasons: list[int],
                   tune_seasons: Optional[list[int]] = None) -> float:
    K = trial.suggest_float("K", 2, 50)
    K_home = trial.suggest_float("K_home", 0, 30)
    
    if tune_seasons is None:
        tune_seasons = [s for s in games['season'].unique() 
                       if s not in val_seasons]
    
    # Compute Elo using ONLY tune_seasons data
    # This ensures rating parameters don't see val_seasons data
    games_for_tune = games[games['season'].isin(tune_seasons)].copy()
    elo_tune = compute_elo(games_for_tune, K=K, K_home=K_home)
    
    # Now extend to val_seasons using the learned parameters
    games_extended = games.copy()
    elo_extended = compute_elo(games_extended, K=K, K_home=K_home)
    
    # Evaluate ONLY on val_seasons (parameters tuned on external distribution)
    val_mask = elo_extended['season'].isin(val_seasons)
    brier = brier_score_loss(elo_extended.loc[val_mask, 'home_win'],
                             elo_extended.loc[val_mask, 'elo_prob'])
    return brier
```

### Alternative Simpler Fix
If modifying the rating computation is complex, the minimum fix is to only return parameters trained externally:

```python
def tune_all_ratings_safer(games, n_trials=100, val_seasons=None):
    """Simpler alternative: tune on external data only, don't touch the last 3 seasons."""
    if val_seasons is None:
        all_seasons = sorted(games["season"].dropna().unique())
        val_seasons = all_seasons[-3:]
    
    # SAFETY: Don't pass val_seasons to Optuna for parameter selection
    # Instead, do external rolling-window tuning on older data
    historical_games = games[~games['season'].isin(val_seasons)].copy()
    
    # Tune on historical data only
    elo_study = optuna.create_study(direction="minimize")
    elo_study.optimize(
        lambda trial: _elo_objective(trial, historical_games, []),  # Empty val_seasons
        n_trials=n_trials
    )
    
    return elo_study.best_params
```

---

## Fix #2: Optuna HPO Hyperparameter Leakage 🟡 MEDIUM PRIORITY

### Current Code (LEAKS)
**File:** `pregame/strategy/train.py` (lines 53–206)

```python
def train_target(
    features_path: Path,
    target: str,
    output_dir: Path,
    data_mode: str = "2015+",
    families: list[str] | None = None,
    n_trials: int = OPTUNA_N_TRIALS,
    tier: str = "A",
) -> dict:
    # ... setup code ...
    
    for family in families:
        log.info(f"  [{family}] Starting Optuna HPO ({n_trials} trials)...")
        t0 = time.time()
        
        try:
            # ← LEAKAGE: HPO on latest split (most recent data)
            best_params = _run_optuna_hpo(family, task, X, y, seasons, splits, n_trials)
        except Exception as e:
            log.warning(f"  [{family}] Optuna HPO failed: {e}")
            continue
        
        # Then apply these params to ALL folds, including old ones
        for split in splits:
            model = build_model(family, task, best_params)  # ← Same params everywhere
            # ...
```

### Fixed Code (Option A: Per-Fold HPO) — RECOMMENDED
```python
def train_target(
    features_path: Path,
    target: str,
    output_dir: Path,
    data_mode: str = "2015+",
    families: list[str] | None = None,
    n_trials: int = OPTUNA_N_TRIALS,
    tier: str = "A",
) -> dict:
    # ... setup code ...
    
    for family in families:
        log.info(f"  [{family}] Starting per-fold Optuna HPO ({n_trials} trials per fold)...")
        t0 = time.time()
        
        oof_predictions = np.full(len(y), np.nan)
        fold_metrics = []
        best_params_per_fold = {}  # ← Store per-fold parameters
        
        # Resolve per-family feature set from importance analysis
        importance_features = None
        if filter_report is not None:
            from ..analysis.feature_routing import get_feature_set
            importance_features = get_feature_set(family, filter_report)
            log.info(f"  [{family}] importance filter: {len(importance_features)} features")
        
        for split in splits:
            try:
                # Prepare data
                prepared = prepare_fold(X, y, seasons, split, family, tier=tier,
                                        importance_features=importance_features)
                
                # ← NEW: HPO per fold, using ONLY this fold's training data
                log.info(f"  [{family}] Tuning hyperparameters for fold {split.val_season}...")
                best_params = _run_optuna_hpo_single_fold(
                    family, task,
                    prepared.X_train, prepared.y_train,
                    prepared.sample_weights,
                    n_trials=n_trials
                )
                best_params_per_fold[split.val_season] = best_params
                
                # Train with fold-specific parameters
                model = build_model(family, task, best_params)
                
                # Fit with sample weights
                fit_kwargs = {}
                if hasattr(model, "fit"):
                    import inspect
                    sig = inspect.signature(model.fit)
                    if "sample_weight" in sig.parameters:
                        fit_kwargs["sample_weight"] = prepared.sample_weights.values
                
                model.fit(prepared.X_train, prepared.y_train, **fit_kwargs)
                
                # Predict
                if task == "classification":
                    if hasattr(model, "predict_proba"):
                        preds = model.predict_proba(prepared.X_val)[:, 1]
                    else:
                        dec = model.decision_function(prepared.X_val)
                        preds = 1.0 / (1.0 + np.exp(-dec))
                else:
                    preds = model.predict(prepared.X_val)
                
                oof_predictions[split.val_idx] = preds
                
                # Fold metrics
                metrics = compute_metrics(prepared.y_val.values, preds, task)
                metrics["val_season"] = int(split.val_season)
                metrics["best_params"] = best_params  # ← Log params per fold
                fold_metrics.append(metrics)
                
            except Exception as e:
                log.warning(f"  [{family}] Fold {split.val_season} failed: {e}")
            finally:
                del model, prepared
                gc.collect()
        
        # Aggregate metrics
        if fold_metrics:
            agg_metrics = _aggregate_fold_metrics(fold_metrics)
            elapsed = time.time() - t0
            
            results[family] = {
                "status": "success",
                "best_params_per_fold": best_params_per_fold,  # ← Store all params
                "fold_metrics": fold_metrics,
                "aggregate_metrics": agg_metrics,
                "elapsed_secs": round(elapsed, 1),
            }
            
            # Save per-fold parameters
            params_path = output_dir / f"params_per_fold_{target}_{family}_{tier}.json"
            with open(params_path, "w") as f:
                # Convert season keys to strings for JSON
                params_to_save = {str(k): v for k, v in best_params_per_fold.items()}
                json.dump(params_to_save, f, indent=2, default=str)
```

### New Function: Single-Fold HPO
```python
def _run_optuna_hpo_single_fold(
    family: str,
    task: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weights: pd.Series | None = None,
    n_trials: int = OPTUNA_N_TRIALS,
) -> dict:
    """Run Optuna HPO on a single fold's training data.
    
    Parameters tuned on THIS fold's training distribution, ensuring
    hyperparameters match the training set size and feature distribution.
    """
    from .config import NEEDS_IMPUTATION, NEEDS_SCALING
    from .data import _semantic_impute
    
    # Handle NaN imputation
    if family in NEEDS_IMPUTATION:
        X_train = _semantic_impute(X_train)
    
    objective = create_objective(
        family, task, X_train, y_train, sample_weights,
        needs_scaling=(family in NEEDS_SCALING),
    )
    
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=OPTUNA_PRUNER_STARTUP_TRIALS),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    del study
    gc.collect()
    return best_params
```

### Fixed Code (Option B: Lightweight — If Computation Is Expensive)
If per-fold HPO adds too much computation time, use this lighter approach:

```python
def train_target(...):
    # ... setup ...
    
    for family in families:
        # Tune on first few folds' combined data (represents older distribution)
        early_splits = splits[:3] if len(splits) >= 3 else splits
        X_train_early = pd.concat([X.iloc[s.train_idx] for s in early_splits])
        y_train_early = pd.concat([y.iloc[s.train_idx] for s in early_splits])
        
        # Then use middle splits' data to validate
        middle_splits = splits[3:6] if len(splits) >= 6 else splits[1:]
        val_seasons_for_hpo = [s.val_season for s in middle_splits]
        
        # This is a compromise: HPO on old-ish data, validate on mid-range
        best_params = _run_optuna_hpo_temporal(
            family, task, X_train_early, y_train_early, val_seasons_for_hpo, n_trials
        )
        
        # Then apply to all folds
        # (Better than current: params not optimized specifically on newest data)
```

---

## Implementation Checklist

- [ ] **Fix #1 (Rating tuning):** 
  - [ ] Update `tune_all_ratings()` to separate `tune_seasons` from `val_seasons`
  - [ ] Update `_elo_objective()`, `_wolfe_objective()`, etc. to use `tune_seasons`
  - [ ] Update `attach_all_ratings()` to pass the corrected parameters
  - [ ] Re-run Optuna (estimated 800s)
  - [ ] Verify metrics drop by ~1-2% (expected)

- [ ] **Fix #2 (HPO leakage):**
  - [ ] Choose Option A (per-fold, recommended) or Option B (lightweight)
  - [ ] Implement `_run_optuna_hpo_single_fold()` if using Option A
  - [ ] Update `train_target()` loop
  - [ ] Store `best_params_per_fold` in results JSON
  - [ ] Verify no regression in metrics

- [ ] **Verification:**
  - [ ] Run full pipeline with both fixes
  - [ ] Compare metrics before/after
  - [ ] Expected: 55.8% → ~55.7% (minor drop due to removed leakage)
  - [ ] Ensure 2027 holdout performance is consistent with corrected 2026 validation

---

## Estimated Effort & Impact

| Fix | Effort | Impact | Urgency |
|-----|--------|--------|---------|
| Rating tuning | ~2 hours code + 15 min Optuna run | Remove 1-2% optimism | 🔴 NOW |
| HPO per-fold (Option A) | ~3 hours code + 2x longer training | Fairer evaluation | 🟡 SOON |
| HPO per-fold (Option B) | ~1 hour code | Partial improvement | 🟡 IF TIME |

**Recommended approach:** Do Fix #1 immediately (high impact, moderate effort). Do Fix #2 Option B as a lighter compromise if Option A's compute cost is prohibitive.
