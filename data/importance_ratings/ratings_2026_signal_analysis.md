# Rating Feature Importance: 2026 Signal Strength Analysis

**Date**: 2026-08-25  
**Pipeline**: 7 tests x 2 CV modes x 10 targets = 140 instances + ONC clustering + PCA cross-check + Empirical Bayes adaptive shrinkage  
**Features**: 59 rating/ranking features (massey, elo, colley, wolfe, log5, pythag, SRS, consensus, + interactions where at least one component is a rating system)  
**Focus**: Fold 7 (2026) signal strength, NOT cross-fold means

## Fold Identity

- **Expanding CV**: 8 LOYO folds, fold 7 = test on 2026 (train: 2015-2025)
- **Sliding_3 CV**: 8 LOYO folds, fold 7 = test on 2026 (train: 2023-2025)
- Sliding_3 fold 7 is closest to production (trains on recent data only)
- Sign pattern convention: `+++++++-` means positive folds 0-6, negative fold 7. Last character = 2026.

## Key Methodology Notes

- **MDA (permutation importance)**: Shuffles feature, measures accuracy drop. Underestimates individual importance under multicollinearity.
- **DESUB (substitution MDA)**: Replaces feature with its mean. Robust to multicollinearity — correctly attributes importance even when correlated features absorb permutation variance.
- **MDA-DESUB divergence**: When DESUB is positive but MDA is zero/negative, the feature carries unique information that other correlated ratings can substitute for. The information is redundant, not absent.

---

## Target-by-Target Results

### HOME_WIN

**Fold 7 (2026)**: Sliding_3: 30 positive / 28 negative | Expanding: 24 positive / 34 negative | BOTH positive: 20/59

#### Still Alive in 2026 (sorted by sliding_3 fold 7 MDA)

| Feature | SL3_F7 | Trend | Last 3/3 |
|---------|--------|-------|----------|
| diff_all_roll20_whip_x_log5_prob_season | +2.54e-4 | RISING | 3/3 |
| away_elo_centered_x_home_team_woba_vs_rhp_roll200pa | +1.72e-4 | RISING | 3/3 |
| pythag_1st_diff | +1.62e-4 | RISING | 3/3 |
| diff_massey_inn5 | +1.55e-4 | RISING | 2/3 |
| diff_massey_inn4 | +1.51e-4 | RISING | 3/3 |
| pythag_2nd_diff | +1.15e-4 | RISING | 3/3 |
| away_pythag_2nd | +1.05e-4 | RISING | 3/3 |
| diff_colley_inn4 | +1.03e-4 | RISING | 2/3 |
| wolfe_diff | +8.26e-5 | RISING | 3/3 |
| consensus_home_win_prob | +7.64e-5 | FALLING | 3/3 |
| diff_massey_inn9_x_home_team_woba | +7.06e-5 | FALLING | 3/3 |
| diff_colley_inn6 | +5.83e-5 | RISING | 2/3 |
| diff_massey_inn6_x_home_elo_centered | +2.94e-5 | FALLING | 3/3 |
| diff_massey_full_x_elo_diff | +2.83e-5 | FALLING | 3/3 |
| log5_prob_medium | +1.44e-5 | RISING | 2/3 |

#### Dead in 2026 (historically dominant, now negative)

| Feature | SL3_F7 | Pattern | Interpretation |
|---------|--------|---------|----------------|
| elo_prob | -3.09e-5 | +++++++-  | 7 years positive, crashed 2026 |
| elo_diff | -6.46e-7 | ++++-++. | Flat-lined to zero |
| elo_prob_x_same_league | -5.16e-5 | +++++++-  | Same pattern as elo_prob |
| home_elo | -1.04e-4 | ++-+--+- | Actively harmful |
| home_pythag_2nd | -2.81e-5 | ++-+-++- | Was positive fold 6, dropped |

---

### HOME_RUN_DIFF

**Fold 7 (2026)**: Sliding_3: 12 positive / 46 negative | Expanding: 7 positive / 52 negative | BOTH positive: 7/59

**Catastrophic collapse.** Features that were 7/8 or 8/8 positive historically are deeply negative in 2026.

#### Survivors (only 7-12 features)

| Feature | SL3_F7 | Trend | Last 3/3 |
|---------|--------|-------|----------|
| pythag_1st_diff | +2.82e-4 | RISING | 3/3 |
| away_pythag_1st | +1.17e-4 | RISING | 3/3 |
| home_pythag_1st | +1.14e-4 | STABLE+ | 3/3 |
| pythag_2nd_diff_x_consensus_prob | +9.93e-5 | RISING | 3/3 |
| diff_colley_inn3 | +9.06e-5 | RISING | 2/3 |
| away_elo_centered_x_home_team_woba | +7.37e-5 | FALLING | 3/3 |
| pythag_2nd_diff | +6.92e-5 | RISING | 2/3 |
| away_pythag_2nd | +6.90e-5 | RISING | 3/3 |

#### Collapsed features (examples)

| Feature | SL3_F7 | Historical Pattern |
|---------|--------|--------------------|
| elo_prob | -3.37e-4 | +++++++-  (positive 7 folds, crashed) |
| elo_diff | -2.36e-4 | +++++++-  |
| consensus_prob | -5.46e-4 | +++++-+- |
| diff_massey_full | -2.08e-4 | +++++++-  |
| elo_prob_x_same_league | -2.03e-4 | +++++++-  |

---

### YRFI

**Fold 7 (2026)**: Sliding_3: 23 positive / 36 negative | Expanding: 23 positive / 35 negative | BOTH positive: 17/59

Sparse, inconsistent signal. No feature has a clean 8/8 pattern. Best candidates:

| Feature | SL3_F7 | Last 3/3 | Notes |
|---------|--------|----------|-------|
| diff_colley_inn1 | +1.14e-4 | 3/3 | Most consistent |
| elo_diff | +1.09e-4 | 2/3 | Positive here but negative for HRD |
| home_pythag_2nd | +9.47e-5 | 3/3 | |
| elo_prob | +8.55e-5 | 3/3 | Positive here but harmful elsewhere |
| diff_massey_inn1 | +5.78e-5 | 3/3 | |

Ratings add marginal signal for YRFI at best.

---

### TOTAL_RUNS

**Fold 7 (2026)**: Sliding_3: 30 positive / 29 negative | Expanding: 41 positive / 18 negative | BOTH positive: 26/59

Expanding/sliding_3 disagreement is wide (41 vs 30) — historical signal inflates expanding.

| Feature | SL3_F7 | Last 3/3 | Notes |
|---------|--------|----------|-------|
| home_pythag_1st | +3.98e-4 | 3/3 | Strongest |
| home_pythag_2nd | +3.34e-4 | 2/3 | |
| diff_massey_inn9_x_home_team_woba | +2.57e-4 | 2/3 | |
| diff_colley_inn9 | +2.28e-4 | 2/3 | |
| home_elo | +2.26e-4 | 1/3 | Unreliable spike |
| diff_massey_full_x_elo_diff | +2.25e-4 | 1/3 | |

Home-team LEVEL features (not differentials) drive total runs — scoring environment matters more than relative advantage.

---

### FIRST_5_TOTAL_RUNS

**Fold 7 (2026)**: Sliding_3: 28 positive / 30 negative | Expanding: 44 positive / 15 negative | BOTH positive: 26/59

Interaction terms dominate:

| Feature | SL3_F7 | Last 3/3 |
|---------|--------|----------|
| diff_massey_inn9_x_home_team_woba | +4.29e-4 | 2/3 |
| diff_massey_full_x_elo_diff | +4.21e-4 | 2/3 |
| diff_massey_inn1 | +1.87e-4 | 3/3 |
| away_elo_centered_x_home_team_woba | +1.85e-4 | 3/3 |
| elo_sum | +1.54e-4 | 3/3 |
| wolfe_sum | +1.31e-4 | 3/3 |

Notable: elo_sum and wolfe_sum are positive here (combined team quality predicts early-inning totals) while universally tier-F elsewhere.

---

### EXTRA_INNINGS

**Fold 7 (2026)**: Sliding_3: 18 positive / 41 negative | Expanding: 12 positive / 47 negative | BOTH positive: 9/59

**Ratings are actively destructive.** The most harmful features in the entire study:

| Feature | SL3_F7 | Pattern | Magnitude |
|---------|--------|---------|-----------|
| diff_massey_inn9_x_home_team_woba | -4.11e-3 | +------- | Extreme damage |
| away_elo_centered_x_home_team_woba | -2.38e-3 | +------- | Extreme damage |
| elo_prob | -1.13e-3 | -------- | ALL 8 folds negative |
| log5_prob_medium | -1.07e-3 | ++++---- | |
| diff_all_roll20_whip_x_log5_prob | -2.97e-4 | -------- | ALL 8 folds negative |
| elo_diff | -5.43e-5 | -------- | ALL 8 folds negative |

Survivors (pythag + structural only):

| Feature | SL3_F7 | Last 3/3 |
|---------|--------|----------|
| pythag_2nd_diff | +1.02e-3 | 3/3 |
| pythag_2nd_diff_x_consensus_prob | +7.04e-4 | 3/3 |
| consensus_home_win_prob | +4.05e-4 | 3/3 |
| home_pythag_1st | +4.01e-4 | 2/3 |
| away_srs | +3.59e-4 | 3/3 |
| pythag_1st_sum | +1.59e-4 | 3/3 (pattern: ++++++++) |
| pythag_1st_diff | +8.32e-5 | 3/3 |

Extra innings are about competitive balance, not team quality. Elo misleads by encouraging splits on strength differential.

---

## Cross-Target Summary (Sliding_3 Fold 7)

### Universally Alive (positive in 5-6/6 targets)

| Feature | Targets Positive | Targets Negative |
|---------|-----------------|-----------------|
| home_pythag_1st | 6/6 | 0/6 |
| pythag_1st_diff | 5/6 | 1/6 |
| elo_prob_same_div_above57 | 5/6 | 1/6 |
| diff_all_roll20_whip_x_log5_prob_season | 5/6 | 1/6 |

### Universally Dead (negative in 6/6 targets)

| Feature | Targets Positive | Targets Negative |
|---------|-----------------|-----------------|
| wolfe_prob | 0/6 | 6/6 |
| diff_massey_inn6 | 0/6 | 6/6 |
| diff_massey_inn8 | 0/6 | 6/6 |
| diff_massey_full | 0/6 | 6/6 |

---

## Structural Findings

### 1. Pure Elo Collapse (2026)

The `+++++++-` pattern across elo_prob, elo_diff, elo_prob_x_same_league for HOME_WIN and HOME_RUN_DIFF is the dominant finding. Seven consecutive years of positive signal followed by collapse in 2026.

Possible causes:
- Market efficiency: books already price Elo perfectly, no residual edge
- Parity increase: league more balanced in 2026
- Roster volatility: mid-season trades make season-long Elo lag behind reality
- Training artifact: expanding CV trains on more data, model may rely less on any single feature

### 2. Pythag Differentials are RISING

`pythag_1st_diff` and `pythag_2nd_diff` are the most reliably rising features across targets. They encode run-scoring environment (pythagorean expectation from RS/RA) rather than just who won.

### 3. Interaction Terms (Rating x Batting) Hold Better Than Pure Ratings

`away_elo_centered_x_home_team_woba` and `diff_all_roll20_whip_x_log5_prob` are RISING for home_win. They combine team quality with recent offensive performance, making them more adaptive than naked Elo.

### 4. DESUB-MDA Divergence Reveals Redundancy

Features like elo_diff have DESUB_SL3_F7=+8.14e-4 (positive) but MDA_SL3_F7=-6.46e-7 (zero). The information is redundant with other correlated ratings, not absent. Under multicollinearity, only one member of a correlated group gets MDA credit.

### 5. Target-Specific Signal is Non-Transferable

- Features strong for HOME_WIN (interactions, massey differentials) are HARMFUL for EXTRA_INNINGS
- Features strong for TOTAL_RUNS (home-team levels) don't help HOME_WIN (which needs differentials)
- YRFI has essentially no reliable rating signal

---

## Actionable Feature Selection (2026-Focused)

### HOME_WIN (keep ~15)
KEEP: pythag_1st_diff, pythag_2nd_diff, diff_all_roll20_whip_x_log5_prob_season, away_elo_centered_x_home_team_woba, diff_massey_inn4, diff_massey_inn5, away_pythag_2nd, diff_colley_inn4, wolfe_diff, diff_massey_inn9_x_home_team_woba, diff_massey_full_x_elo_diff, log5_prob_medium, consensus_home_win_prob, diff_massey_inn6_x_home_elo_centered, diff_colley_inn6

DROP: All pure Elo (elo_prob, elo_diff, elo_prob_x_same_league, home_elo, away_elo), all sums (elo_sum, srs_sum, wolfe_sum, pythag_1st_sum, pythag_2nd_sum), wolfe_prob, home_wolfe, away_wolfe

### HOME_RUN_DIFF (keep ~8)
KEEP: pythag_1st_diff, away_pythag_1st, home_pythag_1st, pythag_2nd_diff_x_consensus_prob, away_elo_centered_x_home_team_woba, pythag_2nd_diff, away_pythag_2nd, diff_colley_inn3

DROP: Everything else (52/59 features harmful in 2026)

### YRFI (keep ~5)
KEEP: diff_colley_inn1, home_pythag_2nd, elo_prob, elo_diff, diff_massey_inn1

DROP: 54/59 features

### TOTAL_RUNS (keep ~10)
KEEP: home_pythag_1st, home_pythag_2nd, diff_massey_inn9_x_home_team_woba, diff_colley_inn9, diff_massey_full_x_elo_diff, home_srs, wolfe_sum, log5_prob_season, away_elo_centered_x_home_team_woba, diff_all_roll20_whip_x_log5_prob_season

### FIRST_5_TOTAL_RUNS (keep ~10)
KEEP: diff_massey_inn9_x_home_team_woba, diff_massey_full_x_elo_diff, diff_massey_inn1, away_elo_centered_x_home_team_woba, elo_sum, wolfe_sum, home_elo, consensus_home_win_prob, home_srs, diff_all_roll20_whip_x_log5_prob_season

### EXTRA_INNINGS (keep ~7)
KEEP: pythag_2nd_diff, pythag_2nd_diff_x_consensus_prob, consensus_home_win_prob, home_pythag_1st, away_srs, pythag_1st_sum, pythag_1st_diff

HARD DROP: All interaction terms with batting metrics, all log5, all elo_prob variants

---

## Analysis Scripts

Scripts in `classical_learning/analysis/importance_ratings/`:
- `clear_folds.py` — Side-by-side expanding vs sliding_3 fold patterns per target
- `cluster_synthesis.py` — ONC cluster x importance x PCA tier synthesis
- `final_summary_2026.py` — Full fold 7 tables, cross-target matrix, temporal trends
- `key_findings.py` — Structured narrative of findings
- `temporal_stability.py` — Expanding vs sliding_3 DECAY/EMERGE/STABLE classification
- `pca_alignment.py` — PCA eigenvalue rank vs importance rank (Kendall tau)
- `deep_analysis.py` — Initial deep fold-level analysis

All scripts expect CSVs at `/tmp/ratings_deep/` (downloaded from S3 during analysis session). Source data at:
- `s3://mlb-265753586044-us-east-1-an/classical_learning/artifacts/importance_ratings/{expanding,sliding_3}/{target}/importance_{mda,desub_mda}_raw.csv`
- `s3://mlb-265753586044-us-east-1-an/classical_learning/artifacts/importance_ratings/{target}/pca_cross_check.csv`
