import pandas as pd
import numpy as np
from scipy import stats

print()
print("=" * 120)
print("KEY FINDINGS: Rating Feature Importance Deep Dive")
print("=" * 120)
print()

print("""
1. FOLD-LEVEL SIGN PATTERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOME_WIN & HOME_RUN_DIFF: The elo/massey/pythag DIFFERENTIAL features show remarkably
consistent positive importance — 7-8/8 folds positive in MDA, 8/8 in DESUB.

Key insight from DESUB vs MDA disagreement:
  - DESUB (substitution method) shows 8/8 positive for nearly ALL features in home_win/home_run_diff
  - MDA (permutation) shows only 5-7/8 positive for the same features
  - This means: shuffling the feature DOES hurt the model (MDA), but not consistently every fold.
    The substitution of the feature's mean value (DESUB) ALWAYS hurts — meaning the feature
    carries signal in every fold, but some folds are dominated by other correlated ratings
    that absorb the permutation variance.
  - Translation: The features carry redundant signal (expected — massey/colley/elo all encode
    team strength). MDA underestimates individual importance because of multicollinearity.
    DESUB correctly attributes importance even under collinearity.

YRFI: ZERO features achieve consistent positive importance. Best feature (home_pythag_2nd)
achieves only 6/8 positive folds, and with near-zero magnitude. Ratings genuinely add nothing.

EXTRA_INNINGS: Features are actively HARMFUL. elo_prob achieves --------  (ALL negative folds).
diff_all_roll20_whip_x_log5_prob_season is -------- in both expanding and sliding_3.
This is NOT noise — it's systematic damage. Including elo in the extra_innings model makes
predictions WORSE by encouraging the tree to split on team strength, which is orthogonal to
whether a game goes to extras.


2. TEMPORAL DECAY
━━━━━━━━━━━━━━━━
For home_win, the top features show a systematic DOWNWARD trend in importance:
  elo_prob:           r=-0.86 (fold 0: 5.84e-4, fold 7: -0.58e-4)
  elo_diff:           r=-0.94 (fold 0: 5.91e-4, fold 7: -0.63e-4)
  elo_prob_x_same_league: r=-0.90

This means recent years (2024-2026) extract LESS signal from Elo than early years (2018-2020).
Possible causes:
  a) Market efficiency: as more bettors use Elo, the edge it provides over the market line shrinks
  b) Game evolution: increased specialization (openers, bullpen games) makes season-long Elo
     less predictive of single-game outcomes
  c) Training artifact: expanding CV trains on progressively more data in later folds,
     the model may rely less on any single feature

BUT: sliding_3 still shows 6-7/8 positive folds for elo_prob/elo_diff. The signal persists,
just at reduced magnitude. The feature is WEAKENING, not DYING.


3. PCA ALIGNMENT
━━━━━━━━━━━━━━━━
HOME_WIN / HOME_RUN_DIFF:
  PC_0 (37.7% variance explained) ranks #1 in both MDI and MDA importance.
  Top-3 PCs hold 62-65% of total MDA importance.
  The dominant linear direction in the rating feature space IS the predictive direction.
  → Implication: Even a linear model would capture most of the rating signal for win probability.

YRFI:
  PC_0 ranks #16-17 in importance. 19/21 PCs have NEGATIVE MDA.
  Negative tau between eigenvalue rank and MDI rank (tau=-0.37, p=0.019).
  The dominant variance direction is ANTI-correlated with predictive value.
  → Implication: There is no linear combination of rating features that helps YRFI prediction.

EXTRA_INNINGS:
  PC_0 ranks LAST (#21) in MDA. 20/21 PCs have negative MDA.
  The strongest eigenvalue direction is the MOST harmful to prediction.
  → Implication: Ratings actively mislead the model by tempting it to split on team strength.


4. CLUSTER STRUCTURE
━━━━━━━━━━━━━━━━━━━━
Cluster 0 (42 features): Differentials, home-team measures, probabilities
  - Contains ALL the tier-S and tier-A features for home_win
  - Captures "how much better is the home team?" signal
  - 88% of these features have positive mean MDA for home_win

Cluster 1 (17 features): Sums, away measures, interactions, symmetric measures
  - Mixed signal — includes top interactions (away_elo_centered_x_...) but also pure noise (wolfe_sum, srs_sum)
  - The 4 features from C1 that are tier-S are ALL interaction terms that cross a rating with a batting metric
  - The 7 pure rating features in C1 (sums, away_X) are majority tier-F

Key insight: Cluster 1 "sums" (elo_sum, srs_sum, pythag_1st_sum, pythag_2nd_sum, wolfe_sum) are
UNIVERSALLY in tier-F across all targets. They encode combined team strength without direction.
For win prediction, you need the DIFFERENTIAL (who's better), not the sum (how good are both teams).
For totals, the sum might seem useful (two good teams = more runs?) but empirically it doesn't help.


5. ACTIONABLE RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FOR THE DL MODEL:
  - Feed the S/A tier features (27 for home_win, 32 for home_run_diff) as fixed inputs
  - These encode knowledge the DL cannot derive from pitch sequences alone
  - DO NOT feed them for yrfi/extra_innings/first_5_total_runs — they add noise

FOR THE CLASSICAL MODEL:
  - Drop tier-F features from the feature allowlist for EACH TARGET separately
  - home_win/home_run_diff: drop 12/7 features (mostly sums + wolfe)
  - yrfi: drop 54/59 features — keep ONLY home_pythag_2nd, diff_massey_inn1, and log5_prob_short
  - extra_innings: drop 43/59 features
  - first_5_total_runs: drop 56/59 features

  WARNING: The top features show temporal decay (r=-0.86 to -0.94).
  Consider game-indexed decay weighting in the rolling Elo calculation itself,
  or a time-interaction feature to let the model learn the decay.
""")
