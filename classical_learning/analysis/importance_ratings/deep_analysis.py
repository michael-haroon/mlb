import pandas as pd
import numpy as np
from scipy import stats
import sys

def load_raw(code):
    mda = pd.read_csv(f'/tmp/ratings_deep/mda_{code}_exp.csv', index_col=0)
    desub = pd.read_csv(f'/tmp/ratings_deep/desub_{code}_exp.csv', index_col=0)
    mda_sl3 = pd.read_csv(f'/tmp/ratings_deep/mda_{code}_sl3.csv', index_col=0)
    desub_sl3 = pd.read_csv(f'/tmp/ratings_deep/desub_{code}_sl3.csv', index_col=0)
    return mda, desub, mda_sl3, desub_sl3

targets = [
    ('hw', 'home_win'),
    ('hrd', 'home_run_diff'),
    ('yrfi', 'yrfi'),
    ('tr', 'total_runs'),
    ('f5tr', 'first_5_total_runs'),
    ('ei', 'extra_innings'),
]

print("=" * 120)
print("FOLD-LEVEL IMPORTANCE DEEP DIVE: Per-feature sign patterns across LOYO folds")
print("=" * 120)
print()
print("'+' = positive MDA in that fold (feature helps prediction)")
print("'-' = negative MDA in that fold (feature HURTS prediction)")
print("Folds are chronological: fold 0 = earliest year, fold 7 = most recent (2026)")
print("t-stat: one-sample t-test H0: mean_importance = 0")
print()

for code, name in targets:
    mda, desub, mda_sl3, desub_sl3 = load_raw(code)
    n_folds = len(mda)
    features = mda.columns.tolist()

    print(f"{'━' * 120}")
    print(f"  TARGET: {name.upper()} | Expanding CV ({n_folds} folds)")
    print(f"{'━' * 120}")

    results = []
    for feat in features:
        vals = mda[feat].values
        desub_vals = desub[feat].values
        mean_mda = vals.mean()
        std_mda = vals.std(ddof=1)
        n_pos = int((vals > 1e-6).sum())
        sign_pattern = ''.join(['+' if v > 1e-6 else ('-' if v < -1e-6 else '.') for v in vals])
        mean_desub = desub_vals.mean()
        n_pos_desub = int((desub_vals > 1e-6).sum())
        desub_pattern = ''.join(['+' if v > 1e-6 else ('-' if v < -1e-6 else '.') for v in desub_vals])
        t_stat = mean_mda / (std_mda / np.sqrt(n_folds)) if std_mda > 0 else 0.0

        # Sliding_3 pattern
        s_vals = mda_sl3[feat].values
        s_n_folds = len(s_vals)
        s_sign = ''.join(['+' if v > 1e-6 else ('-' if v < -1e-6 else '.') for v in s_vals])
        s_mean = s_vals.mean()

        results.append({
            'feat': feat, 'mean_mda': mean_mda, 'std_mda': std_mda,
            'n_pos': n_pos, 'sign_pattern': sign_pattern,
            'mean_desub': mean_desub, 'n_pos_desub': n_pos_desub, 'desub_pattern': desub_pattern,
            't_stat': t_stat,
            's_sign': s_sign, 's_mean': s_mean,
        })

    results.sort(key=lambda x: -x['mean_mda'])

    hdr = f"  {'Feature':<36} {'x_MDA':>9} {'std':>9} {'MDA folds':>9} {'DESUB folds':>11} {'Sl3 folds':>9} {'t':>6}"
    print(hdr)
    print(f"  {'─'*36} {'─'*9} {'─'*9} {'─'*9} {'─'*11} {'─'*9} {'─'*6}")
    for r in results:
        print(f"  {r['feat']:<36} {r['mean_mda']:>9.2e} {r['std_mda']:>9.2e} {r['sign_pattern']:>9} {r['desub_pattern']:>11} {r['s_sign']:>9} {r['t_stat']:>6.2f}")

    # Summary stats
    n_all_pos = sum(1 for r in results if r['n_pos'] == n_folds)
    n_all_pos_both = sum(1 for r in results if r['n_pos'] == n_folds and r['n_pos_desub'] == n_folds)
    n_majority_neg = sum(1 for r in results if r['n_pos'] <= n_folds // 2 - 1)
    n_sig = sum(1 for r in results if r['t_stat'] > 2.365)  # t_crit for df=7, alpha=0.025 one-sided

    print(f"\n  Summary: {n_all_pos} features all-positive MDA | {n_all_pos_both} all-positive BOTH methods | {n_majority_neg} majority-negative | {n_sig} significant (t>{2.365:.3f}, p<0.025)")
    print()

sys.stdout.flush()
