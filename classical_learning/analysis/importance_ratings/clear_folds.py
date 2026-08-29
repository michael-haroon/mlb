import pandas as pd
import numpy as np
from scipy import stats

def load_all(code):
    mda_exp = pd.read_csv(f'/tmp/ratings_deep/mda_{code}_exp.csv', index_col=0)
    desub_exp = pd.read_csv(f'/tmp/ratings_deep/desub_{code}_exp.csv', index_col=0)
    mda_sl3 = pd.read_csv(f'/tmp/ratings_deep/mda_{code}_sl3.csv', index_col=0)
    desub_sl3 = pd.read_csv(f'/tmp/ratings_deep/desub_{code}_sl3.csv', index_col=0)
    return mda_exp, desub_exp, mda_sl3, desub_sl3

def sign_pat(vals):
    return ''.join(['+' if v > 1e-6 else ('-' if v < -1e-6 else '.') for v in vals])

def t_stat(vals):
    n = len(vals)
    m = vals.mean()
    s = vals.std(ddof=1)
    return m / (s / np.sqrt(n)) if s > 0 else 0.0

targets = [
    ('hw', 'HOME_WIN'),
    ('hrd', 'HOME_RUN_DIFF'),
    ('yrfi', 'YRFI'),
    ('tr', 'TOTAL_RUNS'),
    ('f5tr', 'FIRST_5_TOTAL_RUNS'),
    ('ei', 'EXTRA_INNINGS'),
]

for code, name in targets:
    mda_exp, desub_exp, mda_sl3, desub_sl3 = load_all(code)
    features = mda_exp.columns.tolist()
    n_exp = len(mda_exp)
    n_sl3 = len(mda_sl3)

    print(f"\n{'=' * 140}")
    print(f"  TARGET: {name}")
    print(f"  Expanding: {n_exp} LOYO folds (all prior years as train)")
    print(f"  Sliding_3: {n_sl3} LOYO folds (only 3 most recent years as train)")
    print(f"{'=' * 140}")
    print()

    # Build combined stats
    rows = []
    for feat in features:
        me = mda_exp[feat].values
        de = desub_exp[feat].values
        ms = mda_sl3[feat].values
        ds = desub_sl3[feat].values

        rows.append({
            'feat': feat,
            # Expanding
            'exp_mda_mean': me.mean(),
            'exp_mda_pat': sign_pat(me),
            'exp_mda_pos': int((me > 1e-6).sum()),
            'exp_mda_t': t_stat(me),
            'exp_desub_mean': de.mean(),
            'exp_desub_pat': sign_pat(de),
            'exp_desub_pos': int((de > 1e-6).sum()),
            # Sliding_3
            'sl3_mda_mean': ms.mean(),
            'sl3_mda_pat': sign_pat(ms),
            'sl3_mda_pos': int((ms > 1e-6).sum()),
            'sl3_mda_t': t_stat(ms),
            'sl3_desub_mean': ds.mean(),
            'sl3_desub_pat': sign_pat(ds),
            'sl3_desub_pos': int((ds > 1e-6).sum()),
        })

    df = pd.DataFrame(rows)
    # Sort by average of expanding + sliding_3 MDA means
    df['combined_mean'] = (df['exp_mda_mean'] + df['sl3_mda_mean']) / 2
    df = df.sort_values('combined_mean', ascending=False).reset_index(drop=True)

    # Print header
    print(f"  {'Feature':<34} |{'--- EXPANDING ---':^42}|{'--- SLIDING_3 ---':^42}|")
    print(f"  {'':34} | {'MDA_pat':>8} {'MDA+':>4} {'MDA_x':>9} {'t':>5} {'DSUB_pat':>8} {'D+':>3} | {'MDA_pat':>8} {'MDA+':>4} {'MDA_x':>9} {'t':>5} {'DSUB_pat':>8} {'D+':>3} |")
    print(f"  {'-'*34}-+-{'-'*8}-{'-'*4}-{'-'*9}-{'-'*5}-{'-'*8}-{'-'*3}-+-{'-'*8}-{'-'*4}-{'-'*9}-{'-'*5}-{'-'*8}-{'-'*3}-+")

    for _, r in df.iterrows():
        print(f"  {r['feat']:<34} | {r['exp_mda_pat']:>8} {r['exp_mda_pos']:>2}/{n_exp} {r['exp_mda_mean']:>+9.2e} {r['exp_mda_t']:>5.1f} {r['exp_desub_pat']:>8} {r['exp_desub_pos']:>1}/{n_exp} | {r['sl3_mda_pat']:>8} {r['sl3_mda_pos']:>2}/{n_sl3} {r['sl3_mda_mean']:>+9.2e} {r['sl3_mda_t']:>5.1f} {r['sl3_desub_pat']:>8} {r['sl3_desub_pos']:>1}/{n_sl3} |")

    # Summary
    exp_useful = (df['exp_mda_pos'] >= 6).sum()
    sl3_useful = (df['sl3_mda_pos'] >= 6).sum()
    exp_harmful = (df['exp_mda_mean'] < -1e-6).sum()
    sl3_harmful = (df['sl3_mda_mean'] < -1e-6).sum()
    both_useful = ((df['exp_mda_pos'] >= 6) & (df['sl3_mda_pos'] >= 6)).sum()
    both_harmful = ((df['exp_mda_mean'] < -1e-6) & (df['sl3_mda_mean'] < -1e-6)).sum()

    print()
    print(f"  SUMMARY:")
    print(f"    Expanding:  {exp_useful} features with 6+/{n_exp} positive folds | {exp_harmful} features with negative mean")
    print(f"    Sliding_3:  {sl3_useful} features with 6+/{n_sl3} positive folds | {sl3_harmful} features with negative mean")
    print(f"    BOTH modes: {both_useful} features useful in both | {both_harmful} features harmful in both")
    print()
