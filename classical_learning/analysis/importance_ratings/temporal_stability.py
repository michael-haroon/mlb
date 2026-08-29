import pandas as pd
import numpy as np
from scipy import stats

print("=" * 120)
print("TEMPORAL STABILITY: Expanding vs Sliding_3 fold-by-fold comparison")
print("=" * 120)
print()
print("Key question: Do features gain/lose signal in recent years vs. full history?")
print("A feature positive in expanding but negative in sliding_3 = decaying signal (historically useful, recently dead)")
print("A feature negative in expanding but positive in sliding_3 = emerging signal (recently useful, historically diluted)")
print()

targets = [
    ('hw', 'home_win'),
    ('hrd', 'home_run_diff'),
    ('yrfi', 'yrfi'),
    ('tr', 'total_runs'),
    ('ei', 'extra_innings'),
]

for code, name in targets:
    mda_exp = pd.read_csv(f'/tmp/ratings_deep/mda_{code}_exp.csv', index_col=0)
    mda_sl3 = pd.read_csv(f'/tmp/ratings_deep/mda_{code}_sl3.csv', index_col=0)
    desub_exp = pd.read_csv(f'/tmp/ratings_deep/desub_{code}_exp.csv', index_col=0)
    desub_sl3 = pd.read_csv(f'/tmp/ratings_deep/desub_{code}_sl3.csv', index_col=0)

    features = mda_exp.columns.tolist()
    n_exp = len(mda_exp)
    n_sl3 = len(mda_sl3)

    print(f"{'━' * 120}")
    print(f"  TARGET: {name.upper()} | {n_exp} expanding folds vs {n_sl3} sliding_3 folds")
    print(f"{'━' * 120}")

    results = []
    for feat in features:
        exp_mean = mda_exp[feat].mean()
        sl3_mean = mda_sl3[feat].mean()
        exp_std = mda_exp[feat].std(ddof=1)
        sl3_std = mda_sl3[feat].std(ddof=1)

        exp_pos = int((mda_exp[feat] > 1e-6).sum())
        sl3_pos = int((mda_sl3[feat] > 1e-6).sum())

        # Fold-by-fold sign patterns
        exp_sign = ''.join(['+' if v > 1e-6 else ('-' if v < -1e-6 else '.') for v in mda_exp[feat].values])
        sl3_sign = ''.join(['+' if v > 1e-6 else ('-' if v < -1e-6 else '.') for v in mda_sl3[feat].values])

        # Direction change
        if exp_mean > 1e-6 and sl3_mean < -1e-6:
            direction = "DECAY"
        elif exp_mean < -1e-6 and sl3_mean > 1e-6:
            direction = "EMERGE"
        elif exp_mean > 1e-6 and sl3_mean > 1e-6:
            ratio = sl3_mean / exp_mean if exp_mean != 0 else 1
            if ratio > 1.5:
                direction = "STRENGTHEN"
            elif ratio < 0.5:
                direction = "WEAKEN"
            else:
                direction = "STABLE+"
        elif exp_mean < -1e-6 and sl3_mean < -1e-6:
            direction = "STABLE-"
        else:
            direction = "FLAT"

        results.append({
            'feat': feat, 'exp_mean': exp_mean, 'sl3_mean': sl3_mean,
            'exp_pos': exp_pos, 'sl3_pos': sl3_pos,
            'exp_sign': exp_sign, 'sl3_sign': sl3_sign,
            'direction': direction, 'delta': sl3_mean - exp_mean,
        })

    # Sort by absolute delta (biggest change first)
    results.sort(key=lambda x: -abs(x['delta']))

    # Show direction summary
    dir_counts = {}
    for r in results:
        dir_counts[r['direction']] = dir_counts.get(r['direction'], 0) + 1
    print(f"  Direction summary: {dir_counts}")
    print()

    # Show features with biggest temporal shifts
    print(f"  {'Feature':<36} {'Exp_mean':>9} {'Sl3_mean':>9} {'Delta':>9} {'Exp_pat':>9} {'Sl3_pat':>9} {'Dir':<10}")
    print(f"  {'─'*36} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*10}")
    for r in results[:20]:
        print(f"  {r['feat']:<36} {r['exp_mean']:>9.2e} {r['sl3_mean']:>9.2e} {r['delta']:>+9.2e} {r['exp_sign']:>9} {r['sl3_sign']:>9} {r['direction']:<10}")

    # Specifically highlight DECAY and EMERGE
    decays = [r for r in results if r['direction'] == 'DECAY']
    emerges = [r for r in results if r['direction'] == 'EMERGE']
    if decays:
        print(f"\n  DECAYING SIGNAL ({len(decays)} features — were useful, now hurt):")
        for r in decays:
            print(f"    {r['feat']:<36} exp={r['exp_mean']:+.2e} sl3={r['sl3_mean']:+.2e}  exp_pat={r['exp_sign']} sl3_pat={r['sl3_sign']}")
    if emerges:
        print(f"\n  EMERGING SIGNAL ({len(emerges)} features — were noise, now useful):")
        for r in emerges:
            print(f"    {r['feat']:<36} exp={r['exp_mean']:+.2e} sl3={r['sl3_mean']:+.2e}  exp_pat={r['exp_sign']} sl3_pat={r['sl3_sign']}")
    print()

print()
print("=" * 120)
print("FOLD-BY-FOLD TEMPORAL TREND: Top features, expanding mode")
print("=" * 120)
print()
print("For the top features in home_win and home_run_diff, show per-fold values to detect temporal trends")
print("(are recent folds stronger or weaker than early folds?)")
print()

for code, name in [('hw', 'home_win'), ('hrd', 'home_run_diff')]:
    mda = pd.read_csv(f'/tmp/ratings_deep/mda_{code}_exp.csv', index_col=0)
    n_folds = len(mda)

    top_feats = mda.mean().nlargest(10).index.tolist()

    print(f"  TARGET: {name.upper()} — Top 10 features, per-fold MDA (x1e4)")
    print(f"  {'Feature':<36} " + " ".join([f"{'F'+str(i):>7}" for i in range(n_folds)]) + f" {'Mean':>8} {'Trend':>7}")
    print(f"  {'─'*36} " + " ".join(["─"*7]*n_folds) + f" {'─'*8} {'─'*7}")

    for feat in top_feats:
        vals = mda[feat].values * 1e4
        mean_v = vals.mean()
        # Linear trend (slope of importance over time)
        slope, _, r, p, _ = stats.linregress(np.arange(n_folds), vals)
        trend_dir = "UP" if slope > 0.1 else ("DOWN" if slope < -0.1 else "FLAT")
        print(f"  {feat:<36} " + " ".join([f"{v:>7.2f}" for v in vals]) + f" {mean_v:>8.2f} {trend_dir:>5} (r={r:+.2f})")
    print()
