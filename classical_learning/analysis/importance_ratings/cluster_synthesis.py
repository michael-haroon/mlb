import pandas as pd
import numpy as np
import json
from scipy import stats

cluster_map = json.load(open('/tmp/ratings_deep/cluster_map.json'))

print("=" * 120)
print("ONC CLUSTER x IMPORTANCE x PCA SYNTHESIS")
print("=" * 120)
print()
print(f"Cluster 0: {len(cluster_map['0'])} features | Cluster 1: {len(cluster_map['1'])} features")
print()

c0 = set(cluster_map['0'])
c1 = set(cluster_map['1'])
print("Cluster 0 (differentials/home team bias):")
for f in sorted(c0):
    print(f"  {f}")
print()
print("Cluster 1 (sums/away/symmetric):")
for f in sorted(c1):
    print(f"  {f}")
print()

print("-" * 120)
print("CLUSTER-LEVEL IMPORTANCE MEANS (expanding MDA, x1e4)")
print("-" * 120)
print()

targets = [
    ('hw', 'home_win'),
    ('hrd', 'home_run_diff'),
    ('yrfi', 'yrfi'),
    ('tr', 'total_runs'),
    ('ei', 'extra_innings'),
]

print(f"  {'Target':<20} {'C0 mean':>9} {'C0 std':>9} {'C0 +frac':>9} | {'C1 mean':>9} {'C1 std':>9} {'C1 +frac':>9} | {'Delta':>9}")
print(f"  {'-'*20} {'-'*9} {'-'*9} {'-'*9}   {'-'*9} {'-'*9} {'-'*9}   {'-'*9}")

for code, name in targets:
    mda = pd.read_csv(f'/tmp/ratings_deep/mda_{code}_exp.csv', index_col=0)
    features = mda.columns.tolist()

    c0_feats = [f for f in features if f in c0]
    c1_feats = [f for f in features if f in c1]

    c0_means = mda[c0_feats].mean(axis=0)
    c1_means = mda[c1_feats].mean(axis=0)

    c0_avg = c0_means.mean() * 1e4
    c0_std = c0_means.std() * 1e4
    c0_pos = (c0_means > 1e-6).mean()

    c1_avg = c1_means.mean() * 1e4
    c1_std = c1_means.std() * 1e4
    c1_pos = (c1_means > 1e-6).mean()

    delta = c0_avg - c1_avg
    print(f"  {name:<20} {c0_avg:>9.3f} {c0_std:>9.3f} {c0_pos:>9.1%} | {c1_avg:>9.3f} {c1_std:>9.3f} {c1_pos:>9.1%} | {delta:>+9.3f}")

print()
print("-" * 120)
print("FINAL FEATURE TIER LIST: Combining all evidence (HOME_WIN)")
print("-" * 120)
print()
print("Tier criteria:")
print("  S: All-positive folds in BOTH MDA+DESUB, t>2.4, consistent across expanding/sliding_3")
print("  A: 7/8+ positive folds in MDA, positive in sliding_3, or 8/8 DESUB")
print("  B: 5-6/8 positive folds, mostly stable")
print("  C: Mixed signal, high variance, decaying or unstable")
print("  F: Majority-negative or zero mean, blocked by EB")
print()

mda = pd.read_csv('/tmp/ratings_deep/mda_hw_exp.csv', index_col=0)
desub = pd.read_csv('/tmp/ratings_deep/desub_hw_exp.csv', index_col=0)
mda_sl3 = pd.read_csv('/tmp/ratings_deep/mda_hw_sl3.csv', index_col=0)

features = mda.columns.tolist()
n = len(mda)

tiers = []
for feat in features:
    vals = mda[feat].values
    d_vals = desub[feat].values
    s_vals = mda_sl3[feat].values

    mean_v = vals.mean()
    std_v = vals.std(ddof=1)
    t_stat = mean_v / (std_v / np.sqrt(n)) if std_v > 0 else 0
    n_pos_mda = int((vals > 1e-6).sum())
    n_pos_desub = int((d_vals > 1e-6).sum())
    n_pos_sl3 = int((s_vals > 1e-6).sum())
    cluster = 0 if feat in c0 else 1

    if n_pos_mda >= n-1 and n_pos_desub >= n-1 and t_stat > 2.4:
        tier = "S"
    elif n_pos_mda >= n-1 and n_pos_desub >= n-1:
        tier = "A"
    elif n_pos_mda >= 6 and mean_v > 0 and n_pos_sl3 >= 5:
        tier = "A"
    elif n_pos_mda >= 5 and mean_v > 0:
        tier = "B"
    elif mean_v > 0:
        tier = "C"
    else:
        tier = "F"

    tiers.append((feat, tier, n_pos_mda, n_pos_desub, n_pos_sl3, t_stat, mean_v, cluster))

tiers.sort(key=lambda x: ('SABCF'.index(x[1]), -x[6]))

print(f"  {'Feature':<38} {'Tier':<4} {'MDA+':>5} {'DSUB+':>6} {'SL3+':>5} {'t':>6} {'x_MDA(e4)':>10} {'Cluster':>8}")
print(f"  {'-'*38} {'-'*4} {'-'*5} {'-'*6} {'-'*5} {'-'*6} {'-'*10} {'-'*8}")

for feat, tier, n_pos_mda, n_pos_desub, n_pos_sl3, t_stat, mean_v, cluster in tiers:
    print(f"  {feat:<38} {tier:<4} {n_pos_mda}/{n}   {n_pos_desub}/{n}    {n_pos_sl3}/{n}   {t_stat:>6.2f} {mean_v*1e4:>10.2f} {'C0' if cluster==0 else 'C1':>8}")

tier_counts = {}
for _, t, *_ in tiers:
    tier_counts[t] = tier_counts.get(t, 0) + 1
print(f"\n  Tier counts: S={tier_counts.get('S',0)} | A={tier_counts.get('A',0)} | B={tier_counts.get('B',0)} | C={tier_counts.get('C',0)} | F={tier_counts.get('F',0)}")
print()

# Same for HOME_RUN_DIFF
print("-" * 120)
print("FINAL FEATURE TIER LIST: HOME_RUN_DIFF")
print("-" * 120)
print()

mda = pd.read_csv('/tmp/ratings_deep/mda_hrd_exp.csv', index_col=0)
desub = pd.read_csv('/tmp/ratings_deep/desub_hrd_exp.csv', index_col=0)
mda_sl3 = pd.read_csv('/tmp/ratings_deep/mda_hrd_sl3.csv', index_col=0)

tiers = []
for feat in features:
    vals = mda[feat].values
    d_vals = desub[feat].values
    s_vals = mda_sl3[feat].values

    mean_v = vals.mean()
    std_v = vals.std(ddof=1)
    t_stat = mean_v / (std_v / np.sqrt(n)) if std_v > 0 else 0
    n_pos_mda = int((vals > 1e-6).sum())
    n_pos_desub = int((d_vals > 1e-6).sum())
    n_pos_sl3 = int((s_vals > 1e-6).sum())
    cluster = 0 if feat in c0 else 1

    if n_pos_mda >= n-1 and n_pos_desub >= n-1 and t_stat > 2.4:
        tier = "S"
    elif n_pos_mda >= n-1 and n_pos_desub >= n-1:
        tier = "A"
    elif n_pos_mda >= 6 and mean_v > 0 and n_pos_sl3 >= 5:
        tier = "A"
    elif n_pos_mda >= 5 and mean_v > 0:
        tier = "B"
    elif mean_v > 0:
        tier = "C"
    else:
        tier = "F"

    tiers.append((feat, tier, n_pos_mda, n_pos_desub, n_pos_sl3, t_stat, mean_v, cluster))

tiers.sort(key=lambda x: ('SABCF'.index(x[1]), -x[6]))

print(f"  {'Feature':<38} {'Tier':<4} {'MDA+':>5} {'DSUB+':>6} {'SL3+':>5} {'t':>6} {'x_MDA(e4)':>10} {'Cluster':>8}")
print(f"  {'-'*38} {'-'*4} {'-'*5} {'-'*6} {'-'*5} {'-'*6} {'-'*10} {'-'*8}")

for feat, tier, n_pos_mda, n_pos_desub, n_pos_sl3, t_stat, mean_v, cluster in tiers:
    print(f"  {feat:<38} {tier:<4} {n_pos_mda}/{n}   {n_pos_desub}/{n}    {n_pos_sl3}/{n}   {t_stat:>6.2f} {mean_v*1e4:>10.2f} {'C0' if cluster==0 else 'C1':>8}")

tier_counts = {}
for _, t, *_ in tiers:
    tier_counts[t] = tier_counts.get(t, 0) + 1
print(f"\n  Tier counts: S={tier_counts.get('S',0)} | A={tier_counts.get('A',0)} | B={tier_counts.get('B',0)} | C={tier_counts.get('C',0)} | F={tier_counts.get('F',0)}")
