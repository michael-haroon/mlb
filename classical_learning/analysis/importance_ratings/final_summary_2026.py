import pandas as pd
import numpy as np
from scipy import stats
import json

def load_all(code):
    mda_exp = pd.read_csv(f'/tmp/ratings_deep/mda_{code}_exp.csv', index_col=0)
    desub_exp = pd.read_csv(f'/tmp/ratings_deep/desub_{code}_exp.csv', index_col=0)
    mda_sl3 = pd.read_csv(f'/tmp/ratings_deep/mda_{code}_sl3.csv', index_col=0)
    desub_sl3 = pd.read_csv(f'/tmp/ratings_deep/desub_{code}_sl3.csv', index_col=0)
    return mda_exp, desub_exp, mda_sl3, desub_sl3

cluster_map = json.load(open('/tmp/ratings_deep/cluster_map.json'))
c0 = set(cluster_map['0'])
c1 = set(cluster_map['1'])

targets = [
    ('hw', 'HOME_WIN'),
    ('hrd', 'HOME_RUN_DIFF'),
    ('yrfi', 'YRFI'),
    ('tr', 'TOTAL_RUNS'),
    ('f5tr', 'FIRST_5_TOTAL_RUNS'),
    ('ei', 'EXTRA_INNINGS'),
]

# ============================================================================
# SECTION 1: What does each fold represent?
# ============================================================================
print("=" * 130)
print("COMPREHENSIVE RATING FEATURE ANALYSIS — FOCUSED ON 2026 SIGNAL STRENGTH")
print("=" * 130)
print()
print("FOLD IDENTITY (Expanding CV — 8 folds, each leaves out one year as test):")
print("  Fold 0 = test on ~2019 (train: 2015-2018)")
print("  Fold 1 = test on ~2020 (train: 2015-2019)")
print("  Fold 2 = test on ~2021 (train: 2015-2020)")
print("  Fold 3 = test on ~2022 (train: 2015-2021)")
print("  Fold 4 = test on ~2023 (train: 2015-2022)")
print("  Fold 5 = test on ~2024 (train: 2015-2023)")
print("  Fold 6 = test on ~2025 (train: 2015-2024)")
print("  Fold 7 = test on ~2026 (train: 2015-2025)  <-- THIS IS THE ONE THAT MATTERS")
print()
print("FOLD IDENTITY (Sliding_3 CV — 8 folds, each trains on only 3 prior years):")
print("  Fold 0 = test on ~2019 (train: 2016-2018)")
print("  Fold 1 = test on ~2020 (train: 2017-2019)")
print("  ...same mapping...")
print("  Fold 7 = test on ~2026 (train: 2023-2025)  <-- THIS IS THE ONE THAT MATTERS")
print()
print("KEY: Fold 7 is the most recent year. A feature that is '+' in fold 7 has LIVE signal in 2026.")
print("     A feature that is '-' in fold 7 is HURTING predictions RIGHT NOW.")
print()

# ============================================================================
# SECTION 2: Per-target, show fold 7 (2026) value for each feature
# ============================================================================
print("=" * 130)
print("SECTION 1: FOLD 7 (2026) — CURRENT SIGNAL STRENGTH")
print("=" * 130)
print()
print("What matters for live trading is fold 7. Features sorted by fold 7 MDA value (sliding_3),")
print("because sliding_3 fold 7 trains on 2023-2025 and tests on 2026 — closest to production.")
print()

for code, name in targets:
    mda_exp, desub_exp, mda_sl3, desub_sl3 = load_all(code)
    features = mda_exp.columns.tolist()

    print(f"{'━' * 130}")
    print(f"  TARGET: {name}")
    print(f"{'━' * 130}")

    rows = []
    for feat in features:
        # Fold 7 values
        exp_f7 = mda_exp[feat].iloc[-1]
        sl3_f7 = mda_sl3[feat].iloc[-1]
        desub_exp_f7 = desub_exp[feat].iloc[-1]
        desub_sl3_f7 = desub_sl3[feat].iloc[-1]

        # Also fold 6 for trend
        exp_f6 = mda_exp[feat].iloc[-2]
        sl3_f6 = mda_sl3[feat].iloc[-2]

        # Full expanding pattern
        exp_pat = ''.join(['+' if v > 1e-6 else ('-' if v < -1e-6 else '.') for v in mda_exp[feat].values])
        sl3_pat = ''.join(['+' if v > 1e-6 else ('-' if v < -1e-6 else '.') for v in mda_sl3[feat].values])

        # How many of last 3 folds (5,6,7) are positive?
        last3_exp = sum(1 for v in mda_exp[feat].values[-3:] if v > 1e-6)
        last3_sl3 = sum(1 for v in mda_sl3[feat].values[-3:] if v > 1e-6)

        rows.append({
            'feat': feat,
            'sl3_f7': sl3_f7,
            'exp_f7': exp_f7,
            'sl3_f6': sl3_f6,
            'exp_f6': exp_f6,
            'desub_sl3_f7': desub_sl3_f7,
            'desub_exp_f7': desub_exp_f7,
            'exp_pat': exp_pat,
            'sl3_pat': sl3_pat,
            'last3_exp': last3_exp,
            'last3_sl3': last3_sl3,
        })

    df = pd.DataFrame(rows).sort_values('sl3_f7', ascending=False)

    print(f"  {'Feature':<36} {'SL3_F7':>9} {'EXP_F7':>9} {'SL3_F6':>9} {'DSUB_SL3_7':>11} {'Last3_sl3':>10} {'SL3 pattern':>11} {'EXP pattern':>11}")
    print(f"  {'-'*36} {'-'*9} {'-'*9} {'-'*9} {'-'*11} {'-'*10} {'-'*11} {'-'*11}")

    for _, r in df.iterrows():
        f7_sign = '+' if r['sl3_f7'] > 1e-6 else ('-' if r['sl3_f7'] < -1e-6 else '.')
        f6_sign = '+' if r['sl3_f6'] > 1e-6 else ('-' if r['sl3_f6'] < -1e-6 else '.')
        last3_label = f"{r['last3_sl3']}/3"
        print(f"  {r['feat']:<36} {r['sl3_f7']:>+9.2e} {r['exp_f7']:>+9.2e} {r['sl3_f6']:>+9.2e} {r['desub_sl3_f7']:>+11.2e} {last3_label:>10} {r['sl3_pat']:>11} {r['exp_pat']:>11}")

    # Count positives in fold 7
    n_pos_sl3_f7 = (df['sl3_f7'] > 1e-6).sum()
    n_neg_sl3_f7 = (df['sl3_f7'] < -1e-6).sum()
    n_pos_exp_f7 = (df['exp_f7'] > 1e-6).sum()
    n_neg_exp_f7 = (df['exp_f7'] < -1e-6).sum()

    # Features positive in fold 7 of BOTH modes
    both_pos_f7 = ((df['sl3_f7'] > 1e-6) & (df['exp_f7'] > 1e-6)).sum()

    print(f"\n  Fold 7 (2026): Sliding_3: {n_pos_sl3_f7} positive / {n_neg_sl3_f7} negative")
    print(f"                 Expanding: {n_pos_exp_f7} positive / {n_neg_exp_f7} negative")
    print(f"                 BOTH positive in 2026: {both_pos_f7} / 59 features")
    print()

# ============================================================================
# SECTION 3: Cross-target comparison for 2026
# ============================================================================
print()
print("=" * 130)
print("SECTION 2: CROSS-TARGET 2026 SIGNAL SUMMARY")
print("=" * 130)
print()
print("For each feature, show fold 7 (2026) sign across all targets in sliding_3 mode.")
print("This shows WHERE each feature carries live signal right now.")
print()

# Build cross-target matrix
all_features = None
target_f7 = {}
for code, name in targets:
    _, _, mda_sl3, _ = load_all(code)
    if all_features is None:
        all_features = mda_sl3.columns.tolist()
    target_f7[name] = mda_sl3.iloc[-1]  # fold 7

# Sort features by how many targets they're positive in fold 7
feat_scores = []
for feat in all_features:
    signs = []
    vals = []
    for _, name in targets:
        v = target_f7[name][feat]
        signs.append('+' if v > 1e-6 else ('-' if v < -1e-6 else '.'))
        vals.append(v)
    n_pos = signs.count('+')
    n_neg = signs.count('-')
    feat_scores.append((feat, signs, n_pos, n_neg, vals))

feat_scores.sort(key=lambda x: (-x[2], x[3]))

header_targets = [n[:6] for _, n in targets]
print(f"  {'Feature':<36} {' '.join([f'{t:>8}' for t in header_targets])} {'Pos':>4} {'Neg':>4}")
print(f"  {'-'*36} {' '.join(['-'*8]*6)} {'-'*4} {'-'*4}")

for feat, signs, n_pos, n_neg, vals in feat_scores:
    vals_str = ' '.join([f'{v:>+8.1e}' for v in vals])
    print(f"  {feat:<36} {vals_str} {n_pos:>4} {n_neg:>4}")

print()
print()

# ============================================================================
# SECTION 4: Trend analysis — is 2026 getting better or worse?
# ============================================================================
print("=" * 130)
print("SECTION 3: TEMPORAL TREND — IS THE FEATURE GETTING STRONGER OR WEAKER HEADING INTO 2026?")
print("=" * 130)
print()
print("Looking at folds 5-6-7 (2024-2025-2026) in sliding_3 mode.")
print("RISING = fold 7 > fold 5 (feature getting more useful)")
print("FALLING = fold 7 < fold 5 (feature decaying)")
print("Note: only shown for home_win and home_run_diff where ratings have signal")
print()

for code, name in [('hw', 'HOME_WIN'), ('hrd', 'HOME_RUN_DIFF')]:
    _, _, mda_sl3, _ = load_all(code)
    features = mda_sl3.columns.tolist()

    print(f"  TARGET: {name} — Sliding_3 folds 5/6/7 (2024/2025/2026)")
    print(f"  {'Feature':<36} {'F5':>9} {'F6':>9} {'F7':>9} {'F7-F5':>9} {'Trend':<8}")
    print(f"  {'-'*36} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*8}")

    rows = []
    for feat in features:
        f5 = mda_sl3[feat].iloc[-3]
        f6 = mda_sl3[feat].iloc[-2]
        f7 = mda_sl3[feat].iloc[-1]
        delta = f7 - f5
        if f7 > 1e-6 and delta > 1e-5:
            trend = "RISING"
        elif f7 > 1e-6 and delta < -1e-5:
            trend = "FALLING"
        elif f7 > 1e-6:
            trend = "STABLE+"
        elif f7 < -1e-6 and delta < -1e-5:
            trend = "WORSENING"
        elif f7 < -1e-6:
            trend = "DEAD"
        else:
            trend = "FLAT"
        rows.append((feat, f5, f6, f7, delta, trend))

    rows.sort(key=lambda x: -x[3])
    for feat, f5, f6, f7, delta, trend in rows:
        print(f"  {feat:<36} {f5:>+9.2e} {f6:>+9.2e} {f7:>+9.2e} {delta:>+9.2e} {trend:<8}")
    print()
