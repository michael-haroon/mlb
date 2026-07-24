"""Model-free single-feature AUC by schedule context.

No model needed: roc_auc_score(y_true, feature_value) measures how well
a raw feature alone discriminates wins from losses. Sliced by matchup type
to detect schedule-dependent signal degradation.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

df = pd.read_parquet(
    "pregame/artifacts/features/game_features.parquet"
)

TEAM_LEAGUE = {
    108: 103, 109: 104, 110: 103, 111: 103, 112: 104, 113: 104, 114: 103,
    115: 104, 116: 103, 117: 103, 118: 103, 119: 104, 120: 104, 121: 104,
    133: 103, 134: 104, 135: 104, 136: 103, 137: 104, 138: 104, 139: 103,
    140: 103, 141: 103, 142: 103, 143: 104, 144: 104, 145: 103, 146: 104,
    147: 103, 158: 104,
}
TEAM_DIVISION = {
    110: 201, 111: 201, 147: 201, 139: 201, 141: 201,
    114: 202, 116: 202, 118: 202, 142: 202, 145: 202,
    108: 203, 117: 203, 133: 203, 136: 203, 140: 203,
    144: 204, 146: 204, 121: 204, 143: 204, 120: 204,
    112: 205, 113: 205, 158: 205, 134: 205, 138: 205,
    109: 206, 115: 206, 119: 206, 135: 206, 137: 206,
}

df["_h_lg"] = df["home_team_id"].map(TEAM_LEAGUE)
df["_a_lg"] = df["away_team_id"].map(TEAM_LEAGUE)
df["_h_div"] = df["home_team_id"].map(TEAM_DIVISION)
df["_a_div"] = df["away_team_id"].map(TEAM_DIVISION)
df["_same_lg"] = (df["_h_lg"] == df["_a_lg"]).astype(int)
df["_same_div"] = ((df["_h_div"] == df["_a_div"]) & (df["_h_lg"] == df["_a_lg"])).astype(int)

conditions = [df["_same_div"] == 1, (df["_same_lg"] == 1) & (df["_same_div"] == 0)]
df["_ctx"] = np.select(conditions, ["intra_div", "intra_lg"], default="interleague")

print("=" * 72)
print("MODEL-FREE SINGLE-FEATURE AUC BY SCHEDULE CONTEXT")
print("=" * 72)
print()
print("Matchup distribution:")
for ctx, cnt in df["_ctx"].value_counts().items():
    print(f"  {ctx}: {cnt:,}")
print()

features = [
    "elo_prob", "elo_diff", "home_elo",
    "home_roll10_winpct", "away_roll10_winpct",
    "home_win_streak",
    "home_roll10_avg", "away_roll10_avg",
    "h2h_home_winrate_10", "h2h_rd_mean_10",
    "park_factor",
    "home_roll10_era", "away_roll10_era",
    "sp_era_diff",
]
features = [f for f in features if f in df.columns]
y = df["home_win"].values

print(f"{'Feature':<28} {'IntraDiv':>9} {'IntraLg':>9} {'InterLg':>9} {'Gap(D-I)':>9} {'N_inter':>8}")
print("-" * 73)

for feat in features:
    vals = df[feat].values.astype(float)
    results = {}
    ns = {}
    for ctx in ["intra_div", "intra_lg", "interleague"]:
        mask = (df["_ctx"].values == ctx) & np.isfinite(vals)
        n = mask.sum()
        ns[ctx] = n
        if n < 100:
            results[ctx] = float("nan")
            continue
        y_sub = y[mask]
        v_sub = vals[mask]
        if len(set(y_sub)) < 2:
            results[ctx] = float("nan")
            continue
        results[ctx] = roc_auc_score(y_sub, v_sub)
    d = results.get("intra_div", 0.5)
    i_val = results.get("interleague", 0.5)
    if np.isnan(d) or np.isnan(i_val):
        gap_str = "     N/A"
    else:
        gap_str = f"{d - i_val:>+9.4f}"
    print(
        f"{feat:<28} "
        f"{results.get('intra_div', float('nan')):>9.4f} "
        f"{results.get('intra_lg', float('nan')):>9.4f} "
        f"{results.get('interleague', float('nan')):>9.4f} "
        f"{gap_str} "
        f"{ns.get('interleague', 0):>8,}"
    )

print()
print("=" * 72)
print("H2H SIGNAL CHECK")
print("=" * 72)
if "h2h_home_winrate_10" in df.columns:
    valid = df[["h2h_home_winrate_10", "home_win"]].dropna()
    r = valid["h2h_home_winrate_10"].corr(valid["home_win"])
    print(f"  Pearson r with outcome: {r:.4f} (n={len(valid):,})")
    print(f"  Non-null fraction: {len(valid)/len(df):.1%}")
    print(f"  AUC (all contexts): {roc_auc_score(valid['home_win'], valid['h2h_home_winrate_10']):.4f}")
print()
print("Interpretation:")
print("  AUC=0.50 = no discrimination (random)")
print("  AUC>0.55 = meaningful single-feature signal")
print("  Gap>0.02 = schedule context materially degrades feature")
