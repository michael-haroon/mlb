import pandas as pd
import numpy as np
from scipy import stats

print("=" * 120)
print("PCA vs IMPORTANCE ALIGNMENT ANALYSIS")
print("=" * 120)
print()
print("Do the features that tree models find important correspond to the directions PCA finds important?")
print("High tau = importance aligns with variance structure. Low tau = importance is orthogonal to variance.")
print()

targets = ['home_win', 'yrfi', 'home_run_diff', 'total_runs', 'first_5_total_runs', 'extra_innings']

for tgt in targets:
    pca = pd.read_csv(f'/tmp/ratings_deep/pca_{tgt}.csv', index_col=0)

    n_pcs = len(pca)

    tau_mdi, p_mdi = stats.kendalltau(pca['eigenvalue_rank'], pca['mdi_rank'])
    tau_mda, p_mda = stats.kendalltau(pca['eigenvalue_rank'], pca['mda_rank'])
    tau_sfi, p_sfi = stats.kendalltau(pca['eigenvalue_rank'], pca['sfi_rank'])

    total_mda = pca['mda_importance'].sum()
    top3_mda = pca.nlargest(3, 'mda_importance')['mda_importance'].sum()
    mda_conc = top3_mda / total_mda if abs(total_mda) > 1e-10 else float('nan')

    pc0_mda_rank = int(pca.loc[pca['eigenvalue_rank'] == 1, 'mda_rank'].values[0])
    pc0_mdi_rank = int(pca.loc[pca['eigenvalue_rank'] == 1, 'mdi_rank'].values[0])
    pc0_var = pca.loc[pca['eigenvalue_rank'] == 1, 'explained_variance_ratio'].values[0]

    print(f"  TARGET: {tgt.upper()}")
    print(f"    PC_0 explains {pc0_var:.1%} of variance")
    print(f"    PC_0 MDA rank: #{pc0_mda_rank} | PC_0 MDI rank: #{pc0_mdi_rank}")
    print(f"    Kendall tau (eigenvalue rank vs importance rank):")
    print(f"      MDI: tau={tau_mdi:+.3f} (p={p_mdi:.4f})  MDA: tau={tau_mda:+.3f} (p={p_mda:.4f})  SFI: tau={tau_sfi:+.3f} (p={p_sfi:.4f})")

    if total_mda > 1e-8:
        print(f"    MDA concentration: top-3 PCs hold {mda_conc:.1%} of total MDA importance")
    else:
        neg_count = (pca['mda_importance'] < 0).sum()
        print(f"    MDA total is {'negative' if total_mda < 0 else 'near-zero'} ({neg_count}/{n_pcs} PCs have negative MDA)")

    print(f"    Top-5 PCs by |MDA|:")
    top5 = pca.reindex(pca['mda_importance'].abs().nlargest(5).index)
    for idx, row in top5.iterrows():
        print(f"      {idx}: var_explained={row['explained_variance_ratio']:.4f} (rank {int(row['eigenvalue_rank'])}) | MDA={row['mda_importance']:.2e}")
    print()

print()
print("-" * 120)
print("INTERPRETATION")
print("-" * 120)
print()
print("Strong signal targets (home_win, home_run_diff):")
print("  - High positive tau -> importance aligns with eigenvalue rank")
print("  - PC_0 ranks #1 in importance -> dominant variance direction IS the predictive direction")
print("  - MDA concentrated in top PCs -> tree models exploit the same linear structure PCA finds")
print()
print("Weak signal targets (yrfi, total_runs, first_5_total_runs, extra_innings):")
print("  - tau near zero or negative -> importance NOT aligned with variance")
print("  - PC_0 ranks poorly -> direction with most variance is NOT predictive")
print("  - MDA scattered or negative -> ratings add noise, trees can't find stable signal")
