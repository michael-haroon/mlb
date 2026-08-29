"""Visualize EB shrinkage on fold-level importance scores.

Shows concrete examples of features that are:
  - Clearly good (all folds above null, tight CI)
  - Clearly bad (all folds near/below null)
  - Ambiguous (oscillating around null — the hard case)

Run: conda run -n pred python pregame/analysis/view_eb_shrinkage.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import t as t_dist

# ── Config ──────────────────────────────────────────────────────────────────
IMPORTANCE_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "importance" / "home_win"
SFI_RAW = IMPORTANCE_DIR / "importance_sfi_raw.csv"
FEATURE_REPORT = IMPORTANCE_DIR / "filtered" / "feature_report.csv"

# EB priors (validated via homogeneity tests)
D0 = 14.1
S0_SQ = 3.92e-06
NULL = np.log(0.5)  # -0.6931 (coin-flip log-loss)
CI_ALPHA = 0.10


def eb_moderated_score(fold_values: np.ndarray) -> dict:
    """Apply EB shrinkage to a single feature's fold values."""
    vals = np.asarray(fold_values, dtype=np.float64)
    n = len(vals)

    # Level: median of last 3 folds
    level = float(np.median(vals[-3:]))

    # Raw (unmoderated) statistics
    s2_raw = float(np.var(vals, ddof=1))
    se_raw = np.sqrt(s2_raw / n)

    # EB-moderated variance (James-Stein shrinkage on variance)
    d_i = n - 1
    mod_var = (d_i * s2_raw + D0 * S0_SQ) / (d_i + D0)
    mod_df = d_i + D0
    se_mod = np.sqrt(mod_var / n)

    # Shrinkage weight toward prior
    shrinkage_weight = D0 / (d_i + D0)

    # Raw CI (no shrinkage)
    t_crit_raw = t_dist.ppf(1 - CI_ALPHA / 2, df=d_i)
    ci_lo_raw = level - t_crit_raw * se_raw
    ci_hi_raw = level + t_crit_raw * se_raw

    # Moderated CI (with shrinkage)
    t_crit_mod = t_dist.ppf(1 - CI_ALPHA / 2, df=mod_df)
    ci_lo_mod = level - t_crit_mod * se_mod
    ci_hi_mod = level + t_crit_mod * se_mod

    # Decision based on moderated CI
    if ci_lo_mod > NULL:
        decision = "ACCEPT"
    elif ci_hi_mod < NULL:
        decision = "REJECT"
    else:
        decision = "NEEDS_SPECIFICATION"

    return {
        "level": level,
        "null": NULL,
        "distance_from_null": level - NULL,
        "n_folds": n,
        "n_above_null": int(np.sum(vals > NULL)),
        "n_below_null": int(np.sum(vals <= NULL)),
        "raw_var": s2_raw,
        "mod_var": mod_var,
        "shrinkage_weight": shrinkage_weight,
        "variance_reduction_pct": (1 - mod_var / s2_raw) * 100 if s2_raw > 0 else 0,
        "se_raw": se_raw,
        "se_mod": se_mod,
        "ci_raw": (ci_lo_raw, ci_hi_raw),
        "ci_mod": (ci_lo_mod, ci_hi_mod),
        "ci_raw_width": ci_hi_raw - ci_lo_raw,
        "ci_mod_width": ci_hi_mod - ci_lo_mod,
        "raw_excludes_null": ci_lo_raw > NULL or ci_hi_raw < NULL,
        "mod_excludes_null": ci_lo_mod > NULL or ci_hi_mod < NULL,
        "decision": decision,
        "fold_values": vals,
    }


def main():
    # Load data
    sfi_raw = pd.read_csv(SFI_RAW, index_col=0)
    report = pd.read_csv(FEATURE_REPORT, index_col="feature")

    print(f"SFI raw: {sfi_raw.shape[0]} folds × {sfi_raw.shape[1]} features")
    print(f"Null value (coin-flip log-loss): {NULL:.6f}")
    print(f"EB priors: d0={D0}, s0²={S0_SQ:.2e}")
    print(f"Shrinkage weight toward prior: {D0 / (sfi_raw.shape[0] - 1 + D0):.1%}")
    print()

    # Score all features
    results = {}
    for feat in sfi_raw.columns:
        vals = sfi_raw[feat].dropna().values
        if len(vals) >= 4:
            results[feat] = eb_moderated_score(vals)

    # Sort by distance from null (best = most above null)
    sorted_feats = sorted(results.keys(), key=lambda f: results[f]["distance_from_null"], reverse=True)

    # ── CLEARLY GOOD: top features, all/most folds above null ──
    good = [f for f in sorted_feats if results[f]["decision"] == "ACCEPT"]
    # ── CLEARLY BAD: bottom features, CI entirely below null ──
    bad = [f for f in sorted_feats if results[f]["decision"] == "REJECT"]
    # ── AMBIGUOUS: CI straddles null, mixed folds ──
    ambiguous = [f for f in sorted_feats if results[f]["decision"] == "NEEDS_SPECIFICATION"]
    # Among ambiguous, find the most oscillating (closest to 50/50 split)
    ambiguous_by_oscillation = sorted(
        ambiguous,
        key=lambda f: abs(results[f]["n_above_null"] - results[f]["n_below_null"])
    )

    def print_feature(feat, label=""):
        r = results[feat]
        tier = report.loc[feat, "tier"] if feat in report.index else "?"
        print(f"{'─' * 70}")
        print(f"  {label}{feat}")
        print(f"  Final tier in report: {tier}")
        print(f"{'─' * 70}")
        print(f"  Fold values (vs null={NULL:.4f}):")
        for i, v in enumerate(r["fold_values"]):
            marker = "▲" if v > NULL else "▼"
            delta = (v - NULL) * 1e4  # in units of 1e-4
            print(f"    Fold {i}: {v:.6f}  ({marker} {delta:+.2f}×10⁻⁴ from null)")
        print()
        print(f"  Folds above null: {r['n_above_null']}/{r['n_folds']}")
        print(f"  Level (median last 3): {r['level']:.6f} (distance from null: {r['distance_from_null']*1e4:+.2f}×10⁻⁴)")
        print()
        print(f"  ── Before EB shrinkage ──")
        print(f"  Raw variance:    {r['raw_var']:.2e}")
        print(f"  Raw SE:          {r['se_raw']:.2e}")
        print(f"  Raw 90% CI:      [{r['ci_raw'][0]:.6f}, {r['ci_raw'][1]:.6f}]  (width: {r['ci_raw_width']*1e4:.2f}×10⁻⁴)")
        print(f"  Raw excludes null: {r['raw_excludes_null']}")
        print()
        print(f"  ── After EB shrinkage ──")
        print(f"  Moderated var:   {r['mod_var']:.2e}  (shrinkage weight: {r['shrinkage_weight']:.1%})")
        print(f"  Variance change: {r['variance_reduction_pct']:+.1f}%")
        print(f"  Moderated SE:    {r['se_mod']:.2e}")
        print(f"  Mod 90% CI:      [{r['ci_mod'][0]:.6f}, {r['ci_mod'][1]:.6f}]  (width: {r['ci_mod_width']*1e4:.2f}×10⁻⁴)")
        print(f"  Mod excludes null: {r['mod_excludes_null']}")
        print()
        print(f"  Decision: {r['decision']}")
        print()

    # ── Print examples ──────────────────────────────────────────────────────
    print("=" * 70)
    print("  CLEARLY GOOD — EB confirms signal (moderated CI entirely above null)")
    print("=" * 70)
    for feat in good[:3]:
        print_feature(feat, "✓ ")

    print()
    print("=" * 70)
    print("  CLEARLY BAD — EB confirms noise (moderated CI includes or below null)")
    print("=" * 70)
    # Show some that are clearly noise (folds evenly split, near null)
    for feat in bad[:3] if bad else ambiguous[-3:]:
        print_feature(feat, "✗ ")

    print()
    print("=" * 70)
    print("  AMBIGUOUS / OSCILLATING — the hard case you asked about")
    print("=" * 70)
    # Show features where folds genuinely oscillate around null
    shown = 0
    for feat in ambiguous_by_oscillation[:5]:
        r = results[feat]
        # Only show truly oscillating ones (3-5 folds on each side)
        if 3 <= r["n_above_null"] <= 5:
            print_feature(feat, "? ")
            shown += 1
            if shown >= 3:
                break
    # If we didn't find enough perfectly split ones, show the most ambiguous
    if shown < 3:
        for feat in ambiguous_by_oscillation[:3 - shown]:
            print_feature(feat, "? ")

    # ── Summary statistics ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  SUMMARY: EB shrinkage impact across all features")
    print("=" * 70)
    print(f"  Total features scored: {len(results)}")
    print(f"  ACCEPT (mod CI > null):          {len(good):3d} ({100*len(good)/len(results):.1f}%)")
    print(f"  REJECT (mod CI < null):          {len(bad):3d} ({100*len(bad)/len(results):.1f}%)")
    print(f"  NEEDS_SPECIFICATION (straddles): {len(ambiguous):3d} ({100*len(ambiguous)/len(results):.1f}%)")
    print()

    # Show how many features CHANGED decision due to EB
    rescued = [f for f in results if not results[f]["raw_excludes_null"] and results[f]["mod_excludes_null"]]
    demoted = [f for f in results if results[f]["raw_excludes_null"] and not results[f]["mod_excludes_null"]]
    print(f"  Features where EB CHANGED the decision:")
    print(f"    Rescued (raw CI straddles, mod CI excludes null): {len(rescued)}")
    print(f"    Demoted (raw CI excludes, mod CI straddles null): {len(demoted)}")
    print()

    if rescued:
        print(f"  Example RESCUED feature (EB tightened CI enough to clear null):")
        print_feature(rescued[0], "  → ")

    if demoted:
        print(f"  Example DEMOTED feature (EB inflated variance, lost significance):")
        print_feature(demoted[0], "  → ")


if __name__ == "__main__":
    main()
