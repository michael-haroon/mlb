"""Empirical comparison: MAD instability detector vs CI-only path.

Question: Does the MAD check at n=8 (4 values per half) catch anything
that the moderated-t CI would miss? Or does it just produce false positives?

Run: conda run -n pred python tests/test_mad_vs_ci.py
(Or: python3 tests/test_mad_vs_ci.py — needs only numpy)
"""
import numpy as np
from math import gamma, sqrt, log

# EB priors for desub_mda (typical test)
D0 = 5.69
S0_SQ = 6.94e-06
NULL = 0.0
CI_ALPHA = 0.10


def t_ppf(p, df):
    """Approximate t-distribution ppf using Abramowitz & Stegun 26.7.5.
    Good to ~4 decimal places for df > 3."""
    # Normal quantile via rational approximation (Beasley-Springer-Moro)
    from math import log, sqrt
    a = p if p <= 0.5 else 1 - p
    t_val = sqrt(-2 * log(a))
    # Rational approximation constants
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t_val - (c0 + c1*t_val + c2*t_val**2) / (1 + d1*t_val + d2*t_val**2 + d3*t_val**3)
    if p <= 0.5:
        z = -z

    # Cornish-Fisher correction for t with finite df
    g1 = (z**3 + z) / (4 * df)
    g2 = (5*z**5 + 16*z**3 + 3*z) / (96 * df**2)
    return z + g1 + g2


def kendalltau_statistic(x, y):
    """Kendall's tau-b (handles ties)."""
    n = len(x)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i+1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1
    denom = n * (n-1) / 2
    tau = (concordant - discordant) / denom if denom > 0 else 0
    # Approximate p-value for n=8 (variance under null = 2(2n+5)/(9n(n-1)))
    var_tau = 2 * (2*n + 5) / (9 * n * (n - 1))
    z = tau / sqrt(var_tau) if var_tau > 0 else 0
    # Two-tailed p from normal approximation
    p = 2 * (1 - _norm_cdf(abs(z)))
    return tau, p


def _norm_cdf(x):
    """Standard normal CDF approximation."""
    from math import erf
    return 0.5 * (1 + erf(x / sqrt(2)))


def mad_fires(vals, threshold):
    """Check if MAD instability detector would fire at given threshold."""
    mid = len(vals) // 2
    mad_first = np.median(np.abs(vals[:mid] - np.median(vals[:mid])))
    mad_second = np.median(np.abs(vals[mid:] - np.median(vals[mid:])))
    return mad_second > threshold * max(mad_first, 1e-15)


def ci_decision(vals, null=NULL, d0=D0, s0_sq=S0_SQ, ci_alpha=CI_ALPHA):
    """What would the CI decide without MAD override?"""
    n = len(vals)
    level = float(np.median(vals[-3:]))
    d_i = n - 1
    s2_i = float(np.var(vals, ddof=1))
    mod_var = (d_i * s2_i + d0 * s0_sq) / (d_i + d0)
    mod_df = d_i + d0
    se = sqrt(mod_var / n)

    t_crit = t_ppf(1 - ci_alpha / 2, df=mod_df)
    ci_lo = level - t_crit * se
    ci_hi = level + t_crit * se

    # Trend
    tau, p_trend = kendalltau_statistic(np.arange(n), vals)
    trend_sig = p_trend < 0.05

    # Decision cascade (without MAD)
    if trend_sig and tau < 0 and ci_lo <= null:
        return "REJECT", ci_lo, ci_hi, se
    elif trend_sig and tau > 0 and level > null:
        return "ACCEPT", ci_lo, ci_hi, se
    elif not trend_sig:
        if ci_lo > null:
            return "ACCEPT_FLAGGED", ci_lo, ci_hi, se
        elif ci_hi < null:
            return "REJECT", ci_lo, ci_hi, se
        else:
            return "NEEDS_SPECIFICATION", ci_lo, ci_hi, se
    else:
        if ci_lo > null:
            return "ACCEPT_FLAGGED", ci_lo, ci_hi, se
        elif ci_hi < null:
            return "REJECT", ci_lo, ci_hi, se
        else:
            return "NEEDS_SPECIFICATION", ci_lo, ci_hi, se


def main():
    rng = np.random.default_rng(42)
    N_SIM = 50_000
    N_FOLDS = 8

    print("=" * 70)
    print("EXPERIMENT 1: FALSE POSITIVE RATE UNDER NULL (no regime shift)")
    print("=" * 70)
    print()
    print("Simulating stable features (iid draws from same distribution).")
    print("MAD should NOT fire — any firing is a false positive.")
    print()

    for scale_name, sigma in [("tight (σ=0.002)", 0.002),
                               ("medium (σ=0.01)", 0.01),
                               ("wide (σ=0.05)", 0.05)]:
        fp_3x = 0
        fp_5x = 0
        fp_7x = 0
        for _ in range(N_SIM):
            vals = rng.normal(0.005, sigma, size=N_FOLDS)
            if mad_fires(vals, 3.0):
                fp_3x += 1
            if mad_fires(vals, 5.0):
                fp_5x += 1
            if mad_fires(vals, 7.0):
                fp_7x += 1

        print(f"  Scale {scale_name}:")
        print(f"    3x threshold: {fp_3x/N_SIM*100:.1f}% false positive rate")
        print(f"    5x threshold: {fp_5x/N_SIM*100:.1f}% false positive rate")
        print(f"    7x threshold: {fp_7x/N_SIM*100:.1f}% false positive rate")
        print()

    print()
    print("=" * 70)
    print("EXPERIMENT 2: DETECTION POWER — GENUINE REGIME SHIFT")
    print("=" * 70)
    print()
    print("First half: σ_1=0.002, Second half: σ_2 varies. Mean=0.005.")
    print()

    sigma_1 = 0.002
    for sigma_2_mult, sigma_2 in [(3, 0.006), (5, 0.01), (10, 0.02), (20, 0.04)]:
        det_3x = 0
        det_5x = 0
        ci_catches = 0
        ci_misses = 0

        for _ in range(N_SIM):
            first = rng.normal(0.005, sigma_1, size=4)
            second = rng.normal(0.005, sigma_2, size=4)
            vals = np.concatenate([first, second])

            if mad_fires(vals, 3.0):
                det_3x += 1
            if mad_fires(vals, 5.0):
                det_5x += 1

            decision, ci_lo, ci_hi, se = ci_decision(vals)
            if "ACCEPT" in decision:
                ci_misses += 1
            else:
                ci_catches += 1

        print(f"  σ_2 = {sigma_2_mult}x σ_1 (σ_1={sigma_1}, σ_2={sigma_2}):")
        print(f"    MAD 3x detection: {det_3x/N_SIM*100:.1f}%")
        print(f"    MAD 5x detection: {det_5x/N_SIM*100:.1f}%")
        print(f"    CI catches (NEEDS_SPEC or REJECT): {ci_catches/N_SIM*100:.1f}%")
        print(f"    CI misses (still ACCEPTs):         {ci_misses/N_SIM*100:.1f}%")
        print()

    print()
    print("=" * 70)
    print("EXPERIMENT 3: DISAGREEMENT — MAD fires but CI would ACCEPT")
    print("=" * 70)
    print()
    print("The critical question: does MAD catch dangerous cases CI misses?")
    print()

    sigma_1 = 0.002
    scenarios = [
        ("moderate shift (5x σ)", 0.01),
        ("large shift (10x σ)", 0.02),
        ("extreme shift (20x σ)", 0.04),
    ]

    for name, sigma_2 in scenarios:
        mad_fires_ci_accepts = 0
        mad_fires_ci_catches = 0
        mad_silent_ci_accepts = 0
        mad_silent_ci_catches = 0
        total = N_SIM

        for _ in range(total):
            first = rng.normal(0.005, sigma_1, size=4)
            second = rng.normal(0.005, sigma_2, size=4)
            vals = np.concatenate([first, second])

            fired = mad_fires(vals, 3.0)
            decision, ci_lo, ci_hi, se = ci_decision(vals)
            accepts = "ACCEPT" in decision

            if fired and accepts:
                mad_fires_ci_accepts += 1
            elif fired and not accepts:
                mad_fires_ci_catches += 1
            elif not fired and accepts:
                mad_silent_ci_accepts += 1
            else:
                mad_silent_ci_catches += 1

        print(f"  {name}:")
        print(f"    MAD fires + CI accepts (MAD ADDS VALUE): {mad_fires_ci_accepts/total*100:.2f}%")
        print(f"    MAD fires + CI catches (redundant):      {mad_fires_ci_catches/total*100:.2f}%")
        print(f"    MAD silent + CI accepts (both miss):     {mad_silent_ci_accepts/total*100:.2f}%")
        print(f"    MAD silent + CI catches (CI enough):     {mad_silent_ci_catches/total*100:.2f}%")
        print()

    print()
    print("=" * 70)
    print("EXPERIMENT 4: THE DESIGNED-FOR CASE — same mean, explosive variance")
    print("=" * 70)
    print()
    print("Mean stays 0.01 (above null). Only variance changes in second half.")
    print("This is where MAD SHOULD shine — CI might not widen enough due to")
    print("EB shrinkage pulling moderated variance toward the prior.")
    print()

    sigma_1 = 0.001
    for sigma_2 in [0.01, 0.05, 0.10, 0.20]:
        mad_catches_ci_misses = 0
        ci_catches_alone = 0
        both_catch = 0
        neither_catches = 0
        total = N_SIM

        for _ in range(total):
            first = rng.normal(0.01, sigma_1, size=4)
            second = rng.normal(0.01, sigma_2, size=4)
            vals = np.concatenate([first, second])

            fired = mad_fires(vals, 3.0)
            decision, ci_lo, ci_hi, se = ci_decision(vals)
            accepts = "ACCEPT" in decision

            if fired and accepts:
                mad_catches_ci_misses += 1
            elif fired and not accepts:
                both_catch += 1
            elif not fired and not accepts:
                ci_catches_alone += 1
            else:
                neither_catches += 1

        print(f"  σ_1={sigma_1}, σ_2={sigma_2} (mean=0.01 both halves, null=0.0):")
        print(f"    MAD fires + CI misses (MAD UNIQUE VALUE): {mad_catches_ci_misses/total*100:.2f}%")
        print(f"    Both catch:                               {both_catch/total*100:.2f}%")
        print(f"    CI catches alone (MAD missed):            {ci_catches_alone/total*100:.2f}%")
        print(f"    Neither catches:                          {neither_catches/total*100:.2f}%")
        print()

    print()
    print("=" * 70)
    print("EXPERIMENT 5: EB SHRINKAGE — how fast does CI widen?")
    print("=" * 70)
    print()

    d_i = 7  # n-1 for n=8
    prior_weight = D0 / (D0 + d_i)
    sample_weight = d_i / (D0 + d_i)
    print(f"  Prior weight: {prior_weight:.1%}")
    print(f"  Sample weight: {sample_weight:.1%}")
    print()

    t_crit = t_ppf(1 - CI_ALPHA / 2, df=D0 + d_i)
    print(f"  t_crit at df={D0+d_i:.1f}: ~{t_crit:.3f}")
    print()
    print("  sample_var (s²) | moderated_var  | SE        | CI half-width | CI straddles null?")
    print("                  |                |           |               | (for level=0.01)")
    print("  " + "-" * 80)
    for s2_i in [1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
        mod_var = (d_i * s2_i + D0 * S0_SQ) / (d_i + D0)
        se = sqrt(mod_var / 8)
        hw = t_crit * se
        straddles = "YES" if (0.01 - hw) <= 0 else "no"
        print(f"  {s2_i:.1e}         | {mod_var:.4e}    | {se:.4e} | {hw:.5f}       | {straddles}")

    print()
    print("  Key: for typical feature level=0.01, CI straddles null when")
    print("  half-width >= 0.01, which requires sample_var >= ~5e-4.")
    print("  A σ_2=0.02 gives s² ~ 4e-4, which is borderline.")
    print("  A σ_2=0.05 gives s² ~ 2.5e-3, CI clearly straddles → NEEDS_SPEC.")
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("Check the numbers above. The key metric is Experiment 4:")
    print("'MAD fires + CI misses' — this is MAD's unique contribution.")
    print("If this is near-zero, MAD adds no value beyond CI.")
    print("If this is substantial, MAD catches a real gap.")


if __name__ == "__main__":
    main()
