"""
Empirical Bayes Adaptive Shrinkage for Feature Importance Classification.

Framework: ash (Stephens 2017, "False discovery rates, a new deal")
Applied to: LOYO-CV permutation importance (MDA) and desubstituted MDA (DESUB)

Model:
    x̄_f | μ_f ~ N(μ_f, SE²_f)         [likelihood — 8-fold sample mean]
    μ_f ~ g = π₀δ(0) + Σ_k π_k N(0, σ²_k)  [prior — spike-and-slab normal mixture]

Key outputs per feature:
    - Posterior mean (shrunk estimate of true importance)
    - P(μ_f = 0 | data)  — lfdr (local false discovery rate)
    - P(μ_f < 0 | data)  — posterior probability of harmful
    - P(μ_f > 0 | data)  — posterior probability of useful

Classification:
    HARMFUL:  P(μ_f < 0 | data) > 0.95 from BOTH MDA and DESUB
    NOISE:    Not harmful, and max(P_mda(>0), P_desub(>0)) < 0.80
    USEFUL:   Everything else

LOYO correlation adjustment:
    With K=8 Leave-One-Year-Out folds sharing ~7/8 of training data,
    fold estimates are positively correlated. Naive SE = sd/√8 underestimates
    true uncertainty. We detect this via the empirical null (Efron 2004):
    if z-scores are overdispersed (σ₀ > 1), inflate SEs by σ₀.
"""

import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data/logs/eb_shrinkage.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

TARGETS = [
    "home_win", "yrfi", "first_5_home_win", "extra_innings",
    "home_run_diff", "total_runs", "home_runs", "away_runs",
    "first_5_home_run_diff", "first_5_total_runs",
]

K_FOLDS = 8
HARMFUL_THRESHOLD = 0.95
USEFUL_THRESHOLD = 0.80
BASE_DIR = "data/importance"


def build_sigma_grid(x_bar, se, n_grid=30):
    """
    Build a geometric grid of prior standard deviations.
    Range: from smallest detectable effect to largest observed effect.
    Grid includes 0 (point mass = null hypothesis).
    """
    abs_effects = np.abs(x_bar)
    nonzero_mask = abs_effects > 0
    if nonzero_mask.sum() == 0:
        return np.array([0.0] + list(np.logspace(-8, -3, n_grid)))

    min_effect = np.percentile(abs_effects[nonzero_mask], 5)
    max_effect = np.percentile(abs_effects[nonzero_mask], 99)

    if min_effect <= 0:
        min_effect = 1e-8
    if max_effect <= min_effect:
        max_effect = min_effect * 100

    sigma_grid = np.logspace(
        np.log10(min_effect / 2),
        np.log10(max_effect * 2),
        n_grid,
    )
    # Prepend 0 for the point mass (null)
    return np.concatenate([[0.0], sigma_grid])


def log_marginal_likelihood_component(x_bar, se, sigma_k):
    """
    Log N(x̄_f; 0, σ²_k + SE²_f) for one grid component.
    When σ_k = 0 (null): this is log N(x̄_f; 0, SE²_f).
    """
    total_var = sigma_k**2 + se**2
    return norm.logpdf(x_bar, loc=0, scale=np.sqrt(total_var))


def fit_ash_em(x_bar, se, sigma_grid, max_iter=2000, tol=1e-8):
    """
    Fit the ash model via EM algorithm.

    Maximize: Σ_f log(Σ_k π_k · N(x̄_f; 0, σ²_k + SE²_f))
    Subject to: π_k ≥ 0, Σ π_k = 1

    Returns: π (mixing weights), log-likelihood trajectory
    """
    n_features = len(x_bar)
    n_components = len(sigma_grid)

    # Precompute log-likelihood matrix: L[f, k] = log N(x̄_f; 0, σ²_k + SE²_f)
    log_lik = np.zeros((n_features, n_components))
    for k, sigma_k in enumerate(sigma_grid):
        log_lik[:, k] = log_marginal_likelihood_component(x_bar, se, sigma_k)

    # Initialize π uniformly
    pi = np.ones(n_components) / n_components
    ll_history = []

    for iteration in range(max_iter):
        # E-step: compute responsibilities
        # w[f, k] ∝ π_k · exp(log_lik[f, k])
        log_numerator = np.log(pi + 1e-300) + log_lik  # (F, K)
        log_denominator = np.logaddexp.reduce(log_numerator, axis=1, keepdims=True)
        log_w = log_numerator - log_denominator
        w = np.exp(log_w)

        # Marginal log-likelihood
        ll = np.sum(log_denominator)
        ll_history.append(ll)

        if iteration > 0 and abs(ll - ll_history[-2]) < tol:
            log.debug(f"EM converged at iteration {iteration}, LL={ll:.4f}")
            break

        # M-step: update π
        pi = w.mean(axis=0)
        pi = np.maximum(pi, 1e-10)
        pi /= pi.sum()

    return pi, w, ll_history


def compute_posteriors(x_bar, se, sigma_grid, pi, w):
    """
    Compute posterior quantities for each feature.

    For component k (k > 0, i.e., non-null):
        Posterior of μ_f | (x̄_f, component k):
            mean:  m_{f,k} = (σ²_k / (σ²_k + SE²_f)) · x̄_f
            var:   v_{f,k} = σ²_k · SE²_f / (σ²_k + SE²_f)

    For component k=0 (null): μ_f = 0 with certainty.

    Returns dict with:
        posterior_mean: shrunk estimate of μ_f
        p_null: P(μ_f = 0 | data) = lfdr
        p_negative: P(μ_f < 0 | data)
        p_positive: P(μ_f > 0 | data)
    """
    n_features = len(x_bar)
    n_components = len(sigma_grid)

    posterior_mean = np.zeros(n_features)
    p_null = w[:, 0].copy()  # weight on the point mass at 0
    p_negative = np.zeros(n_features)
    p_positive = np.zeros(n_features)

    for k in range(n_components):
        sigma_k = sigma_grid[k]
        if sigma_k == 0:
            # Null component: μ_f = 0, contributes 0 to negative/positive
            # (0 is neither < 0 nor > 0)
            continue

        # Posterior parameters within this component
        shrinkage = sigma_k**2 / (sigma_k**2 + se**2)
        m_fk = shrinkage * x_bar  # posterior mean
        v_fk = sigma_k**2 * se**2 / (sigma_k**2 + se**2)  # posterior var
        sd_fk = np.sqrt(v_fk)

        # Contribution to posterior mean
        posterior_mean += w[:, k] * m_fk

        # P(μ_f < 0 | component k) = Φ(-m_{f,k} / sd_{f,k})
        p_neg_k = norm.cdf(0, loc=m_fk, scale=sd_fk)
        p_pos_k = 1 - p_neg_k

        p_negative += w[:, k] * p_neg_k
        p_positive += w[:, k] * p_pos_k

    return {
        "posterior_mean": posterior_mean,
        "p_null": p_null,
        "p_negative": p_negative,
        "p_positive": p_positive,
    }


def estimate_empirical_null(z_scores, z_cutoff=2.0):
    """
    Efron (2004) empirical null estimation.
    Fit N(δ₀, σ₀²) to central z-scores (|z| < z_cutoff).

    If σ₀ > 1, fold estimates are overdispersed (LOYO correlation).
    Returns (delta_0, sigma_0).
    """
    central = z_scores[np.abs(z_scores) < z_cutoff]
    if len(central) < 20:
        return 0.0, 1.0

    delta_0 = np.median(central)
    sigma_0 = np.median(np.abs(central - delta_0)) / 0.6745  # MAD estimator (robust)
    return delta_0, sigma_0


def run_eb_for_method(raw_csv_path, method_name):
    """
    Run EB adaptive shrinkage on one method's raw fold data.

    Steps:
        1. Load 8×F matrix of fold-level importance values
        2. Compute x̄_f and SE_f per feature
        3. Compute z-scores, estimate empirical null
        4. If σ₀ > 1.2, inflate SEs (correct for LOYO overlap)
        5. Fit ash model (EM on spike-and-slab normal mixture)
        6. Compute posteriors
    """
    df = pd.read_csv(raw_csv_path, index_col=0)
    features = df.columns.tolist()
    values = df.values  # (K, F)

    x_bar = values.mean(axis=0)
    x_sd = values.std(axis=0, ddof=1)
    se = x_sd / np.sqrt(K_FOLDS)

    # Handle features with zero variance (constant across folds)
    zero_var_mask = se < 1e-15
    se[zero_var_mask] = np.abs(x_bar[zero_var_mask]) + 1e-15

    # Empirical null calibration
    z_scores = x_bar / se
    delta_0, sigma_0 = estimate_empirical_null(z_scores)
    log.info(
        f"  {method_name} empirical null: δ₀={delta_0:.3f}, σ₀={sigma_0:.3f} "
        f"(inflation factor for LOYO correlation)"
    )

    if sigma_0 > 1.2:
        log.info(
            f"  {method_name}: σ₀={sigma_0:.3f} > 1.2 → inflating SEs by {sigma_0:.3f} "
            f"to correct for LOYO fold correlation"
        )
        se = se * sigma_0

    # Build sigma grid and fit
    sigma_grid = build_sigma_grid(x_bar, se)
    log.debug(f"  Sigma grid: {len(sigma_grid)} components, range [{sigma_grid[1]:.2e}, {sigma_grid[-1]:.2e}]")

    pi, w, ll_history = fit_ash_em(x_bar, se, sigma_grid)

    # Posterior computation
    posteriors = compute_posteriors(x_bar, se, sigma_grid, pi, w)

    # Null proportion
    pi_null = pi[0]
    log.info(f"  {method_name}: π₀ (null proportion) = {pi_null:.3f}, "
             f"converged in {len(ll_history)} iterations")

    return pd.DataFrame({
        "feature": features,
        "x_bar": x_bar,
        "se": se,
        "sigma_0": sigma_0,
        "posterior_mean": posteriors["posterior_mean"],
        "p_null": posteriors["p_null"],
        "p_negative": posteriors["p_negative"],
        "p_positive": posteriors["p_positive"],
        "n_positive_folds": (values > 0).sum(axis=0),
    }).set_index("feature")


def classify_features(mda_results, desub_results):
    """
    Classify features using BOTH MDA and DESUB posteriors.

    HARMFUL: P(μ < 0) > 0.95 from BOTH methods
        Interpretation: strong evidence from two independent estimators
        that including this feature degrades out-of-sample performance.

    NOISE: Not harmful, and neither method shows P(μ > 0) > 0.80
        Interpretation: no credible evidence of positive contribution,
        but not definitively harmful either. The posterior is centered
        near zero or has too much uncertainty.

    USEFUL: Everything else (at least one method shows P(μ > 0) > 0.80)
    """
    # Align on common features
    common = mda_results.index.intersection(desub_results.index)
    mda = mda_results.loc[common]
    desub = desub_results.loc[common]

    classifications = []
    for feat in common:
        mda_p_neg = mda.loc[feat, "p_negative"]
        desub_p_neg = desub.loc[feat, "p_negative"]
        mda_p_pos = mda.loc[feat, "p_positive"]
        desub_p_pos = desub.loc[feat, "p_positive"]

        if mda_p_neg > HARMFUL_THRESHOLD and desub_p_neg > HARMFUL_THRESHOLD:
            label = "HARMFUL"
        elif max(mda_p_pos, desub_p_pos) < USEFUL_THRESHOLD:
            label = "NOISE"
        else:
            label = "USEFUL"

        classifications.append({
            "feature": feat,
            "classification": label,
            "mda_posterior_mean": mda.loc[feat, "posterior_mean"],
            "desub_posterior_mean": desub.loc[feat, "posterior_mean"],
            "mda_p_negative": mda_p_neg,
            "desub_p_negative": desub_p_neg,
            "mda_p_positive": mda_p_pos,
            "desub_p_positive": desub_p_pos,
            "mda_p_null": mda.loc[feat, "p_null"],
            "desub_p_null": desub.loc[feat, "p_null"],
            "mda_x_bar": mda.loc[feat, "x_bar"],
            "desub_x_bar": desub.loc[feat, "x_bar"],
            "mda_n_positive_folds": int(mda.loc[feat, "n_positive_folds"]),
            "desub_n_positive_folds": int(desub.loc[feat, "n_positive_folds"]),
        })

    return pd.DataFrame(classifications).set_index("feature")


def format_table(df_classified, target):
    """Format results as a box-drawing table for one target."""
    harmful = df_classified[df_classified["classification"] == "HARMFUL"].sort_values("mda_p_negative", ascending=False)
    noise = df_classified[df_classified["classification"] == "NOISE"].sort_values("mda_p_negative", ascending=False)
    blocked = pd.concat([harmful, noise])

    if len(blocked) == 0:
        return f"TARGET: {target.upper()} — No features blocked (all have credible positive signal)\n"

    lines = []
    lines.append(f"TARGET: {target.upper()} — Block {len(blocked)} features "
                 f"({len(harmful)} harmful + {len(noise)} noise)")
    lines.append("┌──────────┬─────────────────────────────────┬───────────┬───────────┬──────────┬──────────┬──────────┬──────────┬──────┬──────┐")
    lines.append("│   Class  │Feature                          │ MDA_μ_post│DSUB_μ_post│ P(μ<0)   │ P(μ<0)   │ P(μ=0)   │ P(μ=0)   │ MDA+ │DSUB+ │")
    lines.append("│          │                                 │  (shrunk) │  (shrunk) │   MDA    │   DESUB  │   MDA    │   DESUB  │  /8  │  /8  │")
    lines.append("├──────────┼─────────────────────────────────┼───────────┼───────────┼──────────┼──────────┼──────────┼──────────┼──────┼──────┤")

    for feat, row in blocked.iterrows():
        cls = row["classification"]
        mda_pm = row["mda_posterior_mean"]
        desub_pm = row["desub_posterior_mean"]
        mda_pn = row["mda_p_negative"]
        desub_pn = row["desub_p_negative"]
        mda_p0 = row["mda_p_null"]
        desub_p0 = row["desub_p_null"]
        mda_folds = int(row["mda_n_positive_folds"])
        desub_folds = int(row["desub_n_positive_folds"])

        feat_str = feat[:33].ljust(33)
        lines.append(
            f"│ {cls:8s} │{feat_str}│ {mda_pm:9.2e} │ {desub_pm:9.2e} │ {mda_pn:8.4f} │ {desub_pn:8.4f} │ {mda_p0:8.4f} │ {desub_p0:8.4f} │  {mda_folds}/8 │  {desub_folds}/8 │"
        )

    lines.append("└──────────┴─────────────────────────────────┴───────────┴───────────┴──────────┴──────────┴──────────┴──────────┴──────┴──────┘")
    return "\n".join(lines)


def main():
    log.info("=" * 70)
    log.info("Empirical Bayes Adaptive Shrinkage — Feature Importance Classification")
    log.info("=" * 70)
    log.info(f"Model: x̄_f | μ_f ~ N(μ_f, SE²_f)")
    log.info(f"Prior: μ_f ~ π₀·δ(0) + Σ_k π_k·N(0, σ²_k)")
    log.info(f"Estimation: EM on marginal likelihood (convex)")
    log.info(f"Correlation correction: Efron empirical null (inflate SE if σ₀ > 1.2)")
    log.info(f"HARMFUL threshold: P(μ<0) > {HARMFUL_THRESHOLD} from BOTH MDA and DESUB")
    log.info(f"USEFUL threshold: P(μ>0) > {USEFUL_THRESHOLD} from at least one method")
    log.info("")

    all_results = {}
    all_tables = []

    for target in TARGETS:
        log.info(f"{'─'*50}")
        log.info(f"TARGET: {target}")
        log.info(f"{'─'*50}")

        mda_path = f"{BASE_DIR}/{target}/importance_mda_raw.csv"
        desub_path = f"{BASE_DIR}/{target}/importance_desub_mda_raw.csv"

        mda_results = run_eb_for_method(mda_path, "MDA")
        desub_results = run_eb_for_method(desub_path, "DESUB")

        classified = classify_features(mda_results, desub_results)

        n_harmful = (classified["classification"] == "HARMFUL").sum()
        n_noise = (classified["classification"] == "NOISE").sum()
        n_useful = (classified["classification"] == "USEFUL").sum()
        log.info(f"  Classification: {n_harmful} harmful, {n_noise} noise, {n_useful} useful")

        all_results[target] = {
            "harmful": classified[classified["classification"] == "HARMFUL"].index.tolist(),
            "noise": classified[classified["classification"] == "NOISE"].index.tolist(),
            "n_harmful": int(n_harmful),
            "n_noise": int(n_noise),
            "n_useful": int(n_useful),
            "n_total": len(classified),
        }

        table = format_table(classified, target)
        all_tables.append(table)
        print()
        print(table)
        print()

    # Save machine-readable output
    output_path = f"{BASE_DIR}/eb_blocklists.json"
    # Simplify for JSON
    json_output = {}
    for target, res in all_results.items():
        json_output[target] = {
            "harmful": res["harmful"],
            "noise": res["noise"],
            "block_all": res["harmful"] + res["noise"],
            "n_harmful": res["n_harmful"],
            "n_noise": res["n_noise"],
            "n_blocked": res["n_harmful"] + res["n_noise"],
            "n_useful": res["n_useful"],
        }
    with open(output_path, "w") as f:
        json.dump(json_output, f, indent=2)
    log.info(f"\nSaved blocklists to {output_path}")

    # Save tables
    tables_path = f"{BASE_DIR}/eb_blocklist_tables.txt"
    with open(tables_path, "w") as f:
        f.write("EMPIRICAL BAYES ADAPTIVE SHRINKAGE — FEATURE BLOCKLISTS\n")
        f.write("=" * 70 + "\n\n")
        f.write("Model: x̄_f | μ_f ~ N(μ_f, SE²_f)\n")
        f.write("Prior: μ_f ~ π₀·δ(0) + Σ_k π_k·N(0, σ²_k)  [ash, Stephens 2017]\n")
        f.write(f"HARMFUL: P(μ<0|data) > {HARMFUL_THRESHOLD} from BOTH MDA and DESUB\n")
        f.write(f"NOISE: Not harmful, max P(μ>0|data) < {USEFUL_THRESHOLD}\n")
        f.write("SE correction: Efron (2004) empirical null for LOYO correlation\n\n")
        for table in all_tables:
            f.write(table + "\n\n")
    log.info(f"Saved tables to {tables_path}")

    # Summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    for target in TARGETS:
        r = all_results[target]
        log.info(f"  {target:25s}: {r['n_harmful']:3d} harmful + {r['n_noise']:3d} noise = "
                 f"{r['n_harmful']+r['n_noise']:3d} blocked / {r['n_total']}")


if __name__ == "__main__":
    main()
