"""
Empirical Bayes Adaptive Shrinkage — RATINGS SUBSET.

Identical methodology to data/importance/eb_adaptive_shrinkage.py but reads
from the ratings-subset importance output path.

Reads: s3://BUCKET/classical_learning/artifacts/importance_ratings/expanding/<target>/
Outputs: data/importance_ratings/eb_blocklists.json
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
        logging.FileHandler("data/logs/eb_shrinkage_ratings.log", mode="w"),
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
BASE_DIR = "data/importance_ratings"


def build_sigma_grid(x_bar, se, n_grid=30):
    """Build a geometric grid of prior standard deviations."""
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
    return np.concatenate([[0.0], sigma_grid])


def log_marginal_likelihood_component(x_bar, se, sigma_k):
    """Log N(x̄_f; 0, σ²_k + SE²_f) for one grid component."""
    total_var = sigma_k**2 + se**2
    return norm.logpdf(x_bar, loc=0, scale=np.sqrt(total_var))


def fit_ash_em(x_bar, se, sigma_grid, max_iter=2000, tol=1e-8):
    """Fit the ash model via EM algorithm."""
    n_features = len(x_bar)
    n_components = len(sigma_grid)

    log_lik = np.zeros((n_features, n_components))
    for k, sigma_k in enumerate(sigma_grid):
        log_lik[:, k] = log_marginal_likelihood_component(x_bar, se, sigma_k)

    pi = np.ones(n_components) / n_components
    ll_history = []

    for iteration in range(max_iter):
        log_numerator = np.log(pi + 1e-300) + log_lik
        log_denominator = np.logaddexp.reduce(log_numerator, axis=1, keepdims=True)
        log_w = log_numerator - log_denominator
        w = np.exp(log_w)

        ll = np.sum(log_denominator)
        ll_history.append(ll)

        if iteration > 0 and abs(ll - ll_history[-2]) < tol:
            log.debug(f"EM converged at iteration {iteration}, LL={ll:.4f}")
            break

        pi = w.mean(axis=0)
        pi = np.maximum(pi, 1e-10)
        pi /= pi.sum()

    return pi, w, ll_history


def compute_posteriors(x_bar, se, sigma_grid, pi, w):
    """Compute posterior quantities for each feature."""
    n_features = len(x_bar)
    n_components = len(sigma_grid)

    posterior_mean = np.zeros(n_features)
    p_null = w[:, 0].copy()
    p_negative = np.zeros(n_features)
    p_positive = np.zeros(n_features)

    for k in range(n_components):
        sigma_k = sigma_grid[k]
        if sigma_k == 0:
            continue

        shrinkage = sigma_k**2 / (sigma_k**2 + se**2)
        m_fk = shrinkage * x_bar
        v_fk = sigma_k**2 * se**2 / (sigma_k**2 + se**2)
        sd_fk = np.sqrt(v_fk)

        posterior_mean += w[:, k] * m_fk

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
    """Efron (2004) empirical null estimation."""
    central = z_scores[np.abs(z_scores) < z_cutoff]
    if len(central) < 20:
        return 0.0, 1.0

    delta_0 = np.median(central)
    sigma_0 = np.median(np.abs(central - delta_0)) / 0.6745
    return delta_0, sigma_0


def run_eb_for_method(raw_csv_path, method_name):
    """Run EB adaptive shrinkage on one method's raw fold data."""
    df = pd.read_csv(raw_csv_path, index_col=0)
    features = df.columns.tolist()
    values = df.values

    x_bar = values.mean(axis=0)
    x_sd = values.std(axis=0, ddof=1)
    se = x_sd / np.sqrt(K_FOLDS)

    zero_var_mask = se < 1e-15
    se[zero_var_mask] = np.abs(x_bar[zero_var_mask]) + 1e-15

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

    sigma_grid = build_sigma_grid(x_bar, se)
    pi, w, ll_history = fit_ash_em(x_bar, se, sigma_grid)
    posteriors = compute_posteriors(x_bar, se, sigma_grid, pi, w)

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
    """Classify features using BOTH MDA and DESUB posteriors."""
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


def run_eb_pipeline(base_dir, cv_mode_label):
    """Run full EB pipeline on one CV mode's results."""
    log.info(f"\n{'='*70}")
    log.info(f"Empirical Bayes Adaptive Shrinkage — RATINGS SUBSET ({cv_mode_label})")
    log.info(f"{'='*70}")
    log.info(f"Model: x̄_f | μ_f ~ N(μ_f, SE²_f)")
    log.info(f"Prior: μ_f ~ π₀·δ(0) + Σ_k π_k·N(0, σ²_k)")
    log.info(f"HARMFUL threshold: P(μ<0) > {HARMFUL_THRESHOLD} from BOTH MDA and DESUB")
    log.info(f"USEFUL threshold: P(μ>0) > {USEFUL_THRESHOLD} from at least one method")
    log.info("")

    all_results = {}
    all_tables = []

    for target in TARGETS:
        log.info(f"{'─'*50}")
        log.info(f"TARGET: {target}")
        log.info(f"{'─'*50}")

        mda_path = f"{base_dir}/{target}/importance_mda_raw.csv"
        desub_path = f"{base_dir}/{target}/importance_desub_mda_raw.csv"

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
    output_path = f"{base_dir}/eb_blocklists.json"
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
    tables_path = f"{base_dir}/eb_blocklist_tables.txt"
    with open(tables_path, "w") as f:
        f.write(f"EMPIRICAL BAYES ADAPTIVE SHRINKAGE — RATINGS SUBSET ({cv_mode_label})\n")
        f.write("=" * 70 + "\n\n")
        f.write("Features: elo, massey, colley, wolfe, pythag, srs, log5, consensus\n")
        f.write("+ all interactions containing at least one rating system component\n\n")
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
    log.info(f"SUMMARY — RATINGS SUBSET ({cv_mode_label})")
    log.info("=" * 70)
    for target in TARGETS:
        r = all_results[target]
        log.info(f"  {target:25s}: {r['n_harmful']:3d} harmful + {r['n_noise']:3d} noise = "
                 f"{r['n_harmful']+r['n_noise']:3d} blocked / {r['n_total']}")

    return json_output


def main():
    # Run EB on expanding (all prior years) folds
    expanding_results = run_eb_pipeline(BASE_DIR, "expanding")

    # Run EB on sliding_3 (last 3 years only) folds
    sliding3_dir = "data/importance_ratings_sliding3"
    sliding3_results = run_eb_pipeline(sliding3_dir, "sliding_3")

    # Combined summary
    log.info("\n" + "=" * 70)
    log.info("COMPARISON: expanding vs sliding_3")
    log.info("=" * 70)
    for target in TARGETS:
        exp_blocked = expanding_results[target]["n_blocked"]
        sl3_blocked = sliding3_results[target]["n_blocked"]
        exp_total = expanding_results[target]["n_blocked"] + expanding_results[target]["n_useful"]
        log.info(f"  {target:25s}: expanding={exp_blocked} blocked, sliding_3={sl3_blocked} blocked (of {exp_total})")


if __name__ == "__main__":
    main()
