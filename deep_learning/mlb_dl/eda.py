from __future__ import annotations

import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eda_data(feature_store_path, seasons=None):
    import pandas as pd

    feature_store_path = Path(feature_store_path)
    team_games = pd.read_parquet(feature_store_path / "team_games.parquet")
    game_targets = pd.read_parquet(feature_store_path / "game_targets.parquet")

    if seasons is not None:
        team_games = team_games[team_games["season"].isin(seasons)]
        game_targets = game_targets[game_targets["season"].isin(seasons)]

    for df in (team_games, game_targets):
        num_cols = df.select_dtypes(include=["number"]).columns
        df[num_cols] = df[num_cols].astype("float64")

    log.info("team_games: %d rows, %d cols", len(team_games), len(team_games.columns))
    log.info("game_targets: %d rows, %d cols", len(game_targets), len(game_targets.columns))
    if seasons is not None:
        log.debug("Filtered to seasons: %s", seasons)
    return team_games, game_targets


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------

def classify_columns(team_games, game_targets):
    from .datasets import infer_feature_columns, GAME_TARGET_COLUMNS

    feature_cols = infer_feature_columns(team_games)
    target_cols = [c for c in GAME_TARGET_COLUMNS if c in game_targets.columns]
    log.debug("Feature columns: %d, target columns: %d", len(feature_cols), len(target_cols))
    return {"feature_cols": feature_cols, "target_cols": target_cols}


# ---------------------------------------------------------------------------
# Per-column statistics
# ---------------------------------------------------------------------------

def compute_column_stats(values, col_name):
    import numpy as np
    from scipy import stats as spstats
    from .distributions import suggest_distribution

    arr = np.asarray(values, dtype="float64")
    n_nan = int(np.sum(~np.isfinite(arr)))
    arr = arr[np.isfinite(arr)]
    n = len(arr)

    base = {
        "col_name": col_name,
        "n": n,
        "n_nan": n_nan,
        "pct_nan": round(100.0 * n_nan / max(n + n_nan, 1), 2),
    }

    if n < 3:
        base.update({k: None for k in [
            "pct_zero", "mean", "std", "median", "p5", "p25", "p75", "p95",
            "skewness", "excess_kurtosis", "bimodality_coeff",
            "n_unique", "is_binary", "is_integer_valued", "is_non_negative",
        ]})
        base["suggested_family"] = "insufficient_data"
        return base

    pct_zero = round(100.0 * float(np.sum(arr == 0.0)) / n, 2)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    median = float(np.median(arr))
    p5, p25, p75, p95 = [float(x) for x in np.percentile(arr, [5, 25, 75, 95])]
    skewness = float(spstats.skew(arr))
    excess_kurtosis = float(spstats.kurtosis(arr))  # Fisher (excess) definition

    if std < 1e-10:
        bimodality_coeff = float("nan")
    else:
        pearson_kurtosis = excess_kurtosis + 3.0
        if abs(pearson_kurtosis) < 1e-10:
            bimodality_coeff = float("nan")
        else:
            bimodality_coeff = round((skewness ** 2 + 1.0) / pearson_kurtosis, 4)

    n_unique = int(np.unique(arr).size)
    unique_set = set(np.unique(arr))
    is_binary = unique_set.issubset({0.0, 1.0})
    is_integer_valued = bool(np.allclose(arr, np.round(arr), atol=0.01))
    is_non_negative = bool(arr.min() >= 0.0)

    suggested_family = suggest_distribution(arr, col_name).get("family", "unknown")

    base.update({
        "pct_zero": pct_zero,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "median": round(median, 6),
        "p5": round(p5, 6),
        "p25": round(p25, 6),
        "p75": round(p75, 6),
        "p95": round(p95, 6),
        "skewness": round(skewness, 4),
        "excess_kurtosis": round(excess_kurtosis, 4),
        "bimodality_coeff": bimodality_coeff,
        "n_unique": n_unique,
        "is_binary": is_binary,
        "is_integer_valued": is_integer_valued,
        "is_non_negative": is_non_negative,
        "suggested_family": suggested_family,
    })
    return base


def compute_all_stats(df, columns, label):
    import numpy as np
    import pandas as pd

    rows = []
    for col in columns:
        if col not in df.columns:
            log.debug("Column %s not in dataframe, skipping", col)
            continue
        values = df[col].to_numpy(dtype="float64", na_value=np.nan)
        row = compute_column_stats(values, col)
        row["label"] = label
        rows.append(row)
        log.debug("Stats computed: %s", col)

    log.info("Computed stats for %d %s columns", len(rows), label)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Distribution fitting
# ---------------------------------------------------------------------------

def fit_distributions(values, col_name, stats):
    import warnings
    import json
    import numpy as np
    from scipy import stats as spstats

    if stats.get("is_binary"):
        return {"bernoulli": {"ks_stat": None, "ks_pvalue": None, "params": {}}, "best_fit": "bernoulli", "skipped": True}

    if len(values) < 10:
        return {"best_fit": "insufficient_data"}

    results = {}

    def _ks_continuous(dist_name, params):
        try:
            stat, pval = spstats.kstest(values, dist_name, args=params)
            return {"ks_stat": float(stat), "ks_pvalue": float(pval), "params": list(params)}
        except Exception as e:
            log.debug("KS test failed for %s on %s: %s", dist_name, col_name, e)
            return {"ks_stat": None, "ks_pvalue": 0.0, "params": []}

    # Normal
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            loc, scale = spstats.norm.fit(values)
            results["normal"] = _ks_continuous("norm", (loc, scale))
            results["normal"]["params"] = {"loc": loc, "scale": scale}
        except Exception:
            results["normal"] = {"ks_stat": None, "ks_pvalue": 0.0, "params": {}}

    # Log-Normal (non-negative only)
    if stats.get("is_non_negative"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                shifted = values + 1e-9
                s, loc_ln, scale_ln = spstats.lognorm.fit(shifted, floc=0)
                res = _ks_continuous("lognorm", (s, loc_ln, scale_ln))
                res["params"] = {"s": s, "loc": loc_ln, "scale": scale_ln}
                results["lognormal"] = res
            except Exception:
                results["lognormal"] = {"ks_stat": None, "ks_pvalue": 0.0, "params": {}}
    else:
        results["lognormal"] = None

    # Student-t
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df_fit, loc_t, scale_t = spstats.t.fit(values)
            res = _ks_continuous("t", (df_fit, loc_t, scale_t))
            res["params"] = {"df": df_fit, "loc": loc_t, "scale": scale_t}
            results["student_t"] = res
        except Exception:
            results["student_t"] = {"ks_stat": None, "ks_pvalue": 0.0, "params": {}}

    # Poisson (integer non-negative counts)
    if stats.get("is_integer_valued") and stats.get("is_non_negative"):
        try:
            lam = float(np.mean(values))
            rv = spstats.poisson(lam)
            stat, pval = spstats.kstest(values, rv.cdf)
            results["poisson"] = {"ks_stat": float(stat), "ks_pvalue": float(pval), "params": {"lambda": lam}}
        except Exception:
            results["poisson"] = {"ks_stat": None, "ks_pvalue": 0.0, "params": {}}

        # Negative Binomial (over-dispersed counts)
        mean_ = float(np.mean(values))
        var_ = float(np.var(values))
        if var_ > mean_ * 1.1 and mean_ > 0:
            try:
                p_hat = mean_ / var_
                n_hat = mean_ ** 2 / (var_ - mean_)
                if n_hat > 0 and 0 < p_hat < 1:
                    rv_nb = spstats.nbinom(n_hat, p_hat)
                    stat_nb, pval_nb = spstats.kstest(values, rv_nb.cdf)
                    results["negative_binomial"] = {
                        "ks_stat": float(stat_nb), "ks_pvalue": float(pval_nb),
                        "params": {"n": n_hat, "p": p_hat},
                    }
                else:
                    results["negative_binomial"] = None
            except Exception:
                results["negative_binomial"] = None
        else:
            results["negative_binomial"] = None
    else:
        results["poisson"] = None
        results["negative_binomial"] = None

    # GMM-2 (two-sample KS against fitted GMM samples)
    try:
        from sklearn.mixture import GaussianMixture
        gmm1 = GaussianMixture(n_components=1, random_state=42).fit(values.reshape(-1, 1))
        gmm2 = GaussianMixture(n_components=2, random_state=42).fit(values.reshape(-1, 1))
        gmm_samples, _ = gmm2.sample(min(10000, max(len(values), 200)))
        gmm_samples = gmm_samples.ravel()
        ks_stat_g, ks_pval_g = spstats.ks_2samp(values, gmm_samples)
        results["gmm2"] = {
            "ks_stat": float(ks_stat_g),
            "ks_pvalue": float(ks_pval_g),
            "bic_1comp": float(gmm1.bic(values.reshape(-1, 1))),
            "bic_2comp": float(gmm2.bic(values.reshape(-1, 1))),
            "params": {
                "means": gmm2.means_.flatten().tolist(),
                "covariances": gmm2.covariances_.flatten().tolist(),
                "weights": gmm2.weights_.tolist(),
            },
        }
    except Exception as e:
        log.debug("GMM fitting failed for %s: %s", col_name, e)
        results["gmm2"] = {"ks_stat": None, "ks_pvalue": 0.0, "params": {}}

    # Determine best fit by highest ks_pvalue
    candidate_pvals = {
        fam: res["ks_pvalue"]
        for fam, res in results.items()
        if res is not None and isinstance(res, dict) and res.get("ks_pvalue") is not None
    }
    if candidate_pvals:
        results["best_fit"] = max(candidate_pvals, key=candidate_pvals.get)
    else:
        results["best_fit"] = "unknown"

    return results


def fit_all_distributions(df, columns, stats_df, label, sample_n=100_000):
    import numpy as np
    import pandas as pd
    import json

    stats_lookup = {row["col_name"]: row for row in stats_df.to_dict("records")}

    if len(df) > sample_n:
        df_sample = df.sample(n=sample_n, random_state=42)
    else:
        df_sample = df

    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        values = df_sample[col].dropna().to_numpy(dtype="float64")
        values = values[np.isfinite(values)]
        col_stats = stats_lookup.get(col, {})
        fits = fit_distributions(values, col, col_stats)
        best_fit = fits.pop("best_fit", "unknown")
        skipped = fits.pop("skipped", False)
        for fam, res in fits.items():
            if res is None:
                continue
            rows.append({
                "col_name": col,
                "label": label,
                "family": fam,
                "ks_stat": res.get("ks_stat"),
                "ks_pvalue": res.get("ks_pvalue"),
                "params_json": json.dumps(res.get("params", {})),
                "bic_1comp": res.get("bic_1comp"),
                "bic_2comp": res.get("bic_2comp"),
                "best_fit": best_fit,
                "skipped": skipped,
            })
        log.debug("Distributions fitted: %s (best=%s)", col, best_fit)

    log.info("Fitted distributions for %d %s columns", len(columns), label)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _ensure_agg():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_column_panel(values, col_name, fit_results_df, season_values, output_path):
    import numpy as np
    import warnings
    from scipy import stats as spstats

    plt = _ensure_agg()
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(col_name, fontsize=14, fontweight="bold")

    # --- [0,0] Histogram + KDE + PDFs ---
    ax = axes[0, 0]
    clipped = np.clip(values, np.percentile(values, 1), np.percentile(values, 99))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.histplot(clipped, bins=50, stat="density", alpha=0.4, ax=ax, color="steelblue")
        sns.kdeplot(clipped, ax=ax, linewidth=2, label="KDE", color="navy")

    x_range = np.linspace(clipped.min(), clipped.max(), 300)

    # Overlay normal PDF
    norm_row = fit_results_df[(fit_results_df["col_name"] == col_name) & (fit_results_df["family"] == "normal")]
    if not norm_row.empty and norm_row.iloc[0]["ks_pvalue"] is not None:
        import json
        params = json.loads(norm_row.iloc[0]["params_json"])
        ax.plot(x_range, spstats.norm.pdf(x_range, params["loc"], params["scale"]),
                label=f"Normal (p={norm_row.iloc[0]['ks_pvalue']:.3f})", linestyle="--", color="red", linewidth=1.5)

    # Overlay best alternative PDF if different from normal
    best_fit_vals = fit_results_df[fit_results_df["col_name"] == col_name]["best_fit"]
    best_fit = best_fit_vals.iloc[0] if not best_fit_vals.empty else "normal"
    if best_fit not in ("normal", "bernoulli", "poisson", "negative_binomial", "gmm2", "insufficient_data", "unknown"):
        alt_row = fit_results_df[(fit_results_df["col_name"] == col_name) & (fit_results_df["family"] == best_fit)]
        if not alt_row.empty and alt_row.iloc[0]["ks_pvalue"] is not None:
            import json
            params = json.loads(alt_row.iloc[0]["params_json"])
            try:
                if best_fit == "lognormal":
                    y = spstats.lognorm.pdf(x_range[x_range > 0], params["s"], params["loc"], params["scale"])
                    ax.plot(x_range[x_range > 0], y, label=f"LogNormal (p={alt_row.iloc[0]['ks_pvalue']:.3f})",
                            linestyle="-.", color="green", linewidth=1.5)
                elif best_fit == "student_t":
                    y = spstats.t.pdf(x_range, params["df"], params["loc"], params["scale"])
                    ax.plot(x_range, y, label=f"Student-t (p={alt_row.iloc[0]['ks_pvalue']:.3f})",
                            linestyle="-.", color="purple", linewidth=1.5)
            except Exception:
                pass

    # GMM overlay if bimodal evidence
    gmm_row = fit_results_df[(fit_results_df["col_name"] == col_name) & (fit_results_df["family"] == "gmm2")]
    if not gmm_row.empty:
        import json
        g = gmm_row.iloc[0]
        if g["bic_2comp"] is not None and g["bic_1comp"] is not None:
            bic_delta = g["bic_1comp"] - g["bic_2comp"]
            if bic_delta > 10:
                params = json.loads(g["params_json"])
                means = params.get("means", [])
                covs = params.get("covariances", [])
                weights = params.get("weights", [])
                for i in range(min(2, len(means))):
                    std_i = float(covs[i]) ** 0.5 if covs[i] > 0 else 1e-6
                    y_comp = weights[i] * spstats.norm.pdf(x_range, means[i], std_i)
                    ax.plot(x_range, y_comp, linestyle=":", linewidth=1.2,
                            label=f"GMM comp {i+1} (w={weights[i]:.2f})", alpha=0.8)

    skew_val = fit_results_df[fit_results_df["col_name"] == col_name]
    ax.legend(fontsize=8)
    ax.set_title("Distribution")
    ax.set_xlabel(col_name)

    # --- [0,1] QQ vs Normal ---
    ax = axes[0, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            (osm, osr), (slope, intercept, r) = spstats.probplot(values, dist="norm", plot=None)
            ax.scatter(osm, osr, s=4, alpha=0.4, color="steelblue")
            line_x = np.array([osm.min(), osm.max()])
            ax.plot(line_x, slope * line_x + intercept, color="red", linewidth=1.5)
            ax.text(0.05, 0.95, f"R²={r**2:.4f}", transform=ax.transAxes, va="top", fontsize=9)
        except Exception:
            ax.text(0.5, 0.5, "QQ plot failed", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("QQ Plot vs Normal")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")

    # --- [1,0] QQ vs best-fit alternative ---
    ax = axes[1, 0]
    _plot_alt_qq(ax, values, col_name, best_fit, fit_results_df)

    # --- [1,1] Violin by season ---
    ax = axes[1, 1]
    if season_values:
        import pandas as pd
        long_rows = []
        for season, sv in sorted(season_values.items()):
            for v in sv:
                long_rows.append({"season": str(season), "value": v})
        if long_rows:
            df_long = pd.DataFrame(long_rows)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    sns.violinplot(data=df_long, x="season", y="value", ax=ax, inner="box", cut=0,
                                   color="steelblue", linewidth=0.8)
                    ax.tick_params(axis="x", rotation=45)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Violin failed:\n{e}", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No season data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"{col_name} by Season")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close("all")
    return output_path


def _plot_alt_qq(ax, values, col_name, best_fit, fit_results_df):
    import numpy as np
    import warnings
    from scipy import stats as spstats
    import json

    alt_label = best_fit
    if best_fit in ("normal", "bernoulli", "insufficient_data", "unknown"):
        ax.text(0.5, 0.5, "No alternative distribution\n(best fit is Normal or N/A)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title("QQ Plot vs Best Alternative")
        return

    alt_row = fit_results_df[(fit_results_df["col_name"] == col_name) & (fit_results_df["family"] == best_fit)]

    try:
        if best_fit == "lognormal" and alt_row.empty is False:
            params = json.loads(alt_row.iloc[0]["params_json"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                (osm, osr), (slope, intercept, r) = spstats.probplot(
                    values, dist=spstats.lognorm,
                    sparams=(params["s"], params["loc"], params["scale"]), plot=None
                )
            ax.scatter(osm, osr, s=4, alpha=0.4, color="green")
            line_x = np.array([osm.min(), osm.max()])
            ax.plot(line_x, slope * line_x + intercept, color="red", linewidth=1.5)
            ax.text(0.05, 0.95, f"R²={r**2:.4f}", transform=ax.transAxes, va="top", fontsize=9)

        elif best_fit == "student_t" and not alt_row.empty:
            params = json.loads(alt_row.iloc[0]["params_json"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                (osm, osr), (slope, intercept, r) = spstats.probplot(
                    values, dist=spstats.t,
                    sparams=(params["df"], params["loc"], params["scale"]), plot=None
                )
            ax.scatter(osm, osr, s=4, alpha=0.4, color="purple")
            line_x = np.array([osm.min(), osm.max()])
            ax.plot(line_x, slope * line_x + intercept, color="red", linewidth=1.5)
            ax.text(0.05, 0.95, f"R²={r**2:.4f}", transform=ax.transAxes, va="top", fontsize=9)

        elif best_fit in ("poisson", "negative_binomial"):
            # For discrete: empirical vs theoretical quantiles
            alt_row2 = fit_results_df[(fit_results_df["col_name"] == col_name) & (fit_results_df["family"] == best_fit)]
            if not alt_row2.empty:
                params = json.loads(alt_row2.iloc[0]["params_json"])
                if best_fit == "poisson":
                    rv = spstats.poisson(params["lambda"])
                else:
                    rv = spstats.nbinom(params["n"], params["p"])
                percs = np.linspace(0, 100, min(len(values), 200))
                emp_q = np.percentile(values, percs)
                theo_q = np.array([rv.ppf(p / 100.0) for p in percs])
                ax.scatter(theo_q, emp_q, s=4, alpha=0.4, color="orange")
                lim = max(theo_q.max(), emp_q.max())
                ax.plot([0, lim], [0, lim], color="red", linewidth=1.5)
            alt_label = best_fit

        elif best_fit == "gmm2":
            ax.text(0.5, 0.5, "GMM-2 selected as best fit\n(see histogram panel for components)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
        else:
            ax.text(0.5, 0.5, f"QQ for {best_fit}\nnot implemented",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
    except Exception as e:
        ax.text(0.5, 0.5, f"QQ plot failed:\n{e}", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)

    ax.set_title(f"QQ Plot vs {alt_label}")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")


def plot_all_column_panels(df, columns, fits_df, stats_df, output_dir, label, sample_n=100_000):
    import numpy as np

    stats_lookup = {row["col_name"]: row for row in stats_df.to_dict("records")}
    paths = []

    if len(df) > sample_n:
        df_plot = df.sample(n=sample_n, random_state=42)
    else:
        df_plot = df

    seasons = sorted(df["season"].dropna().unique().astype(int)) if "season" in df.columns else []

    for i, col in enumerate(columns):
        if col not in df.columns:
            continue
        col_stats = stats_lookup.get(col, {})
        if col_stats.get("is_binary"):
            log.debug("Skipping panel for binary column: %s", col)
            continue

        values = df_plot[col].dropna().to_numpy(dtype="float64")
        values = values[np.isfinite(values)]
        if len(values) < 3:
            log.debug("Skipping panel for %s — too few values (%d)", col, len(values))
            continue

        season_values = {}
        if seasons:
            for s in seasons:
                sv = df[df["season"] == s][col].dropna().to_numpy(dtype="float64")
                sv = sv[np.isfinite(sv)]
                if len(sv) >= 2:
                    # Cap per-season to 20k rows for violin
                    if len(sv) > 20_000:
                        rng = np.random.default_rng(42)
                        sv = rng.choice(sv, size=20_000, replace=False)
                    season_values[s] = sv

        output_path = output_dir / label / f"{col}_panel.png"
        try:
            plot_column_panel(values, col, fits_df, season_values, output_path)
            paths.append(output_path)
        except Exception as e:
            log.warning("Panel plot failed for %s: %s", col, e)

        if (i + 1) % 10 == 0:
            log.info("Column panels: %d/%d (%s)", i + 1, len(columns), label)

    log.info("Generated %d %s column panels", len(paths), label)
    return paths


def plot_ks_heatmap(ks_df, output_path, title):
    import pandas as pd
    import numpy as np

    plt = _ensure_agg()
    import matplotlib.pyplot as plt
    import seaborn as sns

    pivot = ks_df.pivot_table(index="col_name", columns="family", values="ks_pvalue", aggfunc="first")
    if pivot.empty:
        log.warning("Empty KS data for heatmap: %s", title)
        return output_path

    # Drop skipped columns (all NaN)
    pivot = pivot.dropna(how="all")
    # Sort: worst max p-value at top
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=True).index]

    n_rows = len(pivot)
    fig_h = max(6, 0.35 * n_rows)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    sns.heatmap(
        pivot.astype(float), vmin=0, vmax=1, cmap="RdYlGn", annot=True, fmt=".2f",
        ax=ax, linewidths=0.3, cbar_kws={"label": "KS p-value"},
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Distribution Family")
    ax.set_ylabel("Column")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close("all")
    return output_path


def plot_bimodality_analysis(stats_df, df_data, fits_df, output_path, top_n=20):
    import numpy as np
    import pandas as pd
    import math
    import json
    from scipy import stats as spstats

    plt = _ensure_agg()
    import matplotlib.pyplot as plt
    import seaborn as sns

    sub = stats_df[~stats_df["is_binary"].fillna(False)].copy()
    sub = sub[sub["bimodality_coeff"].notna() & (sub["bimodality_coeff"] > 0)]
    sub = sub.sort_values("bimodality_coeff", ascending=False).head(top_n)

    cols = sub["col_name"].tolist()
    if not cols:
        log.warning("No bimodal columns found for %s", output_path.name)
        return output_path

    n_cols_grid = 4
    n_rows_grid = math.ceil(len(cols) / n_cols_grid)
    fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(16, n_rows_grid * 3.5))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, col in enumerate(cols):
        ax = axes_flat[idx]
        if col not in df_data.columns:
            ax.set_visible(False)
            continue

        values = df_data[col].dropna().to_numpy(dtype="float64")
        values = values[np.isfinite(values)]
        if len(values) < 3:
            ax.set_visible(False)
            continue

        clipped = np.clip(values, np.percentile(values, 1), np.percentile(values, 99))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.histplot(clipped, bins=40, stat="density", alpha=0.4, ax=ax, color="steelblue")
            sns.kdeplot(clipped, ax=ax, color="navy", linewidth=1.5)

        # Overlay GMM components if BIC improvement
        gmm_row = fits_df[(fits_df["col_name"] == col) & (fits_df["family"] == "gmm2")]
        if not gmm_row.empty:
            g = gmm_row.iloc[0]
            if g["bic_1comp"] is not None and g["bic_2comp"] is not None:
                bic_delta = float(g["bic_1comp"]) - float(g["bic_2comp"])
                params = json.loads(g["params_json"])
                means = params.get("means", [])
                covs = params.get("covariances", [])
                weights = params.get("weights", [])
                colors = ["red", "green"]
                x_range = np.linspace(clipped.min(), clipped.max(), 300)
                for i in range(min(2, len(means))):
                    std_i = float(covs[i]) ** 0.5 if float(covs[i]) > 0 else 1e-6
                    y = weights[i] * spstats.norm.pdf(x_range, means[i], std_i)
                    ax.plot(x_range, y, color=colors[i], linewidth=1.5, linestyle="--",
                            label=f"comp {i+1} μ={means[i]:.1f}")
                if bic_delta > 10:
                    ax.legend(fontsize=7)

        bc = sub[sub["col_name"] == col]["bimodality_coeff"].values[0]
        ax.set_title(f"{col}\nBC={bc:.3f}", fontsize=8)
        ax.tick_params(labelsize=7)

    for i in range(len(cols), len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(f"Bimodality Analysis — Top {len(cols)} Columns by Bimodality Coefficient", fontsize=11)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close("all")
    return output_path


def plot_seasonal_drift(game_targets, target_cols, output_dir):
    import numpy as np
    import pandas as pd
    import math
    import warnings
    from scipy import stats as spstats

    plt = _ensure_agg()
    import matplotlib.pyplot as plt
    import seaborn as sns

    seasons = sorted(game_targets["season"].dropna().unique().astype(int))
    non_binary_targets = [
        c for c in target_cols
        if c in game_targets.columns and game_targets[c].nunique() > 2
    ]

    paths = []

    # Violin plots (6 per figure)
    chunk_size = 6
    violin_paths = []
    for chunk_start in range(0, max(len(non_binary_targets), 1), chunk_size):
        chunk = non_binary_targets[chunk_start: chunk_start + chunk_size]
        if not chunk:
            break
        fig, axes = plt.subplots(len(chunk), 1, figsize=(14, 3.5 * len(chunk)))
        if len(chunk) == 1:
            axes = [axes]
        for ax, col in zip(axes, chunk):
            season_frames = []
            for s in seasons:
                sv = game_targets[game_targets["season"] == s][col].dropna().to_numpy(dtype="float64")
                sv = sv[np.isfinite(sv)]
                if len(sv):
                    season_frames.append(pd.DataFrame({"season": str(s), "value": sv}))
            long_rows = season_frames  # sentinel: non-empty means we have data
            if long_rows:
                df_long = pd.concat(long_rows, ignore_index=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        sns.violinplot(data=df_long, x="season", y="value", ax=ax, inner="box", cut=0,
                                       color="steelblue", linewidth=0.8)
                    except Exception:
                        sns.boxplot(data=df_long, x="season", y="value", ax=ax)
                ax.tick_params(axis="x", rotation=45)
                ax.set_title(col)
        suffix = f"_{chunk_start // chunk_size}" if chunk_start > 0 else ""
        vpath = output_dir / "summary" / f"seasonal_drift_targets{suffix}.png"
        plt.tight_layout()
        vpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(vpath, dpi=100, bbox_inches="tight")
        plt.close("all")
        violin_paths.append(vpath)
    paths.extend(violin_paths)

    # KS distance matrix per target column (seasons × seasons)
    if len(seasons) >= 2 and non_binary_targets:
        n_targets = len(non_binary_targets)
        n_cols_grid = min(3, n_targets)
        n_rows_grid = math.ceil(n_targets / n_cols_grid)
        fig, axes = plt.subplots(n_rows_grid, n_cols_grid,
                                 figsize=(6 * n_cols_grid, 5 * n_rows_grid))
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for idx, col in enumerate(non_binary_targets):
            ax = axes_flat[idx]
            matrix = np.zeros((len(seasons), len(seasons)))
            for i, s1 in enumerate(seasons):
                v1 = game_targets[game_targets["season"] == s1][col].dropna().to_numpy(dtype="float64")
                v1 = v1[np.isfinite(v1)]
                for j, s2 in enumerate(seasons):
                    if i == j:
                        continue
                    v2 = game_targets[game_targets["season"] == s2][col].dropna().to_numpy(dtype="float64")
                    v2 = v2[np.isfinite(v2)]
                    if len(v1) >= 2 and len(v2) >= 2:
                        stat, _ = spstats.ks_2samp(v1, v2)
                        matrix[i, j] = stat
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sns.heatmap(
                    pd.DataFrame(matrix, index=[str(s) for s in seasons], columns=[str(s) for s in seasons]),
                    cmap="Blues_r", vmin=0, vmax=1, ax=ax, annot=True, fmt=".2f",
                    linewidths=0.3, cbar_kws={"label": "KS distance"},
                )
            ax.set_title(f"{col}\nSeason-to-season KS distance", fontsize=9)
            ax.tick_params(labelsize=7)

        for i in range(len(non_binary_targets), len(axes_flat)):
            axes_flat[i].set_visible(False)

        ks_path = output_dir / "summary" / "seasonal_drift_ks_matrix.png"
        plt.tight_layout()
        fig.savefig(ks_path, dpi=100, bbox_inches="tight")
        plt.close("all")
        paths.append(ks_path)

    log.info("Generated %d seasonal drift plots", len(paths))
    return paths


def plot_feature_target_correlations(team_games, game_targets, feature_cols, target_cols, output_path, top_n=50):
    import numpy as np
    import pandas as pd
    import warnings

    plt = _ensure_agg()
    import matplotlib.pyplot as plt
    import seaborn as sns

    home_tg = team_games[team_games.get("side", pd.Series()).eq("home")] if "side" in team_games.columns else team_games
    merged = home_tg.merge(game_targets[["game_pk"] + target_cols], on="game_pk", how="inner")

    available_features = [c for c in feature_cols if c in merged.columns]
    available_targets = [c for c in target_cols if c in merged.columns]

    if not available_features or not available_targets:
        log.warning("No features or targets available for correlation plot")
        return output_path

    # Sample down for speed — Spearman on 300k rows is very slow
    if len(merged) > 20_000:
        merged = merged.sample(n=20_000, random_state=42)

    feat_df = merged[available_features].astype("float64")
    tgt_df = merged[available_targets].astype("float64")
    combined = pd.concat([feat_df, tgt_df], axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_full = combined.corr(method="spearman")

    corr_ft = corr_full.loc[available_features, available_targets]
    max_abs = corr_ft.abs().max(axis=1).sort_values(ascending=False)
    top_features = max_abs.head(top_n).index.tolist()

    corr_plot = corr_ft.loc[top_features]
    fig_h = max(8, len(top_features) * 0.22)
    fig, ax = plt.subplots(figsize=(max(10, len(available_targets) * 0.8 + 2), fig_h))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.heatmap(
            corr_plot.astype(float), cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            ax=ax, annot=False, linewidths=0.2,
            cbar_kws={"label": "Spearman ρ"},
        )
    ax.set_title(f"Feature-Target Spearman Correlations (top {len(top_features)} features)", fontsize=11)
    ax.set_xlabel("Target")
    ax.set_ylabel("Feature")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close("all")
    return output_path


def plot_normalization_comparison(df, columns, stats_df, fits_df, output_path, top_n=12):
    import numpy as np
    import warnings

    plt = _ensure_agg()
    import matplotlib.pyplot as plt
    import seaborn as sns

    stats_lookup = {row["col_name"]: row for row in stats_df.to_dict("records")}

    # Select top-N most non-normal: lowest Normal KS p-value, exclude binary
    norm_pvals = fits_df[(fits_df["family"] == "normal") & (fits_df["ks_pvalue"].notna())].copy()
    norm_pvals = norm_pvals[norm_pvals["col_name"].map(
        lambda c: not stats_lookup.get(c, {}).get("is_binary", True)
    )]
    norm_pvals = norm_pvals.sort_values("ks_pvalue", ascending=True).head(top_n)
    selected_cols = norm_pvals["col_name"].tolist()

    if not selected_cols:
        log.warning("No columns selected for normalization comparison")
        return output_path

    from sklearn.preprocessing import QuantileTransformer

    fig, axes = plt.subplots(len(selected_cols), 3, figsize=(15, len(selected_cols) * 3))
    if len(selected_cols) == 1:
        axes = [axes]

    panel_titles = ["Raw (clipped p1–p99)", "Z-Score (current)", "Log1p / Quantile Transform"]
    bg_colors = ["#f0f0f0", "#ffe0e0", "#e0ffe0"]

    for row_idx, col in enumerate(selected_cols):
        if col not in df.columns:
            for ax in axes[row_idx]:
                ax.set_visible(False)
            continue

        values = df[col].dropna().to_numpy(dtype="float64")
        values = values[np.isfinite(values)]
        if len(values) < 3:
            for ax in axes[row_idx]:
                ax.set_visible(False)
            continue

        clipped = np.clip(values, np.percentile(values, 1), np.percentile(values, 99))
        col_mean = float(np.mean(values))
        col_std = float(np.std(values))
        z_scored = (values - col_mean) / max(col_std, 1e-6)

        col_stats = stats_lookup.get(col, {})
        is_non_neg = col_stats.get("is_non_negative", False)
        if is_non_neg:
            transformed = np.log1p(values)
            transform_label = "Log1p"
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                qt = QuantileTransformer(output_distribution="normal", random_state=42)
                transformed = qt.fit_transform(values.reshape(-1, 1)).ravel()
            transform_label = "Quantile→Normal"

        panels = [clipped, z_scored, transformed]
        ks_pval = norm_pvals[norm_pvals["col_name"] == col]["ks_pvalue"].values
        pval_str = f"(Normal KS p={ks_pval[0]:.4f})" if len(ks_pval) else ""

        for col_idx, (data, bg, title_suffix) in enumerate(zip(panels, bg_colors, panel_titles)):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor(bg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    sns.histplot(data, bins=50, stat="density", alpha=0.6, ax=ax, color="steelblue")
                    sns.kdeplot(data, ax=ax, color="navy", linewidth=1.5)
                except Exception:
                    ax.hist(data, bins=50, density=True, alpha=0.6, color="steelblue")
            if col_idx == 0:
                ax.set_ylabel(f"{col}\n{pval_str}", fontsize=8)
            else:
                ax.set_ylabel("")
            if row_idx == 0:
                ax.set_title(title_suffix if col_idx < 2 else f"{transform_label}\n(alternative)", fontsize=9)
            ax.tick_params(labelsize=7)

    fig.suptitle("Normalization Comparison: why z-score is insufficient", fontsize=12, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close("all")
    return output_path


# ---------------------------------------------------------------------------
# Summary stats CSV
# ---------------------------------------------------------------------------

def save_summary_stats(stats_df, fits_df, output_dir):
    import pandas as pd

    stats_path = output_dir / "stats" / "summary_stats.csv"
    ks_path = output_dir / "stats" / "ks_results.csv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    stats_df.to_csv(stats_path, index=False)
    fits_df.to_csv(ks_path, index=False)

    log.info("Saved summary_stats.csv (%d rows)", len(stats_df))
    log.info("Saved ks_results.csv (%d rows)", len(fits_df))
    return stats_path, ks_path


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def build_html_index(output_dir, stats_df, fits_df, image_paths, generated_at):
    import html as html_mod
    import pandas as pd

    SORTABLE_JS = r"""
<script>
function sortTable(th) {
  var table = th.closest('table');
  var tbody = table.querySelector('tbody');
  var rows = Array.from(tbody.querySelectorAll('tr'));
  var idx = Array.from(th.parentNode.children).indexOf(th);
  var asc = th.dataset.asc !== '1';
  rows.sort(function(a, b) {
    var va = a.children[idx].textContent.trim();
    var vb = b.children[idx].textContent.trim();
    var na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  });
  rows.forEach(function(r) { tbody.appendChild(r); });
  th.dataset.asc = asc ? '1' : '0';
  Array.from(th.parentNode.children).forEach(function(h) { h.style.fontWeight = 'normal'; });
  th.style.fontWeight = 'bold';
}
</script>
"""

    CSS = """
<style>
body { font-family: sans-serif; margin: 20px; color: #222; }
h1 { border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { margin-top: 40px; border-bottom: 1px solid #aaa; }
nav { margin: 10px 0 20px; }
nav a { margin-right: 12px; color: #0066cc; text-decoration: none; }
nav a:hover { text-decoration: underline; }
table { border-collapse: collapse; font-size: 12px; width: 100%; }
th { cursor: pointer; background: #ddd; padding: 5px 8px; border: 1px solid #bbb; user-select: none; }
th:hover { background: #ccc; }
td { padding: 4px 8px; border: 1px solid #ddd; }
tr:nth-child(even) { background: #f9f9f9; }
.skewed { background: #ffe0e0 !important; }
.bimodal { background: #fff0c0 !important; }
.panel-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
.panel-grid a img { max-width: 100%; border: 1px solid #ccc; }
.panel-grid a:hover img { border-color: #0066cc; }
.summary-img { max-width: 100%; margin: 10px 0; border: 1px solid #ccc; }
.insight { background: #fffbe6; border-left: 4px solid #f0ad4e; padding: 10px 14px; margin: 10px 0; font-size: 13px; }
</style>
"""

    def _fmt(v):
        if v is None:
            return ""
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            if v != v:
                return ""
            return f"{v:.4f}"
        return html_mod.escape(str(v))

    # Build stats table rows
    display_cols = [
        "col_name", "label", "n", "pct_nan", "pct_zero",
        "mean", "std", "skewness", "excess_kurtosis", "bimodality_coeff",
        "suggested_family",
    ]

    # Attach best_fit from fits_df
    best_fit_map = {}
    if not fits_df.empty and "best_fit" in fits_df.columns:
        for _, r in fits_df.drop_duplicates("col_name").iterrows():
            best_fit_map[r["col_name"]] = r["best_fit"]

    stats_display = stats_df.copy()
    stats_display["best_fit"] = stats_display["col_name"].map(best_fit_map)
    display_cols_ext = display_cols + ["best_fit"]

    stats_sorted = stats_display.sort_values("bimodality_coeff", ascending=False, na_position="last")

    table_rows = []
    for _, row in stats_sorted.iterrows():
        cells = []
        for c in display_cols_ext:
            val = row.get(c)
            cell_class = ""
            if c == "skewness" and isinstance(val, float) and abs(val) > 1:
                cell_class = ' class="skewed"'
            if c == "bimodality_coeff" and isinstance(val, float) and val > 0.555:
                cell_class = ' class="bimodal"'
            cells.append(f"<td{cell_class}>{_fmt(val)}</td>")
        table_rows.append("<tr>" + "".join(cells) + "</tr>")

    header_cells = "".join(f'<th onclick="sortTable(this)">{c}</th>' for c in display_cols_ext)
    table_html = (
        f"<table><thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody></table>"
    )

    def _img_section(paths, base_dir, heading_id, heading):
        if not paths:
            return f'<h2 id="{heading_id}">{heading}</h2><p>No images generated.</p>'
        items = []
        for p in paths:
            try:
                rel = p.relative_to(base_dir)
            except ValueError:
                rel = p.name
            items.append(f'<a href="{rel}"><img src="{rel}" loading="lazy"></a>')
        return (
            f'<h2 id="{heading_id}">{heading}</h2>'
            f'<div class="panel-grid">{"".join(items)}</div>'
        )

    def _single_img(path, base_dir, label=""):
        if path is None or not Path(path).exists():
            return f"<p>{label}: not generated.</p>"
        try:
            rel = Path(path).relative_to(base_dir)
        except ValueError:
            rel = Path(path).name
        return f'<img src="{rel}" class="summary-img" alt="{label}">'

    summary_dir = output_dir / "summary"
    target_panels = image_paths.get("target_panels", [])
    feature_panels = image_paths.get("feature_panels", [])
    summary_imgs = image_paths.get("summary", {})

    body = f"""
<h1>MLB Feature &amp; Target Distribution Analysis</h1>
<p>Generated: {generated_at}</p>

<nav>
  <a href="#summary">Summary Stats</a>
  <a href="#ks-heatmaps">KS Heatmaps</a>
  <a href="#normalization">Normalization</a>
  <a href="#bimodality">Bimodality</a>
  <a href="#drift">Seasonal Drift</a>
  <a href="#correlations">Correlations</a>
  <a href="#target-panels">Target Panels</a>
  <a href="#feature-panels">Feature Panels</a>
</nav>

<h2 id="summary">Summary Statistics</h2>
<p>Red cells: |skewness| &gt; 1. Yellow cells: bimodality coefficient &gt; 0.555.
Sorted by bimodality coefficient (highest first).</p>
{table_html}

<h2 id="ks-heatmaps">KS Fit Heatmaps</h2>
<p>Higher p-value (greener) = better fit. Red = Normal assumption is poor.</p>
<h3>Targets</h3>
{_single_img(summary_imgs.get("ks_heatmap_targets"), output_dir, "KS heatmap targets")}
<h3>Features</h3>
{_single_img(summary_imgs.get("ks_heatmap_features"), output_dir, "KS heatmap features")}

<h2 id="normalization">Normalization Comparison</h2>
<div class="insight">
  <strong>Why z-score normalization is insufficient:</strong>
  Z-score applies <code>(x - μ) / σ</code> which centers and scales data but
  <em>preserves the distribution shape</em>. Count stats (hits, HR, runs) follow
  Poisson or Negative-Binomial distributions with right skew and zero-inflation —
  after z-scoring they remain right-skewed, not Normal. Bimodal features
  (e.g., pitcher innings: starters ≈ 6 IP, relievers ≈ 1 IP) remain bimodal.
  Log1p transform normalizes right-skewed non-negative data;
  quantile transform works for general non-Normal distributions.
  The plots below show raw (grey), z-scored (red background, shape unchanged),
  and an alternative transform (green background, more Normal).
</div>
{_single_img(summary_imgs.get("normalization_comparison"), output_dir, "Normalization comparison")}

<h2 id="bimodality">Bimodality Analysis — Top-20 by Bimodality Coefficient</h2>
<p>GMM component curves overlaid when 2-component BIC is substantially better than 1-component (ΔBIC &gt; 10).</p>
<h3>Targets</h3>
{_single_img(summary_imgs.get("bimodality_targets"), output_dir, "Bimodality targets")}
<h3>Features</h3>
{_single_img(summary_imgs.get("bimodality_features"), output_dir, "Bimodality features")}

<h2 id="drift">Seasonal Distribution Drift</h2>
<p>KS distance matrix: darker = more similar seasons. Violin plots show per-season spread.</p>
{''.join(_single_img(p, output_dir) for p in image_paths.get("seasonal_drift", []))}

<h2 id="correlations">Feature-Target Spearman Correlations</h2>
{_single_img(summary_imgs.get("feature_target_correlations"), output_dir, "Correlations")}

{_img_section(target_panels, output_dir, "target-panels", "Target Distribution Panels")}
{_img_section(feature_panels, output_dir, "feature-panels", "Feature Distribution Panels")}
"""

    html_content = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>MLB EDA Report</title>{CSS}{SORTABLE_JS}</head><body>{body}</body></html>"

    index_path = output_dir / "index.html"
    index_path.write_text(html_content, encoding="utf-8")
    log.info("HTML report written: %s", index_path)
    return index_path


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_eda(feature_store_path, output_dir, seasons=None, targets_only=False, features_only=False):
    import time
    from datetime import datetime, timezone

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Two-handler logging setup (CLAUDE.md pattern) ---
    from pathlib import Path as _Path
    LOG_DIR = _Path("data/logs")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log.setLevel(logging.DEBUG)
    if not log.handlers:
        _fh = logging.FileHandler(LOG_DIR / "eda.log")
        _fh.setLevel(logging.DEBUG)
        _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(threadName)s %(message)s"))
        log.addHandler(_fh)

        _sh = logging.StreamHandler(sys.stdout)
        _sh.setLevel(logging.INFO)
        _sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
        log.addHandler(_sh)

    # --- Check visualization deps ---
    try:
        import matplotlib
        import seaborn
    except ImportError as exc:
        raise ImportError(
            "matplotlib and seaborn are required for EDA. "
            "Install with: conda run -n pred python -m pip install matplotlib seaborn"
        ) from exc

    # --- Create output subdirs ---
    for subdir in ("targets", "features", "summary", "stats"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log.info("Starting EDA — feature_store=%s, output=%s", feature_store_path, output_dir)

    # --- Load data ---
    team_games, game_targets = load_eda_data(feature_store_path, seasons)
    col_groups = classify_columns(team_games, game_targets)
    feature_cols = col_groups["feature_cols"]
    target_cols = col_groups["target_cols"]

    # --- Stats ---
    import pandas as pd
    stats_parts = []
    if not features_only:
        stats_parts.append(compute_all_stats(game_targets, target_cols, "target"))
    if not targets_only:
        stats_parts.append(compute_all_stats(team_games, feature_cols, "feature"))
    all_stats_df = pd.concat(stats_parts, ignore_index=True)

    # --- Distribution fits ---
    fits_parts = []
    if not features_only:
        fits_parts.append(fit_all_distributions(game_targets, target_cols, all_stats_df[all_stats_df["label"] == "target"], "target"))
    if not targets_only:
        fits_parts.append(fit_all_distributions(team_games, feature_cols, all_stats_df[all_stats_df["label"] == "feature"], "feature"))
    all_fits_df = pd.concat(fits_parts, ignore_index=True) if fits_parts else pd.DataFrame()

    # --- Summary plots ---
    summary_imgs = {}
    log.info("Generating summary plots...")

    if not features_only and not all_fits_df.empty:
        target_fits = all_fits_df[all_fits_df["label"] == "target"]
        if not target_fits.empty:
            p = plot_ks_heatmap(target_fits, output_dir / "summary" / "ks_heatmap_targets.png", "KS Fit Heatmap — Targets")
            summary_imgs["ks_heatmap_targets"] = p

    if not targets_only and not all_fits_df.empty:
        feature_fits = all_fits_df[all_fits_df["label"] == "feature"]
        if not feature_fits.empty:
            p = plot_ks_heatmap(feature_fits, output_dir / "summary" / "ks_heatmap_features.png", "KS Fit Heatmap — Features")
            summary_imgs["ks_heatmap_features"] = p

    target_stats = all_stats_df[all_stats_df["label"] == "target"]
    feature_stats = all_stats_df[all_stats_df["label"] == "feature"]

    if not features_only and not target_stats.empty:
        p = plot_bimodality_analysis(target_stats, game_targets, all_fits_df, output_dir / "summary" / "bimodality_targets.png")
        summary_imgs["bimodality_targets"] = p

    if not targets_only and not feature_stats.empty:
        p = plot_bimodality_analysis(feature_stats, team_games, all_fits_df, output_dir / "summary" / "bimodality_features.png")
        summary_imgs["bimodality_features"] = p

    seasonal_drift_paths = []
    if not features_only and "season" in game_targets.columns:
        seasonal_drift_paths = plot_seasonal_drift(game_targets, target_cols, output_dir)

    if not (targets_only or features_only):
        p = plot_feature_target_correlations(
            team_games, game_targets, feature_cols, target_cols,
            output_dir / "summary" / "feature_target_correlations.png",
        )
        summary_imgs["feature_target_correlations"] = p

    if not all_fits_df.empty:
        norm_df = all_fits_df
        norm_stats = all_stats_df
        if targets_only:
            norm_df = all_fits_df[all_fits_df["label"] == "target"]
            norm_stats = target_stats
            norm_data = game_targets
        elif features_only:
            norm_df = all_fits_df[all_fits_df["label"] == "feature"]
            norm_stats = feature_stats
            norm_data = team_games
        else:
            norm_data = pd.concat([
                game_targets[[c for c in target_cols if c in game_targets.columns]],
                team_games[[c for c in feature_cols if c in team_games.columns]],
            ], axis=1)
        p = plot_normalization_comparison(
            norm_data,
            target_cols + feature_cols if not targets_only and not features_only else (target_cols if targets_only else feature_cols),
            norm_stats, norm_df,
            output_dir / "summary" / "normalization_comparison.png",
        )
        summary_imgs["normalization_comparison"] = p

    # --- Per-column panels ---
    log.info("Generating per-column distribution panels...")
    image_paths = {"summary": summary_imgs, "seasonal_drift": seasonal_drift_paths}

    if not features_only:
        tgt_panels = plot_all_column_panels(game_targets, target_cols, all_fits_df, target_stats, output_dir, "targets")
        image_paths["target_panels"] = tgt_panels
    else:
        image_paths["target_panels"] = []

    if not targets_only:
        feat_panels = plot_all_column_panels(team_games, feature_cols, all_fits_df, feature_stats, output_dir, "features")
        image_paths["feature_panels"] = feat_panels
    else:
        image_paths["feature_panels"] = []

    # --- CSVs ---
    stats_csv, ks_csv = save_summary_stats(all_stats_df, all_fits_df, output_dir)

    # --- HTML index ---
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    index_path = build_html_index(output_dir, all_stats_df, all_fits_df, image_paths, generated_at)

    elapsed = time.time() - t0
    log.info("EDA complete in %.1fs — report: %s", elapsed, index_path)

    return {
        "index_html": str(index_path),
        "summary_stats_csv": str(stats_csv),
        "ks_results_csv": str(ks_csv),
        "target_panels": len(image_paths["target_panels"]),
        "feature_panels": len(image_paths["feature_panels"]),
        "output_dir": str(output_dir),
    }
