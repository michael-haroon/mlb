"""Rigorous AR analysis of per-pitch-type league-average wOBA.

Memory-efficient: processes one season at a time, accumulates weekly stats.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import s3fs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')
import gc

S3_BASE = "mlb-265753586044-us-east-1-an/data"
SEASONS = [y for y in range(2015, 2025) if y != 2020]
TRACKED_PITCH_TYPES = ("FF", "SL", "CH", "CU", "FC", "SI", "FS", "ST")

WOBA_WEIGHTS = {
    "Walk": 0.690,
    "Intent Walk": 0.690,
    "Hit By Pitch": 0.720,
    "Single": 0.880,
    "Double": 1.260,
    "Triple": 1.590,
    "Home Run": 2.080,
}

PA_EVENTS = set(WOBA_WEIGHTS.keys()) | {
    "Strikeout", "Groundout", "Flyout", "Lineout", "Pop Out",
    "Grounded Into DP", "Forceout", "Fielders Choice",
    "Double Play", "Triple Play", "Strikeout - DP",
    "Sac Fly", "Sac Bunt", "Field Error",
    "Fielders Choice Out", "Bunt Groundout", "Bunt Pop Out",
}

FS = s3fs.S3FileSystem()


def process_season_weekly(season: int) -> pd.DataFrame:
    """Load one season, compute weekly wOBA per pitch type, return aggregated rows."""
    prefix = f"{S3_BASE}/season={season}/"
    all_files = FS.ls(prefix)
    pitch_files = [f for f in all_files if "pitches_batch" in f]
    print(f"  Season {season}: {len(pitch_files)} files...", end=" ", flush=True)

    needed = ["at_bat_index", "game_pk", "game_date", "pitch_type", "at_bat_event", "batter_id"]
    frames = []
    for pf in pitch_files:
        with FS.open(pf) as f:
            table = pq.read_table(f, columns=needed)
            frames.append(table.to_pandas())
    pitches = pd.concat(frames, ignore_index=True)
    print(f"{len(pitches):,} pitches", end=" -> ")
    del frames
    gc.collect()

    # Keep PA-ending pitches, dedup to one per PA
    pa = pitches.dropna(subset=["at_bat_event"]).copy()
    del pitches
    gc.collect()

    pa = pa.sort_values(["game_pk", "at_bat_index"]).drop_duplicates(
        subset=["game_pk", "batter_id", "at_bat_index"], keep="last"
    )
    pa = pa[pa["at_bat_event"].isin(PA_EVENTS)]
    pa["pitch_type"] = pa["pitch_type"].str.upper().str.strip()
    pa["game_date"] = pd.to_datetime(pa["game_date"])
    pa["week"] = pa["game_date"].dt.to_period("W")

    # Compute per (week, pitch_type): numerator sum and PA count
    rows = []
    for (week, ptype), grp in pa.groupby(["week", "pitch_type"]):
        if ptype not in TRACKED_PITCH_TYPES:
            continue
        n_pa = len(grp)
        numerator = sum(
            (grp["at_bat_event"] == event).sum() * weight
            for event, weight in WOBA_WEIGHTS.items()
        )
        rows.append({"week": week, "pitch_type": ptype, "n_pa": n_pa, "woba_num": numerator, "season": season})

    result = pd.DataFrame(rows)
    print(f"{len(result)} week-type rows")
    del pa
    gc.collect()
    return result


def ar_analysis(series: np.ndarray) -> dict:
    """Fit AR(1), compute rho, ACF, and diagnostic stats."""
    n = len(series)

    # AR(1) via OLS: y_t = c + rho * y_{t-1} + e_t
    y = series[1:]
    x = add_constant(series[:-1])
    model = OLS(y, x).fit()
    rho = model.params[1]
    rho_se = model.bse[1]
    rho_ci_low = rho - 1.96 * rho_se
    rho_ci_high = rho + 1.96 * rho_se

    # Half-life of mean reversion
    if 0 < rho < 1:
        half_life = -np.log(2) / np.log(rho)
    else:
        half_life = np.inf

    # ACF
    nlags = min(20, n // 3)
    acf_vals = acf(series, nlags=nlags, fft=True)
    pacf_vals = pacf(series, nlags=nlags)

    # ADF
    adf_result = adfuller(series, maxlag=int(np.sqrt(n)), autolag='AIC')
    adf_stat, adf_pval, adf_lags = adf_result[0], adf_result[1], adf_result[2]

    # KPSS
    kpss_result = kpss(series, regression='c', nlags='auto')
    kpss_stat, kpss_pval = kpss_result[0], kpss_result[1]

    # Ljung-Box
    lb = acorr_ljungbox(series - series.mean(), lags=10, return_df=True)
    lb_pval_lag10 = lb['lb_pvalue'].iloc[-1]

    # Residual std
    resid_std = model.resid.std()
    unconditional_std = series.std()

    return {
        "n": n,
        "mean": series.mean(),
        "std": unconditional_std,
        "rho": rho,
        "rho_se": rho_se,
        "rho_ci": (rho_ci_low, rho_ci_high),
        "half_life_weeks": half_life,
        "innovation_std": resid_std,
        "acf_lag1": acf_vals[1] if len(acf_vals) > 1 else np.nan,
        "acf_lag4": acf_vals[4] if len(acf_vals) > 4 else np.nan,
        "acf_lag13": acf_vals[13] if len(acf_vals) > 13 else np.nan,
        "pacf_lag1": pacf_vals[1] if len(pacf_vals) > 1 else np.nan,
        "pacf_lag2": pacf_vals[2] if len(pacf_vals) > 2 else np.nan,
        "adf_stat": adf_stat,
        "adf_pval": adf_pval,
        "adf_lags": adf_lags,
        "kpss_stat": kpss_stat,
        "kpss_pval": kpss_pval,
        "lb_pval_lag10": lb_pval_lag10,
        "acf_vals": acf_vals,
    }


def plot_time_series(weekly_data: dict, ar_results: dict):
    """Plot time series + ACF for each pitch type."""
    n_types = len(weekly_data)
    fig, axes = plt.subplots(n_types, 2, figsize=(16, 3.5 * n_types))
    if n_types == 1:
        axes = axes.reshape(1, -1)

    for idx, (ptype, df) in enumerate(sorted(weekly_data.items())):
        ar = ar_results[ptype]
        series = df["woba"].values
        weeks = range(len(series))

        # Time series plot
        ax = axes[idx, 0]
        ax.plot(weeks, series, linewidth=0.7, alpha=0.8, color='steelblue')
        expanding_mean = pd.Series(series).expanding().mean().values
        ax.plot(weeks, expanding_mean, linewidth=1.5, color='red', label='Expanding mean')
        ax.axhline(ar["mean"], color='gray', linestyle='--', linewidth=0.8, label=f'Grand mean={ar["mean"]:.3f}')
        ax.set_title(f'{ptype} — Weekly League wOBA (n={ar["n"]}, ρ={ar["rho"]:.3f}, HL={ar["half_life_weeks"]:.1f}w)')
        ax.set_xlabel('Week index (across 9 seasons)')
        ax.set_ylabel('wOBA')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ACF plot
        ax2 = axes[idx, 1]
        nlags = min(20, len(series) // 3)
        acf_vals = ar["acf_vals"][:nlags+1]
        ax2.bar(range(len(acf_vals)), acf_vals, width=0.6, color='steelblue', alpha=0.7)
        ci = 1.96 / np.sqrt(len(series))
        ax2.axhline(ci, color='red', linestyle='--', linewidth=0.8, label=f'95% CI (±{ci:.3f})')
        ax2.axhline(-ci, color='red', linestyle='--', linewidth=0.8)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.set_title(f'{ptype} — ACF (ρ={ar["rho"]:.3f} ± {ar["rho_se"]:.3f}, |1-ρ|={abs(1-ar["rho"]):.3f})')
        ax2.set_xlabel('Lag (weeks)')
        ax2.set_ylabel('Autocorrelation')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = "/tmp/woba_stationarity_analysis.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: {outpath}")
    plt.close()
    return outpath


def main():
    print("=" * 70)
    print("RIGOROUS AR ANALYSIS — Per-Pitch-Type League wOBA")
    print("Weekly granularity, AR root characterization")
    print("=" * 70)
    print()

    # Process season by season (memory efficient)
    print("Loading and processing pitches from S3 (season by season)...")
    all_weekly = []
    for season in SEASONS:
        weekly = process_season_weekly(season)
        all_weekly.append(weekly)
        gc.collect()

    combined = pd.concat(all_weekly, ignore_index=True)
    del all_weekly
    gc.collect()
    print(f"\n  Total weekly-type observations: {len(combined):,}")
    print()

    # Build per-type weekly time series
    print("Building weekly time series per pitch type...")
    weekly_data = {}
    for ptype in TRACKED_PITCH_TYPES:
        type_data = combined[combined["pitch_type"] == ptype].copy()
        if type_data.empty:
            continue
        # Aggregate by week (sum numerator, sum PA)
        agg = type_data.groupby("week").agg({"woba_num": "sum", "n_pa": "sum", "season": "first"}).reset_index()
        # Filter weeks with minimum 50 PAs
        agg = agg[agg["n_pa"] >= 50].reset_index(drop=True)
        agg["woba"] = agg["woba_num"] / agg["n_pa"]
        # Sort chronologically
        agg = agg.sort_values("week").reset_index(drop=True)
        weekly_data[ptype] = agg
        print(f"  {ptype}: {len(agg)} weeks, {type_data['n_pa'].sum():,} total PAs")

    print()
    print("Running AR(1) analysis...")
    ar_results = {}
    for ptype, df in sorted(weekly_data.items()):
        series = df["woba"].values
        if len(series) < 30:
            print(f"  {ptype}: only {len(series)} weeks — skipping (need ≥30)")
            continue
        ar = ar_analysis(series)
        ar_results[ptype] = ar

    # Summary table
    print()
    print("=" * 70)
    print("QUANTITATIVE SUMMARY")
    print("=" * 70)
    print()
    header = f"{'Type':<5} {'n':>5} {'Mean':>7} {'Std':>7} {'rho':>7} {'rho_SE':>7} {'95% CI':>18} {'|1-rho|':>8} {'HL(wks)':>8} {'ADF_p':>7} {'KPSS_p':>7}"
    print(header)
    print("-" * len(header))

    for ptype in TRACKED_PITCH_TYPES:
        if ptype not in ar_results:
            continue
        ar = ar_results[ptype]
        ci_str = f"[{ar['rho_ci'][0]:.3f}, {ar['rho_ci'][1]:.3f}]"
        dist_from_unit = abs(1.0 - ar['rho'])
        hl_str = f"{ar['half_life_weeks']:.1f}" if ar['half_life_weeks'] < 1000 else "inf"
        print(f"{ptype:<5} {ar['n']:>5} {ar['mean']:>7.4f} {ar['std']:>7.4f} {ar['rho']:>7.4f} {ar['rho_se']:>7.4f} {ci_str:>18} {dist_from_unit:>8.4f} {hl_str:>8} {ar['adf_pval']:>7.4f} {ar['kpss_pval']:>7.3f}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("  rho close to 1.0 (|1-rho| < 0.05): Near-unit-root. ADF underpowered.")
    print("  rho moderately <1 (|1-rho| 0.05-0.20): Persistent but mean-reverting. ADF marginal.")
    print("  rho well below 1 (|1-rho| > 0.20): Strong mean reversion. ADF reliable.")
    print("  KPSS p < 0.05 with ACF(1) > 0.3: Likely size distortion (over-rejecting stationarity).")
    print()

    for ptype in TRACKED_PITCH_TYPES:
        if ptype not in ar_results:
            continue
        ar = ar_results[ptype]
        dist = abs(1.0 - ar['rho'])
        print(f"  {ptype}:")
        print(f"    rho = {ar['rho']:.4f} (SE={ar['rho_se']:.4f}), 95% CI: [{ar['rho_ci'][0]:.4f}, {ar['rho_ci'][1]:.4f}]")
        print(f"    |1 - rho| = {dist:.4f}")
        print(f"    Half-life = {ar['half_life_weeks']:.1f} weeks")
        print(f"    ACF decay: lag1={ar['acf_lag1']:.3f}, lag4={ar['acf_lag4']:.3f}, lag13={ar['acf_lag13']:.3f}")
        print(f"    Innovation std: {ar['innovation_std']:.4f} (unconditional: {ar['std']:.4f}, ratio: {ar['innovation_std']/ar['std']:.3f})")

        if dist < 0.05:
            diagnosis = "NEAR UNIT ROOT — ADF underpowered, treat as non-stationary"
        elif dist < 0.20:
            diagnosis = "MODERATELY PERSISTENT — mean-reverts slowly"
        else:
            diagnosis = "STRONGLY MEAN-REVERTING — stationary"

        kpss_issue = ""
        if ar['kpss_pval'] < 0.05 and ar['acf_lag1'] > 0.3:
            kpss_issue = f" [KPSS likely false positive: high ACF(1)={ar['acf_lag1']:.3f} causes size distortion]"

        print(f"    DIAGNOSIS: {diagnosis}{kpss_issue}")
        print()

    # Generate and upload plot
    print("Generating time series + ACF plots...")
    plot_path = plot_time_series(weekly_data, ar_results)

    print("\nUploading to S3...")
    FS.put(plot_path, "mlb-265753586044-us-east-1-an/analysis/woba_stationarity_analysis.png")
    print("Done: s3://mlb-265753586044-us-east-1-an/analysis/woba_stationarity_analysis.png")


if __name__ == "__main__":
    main()
