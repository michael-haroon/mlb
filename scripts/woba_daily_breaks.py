"""Daily-granularity wOBA stationarity with PA-weighted AR(1) and structural break tests.

Addresses two gaps in the weekly analysis:
(a) PA-weighted least squares (WLS) AR(1) at daily granularity — no frequency aggregation
    Auto-bandwidth KPSS with Newey-West (data-driven lag selection)
(b) Chow test at known structural break dates:
    - 2021 season start: deadened ball
    - 2022 season start: universal humidor
    - 2023 season start: Hawk-Eye upgrade
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import s3fs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools import add_constant
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import gc

S3_BASE = "mlb-265753586044-us-east-1-an/data"
SEASONS = [y for y in range(2015, 2025) if y != 2020]
TRACKED_PITCH_TYPES = ("FF", "SL", "CH", "CU", "FC", "SI", "FS", "ST")

WOBA_WEIGHTS = {
    "Walk": 0.690, "Intent Walk": 0.690, "Hit By Pitch": 0.720,
    "Single": 0.880, "Double": 1.260, "Triple": 1.590, "Home Run": 2.080,
}

PA_EVENTS = set(WOBA_WEIGHTS.keys()) | {
    "Strikeout", "Groundout", "Flyout", "Lineout", "Pop Out",
    "Grounded Into DP", "Forceout", "Fielders Choice",
    "Double Play", "Triple Play", "Strikeout - DP",
    "Sac Fly", "Sac Bunt", "Field Error",
    "Fielders Choice Out", "Bunt Groundout", "Bunt Pop Out",
}

FS = s3fs.S3FileSystem()

# Known structural break dates (first game of season or rule effective date)
BREAK_DATES = {
    "2021_deadened_ball": "2021-04-01",
    "2022_humidor": "2022-04-07",
    "2023_hawkeye": "2023-03-30",
}


def process_season_daily(season: int) -> pd.DataFrame:
    """Load one season, compute daily wOBA per pitch type."""
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

    pa = pitches.dropna(subset=["at_bat_event"]).copy()
    del pitches
    gc.collect()

    pa = pa.sort_values(["game_pk", "at_bat_index"]).drop_duplicates(
        subset=["game_pk", "batter_id", "at_bat_index"], keep="last"
    )
    pa = pa[pa["at_bat_event"].isin(PA_EVENTS)]
    pa["pitch_type"] = pa["pitch_type"].str.upper().str.strip()
    pa["game_date"] = pd.to_datetime(pa["game_date"])

    rows = []
    for (date, ptype), grp in pa.groupby(["game_date", "pitch_type"]):
        if ptype not in TRACKED_PITCH_TYPES:
            continue
        n_pa = len(grp)
        numerator = sum(
            (grp["at_bat_event"] == event).sum() * weight
            for event, weight in WOBA_WEIGHTS.items()
        )
        rows.append({"date": date, "pitch_type": ptype, "n_pa": n_pa,
                     "woba_num": numerator, "season": season})

    result = pd.DataFrame(rows)
    print(f"{len(result)} day-type rows")
    del pa
    gc.collect()
    return result


def pa_weighted_ar1(series: np.ndarray, weights: np.ndarray) -> dict:
    """PA-weighted AR(1) via WLS: y_t = c + rho * y_{t-1} + e_t, weighted by sqrt(PA_t)."""
    n = len(series)
    y = series[1:]
    x = add_constant(series[:-1])
    w = weights[1:]  # weight for observation t uses PA at time t

    model = WLS(y, x, weights=w).fit()
    rho = model.params[1]
    rho_se = model.bse[1]

    # Also fit OLS for comparison
    ols_model = OLS(y, x).fit()
    rho_ols = ols_model.params[1]
    rho_ols_se = ols_model.bse[1]

    # Half-life
    if 0 < rho < 1:
        half_life = -np.log(2) / np.log(rho)
    elif rho <= 0:
        half_life = None  # N/A for non-positive rho
    else:
        half_life = np.inf

    return {
        "n": n,
        "rho_wls": rho,
        "rho_wls_se": rho_se,
        "rho_wls_ci": (rho - 1.96 * rho_se, rho + 1.96 * rho_se),
        "rho_ols": rho_ols,
        "rho_ols_se": rho_ols_se,
        "half_life": half_life,
        "dist_from_unit": abs(1.0 - rho),
        "innovation_std": model.resid.std(),
        "unconditional_std": series.std(),
    }


def auto_bandwidth_kpss(series: np.ndarray) -> dict:
    """KPSS with automatic Newey-West bandwidth selection (nlags='auto').

    This uses the data-driven lag selection that adapts to the autocorrelation
    structure rather than a fixed lag count.
    """
    # statsmodels kpss with nlags='auto' uses Schwert's formula:
    # int(12 * (n/100)^{1/4}) which is data-driven
    result = kpss(series, regression='c', nlags='auto')
    stat, pval, lags_used = result[0], result[1], result[2]

    # Also run with 'legacy' fixed lag for comparison (Schwert's original)
    n = len(series)
    legacy_lags = int(np.ceil(12 * (n / 100) ** 0.25))
    result_fixed = kpss(series, regression='c', nlags=legacy_lags)

    return {
        "kpss_stat_auto": stat,
        "kpss_pval_auto": pval,
        "kpss_lags_auto": lags_used,
        "kpss_stat_fixed": result_fixed[0],
        "kpss_pval_fixed": result_fixed[1],
        "kpss_lags_fixed": legacy_lags,
    }


def chow_test(series: np.ndarray, dates: np.ndarray, break_date: str, ptype: str) -> dict:
    """Chow test for structural break at a specific date.

    Tests whether AR(1) parameters (intercept + slope) differ before vs after the break.
    Also reports simple pre/post mean comparison (Welch's t-test).
    """
    break_dt = pd.Timestamp(break_date)
    break_idx = np.searchsorted(dates, break_dt)

    if break_idx < 20 or break_idx > len(series) - 20:
        return {"skip": True, "reason": f"break at idx {break_idx}, need >=20 on each side"}

    pre = series[:break_idx]
    post = series[break_idx:]

    # Simple mean comparison (Welch's t-test)
    t_stat, t_pval = stats.ttest_ind(pre, post, equal_var=False)
    mean_diff = post.mean() - pre.mean()
    effect_size = mean_diff / series.std()  # Cohen's d

    # Chow test via F-test on AR(1) model
    # Full model: AR(1) on entire series
    y_full = series[1:]
    x_full = add_constant(series[:-1])
    model_full = OLS(y_full, x_full).fit()
    rss_full = (model_full.resid ** 2).sum()

    # Split models
    if break_idx >= 2 and break_idx < len(series) - 1:
        y_pre = series[1:break_idx]
        x_pre = add_constant(series[:break_idx - 1])
        y_post = series[break_idx:]
        x_post = add_constant(series[break_idx - 1:-1])

        if len(y_pre) > 3 and len(y_post) > 3:
            model_pre = OLS(y_pre, x_pre).fit()
            model_post = OLS(y_post, x_post).fit()
            rss_split = (model_pre.resid ** 2).sum() + (model_post.resid ** 2).sum()

            k = 2  # number of parameters (intercept + rho)
            n_total = len(y_full)
            f_stat = ((rss_full - rss_split) / k) / (rss_split / (n_total - 2 * k))
            f_pval = 1 - stats.f.cdf(f_stat, k, n_total - 2 * k)
        else:
            f_stat, f_pval = np.nan, np.nan
    else:
        f_stat, f_pval = np.nan, np.nan

    return {
        "skip": False,
        "break_idx": break_idx,
        "n_pre": len(pre),
        "n_post": len(post),
        "mean_pre": pre.mean(),
        "mean_post": post.mean(),
        "mean_diff": mean_diff,
        "effect_size_d": effect_size,
        "welch_t": t_stat,
        "welch_p": t_pval,
        "chow_F": f_stat,
        "chow_p": f_pval,
    }


def main():
    print("=" * 80)
    print("DAILY wOBA STATIONARITY: PA-Weighted AR(1) + Auto-Bandwidth KPSS + Break Tests")
    print("=" * 80)
    print()

    # Load data
    print("Loading pitches from S3 (season by season, memory efficient)...")
    all_daily = []
    for season in SEASONS:
        daily = process_season_daily(season)
        all_daily.append(daily)
        gc.collect()

    combined = pd.concat(all_daily, ignore_index=True)
    del all_daily
    gc.collect()
    print(f"\n  Total daily-type observations: {len(combined):,}")
    print()

    # Build daily time series per pitch type
    print("Building daily time series per pitch type...")
    daily_data = {}
    for ptype in TRACKED_PITCH_TYPES:
        type_data = combined[combined["pitch_type"] == ptype].copy()
        if type_data.empty:
            continue
        agg = type_data.groupby("date").agg({"woba_num": "sum", "n_pa": "sum"}).reset_index()
        # Keep all days with at least 20 PAs (lower threshold than weekly since daily)
        agg = agg[agg["n_pa"] >= 20].sort_values("date").reset_index(drop=True)
        agg["woba"] = agg["woba_num"] / agg["n_pa"]
        daily_data[ptype] = agg
        print(f"  {ptype}: {len(agg)} days, {type_data['n_pa'].sum():,} total PAs, "
              f"mean PA/day={agg['n_pa'].mean():.0f}")

    # =========================================================================
    # (a) PA-WEIGHTED AR(1) + AUTO-BANDWIDTH KPSS AT DAILY GRANULARITY
    # =========================================================================
    print()
    print("=" * 80)
    print("(a) PA-WEIGHTED AR(1) + AUTO-BANDWIDTH KPSS — DAILY GRANULARITY")
    print("=" * 80)
    print()

    ar_results = {}
    kpss_results = {}

    header = f"{'Type':<5} {'n':>6} {'rho_WLS':>8} {'SE':>7} {'95% CI':>18} {'|1-rho|':>8} {'HL':>8} {'rho_OLS':>8} {'KPSS_p(auto)':>13} {'KPSS_lags':>10}"
    print(header)
    print("-" * len(header))

    for ptype in TRACKED_PITCH_TYPES:
        if ptype not in daily_data:
            continue
        df = daily_data[ptype]
        series = df["woba"].values
        weights = df["n_pa"].values.astype(float)

        if len(series) < 50:
            print(f"  {ptype}: only {len(series)} days — skipping")
            continue

        ar = pa_weighted_ar1(series, weights)
        ar_results[ptype] = ar

        kp = auto_bandwidth_kpss(series)
        kpss_results[ptype] = kp

        ci_str = f"[{ar['rho_wls_ci'][0]:.3f}, {ar['rho_wls_ci'][1]:.3f}]"
        hl_str = f"{ar['half_life']:.1f}d" if ar['half_life'] is not None and ar['half_life'] < 1000 else "N/A"
        print(f"{ptype:<5} {ar['n']:>6} {ar['rho_wls']:>8.4f} {ar['rho_wls_se']:>7.4f} {ci_str:>18} {ar['dist_from_unit']:>8.4f} {hl_str:>8} {ar['rho_ols']:>8.4f} {kp['kpss_pval_auto']:>13.4f} {kp['kpss_lags_auto']:>10}")

    print()
    print("  Notes:")
    print("  - rho_WLS: PA-weighted AR(1) coefficient (higher-PA days get more influence)")
    print("  - rho_OLS: unweighted AR(1) for comparison")
    print("  - KPSS uses Newey-West auto-bandwidth (data-driven lag selection)")
    print("  - HL: half-life in DAYS (N/A if rho <= 0)")
    print()

    # Detailed KPSS comparison
    print("  KPSS Detail (auto-bandwidth vs fixed Schwert):")
    print(f"  {'Type':<5} {'Auto stat':>10} {'Auto p':>8} {'Auto lags':>10} {'Fixed stat':>11} {'Fixed p':>9} {'Fixed lags':>11}")
    for ptype in TRACKED_PITCH_TYPES:
        if ptype not in kpss_results:
            continue
        kp = kpss_results[ptype]
        print(f"  {ptype:<5} {kp['kpss_stat_auto']:>10.4f} {kp['kpss_pval_auto']:>8.4f} {kp['kpss_lags_auto']:>10} {kp['kpss_stat_fixed']:>11.4f} {kp['kpss_pval_fixed']:>9.4f} {kp['kpss_lags_fixed']:>11}")

    # =========================================================================
    # (b) STRUCTURAL BREAK TESTS AT KNOWN DATES
    # =========================================================================
    print()
    print("=" * 80)
    print("(b) STRUCTURAL BREAK TESTS — Known regime changes")
    print("    2021: deadened ball | 2022: universal humidor | 2023: Hawk-Eye")
    print("=" * 80)
    print()

    break_results = {}
    for ptype in TRACKED_PITCH_TYPES:
        if ptype not in daily_data:
            continue
        df = daily_data[ptype]
        series = df["woba"].values
        dates = df["date"].values

        print(f"  {ptype}:")
        break_results[ptype] = {}
        for break_name, break_date in BREAK_DATES.items():
            result = chow_test(series, dates, break_date, ptype)
            break_results[ptype][break_name] = result
            if result.get("skip"):
                print(f"    {break_name}: SKIPPED ({result['reason']})")
            else:
                sig_welch = "***" if result['welch_p'] < 0.001 else "**" if result['welch_p'] < 0.01 else "*" if result['welch_p'] < 0.05 else "ns"
                sig_chow = "***" if result['chow_p'] < 0.001 else "**" if result['chow_p'] < 0.01 else "*" if result['chow_p'] < 0.05 else "ns"
                print(f"    {break_name}:")
                print(f"      n_pre={result['n_pre']}, n_post={result['n_post']}")
                print(f"      Mean: pre={result['mean_pre']:.4f}, post={result['mean_post']:.4f}, diff={result['mean_diff']:+.4f} (d={result['effect_size_d']:+.3f})")
                print(f"      Welch t={result['welch_t']:.3f}, p={result['welch_p']:.4f} {sig_welch}")
                print(f"      Chow F={result['chow_F']:.3f}, p={result['chow_p']:.4f} {sig_chow}")
        print()

    # =========================================================================
    # SUMMARY TABLE — Break tests
    # =========================================================================
    print("=" * 80)
    print("BREAK TEST SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Type':<5} | {'2021 deadened ball':^35} | {'2022 humidor':^35} | {'2023 Hawk-Eye':^35}")
    print(f"{'':5} | {'diff':>7} {'d':>6} {'Welch_p':>8} {'Chow_p':>8} | {'diff':>7} {'d':>6} {'Welch_p':>8} {'Chow_p':>8} | {'diff':>7} {'d':>6} {'Welch_p':>8} {'Chow_p':>8}")
    print("-" * 120)
    for ptype in TRACKED_PITCH_TYPES:
        if ptype not in break_results:
            continue
        parts = [f"{ptype:<5}"]
        for break_name in BREAK_DATES:
            r = break_results[ptype].get(break_name, {})
            if r.get("skip"):
                parts.append(f"{'--':>7} {'--':>6} {'--':>8} {'--':>8}")
            else:
                parts.append(f"{r['mean_diff']:>+7.4f} {r['effect_size_d']:>+6.3f} {r['welch_p']:>8.4f} {r['chow_p']:>8.4f}")
        print(" | ".join(parts))

    # =========================================================================
    # FINAL DIAGNOSIS
    # =========================================================================
    print()
    print("=" * 80)
    print("FINAL DIAGNOSIS")
    print("=" * 80)
    print()

    for ptype in TRACKED_PITCH_TYPES:
        if ptype not in ar_results:
            continue
        ar = ar_results[ptype]
        kp = kpss_results[ptype]

        # AR diagnosis
        if ar['dist_from_unit'] < 0.05:
            ar_diag = "NEAR UNIT ROOT — cannot distinguish from random walk"
        elif ar['dist_from_unit'] < 0.20:
            ar_diag = "MODERATELY PERSISTENT"
        else:
            ar_diag = "STRONGLY MEAN-REVERTING (|1-rho| > 0.2)"

        # KPSS diagnosis
        if kp['kpss_pval_auto'] > 0.10:
            kpss_diag = "STATIONARY (fail to reject level stationarity)"
        elif kp['kpss_pval_auto'] > 0.05:
            kpss_diag = "MARGINAL (0.05 < p < 0.10)"
        else:
            kpss_diag = "NON-STATIONARY (reject level stationarity)"

        # Break diagnosis
        any_break = False
        break_details = []
        for break_name in BREAK_DATES:
            r = break_results.get(ptype, {}).get(break_name, {})
            if not r.get("skip") and r.get("welch_p", 1) < 0.01 and abs(r.get("effect_size_d", 0)) > 0.1:
                any_break = True
                break_details.append(f"{break_name}(d={r['effect_size_d']:+.3f})")

        print(f"  {ptype}:")
        print(f"    AR(1) WLS: rho={ar['rho_wls']:.4f}, |1-rho|={ar['dist_from_unit']:.4f} -> {ar_diag}")
        print(f"    KPSS auto-BW: p={kp['kpss_pval_auto']:.4f} -> {kpss_diag}")
        if any_break:
            print(f"    BREAKS DETECTED: {', '.join(break_details)}")
            print(f"    -> Static constant NOT safe. Use regime-specific or expanding mean.")
        else:
            print(f"    No significant structural breaks at known dates.")
            print(f"    -> Static constant is justified.")
        print()


if __name__ == "__main__":
    main()
