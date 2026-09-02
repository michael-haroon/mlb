"""Prove per-pitch-type league-average wOBA stationarity at DAILY granularity.

Computes league-avg wOBA per pitch type per game-day (~1600 data points),
then runs ADF, KPSS, and Mann-Kendall trend tests with proper statistical power.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import s3fs
from collections import defaultdict

S3_BASE = "mlb-265753586044-us-east-1-an/data"
SEASONS = list(range(2015, 2025))
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


def load_season(season: int) -> pd.DataFrame:
    """Load pitches for a season from S3."""
    prefix = f"{S3_BASE}/season={season}/"
    all_files = FS.ls(prefix)
    pitch_files = [f for f in all_files if "pitches_batch" in f]
    if not pitch_files:
        return pd.DataFrame()

    needed = ["game_date", "at_bat_index", "game_pk", "pitch_type", "at_bat_event", "batter_id"]
    frames = []
    for pf in pitch_files:
        with FS.open(pf) as f:
            table = pq.read_table(f, columns=needed)
            frames.append(table.to_pandas())
    return pd.concat(frames, ignore_index=True)


def compute_daily_woba(pitches: pd.DataFrame) -> pd.DataFrame:
    """Compute league-average wOBA per pitch type per game_date."""
    pa = pitches.dropna(subset=["at_bat_event"]).copy()
    pa = pa.sort_values(["game_pk", "at_bat_index"]).drop_duplicates(
        subset=["game_pk", "batter_id", "at_bat_index"], keep="last"
    )
    pa = pa[pa["at_bat_event"].isin(PA_EVENTS)]
    pa["pitch_type"] = pa["pitch_type"].str.upper().str.strip()
    pa["game_date"] = pd.to_datetime(pa["game_date"], errors="coerce")

    # Compute wOBA value for each PA
    pa["woba_value"] = pa["at_bat_event"].map(WOBA_WEIGHTS).fillna(0.0)

    # Group by (game_date, pitch_type) and compute mean wOBA
    daily = (
        pa.groupby(["game_date", "pitch_type"])
        .agg(woba_mean=("woba_value", "mean"), n_pa=("woba_value", "count"))
        .reset_index()
    )
    return daily


def run_stationarity_tests(series: pd.Series, name: str) -> dict:
    """Run ADF, KPSS, and Mann-Kendall on a time series."""
    from statsmodels.tsa.stattools import adfuller, kpss

    clean = series.dropna()
    n = len(clean)
    result = {"n": n, "mean": clean.mean(), "std": clean.std()}

    if n < 30:
        result["adf_stationary"] = None
        result["kpss_stationary"] = None
        result["trend"] = None
        return result

    # ADF test (null = unit root / non-stationary)
    try:
        adf_stat, adf_p, _, _, _, _ = adfuller(clean.values, maxlag=int(np.sqrt(n)))
        result["adf_stat"] = adf_stat
        result["adf_p"] = adf_p
        result["adf_stationary"] = adf_p < 0.05  # reject unit root
    except Exception as e:
        result["adf_stationary"] = None
        result["adf_error"] = str(e)

    # KPSS test (null = stationary)
    try:
        kpss_stat, kpss_p, _, _ = kpss(clean.values, regression="c", nlags="auto")
        result["kpss_stat"] = kpss_stat
        result["kpss_p"] = kpss_p
        result["kpss_stationary"] = kpss_p > 0.05  # fail to reject stationarity
    except Exception as e:
        result["kpss_stationary"] = None
        result["kpss_error"] = str(e)

    # Mann-Kendall trend test
    try:
        from scipy.stats import kendalltau
        x = np.arange(n)
        tau, mk_p = kendalltau(x, clean.values)
        result["mk_tau"] = tau
        result["mk_p"] = mk_p
        result["has_trend"] = mk_p < 0.05
    except Exception as e:
        result["has_trend"] = None

    # CV
    result["cv"] = clean.std() / clean.mean() if clean.mean() > 0 else float("inf")

    return result


def main():
    print("=" * 70)
    print("wOBA STATIONARITY — DAILY GRANULARITY (ADF + KPSS + Mann-Kendall)")
    print("=" * 70)
    print()

    all_daily = []

    for season in SEASONS:
        if season == 2020:
            print(f"  Season {season}: SKIPPED (COVID)")
            continue
        print(f"  Loading season {season}...", end=" ", flush=True)
        pitches = load_season(season)
        if pitches.empty:
            print("NO DATA")
            continue
        print(f"{len(pitches):,} pitches", end=" -> ")
        daily = compute_daily_woba(pitches)
        print(f"{len(daily):,} daily obs")
        all_daily.append(daily)

    combined = pd.concat(all_daily, ignore_index=True)
    combined = combined.sort_values("game_date").reset_index(drop=True)

    print(f"\n  Total daily observations: {len(combined):,}")
    print()

    # Filter to days with sufficient sample size (min 50 PAs per type per day)
    combined_filtered = combined[combined["n_pa"] >= 50]
    print(f"  After filtering (>=50 PA/day): {len(combined_filtered):,} obs")
    print()

    # Run stationarity tests per pitch type
    print("=" * 70)
    print("STATIONARITY TEST RESULTS")
    print("=" * 70)
    print()
    print(f"{'Type':<6} {'N':>6} {'Mean':>7} {'Std':>7} {'CV%':>7} "
          f"{'ADF_p':>7} {'ADF?':>5} {'KPSS_p':>7} {'KPSS?':>5} "
          f"{'MK_tau':>7} {'MK_p':>7} {'Trend?':>6}")
    print("-" * 90)

    results = {}
    for ptype in TRACKED_PITCH_TYPES:
        ptype_data = combined_filtered[combined_filtered["pitch_type"] == ptype]
        if ptype_data.empty:
            print(f"{ptype:<6} -- insufficient data --")
            continue

        # Create a proper time series indexed by date (average if multiple entries per day)
        ts = ptype_data.groupby("game_date")["woba_mean"].mean()
        ts = ts.sort_index()

        test_results = run_stationarity_tests(ts, ptype)
        results[ptype] = test_results

        adf_flag = {True: "YES", False: "NO", None: "N/A"}[test_results.get("adf_stationary")]
        kpss_flag = {True: "YES", False: "NO", None: "N/A"}[test_results.get("kpss_stationary")]
        trend_flag = {True: "YES", False: "NO", None: "N/A"}[test_results.get("has_trend")]

        print(f"{ptype:<6} {test_results['n']:>6} {test_results['mean']:>7.4f} "
              f"{test_results['std']:>7.4f} {test_results['cv']*100:>6.2f}% "
              f"{test_results.get('adf_p', 0):>7.4f} {adf_flag:>5} "
              f"{test_results.get('kpss_p', 0):>7.4f} {kpss_flag:>5} "
              f"{test_results.get('mk_tau', 0):>7.4f} {test_results.get('mk_p', 0):>7.4f} "
              f"{trend_flag:>6}")

    print("-" * 90)
    print()
    print("Legend: ADF? = reject unit root (stationary). KPSS? = fail to reject stationarity.")
    print("        Both YES = strong evidence of stationarity.")
    print("        Trend? = Mann-Kendall detects monotonic trend (p<0.05).")
    print()

    # Final verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    for ptype, r in results.items():
        adf_ok = r.get("adf_stationary", False)
        kpss_ok = r.get("kpss_stationary", False)
        has_trend = r.get("has_trend", False)

        if adf_ok and kpss_ok and not has_trend:
            verdict = "STATIONARY — static constant justified"
        elif adf_ok and kpss_ok and has_trend:
            verdict = "TREND-STATIONARY — constant OK but drifting slowly"
        elif not adf_ok and not kpss_ok:
            verdict = "NON-STATIONARY — use rolling/expanding average"
        elif adf_ok and not kpss_ok:
            verdict = "MIXED (ADF=stat, KPSS=non-stat) — likely trend"
        else:
            verdict = "MIXED (ADF=non-stat, KPSS=stat) — borderline"

        print(f"  {ptype}: {verdict} (mean={r['mean']:.4f}, CV={r['cv']*100:.2f}%)")

    print()
    print("RECOMMENDED CONSTANTS (for stationary types):")
    print("  _LEAGUE_AVG_WOBA_BY_TYPE = {")
    for ptype, r in sorted(results.items()):
        adf_ok = r.get("adf_stationary", False)
        kpss_ok = r.get("kpss_stationary", False)
        tag = "" if (adf_ok and kpss_ok) else "  # TODO: validate — non-stationary"
        print(f'      "{ptype}": {r["mean"]:.3f},{tag}')
    print("  }")


if __name__ == "__main__":
    main()
