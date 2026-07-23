"""Prove per-pitch-type league-average wOBA stationarity across 2015-2024.

Loads raw pitches from S3, computes wOBA per pitch type per season,
reports CV and determines whether static constants are justified.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import s3fs

S3_BASE = "mlb-265753586044-us-east-1-an/data"
SEASONS = list(range(2015, 2025))  # 2015-2024 (exclude 2020 shortened season)
TRACKED_PITCH_TYPES = ("FF", "SL", "CH", "CU", "FC", "SI", "FS", "ST")

# wOBA weights (linear weights from FanGraphs methodology)
# Keys match the title-case at_bat_event values in Statcast data
WOBA_WEIGHTS = {
    "Walk": 0.690,
    "Intent Walk": 0.690,
    "Hit By Pitch": 0.720,
    "Single": 0.880,
    "Double": 1.260,
    "Triple": 1.590,
    "Home Run": 2.080,
}

# Events that count as a PA denominator (excludes sacrifices per FanGraphs definition)
PA_EVENTS = set(WOBA_WEIGHTS.keys()) | {
    "Strikeout", "Groundout", "Flyout", "Lineout", "Pop Out",
    "Grounded Into DP", "Forceout", "Fielders Choice",
    "Double Play", "Triple Play", "Strikeout - DP",
    "Sac Fly", "Sac Bunt", "Field Error",
    "Fielders Choice Out", "Bunt Groundout", "Bunt Pop Out",
}

FS = s3fs.S3FileSystem()


def load_pitches_for_season(season: int) -> pd.DataFrame:
    """Load pitches parquet files from S3 for a given season."""
    prefix = f"{S3_BASE}/season={season}/"
    try:
        all_files = FS.ls(prefix)
        pitch_files = [f for f in all_files if "pitches_batch" in f]
        if not pitch_files:
            print(f"no pitches files found")
            return pd.DataFrame()

        needed = ["at_bat_index", "game_pk", "pitch_type", "at_bat_event", "batter_id"]
        frames = []
        for pf in pitch_files:
            with FS.open(pf) as f:
                table = pq.read_table(f, columns=needed)
                frames.append(table.to_pandas())
        df = pd.concat(frames, ignore_index=True)
        return df
    except Exception as e:
        print(f"error - {e}")
        return pd.DataFrame()


def compute_woba_per_pitch_type(pitches: pd.DataFrame, season: int) -> dict:
    """Compute league-average wOBA for each pitch type in a season.

    wOBA is computed on the LAST pitch of each PA (the pitch that ends the AB).
    """
    if pitches.empty:
        return {}

    # Keep only rows with an at_bat_event (PA-ending pitches)
    pa = pitches.dropna(subset=["at_bat_event"]).copy()

    # Dedup: keep only the last pitch of each PA
    pa = pa.sort_values(["game_pk", "at_bat_index"]).drop_duplicates(
        subset=["game_pk", "batter_id", "at_bat_index"], keep="last"
    )

    # Filter to PA events only
    pa = pa[pa["at_bat_event"].isin(PA_EVENTS)]

    if pa.empty:
        return {}

    # Normalize pitch type
    pa["pitch_type"] = pa["pitch_type"].str.upper().str.strip()

    results = {}
    for ptype in TRACKED_PITCH_TYPES:
        subset = pa[pa["pitch_type"] == ptype]
        if len(subset) < 100:
            continue

        # Compute wOBA numerator
        numerator = sum(
            (subset["at_bat_event"] == event).sum() * weight
            for event, weight in WOBA_WEIGHTS.items()
        )
        denominator = len(subset)

        if denominator > 0:
            results[ptype] = numerator / denominator

    return results


def main():
    print("=" * 70)
    print("wOBA STATIONARITY ANALYSIS — Per Pitch Type (2015-2024)")
    print("=" * 70)
    print()

    all_results = {}

    for season in SEASONS:
        if season == 2020:
            print(f"  Season {season}: SKIPPED (60-game COVID season)")
            continue
        print(f"  Loading season {season}...", end=" ", flush=True)
        pitches = load_pitches_for_season(season)
        if pitches.empty:
            print("NO DATA")
            continue
        print(f"{len(pitches):,} pitches", end=" -> ")
        woba = compute_woba_per_pitch_type(pitches, season)
        print(f"{len(woba)} pitch types computed")
        all_results[season] = woba

    print()
    print("=" * 70)
    print("RESULTS: Per-Pitch-Type League-Average wOBA")
    print("=" * 70)
    print()

    # Build results table
    result_df = pd.DataFrame(all_results).T
    result_df.index.name = "season"

    print(result_df.round(4).to_string())
    print()

    # Stationarity metrics
    print("-" * 70)
    print(f"{'Pitch Type':<12} {'Mean':>8} {'Std':>8} {'CV%':>8} {'Min':>8} {'Max':>8} {'Range':>8} {'Stationary?':<12}")
    print("-" * 70)

    stationarity_results = {}
    for ptype in TRACKED_PITCH_TYPES:
        if ptype not in result_df.columns:
            continue
        vals = result_df[ptype].dropna()
        if len(vals) < 5:
            continue
        mean_v = vals.mean()
        std_v = vals.std(ddof=1)
        cv = std_v / mean_v if mean_v > 0 else float("inf")
        stationary = cv < 0.05
        stationarity_results[ptype] = {
            "mean": mean_v, "std": std_v, "cv": cv,
            "min": vals.min(), "max": vals.max(),
            "stationary": stationary,
        }
        print(f"{ptype:<12} {mean_v:>8.4f} {std_v:>8.4f} {cv*100:>7.2f}% {vals.min():>8.4f} {vals.max():>8.4f} {vals.max()-vals.min():>8.4f} {'YES' if stationary else 'NO'}")

    print("-" * 70)
    print()

    # Final recommendation
    all_stationary = all(r["stationary"] for r in stationarity_results.values())
    print("CONCLUSION:")
    if all_stationary:
        print("  ALL pitch types have CV < 5% -> static constants ARE justified.")
        print()
        print("  Recommended _LEAGUE_AVG_WOBA_BY_TYPE = {")
        for ptype, r in sorted(stationarity_results.items()):
            print(f'      "{ptype}": {r["mean"]:.3f},')
        print("  }")
        print()
        overall_mean = np.mean([r["mean"] for r in stationarity_results.values()])
        print(f"  Overall mean (fallback): {overall_mean:.3f}")
    else:
        non_stat = [k for k, v in stationarity_results.items() if not v["stationary"]]
        print(f"  NON-STATIONARY pitch types: {non_stat}")
        print("  These need per-season or rolling league average, not static constants.")


if __name__ == "__main__":
    main()
