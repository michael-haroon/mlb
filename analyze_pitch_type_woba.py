"""Analyze year-to-year stability of per-pitch-type league-average wOBA.

Validates whether _LEAGUE_AVG_WOBA can be a static constant or needs
to be pitch-type-specific and/or seasonally adjusted.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from classical_learning.engineering.pitch_level_features import (
    WOBA_WEIGHTS,
    WOBA_PA_DENOM_EVENTS,
    TRACKED_PITCH_TYPES,
)

# Map at_bat_event strings to wOBA weight keys (copied from pitch_level_features.py)
_EVENT_TO_WOBA_KEY = {
    "walk":          "walk",
    "intent_walk":   "walk",
    "hit_by_pitch":  "hbp",
    "single":        "single",
    "double":        "double",
    "triple":        "triple",
    "home_run":      "home_run",
}


def main():
    print("Loading raw pitches data (2015-2024, regular season only)...")
    from classical_learning.engineering.data_loader import load_all

    # Use S3 source from trading config
    SOURCE = "s3://mlb-265753586044-us-east-1-an/data"
    SEASON_START = 2015
    SEASON_END = 2024

    # Load parquet — load_all returns dict keyed by table name
    data = load_all(SOURCE, SEASON_START, SEASON_END)
    pitches = data["pitches_raw"]

    # Filter: regular season only, 2015-2024 (skip 2020 pandemic season as per project standard)
    pitches = pitches[
        (pitches["game_type_code"] == "R") &
        (pitches["season"] >= 2015) &
        (pitches["season"] <= 2024) &
        (pitches["season"] != 2020)
    ].copy()

    print(f"Loaded {len(pitches):,} regular-season pitch rows (2020 excluded)")

    # Filter to actual pitches (is_pitch == True) with a completed PA (at_bat_event non-null)
    pa = pitches[
        (pitches["is_pitch"] == True) &
        pitches["at_bat_event"].notna()
    ].copy()

    # Keep only events that count in wOBA denominator
    pa = pa[pa["at_bat_event"].isin(WOBA_PA_DENOM_EVENTS)]

    # Map at_bat_event to wOBA weight
    pa["woba_num"] = pa["at_bat_event"].map(_EVENT_TO_WOBA_KEY).map(WOBA_WEIGHTS).fillna(0.0)

    # Normalize pitch type: collapse anything outside TRACKED_PITCH_TYPES to "other"
    pa["pitch_type_norm"] = pa["pitch_type"].where(
        pa["pitch_type"].isin(TRACKED_PITCH_TYPES),
        other="other"
    )

    # For each PA, keep only the LAST pitch (the one that determines the outcome).
    # The pitch_type of the final pitch is what we attribute the wOBA result to.
    # Sort by pitch_number to ensure we get the last pitch per PA.
    pa = pa.sort_values(["season", "game_pk", "at_bat_index", "pitch_number"])
    pa = pa.groupby(["season", "game_pk", "at_bat_index", "batter_id"], as_index=False).last()

    print(f"Deduped to {len(pa):,} PA events")

    # Compute wOBA per season per pitch type
    results = []
    all_types = list(TRACKED_PITCH_TYPES) + ["other"]

    for season in sorted(pa["season"].unique()):
        season_pa = pa[pa["season"] == season]

        for pt in all_types:
            pt_pa = season_pa[season_pa["pitch_type_norm"] == pt]

            if len(pt_pa) == 0:
                continue

            woba = pt_pa["woba_num"].sum() / len(pt_pa)
            results.append({
                "season": season,
                "pitch_type": pt,
                "woba": woba,
                "n_pa": len(pt_pa),
            })

    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("PER-PITCH-TYPE LEAGUE-AVERAGE WOBA BY SEASON (2015-2024, excl. 2020)")
    print("="*80)

    # Pivot for easier viewing
    pivot = df.pivot(index="season", columns="pitch_type", values="woba")
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n" + "="*80)
    print("STABILITY ANALYSIS (10-year statistics)")
    print("="*80)

    stability_results = []

    for pt in all_types:
        pt_data = df[df["pitch_type"] == pt]

        if len(pt_data) < 2:
            continue

        wobas = pt_data["woba"].values
        mean_woba = wobas.mean()
        std_woba = wobas.std()
        cv = std_woba / mean_woba if mean_woba > 0 else np.nan
        min_woba = wobas.min()
        max_woba = wobas.max()
        range_woba = max_woba - min_woba

        stationary = "YES" if cv < 0.05 else "NO"

        stability_results.append({
            "pitch_type": pt,
            "mean": mean_woba,
            "std": std_woba,
            "cv": cv,
            "min": min_woba,
            "max": max_woba,
            "range": range_woba,
            "stationary_5pct": stationary,
        })

    stab_df = pd.DataFrame(stability_results).sort_values("pitch_type")

    print(stab_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    # Check if ALL pitch types are stationary
    all_stationary = all(r["stationary_5pct"] == "YES" for r in stability_results)

    if all_stationary:
        print("\nALL pitch types have CV < 0.05 → STATIC per-pitch-type constants are VALID.")
        print("\nRecommended replacement for _LEAGUE_AVG_WOBA:")
        print("\n_LEAGUE_AVG_WOBA_BY_PITCH_TYPE: dict[str, float] = {")
        for r in stability_results:
            print(f'    "{r["pitch_type"]}": {r["mean"]:.4f},')
        print("}")

        # Also show the simple mean across all pitch types for comparison
        global_mean = stab_df["mean"].mean()
        print(f"\nGlobal mean (all pitch types): {global_mean:.4f}")
        print(f"Current static constant: 0.320")
        print(f"Difference: {abs(global_mean - 0.320):.4f}")

    else:
        print("\nSOME pitch types have CV >= 0.05 → seasonal adjustment MAY be needed.")
        print("Non-stationary pitch types:")
        for r in stability_results:
            if r["stationary_5pct"] == "NO":
                print(f"  - {r['pitch_type']}: CV={r['cv']:.4f}, range={r['range']:.4f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
