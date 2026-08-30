#!/usr/bin/env python3.11
"""Measure how often each obs source column is unreported, on the venue stations only.

Why this exists. OBS_DIM_SOURCES decides which obs dims the mask may claim, and the only
way to know whether a build's coverage loss is the intended effect or a bug is to compare
it against the per-column null rate on the reports the tensor actually reads. Rates
measured over the whole ASOS archive are the wrong denominator -- the tensor never touches
the ~1,050 non-venue station files -- and using them made the first post-fix diff look like
a 2x over-disclaim when it was a population difference.

It also checks the identity the mask depends on: wind_u_10m is NaN exactly when direction
or speed was unreported, and speed is never NaN while u survives. If that breaks, the mask
is disclaiming dims for some reason other than an absent METAR group.

Measured baseline (season=2015, 662,306 reports across 30 venues):
    dims 2,3  wind_u_10m            6.4405%      dims 11  visibility     2.8458%
    dims 4    wind_speed_10m        3.0021%      dims 12  precipitation  0.2194%
    dims 5    wind_gusts_10m        3.0019%      dims 7   relative_hum   0.2387%
    dims 10   cloud_cover           3.7385%      dims 13  surface_press  0.0442%
                                                 dims 9   temperature    0.1001%
The wind decomposition is exact: u null 6.4405% = speed null 3.0021% + 3.4384% reporting a
measured speed with a variable (unreported) direction. Those 22,773 reports are why the
mask disclaims dims 2-3 without touching dim 4.

The resulting per-dim coverage losses in the built tensor were 8.43% (dims 2,3), 3.65%
(4,5), 3.28% (10), 3.16% (11), 0.15% (7), 0.08% (12) of claimed obs entries -- the same
order as each source rate, scattered around it in both directions because
select_asof_obs subsamples one report per (game, decision, hour) slot rather than reading
the archive uniformly.

Usage:
  python3.11 data_curation/scripts/measure_obs_source_nullity.py --season 2015
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "deep_learning"))

from mlb_dl.build_weather_asof import (  # noqa: E402
    STATION_MAP_KEY,
    _read_json,
    load_obs_for_venues,
    load_population,
)
from mlb_dl.weather_asof import OBS_DIM_SOURCES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    args = ap.parse_args()

    pop = load_population(args.season)
    if not len(pop):
        print(f"{args.season}: no population games")
        sys.exit(0)
    vids = sorted(pop["venue_id"].unique().tolist())
    frames = load_obs_for_venues(vids, args.season, _read_json(STATION_MAP_KEY))
    obs = pd.concat([f for f in frames.values() if f is not None and len(f)],
                    ignore_index=True)
    print(f"{args.season}: {len(obs):,} venue-station reports across {len(frames)} venues")

    # Report per source column, listing the dims each one gates, so a surprising rate
    # points straight at the dims whose coverage it explains.
    by_col: dict[str, list[int]] = {}
    for dim, cols in sorted(OBS_DIM_SOURCES.items()):
        for c in cols:
            by_col.setdefault(c, []).append(dim)
    fails = []
    for col, dims in sorted(by_col.items(), key=lambda kv: -len(kv[1])):
        if col not in obs.columns:
            print(f"  MISSING COLUMN {col} (gates dims {dims})")
            fails.append(col)
            continue
        print(f"  {col:24s} {obs[col].isna().mean():8.4%} null   gates dims {dims}")

    if "wind_u_10m" in obs.columns and "wind_speed_10m" in obs.columns:
        u_nan = obs["wind_u_10m"].isna()
        s_nan = obs["wind_speed_10m"].isna()
        variable = int((u_nan & ~s_nan).sum())
        orphan = int((~u_nan & s_nan).sum())
        print(f"\n  wind: {int(u_nan.sum()):,} missing direction, of which {variable:,} "
              f"report a measured speed (variable direction -> dim 4 must survive)")
        if orphan:
            print(f"  FAIL {orphan:,} reports have a speed but no direction components — "
                  f"the mask would disclaim dims 2-3 for a reason other than a missing "
                  f"METAR group")
            fails.append("wind identity")

    print(f"\n{'NULLITY MEASURED' if not fails else f'{len(fails)} PROBLEMS'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
