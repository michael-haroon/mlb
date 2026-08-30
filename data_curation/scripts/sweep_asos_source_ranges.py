#!/usr/bin/env python3.11
"""Range-sweep the RAW asos_obs archive: how much of it is not a measurement?

Why this exists. The visibility defect (raw vsby reaching 34,007 statute miles, further
than the circumference of the Earth) was found only because one such report happened to
survive into a built tensor and trip the artifact gate. That is late and lucky: the same
corruption in any other METAR column reaches a tensor by the same route, and a single
out-of-range reading propagates into several derived dims before anything notices.

So this checks the source directly, ahead of the builds that consume it. It reports, per
column, how many finite readings fall outside physical possibility, the observed range,
and which stations are responsible -- the station breakdown is what identifies the cause,
since in the measured sweep the offenders were almost entirely the non-US stations in the
venue map (CYQG for Comerica, MMMX/MMMY/MMTO for the Mexico series, EGWU for the London
series, RJTY for Tokyo).

The bounds are NOT defined here. Consumed columns import METAR_PHYSICAL_LIMITS from
mlb_dl.weather_asof -- the same table _drop_impossible_reports filters on -- so the sweep
and the filter cannot drift apart, and a violation reported here is exactly a report the
builder will drop. Columns the as-of path never reads are swept separately under
UNCONSUMED_LIMITS and reported as informational only: mslp and skyl* are measurably
corrupt in the archive but cannot reach a tensor, so filtering on them would discard
reports for no benefit.

Measured baseline (2026-08-30, 1,079 station-season files, 13,689,878 reports):
  consumed:    tmpf 66 [-80, 149] °F | dwpf 6 (min -268.6 °F) | sknt 5 (max 910 kt)
               gust 7 (max 525 kt)   | alti 319 [0, 99.99] inHg | p01i 44 (max 24 in/h)
               relh, drct, peak_wind_gust CLEAN
  vsby:        95,547 above the 10 SM ceiling (0.71%) — clamped, not dropped, because
               "10SM" means "10 or more" so the reading saturates and no signal is lost
  unconsumed:  mslp 221 (max 1120.6 hPa) | skyl1 1,711 (max 709,000 ft)
Consumed-column violations total 447 of 13.6M readings (0.003%).

Usage:
  python3.11 data_curation/scripts/sweep_asos_source_ranges.py
  python3.11 data_curation/scripts/sweep_asos_source_ranges.py --stations BOS CYQG
Exit code is nonzero only if a CONSUMED column regresses beyond its measured baseline,
so this is safe to wire into a gate.
"""

from __future__ import annotations

import argparse
import io
import sys
from collections import defaultdict
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "deep_learning"))

from mlb_dl.weather_asof import (  # noqa: E402
    METAR_PHYSICAL_LIMITS,
    METAR_VIS_CEILING_MI,
)

S3_BUCKET = "mlb-265753586044-us-east-1-an"
PREFIX = "data/weather/source=asos_obs/"

# Read nowhere in the as-of path, so corruption here cannot reach a tensor. Swept to keep
# the finding on the record (and to catch it becoming relevant if a dim is ever added),
# but never a failure.
UNCONSUMED_LIMITS = {
    "mslp": (850.0, 1090.0),      # hPa sea-level; global records 870-1084
    "skyl1": (0.0, 60000.0),      # ft cloud base
    "skyl2": (0.0, 60000.0),
    "skyl3": (0.0, 60000.0),
}

# vsby is clamped rather than dropped, so it is reported on its own terms: exceeding the
# ceiling is expected and harmless, and the count is informational.
VIS_COL = "vsby"

# Measured counts above. A sweep that finds materially more than this in a consumed
# column means the upstream feed changed, which is worth failing on -- the filter would
# then be discarding reports at a rate nobody has looked at. 5x the measured 447 leaves
# room for the seasons still being fetched without tolerating a regression.
MAX_CONSUMED_VIOLATIONS = 2500


def sweep(keys: list[str]) -> tuple[dict, int]:
    agg = defaultdict(lambda: dict(n=0, bad=0, lo=np.inf, hi=-np.inf, worst=None,
                                   stations=defaultdict(int)))
    for i, key in enumerate(keys):
        body = boto3.client("s3", region_name="us-east-1").get_object(
            Bucket=S3_BUCKET, Key=key)["Body"].read()
        df = pd.read_parquet(io.BytesIO(body))
        station = key.split("station=")[1].split("/")[0]
        allb = {**METAR_PHYSICAL_LIMITS, **UNCONSUMED_LIMITS,
                VIS_COL: (0.0, METAR_VIS_CEILING_MI)}
        for col, (lo, hi) in allb.items():
            if col not in df.columns:
                continue
            v = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            a = agg[col]
            a["n"] += v.size
            a["lo"] = min(a["lo"], float(v.min()))
            a["hi"] = max(a["hi"], float(v.max()))
            bad = v[(v < lo) | (v > hi)]
            if bad.size:
                a["bad"] += int(bad.size)
                a["stations"][station] += int(bad.size)
                w = float(bad[np.argmax(np.abs(bad - np.clip(bad, lo, hi)))])
                if a["worst"] is None or abs(w) > abs(a["worst"]):
                    a["worst"] = w
        if (i + 1) % 200 == 0:
            print(f"  ..{i + 1}/{len(keys)}", flush=True)
    return agg, len(keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", nargs="*", help="limit to these station codes")
    args = ap.parse_args()

    s3 = boto3.client("s3", region_name="us-east-1")
    pg = s3.get_paginator("list_objects_v2")
    keys = [o["Key"] for p in pg.paginate(Bucket=S3_BUCKET, Prefix=PREFIX)
            for o in p.get("Contents", []) if o["Key"].endswith(".parquet")]
    if args.stations:
        want = set(args.stations)
        keys = [k for k in keys if k.split("station=")[1].split("/")[0] in want]
    print(f"asos_obs parquet files: {len(keys)}", flush=True)

    agg, _ = sweep(keys)
    consumed_bad = 0
    print(f"\n{'column':16s}{'n_finite':>12s}{'out_of_bounds':>18s}"
          f"{'observed_range':>28s}{'worst':>14s}  class")
    groups = [("consumed", METAR_PHYSICAL_LIMITS),
              ("clamped", {VIS_COL: (0.0, METAR_VIS_CEILING_MI)}),
              ("unconsumed", UNCONSUMED_LIMITS)]
    for label, table in groups:
        for col, (lo, hi) in table.items():
            a = agg.get(col)
            if a is None or a["n"] == 0:
                print(f"{col:16s}{'absent':>12s}{'':>18s}{'':>28s}{'':>14s}  {label}")
                continue
            if label == "consumed":
                consumed_bad += a["bad"]
            rate = a["bad"] / a["n"]
            print(f"{col:16s}{a['n']:>12d}{a['bad']:>10d} ({rate:6.4%})"
                  f"  [{a['lo']:>11.4g},{a['hi']:>11.4g}]{str(a['worst']):>14s}  {label}")
            if a["bad"]:
                top = sorted(a["stations"].items(), key=lambda x: -x[1])[:6]
                print(f"{'':18s}bound [{lo:g},{hi:g}]  stations: {top}")

    print(f"\nconsumed-column violations: {consumed_bad} "
          f"(these reports are DROPPED by _drop_impossible_reports)")
    if consumed_bad > MAX_CONSUMED_VIOLATIONS:
        sys.exit(f"SOURCE SWEEP FAILED: {consumed_bad} consumed-column violations exceeds "
                 f"the {MAX_CONSUMED_VIOLATIONS} baseline ceiling — the upstream feed "
                 f"changed and the drop rate needs re-examining before it is trusted")
    print("SOURCE SWEEP PASSED (within the measured baseline)")


if __name__ == "__main__":
    main()
