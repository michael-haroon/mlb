"""Assert every population game date in a season has an hrrr_asissued archive file.

WHY THIS EXISTS (2026-08-30): the weather backfill is driven by the game population, so a
truncated population silently truncates the archive to match — and nothing downstream
notices. `game_meta` stopped at 2026-06-20, so `hrrr_asissued` also stopped at exactly
2026-06-20. When the store was rebuilt and 916 new 2026 games appeared, building
`weather_asof` against them would have emitted rows whose forecast channel is entirely
mask=0: structurally valid, statistically empty, and invisible to a row-count check.

`verify_weather_archives.py completeness` samples an existing archive for holes; it cannot
see dates that were never attempted because the population did not yet contain them. This
script closes that specific gap by diffing the archive against the CURRENT population.

Missing dates are classified, because one class is irreducible and must not be reported as a
failure — a check that can never go green gets ignored, which is how the 2026 gap survived:

  UNCOVERABLE  every population game that day is outside the HRRR CONUS grid, so no file can
               ever exist. In practice this is the international series (Tokyo, Seoul, London,
               Mexico City) — e.g. 2025-03-18/19 is the Dodgers/Cubs Tokyo Series, which are
               regular-season game types and therefore in the population.
  MISSING      at least one CONUS game that day has no archive file. Actionable: either the
               backfill never ran for that date, or it ran and lost the data.

Exit 0 = no MISSING dates (UNCOVERABLE ones are reported and tolerated). Exit 1 = MISSING.

Usage:
    python data_curation/scripts/verify_asof_population_coverage.py --year 2026
    python data_curation/scripts/verify_asof_population_coverage.py --year 2026 --quiet
"""
from __future__ import annotations

import argparse
import sys

import boto3
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

S3_BUCKET = "mlb-265753586044-us-east-1-an"
FS_PREFIX = "deep_learning/feature_store"
ARCHIVE_PREFIX = "data/weather/source=hrrr_asissued"

# Must stay in sync with build_weather_asof.py — these define which games get a weather
# tensor at all, and therefore which dates the archive is required to cover.
POP_MIN_DATE = "2015-01-15"
POP_GAME_TYPES = ["R", "F", "D", "L", "W"]

# Conservative bounding box around the HRRR CONUS Lambert-conformal grid. Deliberately loose:
# its only job is to separate "continental North America" from the overseas series, and a
# false "inside" is the safe error (it reports an actionable MISSING for a human to look at,
# rather than silently excusing a real hole). Note Rogers Centre (43.64N) is INSIDE — HRRR
# does cover Toronto, contra the stale comment on TORONTO_VENUE_ID in fetch_weather.py.
CONUS_LAT = (21.0, 53.0)
CONUS_LON = (-135.0, -60.0)


def _in_conus(lat: float, lon: float) -> bool:
    if pd.isna(lat) or pd.isna(lon):
        return True  # unknown coords -> assume it needed a file, so the gap stays actionable
    return CONUS_LAT[0] <= lat <= CONUS_LAT[1] and CONUS_LON[0] <= lon <= CONUS_LON[1]


def population_frame(year: int) -> pd.DataFrame:
    """Population games for `year` with venue coords, flagged for CONUS membership."""
    fs = pafs.S3FileSystem(region="us-east-1")
    table = ds.dataset(
        f"{S3_BUCKET}/{FS_PREFIX}/game_meta.parquet", filesystem=fs, format="parquet"
    ).to_table(columns=["game_pk", "game_date", "game_type_code", "venue_name",
                        "venue_latitude", "venue_longitude"]).to_pandas()
    table["game_date"] = pd.to_datetime(table["game_date"])
    pop = table[
        (table["game_date"] >= POP_MIN_DATE)
        & (table["game_type_code"].isin(POP_GAME_TYPES))
        & (table["game_date"].dt.year == year)
    ].copy()
    pop["date"] = pop["game_date"].dt.strftime("%Y-%m-%d")
    pop["in_conus"] = [
        _in_conus(a, o) for a, o in zip(pop["venue_latitude"], pop["venue_longitude"])
    ]
    return pop


def archive_dates(year: int) -> set[str]:
    """Dates present in the flat date=YYYY-MM-DD.parquet archive for `year`."""
    s3 = boto3.client("s3")
    found: set[str] = set()
    # The archive is a single flat prefix (~2k objects across all seasons), so paginate
    # rather than assuming one ListObjectsV2 page.
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=S3_BUCKET, Prefix=f"{ARCHIVE_PREFIX}/date={year}-"
    ):
        for obj in page.get("Contents", []):
            stem = obj["Key"].rsplit("date=", 1)[-1]
            if stem.endswith(".parquet"):
                found.add(stem[: -len(".parquet")])
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--quiet", action="store_true", help="print only the verdict line")
    args = ap.parse_args()

    pop = population_frame(args.year)
    if pop.empty:
        print(f"COVERAGE n/a — {args.year} has no population games in game_meta")
        return 0
    need = sorted(pop["date"].unique())
    have = archive_dates(args.year)
    absent = [d for d in need if d not in have]

    # A date is only excusable if EVERY population game on it is outside CONUS. A mixed date
    # (overseas opener plus a domestic game) still requires the file.
    conus_by_date = pop.groupby("date")["in_conus"].any()
    missing = [d for d in absent if bool(conus_by_date.get(d, True))]
    uncoverable = [d for d in absent if d not in missing]

    if not args.quiet:
        print(f"{args.year}: {len(pop):,} population games over {len(need)} dates "
              f"({need[0]} .. {need[-1]})")
        print(f"  archive has {len(need) - len(absent)}/{len(need)}")
    for d in uncoverable:
        venues = sorted(pop.loc[pop["date"] == d, "venue_name"].dropna().unique())
        print(f"  UNCOVERABLE {d} — outside HRRR CONUS grid: {', '.join(venues) or 'unknown'}")
    if missing:
        print(f"COVERAGE FAILED — {len(missing)} CONUS population dates have no "
              f"hrrr_asissued file: {missing[0]} .. {missing[-1]}")
        if not args.quiet:
            for d in missing:
                print(f"    {d}")
        return 1
    print(f"COVERAGE OK — all {len(need) - len(uncoverable)} coverable population dates of "
          f"{args.year} present ({len(uncoverable)} uncoverable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
