"""
Backfill era5 archives for venues fetch_weather's venue discovery never saw.

fetch_weather._load_venues() samples the first ~50 pitch files of ONE recent
season, so any venue absent from that sample was silently skipped by the
2026-08 weather rebuild — including three FULL-TIME current parks (Rogers
Centre 14, T-Mobile Park 680, Busch Stadium 2889) plus retired/special venues
(Turner Field 16, Globe Life Park 13, Sahlen Field 2756, internationals).
3,229 population games (12%) had no era5 rows at all.

This runner reuses fetch_weather's own _fetch_archive/_parse_hourly (identical
schema and units) with coordinates from the validated station_venue_map, and
writes the standard S3 layout. Only `era5` is fetched: the as-of weather v1
consumes era5 solely for the soil-moisture persistence dim; AQI dims are
dropped in v1 (CAMS starts 2022-07 — an 82%-missing mask-flip era regressor).

Usage:
  python3.11 data_curation/scripts/backfill_missing_venue_era5.py
"""

import io
import json
import sys
import time
from pathlib import Path

import boto3
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_weather import _fetch_archive  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
S3_BUCKET = "mlb-265753586044-us-east-1-an"

MISSING_VENUE_IDS = [13, 14, 16, 680, 2397, 2535, 2701, 2756, 2889,
                     3949, 5010, 5340, 5365, 5445, 6130]

s3 = boto3.client("s3", region_name="us-east-1")


def key_for(vid: int, year: int) -> str:
    return f"data/weather/source=era5/venue_id={vid}/year={year}.parquet"


def exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except Exception:
        return False


def main() -> None:
    with open(REPO / "data_curation" / "station_venue_map.json") as f:
        vmap = json.load(f)
    n_ok = n_skip = n_fail = 0
    for vid in MISSING_VENUE_IDS:
        m = vmap[str(vid)]
        for year in range(2015, 2027):
            key = key_for(vid, year)
            if exists(key):
                n_skip += 1
                continue
            try:
                df = _fetch_archive(vid, m["venue_lat"], m["venue_lon"], "era5", year)
            except Exception as exc:
                print(f"FAIL {m['venue_name']} {year}: {exc}")
                n_fail += 1
                time.sleep(5)
                continue
            if df is None or df.empty:
                n_fail += 1
                continue
            buf = io.BytesIO()
            df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
            print(f"ok {m['venue_name']} {year}: {len(df)} rows")
            n_ok += 1
            time.sleep(1.0)  # Open-Meteo courtesy throttle
    print(f"done: {n_ok} written, {n_skip} skipped, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
