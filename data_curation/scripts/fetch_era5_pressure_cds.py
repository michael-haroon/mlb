#!/usr/bin/env python3
"""
Fetch ERA5 pressure-level reanalysis from Copernicus CDS API.

CDS returns gridded NetCDF for a bounding box; we extract the nearest
0.25-degree grid point for each MLB venue and write the same parquet
layout as fetch_weather.py:
  s3://{bucket}/data/weather/source=era5_pressure/venue_id={id}/year={year}.parquet

Variables match ERA5_PRESSURE_VARS in fetch_weather.py:
  temperature, relative_humidity, cloud_cover, wind_speed, wind_direction,
  geopotential_height — at 19 pressure levels (1000…30 hPa)

CDS job queue: max 2 concurrent requests. Each year/month batch is one job.
Run with --test for a single-venue 1-week probe to measure queue latency.

Usage:
  conda run -n pred python fetch_era5_pressure_cds.py --test
  conda run -n pred python fetch_era5_pressure_cds.py --mode backfill --start-year 2015
"""

import argparse
import io
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import boto3
import cdsapi
import netCDF4 as nc
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parents[2] / ".env")

CDS_URL = os.environ["CDS_URL"].strip()
CDS_KEY = os.environ["CDS_KEY"].strip()

S3_BUCKET = "mlb-265753586044-us-east-1-an"
S3_PREFIX = "data"
S3_REGION = "us-east-1"

# Bounding box covering all MLB venues (+ Toronto) with 0.5° padding
# ERA5 grid: 0.25° resolution; CDS snaps area to nearest grid points
CONUS_AREA = [50.0, -125.0, 24.0, -66.0]   # [N, W, S, E]

ERA5_PRESSURE_LEVELS = [
    1000, 975, 950, 925, 900, 850, 800, 700, 600, 500,
    400, 300, 250, 200, 150, 100, 70, 50, 30,
]

# CDS variable names → what we want. Wind comes as u+v; we compute speed+direction.
CDS_VARIABLES = [
    "temperature",             # K → °C
    "relative_humidity",       # % (0-100)
    "fraction_of_cloud_cover", # 0-1 → ×100 → %
    "u_component_of_wind",     # m/s → combined with v for speed/direction
    "v_component_of_wind",     # m/s
    "geopotential",            # m²/s² → ÷9.80665 → meters
]

# Matches ERA5_PRESSURE_LEVELS × ERA5_PRESSURE_BASE_VARS in fetch_weather.py
def _colname(base: str, level: int) -> str:
    return f"{base}_{level}hPa"

# Max concurrent CDS jobs (hard quota: 2 active per user)
MAX_CONCURRENT = 2

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[CDS] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cds_era5")

# ── S3 helpers ────────────────────────────────────────────────────────────────
_s3 = None
def _get_s3():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3", region_name=S3_REGION)
    return _s3

def _s3_key(venue_id: int, year: int) -> str:
    return f"{S3_PREFIX}/weather/source=era5_pressure/venue_id={venue_id}/year={year}.parquet"

def _s3_exists(key: str) -> bool:
    try:
        _get_s3().head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError as e:
        return e.response["Error"]["Code"] in ("404", "NoSuchKey") and False

def _write_s3(df: pd.DataFrame, key: str) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
    buf.seek(0)
    _get_s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())

# ── Venue loading (same logic as fetch_weather.py) ───────────────────────────
def _load_venues() -> pd.DataFrame:
    COLS = ["venue_id", "venue_latitude", "venue_longitude", "venue_name"]
    for year in range(datetime.now().year, 2014, -1):
        prefix = f"{S3_PREFIX}/season={year}/pitches_batch_"
        try:
            resp = _get_s3().list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, MaxKeys=50)
            keys = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".parquet")]
            if not keys:
                continue
            accumulated = pd.DataFrame()
            for key in keys:
                buf = io.BytesIO()
                _get_s3().download_fileobj(S3_BUCKET, key, buf)
                buf.seek(0)
                df = pd.read_parquet(buf, columns=COLS)
                accumulated = (
                    pd.concat([accumulated, df])
                    .dropna(subset=["venue_latitude", "venue_longitude"])
                    .drop_duplicates("venue_id")
                )
            if len(accumulated) >= 20:
                log.info(f"Loaded {len(accumulated)} venues from season={year}")
                return accumulated.reset_index(drop=True)
        except Exception as exc:
            log.debug(f"season={year} failed: {exc}")
    raise RuntimeError("Could not load venues from S3")

# ── CDS client ────────────────────────────────────────────────────────────────
def _make_client() -> cdsapi.Client:
    return cdsapi.Client(url=CDS_URL, key=CDS_KEY, quiet=True, progress=False)

# ── NetCDF → per-venue DataFrame ─────────────────────────────────────────────
def _extract_venues(nc_path: str, venues: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """
    Read a monthly ERA5 pressure-level NetCDF and extract the nearest
    0.25-degree grid point for each venue. Returns {venue_id: DataFrame}.
    """
    ds = nc.Dataset(nc_path)

    lats = ds.variables["latitude"][:]
    lons = ds.variables["longitude"][:]
    times = nc.num2date(
        ds.variables["time"][:],
        ds.variables["time"].units,
        calendar="gregorian",
    )
    timestamps = pd.to_datetime([t.isoformat() for t in times], utc=True)

    # Pre-compute nearest grid indices for each venue
    def _nearest(arr, val):
        return int(np.argmin(np.abs(arr - val)))

    venue_indices = {
        int(row.venue_id): (
            _nearest(lats, row.venue_latitude),
            _nearest(lons, row.venue_longitude % 360 if lons.min() >= 0 else row.venue_longitude),
        )
        for _, row in venues.iterrows()
    }

    results = {}
    for vid, (li, loi) in venue_indices.items():
        rows = {"time": timestamps}
        for level in ERA5_PRESSURE_LEVELS:
            level_str = str(level)
            level_idx = list(ds.variables["level"][:]).index(level) if "level" in ds.variables else None

            def _get(varname):
                v = ds.variables[varname]
                # shape: (time, level, lat, lon) or (time, lat, lon)
                if level_idx is not None and v.ndim == 4:
                    return v[:, level_idx, li, loi]
                elif v.ndim == 3:
                    return v[:, li, loi]
                return v[:]

            t_k   = np.array(_get("t"))   if "t"  in ds.variables else np.array(_get("temperature"))
            rh    = np.array(_get("r"))   if "r"  in ds.variables else np.array(_get("relative_humidity"))
            cc    = np.array(_get("cc"))  if "cc" in ds.variables else np.array(_get("fraction_of_cloud_cover"))
            u     = np.array(_get("u"))   if "u"  in ds.variables else np.array(_get("u_component_of_wind"))
            v_    = np.array(_get("v"))   if "v"  in ds.variables else np.array(_get("v_component_of_wind"))
            z     = np.array(_get("z"))   if "z"  in ds.variables else np.array(_get("geopotential"))

            rows[_colname("temperature",        level)] = t_k - 273.15          # K → °C
            rows[_colname("relative_humidity",  level)] = rh                    # already %
            rows[_colname("cloud_cover",        level)] = cc * 100.0            # 0-1 → %
            rows[_colname("wind_speed",         level)] = np.sqrt(u**2 + v_**2) # m/s
            rows[_colname("wind_direction",     level)] = (
                (180.0 / math.pi) * np.arctan2(u, v_) + 180.0                  # met convention
            )
            rows[_colname("geopotential_height", level)] = z / 9.80665          # m²/s² → m

        results[vid] = pd.DataFrame(rows)

    ds.close()
    return results

# ── CDS request: one month ────────────────────────────────────────────────────
def _fetch_month(client: cdsapi.Client, year: int, month: int,
                 area: list, tmp_dir: Path) -> str:
    """Submit one CDS request for a calendar month. Blocks until download done."""
    days = pd.date_range(f"{year}-{month:02d}-01",
                         periods=1, freq="MS")[0]
    n_days = (days + pd.offsets.MonthEnd(0)).day

    out = tmp_dir / f"era5_pressure_{year}_{month:02d}.nc"
    if out.exists():
        log.info(f"  cache hit: {out.name}")
        return str(out)

    t0 = time.time()
    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": CDS_VARIABLES,
            "pressure_level": [str(lv) for lv in ERA5_PRESSURE_LEVELS],
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, n_days + 1)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area,
            "format": "netcdf",
            "download_format": "unarchived",
        },
        str(out),
    )
    log.info(f"  {year}-{month:02d} downloaded in {time.time()-t0:.0f}s → {out.stat().st_size/1e6:.0f} MB")
    return str(out)

# ── Backfill one year ─────────────────────────────────────────────────────────
def _backfill_year(year: int, venues: pd.DataFrame, tmp_dir: Path, force: bool = False) -> None:
    # Skip if all venues already written
    if not force:
        done = all(
            _s3_exists(_s3_key(int(v.venue_id), year))
            for _, v in venues.iterrows()
        )
        if done:
            log.info(f"{year}: all venues already in S3, skipping")
            return

    client = _make_client()
    accumulated: dict[int, list[pd.DataFrame]] = {int(v.venue_id): [] for _, v in venues.iterrows()}

    for month in range(1, 13):
        log.info(f"{year}-{month:02d}: requesting CDS...")
        nc_path = _fetch_month(client, year, month, CONUS_AREA, tmp_dir)
        monthly = _extract_venues(nc_path, venues)
        for vid, df in monthly.items():
            accumulated[vid].append(df)
        Path(nc_path).unlink(missing_ok=True)  # free disk after each month

    log.info(f"{year}: writing {len(accumulated)} venues to S3...")
    for vid, frames in accumulated.items():
        full = pd.concat(frames, ignore_index=True)
        key = _s3_key(vid, year)
        _write_s3(full, key)
        log.debug(f"  wrote {key}  ({len(full)} rows, {full.shape[1]} cols)")

    log.info(f"{year}: done")

# ── Quick test ────────────────────────────────────────────────────────────────
def _run_test():
    """
    One venue (Yankee Stadium), 1 week, all pressure levels.
    Prints queue + download time so we know CDS latency.
    """
    log.info("=== CDS SPEED TEST: Yankee Stadium, 2022-06-01..07 ===")
    client = _make_client()
    tmp = Path("/tmp/cds_test")
    tmp.mkdir(exist_ok=True)
    out = tmp / "era5_pressure_test.nc"

    t0 = time.time()
    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": CDS_VARIABLES,
            "pressure_level": [str(lv) for lv in ERA5_PRESSURE_LEVELS],
            "year": "2022",
            "month": "06",
            "day": [f"{d:02d}" for d in range(1, 8)],   # 7 days only
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [41.5, -74.5, 40.0, -73.0],          # tight box NYC
            "format": "netcdf",
            "download_format": "unarchived",
        },
        str(out),
    )
    elapsed = time.time() - t0
    size_mb = out.stat().st_size / 1e6
    log.info(f"Download complete: {size_mb:.1f} MB in {elapsed:.1f}s ({size_mb/elapsed:.1f} MB/s)")

    # Quick sanity: read one variable
    ds = nc.Dataset(str(out))
    lats = ds.variables["latitude"][:]
    lons = ds.variables["longitude"][:]
    t_var = ds.variables.get("t") or ds.variables.get("temperature")
    log.info(f"Grid: {len(lats)} lats × {len(lons)} lons, vars: {list(ds.variables.keys())}")
    if t_var is not None:
        t_sample = float(t_var[0, 0, 0, 0]) - 273.15
        log.info(f"temperature_1000hPa at t=0, point=0: {t_sample:.1f} °C")
    ds.close()
    out.unlink()
    log.info("=== TEST DONE ===")

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test",        action="store_true", help="speed test only")
    ap.add_argument("--mode",        choices=["backfill"], default="backfill")
    ap.add_argument("--start-year",  type=int, default=2015)
    ap.add_argument("--end-year",    type=int, default=datetime.now().year)
    ap.add_argument("--force",       action="store_true")
    ap.add_argument("--tmp-dir",     default="/tmp/cds_era5", help="scratch dir for NetCDF downloads")
    args = ap.parse_args()

    if args.test:
        _run_test()
    else:
        tmp = Path(args.tmp_dir)
        tmp.mkdir(parents=True, exist_ok=True)
        venues = _load_venues()
        log.info(f"Venues: {len(venues)} | Years: {args.start_year}–{args.end_year} | force={args.force}")

        # Sequential years (CDS queue only allows 2 concurrent; monthly batches inside each year
        # already max out that quota with 1 active job, so no benefit to parallel years)
        for year in range(args.start_year, args.end_year + 1):
            _backfill_year(year, venues, tmp, force=args.force)

        log.info("=== CDS ERA5 pressure backfill complete ===")
