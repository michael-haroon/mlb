"""
Hourly live feeder for the as-of weather tensor.

Every run appends to daily-accumulating S3 files (never overwrite-in-place —
intra-day issues/reports must all survive so availability filtering works):
  data/weather/source=asos_obs_live/station={ICAO}/date={D}.parquet
  data/weather/source=hrrr_asissued_live/date={D}.parquet

The inference engine's weather_asof.fetch_live_asof reads these and assembles
the same tensor the training builder wrote. Sources are the SAME as training:
AWC serves the identical METAR reports the IEM archive stores, and the HRRR
points come through the same fetch_nwp_asissued extraction (never Open-Meteo's
interpolated live products — quiet distribution shift).

Run inside live_daemon (hourly thread) or standalone:
  python3.11 data_curation/scripts/live_weather_asof.py once
  python3.11 data_curation/scripts/live_weather_asof.py loop --interval 3600
"""

from __future__ import annotations

import argparse
import io
import logging
import time

import boto3
import pandas as pd

from fetch_asos_obs import fetch_recent, load_station_venue_map, stations_from_map
from fetch_nwp_asissued import fetch_issue_points, freshest_issue, load_venue_points

S3_BUCKET = "mlb-265753586044-us-east-1-an"
OBS_LIVE_PREFIX = "data/weather/source=asos_obs_live"
HRRR_LIVE_PREFIX = "data/weather/source=hrrr_asissued_live"

# Leads 1..8 from the freshest issue cover [now+1h .. now+8h] valid hours; the
# previous hours' runs already covered earlier valid times in the daily file.
LIVE_FXX = range(1, 9)

logger = logging.getLogger("LIVE_WX_ASOF")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[LIVEWX] %(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)

_s3 = None


def s3():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3", region_name="us-east-1")
    return _s3


def _append_parquet(df: pd.DataFrame, key: str, dedupe_cols: list[str]) -> int:
    """Read-concat-dedupe-write; returns row count after append."""
    try:
        old = pd.read_parquet(io.BytesIO(s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()))
        df = pd.concat([old, df], ignore_index=True)
    except Exception:
        pass
    df = df.drop_duplicates(dedupe_cols, keep="last")
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
    s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    return len(df)


def refresh_obs(now: pd.Timestamp) -> None:
    stations = stations_from_map(load_station_venue_map())
    # 6h window: hourly appends only need ~1h, but a daemon restart must not
    # leave an obs hole across the game window (dedupe absorbs the overlap).
    df = fetch_recent(stations, lookback_hours=6)
    if df.empty:
        logger.warning("AWC returned no reports")
        return
    for st, grp in df.groupby("station"):
        day = grp["valid_utc"].max().normalize()
        n = _append_parquet(grp, f"{OBS_LIVE_PREFIX}/station={st}/date={day:%Y-%m-%d}.parquet",
                            dedupe_cols=["valid_utc"])
        logger.debug(f"obs {st}: {len(grp)} new, {n} total")
    logger.info(f"obs refresh: {df['station'].nunique()} stations, {len(df)} reports")


def refresh_hrrr(now: pd.Timestamp) -> None:
    points = load_venue_points()
    issue = freshest_issue(now)
    frames = []
    for fxx in LIVE_FXX:
        df = fetch_issue_points(issue, fxx, points, all_points=points)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        logger.warning(f"no HRRR data for issue {issue} — will retry next cycle")
        return
    out = pd.concat(frames, ignore_index=True)
    day = now.normalize()
    n = _append_parquet(out, f"{HRRR_LIVE_PREFIX}/date={day:%Y-%m-%d}.parquet",
                        dedupe_cols=["venue_id", "issue_time_utc", "valid_time_utc"])
    logger.info(f"hrrr refresh: issue {issue:%H}Z, {len(out)} rows appended ({n} in day file)")


def run_once() -> None:
    now = pd.Timestamp.now(tz="UTC")
    try:
        refresh_obs(now)
    except Exception:
        logger.error("obs refresh failed", exc_info=True)
    try:
        refresh_hrrr(now)
    except Exception:
        logger.error("hrrr refresh failed", exc_info=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("once")
    lp = sub.add_parser("loop")
    lp.add_argument("--interval", type=int, default=3600)
    args = ap.parse_args()
    if args.cmd == "once":
        run_once()
    else:
        while True:
            run_once()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
