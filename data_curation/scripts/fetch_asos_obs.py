"""
ASOS/METAR station observations — the OBSERVED channel of the as-of weather tensor.

Two modes, one physical data source:
  backfill — IEM ASOS archive (mesonet.agron.iastate.edu) per station-year, 2015→present.
  recent   — NOAA AWC (aviationweather.gov) latest reports for live inference.

IEM archives and AWC serve the SAME station METAR reports, so train/inference parity
for the observation channel holds by identity — no NWP-model-matching argument needed
(measured 2026-08-29: IEM 1 station-year in 1.3s; AWC 234ms with `receiptTime`).

Values are stored in RAW METAR units (°F, knots, inHg, miles, feet); unit conversion
and physics (air density, park-relative wind, ...) happen in mlb_dl.weather_asof at
tensor-assembly time, shared between the training builder and the live path.

S3 layout (year-partitioned, mirrors fetch_weather.py archives):
  data/weather/source=asos_obs/station={ICAO}/year={year}.parquet
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd
import requests
from botocore.exceptions import ClientError

# ── Storage (same bucket/prefix as fetch_weather.py) ─────────────────────────
S3_BUCKET = "mlb-265753586044-us-east-1-an"
S3_PREFIX = "data"
S3_REGION = "us-east-1"

DATA_DIR = "data"
LOG_DIR = os.path.join(DATA_DIR, "logs")

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
AWC_URL = "https://aviationweather.gov/api/data/metar"
REQUEST_TIMEOUT = 120

# IEM asks for courtesy throttling; it has no hard limit. One request per
# station-year is ~360 requests total for the full backfill, so a short pause
# costs minutes and keeps us an obviously well-behaved client.
IEM_PAUSE_SECONDS = 1.0

# Dissemination lag — measured 2026-08-30 via AWC receiptTime-reportTime over
# 33 recent reports at 8 major airports: p50 4.1 / p95 8.9 / p99 29.2 min.
# 10 min covers p95; the p99 tail is late corrections/re-transmissions whose
# values barely differ from the on-time reading, and the LIVE path uses the
# per-report measured receiptTime, so any residual optimism is training-side
# only and ~1% of reports. Re-measure over a full season of live polls before
# tightening below 10.
ASOS_AVAILABILITY_LAG_MIN = 10

# Every game-relevant METAR field IEM serves (all reconstructible live from the
# same AWC reports — parsed fields or rawOb remarks — so parity holds by
# identity). Kept raw — see module docstring. Deliberately absent: `feel`
# (derived from temp/RH/wind in our own physics) and raw `metar` text.
IEM_FIELDS = [
    "tmpf", "dwpf", "relh", "drct", "sknt", "gust", "alti", "mslp",
    "vsby", "skyc1", "skyl1", "skyc2", "skyl2", "skyc3", "skyl3",
    "skyc4", "skyl4", "p01i",
    # Present-weather codes (TS/RA/FG/...): the only field that says WHAT is
    # falling and whether thunder is in the area — delay/PPD signal.
    "wxcodes",
    # Hour-peak wind from remarks (PK WND) — gustiness beyond the snapshot gust.
    "peak_wind_gust", "peak_wind_drct", "peak_wind_time",
    # Freezing precip + snow cover: near-constant zero in-season, archived for
    # completeness (edge April/October games); tensor inclusion decided later.
    "ice_accretion_1hr", "ice_accretion_3hr", "ice_accretion_6hr",
    "snowdepth",
]

# Columns that are not numeric readings (everything else passes to_numeric).
NON_NUMERIC_FIELDS = {"skyc1", "skyc2", "skyc3", "skyc4", "wxcodes", "peak_wind_time"}

logger = logging.getLogger("ASOS_INGEST")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    os.makedirs(LOG_DIR, exist_ok=True)
    fh = logging.FileHandler(os.path.join(LOG_DIR, "asos_ingest.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[ASOS] %(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)


# ── boto3 singleton ───────────────────────────────────────────────────────────
_s3_client = None


def _get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=S3_REGION)
    return _s3_client


def _s3_key_exists(key: str) -> bool:
    try:
        _get_s3().head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def _write_parquet_s3(df: pd.DataFrame, key: str) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
    buf.seek(0)
    _get_s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())


def _write_parquet_local(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)


def _obs_key(station: str, year: int) -> str:
    return f"{S3_PREFIX}/weather/source=asos_obs/station={station}/year={year}.parquet"


# ── Station list ──────────────────────────────────────────────────────────────
def load_station_venue_map(path: Optional[str] = None) -> dict:
    """station_venue_map.json: venue_id -> primary/backup ICAO (built by T0.1)."""
    import json

    p = Path(path or Path(__file__).resolve().parent.parent / "station_venue_map.json")
    with open(p) as f:
        return json.load(f)


def stations_from_map(venue_map: dict) -> list[str]:
    out: set[str] = set()
    for m in venue_map.values():
        out.add(m["primary_station"])
        out.add(m["backup_station"])
    return sorted(out)


# ── Normalization ─────────────────────────────────────────────────────────────
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """IEM CSV -> typed frame with valid_utc / available_time_utc.

    'M' is IEM's missing marker and 'T' is trace precip; both arrive in numeric
    columns, so every numeric field goes through to_numeric(errors='coerce')
    with trace mapped to 0.0 first (a trace is a real, ~zero measurement — NaN
    would wrongly mask a populated report).
    """
    df = df.rename(columns={"valid": "valid_utc"})
    df["valid_utc"] = pd.to_datetime(df["valid_utc"], utc=True)
    numeric = [c for c in IEM_FIELDS if c not in NON_NUMERIC_FIELDS]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].replace({"T": "0.0"}), errors="coerce")
    for c in NON_NUMERIC_FIELDS - {"peak_wind_time"}:
        if c in df.columns:
            df[c] = df[c].mask(df[c] == "M", pd.NA)
    if "peak_wind_time" in df.columns:
        df["peak_wind_time"] = pd.to_datetime(
            df["peak_wind_time"].mask(df["peak_wind_time"] == "M"), utc=True,
            errors="coerce")
    df["available_time_utc"] = df["valid_utc"] + pd.Timedelta(minutes=ASOS_AVAILABILITY_LAG_MIN)
    return df


# ── Backfill (IEM) ────────────────────────────────────────────────────────────
def _iem_params(station: str, year: int) -> dict:
    """Request params for one station-year.

    report_type is load-bearing: without it IEM interleaves 5-minute MADIS
    (HFMETAR, type 1) rows that carry only wind/altimeter — measured 91% NaN
    tmpf at major airports — and a "latest report in window" selection would
    routinely land on one, zeroing dims behind a populated obs_mask. Types 3
    (routine hourly METAR) + 4 (specials) are complete reports; specials still
    fire on significant weather changes, which is when freshness matters.
    """
    return {
        "station": station,
        "data": ",".join(IEM_FIELDS),
        "year1": year, "month1": 1, "day1": 1,
        "year2": year + 1, "month2": 1, "day2": 1,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "missing": "M",
        "trace": "T",
        "report_type": [3, 4],
    }


def fetch_station_year(station: str, year: int) -> Optional[pd.DataFrame]:
    """One station-year of METAR (routine + specials) from the IEM archive."""
    resp = requests.get(IEM_URL, params=_iem_params(station, year), timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    if not resp.text.strip() or resp.text.startswith("ERROR"):
        logger.warning(f"{station} {year}: empty/error response from IEM")
        return None
    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty:
        logger.warning(f"{station} {year}: 0 rows")
        return None
    return _normalize(df)


def run_backfill(start_year: int, end_year: int, local: bool = False,
                 force: bool = False, stations: Optional[list[str]] = None) -> None:
    stations = stations or stations_from_map(load_station_venue_map())
    logger.info(f"Backfill {start_year}-{end_year} for {len(stations)} stations")
    n_ok = n_skip = n_fail = 0
    for station in stations:
        for year in range(start_year, end_year + 1):
            key = _obs_key(station, year)
            if not force and not local and _s3_key_exists(key):
                n_skip += 1
                continue
            try:
                df = fetch_station_year(station, year)
            except requests.RequestException as exc:
                logger.error(f"{station} {year}: {exc}")
                n_fail += 1
                continue
            if df is None:
                n_fail += 1
                continue
            if local:
                _write_parquet_local(df, Path(DATA_DIR) / "weather" / "source=asos_obs"
                                     / f"station={station}" / f"year={year}.parquet")
            else:
                _write_parquet_s3(df, key)
            logger.info(f"{station} {year}: {len(df)} obs rows")
            n_ok += 1
            time.sleep(IEM_PAUSE_SECONDS)
    logger.info(f"Backfill done: {n_ok} written, {n_skip} skipped (exist), {n_fail} failed")


# ── METAR remark parsers (live parity for fields IEM pre-parses) ─────────────
# Time alternation must try hhmm before mm, and be boundary-anchored —
# (\d{2}|\d{4}) would match only "23" of "2317".
_PK_WND_RE = re.compile(r"PK WND (\d{3})(\d{2,3})/(\d{4}|\d{2})\b")
_ICE_RE = re.compile(r"\bI([136])(\d{3})\b")
_SNOWDEPTH_RE = re.compile(r"\b4/(\d{3})\b")


def parse_peak_wind(raw_metar: str, report_time: pd.Timestamp):
    """PK WND dddss/(hh)mm remark -> (gust_kt, drct_deg, time_utc).

    The time group is minutes-only when the peak occurred within the report's
    hour; hhmm otherwise. Matches IEM's peak_wind_* parsed columns."""
    m = _PK_WND_RE.search(raw_metar or "")
    if not m:
        return None, None, None
    drct, sknt, t = int(m.group(1)), int(m.group(2)), m.group(3)
    if len(t) == 2:
        when = report_time.replace(minute=int(t), second=0, microsecond=0)
    else:
        when = report_time.replace(hour=int(t[:2]), minute=int(t[2:]),
                                   second=0, microsecond=0)
    if when > report_time:  # peak from the previous hour/day wraps backward
        when -= pd.Timedelta(hours=1) if len(t) == 2 else pd.Timedelta(days=1)
    return float(sknt), float(drct), when


def parse_ice_accretion(raw_metar: str) -> dict:
    """I1/I3/I6 remarks (hundredths of an inch) -> inches, matching IEM units."""
    out = {"ice_accretion_1hr": None, "ice_accretion_3hr": None, "ice_accretion_6hr": None}
    for hours, hundredths in _ICE_RE.findall(raw_metar or ""):
        out[f"ice_accretion_{hours}hr"] = int(hundredths) / 100.0
    return out


def parse_snowdepth(raw_metar: str):
    """4/sss remark -> snow depth in inches."""
    m = _SNOWDEPTH_RE.search(raw_metar or "")
    return float(m.group(1)) if m else None


# ── Live (AWC) ────────────────────────────────────────────────────────────────
def fetch_recent(stations: list[str], lookback_hours: int = 6) -> pd.DataFrame:
    """Latest METARs from NOAA AWC — same physical reports the IEM archive stores.

    Returns the backfill schema so weather_asof consumes both identically.
    AWC serves parsed fields in different units for a few (altim in hPa, temp in
    °C); converted here so raw-unit semantics match IEM columns.

    Station ids: the map stores IEM ids (US = 3-letter, e.g. BOS) while AWC
    only answers ICAO (KBOS) — measured 2026-08-30: raw 3-letter ids returned
    12/90 stations (internationals only). US ids get the K prefix for the
    query and are stripped back so the output keys match the archive's.
    """
    icao_by_station = {st: (f"K{st}" if len(st) == 3 else st) for st in stations}
    station_by_icao = {v: k for k, v in icao_by_station.items()}
    resp = requests.get(AWC_URL, params={
        "ids": ",".join(icao_by_station.values()), "format": "json",
        "hours": lookback_hours,
    }, timeout=30)
    resp.raise_for_status()
    rows = []
    for r in resp.json():
        clouds = r.get("clouds") or []
        def _cloud(i, field):
            return clouds[i].get(field) if i < len(clouds) else None
        report_time = pd.to_datetime(r.get("reportTime"), utc=True)
        raw = r.get("rawOb") or ""
        pk_gust, pk_drct, pk_time = parse_peak_wind(raw, report_time)
        rows.append({
            "station": station_by_icao.get(r.get("icaoId"), r.get("icaoId")),
            "valid_utc": report_time,
            # AWC gives the actual receipt timestamp — use it instead of the
            # fixed lag so live availability is measured, not assumed.
            "available_time_utc": pd.to_datetime(r.get("receiptTime"), utc=True),
            "tmpf": _c_to_f(r.get("temp")),
            "dwpf": _c_to_f(r.get("dewp")),
            "relh": None,  # derivable from temp/dewp in weather_asof
            "drct": r.get("wdir") if isinstance(r.get("wdir"), (int, float)) else None,
            "sknt": r.get("wspd"),
            "gust": r.get("wgst"),
            "alti": _hpa_to_inhg(r.get("altim")),
            "mslp": r.get("slp"),
            "vsby": pd.to_numeric(r.get("visib"), errors="coerce"),
            "skyc1": _cloud(0, "cover"), "skyl1": _cloud(0, "base"),
            "skyc2": _cloud(1, "cover"), "skyl2": _cloud(1, "base"),
            "skyc3": _cloud(2, "cover"), "skyl3": _cloud(2, "base"),
            "skyc4": _cloud(3, "cover"), "skyl4": _cloud(3, "base"),
            "p01i": r.get("precip"),
            "wxcodes": r.get("wxString"),
            "peak_wind_gust": pk_gust, "peak_wind_drct": pk_drct,
            "peak_wind_time": pk_time,
            **parse_ice_accretion(raw),
            "snowdepth": parse_snowdepth(raw),
        })
    df = pd.DataFrame(rows)
    logger.debug(f"AWC recent: {len(df)} reports for {len(stations)} stations")
    return df


def _c_to_f(c) -> Optional[float]:
    return None if c is None else c * 9.0 / 5.0 + 32.0


def _hpa_to_inhg(hpa) -> Optional[float]:
    return None if hpa is None else hpa / 33.8639


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="ASOS/METAR observation ingestion")
    sub = parser.add_subparsers(dest="command", required=True)

    bf = sub.add_parser("backfill", help="IEM archive backfill per station-year")
    bf.add_argument("--start", type=int, default=2015)
    bf.add_argument("--end", type=int, default=datetime.now(timezone.utc).year)
    bf.add_argument("--local", action="store_true", help="write under data/ instead of S3")
    bf.add_argument("--force", action="store_true", help="rewrite existing keys")
    bf.add_argument("--stations", nargs="*", default=None)

    rc = sub.add_parser("recent", help="print latest AWC reports (live-mode smoke test)")
    rc.add_argument("--stations", nargs="*", default=None)
    rc.add_argument("--hours", type=int, default=6)

    args = parser.parse_args()
    if args.command == "backfill":
        run_backfill(args.start, args.end, local=args.local, force=args.force,
                     stations=args.stations)
    elif args.command == "recent":
        stations = args.stations or stations_from_map(load_station_venue_map())
        print(fetch_recent(stations, args.hours).to_string(index=False))


if __name__ == "__main__":
    main()
