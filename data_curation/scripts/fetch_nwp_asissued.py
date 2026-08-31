"""
HRRR forecasts AS ISSUED — the FORECAST channel of the as-of weather tensor.

Replaces Open-Meteo Historical Forecast for the DL live models. That product is a
stitched 0-2h-lead composite (near-analysis), so training on it leaks skill the
model can never have live; here every row is one (issue_time, valid_time) pair
pulled from the operational HRRR archive (s3://noaa-hrrr-bdp-pds via Herbie), so
training sees forecasts at the same lead times inference does.

v1 is HRRR-only (verified 2026-08-30):
  - Rogers Centre is INSIDE the HRRR CONUS grid (nearest cell 0.83 km), removing
    GFS's primary purpose (Toronto).
  - AWS has GFS only from 2021; a channel whose mask flips on mid-era is a free
    era regressor (same argument that excluded ECMWF), and NCAR's 2015+ GFS
    archive needs registration. International-venue games (~27 of 26,751; mostly
    domes) get the zero-mask "unknown" path instead.
  - A missing HRRR issue falls back to the previous issue at higher lead — the
    same thing the live path would do.

Values are stored in RAW GRIB SI units (K, m/s, Pa, kg/m^2); physics happens in
mlb_dl.weather_asof, shared between the training builder and the live path.

S3 layout (date-partitioned — the extraction loop is date-major and one date's
file is the resumability unit):
  data/weather/source=hrrr_asissued/date={YYYY-MM-DD}.parquet

Usage:
  conda run -n pred python data_curation/scripts/fetch_nwp_asissued.py backfill \
      --start 2015-01-15 --end 2025-11-01 [--workers 6]
  conda run -n pred python data_curation/scripts/fetch_nwp_asissued.py latest \
      --venue-id 3313 --game-hour-utc 2026-08-30T23:00
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

# ── Storage (same bucket/prefix as fetch_weather.py / fetch_asos_obs.py) ─────
S3_BUCKET = "mlb-265753586044-us-east-1-an"
S3_PREFIX = "data"
S3_REGION = "us-east-1"

DATA_DIR = "data"
LOG_DIR = os.path.join(DATA_DIR, "logs")

REPO = Path(__file__).resolve().parent.parent.parent
FEATURE_STORE = REPO / "deep_learning" / "feature_store"
STATION_VENUE_MAP = REPO / "data_curation" / "station_venue_map.json"

# Herbie downloads GRIB byte-range subsets here; each is deleted after point
# extraction (260k subsets x ~2MB would otherwise fill the disk).
# PID-scoped: the between-dates purge below is only safe against this process's
# own workers. When three shards ran on one box against a shared /tmp/herbie_nwp
# (2026-08-30), one process's purge deleted another's in-flight subsets — 1,319
# ENOENT failures silently dropped 41% of planned tasks.
HERBIE_SAVE_DIR = Path(f"/tmp/herbie_nwp_{os.getpid()}")

# Transient S3/decode failures are retried; a task that still fails after this
# many attempts must surface, because a date written short of its planned tasks
# is indistinguishable from a complete one to the existence-keyed resume logic.
FETCH_ATTEMPTS = 3
FETCH_RETRY_SLEEP_S = 2.0

# Retries alone do not close the hazard named in the comment above. The shared-/tmp
# purge race described at HERBIE_SAVE_DIR exhausted all 3 attempts on many tasks and
# wrote 25 dates at fill 0.17-0.81 between 07:43-08:23Z on 2026-08-30; the existence-
# keyed resume then skipped them permanently, and they were only recovered by a
# separate verifier sweep plus a --force rerun. PID-scoping fixed that particular
# cause, but ANY sustained transient-failure source reproduces the same permanent
# damage, so the write itself has to refuse. A date whose shortfall is due to
# TransientFetchError is not written
# at all, leaving the key absent for the next run to retry. Set to the verifier's
# DATE_FILL_REPORT_FLOOR so the writer refuses exactly what the completeness gate
# would reject — any gap between the two floors is a date that gets persisted and then
# flagged forever. Leaving a date absent is only recoverable because
# verify_weather_archives.py check_coverage() fails on a population date with no object.
MIN_WRITE_FILL = 0.98


class TransientFetchError(Exception):
    """Download or GRIB decode failed — retryable, unlike an archive gap."""

# Population definition — must match the training population.
POP_MIN_DATE = "2015-01-15"  # HRRR on AWS is complete from here; also GFS-era floor
POP_GAME_TYPES = ["R", "F", "D", "L", "W"]

# Dissemination lag — VALIDATED 2026-08-30 (research/weather_source_selection.py):
# S3 Last-Modified minus issue time over 12 recent cycles measured 54.8-65.6 min,
# so 75 min never admits an issue before it was actually retrievable while
# sacrificing only ~10-20 min of freshness.
HRRR_AVAILABILITY_LAG_MIN = 75

# As-of window geometry (mirrors mlb_dl.weather_asof): target hours -1..5
# relative to floor(game_datetime_utc), decision hours 0..6.
TARGET_HOURS = range(-1, 6)
DECISION_HOURS = range(0, 7)

# HRRR grid is 3 km; a nearest-cell distance beyond this means the venue is
# outside the CONUS domain (international games) — no forecast rows emitted.
MAX_GRID_DISTANCE_KM = 5.0

# Outer bounds of the HRRR CONUS domain, used ONLY as a definitely-outside test.
# A venue beyond these bounds is thousands of km from any grid cell, so no rerun can
# ever produce rows for it; a venue inside them is still subject to the exact
# MAX_GRID_DISTANCE_KM nearest-cell test above. The asymmetry is deliberate and is the
# safe direction: the verifier may only use this to excuse an absence it can PROVE is
# structural, never to excuse a real extraction failure as a domain gap.
HRRR_CONUS_BOUNDS = (21.0, 53.0, -135.0, -60.0)  # lat_min, lat_max, lon_min, lon_max


def venue_in_hrrr_domain(lat: float, lon: float) -> bool:
    """Could HRRR ever supply a forecast for this venue?

    Six real population dates have no in-domain venue: the Tokyo Dome openers
    (2019-03-20/21, 2025-03-18/19) and the Seoul series (2024-03-20/21). Those were the
    only games played on those days, so no CONUS game created an archive object and the
    date is absent from S3 permanently and correctly. Coverage must be able to tell that
    apart from a dropped extraction, because the upstream GRIBs for those hours plainly
    do exist.
    """
    lat_min, lat_max, lon_min, lon_max = HRRR_CONUS_BOUNDS
    return lat_min <= float(lat) <= lat_max and lon_min <= float(lon) <= lon_max


def dates_without_in_domain_venue(games: pd.DataFrame) -> set[str]:
    """Population dates on which no venue lies inside the HRRR domain.

    A date with even one in-domain venue yields rows and an object, so only dates where
    EVERY venue is out of domain are structurally unfillable.
    """
    pts = load_venue_points().set_index("venue_id")
    in_domain = {
        int(vid): venue_in_hrrr_domain(r["latitude"], r["longitude"])
        for vid, r in pts.iterrows()
    }
    out: set[str] = set()
    for d, grp in games.groupby(games["game_date"].dt.normalize()):
        vids = [int(x) for x in grp["venue_id"].unique()]
        # An unknown venue is treated as IN domain: assuming otherwise would let a
        # missing venue_points row silently excuse a genuine gap.
        if vids and not any(in_domain.get(v, True) for v in vids):
            out.add(f"{d:%Y-%m-%d}")
    return out

# One byte-range subset per (issue, fxx): every raw field the 22-dim layout
# needs, verified present in the cheap `sfc` product back to 2015-04.
# APCP is appended per-fxx because its bucket string embeds the lead hour.
# RH:2m is deliberately absent — 2015 files don't carry it (added later), and a
# column whose presence flips mid-era is a free era regressor; weather_asof
# derives relative humidity from t2m/d2m identically for every era.
SEARCH_BASE = (
    ":(TMP|DPT):2 m above ground:"
    "|:(UGRD|VGRD):10 m above ground:"
    "|:GUST:surface:|:PRES:surface:|:TCDC:entire atmosphere:"
    "|:VIS:surface:|:HPBL:surface:|:DSWRF:surface:"
    "|:(TMP|HGT):(850|1000) mb:|:(UGRD|VGRD):850 mb:"
)

# cfgrib variable name -> output column (surface/near-surface hypercubes).
SURFACE_RENAMES = {
    "t2m": "t2m_k", "d2m": "d2m_k",
    "u10": "u10_ms", "v10": "v10_ms", "gust": "gust_ms",
    "sp": "sp_pa", "tcc": "tcc_pct", "vis": "vis_m",
    "tp": "apcp_mm", "blh": "hpbl_m",
    "sdswrf": "dswrf_wm2", "dswrf": "dswrf_wm2",
}
# Isobaric hypercubes: (cfgrib name, level hPa) -> output column.
LEVEL_RENAMES = {
    ("t", 850): "t850_k", ("t", 1000): "t1000_k",
    ("gh", 850): "z850_m", ("gh", 1000): "z1000_m",
    ("u", 850): "u850_ms", ("v", 850): "v850_ms",
}

logger = logging.getLogger("NWP_ASISSUED")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    os.makedirs(LOG_DIR, exist_ok=True)
    fh = logging.FileHandler(os.path.join(LOG_DIR, "nwp_asissued.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[NWP] %(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)


# ── Pure planning functions (unit-tested; the leakage guarantee lives here) ──
def apcp_search(fxx: int) -> str:
    """Full search regex for one lead: base fields + the 1-hour APCP bucket.

    The sfc file carries two APCP records (0-fxx total and (fxx-1)-fxx bucket);
    only the 1-hour bucket maps onto the layout's hourly `precipitation`.
    """
    return SEARCH_BASE + f"|:APCP:surface:{fxx - 1}-{fxx} hour acc"


def issue_available_time(issue_time: pd.Timestamp) -> pd.Timestamp:
    return issue_time + pd.Timedelta(minutes=HRRR_AVAILABILITY_LAG_MIN)


def freshest_issue(decision_time: pd.Timestamp) -> pd.Timestamp:
    """Latest hourly HRRR issue whose availability time is <= decision_time.

    With a 75-min lag this is decision_hour - 2 when the decision falls on the
    hour: issue h-1 lands at h-1+01:15 > h, so it is NOT available at h.
    """
    candidate = decision_time.floor("h")
    while issue_available_time(candidate) > decision_time:
        candidate -= pd.Timedelta(hours=1)
    return candidate


def plan_game_tasks(game_hour_utc: pd.Timestamp) -> set[tuple[pd.Timestamp, int]]:
    """All (issue_time, fxx) pairs one game's as-of tensor can reference.

    For each decision hour d, the tensor wants the freshest available issue for
    every target hour h in -1..5 with h >= d-1 (hours before that are covered by
    earlier decisions' extractions — D1 keeps the LAST pre-observation forecast
    at observed hours). One extra preceding issue per decision is planned as the
    missing-file fallback, mirroring what the live path would do.
    """
    tasks: set[tuple[pd.Timestamp, int]] = set()
    for d in DECISION_HOURS:
        decision_time = game_hour_utc + pd.Timedelta(hours=d)
        primary = freshest_issue(decision_time)
        for issue in (primary, primary - pd.Timedelta(hours=1)):
            for h in TARGET_HOURS:
                if h < d - 1:
                    continue
                valid = game_hour_utc + pd.Timedelta(hours=h)
                fxx = int((valid - issue) / pd.Timedelta(hours=1))
                # fxx 0 is an analysis with no 1-h APCP bucket; 18 is the
                # hourly-cycle horizon. Both bounds hold for every reachable
                # (d, h) here, but guard against future window changes.
                if 1 <= fxx <= 18:
                    tasks.add((issue, fxx))
    return tasks


# ── Venue coordinates (single source: the validated station-venue map) ───────
def load_venue_points() -> pd.DataFrame:
    with open(STATION_VENUE_MAP) as f:
        vmap = json.load(f)
    return pd.DataFrame([
        {"venue_id": int(vid), "latitude": m["venue_lat"], "longitude": m["venue_lon"]}
        for vid, m in vmap.items()
    ])


def load_population_games() -> pd.DataFrame:
    gm = pd.read_parquet(FEATURE_STORE / "game_meta.parquet", columns=[
        "game_pk", "game_date", "game_type_code", "venue_id", "game_datetime_utc"])
    gm["game_date"] = pd.to_datetime(gm["game_date"])
    pop = gm[(gm["game_date"] >= POP_MIN_DATE) & gm["game_type_code"].isin(POP_GAME_TYPES)].copy()
    dt = pd.to_datetime(pop["game_datetime_utc"], utc=True)
    pop["game_hour_utc"] = dt.dt.floor("h")
    return pop.dropna(subset=["game_hour_utc", "venue_id"])


# ── GRIB extraction ───────────────────────────────────────────────────────────
# The venue set is static, so each venue's nearest grid cell is computed ONCE
# per grid shape and reused. Herbie's per-call pick_points loads/builds a
# 1.9M-point BallTree per hypercube per file — measured 7.3 GB RSS with 4
# workers, OOM-killing the t3.large backfill box. Direct flat indexing is both
# the memory and the throughput fix.
_grid_cache: dict[tuple, dict] = {}
_grid_lock = threading.Lock()


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def _grid_index(ds, all_points: pd.DataFrame) -> dict:
    """venue_id -> (flat grid index, distance km) for one grid geometry."""
    key = tuple(int(ds.sizes[d]) for d in ("y", "x") if d in ds.sizes)
    with _grid_lock:
        if key not in _grid_cache:
            glat = np.asarray(ds["latitude"].values, dtype=np.float64).ravel()
            # HRRR GRIB longitudes are 0..360; venue lons are -180..180
            glon = (np.asarray(ds["longitude"].values, dtype=np.float64).ravel()
                    + 180.0) % 360.0 - 180.0
            out = {}
            for _, p in all_points.iterrows():
                d = _haversine_km(p["latitude"], p["longitude"], glat, glon)
                j = int(np.argmin(d))
                out[int(p["venue_id"])] = (j, float(d[j]))
            _grid_cache[key] = out
            logger.info(f"grid index built for shape {key} ({len(out)} venues)")
    return _grid_cache[key]


def _extract_columns(ds, flat_idx: np.ndarray) -> dict[str, np.ndarray]:
    """Map one cfgrib hypercube's values at the venue grid cells onto columns."""
    out: dict[str, np.ndarray] = {}
    levels = ds.coords.get("isobaricInhPa")
    for var in ds.data_vars:
        if var in ("gribfile_projection",):
            continue
        vals = np.asarray(ds[var].values)
        if levels is not None and levels.size > 1:
            flat = vals.reshape(levels.size, -1)
            for i, lvl in enumerate(np.atleast_1d(levels.values)):
                col = LEVEL_RENAMES.get((var, int(lvl)))
                if col:
                    out[col] = flat[i, flat_idx]
        elif levels is not None:
            col = LEVEL_RENAMES.get((var, int(np.atleast_1d(levels.values)[0])))
            if col:
                out[col] = vals.reshape(-1)[flat_idx]
        else:
            col = SURFACE_RENAMES.get(var)
            if col:
                out[col] = vals.reshape(-1)[flat_idx]
    return out


def fetch_issue_points(issue_time: pd.Timestamp, fxx: int,
                       points: pd.DataFrame,
                       all_points: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """One (issue, fxx) subset download -> one row per in-grid venue.

    `points` is the day's venues; `all_points` (default: the full map) seeds
    the one-time grid index so the cache is shared across all dates.
    """
    from herbie import Herbie

    if all_points is None:
        all_points = load_venue_points()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        H = Herbie(issue_time.strftime("%Y-%m-%d %H:%M"), model="hrrr", product="sfc",
                   fxx=fxx, save_dir=str(HERBIE_SAVE_DIR), verbose=False)
        if H.grib is None:
            logger.debug(f"issue {issue_time} f{fxx:02d}: not in archive")
            return None
        try:
            dss = H.xarray(apcp_search(fxx), remove_grib=True)
        except Exception as exc:
            raise TransientFetchError(
                f"issue {issue_time} f{fxx:02d}: download/decode failed: {exc}") from exc
        if not isinstance(dss, list):
            dss = [dss]
        cols: dict[str, np.ndarray] = {}
        gi = _grid_index(dss[0], all_points)
        vids = [int(v) for v in points["venue_id"].values]
        flat_idx = np.array([gi[v][0] for v in vids], dtype=np.int64)
        dist = np.array([gi[v][1] for v in vids])
        for ds in dss:
            cols.update(_extract_columns(ds, flat_idx))
            ds.close()

    df = pd.DataFrame(cols)
    df.insert(0, "venue_id", vids)
    df = df[dist <= MAX_GRID_DISTANCE_KM]
    df["model"] = "hrrr"
    df["issue_time_utc"] = issue_time
    df["available_time_utc"] = issue_available_time(issue_time)
    df["valid_time_utc"] = issue_time + pd.Timedelta(hours=fxx)
    df["lead_hours"] = fxx
    return df


def fetch_issue_points_retrying(issue_time: pd.Timestamp, fxx: int,
                                points: pd.DataFrame,
                                all_points: Optional[pd.DataFrame] = None,
                                attempts: int = FETCH_ATTEMPTS) -> Optional[pd.DataFrame]:
    """`fetch_issue_points` with retries on transient failures only.

    An archive gap (issue never published) returns None on the first call and is
    NOT retried — the 2015 era has thousands of real holes and retrying them
    would multiply wall-clock for nothing.
    """
    last: Optional[TransientFetchError] = None
    for i in range(attempts):
        try:
            return fetch_issue_points(issue_time, fxx, points, all_points)
        except TransientFetchError as exc:
            last = exc
            logger.warning(f"attempt {i + 1}/{attempts}: {exc}")
            if i + 1 < attempts:
                time.sleep(FETCH_RETRY_SLEEP_S * (i + 1))
    raise last  # type: ignore[misc]


# ── S3 helpers ────────────────────────────────────────────────────────────────
_s3_client = None


def _get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=S3_REGION)
    return _s3_client


def _date_key(date: pd.Timestamp) -> str:
    return f"{S3_PREFIX}/weather/source=hrrr_asissued/date={date:%Y-%m-%d}.parquet"


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


# ── Backfill driver ───────────────────────────────────────────────────────────
def run_backfill(start: str, end: str, workers: int = 6, force: bool = False,
                 local: bool = False, allow_empty: bool = False) -> None:
    HERBIE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    games = load_population_games()
    venue_points = load_venue_points()
    pop_max = pd.to_datetime(games["game_date"]).max()
    games = games[(games["game_date"] >= start) & (games["game_date"] <= end)]
    dates = sorted(games["game_date"].dt.normalize().unique())

    # An empty window must not report success. load_population_games() reads a LOCAL
    # game_meta.parquet, so a box holding a pre-refresh snapshot sees no games in a window
    # that is actually full of them: on 2026-08-30 this logged "0 games over 0 dates" and
    # exited 0 for a 68-date gap, leaving the archive short. A downstream build would then
    # emit rows whose forecast channel is entirely mask=0 — correct row count, no signal.
    # Genuinely gameless windows (offseason) are legitimate, hence --allow-empty.
    if len(games) == 0 and not allow_empty:
        raise RuntimeError(
            f"No population games in {start}..{end}, so there is nothing to fetch and the "
            f"archive would silently stay short. The local game_meta only reaches "
            f"{pop_max:%Y-%m-%d}. If that is behind the feature store, refresh it:\n"
            f"  aws s3 cp s3://{S3_BUCKET}/deep_learning/feature_store/game_meta.parquet "
            f"{FEATURE_STORE}/game_meta.parquet\n"
            f"If the window is genuinely gameless (offseason), pass --allow-empty.")

    logger.info(f"Backfill {start}..{end}: {len(games)} games over {len(dates)} dates, "
                f"{workers} workers (population reaches {pop_max:%Y-%m-%d})")

    n_done = n_skip = n_empty = n_incomplete = 0
    for date in dates:
        key = _date_key(pd.Timestamp(date))
        if not force and not local and _s3_key_exists(key):
            n_skip += 1
            continue
        day_games = games[games["game_date"].dt.normalize() == date]
        day_venues = venue_points[venue_points["venue_id"].isin(day_games["venue_id"].unique())]
        tasks: set[tuple[pd.Timestamp, int]] = set()
        for gh in day_games["game_hour_utc"].unique():
            tasks |= plan_game_tasks(pd.Timestamp(gh))

        frames = []
        n_gap = n_fail = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch_issue_points_retrying, issue, fxx, day_venues,
                                   venue_points): (issue, fxx)
                       for issue, fxx in tasks}
            for fut in as_completed(futures):
                issue, fxx = futures[fut]
                try:
                    df = fut.result()
                except Exception as exc:
                    n_fail += 1
                    logger.error(f"{date:%Y-%m-%d} issue {issue} f{fxx:02d}: {exc}")
                    continue
                if df is None or df.empty:
                    n_gap += 1
                else:
                    frames.append(df)

        if not frames:
            logger.warning(f"{pd.Timestamp(date):%Y-%m-%d}: no data for any of {len(tasks)} tasks")
            n_empty += 1
            continue

        # Refuse to persist a shortfall we caused. n_fail counts tasks whose data
        # EXISTS upstream but which we failed to retrieve after every retry; n_gap
        # counts tasks the archive never published. Only the former is recoverable, so
        # only the former justifies withholding the write — withholding an unfillable
        # date would leave check_coverage alarming with no way to clear it.
        fill = len(frames) / max(len(tasks), 1)
        if n_fail and fill < MIN_WRITE_FILL:
            logger.error(
                f"{pd.Timestamp(date):%Y-%m-%d}: NOT WRITING — fill {fill:.2f} "
                f"({len(frames)}/{len(tasks)} tasks) with {n_fail} transient failures "
                f"and {n_gap} archive gaps. Leaving the key absent so a rerun retries "
                f"it; persisting it would make the loss permanent."
            )
            n_incomplete += 1
            continue
        out = pd.concat(frames, ignore_index=True).sort_values(
            ["venue_id", "issue_time_utc", "valid_time_utc"])
        if local:
            p = Path(DATA_DIR) / "weather" / "source=hrrr_asissued" / f"date={pd.Timestamp(date):%Y-%m-%d}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(p, index=False)
        else:
            _write_parquet_s3(out, key)
        n_done += 1
        # gap vs fail matters: gaps are real archive holes the fallback-issue
        # planning already covers, failures are lost data behind a written file.
        logger.info(f"{pd.Timestamp(date):%Y-%m-%d}: {len(out)} rows "
                    f"({len(frames)}/{len(tasks)} tasks, {n_gap} gap, {n_fail} FAIL, "
                    f"{day_venues.shape[0]} venues) [{n_done} done, {n_skip} skipped]")
        if n_fail:
            logger.error(f"{pd.Timestamp(date):%Y-%m-%d}: WRITTEN SHORT — {n_fail} tasks "
                         f"failed after {FETCH_ATTEMPTS} attempts; rerun this date with --force")
        # Herbie's remove_grib leaves subset files behind (measured: ~110 MB after
        # 3 dates); the full backfill would fill the disk. Between dates no worker
        # is live, so purging the save dir is safe.
        shutil.rmtree(HERBIE_SAVE_DIR, ignore_errors=True)
        HERBIE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Backfill complete: {n_done} written, {n_skip} skipped (exist), "
                f"{n_empty} empty, {n_incomplete} withheld (transient loss)")
    if n_incomplete:
        # Loud, because these dates look identical to "never attempted" and are
        # only recoverable by rerunning this command.
        logger.error(f"{n_incomplete} date(s) were NOT written due to transient "
                     f"fetch loss — rerun this range to fill them")


# ── Live path ─────────────────────────────────────────────────────────────────
def fetch_latest_issue(venue_id: int, game_hour_utc: pd.Timestamp,
                       now: Optional[pd.Timestamp] = None,
                       max_fallback: int = 3) -> Optional[pd.DataFrame]:
    """Freshest available issue covering the game window, for live inference.

    Walks back through issues (same fallback the training extraction plans for)
    until one exists; same schema as the backfill rows.
    """
    now = now or pd.Timestamp.now(tz="UTC")
    points = load_venue_points()
    points = points[points["venue_id"] == venue_id]
    if points.empty:
        logger.warning(f"venue {venue_id} not in station_venue_map — no forecast")
        return None
    issue = freshest_issue(now)
    for _ in range(max_fallback):
        frames = []
        for h in TARGET_HOURS:
            valid = game_hour_utc + pd.Timedelta(hours=h)
            fxx = int((valid - issue) / pd.Timedelta(hours=1))
            if not 1 <= fxx <= 18:
                continue
            df = fetch_issue_points(issue, fxx, points)
            if df is not None and not df.empty:
                frames.append(df)
        if frames:
            return pd.concat(frames, ignore_index=True)
        issue -= pd.Timedelta(hours=1)
        logger.warning(f"live: issue missing, falling back to {issue}")
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="HRRR as-issued point extraction")
    sub = parser.add_subparsers(dest="command", required=True)

    bf = sub.add_parser("backfill", help="date-partitioned archive backfill")
    bf.add_argument("--start", default=POP_MIN_DATE)
    bf.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    bf.add_argument("--workers", type=int, default=6)
    bf.add_argument("--force", action="store_true", help="rewrite existing keys")
    bf.add_argument("--local", action="store_true", help="write under data/ instead of S3")
    bf.add_argument("--allow-empty", action="store_true",
                    help="permit a window with no population games (offseason); without this "
                         "an empty window is an error, since the usual cause is a stale "
                         "local game_meta.parquet rather than a genuinely gameless range")

    lt = sub.add_parser("latest", help="freshest issue for one venue (live smoke test)")
    lt.add_argument("--venue-id", type=int, required=True)
    lt.add_argument("--game-hour-utc", required=True)

    args = parser.parse_args()
    if args.command == "backfill":
        run_backfill(args.start, args.end, workers=args.workers, force=args.force,
                     local=args.local, allow_empty=args.allow_empty)
    elif args.command == "latest":
        df = fetch_latest_issue(args.venue_id, pd.Timestamp(args.game_hour_utc, tz="UTC"))
        print(df.to_string(index=False) if df is not None else "no forecast available")


if __name__ == "__main__":
    main()
