"""
Integrity + realism gates for the as-of weather archives. Run after (or during)
any backfill and before every downstream transformation — catching a defect
here is hours; catching it after a retrain is days.

  asos  — schema/types/UTC, completeness vs the station map, MADIS
          contamination (the report_type bug: >30% NaN tmpf), value ranges,
          'M' sentinel leaks, availability lag invariant.
  hrrr  — schema, leakage invariants AS DATA (available=issue+lag,
          valid=issue+lead, lead in [1,18]), duplicates, per-venue physical
          plausibility (surface pressure vs venue elevation), per-date task
          coverage, d=0..6 assemblability for every game.
  completeness — per-date (issue, fxx) task fill vs `plan_game_tasks`. Every
          per-row check passes on a file that is missing half its tasks, so
          completeness must be checked against the PLAN, not the file.
  cross — HRRR 1-2h-lead temperature vs the INDEPENDENT ASOS observation at
          the same venue-hour: |bias| < 1.5°C, MAE < 3.0°C (hourly-cycle NWP
          2m-temp skill; Benjamin et al. 2016 report ~1.5-2K RMSE at short
          leads). Catches unit errors, wrong grid cells, timestamp shifts that
          per-source checks cannot.

Exits non-zero on any failure so backfill chains can gate on it.

Usage:
  python3.11 verify_weather_archives.py asos [--sample 30]
  python3.11 verify_weather_archives.py hrrr [--sample 30]
  python3.11 verify_weather_archives.py cross [--dates 8]
  python3.11 verify_weather_archives.py all
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import pandas as pd

S3_BUCKET = "mlb-265753586044-us-east-1-an"
REPO = Path(__file__).resolve().parent.parent.parent
STATION_VENUE_MAP = REPO / "data_curation" / "station_venue_map.json"

HRRR_COLS = {
    "venue_id", "tcc_pct", "u10_ms", "v10_ms", "t2m_k", "d2m_k", "t1000_k",
    "t850_k", "z1000_m", "z850_m", "u850_ms", "v850_ms", "sp_pa", "hpbl_m",
    "vis_m", "apcp_mm", "gust_ms", "dswrf_wm2", "model", "issue_time_utc",
    "available_time_utc", "valid_time_utc", "lead_hours",
}

_s3 = boto3.client("s3", region_name="us-east-1")
_fails: list[str] = []


def fail(msg: str) -> None:
    _fails.append(msg)
    print(f"FAIL  {msg}")


def ok(msg: str) -> None:
    print(f"ok    {msg}")


def _list(prefix: str) -> list[tuple[str, int]]:
    out = []
    for page in _s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=prefix):
        out += [(o["Key"], o["Size"]) for o in page.get("Contents", [])]
    return out


def _read(key: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(_s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()))


def _venue_map() -> dict:
    with open(STATION_VENUE_MAP) as f:
        return json.load(f)


# ── ASOS ──────────────────────────────────────────────────────────────────────
def check_asos(sample_n: int = 30) -> None:
    keys = _list("data/weather/source=asos_obs/")
    vm = _venue_map()
    stations = sorted({m["primary_station"] for m in vm.values()}
                      | {m["backup_station"] for m in vm.values()})
    have = {tuple(k.split("station=")[1].replace(".parquet", "").split("/year="))
            for k, _ in keys}
    missing = [(s, y) for s in stations for y in map(str, range(2015, 2027))
               if (s, y) not in have]
    # DMH 2021 is a known real outage (station dark), not a fetch failure.
    missing = [m for m in missing if m != ("DMH", "2021")]
    if missing:
        fail(f"asos completeness: missing {missing[:10]}{'...' if len(missing) > 10 else ''}")
    else:
        ok(f"asos completeness: {len(keys)} files cover all station-years")

    random.seed(1)
    for key, _ in random.sample(keys, min(sample_n, len(keys))):
        df = _read(key)
        st, yr = key.split("station=")[1].replace(".parquet", "").split("/year=")
        tag = f"{st} {yr}"
        for col in ("valid_utc", "available_time_utc", "tmpf", "wxcodes",
                    "peak_wind_gust", "ice_accretion_1hr", "skyc4"):
            if col not in df.columns:
                fail(f"asos {tag}: missing column {col}")
        if str(df["valid_utc"].dt.tz) != "UTC":
            fail(f"asos {tag}: valid_utc not UTC")
        nan_share = df["tmpf"].isna().mean()
        if nan_share > 0.30:
            fail(f"asos {tag}: tmpf {nan_share:.0%} NaN — MADIS contamination")
        lag = (df["available_time_utc"] - df["valid_utc"]).dt.total_seconds() / 60
        if not (lag == 10).all():
            fail(f"asos {tag}: availability lag != 10min")
        for col in ("tmpf", "dwpf", "sknt", "alti", "p01i", "peak_wind_gust"):
            if df[col].dtype == object:
                fail(f"asos {tag}: {col} object dtype ('M' leaked)")
        t = df["tmpf"].dropna()
        if len(t) and (t.min() < -60 or t.max() > 130):
            fail(f"asos {tag}: tmpf out of range [{t.min()},{t.max()}]")
        if (df["skyc1"].astype(str) == "M").any():
            fail(f"asos {tag}: literal 'M' in skyc1")
        if int(yr) < 2026 and df["valid_utc"].dt.floor("h").nunique() < 4000:
            fail(f"asos {tag}: only {df['valid_utc'].dt.floor('h').nunique()} hours")
    ok(f"asos deep checks on {min(sample_n, len(keys))} sampled files done")


# ── HRRR ──────────────────────────────────────────────────────────────────────
def check_hrrr(sample_n: int = 30) -> None:
    keys = _list("data/weather/source=hrrr_asissued/")
    if not keys:
        fail("hrrr: no files at all")
        return
    ok(f"hrrr: {len(keys)} date files, {sum(s for _, s in keys) / 1e6:.0f} MB")
    small = [k for k, s in keys if s < 3000]
    if small:
        fail(f"hrrr: suspiciously small files {small[:5]}")

    vm = _venue_map()
    elev_by_vid: dict[int, float] = {}
    for vid, m in vm.items():
        if m.get("primary_elev_m") is not None:
            elev_by_vid[int(vid)] = float(m["primary_elev_m"])

    random.seed(2)
    for key, _ in random.sample(keys, min(sample_n, len(keys))):
        df = _read(key)
        tag = key.split("date=")[1].replace(".parquet", "")
        if set(df.columns) != HRRR_COLS:
            fail(f"hrrr {tag}: column mismatch {set(df.columns) ^ HRRR_COLS}")
            continue
        if df.isna().any().any():
            fail(f"hrrr {tag}: NaNs in {df.columns[df.isna().any()].tolist()}")
        # Leakage invariants as data properties
        lag = (df["available_time_utc"] - df["issue_time_utc"]).dt.total_seconds() / 60
        if not (lag == 75).all():
            fail(f"hrrr {tag}: available != issue+75min")
        span = (df["valid_time_utc"] - df["issue_time_utc"]).dt.total_seconds() / 3600
        if not (span == df["lead_hours"]).all():
            fail(f"hrrr {tag}: valid != issue+lead")
        if df["lead_hours"].min() < 1 or df["lead_hours"].max() > 18:
            fail(f"hrrr {tag}: lead out of [1,18]")
        if df.duplicated(["venue_id", "issue_time_utc", "valid_time_utc"]).any():
            fail(f"hrrr {tag}: duplicate (venue, issue, valid) rows")
        # Physical plausibility
        t = df["t2m_k"]
        if t.min() < 230 or t.max() > 330:
            fail(f"hrrr {tag}: t2m_k out of range [{t.min():.0f},{t.max():.0f}]")
        if (df["apcp_mm"] < 0).any() or (df["tcc_pct"].max() > 100.5):
            fail(f"hrrr {tag}: negative precip or cloud > 100%")
        # HRRR emits slight supersaturation in saturated conditions (measured
        # max +0.77 K on 2015-05-03, ~0.8% of rows) — a model artifact, and
        # weather_asof clips RH at 100%. Beyond ~1.5 K means misaligned fields.
        if (df["d2m_k"] > df["t2m_k"] + 1.5).any():
            fail(f"hrrr {tag}: dewpoint above temperature beyond supersaturation artifact")
        # Surface pressure must track venue elevation (barometric formula
        # within generous weather bounds) — catches wrong-grid-cell errors.
        for vid, grp in df.groupby("venue_id"):
            e = elev_by_vid.get(int(vid))
            if e is None:
                continue
            expected = 1013.25 * (1 - 2.25577e-5 * e) ** 5.25588
            p = grp["sp_pa"].mean() / 100.0
            if abs(p - expected) > 40:
                fail(f"hrrr {tag}: venue {vid} sp {p:.0f} hPa vs elevation-expected {expected:.0f}")
    ok(f"hrrr deep checks on {min(sample_n, len(keys))} sampled files done")


# ── HRRR completeness (per-date task fill) ────────────────────────────────────
# The gate that was missing on 2026-08-30: three shards sharing one Herbie save
# dir dropped 41% of planned (issue, fxx) tasks, and every per-row check above
# passed — the rows that survived were perfectly valid. Completeness has to be
# checked against the PLAN, not against the file.
#
# Thresholds alone cannot do this job. Measured 2026-08-30: a season median of
# 1.000 coexisted with individual dates at 0.26-0.39 fill, because only the
# dates written during the 3-process window were damaged and the median washed
# them out. And a hard per-date floor mis-flags the 2015 era, which has genuine
# multi-hour HRRR outages (2015-09-11 legitimately sits at 0.60).
#
# So low fill is classified, not thresholded: for each suspect date, HEAD the
# missing (issue, fxx) objects in the upstream NOAA bucket. Object absent = real
# archive gap, which the fallback-issue planning already covers. Object present
# = OUR extraction lost it, which is recoverable and must fail the gate.
# Set just under 1.0 rather than at a "suspicious" level: because the upstream
# probe classifies rather than guesses, a false positive costs 10 HEAD requests
# and prints "genuine archive gap". Sensitivity is nearly free here.
DATE_FILL_REPORT_FLOOR = 0.98
UPSTREAM_PROBES_PER_DATE = 10
NOAA_HRRR_BUCKET = "noaa-hrrr-bdp-pds"


def _noaa_s3():
    global _noaa
    try:
        return _noaa
    except NameError:
        pass
    from botocore import UNSIGNED
    from botocore.client import Config
    # Public bucket 403s on SIGNED cross-account requests.
    _noaa = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
    return _noaa


def _upstream_exists(issue: pd.Timestamp, fxx: int) -> bool:
    key = f"hrrr.{issue:%Y%m%d}/conus/hrrr.t{issue.hour:02d}z.wrfsfcf{fxx:02d}.grib2"
    try:
        _noaa_s3().head_object(Bucket=NOAA_HRRR_BUCKET, Key=key)
        return True
    except Exception:
        return False


def check_coverage(year: int | None = None, workers: int = 16,
                   repair_out: str | None = None) -> None:
    """Every population date must HAVE an archive object, or provably not need one.

    check_completeness only inspects dates that exist, so a date the extraction never
    wrote is invisible to it — including a whole season lost to a dead shard. This is
    also what makes it safe for the fetcher to refuse to persist a partial date
    (fetch_nwp_asissued.MIN_WRITE_FILL): withholding a write is only recoverable if
    something notices the absence.

    A missing date is NOT automatically a failure. run_backfill writes nothing when
    every planned task is an upstream archive gap (`n_empty`), which genuinely happens
    in the 2015-2016 era. So each missing date is classified against the upstream
    bucket exactly as check_completeness classifies an under-filled one: recoverable
    (tasks exist upstream) fails, unobtainable passes. Without this, a hard coverage
    gate would block those seasons' builds forever.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fetch_nwp_asissued import (dates_without_in_domain_venue,
                                    load_population_games, plan_game_tasks)

    games = load_population_games()
    # HRRR is CONUS-only. A date whose every venue is out of domain yields no rows and
    # therefore no object, no matter how many times it is refetched, even though its
    # upstream GRIBs exist. Without this, the six international-opener dates would fail
    # coverage forever and block the 2019, 2024 and 2025 builds.
    out_of_domain = dates_without_in_domain_venue(games)
    games["d"] = games["game_date"].dt.normalize()
    hours_by_date = {d: g["game_hour_utc"].unique() for d, g in games.groupby("d")}
    expected = {f"{d:%Y-%m-%d}" for d in games["d"].unique()}
    have = {k.split("date=")[1].replace(".parquet", "")
            for k, _ in _list("data/weather/source=hrrr_asissued/")}
    if year is not None:
        expected = {t for t in expected if t.startswith(f"{year}-")}
        have = {t for t in have if t.startswith(f"{year}-")}

    missing = sorted(expected - have)
    extra = sorted(have - expected)
    scope = f"{year}" if year is not None else "all seasons"

    if not missing:
        ok(f"coverage {scope}: all {len(expected)} population dates present")
    else:
        def classify(tag: str):
            planned: set = set()
            for gh in hours_by_date.get(pd.Timestamp(tag), []):
                planned |= plan_game_tasks(pd.Timestamp(gh))
            probe = sorted(planned)[:UPSTREAM_PROBES_PER_DATE]
            present = [(i, x) for i, x in probe
                       if _upstream_exists(pd.Timestamp(i), int(x))]
            return tag, len(probe), present

        recoverable: list[str] = []
        era_gap: list[str] = []
        no_domain = sorted(d for d in missing if d in out_of_domain)
        # Settled before probing: an out-of-domain date's GRIBs exist, so the upstream
        # probe would call it recoverable and demand a rerun that can never succeed.
        probe_set = [d for d in missing if d not in out_of_domain]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for tag, n_probe, present in pool.map(classify, probe_set):
                (recoverable if present else era_gap).append(tag)
        # Two distinct reasons an absence is permanent, kept apart because they are
        # reported separately: conflating them would claim the international dates have
        # no upstream GRIB, which is false and would send the next reader hunting for a
        # nonexistent archive gap.
        unobtainable = sorted(era_gap + no_domain)

        if no_domain:
            ok(f"coverage {scope}: {len(no_domain)} population date(s) absent because no "
               f"venue is inside the HRRR CONUS domain (international games — HRRR "
               f"cannot cover them at any lead), e.g. {no_domain[:5]}")

        if recoverable:
            by_season: dict[str, int] = {}
            for t in recoverable:
                by_season[t[:4]] = by_season.get(t[:4], 0) + 1
            fail(f"coverage {scope}: {len(recoverable)}/{len(expected)} population dates "
                 f"have NO archive object though their tasks EXIST upstream — per season "
                 f"{by_season}; first {recoverable[:5]}; rerun those ranges")
        if era_gap:
            ok(f"coverage {scope}: {len(era_gap)} population date(s) absent because "
               f"no planned task exists upstream (genuine era gap), e.g. "
               f"{era_gap[:5]}")
        if not recoverable:
            ok(f"coverage {scope}: {len(have)} present + {len(unobtainable)} provably "
               f"unobtainable = all {len(expected)} population dates accounted for")
        if repair_out:
            Path(repair_out).write_text(
                "\n".join(recoverable) + ("\n" if recoverable else ""))
            ok(f"coverage {scope}: wrote {len(recoverable)} missing date(s) to {repair_out}")

    if extra:
        # Not fatal: a date can be archived and later drop out of the population if
        # game_meta is rebuilt. Worth surfacing because it also means the population
        # shrank, which would silently shrink training data.
        ok(f"coverage {scope}: {len(extra)} archived dates not in the population "
           f"(stale or population shrank), e.g. {extra[:5]}")


def check_completeness(sample_n: int = 60, year: int | None = None,
                       workers: int = 16, repair_out: str | None = None) -> None:
    """Compare each archived date's task set against the planned task set.

    `sample_n <= 0` means EVERY date, which is the only setting that can serve as a
    release gate: a sample can estimate the loss rate but cannot enumerate the dates
    to repair, and a lost date is invisible to a normal rerun (run_backfill skips any
    date whose S3 key exists, and a partial object looks identical to a complete one
    through head_object). Reads are I/O-bound S3 GETs, so they are threaded --
    serially, a full 12-season sweep takes over an hour.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fetch_nwp_asissued import load_population_games, plan_game_tasks

    keys = _list("data/weather/source=hrrr_asissued/")
    if not keys:
        fail("completeness: no hrrr files")
        return
    games = load_population_games()
    games["d"] = games["game_date"].dt.normalize()
    hours_by_date = {d: g["game_hour_utc"].unique() for d, g in games.groupby("d")}

    by_date = {k.split("date=")[1].replace(".parquet", ""): k for k, _ in keys}
    if year is not None:
        by_date = {t: k for t, k in by_date.items() if t.startswith(f"{year}-")}
        if not by_date:
            fail(f"completeness: no hrrr date files for {year}")
            return
    if sample_n <= 0:
        sample = sorted(by_date)
        ok(f"completeness: FULL sweep over all {len(sample)} archived dates")
    else:
        random.seed(3)
        sample = random.sample(sorted(by_date), min(sample_n, len(by_date)))

    def one(tag: str):
        """Returns (year, fill, missing_tasks) or None if the date has no planned tasks."""
        d = pd.Timestamp(tag)
        hrs = hours_by_date.get(d)
        if hrs is None or len(hrs) == 0:
            return None
        planned = set()
        for gh in hrs:
            planned |= plan_game_tasks(pd.Timestamp(gh))
        if not planned:
            return None
        try:
            df = _read(by_date[tag])
        except Exception as exc:            # unreadable object is itself a failure
            return (d.year, -1.0, set(), f"unreadable: {exc}")
        got = set(zip(pd.to_datetime(df["issue_time_utc"], utc=True),
                      df["lead_hours"].astype(int)))
        return (d.year, len(got & planned) / len(planned), planned - got, None)

    fills: dict[int, list[float]] = {}
    low: list[tuple[str, float, set]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for tag, res in zip(sample, pool.map(one, sample)):
            if res is None:
                continue
            yr, f, missing, err = res
            if err is not None:
                fail(f"completeness {tag}: {err}")
                continue
            fills.setdefault(yr, []).append(f)
            if f < DATE_FILL_REPORT_FLOOR:
                low.append((tag, f, missing))

    if not fills:
        fail("completeness: no sampled date had planned tasks — population mismatch")
        return
    for yr in sorted(fills):
        v = np.asarray(fills[yr])
        ok(f"completeness {yr}: task fill median {np.median(v):.3f} min {v.min():.3f} "
           f"over {len(v)} dates")

    # Classify every suspect date against the upstream bucket. Threaded for the same
    # reason as the reads: UPSTREAM_PROBES_PER_DATE head_objects per suspect date.
    def classify(item):
        tag, f, missing = item
        probe = sorted(missing)[:UPSTREAM_PROBES_PER_DATE]
        present = [(i, x) for i, x in probe if _upstream_exists(pd.Timestamp(i), int(x))]
        return tag, f, probe, present

    repairable: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for tag, f, probe, present in pool.map(classify, sorted(low)):
            if present:
                repairable.append(tag)
                fail(f"completeness {tag}: fill {f:.2f} and {len(present)}/{len(probe)} probed "
                     f"missing tasks EXIST upstream (e.g. {present[0][0]} f{present[0][1]:02d}) "
                     f"— our extraction lost them; rerun this date with --force")
            else:
                ok(f"completeness {tag}: fill {f:.2f} but 0/{len(probe)} probed missing tasks "
                   f"exist upstream — genuine archive gap, fallback planning covers it")

    # Machine-readable handoff to repair_hrrr_dates.sh. Written even when empty so a
    # consumer can tell "gate ran, nothing to repair" from "gate never ran".
    if repair_out:
        Path(repair_out).write_text("\n".join(sorted(repairable)) + ("\n" if repairable else ""))
        ok(f"completeness: wrote {len(repairable)} repairable date(s) to {repair_out}")


# ── Cross-source realism ──────────────────────────────────────────────────────
def check_cross(n_dates: int = 8) -> None:
    hrrr_keys = [k for k, _ in _list("data/weather/source=hrrr_asissued/")]
    if not hrrr_keys:
        fail("cross: no hrrr files")
        return
    vm = _venue_map()
    prim = {int(vid): m["primary_station"] for vid, m in vm.items()}
    random.seed(3)
    pairs = []
    for key in random.sample(hrrr_keys, min(n_dates, len(hrrr_keys))):
        h = _read(key)
        h = h[h["lead_hours"] <= 2]
        h["hour"] = h["valid_time_utc"].dt.floor("h")
        year = h["valid_time_utc"].dt.year.mode()[0]
        for vid, grp in h.groupby("venue_id"):
            st = prim.get(int(vid))
            try:
                obs = _read(f"data/weather/source=asos_obs/station={st}/year={year}.parquet")
            except Exception:
                continue
            obs = obs.dropna(subset=["tmpf"]).copy()
            obs["hour"] = obs["valid_utc"].dt.floor("h")
            o = obs.groupby("hour", as_index=False)["tmpf"].mean()
            m = grp.merge(o, on="hour", how="inner")
            if len(m):
                fc_c = m["t2m_k"] - 273.15
                ob_c = (m["tmpf"] - 32) * 5 / 9
                pairs.append(pd.DataFrame({"err": fc_c - ob_c}))
    if not pairs:
        fail("cross: no overlapping venue-hours between HRRR and ASOS")
        return
    err = pd.concat(pairs)["err"]
    bias, mae = err.mean(), err.abs().mean()
    msg = f"cross: n={len(err)} venue-hours, bias {bias:+.2f}°C, MAE {mae:.2f}°C"
    # Station is km from the venue grid cell, so tolerance is siting + NWP error.
    if abs(bias) > 1.5 or mae > 3.0:
        fail(msg + " — exceeds short-lead NWP skill bounds; suspect units/grid/timestamps")
    else:
        ok(msg)


# ── Persistence sources (T1.3): soil at -7d, AQI at -24h ─────────────────────
SOIL_LAG = pd.Timedelta(days=7)
AQI_LAG = pd.Timedelta(hours=24)


def check_persistence() -> None:
    """Every population game must have era5 soil at -7d and CAMS AQI at -24h
    for all target hours -1..5 — the lag rule training will share with live."""
    gm = pd.read_parquet(REPO / "deep_learning" / "feature_store" / "game_meta.parquet",
                         columns=["game_pk", "game_date", "game_type_code",
                                  "venue_id", "game_datetime_utc"])
    gm["game_date"] = pd.to_datetime(gm["game_date"])
    pop = gm[(gm["game_date"] >= "2015-01-15")
             & gm["game_type_code"].isin(["R", "F", "D", "L", "W"])].dropna(subset=["venue_id"])
    pop["gh"] = pd.to_datetime(pop["game_datetime_utc"], utc=True).dt.floor("h")

    specs = [("era5", SOIL_LAG, "soil_moisture_0_to_7cm"),
             ("air_quality", AQI_LAG, ["us_aqi", "pm2_5", "ozone"])]
    for source, lag, cols in specs:
        cols = [cols] if isinstance(cols, str) else cols
        n_missing_hours = n_null = n_total = 0
        missing_venues = []
        for vid, grp in pop.groupby("venue_id"):
            vid = int(vid)
            years = set()
            for gh in grp["gh"]:
                years.add((gh - lag - pd.Timedelta(hours=1)).year)
                years.add((gh - lag + pd.Timedelta(hours=5)).year)
            frames = []
            for y in sorted(years):
                try:
                    frames.append(_read(f"data/weather/source={source}/venue_id={vid}/year={y}.parquet"))
                except Exception:
                    pass
            if not frames:
                missing_venues.append(vid)
                continue
            arch = pd.concat(frames, ignore_index=True)
            tcol = "timestamp" if "timestamp" in arch.columns else "time"
            ts = pd.to_datetime(arch[tcol], utc=True)
            by_hour = arch.assign(_h=ts.dt.floor("h")).drop_duplicates("_h").set_index("_h")
            for gh in grp["gh"].unique():
                for h in range(-1, 6):
                    want = gh - lag + pd.Timedelta(hours=h)
                    n_total += 1
                    if want not in by_hour.index:
                        n_missing_hours += 1
                    else:
                        row = by_hour.loc[want]
                        if any(pd.isna(row.get(c)) for c in cols if c in by_hour.columns):
                            n_null += 1
        if missing_venues:
            fail(f"persistence {source}: no archive at all for venues {missing_venues}")
        share_bad = (n_missing_hours + n_null) / max(n_total, 1)
        msg = (f"persistence {source} (lag {lag}): {n_total} game-hours, "
               f"{n_missing_hours} missing, {n_null} null {cols}")
        if share_bad > 0.02:
            fail(msg + f" — {share_bad:.1%} unusable exceeds 2%")
        else:
            ok(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["asos", "hrrr", "coverage", "completeness", "cross",
                                     "persistence", "all"])
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--dates", type=int, default=8)
    ap.add_argument("--year", type=int, default=None,
                    help="restrict the completeness gate to one season")
    ap.add_argument("--workers", type=int, default=16,
                    help="thread count for the completeness sweep's S3 reads/probes")
    ap.add_argument("--repair-out", default=None,
                    help="write the under-filled dates needing a --force refetch, one "
                         "per line, for repair_hrrr_dates.sh to consume")
    ap.add_argument("--coverage-repair-out", default=None,
                    help="write the entirely-absent population dates, one per line "
                         "(these need a plain rerun, not --force — there is no key yet)")
    args = ap.parse_args()
    if args.what in ("asos", "all"):
        check_asos(args.sample)
    if args.what in ("hrrr", "all"):
        check_hrrr(args.sample)
    if args.what in ("coverage", "completeness", "all"):
        check_coverage(year=args.year, workers=args.workers,
                       repair_out=args.coverage_repair_out)
    if args.what in ("completeness", "all"):
        # `--sample 0` must survive as 0 (= full sweep); only a positive sample gets
        # floored to 60, which is the smallest sample the per-year medians are worth
        # reading. A max() over 0 here would silently downgrade the release gate.
        n = args.sample if args.sample <= 0 else max(args.sample, 60)
        check_completeness(n, year=args.year, workers=args.workers,
                           repair_out=args.repair_out)
    if args.what in ("cross", "all"):
        check_cross(args.dates)
    if args.what in ("persistence", "all"):
        check_persistence()
    print(f"\n{'ALL CHECKS PASSED' if not _fails else f'{len(_fails)} FAILURES'}")
    sys.exit(1 if _fails else 0)


if __name__ == "__main__":
    main()
