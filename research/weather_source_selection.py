"""
Phase 2: empirical source-selection study + dissemination-lag validation.

Outputs deep_learning/mlb_dl/weather_source_winners.json:
  - per-dim, per-lead-bucket HRRR error vs the independent ASOS observation
    (bias / MAE / n) for every dim a METAR can verify
  - measured dissemination lags: AWC receiptTime-reportTime percentiles (live
    probe) and HRRR S3 object Last-Modified - issue_time for recent cycles
  - the v1 winner per dim (HRRR is the only leakage-free 2015+ source, so the
    value of this study is the error curves and lag constants, not the vote)

Statistical notes (assumptions stated per repo policy):
  - Station-vs-grid-cell siting error is folded into MAE; it is constant per
    venue and does not vary with lead, so LEAD SLOPES are unbiased even though
    absolute MAE overstates pure forecast error.
  - Errors are averaged over venue-hours pooled across seasons; no independence
    assumption is made beyond reporting n (hours within a game are correlated).

Run on EC2 (reads S3):
  python3.11 research/weather_source_selection.py [--dates 60]
"""

from __future__ import annotations

import argparse
import io
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import requests

S3_BUCKET = "mlb-265753586044-us-east-1-an"
REPO = Path(__file__).resolve().parent.parent
OUT_PATH = REPO / "deep_learning" / "mlb_dl" / "weather_source_winners.json"

LEAD_BUCKETS = [(1, 2), (3, 5), (6, 9), (10, 18)]

# (name, hrrr column -> obs comparable) — dims a METAR can verify.
# Wind speed uses hypot(u,v); cloud/visibility verify category-level skill.
COMPARISONS = {
    "temperature_c": ("t2m_k", "tmpf", lambda h: h - 273.15, lambda o: (o - 32) * 5 / 9),
    "dewpoint_c": ("d2m_k", "dwpf", lambda h: h - 273.15, lambda o: (o - 32) * 5 / 9),
    "wind_speed_ms": (("u10_ms", "v10_ms"), "sknt", None, lambda o: o * 0.514444),
    "gust_ms": ("gust_ms", "gust", lambda h: h, lambda o: o * 0.514444),
    "pressure_hpa": ("sp_pa", None, lambda h: h / 100.0, None),  # vs altimeter+elev, below
    "visibility_km": ("vis_m", "vsby", lambda h: h / 1000.0, lambda o: o * 1.60934),
}

s3 = boto3.client("s3", region_name="us-east-1")


def _read(key):
    return pd.read_parquet(io.BytesIO(s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()))


def _station_pressure_hpa(alti_inhg, elev_m):
    return alti_inhg * 33.8639 * (1.0 - 2.25577e-5 * elev_m) ** 5.25588


def measure_hrrr_errors(n_dates: int) -> dict:
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=S3_BUCKET, Prefix="data/weather/source=hrrr_asissued/"):
        keys += [o["Key"] for o in page.get("Contents", [])]
    random.seed(11)
    sample = random.sample(keys, min(n_dates, len(keys)))
    vmap = json.loads(s3.get_object(Bucket=S3_BUCKET, Key="data/weather/station_venue_map.json")["Body"].read())
    prim = {int(v): (m["primary_station"], float(m.get("primary_elev_m") or 0.0))
            for v, m in vmap.items()}

    errors: dict[str, list[pd.DataFrame]] = {k: [] for k in COMPARISONS}
    obs_cache: dict[str, pd.DataFrame] = {}
    for key in sample:
        h = _read(key)
        h["hour"] = h["valid_time_utc"].dt.floor("h")
        year = int(h["valid_time_utc"].dt.year.mode()[0])
        for vid, grp in h.groupby("venue_id"):
            st, elev = prim.get(int(vid), (None, 0.0))
            if st is None:
                continue
            ck = f"{st}:{year}"
            if ck not in obs_cache:
                try:
                    o = _read(f"data/weather/source=asos_obs/station={st}/year={year}.parquet")
                    o["hour"] = o["valid_utc"].dt.floor("h")
                    obs_cache[ck] = o.groupby("hour", as_index=False).agg(
                        tmpf=("tmpf", "mean"), dwpf=("dwpf", "mean"),
                        sknt=("sknt", "mean"), gust=("gust", "max"),
                        alti=("alti", "mean"), vsby=("vsby", "min"))
                except Exception:
                    obs_cache[ck] = pd.DataFrame()
            o = obs_cache[ck]
            if o.empty:
                continue
            m = grp.merge(o, on="hour", how="inner")
            if m.empty:
                continue
            for name, (hcol, ocol, hconv, oconv) in COMPARISONS.items():
                if name == "wind_speed_ms":
                    fc = np.hypot(m["u10_ms"], m["v10_ms"])
                    ob = oconv(m[ocol])
                elif name == "pressure_hpa":
                    fc = hconv(m[hcol])
                    ob = _station_pressure_hpa(m["alti"], elev)
                else:
                    fc = hconv(m[hcol])
                    ob = oconv(m[ocol])
                err = pd.DataFrame({"lead": m["lead_hours"], "err": fc - ob}).dropna()
                if len(err):
                    errors[name].append(err)

    out = {}
    for name, frames in errors.items():
        if not frames:
            continue
        e = pd.concat(frames, ignore_index=True)
        buckets = {}
        for lo, hi in LEAD_BUCKETS:
            b = e[(e["lead"] >= lo) & (e["lead"] <= hi)]["err"]
            if len(b):
                buckets[f"{lo}-{hi}h"] = {"bias": round(float(b.mean()), 3),
                                          "mae": round(float(b.abs().mean()), 3),
                                          "n": int(len(b))}
        out[name] = {"winner": "hrrr", "lead_buckets": buckets}
    return out


def measure_asos_lag() -> dict:
    """Live AWC probe: actual receipt latency of the reports we consume."""
    r = requests.get("https://aviationweather.gov/api/data/metar",
                     params={"ids": "KBOS,KORD,KLGA,KDEN,KSEA,KMIA,KPHX,KSTL",
                             "format": "json", "hours": 24}, timeout=30)
    lags = []
    for rec in r.json():
        rt, rcpt = rec.get("reportTime"), rec.get("receiptTime")
        if rt and rcpt:
            lag = (pd.Timestamp(rcpt) - pd.Timestamp(rt)).total_seconds() / 60
            if 0 <= lag < 120:
                lags.append(lag)
    lags = np.array(lags)
    return {"n": int(len(lags)), "p50_min": round(float(np.percentile(lags, 50)), 1),
            "p95_min": round(float(np.percentile(lags, 95)), 1),
            "p99_min": round(float(np.percentile(lags, 99)), 1),
            "constant_used_min": 10,
            "verdict": "ok" if np.percentile(lags, 99) <= 10 else "constant too optimistic"}


def measure_hrrr_lag(n_cycles: int = 12) -> dict:
    """HRRR dissemination: S3 Last-Modified minus issue time for RECENT cycles
    (historical objects were bulk-backfilled; only fresh ones carry true lag)."""
    lags = []
    now = datetime.now(timezone.utc)
    for k in range(2, 2 + n_cycles):
        t = pd.Timestamp(now).floor("h") - pd.Timedelta(hours=k)
        key = f"hrrr.{t:%Y%m%d}/conus/hrrr.t{t:%H}z.wrfsfcf02.grib2"
        try:
            # Public NOAA bucket rejects cross-account SIGNED requests (403)
            from botocore import UNSIGNED
            from botocore.config import Config
            anon = boto3.client("s3", region_name="us-east-1",
                                config=Config(signature_version=UNSIGNED))
            head = anon.head_object(Bucket="noaa-hrrr-bdp-pds", Key=key)
            lag = (pd.Timestamp(head["LastModified"]) - t.tz_localize("UTC") if t.tzinfo is None
                   else pd.Timestamp(head["LastModified"]) - t).total_seconds() / 60
            lags.append(lag)
        except Exception:
            continue
    if not lags:
        return {"n": 0, "verdict": "no recent cycles readable"}
    lags = np.array(lags)
    return {"n": int(len(lags)), "min_min": round(float(lags.min()), 1),
            "max_min": round(float(lags.max()), 1),
            "constant_used_min": 75,
            "verdict": "ok" if lags.max() <= 75 else "constant too optimistic"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", type=int, default=60)
    args = ap.parse_args()
    result = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "hrrr_vs_asos": measure_hrrr_errors(args.dates),
        "asos_dissemination_lag": measure_asos_lag(),
        "hrrr_dissemination_lag": measure_hrrr_lag(),
        "notes": "v1 is HRRR-only (Toronto in-grid; AWS GFS starts 2021 = mask-flip era regressor; "
                 "ECMWF as-issued unavailable free pre-2022). Winner fields will become "
                 "meaningful when a second leakage-free source spans 2015+.",
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
