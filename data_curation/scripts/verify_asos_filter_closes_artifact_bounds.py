"""Prove the METAR report filter cannot admit an observation the artifact will reject.

The two layers make DIFFERENT claims and drifted apart, which is how four range failures
reached the built weather_asof artifact:

  _drop_impossible_reports (ingestion) -- "no real weather may be dropped", so its bounds
      are world records widened, one field at a time.
  PHYSICAL_RANGES (artifact audit)      -- "plausible for a played MLB game", which is
      strictly tighter.

A value can therefore pass ingestion and fail the audit, and the audit runs AFTER a
multi-hour season build, so the disagreement is only ever discovered expensively. This
script closes the loop cheaply: it runs the real filter over every station-season and
reports the maximum SURVIVING value per consumed field, converted into the artifact's
units, against the artifact's own bound. Imports both sides rather than restating them so
neither can drift from this check.

Exit 1 if any surviving maximum exceeds its artifact bound. Read-only.

SCOPE MATTERS, and getting it wrong makes this check cry wolf. The artifact bounds are
game-hour bounds ("the coldest MLB game on record is ~18 F"), while the ingestion filter is
an all-hours filter bracketing the world record -- so a station legitimately reporting
-58 F in January violates the artifact's dim-9 floor while being no threat to it at all,
because no population game can ever select that hour. The first version of this script
scanned every report in the archive and duly reported two such phantoms. It therefore
restricts to the hours the as-of window can actually reach: for each venue's mapped
stations, the union over that venue's population games of [anchor-2h, anchor+6h], where
anchor = floor(game_datetime_utc, 1h). select_asof_obs searches [hour_start-1h,
hour_start+1h) for target hours -1..5, so that span is the reachable envelope.

KNOWN RESIDUALS as of 2026-08-30 (this script exits 1 on them by design; they are open
design questions, not regressions):

  dim 12  MCF 2018, 172.21 mm/h surviving. MCF's p01i column is not inches in any usable
      sense (isolated 6.78/11.2/.../19.2, and its own primary SPG covered the same hours at
      0.14 in/h max), so 25.4 inflates it.
  dim  9  NZY 2024, -53.0 F surviving. NZY's temperature sensor was faulted across
      March-April 2024: dozens of reports from -53 to -80 F, all with dwpf and relh NaN
      while alti and vsby stayed normal. METAR_PHYSICAL_LIMITS["tmpf"] floor of -60 drops
      -61..-80 and passes -53..-58, so what survives is simply the warmest garbage.

Both are the same structural gap, and it is worth stating because it bounds what this
script can ever prove: a physical bound sits, by construction, at the edge of the
physically possible, while a broken sensor emits a SPREAD of values, some of which land
inside that edge. Global field bounds therefore cannot catch an episodic single-station
fault -- that needs a screen comparing a station against its own history and against its
paired station, which is a QC subsystem this pipeline does not have. Two independent
instances (MCF precip, NZY temperature) now motivate one.

Usage:  python3.11 data_curation/scripts/verify_asos_filter_closes_artifact_bounds.py
        --all-hours   # the conservative superset, for latent-risk review
"""
import argparse
import io
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, "deep_learning")
from mlb_dl.weather_asof import KT_TO_MPH, metar_to_era5  # noqa: E402

BUCKET = "mlb-265753586044-us-east-1-an"
PREFIX = "data/weather/source=asos_obs/"

# (era5 column produced by metar_to_era5, artifact dim, label). Only the dims a METAR can
# actually observe and that a corrupt report could push past the audit's ceiling.
CHECKS = [
    ("wind_gusts_10m", 5, "wind_gusts mph"),
    ("wind_speed_10m", 4, "wind_speed mph"),
    ("precipitation", 12, "precip mm"),
    ("temperature_2m", 9, "temperature_f"),
]

s3 = boto3.client("s3")
log = logging.getLogger("closes")


def artifact_bounds() -> dict[int, tuple[float, float]]:
    """The audit's own table, imported so the two cannot disagree."""
    sys.path.insert(0, "data_curation/scripts")
    from verify_weather_asof_artifact import PHYSICAL_RANGES
    return {i: r for i, r in enumerate(PHYSICAL_RANGES)}


def reachable_hours() -> dict[str, set]:
    """station -> set of UTC hours some population game at a venue it serves can select.

    Built from game_meta and the station map so it tracks the real population rather than
    a month/hour heuristic. A station serving several venues gets the union.
    """
    sys.path.insert(0, "deep_learning")
    from mlb_dl.build_weather_asof import (FS_PREFIX, POP_GAME_TYPES, POP_MIN_DATE,
                                           STATION_MAP_KEY, _read_json, _read_parquet)
    vmap = _read_json(STATION_MAP_KEY)
    gm = _read_parquet(f"{FS_PREFIX}/game_meta.parquet",
                       columns=["game_pk", "game_date", "game_datetime_utc",
                                "venue_id", "game_type_code"])
    gm["game_date"] = pd.to_datetime(gm["game_date"])
    pop = gm[(gm["game_date"] >= POP_MIN_DATE)
             & gm["game_type_code"].isin(POP_GAME_TYPES)].dropna(subset=["venue_id"])
    anchor = pd.to_datetime(pop["game_datetime_utc"], utc=True).dt.floor("h")
    venue = pop["venue_id"].astype(int).to_numpy()

    per_venue: dict[int, set] = {}
    # -2h..+6h inclusive: obs search opens 1 h before target hour -1 and the last decision
    # sits at +6h, so this is the envelope select_asof_obs can reach.
    offsets = [pd.Timedelta(hours=k) for k in range(-2, 7)]
    for vid, a in zip(venue, anchor.to_numpy()):
        s = per_venue.setdefault(int(vid), set())
        for off in offsets:
            s.add(pd.Timestamp(a) + off)

    out: dict[str, set] = {}
    for vid_str, m in vmap.items():
        hrs = per_venue.get(int(vid_str))
        if not hrs:
            continue
        for st in (m.get("primary_station"), m.get("backup_station")):
            if st:
                out.setdefault(st, set()).update(hrs)
    log.info("reachable envelope: %d stations, %d venue(s) with population games",
             len(out), len(per_venue))
    return out


def list_files() -> list[str]:
    keys, tok = [], None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": PREFIX}
        if tok:
            kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in r.get("Contents", []) if o["Key"].endswith(".parquet")]
        if not r.get("IsTruncated"):
            return keys
        tok = r["NextContinuationToken"]


def scan(key: str, envelope: Optional[dict] = None):
    m = re.search(r"station=([^/]+)/year=(\d+)", key)
    station = m.group(1)
    try:
        raw = pd.read_parquet(io.BytesIO(
            s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()))
    except Exception as exc:
        log.warning("unreadable %s: %s", key, exc)
        return None
    n_in = len(raw)
    # Station elevation only shifts pressure, which is not among CHECKS; 0 is fine here.
    out = metar_to_era5(raw, station_elev_m=0.0)
    if envelope is not None:
        hrs = envelope.get(station)
        if not hrs:
            return None          # station serves no population game; nothing can reach it
        keep = pd.to_datetime(out["valid_utc"], utc=True).dt.floor("h").isin(hrs)
        out = out[keep]
    rec = {"station": station, "year": int(m.group(2)),
           "n_in": n_in, "n_out": len(out)}
    for col, _dim, _label in CHECKS:
        v = pd.to_numeric(out[col], errors="coerce") if col in out else None
        rec[col] = float(v.max()) if v is not None and v.notna().any() else np.nan
        rec[col + "_min"] = float(v.min()) if v is not None and v.notna().any() else np.nan
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-hours", action="store_true",
                    help="scan every report, not just the reachable envelope (superset)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    bounds = artifact_bounds()
    envelope = None if a.all_hours else reachable_hours()
    keys = list_files()
    scope = "ALL HOURS (latent-risk superset)" if a.all_hours else "reachable game hours"
    log.info("running the real filter over %d station-season files, scope: %s",
             len(keys), scope)
    rows = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        for i, r in enumerate(pool.map(lambda k: scan(k, envelope), keys), 1):
            if r:
                rows.append(r)
            if i % 200 == 0:
                log.info("  %d/%d", i, len(keys))
    df = pd.DataFrame(rows)
    kept, total = int(df["n_out"].sum()), int(df["n_in"].sum())
    print(f"\n=== scope: {scope} ===")
    print(f"=== {len(df)} station-seasons contribute; {kept:,} reports in scope "
          f"(of {total:,} read) ===\n")

    failures = 0
    for col, dim, label in CHECKS:
        lo, hi = bounds[dim]
        mx = df[col].max()
        mn = df[col + "_min"].min()
        worst = df.loc[df[col].idxmax()] if df[col].notna().any() else None
        ok = (mx <= hi) and (mn >= lo)
        tag = "ok  " if ok else "FAIL"
        print(f"{tag} dim {dim:2d} {label:18s} surviving range "
              f"[{mn:.2f}, {mx:.2f}]  artifact bound [{lo}, {hi}]"
              + (f"   worst: {worst['station']} {int(worst['year'])}" if worst is not None
                 else ""))
        if not ok:
            failures += 1
            # Which station-seasons still exceed it, so the next rule has a target. Both
            # sides: a floor violation is as real as a ceiling one and the first version of
            # this report only printed ceilings, which hid the dim-9 cold outlier entirely.
            if mx > hi:
                bad = df[df[col] > hi][["station", "year", col]].sort_values(
                    col, ascending=False)
                print("  above the ceiling:")
                print(bad.head(10).to_string(index=False))
            if mn < lo:
                bad = df[df[col + "_min"] < lo][["station", "year", col + "_min"]
                                                ].sort_values(col + "_min")
                print("  below the floor:")
                print(bad.head(10).to_string(index=False))

    if failures:
        print(f"\n{failures} FAILURE(S): the filter still admits reports the artifact "
              f"audit will reject. Fix ingestion, not the audit bound.")
        sys.exit(1)
    print("\nALL SURVIVING OBSERVATIONS FALL INSIDE THE ARTIFACT BOUNDS")


if __name__ == "__main__":
    main()
