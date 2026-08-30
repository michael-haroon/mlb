"""Size two cross-field METAR defects across the whole raw asos_obs archive.

Both defects reached the built weather_asof artifact and failed its plausibility gate,
and neither is catchable by the field-at-a-time bounds in METAR_PHYSICAL_LIMITS, because
each offending value is legal in isolation:

  gust:  GPM 2024-04-28 20:50Z reported sknt=11 with gust=202 kt. 202 kt is BELOW the
         220 kt Barrow Island world record, so the (0,250) bound passes it. What is
         impossible is the pairing -- a gust factor of 18.4 against neighbours at 16-26 kt.
  p01i:  MCF (MacDill AFB) reported 0.8 for eight consecutive hours with wxcodes absent
         and vsby 10 SM, plus isolated 6.78/11.2/11.6/12.8/16.4/19.2. As inches, 19.2 in/h
         is 6x the world hourly record; the (0,12) bound drops the loudest and passes the
         rest. Meanwhile the primary station SPG covered the same hours at a realistic
         0.14 in/h max with -TSRA coded. So MCF's p01i is not inches in any usable sense,
         and the 25.4 in->mm conversion inflated 6.78 into 172 mm/h in the 2018 artifact.

This script measures, it does not decide: it reports how many reports each candidate rule
would remove and which stations carry the mass, so a rule can be chosen on scope rather
than on the single game that surfaced it. Read-only.

Usage:  python3.11 data_curation/scripts/sweep_asos_crossfield_qc.py [--workers 16]
"""
import argparse
import io
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import pandas as pd

BUCKET = "mlb-265753586044-us-east-1-an"
PREFIX = "data/weather/source=asos_obs/"

# Candidate rule constants, stated as hypotheses to be scoped -- not yet adopted.
# GUST_FACTOR_MAX: the 3-second peak to mean-wind ratio over open airport terrain. Durst
# (1960) and ASCE 7-22 put the 3-s gust at ~1.53x the 1-hour mean; convective downbursts
# reach ~2.5-3x. 4.0 sits above every documented value, so it cannot remove real weather.
# GUST_RATIO_FLOOR_KT: below this, the ratio is meaningless -- a METAR only encodes a gust
# at >=10 kt above the mean, so sknt=1/gust=11 is the SMALLEST reportable gust and carries
# a factor of 11. Judging light-and-variable wind by ratio would delete valid reports.
GUST_FACTOR_MAX = 4.0
GUST_RATIO_FLOOR_KT = 60.0

s3 = boto3.client("s3")
log = logging.getLogger("sweep")


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


def scan(key: str) -> dict | None:
    m = re.search(r"station=([^/]+)/year=(\d+)", key)
    if not m:
        return None
    station, year = m.group(1), int(m.group(2))
    try:
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        df = pd.read_parquet(io.BytesIO(body))
    except Exception as exc:
        log.warning("unreadable %s: %s", key, exc)
        return None

    out = {"station": station, "year": year, "n": len(df)}
    sk = pd.to_numeric(df.get("sknt"), errors="coerce")
    gu = pd.to_numeric(df.get("gust"), errors="coerce")
    pi = pd.to_numeric(df.get("p01i"), errors="coerce")
    wx = df["wxcodes"] if "wxcodes" in df else pd.Series(np.nan, index=df.index)

    # ── gust: how many reports the candidate ratio rule would remove ──────────
    if sk is not None and gu is not None:
        ok = sk.notna() & gu.notna()
        ratio = (gu / sk.replace(0, np.nan))
        hit = ok & (gu > GUST_RATIO_FLOOR_KT) & (ratio > GUST_FACTOR_MAX)
        out["gust_rule_hits"] = int(hit.sum())
        out["gust_max"] = float(gu.max()) if gu.notna().any() else np.nan
        # Worst ratio seen ABOVE the floor, to check the floor is not hiding cases.
        r_hi = ratio[ok & (gu > GUST_RATIO_FLOOR_KT)]
        out["gust_ratio_max_above_floor"] = float(r_hi.max()) if len(r_hi) else np.nan
        # And the worst ratio BELOW the floor, to confirm the floor's exemption is small.
        r_lo = ratio[ok & (gu <= GUST_RATIO_FLOOR_KT)]
        out["gust_ratio_max_below_floor"] = float(r_lo.max()) if len(r_lo) else np.nan

    # ── p01i: is this station's accumulation consistent with inches? ──────────
    if pi is not None and pi.notna().any():
        p = pi.dropna()
        nz = p[p > 0]
        out["p01i_max"] = float(p.max())
        out["p01i_p999"] = float(np.percentile(p, 99.9))
        out["p01i_n_nonzero"] = int(len(nz))
        # A stuck/miscalibrated gauge parks on one nonzero value. Mass at the single most
        # common nonzero reading separates that from real rainfall, which is continuous.
        if len(nz):
            vc = nz.round(3).value_counts()
            out["p01i_mode"] = float(vc.index[0])
            out["p01i_mode_frac_of_nonzero"] = float(vc.iloc[0] / len(nz))
        # Accumulation with no precipitation code and unrestricted visibility is
        # self-contradictory; count it as the cross-field signal.
        has_pcode = wx.astype(str).str.contains(
            "RA|DZ|SN|SG|IC|PL|GR|GS|UP|TS", regex=True, na=False)
        vs = pd.to_numeric(df.get("vsby"), errors="coerce")
        contradictory = (pi > 0.05) & ~has_pcode & (vs >= 9.0)
        out["p01i_dry_accum"] = int(contradictory.sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    keys = list_files()
    log.info("scanning %d station-season files", len(keys))
    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, r in enumerate(pool.map(scan, keys), 1):
            if r:
                rows.append(r)
            if i % 200 == 0:
                log.info("  %d/%d", i, len(keys))
    df = pd.DataFrame(rows)
    total = int(df["n"].sum())
    print(f"\n=== archive: {len(df)} station-seasons, {total:,} reports ===")

    # ── gust rule scope ───────────────────────────────────────────────────────
    g = int(df["gust_rule_hits"].fillna(0).sum())
    print(f"\n--- candidate gust rule: drop when gust > {GUST_RATIO_FLOOR_KT} kt AND "
          f"gust/sknt > {GUST_FACTOR_MAX} ---")
    print(f"reports removed: {g} of {total:,} ({100.0*g/max(total,1):.5f}%)")
    print(f"worst ratio above the {GUST_RATIO_FLOOR_KT} kt floor: "
          f"{df['gust_ratio_max_above_floor'].max():.1f}")
    print(f"worst ratio below the floor (exempted by design): "
          f"{df['gust_ratio_max_below_floor'].max():.1f}")
    by_st = df.groupby("station")["gust_rule_hits"].sum().sort_values(ascending=False)
    print("stations carrying the mass:")
    print(by_st[by_st > 0].head(15).to_string())

    # ── p01i station screen ───────────────────────────────────────────────────
    print("\n--- p01i: stations whose accumulation is inconsistent with inches ---")
    agg = df.groupby("station").agg(
        n=("n", "sum"), p01i_max=("p01i_max", "max"),
        p01i_p999=("p01i_p999", "max"), dry_accum=("p01i_dry_accum", "sum"),
        mode=("p01i_mode", "median"),
        mode_frac=("p01i_mode_frac_of_nonzero", "median")).reset_index()
    # 12 in/h is the world hourly rainfall record; a station whose 99.9th percentile
    # approaches it is not reporting inches.
    susp = agg[(agg["p01i_p999"] > 1.0) | (agg["dry_accum"] > 50)].sort_values(
        "dry_accum", ascending=False)
    print(f"{len(susp)} suspect station(s) of {len(agg)}:")
    with pd.option_context("display.width", 200):
        print(susp.to_string(index=False))
    print("\nfor reference, the same columns for 8 healthy stations:")
    ok = agg[~agg["station"].isin(susp["station"])].nlargest(8, "n")
    with pd.option_context("display.width", 200):
        print(ok.to_string(index=False))


if __name__ == "__main__":
    main()
