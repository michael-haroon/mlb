#!/usr/bin/env python3.11
"""Diff two weather_asof artifacts of the same season, per channel and per dim.

Why this exists. A passing audit proves the artifact is self-consistent; it does not prove
a builder change did what it claimed. The three data-integrity fixes landing in this
season (obs mask honesty, the METAR visibility clamp, the impossible-report drop) are all
of the form "some populated entries should stop existing or change value" -- and the audit
would report PASSED for an artifact where the fix silently no-oped, because a clean tensor
and an unfixed-but-lucky tensor look identical to a structural check.

So this compares old against new directly and answers the three questions a reviewer would
actually ask: did the defect counts go to zero, did the fix reach only the dims it was
supposed to reach, and did coverage survive? The last one is the real risk. Dropping a
whole METAR report to avoid fabricating a reading is the honest choice, but it spends
coverage to buy it, and a fix that quietly halved obs coverage would be a much worse bug
than the one it repaired.

Usage:
  python3.11 data_curation/scripts/diff_weather_asof_artifacts.py --season 2015 \
      --old deep_learning/feature_store/_backups/weather_asof/_prefix_backup/season=2015.parquet

The backups moved out from under weather_asof/ on 2026-08-31. They used to sit at
weather_asof/_prefix_backup/ and weather_asof/_srcmask_backup/, which every current reader
happens to miss (they key on the literal `weather_asof/season=` prefix, and a single-star glob
does not cross `/`) -- but any future reader written as `weather_asof/**/*.parquet` would have
silently triple-counted 2015. Keeping backups outside the read prefix removes the possibility
rather than relying on the next reader being careful.

Exit code is nonzero if coverage regressed beyond --max-coverage-drop, so this is safe to
wire into a gate.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "deep_learning"))

from mlb_dl.weather_asof import (  # noqa: E402
    ASOF_CHANNELS,
    IMPOSSIBLE_ZERO_OBS_DIMS,
    N_DECISIONS,
    N_DIMS,
    N_OBS_DIMS,
    N_TARGET_HOURS,
    OBS_EXTRA_NAMES,
    OFF_FCST,
    OFF_FCST_MASK,
    OFF_LEAD,
    OFF_OBS,
    OFF_OBS_MASK,
)
from mlb_dl.weather_context import WEATHER_TEMPORAL_COLUMNS  # noqa: E402

DIM_NAMES = list(WEATHER_TEMPORAL_COLUMNS) + list(OBS_EXTRA_NAMES)
S3_BUCKET = "mlb-265753586044-us-east-1-an"
CHANNEL_COLS = [f"wx_c{i:02d}" for i in range(ASOF_CHANNELS)]

# The visibility clamp's target: the METAR reporting ceiling in metres. Entries above this
# are what the clamp was added to remove.
VIS_CEILING_M = 10.0 * 1609.34


def load(key: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (tensor[games, d, h, C], game_pk[games]) sorted canonically."""
    s3 = boto3.client("s3", region_name="us-east-1")
    df = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()))
    df = df.sort_values(["game_pk", "decision_hour", "target_hour"])
    pks = df["game_pk"].unique()
    T = df[CHANNEL_COLS].to_numpy(np.float32).reshape(
        len(pks), N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS)
    return T, pks


def masked_in_zeros(T: np.ndarray, v0: int, m0: int, n_dims: int) -> dict[int, int]:
    out = {}
    for d in IMPOSSIBLE_ZERO_OBS_DIMS:
        if d >= n_dims:
            continue
        out[d] = int(((T[..., m0 + d] == 1.0) & (T[..., v0 + d] == 0.0)).sum())
    return out


def coverage(T: np.ndarray, m0: int, n_dims: int) -> np.ndarray:
    """Per-dim share of entries the mask claims. Compared dim-by-dim rather than in
    aggregate, because a fix that wiped one dim's coverage while leaving the other 26
    intact would barely move an overall mean."""
    return (T[..., m0:m0 + n_dims] == 1.0).mean(axis=(0, 1, 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--old", required=True, help="S3 key of the pre-change artifact")
    ap.add_argument("--new", default=None, help="S3 key of the post-change artifact")
    ap.add_argument("--max-coverage-drop", type=float, default=0.02,
                    help="per-dim absolute coverage loss tolerated before failing")
    args = ap.parse_args()
    new_key = args.new or (f"deep_learning/feature_store/weather_asof/"
                           f"season={args.season}.parquet")

    To, pko = load(args.old)
    Tn, pkn = load(new_key)
    print(f"old {To.shape} ({len(pko)} games)   new {Tn.shape} ({len(pkn)} games)")
    if len(pko) != len(pkn) or not np.array_equal(pko, pkn):
        print(f"FAIL game sets differ: {len(set(pko) - set(pkn))} only in old, "
              f"{len(set(pkn) - set(pko))} only in new")
        sys.exit(1)

    fails: list[str] = []

    # --- 1. did the defects go away? -----------------------------------------
    print("\n--- masked-in impossible zeros (target: all -> 0)")
    for label, v0, m0, n in (("fcst", OFF_FCST, OFF_FCST_MASK, N_DIMS),
                             ("obs", OFF_OBS, OFF_OBS_MASK, N_OBS_DIMS)):
        zo = masked_in_zeros(To, v0, m0, n)
        zn = masked_in_zeros(Tn, v0, m0, n)
        for d in sorted(zo):
            if zo[d] or zn[d]:
                verdict = "FIXED" if zn[d] == 0 else "STILL PRESENT"
                print(f"  {label:4s} dim {d:2d} {DIM_NAMES[d]:22s} {zo[d]:6d} -> {zn[d]:6d}  {verdict}")
                if zn[d]:
                    fails.append(f"{label} dim {d} still has {zn[d]} masked-in zeros")

    print("\n--- visibility above the METAR reporting ceiling (target: -> 0)")
    for label, v0, m0 in (("fcst", OFF_FCST, OFF_FCST_MASK), ("obs", OFF_OBS, OFF_OBS_MASK)):
        for tag, T in (("old", To), ("new", Tn)):
            live = T[..., m0 + 11] == 1.0
            n = int((T[..., v0 + 11][live] > VIS_CEILING_M + 1.0).sum())
            mx = float(T[..., v0 + 11][live].max()) if live.any() else float("nan")
            print(f"  {label:4s} {tag}: {n:6d} above ceiling, max {mx:,.0f} m")
            if tag == "new" and n:
                fails.append(f"{label} visibility still has {n} entries above the ceiling")

    # --- 2. coverage must survive --------------------------------------------
    print(f"\n--- per-dim mask coverage (fail if a dim drops > {args.max_coverage_drop:.1%})")
    for label, m0, n in (("fcst", OFF_FCST_MASK, N_DIMS), ("obs", OFF_OBS_MASK, N_OBS_DIMS)):
        co, cn = coverage(To, m0, n), coverage(Tn, m0, n)
        worst = int(np.argmax(co - cn))
        for d in range(n):
            drop = co[d] - cn[d]
            if abs(drop) > 1e-9:
                flag = "  <-- LOSS" if drop > args.max_coverage_drop else ""
                print(f"  {label:4s} dim {d:2d} {DIM_NAMES[d]:22s} "
                      f"{co[d]:7.4%} -> {cn[d]:7.4%} ({-drop:+.4%}){flag}")
                if drop > args.max_coverage_drop:
                    fails.append(f"{label} dim {d} ({DIM_NAMES[d]}) lost {drop:.2%} coverage")
        print(f"  {label:4s} overall {co.mean():.4%} -> {cn.mean():.4%} "
              f"(worst dim {worst} {DIM_NAMES[worst]}: {co[worst] - cn[worst]:+.4%})")

    # --- 3. what else moved? -------------------------------------------------
    # An unexpected dim changing is the signal that a fix reached further than intended.
    print("\n--- value changes among entries both artifacts claim")
    for label, v0, m0, n in (("fcst", OFF_FCST, OFF_FCST_MASK, N_DIMS),
                             ("obs", OFF_OBS, OFF_OBS_MASK, N_OBS_DIMS)):
        both = (To[..., m0:m0 + n] == 1.0) & (Tn[..., m0:m0 + n] == 1.0)
        for d in range(n):
            b = both[..., d]
            if not b.any():
                continue
            a, c = To[..., v0 + d][b], Tn[..., v0 + d][b]
            nd = int((~np.isclose(a, c, rtol=1e-5, atol=1e-6)).sum())
            if nd:
                print(f"  {label:4s} dim {d:2d} {DIM_NAMES[d]:22s} {nd:7d} of {b.sum():8d} "
                      f"changed, max |delta| {np.abs(a - c).max():.4g}")

    lead_delta = np.abs(To[..., OFF_LEAD] - Tn[..., OFF_LEAD]).max()
    print(f"  lead_norm max |delta| {lead_delta:.4g}")

    print(f"\n{'DIFF PASSED' if not fails else 'DIFF FAILED'}")
    for f in fails:
        print(f"  FAIL {f}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
