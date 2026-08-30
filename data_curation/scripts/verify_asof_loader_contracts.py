#!/usr/bin/env python3.11
"""Verify the POSITIONAL contracts the training loaders rely on, against real artifacts.

`train_unified._load_asof_frames` reads both as-of weather artifacts positionally, with
no validation:

    # weather_asof: fixed 49-row blocks, keyed off whatever game_pk starts the block
    for start in range(0, len(arr), per_game):
        asof[int(pks[start])] = arr[start:start + per_game].reshape(7, 7, C)

    # wx_hour_offset: sorted, then straight to_numpy -- position IS sequence_index
    df = pd.read_parquet(f).sort_values(["game_pk", "sequence_index"])
    offsets[int(gpk)] = grp["wx_hour_offset"].to_numpy(np.int8)

Both are correct only if the artifacts satisfy assumptions nothing checks. If a game ever
has 48 or 50 rows, or two games interleave, the block loop assigns one game's weather to
another game's key and every subsequent game shifts -- silently, with correct shapes and
plausible values. If sequence_index has a gap or duplicate, every later pitch in that
game reads the wrong decision hour.

These are exactly the failure modes that produce a model that trains cleanly and prices
wrong, so they are verified here rather than trusted. The check functions take frames so
they can be unit-tested against deliberately corrupted input; see
tests/test_verify_asof_loader_contracts.py.

Usage:
  python3.11 data_curation/scripts/verify_asof_loader_contracts.py --seasons 2015 2016
  python3.11 data_curation/scripts/verify_asof_loader_contracts.py --all
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

from mlb_dl.weather_asof import DECISION_HOURS, TARGET_HOURS  # noqa: E402

S3_BUCKET = "mlb-265753586044-us-east-1-an"
FS_PREFIX = "deep_learning/feature_store"
ROWS_PER_GAME = len(DECISION_HOURS) * len(TARGET_HOURS)

# The loader's reshape(7, 7, C) means row order must be decision-major, target-minor.
EXPECTED_ORDER = [(d, h) for d in DECISION_HOURS for h in TARGET_HOURS]

# Per-game share of backwards steps that separates upstream jitter from misalignment.
# A reversed or interleaved sequence drives ~50% of steps negative; the worst real game
# measured is 3.76% (game 429523, 16 of 426). 20% sits >5x above the observed jitter
# ceiling and well under half, so neither direction is a near miss.
MAX_GAME_DECREASING_FRACTION = 0.20

# Season-wide backwards-step rate. Measured floor is ~5.1e-05 (525 steps / 10.3M
# pitches); 1e-03 leaves ~20x headroom, so a systemic regression in the timestamp join
# or the sort trips this long before it could reach the per-game test.
MAX_SEASON_DECREASING_RATE = 1e-3

# A rate over a handful of pitches carries no information -- one dip in a 20-pitch test
# frame is 5e-02 -- so the season test only applies once there is enough volume for the
# floor to mean something. Real season artifacts run 440k (2020) to 1M pitches, so this
# never suppresses the check on production data; the per-game fraction test still applies
# at any size.
MIN_PITCHES_FOR_RATE = 10_000

_s3 = boto3.client("s3", region_name="us-east-1")


def _read(key: str, columns=None) -> pd.DataFrame:
    body = _s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    return pd.read_parquet(io.BytesIO(body), columns=columns)


def check_asof_blocks(df: pd.DataFrame) -> list[str]:
    """The 49-row-block contract behind the loader's `range(0, len(arr), per_game)`."""
    fails: list[str] = []
    if df.empty:
        return ["weather_asof frame is empty"]

    if len(df) % ROWS_PER_GAME != 0:
        fails.append(f"row count {len(df)} is not a multiple of {ROWS_PER_GAME}; the "
                     f"final block would be short and reshape would raise")

    pks = df["game_pk"].to_numpy()
    # Contiguity: a game must occupy one unbroken run. groupby-based counts cannot see
    # interleaving, and interleaving is the failure the block loop cannot survive.
    starts = np.r_[0, np.flatnonzero(pks[1:] != pks[:-1]) + 1]
    runs = np.diff(np.r_[starts, len(pks)])
    if len(np.unique(pks[starts])) != len(starts):
        dupes = pd.Series(pks[starts]).duplicated()
        fails.append(f"{int(dupes.sum())} game(s) appear in more than one run "
                     f"(rows interleaved); block loader would overwrite keys, e.g. "
                     f"{pks[starts][dupes.to_numpy()][:3].tolist()}")
    bad = runs != ROWS_PER_GAME
    if bad.any():
        ex = [(int(pks[s]), int(n)) for s, n in zip(starts[bad], runs[bad])][:5]
        fails.append(f"{int(bad.sum())} game(s) do not have exactly {ROWS_PER_GAME} "
                     f"contiguous rows (game_pk, n_rows): {ex}")

    # Row order inside each block, since reshape assigns meaning by position alone.
    if {"decision_hour", "target_hour"} <= set(df.columns) and not bad.any():
        order = list(zip(df["decision_hour"].to_numpy()[:ROWS_PER_GAME],
                         df["target_hour"].to_numpy()[:ROWS_PER_GAME]))
        if [(int(d), int(h)) for d, h in order] != EXPECTED_ORDER:
            fails.append(f"first block is not decision-major/target-minor; reshape "
                         f"would transpose the tensor. got {order[:4]} "
                         f"expected {EXPECTED_ORDER[:4]}")
    return fails


def check_offsets(df: pd.DataFrame) -> list[str]:
    """The positional contract behind `grp["wx_hour_offset"].to_numpy(np.int8)`."""
    fails: list[str] = []
    if df.empty:
        return ["wx_hour_offset frame is empty"]

    lo, hi = DECISION_HOURS[0], DECISION_HOURS[-1]
    off = df["wx_hour_offset"]
    if off.isna().any():
        fails.append(f"{int(off.isna().sum())} NaN offsets; int8 cast would wrap them")
    oob = off[(off < lo) | (off > hi)]
    if len(oob):
        fails.append(f"{len(oob)} offset(s) outside [{lo},{hi}], e.g. "
                     f"{oob.unique()[:5].tolist()}; would index past the decision axis")

    g = df.sort_values(["game_pk", "sequence_index"]).groupby("game_pk", sort=False)
    # sequence_index must be a dense 0..N-1 range: position in the array IS the pitch
    # index, so one gap shifts every later pitch onto the wrong decision hour.
    n = g["sequence_index"].agg(["min", "max", "count", "nunique"])
    bad_start = n[n["min"] != 0]
    if len(bad_start):
        fails.append(f"{len(bad_start)} game(s) do not start at sequence_index 0, e.g. "
                     f"{bad_start.index[:3].tolist()}")
    gaps = n[n["max"] != n["count"] - 1]
    if len(gaps):
        fails.append(f"{len(gaps)} game(s) have gaps/duplicates in sequence_index "
                     f"(max != count-1), e.g. {gaps.index[:3].tolist()}")
    dupe = n[n["nunique"] != n["count"]]
    if len(dupe):
        fails.append(f"{len(dupe)} game(s) have duplicate sequence_index values, e.g. "
                     f"{dupe.index[:3].tolist()}")

    # Offsets are elapsed clock hours, so in principle they cannot run backwards within
    # a game. In practice they do, and measuring it (2026-08-30) showed why: raw
    # pitch_start_time is itself non-monotonic. Game 413650 reports 23:02:07 then
    # 22:55:46 for two pitches of the SAME play, all 260 pitches timed. Across
    # 10,328,602 audited pitches, 205 of 32,193 games (0.64%) contain a backwards step,
    # 525 steps in total; classifying each against a centered 21-pitch rolling median
    # puts 252 pitches (2.4e-05) in the leak direction and 584 in the harmless stale
    # direction. The builder's sort keys match the dataset's exactly, so this is upstream
    # noise, not misalignment.
    #
    # So a bare "any decrease" test cannot separate that floor of upstream jitter from
    # the failure this check exists to catch -- a sort or key mismatch, which reverses or
    # interleaves the sequence and drives roughly half of a game's steps negative. The
    # two thresholds below split them with deliberate headroom in both directions.
    frac = g["wx_hour_offset"].apply(
        lambda s: float((s.diff().dropna() < 0).mean()) if len(s) > 1 else 0.0)
    bad = frac[frac > MAX_GAME_DECREASING_FRACTION]
    if len(bad):
        fails.append(f"{len(bad)} game(s) have more than "
                     f"{MAX_GAME_DECREASING_FRACTION:.0%} of steps decreasing — that is "
                     f"reversal/interleaving, not timestamp jitter, e.g. "
                     f"{bad.index[:3].tolist()} at {bad.head(3).round(3).tolist()}")
    # diff() must be taken WITHIN each game: across a game boundary the offset resets to
    # 0, so a global diff scores every boundary as a backwards step and the rate becomes
    # ~1/pitches-per-game regardless of the data.
    n_steps = int(g["wx_hour_offset"].apply(lambda s: int((s.diff().dropna() < 0).sum())).sum())
    rate = n_steps / max(len(df), 1)
    if len(df) >= MIN_PITCHES_FOR_RATE and rate > MAX_SEASON_DECREASING_RATE:
        fails.append(f"backwards-step rate {rate:.2e} exceeds "
                     f"{MAX_SEASON_DECREASING_RATE:.0e}; the upstream timestamp floor is "
                     f"~5e-05, so this is a systemic regression, not jitter")
    elif n_steps:
        n_g = int((frac > 0).sum())
        print(f"       warn: {n_steps} backwards step(s) across {n_g} game(s) "
              f"(rate {rate:.2e}) — upstream pitch_start_time jitter, within the "
              f"known floor; see the comment in check_offsets")
    return fails


def check_cross(asof: pd.DataFrame, offsets: pd.DataFrame) -> list[str]:
    """Games with weather but no offsets silently fall back to the pregame row."""
    a = set(asof["game_pk"].unique())
    o = set(offsets["game_pk"].unique())
    fails = []
    if a - o:
        fails.append(f"{len(a - o)} game(s) have weather_asof but no wx_hour_offset; "
                     f"every live sample would read d=0, e.g. {sorted(a - o)[:3]}")
    return fails


def verify_season(season: int) -> bool:
    """Each artifact is checked whenever it exists.

    The offsets are built for every season well before the weather tensors are, so
    gating the offset checks on weather_asof's presence would leave the majority of
    them unverified for as long as the builds are still running -- which is exactly
    when a defect is cheapest to fix.
    """
    ok = True
    asof = offsets = None
    try:
        asof = _read(f"{FS_PREFIX}/weather_asof/season={season}.parquet",
                     columns=["game_pk", "decision_hour", "target_hour"])
    except Exception as exc:
        print(f"  ..   {season}: weather_asof not built yet ({type(exc).__name__})")
    try:
        offsets = _read(f"{FS_PREFIX}/wx_hour_offset/season={season}.parquet")
    except Exception as exc:
        print(f"  ..   {season}: wx_hour_offset absent ({type(exc).__name__})")

    if asof is None and offsets is None:
        print(f"  SKIP {season}: neither artifact present")
        return True

    checks = []
    if asof is not None:
        checks.append(("weather_asof", check_asof_blocks(asof)))
    if offsets is not None:
        checks.append(("wx_hour_offset", check_offsets(offsets)))
    if asof is not None and offsets is not None:
        checks.append(("cross", check_cross(asof, offsets)))
    # A weather tensor with no offsets is a real failure: every live sample in the
    # season would silently read the pregame decision row.
    elif asof is not None:
        checks.append(("cross", ["weather_asof exists but wx_hour_offset does not; all "
                                 "live samples would fall back to d=0"]))

    for label, fails in checks:
        for f in fails:
            print(f"  FAIL {season} [{label}]: {f}")
            ok = False
    if ok:
        n_g = asof["game_pk"].nunique() if asof is not None else 0
        n_o = len(offsets) if offsets is not None else 0
        print(f"  ok   {season}: {n_g} games x {ROWS_PER_GAME} rows, "
              f"{n_o} pitch offsets, all positional contracts hold")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    seasons = args.seasons or []
    if args.all:
        pg = _s3.get_paginator("list_objects_v2")
        # Union of both prefixes: enumerating only weather_asof would hide a season whose
        # offsets exist but whose tensor is missing, which is the one cross-artifact
        # failure that degrades silently instead of crashing.
        seasons = sorted({
            int(o["Key"].split("season=")[1][:4])
            for sub in ("weather_asof", "wx_hour_offset")
            for p in pg.paginate(Bucket=S3_BUCKET, Prefix=f"{FS_PREFIX}/{sub}/season=")
            for o in p.get("Contents", [])})
    if not seasons:
        sys.exit("nothing to verify: pass --seasons or --all")

    print(f"as-of loader contract audit: {seasons}")
    results = [verify_season(s) for s in seasons]
    if all(results):
        print("LOADER CONTRACT AUDIT PASSED")
    else:
        sys.exit(f"LOADER CONTRACT AUDIT FAILED ({results.count(False)} season(s))")


if __name__ == "__main__":
    main()
