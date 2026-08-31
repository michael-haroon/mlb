#!/usr/bin/env python3.11
"""Which train cutoff does each norm-stat artifact actually correspond to?

Three different notions of "end of train" coexist in the DL stack, and only one of them is the
truth used at training time:

    rating_sequences.py CLI  --train-end default   2024-04-01   (fits rating means/stds)
    build_weather_asof.py    TRAIN_END_DATE        2024-01-01   (fits the weather norm sidecar)
    datasets.temporal_split_dates(game_targets)    80% quantile over DISTINCT game dates

The third is not a constant. It moves whenever the population changes -- 2024-05-14 on the void
1950-train cache, 2024-08-03 on the corrected one -- so the two hardcoded dates cannot be
assumed to stay on the safe side of it.

Direction is what matters, and it is asymmetric:
  * cutoff EARLIER than the real train_end -> the fit merely drops train games. Wasteful,
    slightly noisier norm stats, no leakage.
  * cutoff LATER  -> val/test statistics enter the standardizer. That is leakage, and it
    inflates held-out metrics in a way no downstream check would catch.

Run this after any population change (new seasons, a changed floor, a SKIP_SEASONS edit) rather
than assuming the last answer still holds.

Usage:
  conda run -n pred python deep_learning/audit_split_cutoffs.py
  conda run -n pred python deep_learning/audit_split_cutoffs.py --game-targets /tmp/gt.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlb_dl.datasets import temporal_split_dates  # noqa: E402

S3_DEFAULT = ("s3://mlb-265753586044-us-east-1-an/deep_learning/feature_store/"
              "game_targets.parquet")

# The cutoffs to audit, as (label, source-of-truth location, date).
CUTOFFS = [
    ("rating_sequences --train-end default", "mlb_dl/rating_sequences.py CLI", "2024-04-01"),
    ("build_weather_asof TRAIN_END_DATE", "mlb_dl/build_weather_asof.py:79", "2024-01-01"),
]

# Population filters that define the trainable set. Kept explicit rather than imported so this
# script states the population it is reasoning about; if these drift from the real loader the
# printed game counts will visibly disagree with the prepared manifest.
STATCAST_FLOOR_YEAR = 2015
SKIP_SEASONS = (2020,)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-targets", default=S3_DEFAULT,
                    help="Path or S3 URI to game_targets.parquet")
    args = ap.parse_args()

    if args.game_targets.startswith("s3://"):
        import s3fs
        with s3fs.S3FileSystem().open(args.game_targets, "rb") as f:
            gt = pd.read_parquet(f, columns=["game_pk", "game_date"])
    else:
        gt = pd.read_parquet(args.game_targets, columns=["game_pk", "game_date"])

    gt["game_date"] = pd.to_datetime(gt["game_date"], errors="coerce")
    print(f"game_targets: {len(gt):,} rows, "
          f"{gt['game_date'].min().date()}..{gt['game_date'].max().date()}")

    # Unfiltered, for contrast only: this is what the split looks like if the population floor
    # is not applied, and it is the shape that produced the 1950-train bug.
    tr_raw, va_raw = temporal_split_dates(gt)
    print(f"\nquantile split on the RAW artifact (no floor): "
          f"train_end={tr_raw.date()}  val_end={va_raw.date()}")

    pop = gt[(gt["game_date"].dt.year >= STATCAST_FLOOR_YEAR)
             & (~gt["game_date"].dt.year.isin(SKIP_SEASONS))]
    train_end, val_end = temporal_split_dates(pop)
    print(f"quantile split on the {STATCAST_FLOOR_YEAR}+ / no-{SKIP_SEASONS[0]} population: "
          f"train_end={train_end.date()}  val_end={val_end.date()}")
    print(f"  population {len(pop):,} games, {pop['game_date'].dt.date.nunique():,} distinct dates")
    n_tr = int((pop["game_date"] < train_end).sum())
    n_va = int(((pop["game_date"] >= train_end) & (pop["game_date"] < val_end)).sum())
    n_te = int((pop["game_date"] >= val_end).sum())
    print(f"  -> train {n_tr:,} / val {n_va:,} / test {n_te:,}")
    print("  (compare against the prepared manifest's per-split n_games; a gap means the real "
          "loader applies filters beyond the floor, e.g. game_type or target availability)")

    leaks = 0
    for label, where, cut in CUTOFFS:
        c = pd.Timestamp(cut)
        print(f"\n{label}  ({where}) = {cut}")
        if c < train_end:
            dropped = int(((pop["game_date"] >= c) & (pop["game_date"] < train_end)).sum())
            print(f"  EARLIER than train_end={train_end.date()} -> conservative, no leakage")
            print(f"  train games excluded from the fit: {dropped:,}")
        else:
            leaked = int(((pop["game_date"] >= train_end) & (pop["game_date"] < c)).sum())
            print(f"  LATER than train_end={train_end.date()} -> VAL/TEST LEAK INTO NORM STATS")
            print(f"  val/test games included in the fit: {leaked:,}")
            leaks += 1

    # Nonzero exit so this is safe to wire into a gate: a leaking cutoff is not a warning.
    return 1 if leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
