"""Count how many BUILT weather_asof entries violate the artifact's plausibility bounds.

The full artifact audit answers "is this season clean, yes or no" and stops there, which is
the right question for a gate but the wrong one for a scheduling decision: a season with
four bad entries and a season with four million both come back FAIL. When an A/B is already
running on the artifact, the choice between "restart now" and "rebuild after" turns entirely
on that magnitude, so this reports the count and the share.

Reuses check_physical_ranges and PHYSICAL_RANGES from the audit rather than restating either,
so this can never disagree with the gate it is quantifying. Skips the spot recomputation and
the coverage checks, which is what makes it cheap enough to run on a shard box.

Read-only. Usage:
    python3.11 data_curation/scripts/count_asof_bound_violations.py [--seasons 2018 2024]
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "deep_learning"))
sys.path.insert(0, str(ROOT / "data_curation" / "scripts"))

from mlb_dl.weather_asof import (  # noqa: E402
    ASOF_CHANNELS, N_DECISIONS, N_TARGET_HOURS, OFF_FCST_MASK, OFF_LEAD, OFF_OBS,
    OFF_OBS_MASK,
)
from verify_weather_asof_artifact import (  # noqa: E402
    DIM_NAMES, PHYSICAL_RANGES, S3_BUCKET, check_physical_ranges,
)

CHANNEL_COLS = [f"wx_c{i:02d}" for i in range(ASOF_CHANNELS)]
s3 = boto3.client("s3", region_name="us-east-1")


def load(season: int) -> np.ndarray:
    key = f"deep_learning/feature_store/weather_asof/season={season}.parquet"
    body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    df = pd.read_parquet(io.BytesIO(body)).sort_values(
        ["game_pk", "decision_hour", "target_hour"])
    n_games = df["game_pk"].nunique()
    arr = df[CHANNEL_COLS].to_numpy(np.float32)
    return arr.reshape(n_games, N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS), n_games


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=list(range(2015, 2027)))
    a = ap.parse_args()

    grand_bad = 0
    grand_pop = 0
    print(f"{'season':>7} {'games':>7} {'populated':>12} {'violations':>11}  detail")
    for season in a.seasons:
        T, n_games = load(season)
        # Populated obs+fcst entries, for the denominator the share is quoted against.
        pop = int((T[..., OFF_FCST_MASK:OFF_OBS] == 1.0).sum()
                  + (T[..., OFF_OBS_MASK:OFF_LEAD] == 1.0).sum())
        fails = check_physical_ranges(T)
        n_bad = sum(int(m.group(1)) for f in fails
                    if (m := re.search(r": (\d+) of \d+ populated", f)))
        grand_bad += n_bad
        grand_pop += pop
        detail = "clean" if not fails else "; ".join(
            f.split(" — ")[0].replace(" populated entries outside", " outside")
            for f in fails)
        print(f"{season:>7} {n_games:>7} {pop:>12,} {n_bad:>11}  {detail}")

    share = 100.0 * grand_bad / max(grand_pop, 1)
    print(f"\nTOTAL {grand_bad} violating entries of {grand_pop:,} populated "
          f"({share:.7f}%)")
    # No exit(1): this script quantifies, the audit gates. Returning nonzero here would
    # make it useless inside the `if audit fails, how bad is it` branch it exists for.


if __name__ == "__main__":
    main()
