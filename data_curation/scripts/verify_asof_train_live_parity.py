"""
Train/live parity replay for the as-of weather tensor.

`assemble_asof_tensor` is shared, so the physics cannot diverge. What CAN
diverge is the retrieval layer feeding it: the training builder reads
`source=asos_obs` (station-year files) + `source=hrrr_asissued` (all planned
issues per date), while the live path reads `source=asos_obs_live` (daemon
appends) + `source=hrrr_asissued_live` (freshest issue per hourly poll). A
mismatch there is invisible to unit tests and to every per-source check, and it
shows up as a silent train/serve skew the moment the model goes live.

This replays completed games through BOTH retrieval paths and compares:

  1. Where both paths have data (mask==1 in both), values must be EXACTLY
     equal — same assembler, same inputs, so any delta is a retrieval bug.
  2. Cells the live path has but training lacks are a FAILURE: it means the
     training archive is missing rows we demonstrably can obtain.
  3. Cells training has but live lacks are reported, not failed: the live
     archive only accumulates from when the daemon started, so early decision
     hours are legitimately empty for games near that boundary.

Usage (needs S3 read; run on any box with the repo):
  python3.11 data_curation/scripts/verify_asof_train_live_parity.py \
      --date 2026-08-30 --n-games 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "deep_learning"))

from mlb_dl.build_weather_asof import (  # noqa: E402
    _read_json,
    load_obs_for_venues,
    load_hrrr_for_dates,
    load_population,
    load_soil_for_venues,
    merge_lagged_soil,
    AZIMUTHS_KEY,
    STATION_MAP_KEY,
)
from mlb_dl.weather_asof import (  # noqa: E402
    N_DECISIONS,
    N_DIMS,
    N_OBS_DIMS,
    N_TARGET_HOURS,
    OFF_FCST,
    OFF_FCST_MASK,
    OFF_OBS,
    OFF_OBS_MASK,
    TARGET_HOURS,
    assemble_asof_tensor,
    fetch_live_asof,
    load_norm_stats,
)

_fails: list[str] = []


def fail(msg: str) -> None:
    _fails.append(msg)
    print(f"FAIL  {msg}")


def ok(msg: str) -> None:
    print(f"ok    {msg}")


def _train_tensor(pk: int, vid: int, game_hour: pd.Timestamp, obs, fcst_all,
                  soil, az: float, norm_stats) -> np.ndarray:
    fcst = fcst_all
    if len(fcst_all):
        fcst = fcst_all[(fcst_all["venue_id"] == vid)
                        & (fcst_all["valid_time_utc"] >= game_hour + pd.Timedelta(hours=TARGET_HOURS[0]))
                        & (fcst_all["valid_time_utc"] <= game_hour + pd.Timedelta(hours=TARGET_HOURS[-1]))]
        fcst = merge_lagged_soil(fcst, soil)
    return assemble_asof_tensor(obs, fcst, game_hour, vid, az, norm_stats=norm_stats)


def compare(train: np.ndarray, live: np.ndarray) -> dict:
    """Per-cell comparison restricted to the group each mask governs."""
    res = {"max_delta": 0.0, "both": 0, "live_only": 0, "train_only": 0}
    groups = ((OFF_FCST, OFF_FCST_MASK, N_DIMS), (OFF_OBS, OFF_OBS_MASK, N_OBS_DIMS))
    for off, moff, n in groups:
        mt = train[..., moff:moff + n]
        ml = live[..., moff:moff + n]
        both = (mt > 0.5) & (ml > 0.5)
        res["both"] += int(both.sum())
        res["live_only"] += int(((ml > 0.5) & (mt <= 0.5)).sum())
        res["train_only"] += int(((mt > 0.5) & (ml <= 0.5)).sum())
        if both.any():
            d = np.abs(train[..., off:off + n][both] - live[..., off:off + n][both])
            res["max_delta"] = max(res["max_delta"], float(d.max()))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=str(pd.Timestamp.utcnow().date()))
    ap.add_argument("--n-games", type=int, default=20)
    args = ap.parse_args()

    day = pd.Timestamp(args.date)
    norm_stats = load_norm_stats()
    if norm_stats is None:
        fail("no weather_asof_norm.json on S3 — live path cannot standardize; "
             "build norm-stats before trusting live inference")
        print(f"\n{len(_fails)} FAILURES")
        sys.exit(1)

    vmap = _read_json(STATION_MAP_KEY)
    azimuths = {int(k): v for k, v in _read_json(AZIMUTHS_KEY).items()}
    pop = load_population(day.year)
    pop = pop[pop["game_date"].dt.normalize() == day.normalize()]
    if pop.empty:
        fail(f"no population games on {day:%Y-%m-%d}")
        print(f"\n{len(_fails)} FAILURES")
        sys.exit(1)
    pop = pop[pop["venue_id"].astype(str).isin(vmap.keys())].head(args.n_games)
    ok(f"replaying {len(pop)} games on {day:%Y-%m-%d}")

    venue_ids = sorted(pop["venue_id"].unique())
    obs_by_v = load_obs_for_venues(venue_ids, day.year, vmap)
    fcst_all = load_hrrr_for_dates([day.normalize()])
    soil_by_v = load_soil_for_venues(venue_ids, {day.year, day.year - 1})

    # `now` must be late enough that every decision hour has passed, otherwise
    # the live path's own availability filter (correctly) empties late cells and
    # the comparison measures the clock rather than the retrieval layer.
    now = pd.Timestamp.now(tz="UTC")
    agg = {"max_delta": 0.0, "both": 0, "live_only": 0, "train_only": 0}
    n_live_missing = 0
    for pk, _gd, vid, gh in pop[["game_pk", "game_date", "venue_id", "game_hour_utc"]].itertuples(
            index=False, name=None):
        vid = int(vid)
        az = float(azimuths.get(vid, 0.0))
        tt = _train_tensor(int(pk), vid, pd.Timestamp(gh), obs_by_v.get(vid),
                           fcst_all, soil_by_v.get(vid), az, norm_stats)
        lt = fetch_live_asof(vid, pd.Timestamp(gh), az, now=now,
                             station_map=vmap, norm_stats=norm_stats)
        if lt is None:
            n_live_missing += 1
            continue
        r = compare(tt, lt)
        agg["max_delta"] = max(agg["max_delta"], r["max_delta"])
        for k in ("both", "live_only", "train_only"):
            agg[k] += r[k]
        if r["max_delta"] > 0:
            fail(f"game {pk} venue {vid}: max |train-live| = {r['max_delta']:.6g} on "
                 f"{r['both']} commonly-populated dims — retrieval divergence")
        if r["live_only"] > 0:
            fail(f"game {pk} venue {vid}: {r['live_only']} dims populated LIVE but not in "
                 f"training — the training archive is missing obtainable rows")

    total_cells = len(pop) * N_DECISIONS * N_TARGET_HOURS * (N_DIMS + N_OBS_DIMS)
    ok(f"compared {agg['both']} commonly-populated dims of {total_cells}; "
       f"max delta {agg['max_delta']:.6g}")
    ok(f"train-only dims (expected near the daemon-start boundary): {agg['train_only']}")
    if n_live_missing:
        ok(f"{n_live_missing}/{len(pop)} games had no live tensor at all "
           f"(daemon coverage, not a parity defect)")
    if agg["both"] == 0:
        fail("zero commonly-populated dims — the replay proved nothing; "
             "pick a date with both archives populated")

    print(f"\n{'PARITY HOLDS' if not _fails else f'{len(_fails)} FAILURES'}")
    sys.exit(1 if _fails else 0)


if __name__ == "__main__":
    main()
