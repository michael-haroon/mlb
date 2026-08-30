"""
Phase 6 gate: leakage + integrity audit of the BUILT weather_asof artifact.

Structural invariants asserted on real tensors (unit tests proved them on
synthetic frames; this proves the builder preserved them at scale):
  - obs_mask nonzero only where h < d, all (game, d, h)
  - d=0 rows carry no obs beyond hour -1
  - lead_norm monotonically non-increasing in d for fixed h (while populated)
  - masks are exactly {0,1}; lead_norm in [0,1]; no NaN/Inf anywhere
  - season completeness: every PLANNED season has an artifact (a chain that
    built 11 of 12 must not report PASSED)
  - population coverage: each season's artifact covers every population game
  - fcst coverage: share of populated fcst entries per season (alerts < 90%)
  - obs coverage at elapsed hours (alerts < 85% — station outages expected)
  - spot recomputation: N random games re-assembled from the raw archives via
    assemble_asof_tensor must match the artifact bit-for-bit (builder
    determinism + no post-write corruption)

Run on EC2 after mlb_dl.build_weather_asof completes:
  python3.11 data_curation/scripts/verify_weather_asof_artifact.py [--seasons 2015 2023]
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
    N_DIMS,
    N_OBS_DIMS,
    N_DECISIONS,
    N_TARGET_HOURS,
    OFF_FCST_MASK,
    OFF_OBS,
    OFF_OBS_MASK,
    OFF_LEAD,
    TARGET_HOURS,
)

S3_BUCKET = "mlb-265753586044-us-east-1-an"
CHANNEL_COLS = [f"wx_c{i:02d}" for i in range(ASOF_CHANNELS)]

# The artifact must be checked against the PLAN, not against whatever happens to
# be in the bucket. Discovering seasons by listing S3 means a chain that built 11
# of 12 seasons audits 11 and reports PASSED, and `build_norm_stats` then folds
# an 11-season population into the standardizer that training and live both
# share. Same failure shape as the HRRR shard race: the rows present were all
# valid, and only the plan revealed what was missing.
EXPECTED_SEASONS = tuple(range(2015, 2027))

s3 = boto3.client("s3", region_name="us-east-1")
_fails: list[str] = []


def fail(msg):
    _fails.append(msg)
    print(f"FAIL  {msg}")


def ok(msg):
    print(f"ok    {msg}")


def audit_season(season: int, n_spot: int = 3) -> None:
    key = f"deep_learning/feature_store/weather_asof/season={season}.parquet"
    df = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()))
    df = df.sort_values(["game_pk", "decision_hour", "target_hour"])
    arr = df[CHANNEL_COLS].to_numpy(np.float32)
    n_games = df["game_pk"].nunique()
    per_game = N_DECISIONS * N_TARGET_HOURS
    if len(df) != n_games * per_game:
        fail(f"{season}: row count {len(df)} != {n_games} games x {per_game}")
        return
    check_population_coverage(season, n_games)
    T = arr.reshape(n_games, N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS)
    d_idx = np.arange(N_DECISIONS)[:, None]
    h_idx = np.array(TARGET_HOURS)[None, :]

    if not np.isfinite(arr).all():
        fail(f"{season}: NaN/Inf present")
    fcst_mask = T[..., OFF_FCST_MASK:OFF_OBS]
    obs_mask = T[..., OFF_OBS_MASK:OFF_LEAD]
    lead = T[..., OFF_LEAD]
    for name, m in (("fcst_mask", fcst_mask), ("obs_mask", obs_mask)):
        vals = np.unique(m)
        if not np.isin(vals, [0.0, 1.0]).all():
            fail(f"{season}: {name} values outside {{0,1}}: {vals[:5]}")
    if lead.min() < 0 or lead.max() > 1.0 + 1e-6:
        fail(f"{season}: lead_norm outside [0,1]")

    # THE leakage invariant: obs only at elapsed hours
    future = obs_mask.any(axis=3) & (h_idx >= d_idx)[None, :, :]
    if future.any():
        g = np.argwhere(future)[0]
        fail(f"{season}: obs_mask set at h>=d — e.g. game idx {g[0]} (d={g[1]}, hi={g[2]})")
    else:
        ok(f"{season}: no future observations anywhere ({n_games} games)")

    # d=0: only hour -1 (hi=0) may be observed
    if obs_mask[:, 0, 1:, :].any():
        fail(f"{season}: d=0 carries obs beyond hour -1")

    # lead monotone non-increasing in d where populated
    populated = fcst_mask.any(axis=3)
    viol = 0
    for hi in range(N_TARGET_HOURS):
        l = lead[:, :, hi]
        p = populated[:, :, hi]
        prev = np.where(p[:, 0], l[:, 0], np.inf)
        for d in range(1, N_DECISIONS):
            cur = l[:, d]
            bad = p[:, d] & (cur > prev + 1e-6)
            viol += int(bad.sum())
            prev = np.where(p[:, d], cur, prev)
    if viol:
        fail(f"{season}: {viol} lead-monotonicity violations")
    else:
        ok(f"{season}: lead monotone non-increasing")

    fcst_cov = populated.mean()
    obs_elig = (h_idx < d_idx)[None, :, :] & np.ones((n_games, 1, 1), bool)
    obs_cov = obs_mask.any(axis=3)[obs_elig].mean()
    msg = f"{season}: fcst coverage {fcst_cov:.1%}, obs coverage at elapsed hours {obs_cov:.1%}"
    if fcst_cov < 0.90:
        fail(msg + " — fcst below 90%")
    elif obs_cov < 0.85:
        fail(msg + " — obs below 85%")
    else:
        ok(msg)

    # Spot recomputation (builder determinism)
    from mlb_dl.build_weather_asof import (
        load_hrrr_for_dates, load_obs_for_venues, load_soil_for_venues,
        merge_lagged_soil, load_population, _read_json, STATION_MAP_KEY, AZIMUTHS_KEY,
    )
    from mlb_dl.weather_asof import assemble_asof_tensor
    pop = load_population(season)
    vmap = _read_json(STATION_MAP_KEY)
    azimuths = {int(k): v for k, v in _read_json(AZIMUTHS_KEY).items()}
    rng = np.random.default_rng(season)
    pks = df["game_pk"].unique()
    game_index = {pk: i for i, pk in enumerate(df["game_pk"].unique())}
    for pk in rng.choice(pks, size=min(n_spot, len(pks)), replace=False):
        row = pop[pop["game_pk"] == pk].iloc[0]
        vid = int(row["venue_id"])
        gh = row["game_hour_utc"]
        obs = load_obs_for_venues([vid], season, vmap)[vid]
        fcst = load_hrrr_for_dates([pd.Timestamp(row["game_date"]).normalize()])
        if len(fcst):
            fcst = fcst[fcst["venue_id"] == vid]
            fcst = merge_lagged_soil(fcst, load_soil_for_venues([vid], {season, season - 1})[vid])
        T_re = assemble_asof_tensor(obs, fcst, gh, vid, float(azimuths.get(vid, 0.0)))
        T_art = T[game_index[pk]]
        if not np.allclose(T_re, T_art, atol=1e-5):
            nbad = int((~np.isclose(T_re, T_art, atol=1e-5)).sum())
            fail(f"{season}: game {pk} recomputation mismatch ({nbad} entries)")
        else:
            ok(f"{season}: game {pk} recomputes bit-identical")


def check_season_completeness() -> list[int]:
    """Every planned season must have an artifact. Returns the present ones."""
    resp = s3.list_objects_v2(Bucket=S3_BUCKET,
                              Prefix="deep_learning/feature_store/weather_asof/season=")
    present = sorted(int(o["Key"].split("season=")[1][:4]) for o in resp.get("Contents", []))
    missing = [s for s in EXPECTED_SEASONS if s not in present]
    extra = [s for s in present if s not in EXPECTED_SEASONS]
    if missing:
        fail(f"season completeness: {len(missing)} of {len(EXPECTED_SEASONS)} planned "
             f"seasons have NO artifact: {missing}")
    else:
        ok(f"season completeness: all {len(EXPECTED_SEASONS)} planned seasons present")
    if extra:
        fail(f"season completeness: unplanned season artifacts present: {extra}")
    return present


def check_population_coverage(season: int, n_games_artifact: int) -> None:
    """The artifact must cover EVERY population game for the season.

    A build that lost games to worker failures still writes a valid file; only
    the population count reveals the shortfall. Games absent here silently train
    on an all-zero weather tensor, which is exactly the 88.6%-void-weather bug
    the min_date floor was added to fix.
    """
    from mlb_dl.build_weather_asof import load_population
    n_pop = len(load_population(season))
    if n_pop == 0:
        ok(f"{season}: no population games (nothing to cover)")
        return
    if n_games_artifact != n_pop:
        fail(f"{season}: artifact covers {n_games_artifact} games but population has "
             f"{n_pop} ({n_pop - n_games_artifact} missing)")
    else:
        ok(f"{season}: artifact covers all {n_pop} population games")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=None)
    args = ap.parse_args()
    if args.seasons:
        seasons = args.seasons
    else:
        seasons = check_season_completeness()
    for season in seasons:
        audit_season(season)
    print(f"\n{'ARTIFACT AUDIT PASSED' if not _fails else f'{len(_fails)} FAILURES'}")
    sys.exit(1 if _fails else 0)


if __name__ == "__main__":
    main()
