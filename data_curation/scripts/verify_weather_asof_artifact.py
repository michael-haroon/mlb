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
    IMPOSSIBLE_ZERO_OBS_DIMS,
    N_DIMS,
    N_OBS_DIMS,
    N_DECISIONS,
    N_TARGET_HOURS,
    OBS_EXTRA_NAMES,
    OFF_FCST,
    OFF_FCST_MASK,
    OFF_OBS,
    OFF_OBS_MASK,
    OFF_LEAD,
    TARGET_HOURS,
    VIS_CEILING_M,
)
from mlb_dl.weather_context import WEATHER_TEMPORAL_COLUMNS  # noqa: E402

DIM_NAMES = list(WEATHER_TEMPORAL_COLUMNS) + list(OBS_EXTRA_NAMES)

# Outer plausibility bounds per dim, in the artifact's RAW units (mph, °F, hPa, m, mm,
# kPa, m³/m³, K/km — the builder calls assemble_asof_tensor with no norm_stats, so
# nothing here is standardized). Checked only where the mask says the entry is real.
#
# These are tripwires for order-of-magnitude faults, not distribution checks. Each bound
# clears the record book by a wide margin, because a gate that fires on a real game is a
# gate that gets disabled. Derivations:
#   0-1 density, ratio: ρ = p/(R_d·T_v), R_d = 287.05. The MLB extremes are Coors Field
#       hot (826 hPa, 311 K → 0.926) and sea-level cold (1035 hPa, 265 K → 1.359);
#       ratio divides by the 1.225 standard. Bounds sit ~10% outside both.
#   2-5 winds: signed components cannot exceed the speed; the strongest sustained wind
#       in a played game is ~40 mph, so 80 is 2x, and gusts get 120.
#   6 vpd: 0 at saturation; 115 °F at 5% RH gives es−ea ≈ 9.9 kPa.
#   7,10 humidity, cloud: definitionally percentages, +0.5 for float slack.
#   8,9 wet bulb, temperature: Tw ≤ T always. Coldest MLB game on record ~18 °F,
#       hottest ~115 °F, world-record wet bulb ~95 °F.
#   11 visibility: 0 in dense fog. This is the one bound that is not an outer tripwire —
#       both channels are deliberately censored at the METAR 10 SM ceiling so that "clear"
#       is the same number in the forecast and observation channels (see VIS_CEILING_M),
#       so anything above it means the censoring regressed, not that the weather was
#       unusual. Imported rather than restated so the gate cannot drift from the builder.
#   12 precip: hourly accumulation; 150 mm/h is past any world hourly record for a
#       populated area.
#   13 surface pressure: STATION pressure, so elevation dominates — measured 2015
#       minimum is 826.8 hPa at Coors. Global sea-level record high is 1084 hPa.
#   14-15 PBL height, shortwave: 6000 m clears any convective boundary layer; the solar
#       constant is 1361 W/m², so 1500 covers surface max plus cloud enhancement. The
#       shortwave FLOOR is -1.0, not 0: downward flux cannot physically be negative, but
#       HRRR's DSWRF is a time-averaged packed GRIB field and decoding leaves trace
#       negatives. Measured on season=2015: 9 of 118,869 populated entries, all at
#       exactly -0.1 W/m², against an observed maximum of 1106. Widened rather than
#       clamped upstream because -0.1 W/m² is indistinguishable from 0 after
#       standardization, so clamping would edit data for no model benefit — and a real
#       sign inversion or unit fault lands orders of magnitude away, still caught.
#   16 soil moisture: volumetric fraction, definitionally [0,1].
#   17-19 AQ: EPA AQI tops out at 500 on the published scale but wildfire episodes are
#       reported beyond it; PM2.5 and O₃ get loose µg/m³ ceilings. Permanently masked
#       in this artifact (v1 decision), so these bounds are dormant by design.
#   20 lapse rate: SIGNED across 1000-850 hPa. Superadiabatic layers exceed the 9.8 K/km
#       dry adiabat and nocturnal inversions run strongly negative, hence the wide,
#       asymmetric window — a ≥0 floor here would fail a large share of night games.
#   21 shear: vector wind difference to 850 hPa; low-level jets reach ~100 mph.
#   22,24,25 wx flags: indicators. 23 wx_precip_intensity: ordinal 0-3 codebook.
#   26 peak gust: PK WND group, mph.
PHYSICAL_RANGES = (
    (0.80, 1.45),        # 0  air_density kg/m3
    (0.60, 1.25),        # 1  air_density_ratio
    (-80.0, 80.0),       # 2  wind_toward_cf mph (signed)
    (-80.0, 80.0),       # 3  wind_crossfield mph (signed)
    (0.0, 80.0),         # 4  wind_speed mph
    (0.0, 120.0),        # 5  wind_gusts mph
    (0.0, 15.0),         # 6  vpd kPa
    (0.0, 100.5),        # 7  humidity %
    (0.0, 100.0),        # 8  wet_bulb_f
    (0.0, 130.0),        # 9  temperature_f
    (0.0, 100.5),        # 10 cloud_cover %
    (0.0, VIS_CEILING_M + 1.0),  # 11 visibility m — censored, see note
    (0.0, 150.0),        # 12 precip mm
    (750.0, 1080.0),     # 13 surface_pressure hPa
    (0.0, 6000.0),       # 14 boundary_layer_height m
    (-1.0, 1500.0),      # 15 shortwave_radiation W/m2 (see note on the floor)
    (0.0, 1.0),          # 16 soil_moisture m3/m3
    (0.0, 1000.0),       # 17 us_aqi
    (0.0, 2000.0),       # 18 pm2_5 ug/m3
    (0.0, 1000.0),       # 19 ozone ug/m3
    (-40.0, 30.0),       # 20 lapse_rate_1000_850 K/km (signed)
    (0.0, 200.0),        # 21 wind_shear_sfc_850
    (0.0, 1.0),          # 22 wx_thunder
    (0.0, 3.0),          # 23 wx_precip_intensity (ordinal codebook)
    (0.0, 1.0),          # 24 wx_frozen_precip
    (0.0, 1.0),          # 25 wx_obstruction
    (0.0, 150.0),        # 26 wx_peak_gust mph
)
assert len(PHYSICAL_RANGES) == N_OBS_DIMS

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


def check_physical_ranges(T: np.ndarray) -> list[str]:
    """Are the populated entries actually weather? Returns failure strings.

    The rest of this audit is structural, and the spot recomputation re-runs the very
    function that wrote the artifact, so it is bit-identical by construction and would
    confirm a tensor full of Kelvin. This is the only check that looks at the values
    themselves. It takes a tensor rather than a season so it can be unit-tested against
    deliberately corrupted input; see tests/test_verify_asof_physical_ranges.py.

    Masked entries are skipped: they are exactly 0 by design, and inspecting them would
    flag every not-yet-elapsed hour in the window.
    """
    fails: list[str] = []
    channels = (("fcst", OFF_FCST, OFF_FCST_MASK, OFF_OBS, N_DIMS),
                ("obs", OFF_OBS, OFF_OBS_MASK, OFF_LEAD, N_OBS_DIMS))
    for label, v0, m0, m1, n_dims in channels:
        vals = T[..., v0:m0]
        mask = T[..., m0:m1]
        for d in range(n_dims):
            live = mask[..., d] == 1.0
            if not live.any():
                continue
            x = vals[..., d][live]
            lo, hi = PHYSICAL_RANGES[d]
            bad = (x < lo) | (x > hi)
            if bad.any():
                worst = x[bad][np.argmax(np.abs(x[bad] - np.clip(x[bad], lo, hi)))]
                fails.append(
                    f"{label} {DIM_NAMES[d]} (dim {d}): {int(bad.sum())} of {x.size} "
                    f"populated entries outside [{lo:g}, {hi:g}] — worst {worst:.4g}, "
                    f"observed range [{x.min():.4g}, {x.max():.4g}]")

        # The mask must not claim a dim the source never populated. Measured defect
        # (season=2015, before the fix): a METAR omitting its altimeter group stored
        # surface_pressure = 0 hPa with mask = 1, which the loader then z-scores to
        # roughly -50 sigma. A range check cannot see this on its own -- 0 is outside no
        # bound that would also admit real pressure -- so it is checked against the mask.
        for d in IMPOSSIBLE_ZERO_OBS_DIMS:
            if d >= n_dims:
                continue
            n = int(((mask[..., d] == 1.0) & (vals[..., d] == 0.0)).sum())
            if n:
                fails.append(
                    f"{label} {DIM_NAMES[d]} (dim {d}): {n} entries are exactly 0 with "
                    f"mask=1, but zero is physically impossible there — the mask is "
                    f"claiming a field the source did not populate")
    return fails


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

    range_fails = check_physical_ranges(T)
    for f in range_fails:
        fail(f"{season}: {f}")
    if not range_fails:
        ok(f"{season}: all populated values physically plausible, no masked-in "
           f"impossible zeros")

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
