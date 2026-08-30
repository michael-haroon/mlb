"""
Phase 3 builder: the as-of weather training artifact.

Produces, under the DL feature store prefix:
  weather_asof.parquet       one row per (game_pk, decision_hour, target_hour)
                             with the 99 raw channels (wx_c00..wx_c98) —
                             values UNSTANDARDIZED; masks/lead final.
  weather_asof_norm.json     per-dim mean/std over populated TRAIN entries
                             (game_date < first val date) — applied at
                             precollate and carried into checkpoints for live.
  wx_hour_offset.parquet     per-pitch decision-hour offset int8, keyed
                             (game_pk, sequence_index) — aligned with
                             build_pitch_sequence_frame's sort + cumcount.

Every tensor entry goes through weather_asof.assemble_asof_tensor — the SAME
function the live path calls — so train/live parity holds by construction.

Obs frames concatenate the venue's primary and backup stations (each converted
with its own elevation): select_asof_obs picks the freshest report regardless
of station, so a dark primary (DMH 2021) degrades to the backup per-hour
instead of masking the whole season.

Run on EC2 (never locally):
  python3.11 -m mlb_dl.build_weather_asof build --season 2015 [--workers 8]
  python3.11 -m mlb_dl.build_weather_asof norm-stats
  python3.11 -m mlb_dl.build_weather_asof pitch-offsets --season 2015
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd

from .weather_asof import (
    ASOF_CHANNELS,
    DECISION_HOURS,
    N_DIMS,
    N_OBS_DIMS,
    OFF_FCST,
    OFF_FCST_MASK,
    OFF_OBS,
    OFF_OBS_MASK,
    TARGET_HOURS,
    assemble_asof_tensor,
    hrrr_to_era5,
    metar_to_era5,
)

S3_BUCKET = "mlb-265753586044-us-east-1-an"
FS_PREFIX = "deep_learning/feature_store"
AZIMUTHS_KEY = "classical_learning/artifacts/features/park_azimuths.json"
STATION_MAP_KEY = "data/weather/station_venue_map.json"

POP_MIN_DATE = "2015-01-15"
POP_GAME_TYPES = ["R", "F", "D", "L", "W"]

# Soil persistence lag (D8): live can only know ERA5 ~7 days back
# (ARCHIVE_LAG_DAYS in fetch_weather.py); training must match.
SOIL_LAG = pd.Timedelta(days=7)

# First val-split date — norm stats must come from train games only.
# Matches the 80/10/10 temporal split of the corrected population.
# TODO: validate — read the actual boundary from the prepared-tensor manifest
# at precollate time; this constant only gates norm-stats inclusion.
TRAIN_END_DATE = "2024-01-01"

CHANNEL_COLS = [f"wx_c{i:02d}" for i in range(ASOF_CHANNELS)]

logger = logging.getLogger("BUILD_WX_ASOF")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    os.makedirs("data/logs", exist_ok=True)
    fh = logging.FileHandler("data/logs/build_weather_asof.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[WXASOF] %(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)

_s3 = None


def s3():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3", region_name="us-east-1")
    return _s3


def _read_parquet(key: str, columns=None) -> pd.DataFrame:
    body = s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    return pd.read_parquet(io.BytesIO(body), columns=columns)


def _read_json(key: str) -> dict:
    return json.loads(s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read())


def _write_parquet(df: pd.DataFrame, key: str) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
    s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())


# ── Inputs ────────────────────────────────────────────────────────────────────
def load_population(season: int) -> pd.DataFrame:
    gm = _read_parquet(f"{FS_PREFIX}/game_meta.parquet")
    gm["game_date"] = pd.to_datetime(gm["game_date"])
    pop = gm[(gm["game_date"] >= POP_MIN_DATE)
             & gm["game_type_code"].isin(POP_GAME_TYPES)
             & (gm["game_date"].dt.year == season)].dropna(subset=["venue_id"]).copy()
    pop["venue_id"] = pop["venue_id"].astype(int)
    pop["game_hour_utc"] = pd.to_datetime(pop["game_datetime_utc"], utc=True).dt.floor("h")
    return pop[["game_pk", "game_date", "venue_id", "game_hour_utc"]]


def load_obs_for_venues(venue_ids: list[int], year: int, vmap: dict) -> dict[int, pd.DataFrame]:
    """venue_id -> concatenated era5-schema obs frame (primary + backup)."""
    cache: dict[str, pd.DataFrame] = {}
    out: dict[int, pd.DataFrame] = {}
    for vid in venue_ids:
        m = vmap.get(str(vid))
        if m is None:
            continue
        frames = []
        for st, elev in ((m["primary_station"], m.get("primary_elev_m")),
                         (m["backup_station"], m.get("backup_elev_m"))):
            ck = f"{st}:{year}:{elev}"
            if ck not in cache:
                try:
                    raw = _read_parquet(f"data/weather/source=asos_obs/station={st}/year={year}.parquet")
                    cache[ck] = metar_to_era5(raw, float(elev or 0.0))
                except Exception as exc:
                    logger.debug(f"obs {st} {year}: {exc}")
                    cache[ck] = pd.DataFrame()
            if len(cache[ck]):
                frames.append(cache[ck])
        out[vid] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out


# A complete season still misses a few HRRR dates for reasons no rerun can fix: genuine
# era gaps where the GRIB was never published (1 date in 2015, 2 in 2017) and dates whose
# only venue is outside the CONUS domain (the Tokyo and Seoul openers — 2 dates each in
# 2019, 2024, 2025). Worst observed legitimate absence is 4 of ~200 dates, or 2%. The
# floor sits well above that and far below the 38.5% shortfall that actually shipped, so
# it cannot block a real season and cannot miss a stale build.
MIN_FCST_DATE_COVERAGE = 0.95


def assert_fcst_dates_complete(season: int, dates: list, missing: list) -> None:
    """Refuse to build a season whose HRRR archive is still filling in.

    weather_asof/season=2015.parquet was written while 77 of 200 date files did not yet
    exist. The build logged a warning per absence and wrote anyway, producing an artifact
    that was correct in shape and covered every population game while carrying only 59%
    of its forecast signal. That is the worst kind of defect for an A/B: it trains
    without complaining and quietly understates the treatment arm.
    """
    if not dates:
        return
    cov = 1.0 - (len(missing) / len(dates))
    if cov < MIN_FCST_DATE_COVERAGE:
        tags = [f"{pd.Timestamp(d):%Y-%m-%d}" for d in missing[:8]]
        raise SystemExit(
            f"REFUSING to build {season}: only {cov:.1%} of the season's "
            f"{len(dates)} HRRR dates are archived ({len(missing)} missing, floor "
            f"{MIN_FCST_DATE_COVERAGE:.0%}). The extraction is probably still running. "
            f"Missing e.g. {tags}. Rerun the backfill for those ranges, confirm with "
            f"verify_weather_archives.py coverage --year {season}, then rebuild."
        )


def load_hrrr_for_dates(dates: list[pd.Timestamp],
                        missing_out: list | None = None) -> pd.DataFrame:
    """`missing_out`, when given, collects the dates that had no archive object.

    Absences used to be logged and nothing more, which is exactly how a 61%-complete
    archive produced a full-looking artifact. Callers that want the old behaviour can
    still ignore it; the two verifier scripts call this positionally.
    """
    frames = []
    for d in dates:
        try:
            frames.append(_read_parquet(f"data/weather/source=hrrr_asissued/date={d:%Y-%m-%d}.parquet"))
        except Exception:
            logger.warning(f"hrrr date file missing: {d:%Y-%m-%d}")
            if missing_out is not None:
                missing_out.append(d)
    if not frames:
        return pd.DataFrame()
    return hrrr_to_era5_with_soil_placeholder(pd.concat(frames, ignore_index=True))


def hrrr_to_era5_with_soil_placeholder(raw: pd.DataFrame) -> pd.DataFrame:
    df = hrrr_to_era5(raw)
    df["venue_id"] = raw["venue_id"].values
    return df


def load_soil_for_venues(venue_ids: list[int], years: set[int]) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for vid in venue_ids:
        frames = []
        for y in sorted(years):
            try:
                frames.append(_read_parquet(
                    f"data/weather/source=era5/venue_id={vid}/year={y}.parquet",
                    columns=["timestamp", "soil_moisture_0_to_7cm"]))
            except Exception:
                pass
        if frames:
            soil = pd.concat(frames, ignore_index=True)
            soil["timestamp"] = pd.to_datetime(soil["timestamp"], utc=True)
            out[vid] = soil.dropna(subset=["soil_moisture_0_to_7cm"])
        else:
            out[vid] = pd.DataFrame()
    return out


def merge_lagged_soil(fcst: pd.DataFrame, soil: pd.DataFrame) -> pd.DataFrame:
    """Attach soil_moisture_0_to_7cm at valid_time - 7d (the live-knowable lag)."""
    if fcst.empty:
        return fcst
    fcst = fcst.copy()
    if soil is None or soil.empty:
        fcst["soil_moisture_0_to_7cm"] = np.nan
        return fcst
    lagged = soil.copy()
    lagged["valid_time_utc"] = lagged["timestamp"] + SOIL_LAG
    return fcst.merge(
        lagged[["valid_time_utc", "soil_moisture_0_to_7cm"]],
        on="valid_time_utc", how="left")


# ── Per-game assembly (workers inherit season frames via fork) ───────────────
_G: dict = {}


def _assemble_one(args) -> tuple[int, np.ndarray]:
    game_pk, venue_id, game_hour = args
    obs = _G["obs"].get(venue_id)
    fcst_all = _G["fcst"]
    fcst = (fcst_all[(fcst_all["venue_id"] == venue_id)
                     & (fcst_all["valid_time_utc"] >= game_hour + pd.Timedelta(hours=TARGET_HOURS[0]))
                     & (fcst_all["valid_time_utc"] <= game_hour + pd.Timedelta(hours=TARGET_HOURS[-1]))]
            if len(fcst_all) else fcst_all)
    az = _G["azimuths"].get(venue_id, _G["azimuths"].get(str(venue_id), 0.0))
    T = assemble_asof_tensor(obs, fcst, game_hour, venue_id, float(az))
    return game_pk, T


def build_season(season: int, workers: int = 8) -> None:
    vmap = _read_json(STATION_MAP_KEY)
    azimuths = {int(k): v for k, v in _read_json(AZIMUTHS_KEY).items()}
    pop = load_population(season)
    if pop.empty:
        logger.info(f"{season}: no population games")
        return
    logger.info(f"{season}: {len(pop)} games, {pop['venue_id'].nunique()} venues")

    venue_ids = sorted(pop["venue_id"].unique())
    dates = sorted(pop["game_date"].dt.normalize().unique())
    obs = load_obs_for_venues(venue_ids, season, vmap)
    fcst_missing: list = []
    fcst = load_hrrr_for_dates([pd.Timestamp(d) for d in dates], missing_out=fcst_missing)
    # Before any expensive assembly: an incomplete archive yields a normal-looking
    # artifact, so this has to be a refusal rather than a warning.
    assert_fcst_dates_complete(season, dates, fcst_missing)
    soil_years = {season, season - 1}
    soil = load_soil_for_venues(venue_ids, soil_years)
    if not fcst.empty:
        parts = []
        for vid, grp in fcst.groupby("venue_id"):
            parts.append(merge_lagged_soil(grp, soil.get(int(vid))))
        fcst = pd.concat(parts, ignore_index=True)
    logger.info(f"{season}: loaded obs({sum(len(v) for v in obs.values())} rows) "
                f"fcst({len(fcst)} rows)")

    _G.update(obs=obs, fcst=fcst, azimuths=azimuths)
    jobs = list(pop[["game_pk", "venue_id", "game_hour_utc"]].itertuples(index=False, name=None))
    results: dict[int, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for game_pk, T in pool.map(_assemble_one, jobs, chunksize=16):
            results[game_pk] = T
    logger.info(f"{season}: assembled {len(results)} tensors")

    rows = []
    d_idx = list(DECISION_HOURS)
    h_idx = list(TARGET_HOURS)
    for game_pk, T in results.items():
        flat = T.reshape(len(d_idx) * len(h_idx), ASOF_CHANNELS)
        idx = pd.MultiIndex.from_product([[game_pk], d_idx, h_idx],
                                         names=["game_pk", "decision_hour", "target_hour"])
        rows.append(pd.DataFrame(flat, columns=CHANNEL_COLS, index=idx).reset_index())
    out = pd.concat(rows, ignore_index=True)
    key = f"{FS_PREFIX}/weather_asof/season={season}.parquet"
    _write_parquet(out, key)
    logger.info(f"{season}: wrote {len(out)} rows -> s3://{S3_BUCKET}/{key}")


# ── Norm stats (train games only, populated entries only) ────────────────────
def build_norm_stats() -> None:
    gm = _read_parquet(f"{FS_PREFIX}/game_meta.parquet", columns=["game_pk", "game_date"])
    gm["game_date"] = pd.to_datetime(gm["game_date"])
    train_pks = set(gm[gm["game_date"] < TRAIN_END_DATE]["game_pk"])

    resp = s3().list_objects_v2(Bucket=S3_BUCKET, Prefix=f"{FS_PREFIX}/weather_asof/season=")
    keys = [o["Key"] for o in resp.get("Contents", [])]
    sums = {"fcst": np.zeros(N_DIMS), "obs": np.zeros(N_OBS_DIMS)}
    sqs = {"fcst": np.zeros(N_DIMS), "obs": np.zeros(N_OBS_DIMS)}
    cnts = {"fcst": np.zeros(N_DIMS), "obs": np.zeros(N_OBS_DIMS)}
    for key in keys:
        df = _read_parquet(key)
        df = df[df["game_pk"].isin(train_pks)]
        if df.empty:
            continue
        arr = df[CHANNEL_COLS].to_numpy(dtype=np.float64)
        for ch, off_v, off_m, n in (("fcst", OFF_FCST, OFF_FCST_MASK, N_DIMS),
                                    ("obs", OFF_OBS, OFF_OBS_MASK, N_OBS_DIMS)):
            v = arr[:, off_v:off_v + n]
            m = arr[:, off_m:off_m + n]
            sums[ch] += (v * m).sum(axis=0)
            sqs[ch] += (v * v * m).sum(axis=0)
            cnts[ch] += m.sum(axis=0)
        logger.info(f"norm-stats: {key} accumulated")
    stats = {}
    for ch in ("fcst", "obs"):
        c = np.maximum(cnts[ch], 1.0)
        mean = sums[ch] / c
        var = np.maximum(sqs[ch] / c - mean ** 2, 0.0)
        stats[f"{ch}_mean"] = mean.tolist()
        stats[f"{ch}_std"] = np.sqrt(var).tolist()
        stats[f"{ch}_count"] = cnts[ch].tolist()
    stats["train_end_date"] = TRAIN_END_DATE
    stats["built_at"] = datetime.now(timezone.utc).isoformat()
    s3().put_object(Bucket=S3_BUCKET, Key=f"{FS_PREFIX}/weather_asof_norm.json",
                    Body=json.dumps(stats).encode())
    logger.info("norm-stats written")


# ── T3.2: per-pitch decision-hour offsets ─────────────────────────────────────
def compute_wx_hour_offsets(pit: pd.DataFrame) -> pd.DataFrame:
    """(game_pk, play_index, pitch_sequence_index, pitch_start_time,
    game_hour_utc) -> (game_pk, sequence_index, wx_hour_offset int8).

    sequence_index MUST replicate build_pitch_sequence_frame's ordering
    (sort by game_pk/play_index/pitch_sequence_index, then cumcount) or the
    offsets misalign with the pitch tokens. Untimed pitches forward-fill from
    the previous timed pitch in the same game; a game with no timed pitch at
    all falls to 0 (the pregame decision row — never a future one)."""
    pit = pit.sort_values(["game_pk", "play_index", "pitch_sequence_index"]).reset_index(drop=True)
    pit["sequence_index"] = pit.groupby("game_pk").cumcount()

    t = pd.to_datetime(pit["pitch_start_time"], utc=True, errors="coerce")
    off = np.floor((t - pit["game_hour_utc"]).dt.total_seconds() / 3600.0)
    off = off.groupby(pit["game_pk"]).ffill()
    off = off.fillna(0).clip(DECISION_HOURS[0], DECISION_HOURS[-1]).astype(np.int8)
    return pd.DataFrame({"game_pk": pit["game_pk"], "sequence_index": pit["sequence_index"],
                         "wx_hour_offset": off})


def build_pitch_offsets(season: int) -> None:
    """wx_hour_offset int8 per pitch, aligned with build_pitch_sequence_frame's
    (game_pk, play_index, pitch_sequence_index) sort + cumcount. Fallbacks:
    forward-fill within game (untimed pitches), then 0 (pregame row)."""
    resp = s3().get_paginator("list_objects_v2").paginate(
        Bucket=S3_BUCKET, Prefix=f"data/season={season}/pitches_batch_")
    keys = [o["Key"] for page in resp for o in page.get("Contents", [])
            if o["Key"].endswith(".parquet")]
    cols = ["game_pk", "play_index", "pitch_sequence_index", "pitch_start_time"]
    parts = []
    for k in keys:
        try:
            parts.append(_read_parquet(k, columns=cols))
        except Exception as exc:
            logger.warning(f"{k}: {exc}")
    pit = pd.concat(parts, ignore_index=True)

    gm = _read_parquet(f"{FS_PREFIX}/game_meta.parquet",
                       columns=["game_pk", "game_datetime_utc"])
    gm["game_hour_utc"] = pd.to_datetime(gm["game_datetime_utc"], utc=True).dt.floor("h")
    pit = pit.merge(gm[["game_pk", "game_hour_utc"]], on="game_pk", how="inner")

    share_timed = pd.to_datetime(pit["pitch_start_time"], utc=True, errors="coerce").notna().mean()
    out = compute_wx_hour_offsets(pit)
    key = f"{FS_PREFIX}/wx_hour_offset/season={season}.parquet"
    _write_parquet(out, key)
    logger.info(f"{season}: {len(out)} pitches ({share_timed:.1%} timed) -> {key}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--season", type=int, required=True)
    b.add_argument("--workers", type=int, default=8)
    sub.add_parser("norm-stats")
    p = sub.add_parser("pitch-offsets")
    p.add_argument("--season", type=int, required=True)
    args = ap.parse_args()
    if args.cmd == "build":
        build_season(args.season, args.workers)
    elif args.cmd == "norm-stats":
        build_norm_stats()
    elif args.cmd == "pitch-offsets":
        build_pitch_offsets(args.season)


if __name__ == "__main__":
    main()
