#!/usr/bin/env python3
"""
data_curation/scripts/fetch_weather.py
---------------------------------------
Fetch hourly weather from Open-Meteo for every MLB venue and write to S3.

Modes:
  backfill  — one-time historical fetch (default 2015 to current year)
  daily     — incremental: refresh current-year archive + today's forecast + ensemble

S3 layout:
  data/weather/source=era5/venue_id={id}/year={year}.parquet                             (ERA5 surface 31km, hourly, 1940+)
  data/weather/source=era5_land/venue_id={id}/year={year}.parquet                        (ERA5-Land surface 9km, hourly, 1950+)
  data/weather/source=era5_pressure/venue_id={id}/year={year}.parquet                    (ERA5 pressure levels 19×6 vars, hourly, 1940+)
  data/weather/source=ecmwf_ifs/venue_id={id}/year={year}.parquet                        (ECMWF IFS archive full suite, 2017+)
  data/weather/source=air_quality/venue_id={id}/year={year}.parquet                      (CAMS gases + AQI, 2013+)
  data/weather/source=hrrr_forecast/venue_id={id}/year={year}.parquet                    (HRRR historical forecast surface, US, 2018+)
  data/weather/source=hrrr_forecast_pressure/venue_id={id}/year={year}.parquet           (HRRR historical forecast 44×8 pressure levels, US, 2018+)
  data/weather/source=ecmwf_ifs_hres_forecast/venue_id={id}/year={year}.parquet          (ECMWF IFS HRES historical forecast surface, global, 2017+)
  data/weather/source=ecmwf_ifs_hres_forecast_pressure/venue_id={id}/year={year}.parquet (ECMWF IFS HRES historical forecast 19×6 pressure levels, global, 2017+)
  data/weather/source=gfs_forecast/venue_id={id}/year={year}.parquet                     (GFS historical forecast surface, global, 2021+)
  data/weather/source=gfs_forecast_pressure/venue_id={id}/year={year}.parquet            (GFS historical forecast 44×8 pressure levels, global, 2021+)
  data/weather/source=marine/venue_id={id}/year={year}.parquet                           (ERA5-Ocean: SST, waves, currents, hourly, 1940+)
  data/weather/source=flood/venue_id={id}/year={year}.parquet                            (GloFAS river discharge daily→hourly, 1984+)
  data/weather/source=forecast/venue_id={id}/date={date}.parquet                         (7-day deterministic, best_match)
  data/weather/source=forecast_ecmwf/venue_id={id}/date={date}.parquet                    (3-day deterministic, models=ecmwf_ifs — DL tensor parity)
  data/weather/source=ensemble/venue_id={id}/date={date}.parquet                         (51-member ECMWF spread)
"""

import argparse
import io
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
import pandas as pd
import requests
from botocore.exceptions import ClientError
from tqdm import tqdm

# ── Storage ──────────────────────────────────────────────────────────────────
S3_BUCKET = "mlb-265753586044-us-east-1-an"
S3_PREFIX = "data"
S3_REGION = "us-east-1"

DATA_DIR = "data"
LOG_DIR  = os.path.join(DATA_DIR, "logs")

# ── Fetch config ─────────────────────────────────────────────────────────────
# Single worker: archive-api.open-meteo.com enforces a concurrent-connection
# limit per IP that is tighter than the documented 600/min. Even 4 workers
# waking together after a 429 sleep triggered another 429 (thundering herd).
# One in-flight request at a time eliminates the burst entirely.
MAX_WORKERS      = 4
RATE_LIMIT_DELAY = 3.5     # ~17 req/min — archive API throttles at ~40 req/min sustained; 3.5s eliminates 429 backoffs
ARCHIVE_LAG_DAYS = 7       # ERA5 typically available ~5-7 days after real-time
REQUEST_TIMEOUT  = 120     # seconds; pressure-level multi-year JSON can be very large

# ── Self-hosted Open-Meteo support ────────────────────────────────────────────
# Set OPENMETEO_API_HOST=http://<host>:8080 to route to a local instance.
# Rate limiting is disabled when using a local host (no API quota).
# Flood API always uses the hosted endpoint — not available in AWS open-data sync.
OPENMETEO_API_HOST = os.environ.get("OPENMETEO_API_HOST", "").rstrip("/")

# ── Historical forecast source configuration ─────────────────────────────────
# HRRR only covers CONUS; ecmwf_ifs_hres_forecast covers all venues globally.
# See TORONTO_VENUE_ID below — the id in that constant is NOT Rogers Centre.
HRRR_START_YEAR    = 2018   # HRRR historical forecast available from 2018-01-01
ECMWF_HRES_START_YEAR = 2017  # ECMWF IFS HRES historical forecast from 2017-01-01
GFS_START_YEAR     = 2021   # GFS historical forecast from 2021-03-23
MARINE_START_YEAR  = 1940   # ERA5-Ocean available from 1940
FLOOD_START_YEAR   = 1984   # GloFAS v4 reanalysis from 1984

# WRONG VALUE, KNOWINGLY LEFT IN PLACE — do not "fix" this in isolation. Verified against
# statsapi /api/v1/venues on 2026-08-30:
#   2523 = George M. Steinbrenner Field, Tampa FL  (27.980, -82.507)
#   14   = Rogers Centre, Toronto ON               (43.642, -79.389)
# So this constant routes a Tampa park to ECMWF and leaves Rogers Centre on HRRR. The
# exclusion is doubly wrong: the id is wrong, AND the HRRR CONUS grid does cover Toronto
# at 43.6N anyway, so no exclusion was ever needed.
#
# Why it stays 2523 for now: every weather artifact in the feature store was BUILT with this
# routing. Train and inference must draw each dim from the same NWP model, so flipping this
# to 14 without rebuilding those artifacts would silently create a train/serve skew — a worse
# bug than the mislabel. Change it only together with a weather_asof/weather_features rebuild.
#
# BLAST RADIUS, NOW MEASURED (2026-08-31, game_meta.parquet, 2015+ population = 31,830 games):
#   venue 2523 Steinbrenner Field  252 games (0.792%) — wrongly on ECMWF today
#   venue 14   Rogers Centre       860 games (2.702%) — correctly on HRRR today
# 2523 is indeed a real regular-season venue: 97 of those games are 2025, when the Rays played
# their home slate there. The rest are ~15/season spring training.
#
# THE FIX IS TO DELETE THIS EXCLUSION, NOT TO REPOINT IT. Both venues' coordinates sit inside
# the HRRR CONUS domain (Rogers Centre 43.642,-79.389; Steinbrenner Field 27.980,-82.507), so
# neither needs the ECMWF route. Setting the constant to 14 would move 860 correctly-served
# games onto the wrong model in order to rescue 252 — 3.4x more harm than good.
#
# And the mechanism misses every venue it was meant to catch. The only populated venues
# genuinely outside the HRRR grid are Tokyo Dome (12 games), Gocheok Sky Dome (6), London
# Stadium (6) and Hiram Bithorn (2) — 26 games, 0.082% of the population — and none of them is
# excluded here. If a non-CONUS route is wanted, it belongs on a coordinate test, not on a
# single hardcoded id. Caveat on that list: 13 of the 100 populated venues carry no lat/lon in
# game_meta and so could not be geo-tested at all.
TORONTO_VENUE_ID = 2523

# ── Pressure level variable generation ───────────────────────────────────────
def _make_pressure_level_vars(levels: list[int], base_vars: list[str]) -> str:
    return ",".join(f"{v}_{lev}hPa" for lev in levels for v in base_vars)

# ERA5 archive supports 19 pressure levels
ERA5_PRESSURE_LEVELS = [
    1000, 975, 950, 925, 900, 850, 800, 700, 600, 500,
    400, 300, 250, 200, 150, 100, 70, 50, 30,
]
ERA5_PRESSURE_BASE_VARS = [
    "temperature", "relative_humidity", "cloud_cover",
    "wind_speed", "wind_direction", "geopotential_height",
]
ERA5_PRESSURE_VARS = _make_pressure_level_vars(ERA5_PRESSURE_LEVELS, ERA5_PRESSURE_BASE_VARS)

# HRRR / GFS historical forecast supports 44 pressure levels
HRRR_PRESSURE_LEVELS = [
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725,
    700, 675, 650, 625, 600, 575, 550, 525, 500, 475, 450, 425,
    400, 375, 350, 325, 300, 275, 250, 225, 200, 175, 150, 125,
    100, 70, 50, 40, 30, 20, 15, 10,
]
HRRR_PRESSURE_BASE_VARS = [
    "temperature", "relative_humidity", "dew_point", "cloud_cover",
    "wind_speed", "wind_direction", "vertical_velocity", "geopotential_height",
]
HRRR_PRESSURE_VARS = _make_pressure_level_vars(HRRR_PRESSURE_LEVELS, HRRR_PRESSURE_BASE_VARS)

# ── Variable sets ─────────────────────────────────────────────────────────────
ERA5_VARS = ",".join([
    # Temperature
    "temperature_2m", "apparent_temperature", "wet_bulb_temperature_2m",
    # Humidity
    "relative_humidity_2m", "dew_point_2m", "vapour_pressure_deficit",
    # Precipitation
    "precipitation", "rain", "snowfall", "snow_depth",
    # Wind (10m)
    "wind_speed_10m", "wind_direction_10m",
    # Cloud
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    # Radiation
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "direct_normal_irradiance", "terrestrial_radiation",
    # Pressure
    "surface_pressure", "pressure_msl",
    # Misc atmosphere
    "weather_code", "boundary_layer_height",
    # Soil — all 4 ERA5 depths
    "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm",
    # Derived
    "et0_fao_evapotranspiration", "sunshine_duration", "is_day",
])

# ERA5-Land: 9km resolution reanalysis (vs 31km ERA5). Available 1950+.
# Provides more detailed soil and surface temperature fields.
ERA5_LAND_VARS = ",".join([
    # Temperature
    "temperature_2m", "apparent_temperature", "wet_bulb_temperature_2m",
    # Humidity
    "relative_humidity_2m", "dew_point_2m", "vapour_pressure_deficit",
    # Precipitation
    "precipitation", "rain", "snowfall", "snow_depth",
    # Wind (10m)
    "wind_speed_10m", "wind_direction_10m",
    # Cloud / radiation
    "cloud_cover",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "direct_normal_irradiance", "terrestrial_radiation",
    # Pressure
    "surface_pressure",
    # Soil — all 4 depths
    "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm",
    # Derived
    "et0_fao_evapotranspiration", "sunshine_duration", "is_day",
])

ECMWF_IFS_VARS = ",".join([
    # Temperature
    "temperature_2m", "apparent_temperature", "wet_bulb_temperature_2m",
    # Humidity
    "relative_humidity_2m", "dew_point_2m", "vapour_pressure_deficit",
    # Precipitation
    "precipitation", "rain", "snowfall", "snow_depth",
    # Wind — multi-level
    "wind_speed_10m", "wind_direction_10m",
    "wind_speed_80m", "wind_direction_80m",
    "wind_gusts_10m",
    # Cloud
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    # Visibility
    "visibility",
    # Radiation
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "direct_normal_irradiance", "terrestrial_radiation",
    # Pressure
    "surface_pressure", "pressure_msl",
    # Atmosphere
    "weather_code", "boundary_layer_height",
    "cape", "lifted_index",
    "uv_index", "uv_index_clear_sky",
    # Soil — all 4 depths
    "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm",
    # Derived
    "et0_fao_evapotranspiration", "sunshine_duration", "is_day",
])

AIR_QUALITY_VARS = ",".join([
    # Particulates
    "pm10", "pm2_5", "dust",
    # Gases
    "carbon_monoxide", "nitrogen_dioxide", "nitrogen_monoxide",
    "sulphur_dioxide", "ozone", "ammonia", "methane",
    # Optics
    "aerosol_optical_depth", "uv_index", "uv_index_clear_sky",
    # US AQI composite and per-pollutant
    "us_aqi", "us_aqi_pm2_5", "us_aqi_pm10",
    "us_aqi_nitrogen_dioxide", "us_aqi_ozone",
    "us_aqi_sulphur_dioxide", "us_aqi_carbon_monoxide",
])

# Historical forecast (HRRR, ECMWF IFS HRES, GFS) — surface variables
HISTORICAL_FORECAST_SURFACE_VARS = ",".join([
    # Temperature
    "temperature_2m", "apparent_temperature", "wet_bulb_temperature_2m",
    "temperature_80m",
    # Humidity
    "relative_humidity_2m", "dew_point_2m", "vapour_pressure_deficit",
    "total_column_integrated_water_vapour",
    # Precipitation
    "precipitation", "rain", "showers", "snowfall", "snow_depth",
    "precipitation_probability",
    # Convection / instability
    "cape", "lifted_index", "convective_inhibition",
    "freezing_level_height", "snowfall_height",
    # Wind — multi-level
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "wind_speed_80m", "wind_direction_80m",
    "wind_speed_100m", "wind_direction_100m",
    "wind_speed_120m", "wind_direction_120m",
    # Cloud
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "visibility",
    # Radiation
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "direct_normal_irradiance", "global_tilted_irradiance", "terrestrial_radiation",
    # Pressure
    "surface_pressure", "pressure_msl",
    # Atmosphere
    "weather_code", "boundary_layer_height",
    "uv_index", "uv_index_clear_sky",
    # Soil — multiple depths
    "soil_temperature_0_to_10cm", "soil_temperature_10_to_40cm",
    "soil_temperature_40_to_100cm", "soil_temperature_100_to_200cm",
    "soil_moisture_0_to_10cm", "soil_moisture_10_to_40cm",
    "soil_moisture_40_to_100cm", "soil_moisture_100_to_200cm",
    # Derived
    "et0_fao_evapotranspiration", "sunshine_duration", "is_day",
    "surface_temperature",
])

# GFS/NBM adds probability variables not available in HRRR
GFS_EXTRA_VARS = ",".join([
    "thunderstorm_probability", "rain_probability",
    "snowfall_probability", "freezing_rain_probability",
    "mass_density_8m",
])

FORECAST_VARS = ",".join([
    "temperature_2m", "apparent_temperature", "wet_bulb_temperature_2m",
    "relative_humidity_2m", "dew_point_2m", "vapour_pressure_deficit",
    "total_column_integrated_water_vapour",
    "precipitation", "rain", "snowfall", "precipitation_probability",
    "cape", "lifted_index", "convective_inhibition",
    "freezing_level_height",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "wind_speed_80m", "wind_direction_80m",
    "wind_speed_100m", "wind_direction_100m",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "direct_normal_irradiance", "visibility",
    "surface_pressure", "pressure_msl",
    "weather_code", "boundary_layer_height", "is_day",
    "uv_index", "uv_index_clear_sky",
    "thunderstorm_probability",
])

# Live counterpart to `ecmwf_ifs_hres_forecast`, which is the source the DL
# weather tensor is *trained* on. Inference must read the same NWP model:
# measured at zero forecast lead, ECMWF HRES vs best_match still disagree by
# 0.17 SD on air_density and ~1.0 SD on wind_speed (i.e. wind carries no signal
# across the source boundary), so the mismatch dominates forecast error itself.
#
# Five FORECAST_VARS entries are omitted because models=ecmwf_ifs returns them
# all-null: lifted_index, freezing_level_height, uv_index, uv_index_clear_sky,
# thunderstorm_probability.
#
# soil_moisture_0_to_7cm is present here but NOT in FORECAST_VARS — best_match
# only serves the 0_to_10cm band, which is why tensor dim 16 was hard-zero at
# inference while training had it 86% non-zero.
ECMWF_FORECAST_VARS = ",".join([
    "temperature_2m", "apparent_temperature", "wet_bulb_temperature_2m",
    "relative_humidity_2m", "dew_point_2m", "vapour_pressure_deficit",
    "total_column_integrated_water_vapour",
    "precipitation", "rain", "snowfall", "precipitation_probability",
    "cape", "convective_inhibition",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "wind_speed_80m", "wind_direction_80m",
    "wind_speed_100m", "wind_direction_100m",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "direct_normal_irradiance", "visibility",
    "surface_pressure", "pressure_msl",
    "weather_code", "boundary_layer_height", "is_day",
    "soil_moisture_0_to_7cm",
])

# Live counterparts to the two archive sources that feed tensor dims 17-21.
# Those archives are gated by ARCHIVE_LAG_DAYS=7, so at inference today's hours
# have no row and the dims silently read zero — while training had them non-zero
# ~25-39% of the time. Both have real forecast endpoints, so the dims are
# recoverable rather than lost.
#
# models=gfs_hrrr matches the training source exactly (hrrr_forecast_pressure),
# so dims 20-21 keep full model parity. ecmwf_ifs serves no pressure levels.
PRESSURE_FORECAST_VARS = ",".join([
    "temperature_1000hPa", "temperature_850hPa",
    "geopotential_height_1000hPa", "geopotential_height_850hPa",
    "wind_speed_850hPa", "wind_direction_850hPa",
    # visibility rides along because the DL weather tensor's dim 11 is an *HRRR*
    # feature: build_multihour_weather_frame overwrites ECMWF's visibility with
    # hrrr_forecast's. The two are not interchangeable — measured live at
    # Cleveland over 48h, ECMWF sd=4981 vs HRRR sd=10812, RMSE=8047 (0.744 SD);
    # ECMWF's diagnostic is much smoother than what the model was fit on.
    "visibility",
])

ENSEMBLE_VARS = ",".join([
    "temperature_2m", "apparent_temperature", "wet_bulb_temperature_2m",
    "relative_humidity_2m", "dew_point_2m",
    "precipitation", "rain", "snowfall", "snow_depth",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "wind_speed_80m", "wind_speed_100m",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "visibility",
    "surface_pressure", "pressure_msl",
    "weather_code",
    "cape", "convective_inhibition",
    "freezing_level_height",
    "uv_index",
    "soil_temperature_0_to_10cm", "soil_moisture_0_to_10cm",
    "et0_fao_evapotranspiration",
    "shortwave_radiation", "direct_radiation",
])

MARINE_VARS = ",".join([
    "wave_height", "wave_direction", "wave_period", "wave_peak_period",
    "wind_wave_height", "wind_wave_direction", "wind_wave_period", "wind_wave_peak_period",
    "swell_wave_height", "swell_wave_direction", "swell_wave_period", "swell_wave_peak_period",
    "secondary_swell_wave_height", "secondary_swell_wave_direction", "secondary_swell_wave_period",
    "sea_surface_temperature",
    "sea_level_height_msl",
    "ocean_current_velocity", "ocean_current_direction",
    "invert_barometer_height",
])

FLOOD_VARS = "river_discharge"

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("WEATHER_INGEST")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    fh = logging.FileHandler(os.path.join(LOG_DIR, "weather_ingest.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[WEATHER] %(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)

# ── Rate limiter (slot-reservation pattern from download_history.py) ──────────
_rate_lock        = threading.Lock()
_last_request_ts: float = 0.0


def _acquire_rate_slot() -> None:
    if OPENMETEO_API_HOST:
        return  # Self-hosted: no rate limiting
    # Stamp inside lock (workers get distinct slots), sleep outside
    # (lock not held during wait, other threads can proceed).
    global _last_request_ts
    with _rate_lock:
        now  = time.monotonic()
        wait = max(0.0, _last_request_ts + RATE_LIMIT_DELAY - now)
        _last_request_ts = now + wait
    if wait:
        time.sleep(wait)


def _om_url(hosted_url: str) -> str:
    """Route to self-hosted Open-Meteo when OPENMETEO_API_HOST is set.

    Flood stays on the hosted endpoint — excluded from AWS open-data sync.
    All other endpoints collapse to http://<host>/v1/<path>.
    """
    if not OPENMETEO_API_HOST or "flood-api" in hosted_url:
        return hosted_url
    # e.g. "https://archive-api.open-meteo.com/v1/archive" → "/v1/archive"
    path = "/" + hosted_url.split("/", 3)[-1]
    return f"{OPENMETEO_API_HOST}{path}"


# ── boto3 singleton ───────────────────────────────────────────────────────────
_s3_client = None


def _get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=S3_REGION)
    return _s3_client


def _s3_key_exists(key: str) -> bool:
    try:
        _get_s3().head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def _write_parquet_s3(df: pd.DataFrame, key: str) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
    buf.seek(0)
    _get_s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())


def _write_parquet_local(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)


# ── Venue loading ─────────────────────────────────────────────────────────────
def _load_venues() -> pd.DataFrame:
    """
    Read unique (venue_id, lat, lon, timezone, name) from S3 games parquet.
    Reads one file from the most recent season that has ≥20 distinct venues.
    Authoritative — no hardcoded coordinates.
    """
    COLS = ["venue_id", "venue_latitude", "venue_longitude", "venue_timezone",
            "venue_name", "venue_roof_type"]
    for year in range(datetime.now().year, 2014, -1):
        prefix = f"{S3_PREFIX}/season={year}/pitches_batch_"
        try:
            resp = _get_s3().list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, MaxKeys=50)
            keys = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".parquet")]
            if not keys:
                continue

            accumulated: pd.DataFrame = pd.DataFrame()
            for key in keys:
                buf = io.BytesIO()
                _get_s3().download_fileobj(S3_BUCKET, key, buf)
                buf.seek(0)
                df = pd.read_parquet(buf, columns=COLS)
                accumulated = (
                    pd.concat([accumulated, df])
                    .dropna(subset=["venue_latitude", "venue_longitude"])
                    .drop_duplicates("venue_id")
                )

            venues = accumulated.reset_index(drop=True)
            if len(venues) >= 20:
                logger.info(f"Loaded {len(venues)} venues from season={year}")
                return venues
        except Exception as exc:
            logger.debug(f"season={year} venue read failed: {exc}")
    raise RuntimeError(
        "Could not load venue coordinates from S3 — run GUMBO backfill first "
        "(data_curation/scripts/download_history.py)"
    )


# ── S3 key helpers ─────────────────────────────────────────────────────────────
def _archive_key(source: str, venue_id: int, year: int) -> str:
    return f"{S3_PREFIX}/weather/source={source}/venue_id={venue_id}/year={year}.parquet"


def _forecast_key(venue_id: int, today: date) -> str:
    return f"{S3_PREFIX}/weather/source=forecast/venue_id={venue_id}/date={today.isoformat()}.parquet"


def _forecast_ecmwf_key(venue_id: int, today: date) -> str:
    return f"{S3_PREFIX}/weather/source=forecast_ecmwf/venue_id={venue_id}/date={today.isoformat()}.parquet"


def _aq_forecast_key(venue_id: int, today: date) -> str:
    return f"{S3_PREFIX}/weather/source=air_quality_forecast/venue_id={venue_id}/date={today.isoformat()}.parquet"


def _pressure_forecast_key(venue_id: int, today: date) -> str:
    return f"{S3_PREFIX}/weather/source=hrrr_pressure_forecast/venue_id={venue_id}/date={today.isoformat()}.parquet"


def _ensemble_key(venue_id: int, today: date) -> str:
    return f"{S3_PREFIX}/weather/source=ensemble/venue_id={venue_id}/date={today.isoformat()}.parquet"


# ── HTTP ───────────────────────────────────────────────────────────────────────
def _get_json(url: str, params: dict) -> dict:
    """GET (or POST if URI too large) with exponential backoff on 429 / 5xx.

    Switches permanently to POST once a 414 is received — pressure level variable
    lists with 352 names exceed nginx's default 8KB URI limit.
    """
    use_post = False
    for attempt in range(7):
        _acquire_rate_slot()
        try:
            if use_post:
                resp = requests.post(url, data=params, timeout=REQUEST_TIMEOUT)
            else:
                resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 414:
                    logger.debug(f"414 URI too large, switching to POST permanently: {url}")
                    use_post = True
                    resp = requests.post(url, data=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                wait = min(60.0, 30.0 * (2 ** attempt))
                logger.warning(f"429 rate-limited — sleeping {wait:.0f}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == 6:
                raise
            time.sleep(2.0 ** attempt)
    raise RuntimeError("Exhausted retries")


# ── Response parsers ───────────────────────────────────────────────────────────
def _parse_hourly(data: dict, venue_id: int) -> pd.DataFrame:
    """Standard Open-Meteo hourly response → DataFrame with UTC timestamps."""
    hourly = data["hourly"]
    df = pd.DataFrame({
        "venue_id":  np.int64(venue_id),
        "timestamp": pd.to_datetime(hourly["time"], utc=True),
    })
    for key, values in hourly.items():
        if key == "time":
            continue
        df[key] = pd.array(values, dtype="Float32")

    # Decompose wind direction into u/v components — avoids 0°/360° discontinuity
    # and allows linear interpolation and model dot-products to work correctly.
    for speed_col, dir_col in [
        ("wind_speed_10m",  "wind_direction_10m"),
        ("wind_speed_80m",  "wind_direction_80m"),
        ("wind_speed_100m", "wind_direction_100m"),
        ("wind_speed_120m", "wind_direction_120m"),
    ]:
        if speed_col in df.columns and dir_col in df.columns:
            dir_rad = np.deg2rad(df[dir_col].astype(float))
            speed   = df[speed_col].astype(float)
            u_col   = speed_col.replace("speed", "u")
            v_col   = speed_col.replace("speed", "v")
            df[u_col] = pd.array((speed * np.sin(dir_rad)).astype("float32"), dtype="Float32")
            df[v_col] = pd.array((speed * np.cos(dir_rad)).astype("float32"), dtype="Float32")

    return df


def _parse_daily_to_hourly(data: dict, venue_id: int) -> pd.DataFrame:
    """Open-Meteo daily response → hourly DataFrame.

    GloFAS only produces daily discharge. Each day's value is replicated to all
    24 hours of that day so the flood parquet shares the same timestamp schema
    as every other weather source — enabling consistent (venue_id, timestamp) joins.
    """
    daily = data["daily"]
    dates = pd.to_datetime(daily["time"], utc=True)

    # Build one row per hour for each date
    hourly_rows = []
    for i, d in enumerate(dates):
        for h in range(24):
            row = {"venue_id": np.int64(venue_id), "timestamp": d + pd.Timedelta(hours=h)}
            for key, values in daily.items():
                if key == "time":
                    continue
                row[key] = values[i]
            hourly_rows.append(row)

    df = pd.DataFrame(hourly_rows)
    for col in df.columns:
        if col not in ("venue_id", "timestamp"):
            df[col] = pd.array(df[col].values, dtype="Float64")
    return df


_MEMBER_RE = re.compile(r"^(.+)_member\d+$")


def _parse_ensemble(data: dict, venue_id: int) -> pd.DataFrame:
    """
    ECMWF ensemble response → per-variable mean + std across members.
    The bare variable key (control run) is ignored; statistics are computed
    directly from member columns so we get both mean and spread.
    """
    hourly = data["hourly"]
    times  = pd.to_datetime(hourly["time"], utc=True)
    df     = pd.DataFrame({"venue_id": np.int64(venue_id), "timestamp": times})

    base_vars: set[str] = set()
    for key in hourly:
        m = _MEMBER_RE.match(key)
        if m:
            base_vars.add(m.group(1))

    for var in sorted(base_vars):
        member_keys = sorted(k for k in hourly if k.startswith(f"{var}_member"))
        arr = np.array([hourly[k] for k in member_keys], dtype=np.float32)
        df[f"{var}_ens_mean"] = arr.mean(axis=0)
        df[f"{var}_ens_std"]  = arr.std(axis=0)

    return df


# ── Fetch functions ───────────────────────────────────────────────────────────
def _fetch_archive(venue_id: int, lat: float, lon: float,
                   source: str, year: int,
                   start_date: Optional[date] = None,
                   end_date: Optional[date] = None) -> Optional[pd.DataFrame]:
    today   = date.today()
    start   = start_date if start_date is not None else date(year, 1, 1)
    end     = end_date   if end_date   is not None else min(date(year, 12, 31), today - timedelta(days=ARCHIVE_LAG_DAYS))

    if start > end:
        return None

    common = dict(
        latitude=lat, longitude=lon,
        start_date=start.isoformat(), end_date=end.isoformat(),
        timezone="UTC",
        wind_speed_unit="mph",
        temperature_unit="fahrenheit",
    )

    if source == "era5":
        url    = _om_url("https://archive-api.open-meteo.com/v1/archive")
        params = {**common, "hourly": ERA5_VARS}
    elif source == "era5_land":
        url    = _om_url("https://archive-api.open-meteo.com/v1/archive")
        params = {**common, "hourly": ERA5_LAND_VARS, "models": "era5_land"}
    elif source == "era5_pressure":
        url    = _om_url("https://archive-api.open-meteo.com/v1/archive")
        params = {**common, "hourly": ERA5_PRESSURE_VARS}
    elif source == "ecmwf_ifs":
        if year < 2017:
            return None
        url    = _om_url("https://archive-api.open-meteo.com/v1/archive")
        params = {**common, "hourly": ECMWF_IFS_VARS, "models": "ecmwf_ifs"}
    elif source == "air_quality":
        if year < 2013:
            return None
        url    = _om_url("https://air-quality-api.open-meteo.com/v1/air-quality")
        params = dict(
            latitude=lat, longitude=lon,
            start_date=start.isoformat(), end_date=end.isoformat(),
            hourly=AIR_QUALITY_VARS,
            timezone="UTC",
            domains="cams_global",
        )
    else:
        raise ValueError(f"Unknown archive source: {source!r}")

    data = _get_json(url, params)
    return _parse_hourly(data, venue_id)


def _fetch_historical_forecast(
    venue_id: int, lat: float, lon: float,
    model: str, year: int,
    pressure_levels: bool = False,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch from Open-Meteo Historical Forecast API — actual archived NWP runs,
    not reanalysis. Closes the ERA5/inference distribution gap for training.

    model must be one of: 'gfs_hrrr', 'ecmwf_ifs', 'gfs_global'

    start_date / end_date override the default full-year window — useful for
    targeted backfills and tests that need a narrow sample.
    """
    today = date.today()
    start = start_date if start_date is not None else date(year, 1, 1)
    end   = end_date   if end_date   is not None else min(date(year, 12, 31), today - timedelta(days=ARCHIVE_LAG_DAYS))

    if start > end:
        return None

    if model == "gfs_hrrr" and year < HRRR_START_YEAR:
        return None
    if model == "gfs_hrrr" and venue_id == TORONTO_VENUE_ID:
        return None  # HRRR is CONUS-only
    if model == "ecmwf_ifs" and year < ECMWF_HRES_START_YEAR:
        return None
    if model == "gfs_global" and year < GFS_START_YEAR:
        return None

    if pressure_levels:
        # HRRR and GFS share 44-level pressure definitions; ECMWF uses ERA5's 19 levels
        hourly_vars = ERA5_PRESSURE_VARS if model == "ecmwf_ifs" else HRRR_PRESSURE_VARS
    else:
        extra = f",{GFS_EXTRA_VARS}" if model == "gfs_global" else ""
        hourly_vars = HISTORICAL_FORECAST_SURFACE_VARS + extra

    params = dict(
        latitude=lat, longitude=lon,
        start_date=start.isoformat(), end_date=end.isoformat(),
        hourly=hourly_vars,
        models=model,
        timezone="UTC",
        wind_speed_unit="mph",
        temperature_unit="fahrenheit",
    )

    data = _get_json(
        _om_url("https://historical-forecast-api.open-meteo.com/v1/forecast"),
        params,
    )
    return _parse_hourly(data, venue_id)


def _fetch_marine(venue_id: int, lat: float, lon: float,
                  year: int,
                  start_date: Optional[date] = None,
                  end_date: Optional[date] = None) -> Optional[pd.DataFrame]:
    """ERA5-Ocean: SST, wave height/direction/period, currents, sea level. Hourly, 1940+."""
    today = date.today()
    start = start_date if start_date is not None else date(max(year, MARINE_START_YEAR), 1, 1)
    end   = end_date   if end_date   is not None else min(date(year, 12, 31), today - timedelta(days=ARCHIVE_LAG_DAYS))

    if start > end:
        return None

    params = dict(
        latitude=lat, longitude=lon,
        start_date=start.isoformat(), end_date=end.isoformat(),
        hourly=MARINE_VARS,
        models="era5_ocean",
        timezone="UTC",
    )
    data = _get_json(_om_url("https://marine-api.open-meteo.com/v1/marine"), params)
    return _parse_hourly(data, venue_id)


def _fetch_flood(venue_id: int, lat: float, lon: float,
                 year: int,
                 start_date: Optional[date] = None,
                 end_date: Optional[date] = None) -> Optional[pd.DataFrame]:
    """GloFAS v4 seamless: daily river discharge at the nearest 5km grid cell. 1984+."""
    today = date.today()
    start = start_date if start_date is not None else date(max(year, FLOOD_START_YEAR), 1, 1)
    end   = end_date   if end_date   is not None else min(date(year, 12, 31), today - timedelta(days=1))

    if start > end:
        return None

    params = dict(
        latitude=lat, longitude=lon,
        start_date=start.isoformat(), end_date=end.isoformat(),
        daily=FLOOD_VARS,
        models="seamless_v4",
    )
    data = _get_json("https://flood-api.open-meteo.com/v1/flood", params)
    return _parse_daily_to_hourly(data, venue_id)


def _fetch_forecast(venue_id: int, lat: float, lon: float) -> Optional[pd.DataFrame]:
    """7-day forecast + last 7 days of actuals via best_match model."""
    data = _get_json(_om_url("https://api.open-meteo.com/v1/forecast"), dict(
        latitude=lat, longitude=lon,
        past_days=7, forecast_days=7,
        hourly=FORECAST_VARS,
        models="best_match",
        timezone="UTC",
        wind_speed_unit="mph",
        temperature_unit="fahrenheit",
    ))
    return _parse_hourly(data, venue_id)


def _fetch_forecast_ecmwf(venue_id: int, lat: float, lon: float) -> Optional[pd.DataFrame]:
    """ECMWF IFS deterministic forecast — the DL weather tensor's inference source.

    Kept separate from _fetch_forecast (best_match) rather than replacing it:
    the classical pregame path reads source=forecast and would lose uv_index /
    thunderstorm_probability under models=ecmwf_ifs.

    forecast_days=3 is enough for the 4-hour game window plus a day of slack if a
    refresh cycle fails; past_days=1 keeps the payload small. HRES only runs at
    00/06/12/18Z, so a longer window buys nothing.
    """
    data = _get_json(_om_url("https://api.open-meteo.com/v1/forecast"), dict(
        latitude=lat, longitude=lon,
        past_days=1, forecast_days=3,
        hourly=ECMWF_FORECAST_VARS,
        models="ecmwf_ifs",
        timezone="UTC",
        wind_speed_unit="mph",
        temperature_unit="fahrenheit",
    ))
    return _parse_hourly(data, venue_id)


def _fetch_air_quality_forecast(venue_id: int, lat: float, lon: float) -> Optional[pd.DataFrame]:
    """CAMS air-quality forecast — recovers tensor dims 17-19 (us_aqi, pm2_5, ozone).

    Same CAMS global domain as the air_quality archive, so no source mismatch;
    the archive simply lags 7 days and cannot serve today's game.
    """
    data = _get_json(_om_url("https://air-quality-api.open-meteo.com/v1/air-quality"), dict(
        latitude=lat, longitude=lon,
        forecast_days=3,
        hourly=AIR_QUALITY_VARS,
        timezone="UTC",
        domains="cams_global",
    ))
    return _parse_hourly(data, venue_id)


def _fetch_pressure_forecast(venue_id: int, lat: float, lon: float) -> Optional[pd.DataFrame]:
    """HRRR pressure-level forecast — recovers dims 20-21 (lapse rate, wind shear).

    Returns None for Toronto: HRRR is CONUS-only, and training excluded the same
    venue, so dims 20-21 are legitimately zero there in both paths.
    """
    if venue_id == TORONTO_VENUE_ID:
        return None
    data = _get_json(_om_url("https://api.open-meteo.com/v1/forecast"), dict(
        latitude=lat, longitude=lon,
        past_days=1, forecast_days=3,
        hourly=PRESSURE_FORECAST_VARS,
        models="gfs_hrrr",
        timezone="UTC",
        wind_speed_unit="mph",
        temperature_unit="fahrenheit",
    ))
    return _parse_hourly(data, venue_id)


def _fetch_ensemble(venue_id: int, lat: float, lon: float) -> Optional[pd.DataFrame]:
    """51-member ECMWF ensemble → per-variable mean+std for next 7 days."""
    data = _get_json(_om_url("https://ensemble-api.open-meteo.com/v1/ensemble"), dict(
        latitude=lat, longitude=lon,
        forecast_days=7,
        hourly=ENSEMBLE_VARS,
        models="ecmwf_ifs025",
        timezone="UTC",
        wind_speed_unit="mph",
        temperature_unit="fahrenheit",
    ))
    return _parse_ensemble(data, venue_id)


# ── Forecast product registry ─────────────────────────────────────────────────
# Every date-partitioned product, refreshed both by daily mode and by the
# 6-hourly run_forecast_refresh(). Single source of truth so the two callers
# cannot drift apart — the original bug class here was a product that existed in
# one path and not the other.
#
# Together these cover all 22 dims of the DL weather tensor:
#   forecast_ecmwf          → dims 0-16 (matches training's ecmwf_ifs_hres_forecast)
#   air_quality_forecast    → dims 17-19
#   hrrr_pressure_forecast  → dims 20-21 (matches training's hrrr_forecast_pressure)
#   forecast (best_match)   → classical pregame path only
#   ensemble                → forecast uncertainty; not yet consumed by any model
FORECAST_PRODUCTS = [
    ("forecast",               _fetch_forecast,             _forecast_key),
    ("forecast_ecmwf",         _fetch_forecast_ecmwf,       _forecast_ecmwf_key),
    ("air_quality_forecast",   _fetch_air_quality_forecast, _aq_forecast_key),
    ("hrrr_pressure_forecast", _fetch_pressure_forecast,    _pressure_forecast_key),
    ("ensemble",               _fetch_ensemble,             _ensemble_key),
]


# ── Backfill source registry ──────────────────────────────────────────────────
# Each entry: (source_key, fetch_fn_or_None, start_year, is_daily)
# fetch_fn receives (venue_id, lat, lon, source, year) for archive sources,
# or has its own signature — handled in _do_backfill_job below.

BACKFILL_SOURCES = [
    # ERA5 reanalysis surface + pressure levels
    ("era5",          2015, False),
    ("era5_pressure", 2015, False),
    # ERA5-Land: 9km resolution reanalysis — better soil/surface fields than 31km ERA5
    ("era5_land",     1950, False),
    # ECMWF IFS archive (full surface suite, 2017+)
    ("ecmwf_ifs",     2017, False),
    # Air quality (CAMS global)
    ("air_quality",   2013, False),
    # Historical forecast — closes training/inference distribution gap
    ("hrrr_forecast",                    HRRR_START_YEAR,      False),
    ("hrrr_forecast_pressure",           HRRR_START_YEAR,      False),
    ("ecmwf_ifs_hres_forecast",          ECMWF_HRES_START_YEAR, False),
    ("ecmwf_ifs_hres_forecast_pressure", ECMWF_HRES_START_YEAR, False),
    ("gfs_forecast",                     GFS_START_YEAR,        False),
    ("gfs_forecast_pressure",            GFS_START_YEAR,        False),
    # Marine (ERA5-Ocean)
    ("marine",        MARINE_START_YEAR, False),
    # Flood (GloFAS daily)
    ("flood",         FLOOD_START_YEAR,  True),
]


def _dispatch_backfill(vid: int, lat: float, lon: float,
                        source: str, year: int) -> Optional[pd.DataFrame]:
    """Route source key to the correct fetch function."""
    if source in ("era5", "era5_land", "era5_pressure", "ecmwf_ifs", "air_quality"):
        return _fetch_archive(vid, lat, lon, source, year)
    elif source == "hrrr_forecast":
        return _fetch_historical_forecast(vid, lat, lon, "gfs_hrrr", year, pressure_levels=False)
    elif source == "hrrr_forecast_pressure":
        return _fetch_historical_forecast(vid, lat, lon, "gfs_hrrr", year, pressure_levels=True)
    elif source == "ecmwf_ifs_hres_forecast":
        return _fetch_historical_forecast(vid, lat, lon, "ecmwf_ifs", year, pressure_levels=False)
    elif source == "ecmwf_ifs_hres_forecast_pressure":
        return _fetch_historical_forecast(vid, lat, lon, "ecmwf_ifs", year, pressure_levels=True)
    elif source == "gfs_forecast":
        return _fetch_historical_forecast(vid, lat, lon, "gfs_global", year, pressure_levels=False)
    elif source == "gfs_forecast_pressure":
        return _fetch_historical_forecast(vid, lat, lon, "gfs_global", year, pressure_levels=True)
    elif source == "marine":
        return _fetch_marine(vid, lat, lon, year)
    elif source == "flood":
        return _fetch_flood(vid, lat, lon, year)
    else:
        raise ValueError(f"Unknown source: {source!r}")


# ── Mode runners ───────────────────────────────────────────────────────────────
def _run_backfill(venues: pd.DataFrame, start_year: int, end_year: int,
                  local_dir: Optional[Path], force: bool = False) -> None:
    today        = date.today()
    current_year = today.year

    jobs = []
    for _, v in venues.iterrows():
        vid, lat, lon = int(v["venue_id"]), float(v["venue_latitude"]), float(v["venue_longitude"])
        for source, source_start_year, _ in BACKFILL_SOURCES:
            effective_start = max(start_year, source_start_year)
            for year in range(effective_start, end_year + 1):
                jobs.append((vid, lat, lon, source, year))

    logger.info(f"Backfill: {len(jobs)} jobs | {len(venues)} venues | "
                f"{start_year}–{end_year} | {MAX_WORKERS} workers")

    written = skipped = errors = 0

    def _do(args):
        vid, lat, lon, source, year = args
        key = _archive_key(source, vid, year)

        if not force and year != current_year and local_dir is None and _s3_key_exists(key):
            logger.debug(f"Skip existing: {key}")
            return "skip"

        try:
            df = _dispatch_backfill(vid, lat, lon, source, year)
            if df is None or df.empty:
                return "skip"
            if local_dir is not None:
                _write_parquet_local(df, local_dir / key)
            else:
                _write_parquet_s3(df, key)
            logger.debug(f"Written {len(df)} rows: {key}")
            return "write"
        except Exception as exc:
            logger.error(f"Job failed ({source}, venue={vid}, year={year}): {exc}")
            return "error"

    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="WeatherWorker") as ex:
        futures = {ex.submit(_do, j): j for j in jobs}
        for fut in tqdm(as_completed(futures), total=len(jobs), desc="Backfill", unit="job"):
            result = fut.result()
            if result == "write":
                written += 1
            elif result == "skip":
                skipped += 1
            else:
                errors += 1

    logger.info(f"Backfill complete — written={written} skipped={skipped} errors={errors}")


def _run_daily(venues: pd.DataFrame, venue_filter: Optional[set],
               local_dir: Optional[Path]) -> None:
    """
    1. Refresh this year's archive files for all sources.
    2. Write today's deterministic 7-day forecast per venue.
    3. Write today's ECMWF ensemble spread per venue.
    """
    today        = date.today()
    current_year = today.year

    if venue_filter:
        venues = venues[venues["venue_id"].isin(venue_filter)].copy()
        logger.info(f"Filtered to {len(venues)} venues: {sorted(venue_filter)}")

    logger.info(f"Daily update: {len(venues)} venues | {today.isoformat()}")
    written = errors = 0

    def _write(df, key):
        if local_dir is not None:
            _write_parquet_local(df, local_dir / key)
        else:
            _write_parquet_s3(df, key)

    for _, v in venues.iterrows():
        vid, lat, lon = int(v["venue_id"]), float(v["venue_latitude"]), float(v["venue_longitude"])
        name = v.get("venue_name", vid)

        # 1. Refresh all archive sources for current year
        for source, source_start_year, is_daily in BACKFILL_SOURCES:
            if current_year < source_start_year:
                continue
            try:
                df = _dispatch_backfill(vid, lat, lon, source, current_year)
                if df is not None and not df.empty:
                    _write(df, _archive_key(source, vid, current_year))
                    logger.info(f"[{source}] {name}: {len(df)} rows")
                    written += 1
            except Exception as exc:
                logger.error(f"[{source}] {name} failed: {exc}")
                errors += 1

        # 2. All forecast products (see FORECAST_PRODUCTS)
        for label, fn, key_fn in FORECAST_PRODUCTS:
            try:
                df = fn(vid, lat, lon)
                if df is not None and not df.empty:
                    _write(df, key_fn(vid, today))
                    logger.info(f"[{label}] {name}: {len(df)} rows")
                    written += 1
            except Exception as exc:
                logger.error(f"[{label}] {name} failed: {exc}")
                errors += 1

    logger.info(f"Daily update complete — written={written} errors={errors}")


# ── Importable daily runner (mirrors daily_enrichment.run_daily_enrichment) ───
def run_daily_weather(local: bool = False) -> None:
    """Refresh current-year archive + today's forecast + ensemble for all venues.

    Designed to be called from live_daemon._daily_enrichment once per day.
    Failures are logged but never propagate — daemon stays alive regardless.
    """
    local_dir = Path("data/weather_local") if local else None
    venues = _load_venues()
    _run_daily(venues, venue_filter=None, local_dir=local_dir)


def run_forecast_refresh(local: bool = False) -> None:
    """Re-pull only the date-partitioned forecast products — no archive sources.

    Called on a ~6h cadence so inference reads a recent HRES run instead of one
    issued up to 24h earlier. Deliberately excludes _dispatch_backfill: refreshing
    all 13 archive sources × 56 venues is hundreds of multi-MB requests and would
    exhaust the Open-Meteo rate limit within one cycle.

    Cadence rationale: ECMWF HRES initialises at 00/06/12/18Z, so 6h is the
    shortest interval that can surface new data. Sub-hourly refresh returns the
    identical run. Measured lead-time sensitivity is mild anyway — air_density
    RMSE/SD rises only 0.169 → 0.201 going from 0h to 72h lead — so this closes a
    real but secondary gap; the source switch to ecmwf_ifs is the larger fix.
    """
    local_dir = Path("data/weather_local") if local else None
    today = date.today()
    venues = _load_venues()
    logger.info(f"Forecast refresh: {len(venues)} venues | {today.isoformat()}")

    written = errors = 0
    for _, v in venues.iterrows():
        vid, lat, lon = int(v["venue_id"]), float(v["venue_latitude"]), float(v["venue_longitude"])
        name = v.get("venue_name", vid)
        for label, fn, key_fn in FORECAST_PRODUCTS:
            try:
                df = fn(vid, lat, lon)
                if df is not None and not df.empty:
                    key = key_fn(vid, today)
                    if local_dir is not None:
                        _write_parquet_local(df, local_dir / key)
                    else:
                        _write_parquet_s3(df, key)
                    logger.debug(f"[{label}] {name}: {len(df)} rows → {key}")
                    written += 1
            except Exception as exc:
                logger.error(f"[{label}] {name} failed: {exc}")
                errors += 1

    logger.info(f"Forecast refresh complete — written={written} errors={errors}")


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch hourly weather from Open-Meteo for all MLB venues → S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["backfill", "daily"], required=True,
                        help="backfill: full historical fetch | daily: incremental update")
    parser.add_argument("--start-year", type=int, default=2015,
                        help="Earliest season to fetch (backfill only)")
    parser.add_argument("--end-year", type=int, default=None,
                        help="Latest season to fetch, inclusive (default: current year)")
    parser.add_argument("--sources", type=str, default=None,
                        help="Comma-separated source keys to restrict (e.g. hrrr_forecast,marine). "
                             "Defaults to all sources.")
    parser.add_argument("--venues", type=str, default=None,
                        help="Comma-separated venue_ids to restrict (daily mode only)")
    parser.add_argument("--partition", type=str, default=None,
                        help="INDEX/TOTAL — e.g. '0/3' takes venues[0::3] (interleaved). "
                             "Backfill mode only. Lets N parallel instances share the work "
                             "without overlap while each uses its own IP's rate limit quota.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing S3 files (backfill only). "
                             "Use with --sources to re-fetch specific sources with expanded variable sets.")
    parser.add_argument("--local", action="store_true",
                        help="Write to data/weather_local/ instead of S3 (for testing)")
    args = parser.parse_args()

    end_year  = args.end_year or datetime.now().year
    local_dir = Path("data/weather_local") if args.local else None

    logger.info(f"Weather ingest starting | mode={args.mode} local={args.local}")

    venues = _load_venues()
    logger.info(f"Venue list: {len(venues)} unique stadiums")

    if args.mode == "backfill":
        if args.partition:
            idx, total = (int(x) for x in args.partition.split("/"))
            venues = venues.iloc[idx::total].reset_index(drop=True)
            logger.info(f"Partition {idx}/{total}: {len(venues)} venues")

        # Optionally restrict to a subset of sources (useful for targeted backfills)
        global BACKFILL_SOURCES
        if args.sources:
            requested = set(args.sources.split(","))
            BACKFILL_SOURCES = [(s, y, d) for s, y, d in BACKFILL_SOURCES if s in requested]
            logger.info(f"Restricted to sources: {[s for s,_,_ in BACKFILL_SOURCES]}")

        _run_backfill(venues, args.start_year, end_year, local_dir, force=args.force)
    else:
        venue_filter = (
            {int(x) for x in args.venues.split(",")} if args.venues else None
        )
        _run_daily(venues, venue_filter, local_dir)


if __name__ == "__main__":
    main()
