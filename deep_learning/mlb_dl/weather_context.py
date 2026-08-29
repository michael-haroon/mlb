"""Weather context computation for the GameTransformer.

Produces a [4, 22] tensor per game: 4 consecutive hourly snapshots starting at
game_hour, each with 22 physics-derived features. Used identically by training
(ERA5 source) and inference (forecast source).

Feature vector layout (22 dims per hour):
  [0-1]   air_density, air_density_ratio       — Magnus formula moist air
  [2-3]   wind_toward_cf, wind_crossfield      — park-relative wind rotation
  [4-5]   wind_speed, wind_gusts               — raw magnitude
  [6-8]   vpd, humidity, wet_bulb_f            — moisture
  [9]     temperature_f                        — direct
  [10-11] cloud_cover, visibility              — optical
  [12]    precip                               — rain/snow
  [13]    surface_pressure                     — barometric
  [14]    boundary_layer_height                — convective mixing depth
  [15]    shortwave_radiation                  — sun glare
  [16]    soil_moisture_0_to_7cm               — turf conditions (zeroed for artificial)
  [17-19] us_aqi, pm2_5, ozone                — air quality (respiratory)
  [20]    lapse_rate_1000_850                  — atmospheric stability
  [21]    wind_shear_sfc_850                   — differential carry at altitude
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEATHER_TEMPORAL_HOURS = 4
WEATHER_TOKEN_DIM = 22

WEATHER_TEMPORAL_COLUMNS = [
    "wxt_air_density",
    "wxt_air_density_ratio",
    "wxt_wind_toward_cf",
    "wxt_wind_crossfield",
    "wxt_wind_speed",
    "wxt_wind_gusts",
    "wxt_vpd",
    "wxt_humidity",
    "wxt_wet_bulb_f",
    "wxt_temperature_f",
    "wxt_cloud_cover",
    "wxt_visibility",
    "wxt_precip",
    "wxt_surface_pressure",
    "wxt_boundary_layer_height",
    "wxt_shortwave_radiation",
    "wxt_soil_moisture",
    "wxt_us_aqi",
    "wxt_pm2_5",
    "wxt_ozone",
    "wxt_lapse_rate_1000_850",
    "wxt_wind_shear_sfc_850",
]

# Physical constants
_R_DRY = 287.05  # J/(kg·K)
_RHO_SEA_LEVEL = 1.225  # kg/m³

# Fixed-roof domes (wind is irrelevant)
CLOSED_ROOF_VENUES: set[int] = {2518, 2530, 3289, 5150}

# Artificial turf venues (soil moisture irrelevant)
TURF_VENUES: set[int] = {2518, 2530, 3289, 5150}

# Fallback CF azimuth when calibration unavailable
_DEFAULT_CF_AZIMUTH = 0.0

# How many issue-dates back fetch_live_weather will accept a forecast from.
# 2 days covers a missed refresh cycle plus the UTC-rollover case for late West
# Coast games, and stays inside the 3-day horizon _fetch_forecast_ecmwf writes.
_MAX_FORECAST_STALENESS_DAYS = 2

# Rogers Centre — the only non-CONUS park, so HRRR pressure levels (dims 20-21)
# are absent in both training and inference. Consistent, not a shift.
_TORONTO_VENUE_ID = 2523

# ERA5 surface columns needed for feature computation
ERA5_SURFACE_COLUMNS = [
    "venue_id", "timestamp",
    "temperature_2m", "dew_point_2m", "relative_humidity_2m",
    "vapour_pressure_deficit", "wet_bulb_temperature_2m",
    "wind_speed_10m", "wind_u_10m", "wind_v_10m",
    "wind_gusts_10m", "surface_pressure",
    "cloud_cover", "visibility", "precipitation",
    "boundary_layer_height", "shortwave_radiation",
    "soil_moisture_0_to_7cm",
]

# ERA5 pressure level columns (subset of 116 — only what we derive from)
ERA5_PRESSURE_COLUMNS = [
    "venue_id", "timestamp",
    "temperature_1000hPa", "temperature_850hPa",
    "geopotential_height_1000hPa", "geopotential_height_850hPa",
    "wind_speed_850hPa", "wind_direction_850hPa",
]

# Air quality columns
AIR_QUALITY_COLUMNS = [
    "venue_id", "timestamp",
    "us_aqi", "pm2_5", "ozone",
]


# ---------------------------------------------------------------------------
# Physics functions
# ---------------------------------------------------------------------------


def compute_air_density(
    temp_f: np.ndarray,
    dew_point_f: np.ndarray,
    pressure_hpa: np.ndarray,
) -> np.ndarray:
    """Moist air density via ideal gas law + Magnus vapor pressure."""
    temp_c = (temp_f - 32.0) * 5.0 / 9.0
    dew_c = (dew_point_f - 32.0) * 5.0 / 9.0
    temp_k = temp_c + 273.15

    e_s = 6.1078 * np.exp((17.27 * dew_c) / (dew_c + 237.3))
    p_pa = pressure_hpa * 100.0
    e_pa = e_s * 100.0

    rho = (p_pa - 0.378 * e_pa) / (_R_DRY * temp_k)
    return rho


def rotate_wind_to_park(
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    cf_azimuth_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate meteorological wind (u=east, v=north) to park frame.

    Returns (toward_cf, crossfield) in same units as input.
    Positive toward_cf = wind blowing out to center field.
    """
    az_rad = np.radians(cf_azimuth_deg)
    toward_cf = wind_u * np.sin(az_rad) + wind_v * np.cos(az_rad)
    crossfield = wind_u * np.cos(az_rad) - wind_v * np.sin(az_rad)
    return toward_cf, crossfield


def compute_lapse_rate(
    temp_1000: np.ndarray,
    temp_850: np.ndarray,
    z_1000: np.ndarray,
    z_850: np.ndarray,
) -> np.ndarray:
    """Environmental lapse rate between 1000 and 850 hPa in °C/km.

    Standard atmosphere is ~6.5°C/km. Higher = more unstable (convective).
    Lower = stable/inverted (traps pollution, reduces mixing).
    """
    dz_km = (z_850 - z_1000) / 1000.0
    dz_km = np.where(np.abs(dz_km) < 0.01, 1.5, dz_km)
    # Temp in ERA5 pressure levels is Kelvin when fetched with default units,
    # but our fetch uses Fahrenheit — convert both to Celsius for the rate
    dt = (temp_1000 - temp_850)  # already in same units (F difference = C difference * 9/5)
    # Convert F difference to C difference
    dt_c = dt * 5.0 / 9.0
    return dt_c / dz_km


def compute_wind_shear(
    wind_speed_850: np.ndarray,
    wind_dir_850: np.ndarray,
    wind_u_sfc: np.ndarray,
    wind_v_sfc: np.ndarray,
) -> np.ndarray:
    """Vector wind shear magnitude between surface and 850 hPa (m/s).

    High shear = ball encounters different wind at apex than at surface.
    """
    dir_rad = np.radians(wind_dir_850)
    u_850 = wind_speed_850 * np.sin(dir_rad)
    v_850 = wind_speed_850 * np.cos(dir_rad)

    du = u_850 - wind_u_sfc
    dv = v_850 - wind_v_sfc
    return np.sqrt(du**2 + dv**2)


# ---------------------------------------------------------------------------
# Single-hour feature computation
# ---------------------------------------------------------------------------


def compute_hour_features(
    era5_row: dict,
    venue_id: int,
    cf_azimuth_deg: float,
    air_quality_row: Optional[dict] = None,
    pressure_row: Optional[dict] = None,
) -> np.ndarray:
    """Compute 22-dim feature vector from one hour of raw weather data.

    Parameters
    ----------
    era5_row : dict-like with ERA5 surface columns
    venue_id : stadium ID (determines roof/turf masking)
    cf_azimuth_deg : center field compass bearing for wind rotation
    air_quality_row : dict-like with us_aqi, pm2_5, ozone (optional)
    pressure_row : dict-like with pressure level columns (optional)

    Returns
    -------
    np.ndarray of shape (22,), dtype float32. NaN inputs produce 0.0.
    """
    out = np.zeros(WEATHER_TOKEN_DIM, dtype=np.float32)

    def _safe(val):
        """Convert to float, returning 0.0 for NaN/None."""
        if val is None:
            return 0.0
        f = float(val)
        return 0.0 if np.isnan(f) else f

    temp_f = _safe(era5_row.get("temperature_2m"))
    dew_f = _safe(era5_row.get("dew_point_2m"))
    pressure = _safe(era5_row.get("surface_pressure"))

    # [0-1] Air density
    if temp_f != 0.0 and pressure != 0.0:
        rho = compute_air_density(
            np.array([temp_f]), np.array([dew_f]), np.array([pressure])
        )[0]
        out[0] = rho
        out[1] = rho / _RHO_SEA_LEVEL
    else:
        out[0] = 0.0
        out[1] = 0.0

    # [2-3] Park-relative wind
    wind_u = _safe(era5_row.get("wind_u_10m"))
    wind_v = _safe(era5_row.get("wind_v_10m"))

    is_closed = venue_id in CLOSED_ROOF_VENUES
    if not is_closed and (wind_u != 0.0 or wind_v != 0.0):
        toward_cf, crossfield = rotate_wind_to_park(
            np.array([wind_u]), np.array([wind_v]), cf_azimuth_deg
        )
        out[2] = toward_cf[0]
        out[3] = crossfield[0]

    # [4-5] Wind magnitude
    if not is_closed:
        out[4] = _safe(era5_row.get("wind_speed_10m"))
        out[5] = _safe(era5_row.get("wind_gusts_10m"))

    # [6-8] Moisture
    out[6] = _safe(era5_row.get("vapour_pressure_deficit"))
    out[7] = _safe(era5_row.get("relative_humidity_2m"))
    out[8] = _safe(era5_row.get("wet_bulb_temperature_2m"))

    # [9] Temperature
    out[9] = temp_f

    # [10-11] Optical
    out[10] = _safe(era5_row.get("cloud_cover"))
    out[11] = _safe(era5_row.get("visibility"))

    # [12] Precipitation
    out[12] = _safe(era5_row.get("precipitation"))

    # [13] Pressure
    out[13] = pressure

    # [14] Boundary layer height
    out[14] = _safe(era5_row.get("boundary_layer_height"))

    # [15] Shortwave radiation
    out[15] = _safe(era5_row.get("shortwave_radiation"))

    # [16] Soil moisture (zeroed for turf venues)
    if venue_id not in TURF_VENUES:
        out[16] = _safe(era5_row.get("soil_moisture_0_to_7cm"))

    # [17-19] Air quality
    if air_quality_row is not None:
        out[17] = _safe(air_quality_row.get("us_aqi"))
        out[18] = _safe(air_quality_row.get("pm2_5"))
        out[19] = _safe(air_quality_row.get("ozone"))

    # [20-21] Pressure-level derived
    if pressure_row is not None:
        t_1000 = _safe(pressure_row.get("temperature_1000hPa"))
        t_850 = _safe(pressure_row.get("temperature_850hPa"))
        z_1000 = _safe(pressure_row.get("geopotential_height_1000hPa"))
        z_850 = _safe(pressure_row.get("geopotential_height_850hPa"))

        if t_1000 != 0.0 and t_850 != 0.0 and z_1000 != 0.0 and z_850 != 0.0:
            out[20] = compute_lapse_rate(
                np.array([t_1000]), np.array([t_850]),
                np.array([z_1000]), np.array([z_850]),
            )[0]

        ws_850 = _safe(pressure_row.get("wind_speed_850hPa"))
        wd_850 = _safe(pressure_row.get("wind_direction_850hPa"))
        if ws_850 != 0.0:
            out[21] = compute_wind_shear(
                np.array([ws_850]), np.array([wd_850]),
                np.array([wind_u]), np.array([wind_v]),
            )[0]

    return out


# ---------------------------------------------------------------------------
# Vectorized multi-hour builder (training path)
# ---------------------------------------------------------------------------


def compute_hour_features_vectorized(
    era5_df: pd.DataFrame,
    venue_ids: np.ndarray,
    cf_azimuths: np.ndarray,
    air_quality_df: Optional[pd.DataFrame] = None,
    pressure_df: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Vectorized feature computation for N rows.

    Parameters
    ----------
    era5_df : DataFrame with ERA5 surface columns, one row per (game, hour)
    venue_ids : array of venue IDs aligned with era5_df rows
    cf_azimuths : array of CF azimuth degrees aligned with era5_df rows
    air_quality_df : aligned DataFrame with AQ columns (optional)
    pressure_df : aligned DataFrame with pressure columns (optional)

    Returns
    -------
    np.ndarray of shape (N, 22), dtype float32
    """
    N = len(era5_df)
    out = np.zeros((N, WEATHER_TOKEN_DIM), dtype=np.float32)

    def _col(df, name):
        if name in df.columns:
            return df[name].to_numpy(dtype=np.float64, na_value=0.0)
        return np.zeros(N, dtype=np.float64)

    temp_f = _col(era5_df, "temperature_2m")
    dew_f = _col(era5_df, "dew_point_2m")
    pressure = _col(era5_df, "surface_pressure")

    # [0-1] Air density
    valid_density = (temp_f != 0.0) & (pressure != 0.0)
    rho = np.where(
        valid_density,
        compute_air_density(temp_f, dew_f, pressure),
        0.0,
    )
    out[:, 0] = rho
    out[:, 1] = np.where(valid_density, rho / _RHO_SEA_LEVEL, 0.0)

    # [2-5] Wind (zeroed for closed-roof venues)
    wind_u = _col(era5_df, "wind_u_10m")
    wind_v = _col(era5_df, "wind_v_10m")
    is_closed = np.isin(venue_ids, list(CLOSED_ROOF_VENUES))

    az_rad = np.radians(cf_azimuths)
    toward_cf = wind_u * np.sin(az_rad) + wind_v * np.cos(az_rad)
    crossfield = wind_u * np.cos(az_rad) - wind_v * np.sin(az_rad)

    out[:, 2] = np.where(is_closed, 0.0, toward_cf)
    out[:, 3] = np.where(is_closed, 0.0, crossfield)
    out[:, 4] = np.where(is_closed, 0.0, _col(era5_df, "wind_speed_10m"))
    out[:, 5] = np.where(is_closed, 0.0, _col(era5_df, "wind_gusts_10m"))

    # [6-8] Moisture
    out[:, 6] = _col(era5_df, "vapour_pressure_deficit")
    out[:, 7] = _col(era5_df, "relative_humidity_2m")
    out[:, 8] = _col(era5_df, "wet_bulb_temperature_2m")

    # [9] Temperature
    out[:, 9] = temp_f

    # [10-11] Optical
    out[:, 10] = _col(era5_df, "cloud_cover")
    out[:, 11] = _col(era5_df, "visibility")

    # [12] Precipitation
    out[:, 12] = _col(era5_df, "precipitation")

    # [13] Pressure
    out[:, 13] = pressure

    # [14] Boundary layer height
    out[:, 14] = _col(era5_df, "boundary_layer_height")

    # [15] Shortwave radiation
    out[:, 15] = _col(era5_df, "shortwave_radiation")

    # [16] Soil moisture (zeroed for turf)
    is_turf = np.isin(venue_ids, list(TURF_VENUES))
    out[:, 16] = np.where(is_turf, 0.0, _col(era5_df, "soil_moisture_0_to_7cm"))

    # [17-19] Air quality
    if air_quality_df is not None:
        out[:, 17] = _col(air_quality_df, "us_aqi")
        out[:, 18] = _col(air_quality_df, "pm2_5")
        out[:, 19] = _col(air_quality_df, "ozone")

    # [20-21] Pressure-level derived
    if pressure_df is not None:
        t_1000 = _col(pressure_df, "temperature_1000hPa")
        t_850 = _col(pressure_df, "temperature_850hPa")
        z_1000 = _col(pressure_df, "geopotential_height_1000hPa")
        z_850 = _col(pressure_df, "geopotential_height_850hPa")

        valid_press = (t_1000 != 0.0) & (t_850 != 0.0) & (z_1000 != 0.0) & (z_850 != 0.0)
        lapse = np.where(
            valid_press,
            compute_lapse_rate(t_1000, t_850, z_1000, z_850),
            0.0,
        )
        out[:, 20] = lapse

        ws_850 = _col(pressure_df, "wind_speed_850hPa")
        wd_850 = _col(pressure_df, "wind_direction_850hPa")
        valid_shear = ws_850 != 0.0
        shear = np.where(
            valid_shear,
            compute_wind_shear(ws_850, wd_850, wind_u, wind_v),
            0.0,
        )
        out[:, 21] = shear

    return out


# ---------------------------------------------------------------------------
# Training path: build multi-hour weather from catalog
# ---------------------------------------------------------------------------


def build_multihour_weather_frame(
    catalog,
    game_meta_df: pd.DataFrame,
    park_azimuths: dict[int, float],
    hours: int = WEATHER_TEMPORAL_HOURS,
) -> pd.DataFrame:
    """Build weather_temporal.parquet from S3 data for all games.

    For each game, extracts `hours` consecutive hourly rows from ERA5 surface,
    ERA5 pressure, and air quality. Computes 22 physics features per hour.

    Parameters
    ----------
    catalog : ParquetCatalog pointing to S3 data warehouse
    game_meta_df : DataFrame with game_pk, venue_id, game_datetime_utc
    park_azimuths : venue_id → CF azimuth degrees from calibration
    hours : number of consecutive hours (default 4)

    Returns
    -------
    DataFrame with columns: game_pk, hour_offset, + 22 feature columns.
    Shape: (num_games * hours, 24).
    """
    if game_meta_df.empty:
        cols = ["game_pk", "hour_offset"] + WEATHER_TEMPORAL_COLUMNS
        return pd.DataFrame(columns=cols)

    meta = game_meta_df[["game_pk", "venue_id", "game_datetime_utc"]].copy()
    meta["game_dt"] = pd.to_datetime(meta["game_datetime_utc"], utc=True, errors="coerce")
    meta = meta.dropna(subset=["game_dt", "venue_id"])
    meta["venue_id"] = meta["venue_id"].astype(int)
    meta["game_hour"] = meta["game_dt"].dt.floor("h")

    venue_ids = meta["venue_id"].unique().tolist()
    years = sorted(meta["game_dt"].dt.year.unique().tolist())

    log.info(
        f"Building multi-hour weather: {len(meta)} games × {hours} hours, "
        f"{len(venue_ids)} venues, years {years[0]}-{years[-1]}"
    )

    # --- Read raw data ---
    era5_surface = catalog.read_weather(
        "era5", venue_ids=venue_ids, years=years,
        columns=ERA5_SURFACE_COLUMNS,
    )
    log.info(f"  ERA5 surface: {len(era5_surface):,} rows")

    era5_pressure = catalog.read_weather(
        "era5_pressure", venue_ids=venue_ids, years=years,
        columns=ERA5_PRESSURE_COLUMNS,
    )
    log.info(f"  ERA5 pressure: {len(era5_pressure):,} rows")

    air_quality = catalog.read_weather(
        "air_quality", venue_ids=venue_ids, years=years,
        columns=AIR_QUALITY_COLUMNS,
    )
    log.info(f"  Air quality: {len(air_quality):,} rows")

    # Parse timestamps
    for df in [era5_surface, era5_pressure, air_quality]:
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df["venue_id"] = df["venue_id"].astype(int)

    # --- Expand game × hour offset ---
    expanded = []
    for h in range(hours):
        chunk = meta[["game_pk", "venue_id", "game_hour"]].copy()
        chunk["hour_offset"] = h
        chunk["target_hour"] = chunk["game_hour"] + pd.Timedelta(hours=h)
        expanded.append(chunk)

    expanded_df = pd.concat(expanded, ignore_index=True)
    log.info(f"  Expanded to {len(expanded_df):,} (game × hour) rows")

    # --- Temporal join: ERA5 surface ---
    joined = expanded_df.merge(
        era5_surface,
        left_on=["venue_id", "target_hour"],
        right_on=["venue_id", "timestamp"],
        how="left",
    )

    # --- Temporal join: pressure ---
    pressure_joined = None
    if not era5_pressure.empty:
        pressure_joined = expanded_df[["game_pk", "hour_offset", "venue_id", "target_hour"]].merge(
            era5_pressure,
            left_on=["venue_id", "target_hour"],
            right_on=["venue_id", "timestamp"],
            how="left",
        )

    # --- Temporal join: air quality ---
    aq_joined = None
    if not air_quality.empty:
        aq_joined = expanded_df[["game_pk", "hour_offset", "venue_id", "target_hour"]].merge(
            air_quality,
            left_on=["venue_id", "target_hour"],
            right_on=["venue_id", "timestamp"],
            how="left",
        )

    # --- Vectorized feature computation ---
    venue_arr = joined["venue_id"].to_numpy()
    azimuth_arr = np.array(
        [park_azimuths.get(v, _DEFAULT_CF_AZIMUTH) for v in venue_arr],
        dtype=np.float64,
    )

    features = compute_hour_features_vectorized(
        era5_df=joined,
        venue_ids=venue_arr,
        cf_azimuths=azimuth_arr,
        air_quality_df=aq_joined,
        pressure_df=pressure_joined,
    )

    # --- Assemble output ---
    result = pd.DataFrame({
        "game_pk": joined["game_pk"].values,
        "hour_offset": joined["hour_offset"].values,
    })
    for i, col_name in enumerate(WEATHER_TEMPORAL_COLUMNS):
        result[col_name] = features[:, i]

    # Deduplicate (merge can produce dupes if multiple weather rows match same hour)
    result = result.drop_duplicates(["game_pk", "hour_offset"], keep="first")
    result = result.sort_values(["game_pk", "hour_offset"]).reset_index(drop=True)

    log.info(
        f"  Output: {len(result):,} rows "
        f"({len(result) // hours:,} games × {hours} hours)"
    )
    return result


# ---------------------------------------------------------------------------
# Inference path: fetch live weather from forecast parquet
# ---------------------------------------------------------------------------


def fetch_live_weather(
    venue_id: int,
    game_hour_utc: pd.Timestamp,
    park_azimuths: dict[int, float],
    s3_bucket: str = "mlb-265753586044-us-east-1-an",
    s3_prefix: str = "data",
    hours: int = WEATHER_TEMPORAL_HOURS,
) -> np.ndarray:
    """Fetch forecast weather for a live game and compute feature tensor.

    Reads source=forecast_ecmwf — the live counterpart of the
    `ecmwf_ifs_hres_forecast` archive the tensor is trained on. Reading
    source=forecast (best_match) instead would feed the model a different NWP
    model than it was fit on: at zero forecast lead the two still disagree by
    0.17 SD on air_density and ~1.0 SD on wind_speed.

    Parameters
    ----------
    venue_id : stadium venue_id
    game_hour_utc : first pitch hour (floored to nearest hour, UTC)
    park_azimuths : venue_id → CF azimuth degrees
    s3_bucket : S3 bucket name
    s3_prefix : S3 key prefix
    hours : number of hours to extract

    Returns
    -------
    np.ndarray of shape (hours, 22), dtype float32.
    Returns zeros if forecast data is unavailable.
    """
    import boto3

    s3 = boto3.client("s3", region_name="us-east-1")

    forecast_df = _read_forecast_product(
        s3, s3_bucket, s3_prefix, "forecast_ecmwf", venue_id, game_hour_utc
    )
    if forecast_df is None:
        log.warning(
            f"ECMWF forecast unavailable for venue={venue_id} "
            f"game_hour={game_hour_utc.isoformat()} (searched back "
            f"{_MAX_FORECAST_STALENESS_DAYS} days) — returning zeros"
        )
        return np.zeros((hours, WEATHER_TOKEN_DIM), dtype=np.float32)

    # Air quality (dims 17-19) and HRRR pressure levels (dims 20-21).
    # These read the *forecast* products, not the year-partitioned archives: the
    # archives are gated by ARCHIVE_LAG_DAYS=7 and have no row for today's game,
    # which hard-zeroed 5 of 22 dims. hrrr_pressure_forecast uses models=gfs_hrrr,
    # the same model as training's hrrr_forecast_pressure.
    # Toronto has no HRRR (CONUS-only) in either training or inference.
    aq_df = _read_forecast_product(
        s3, s3_bucket, s3_prefix, "air_quality_forecast", venue_id, game_hour_utc
    )
    # Toronto has no HRRR (CONUS-only) in either training or inference. Training
    # overwrites visibility with HRRR's unconditionally, so dims 11, 20 and 21 are
    # all NaN→0 there — leaving ECMWF's ~45km visibility in would be the shift.
    pressure_df = None
    if venue_id == _TORONTO_VENUE_ID:
        zero_visibility = True
    else:
        zero_visibility = False
        pressure_df = _read_forecast_product(
            s3, s3_bucket, s3_prefix, "hrrr_pressure_forecast", venue_id, game_hour_utc
        )

    # dim 16 (soil moisture) has no live source: no Open-Meteo model serves the
    # ERA5 0-7cm band with real values (operational ecmwf_ifs returns literal
    # 0.0), and substituting GFS's 0-10cm band would be a different model *and*
    # a different depth. Fall back to persistence from the ERA5 archive instead.
    soil_moisture = _soil_moisture_persistence(
        s3, s3_bucket, s3_prefix, venue_id, game_hour_utc
    )

    # Extract hours
    cf_azimuth = park_azimuths.get(venue_id, _DEFAULT_CF_AZIMUTH)
    result = np.zeros((hours, WEATHER_TOKEN_DIM), dtype=np.float32)

    for h in range(hours):
        target = game_hour_utc + pd.Timedelta(hours=h)

        # Find nearest hour in forecast
        era5_row = _extract_hour_row(forecast_df, venue_id, target)
        # Fallback, not override: if a live model ever starts serving the 0-7cm
        # band, the fresher value must win over a up-to-7-day-old archive value.
        if soil_moisture is not None and not era5_row.get("soil_moisture_0_to_7cm"):
            era5_row["soil_moisture_0_to_7cm"] = soil_moisture

        aq_row = None
        if aq_df is not None:
            aq_row = _extract_hour_row(aq_df, venue_id, target)

        if zero_visibility:
            era5_row["visibility"] = 0.0

        press_row = None
        if pressure_df is not None:
            press_row = _extract_hour_row(pressure_df, venue_id, target)
            # dim 11 is an HRRR feature: training overwrites ECMWF's visibility
            # with hrrr_forecast's, and ECMWF's diagnostic is markedly smoother
            # (sd 4981 vs 10812 at Cleveland). Mirror that overwrite here.
            if "visibility" in press_row:
                era5_row["visibility"] = press_row["visibility"]

        result[h] = compute_hour_features(
            era5_row=era5_row,
            venue_id=venue_id,
            cf_azimuth_deg=cf_azimuth,
            air_quality_row=aq_row,
            pressure_row=press_row,
        )

    # The model has no missingness mask, so an all-zero dim is indistinguishable
    # from a genuine zero. Surface it rather than let a source outage become a
    # silent distribution shift. Dims 20-21 are legitimately zero at Toronto.
    dead = [
        WEATHER_TEMPORAL_COLUMNS[i]
        for i in range(WEATHER_TOKEN_DIM)
        if not result[:, i].any()
    ]
    if dead:
        log.warning(
            f"venue={venue_id} game_hour={game_hour_utc.isoformat()} — "
            f"weather tensor dims all-zero: {dead}"
        )

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_forecast_product(
    s3_client,
    bucket: str,
    s3_prefix: str,
    source: str,
    venue_id: int,
    game_hour_utc: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """Read a date-partitioned forecast product covering `game_hour_utc`.

    Walks back through issue dates instead of keying on the game's own UTC date.
    A 22:00 PT first pitch is 05:00 UTC the *next* day, so the game's UTC date
    names a file that won't be written until the following refresh cycle — which
    silently zeroed the weather tensor for every late West Coast game.

    Returns None if no issue date within _MAX_FORECAST_STALENESS_DAYS produces a
    file that actually spans the game hour.
    """
    for back in range(_MAX_FORECAST_STALENESS_DAYS + 1):
        issue_date = (game_hour_utc - pd.Timedelta(days=back)).strftime("%Y-%m-%d")
        key = (
            f"{s3_prefix}/weather/source={source}/"
            f"venue_id={venue_id}/date={issue_date}.parquet"
        )
        df = _read_s3_parquet(s3_client, bucket, key)
        if df is None:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        # An older issue only helps if its horizon still reaches the game hour.
        if (df["timestamp"] == game_hour_utc).any():
            if back:
                log.info(
                    f"venue={venue_id} {source}: using issue {issue_date} for "
                    f"game_hour={game_hour_utc.isoformat()} ({back}d back)"
                )
            return df
    log.warning(
        f"venue={venue_id} {source}: no issue within "
        f"{_MAX_FORECAST_STALENESS_DAYS}d spans game_hour={game_hour_utc.isoformat()}"
    )
    return None


def _soil_moisture_persistence(
    s3_client,
    bucket: str,
    s3_prefix: str,
    venue_id: int,
    game_hour_utc: pd.Timestamp,
) -> Optional[float]:
    """Most recent non-zero ERA5 0-7cm soil moisture at or before the game hour.

    Justified by persistence, not by forecast skill. Measured autocorrelation of
    ERA5 `soil_moisture_0_to_7cm` over 14 venue-years:

        lag   pearson r    R^2    RMSE      sd    RMSE/sd
         1d      0.911   0.830  0.0290  0.0739     0.393
         3d      0.772   0.597  0.0470  0.0740     0.635
         7d      0.622   0.387  0.0611  0.0741     0.825
        14d      0.494   0.244  0.0721  0.0743     0.969

    At the 7-day ARCHIVE_LAG_DAYS boundary persistence still explains 39% of
    variance, and it comes from the identical source and depth band as training.
    The alternative is not "missing" — training encodes artificial turf as exactly
    0.0 (13.9% of rows), so hard-zeroing a grass park asserts the field is a dome.
    Known failure mode: too dry after heavy rain inside the lag window.

    Returns None when no non-zero value exists, so the caller leaves whatever the
    live forecast supplied in place (0.0 for turf venues, correctly).
    """
    # Fall back to the prior year only if the current-year file is absent — the
    # archive is written per calendar year, so a March game early in a new season
    # can precede the first write of its own year's file.
    df = None
    for year in (game_hour_utc.year, game_hour_utc.year - 1):
        key = (
            f"{s3_prefix}/weather/source=era5/"
            f"venue_id={venue_id}/year={year}.parquet"
        )
        candidate = _read_s3_parquet(s3_client, bucket, key)
        if candidate is not None and "soil_moisture_0_to_7cm" in candidate.columns:
            df = candidate
            break

    if df is None:
        log.warning(
            f"venue={venue_id}: no ERA5 archive for soil-moisture persistence "
            f"at game_hour={game_hour_utc.isoformat()}"
        )
        return None

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    vals = pd.to_numeric(df["soil_moisture_0_to_7cm"], errors="coerce")
    # Zero is the turf sentinel, not a wetness reading — skipping it means turf
    # venues fall through to None and keep their 0.0.
    usable = df.loc[(ts <= game_hour_utc) & vals.notna() & (vals > 0.0)]
    if usable.empty:
        return None

    latest = ts.loc[usable.index].idxmax()
    value = float(vals.loc[latest])
    lag_days = (game_hour_utc - ts.loc[latest]).total_seconds() / 86400.0
    log.debug(
        f"venue={venue_id}: soil_moisture persistence {value:.4f} "
        f"from {ts.loc[latest].isoformat()} ({lag_days:.1f}d lag)"
    )
    return value


def _read_s3_parquet(s3_client, bucket: str, key: str) -> Optional[pd.DataFrame]:
    """Read a single parquet from S3, returning None if not found."""
    try:
        buf = io.BytesIO()
        s3_client.download_fileobj(bucket, key, buf)
        buf.seek(0)
        return pd.read_parquet(buf)
    except Exception:
        return None


def _extract_hour_row(df: pd.DataFrame, venue_id: int, target_hour: pd.Timestamp) -> dict:
    """Extract a single hour row from a weather DataFrame as a dict.

    Falls back to empty dict if no match found.
    """
    if df is None or df.empty:
        return {}

    mask = df["timestamp"] == target_hour
    if "venue_id" in df.columns:
        mask = mask & (df["venue_id"] == venue_id)

    matched = df.loc[mask]
    if matched.empty:
        return {}

    return matched.iloc[0].to_dict()
