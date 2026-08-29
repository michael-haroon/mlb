"""
pregame/engineering/weather.py
-------------------------------
Weather feature engineering for the pregame pipeline.

Two modes:
  - Training: joins ERA5 reanalysis (ground truth) to game_datetime_utc
  - Inference: joins Open-Meteo forecast to scheduled first-pitch hour

Architecture (Option C — climatology + anomaly):
  - Climatological features: venue × month normals, always available, encode
    long-term venue character (altitude, typical humidity, prevailing wind)
  - Anomaly features: deviation from climatology at game time. Only reliable
    within ~2 days (forecast horizon). Beyond that, regresses to zero.
  - Ensemble uncertainty: std across 50 ECMWF members, signals forecast confidence.
    Model learns to down-weight anomaly when uncertainty is high.

Encoding for multiple model types:
  - Trees: raw continuous values (wind_toward_cf_mph, air_density_kgm3, etc.)
  - Linear: pre-computed physics (air_density integrates temp+pressure+humidity),
    anomaly from climatology (centered, directly usable as linear terms),
    interactions (wind × is_open_air)

Park azimuth calibration:
  Uses GUMBO historical weather_wind strings + ERA5 wind vectors to derive
  each park's CF bearing. "Out To CF" at game time + ERA5 wind angle at same
  hour → CF azimuth. Median across hundreds of games per venue.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── S3 config ────────────────────────────────────────────────────────────────
_S3_BUCKET = "mlb-265753586044-us-east-1-an"
_S3_REGION = "us-east-1"
_WEATHER_PREFIX = "data/weather"  # shared data (not under pregame/ or live/)

# ── GUMBO direction → park-coordinate unit vector ────────────────────────────
# Park frame: x = third-base→first-base (positive = toward first), y = home→CF
GUMBO_DIR_TO_PARK_VECTOR: dict[str, tuple[float, float]] = {
    "Out To CF":  (0.0,   1.0),
    "In From CF": (0.0,  -1.0),
    "Out To LF":  (-0.707, 0.707),
    "In From LF": (0.707, -0.707),
    "Out To RF":  (0.707,  0.707),
    "In From RF": (-0.707, -0.707),
    "L To R":     (1.0,   0.0),
    "R To L":     (-1.0,  0.0),
}

# Venues where outdoor wind is irrelevant (fixed-roof domes)
CLOSED_ROOF_VENUES: set[int] = {2518, 2530, 3289, 5150}

# Retractable roof venues (wind sometimes relevant, sometimes not)
RETRACTABLE_VENUES: set[int] = {2529, 5325, 3809, 5380, 22}

# Turf venues where soil moisture is irrelevant
TURF_VENUES: set[int] = {2518, 2530, 3289, 5150}

# Physical constants
R_DRY = 287.05      # J/(kg·K) — specific gas constant for dry air
RHO_SEA_LEVEL = 1.225  # kg/m³ — standard sea-level air density


def calibrate_park_azimuths(
    games: pd.DataFrame,
    weather_hourly: pd.DataFrame,
) -> dict[int, float]:
    """Derive each park's CF azimuth from GUMBO wind strings + ERA5 vectors.

    For each game where GUMBO reports "Out To CF" or "In From CF" (the cleanest
    signal), we know the ERA5 wind vector at that hour points along the CF axis.
    The median compass bearing across all such games gives the CF azimuth.

    Parameters
    ----------
    games : DataFrame
        Must have: game_pk, venue_id, game_datetime_utc, weather_wind
    weather_hourly : DataFrame
        ERA5 hourly with: venue_id, timestamp, wind_u_10m, wind_v_10m

    Returns
    -------
    dict[int, float]
        venue_id → CF azimuth in degrees from north (0–360)
    """
    games = games.copy()
    games["game_dt"] = pd.to_datetime(games["game_datetime_utc"], utc=True, errors="coerce")
    games = games.dropna(subset=["game_dt", "venue_id", "weather_wind"])
    games["venue_id"] = games["venue_id"].astype(int)
    games["game_hour"] = games["game_dt"].dt.floor("h")

    # Parse GUMBO direction
    def _parse_direction(s):
        if pd.isna(s):
            return None
        parts = s.split(" mph, ")
        if len(parts) != 2:
            return None
        return parts[1].strip()

    games["gumbo_dir"] = games["weather_wind"].apply(_parse_direction)

    # Only use "Out To CF" and "In From CF" — these are along the CF axis
    cf_games = games[games["gumbo_dir"].isin(["Out To CF", "In From CF"])].copy()
    cf_games["is_outward"] = cf_games["gumbo_dir"] == "Out To CF"

    # Join ERA5 at game hour
    merged = cf_games.merge(
        weather_hourly[["venue_id", "timestamp", "wind_u_10m", "wind_v_10m"]],
        left_on=["venue_id", "game_hour"],
        right_on=["venue_id", "timestamp"],
        how="inner",
    )

    # Compute ERA5 wind compass bearing (azimuth from north)
    # wind_u = eastward, wind_v = northward
    # azimuth = atan2(u, v) gives angle from north, clockwise positive
    merged["era5_bearing"] = np.degrees(
        np.arctan2(merged["wind_u_10m"], merged["wind_v_10m"])
    ) % 360

    # For "In From CF", the wind is coming FROM CF, so ERA5 bearing points
    # AWAY from CF. Flip by 180° to get the CF direction.
    merged["cf_bearing"] = np.where(
        merged["is_outward"],
        merged["era5_bearing"],
        (merged["era5_bearing"] + 180.0) % 360,
    )

    # Filter out calm winds (< 3 mph) — bearing is meaningless
    merged["era5_speed"] = np.sqrt(
        merged["wind_u_10m"] ** 2 + merged["wind_v_10m"] ** 2
    )
    merged = merged[merged["era5_speed"] >= 3.0]

    # Circular median per venue
    azimuths = {}
    for vid, group in merged.groupby("venue_id"):
        if len(group) < 5:
            continue
        bearings_rad = np.radians(group["cf_bearing"].values)
        # Circular mean (more robust than median for angles)
        mean_sin = np.mean(np.sin(bearings_rad))
        mean_cos = np.mean(np.cos(bearings_rad))
        circular_mean = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360
        # Circular dispersion (concentration)
        R = np.sqrt(mean_sin**2 + mean_cos**2)
        if R < 0.3:
            # High dispersion — unreliable (possibly retractable roof, mixed conditions)
            log.warning(f"Venue {vid}: low concentration R={R:.2f} from {len(group)} games — skipping")
            continue
        azimuths[vid] = round(circular_mean, 1)
        log.debug(f"Venue {vid}: CF azimuth = {circular_mean:.1f}° "
                  f"(n={len(group)}, R={R:.2f})")

    log.info(f"Calibrated CF azimuths for {len(azimuths)} venues")
    return azimuths


def compute_air_density(
    temperature_f: pd.Series,
    dew_point_f: pd.Series,
    surface_pressure_hpa: pd.Series,
) -> pd.Series:
    """Compute real air density from temperature, dew point, and pressure.

    Uses the equation of state for moist air:
        ρ = (P - 0.378·e) / (R_d · T)
    where e is vapor pressure derived from dew point via Magnus formula.

    Returns density in kg/m³ (sea level ≈ 1.225, Coors Field ≈ 1.05).
    """
    temp_c = (temperature_f - 32.0) * 5.0 / 9.0
    dew_c = (dew_point_f - 32.0) * 5.0 / 9.0
    temp_k = temp_c + 273.15
    pressure_pa = surface_pressure_hpa * 100.0

    # Magnus formula for vapor pressure from dew point
    e_hpa = 6.1078 * np.exp(17.27 * dew_c / (dew_c + 237.3))
    e_pa = e_hpa * 100.0

    rho = (pressure_pa - 0.378 * e_pa) / (R_DRY * temp_k)
    return rho


def rotate_wind_to_park(
    wind_u: pd.Series,
    wind_v: pd.Series,
    venue_ids: pd.Series,
    azimuths: dict[int, float],
) -> tuple[pd.Series, pd.Series]:
    """Rotate compass wind (u=east, v=north) into park-relative components.

    Returns
    -------
    wind_toward_cf : positive = tailwind (aids fly ball carry)
    wind_crossfield : positive = third-base-to-first-base direction
    """
    # Build azimuth array aligned to venue_ids
    az_rad = venue_ids.map(lambda v: np.radians(azimuths.get(v, np.nan)))

    # Dot product with CF unit vector (sin(az), cos(az))
    wind_toward_cf = wind_u * np.sin(az_rad) + wind_v * np.cos(az_rad)

    # Perpendicular component (cross product z-component)
    wind_crossfield = wind_u * np.cos(az_rad) - wind_v * np.sin(az_rad)

    # Zero out for closed-roof venues
    is_closed = venue_ids.isin(CLOSED_ROOF_VENUES)
    wind_toward_cf = wind_toward_cf.where(~is_closed, 0.0)
    wind_crossfield = wind_crossfield.where(~is_closed, 0.0)

    return wind_toward_cf, wind_crossfield


def engineer_weather_features(
    games: pd.DataFrame,
    climatology: pd.DataFrame,
    azimuths: dict[int, float],
) -> pd.DataFrame:
    """Compute all weather features from ERA5 columns already joined to games.

    Expects ERA5 columns (temperature_2m, dew_point_2m, surface_pressure,
    wind_u_10m, wind_v_10m, etc.) to already be on the games DataFrame
    (from join_era5_to_games or join_forecast_to_game).

    Produces features for all model types:
      - Trees: raw continuous (air_density, wind_toward_cf, temperature_f, ...)
      - KNN/MLP: standardized anomalies (air_density_anomaly, temperature_f_anomaly, ...)
      - Linear: physics composites + interactions (air_density_ratio, wind_toward_cf_open)

    Parameters
    ----------
    games : DataFrame
        Game frame with ERA5 columns already joined (via join_era5_to_games).
    climatology : DataFrame
        Venue × month normals (from compute_climatology).
    azimuths : dict
        venue_id → CF azimuth degrees (from calibrate_park_azimuths).
    """
    games = games.copy()

    # ── Physics-derived features (trees AND linear) ───────────────────────────

    # Air density — the dominant mechanism for ball flight
    if "temperature_2m" in games.columns and "dew_point_2m" in games.columns:
        games["air_density"] = compute_air_density(
            games["temperature_2m"], games["dew_point_2m"], games["surface_pressure"]
        )
        games["air_density_ratio"] = games["air_density"] / RHO_SEA_LEVEL

    # Park-relative wind
    if "wind_u_10m" in games.columns and "wind_v_10m" in games.columns:
        wind_cf, wind_cross = rotate_wind_to_park(
            games["wind_u_10m"], games["wind_v_10m"], games["venue_id"], azimuths
        )
        games["wind_toward_cf"] = wind_cf
        games["wind_crossfield"] = wind_cross

    if "wind_speed_10m" in games.columns:
        games["wind_speed"] = games["wind_speed_10m"]
    if "wind_gusts_10m" in games.columns:
        games["wind_gusts"] = games["wind_gusts_10m"]

    # ── Ground condition features ─────────────────────────────────────────────

    # Recent precipitation as ground wetness proxy (turf-aware)
    is_turf = games["venue_id"].isin(TURF_VENUES)
    for col in ("precip_6h", "precip_24h"):
        if col in games.columns:
            games.loc[is_turf, col] = 0.0

    # ── Pitcher/grip features ─────────────────────────────────────────────────

    if "vapour_pressure_deficit" in games.columns:
        games["vpd"] = games["vapour_pressure_deficit"]
    if "relative_humidity_2m" in games.columns:
        games["humidity"] = games["relative_humidity_2m"]
    if "wet_bulb_temperature_2m" in games.columns:
        games["wet_bulb_f"] = games["wet_bulb_temperature_2m"]

    # ── Raw values for trees ──────────────────────────────────────────────────

    if "temperature_2m" in games.columns:
        games["temperature_f"] = games["temperature_2m"]
    if "cloud_cover" not in games.columns and "cloud_cover" in games.columns:
        pass  # already present from ERA5 join
    if "visibility" not in games.columns and "visibility" in games.columns:
        pass

    # ── Climatology anomalies (for KNN/MLP/linear) ────────────────────────────

    games["_game_month"] = pd.to_datetime(
        games["game_datetime_utc"], utc=True, errors="coerce"
    ).dt.month

    games = games.merge(
        climatology,
        left_on=["venue_id", "_game_month"],
        right_on=["venue_id", "month"],
        how="left",
    )

    # Anomalies: (actual - climatological mean) / climatological std
    # These are z-scores: comparable across venues, centered, unit-variance
    anomaly_pairs = [
        ("air_density", "clim_air_density_kgm3_mean", "clim_air_density_kgm3_std"),
        ("temperature_f", "clim_temperature_2m_mean", "clim_temperature_2m_std"),
        ("humidity", "clim_relative_humidity_2m_mean", "clim_relative_humidity_2m_std"),
        ("wind_speed", "clim_wind_speed_10m_mean", "clim_wind_speed_10m_std"),
        ("surface_pressure", "clim_surface_pressure_mean", "clim_surface_pressure_std"),
    ]

    for feat, clim_mean, clim_std in anomaly_pairs:
        if feat in games.columns and clim_mean in games.columns:
            raw_anomaly = games[feat] - games[clim_mean]
            std_vals = games[clim_std].replace(0, np.nan)
            games[f"{feat}_anomaly"] = raw_anomaly / std_vals

    # ── Interactions (for linear models) ──────────────────────────────────────

    # Wind × open-air (zero in domes where wind is irrelevant)
    if "wind_toward_cf" in games.columns:
        is_open = ~games["venue_id"].isin(CLOSED_ROOF_VENUES | RETRACTABLE_VENUES)
        games["wind_toward_cf_open"] = games["wind_toward_cf"] * is_open.astype(float)

    # ── Clean up ERA5 raw and merge columns ───────────────────────────────────
    # Drop ERA5 columns that are now redundant (we keep only derived features)
    era5_raw_cols = [
        "timestamp", "game_dt", "game_hour", "_game_month", "month",
        "temperature_2m", "apparent_temperature", "wet_bulb_temperature_2m",
        "relative_humidity_2m", "dew_point_2m", "vapour_pressure_deficit",
        "precipitation", "rain", "snowfall", "snow_depth",
        "wind_speed_10m", "wind_direction_10m", "wind_u_10m", "wind_v_10m",
        "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
        "shortwave_radiation", "direct_radiation", "diffuse_radiation",
        "direct_normal_irradiance", "terrestrial_radiation",
        "surface_pressure", "weather_code", "boundary_layer_height",
        "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm",
        "et0_fao_evapotranspiration", "sunshine_duration", "is_day",
        "wind_gusts_10m",
    ]
    # Also drop climatology merge columns
    clim_cols = [c for c in games.columns if c.startswith("clim_")]
    drop = [c for c in era5_raw_cols + clim_cols if c in games.columns]
    games = games.drop(columns=drop, errors="ignore")

    return games


def compute_climatology(
    weather_hourly: pd.DataFrame,
    game_hour_range: tuple[int, int] = (17, 4),
) -> pd.DataFrame:
    """Compute venue × month climatological normals from ERA5 archive.

    Only uses evening/night hours (typical MLB game times in UTC) to avoid
    dilution from daytime hours that never coincide with games.

    Parameters
    ----------
    weather_hourly : DataFrame
        Full ERA5 archive (all venues, all years). Must have:
        venue_id, timestamp, temperature_2m, relative_humidity_2m,
        wind_speed_10m, surface_pressure, dew_point_2m
    game_hour_range : tuple
        UTC hours that overlap with typical game times. Default (17, 4)
        means 17:00–04:00 UTC (≈ 1pm–midnight ET).

    Returns
    -------
    DataFrame with venue_id, month, and clim_{var}_{mean|std} columns
    """
    df = weather_hourly.copy()
    df["month"] = df["timestamp"].dt.month
    df["hour_utc"] = df["timestamp"].dt.hour

    start_h, end_h = game_hour_range
    if start_h > end_h:
        # Wraps midnight
        game_hours = df[(df["hour_utc"] >= start_h) | (df["hour_utc"] <= end_h)]
    else:
        game_hours = df[(df["hour_utc"] >= start_h) & (df["hour_utc"] <= end_h)]

    # Compute air density for climatology
    game_hours = game_hours.copy()
    game_hours["air_density_kgm3"] = compute_air_density(
        game_hours["temperature_2m"],
        game_hours["dew_point_2m"],
        game_hours["surface_pressure"],
    )

    agg_cols = [
        "air_density_kgm3", "temperature_2m", "relative_humidity_2m",
        "wind_speed_10m", "surface_pressure",
    ]
    available = [c for c in agg_cols if c in game_hours.columns]

    clim = game_hours.groupby(["venue_id", "month"])[available].agg(["mean", "std"])
    clim.columns = [f"clim_{col}_{stat}" for col, stat in clim.columns]
    clim = clim.reset_index()

    log.info(f"Computed climatology: {len(clim)} venue-month entries, "
             f"{game_hours['venue_id'].nunique()} venues")
    return clim


def join_era5_to_games(
    games: pd.DataFrame,
    era5_hourly: pd.DataFrame,
) -> pd.DataFrame:
    """Join ERA5 weather at game hour for training.

    Also computes 6h and 24h cumulative precipitation (ground wetness proxy).
    Returns the games DataFrame with ERA5 columns added (left join — games
    without matching weather get NaN, preserving all rows).
    """
    games = games.copy()
    games["game_dt"] = pd.to_datetime(games["game_datetime_utc"], utc=True, errors="coerce")
    games["game_hour"] = games["game_dt"].dt.floor("h")

    # Coerce venue_id for games that have it
    has_venue = games["venue_id"].notna()
    games.loc[has_venue, "venue_id"] = games.loc[has_venue, "venue_id"].astype(int)

    # Pre-compute rolling precipitation per venue
    era5 = era5_hourly.sort_values(["venue_id", "timestamp"]).copy()
    era5["precip_6h"] = (
        era5.groupby("venue_id")["precipitation"]
        .transform(lambda x: x.rolling(6, min_periods=1).sum())
    )
    era5["precip_24h"] = (
        era5.groupby("venue_id")["precipitation"]
        .transform(lambda x: x.rolling(24, min_periods=1).sum())
    )

    # Drop duplicate ERA5 columns that conflict with existing game columns
    era5_cols_to_join = [c for c in era5.columns if c not in ("venue_id", "timestamp")]
    # Avoid column name collisions — prefix with nothing if clean
    era5_join = era5[["venue_id", "timestamp"] + era5_cols_to_join].copy()

    n_before = len(games)
    games = games.merge(
        era5_join,
        left_on=["venue_id", "game_hour"],
        right_on=["venue_id", "timestamp"],
        how="left",
    )

    n_matched = games["timestamp"].notna().sum()
    log.info(f"ERA5 join: {n_matched}/{n_before} games matched "
             f"({n_matched / max(n_before, 1) * 100:.1f}%)")

    return games


def join_forecast_to_game(
    venue_id: int,
    game_hour_utc: pd.Timestamp,
    forecast_df: pd.DataFrame,
    ensemble_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Extract forecast weather for a single game at inference time.

    Parameters
    ----------
    venue_id : int
    game_hour_utc : Timestamp (UTC, floored to hour)
    forecast_df : DataFrame
        From source=forecast for this venue (7-day window)
    ensemble_df : DataFrame, optional
        From source=ensemble for this venue (uncertainty)

    Returns
    -------
    dict of weather values at game hour, or empty dict if no match
    """
    row = forecast_df[
        (forecast_df["venue_id"] == venue_id) &
        (forecast_df["timestamp"] == game_hour_utc)
    ]

    if row.empty:
        # Try nearest hour within ±1h
        mask = (
            (forecast_df["venue_id"] == venue_id) &
            (abs((forecast_df["timestamp"] - game_hour_utc).dt.total_seconds()) <= 3600)
        )
        row = forecast_df[mask]
        if row.empty:
            return {}

    row = row.iloc[0]
    result = row.to_dict()

    # Compute 6h cumulative precip from forecast history
    venue_forecast = forecast_df[forecast_df["venue_id"] == venue_id].sort_values("timestamp")
    mask_6h = (
        (venue_forecast["timestamp"] <= game_hour_utc) &
        (venue_forecast["timestamp"] > game_hour_utc - pd.Timedelta(hours=6))
    )
    result["precip_6h"] = venue_forecast.loc[mask_6h, "precipitation"].sum()

    mask_24h = (
        (venue_forecast["timestamp"] <= game_hour_utc) &
        (venue_forecast["timestamp"] > game_hour_utc - pd.Timedelta(hours=24))
    )
    result["precip_24h"] = venue_forecast.loc[mask_24h, "precipitation"].sum()

    # Add ensemble uncertainty if available
    if ensemble_df is not None:
        ens_row = ensemble_df[
            (ensemble_df["venue_id"] == venue_id) &
            (ensemble_df["timestamp"] == game_hour_utc)
        ]
        if not ens_row.empty:
            ens_row = ens_row.iloc[0]
            result["temperature_ens_std"] = ens_row.get("temperature_2m_ens_std", np.nan)
            result["wind_speed_ens_std"] = ens_row.get("wind_speed_10m_ens_std", np.nan)
            result["precip_ens_std"] = ens_row.get("precipitation_ens_std", np.nan)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Data loading (S3 and local)
# ══════════════════════════════════════════════════════════════════════════════

def load_era5_from_source(
    source: str,
    venue_ids: list[int],
    seasons: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Load ERA5 hourly weather data from S3 or local path.

    Parameters
    ----------
    source : str
        S3 URI (e.g. "s3://bucket/data") or local directory path.
    venue_ids : list[int]
        Venue IDs to load (from game frame).
    seasons : list[int], optional
        Years to load. If None, loads all available.

    Returns
    -------
    DataFrame with hourly ERA5 data for all requested venues.
    """
    frames = []

    if source.startswith("s3://"):
        import boto3
        s3 = boto3.client("s3", region_name=_S3_REGION)
        paginator = s3.get_paginator("list_objects_v2")

        for vid in venue_ids:
            prefix = f"{_WEATHER_PREFIX}/source=era5/venue_id={vid}/"
            try:
                pages = paginator.paginate(Bucket=_S3_BUCKET, Prefix=prefix)
                for page in pages:
                    for obj in page.get("Contents", []):
                        key = obj["Key"]
                        if not key.endswith(".parquet"):
                            continue
                        # Filter by season if requested
                        if seasons:
                            year_str = key.split("year=")[-1].replace(".parquet", "")
                            try:
                                if int(year_str) not in seasons:
                                    continue
                            except ValueError:
                                pass
                        buf = io.BytesIO()
                        s3.download_fileobj(_S3_BUCKET, key, buf)
                        buf.seek(0)
                        frames.append(pd.read_parquet(buf))
            except Exception as e:
                log.debug(f"ERA5 venue {vid}: {e}")
    else:
        # Local path — check both direct and nested layouts
        base = Path(source)
        for weather_root in [
            base / "weather" / "source=era5",
            base / "weather_local" / "data" / "weather" / "source=era5",
        ]:
            if not weather_root.exists():
                continue
            for vid in venue_ids:
                venue_dir = weather_root / f"venue_id={vid}"
                if not venue_dir.exists():
                    continue
                for f in sorted(venue_dir.glob("*.parquet")):
                    if seasons:
                        year_str = f.stem.replace("year=", "")
                        try:
                            if int(year_str) not in seasons:
                                continue
                        except ValueError:
                            pass
                    frames.append(pd.read_parquet(f))
            if frames:
                break

    if not frames:
        log.warning(f"No ERA5 weather data found at {source} for {len(venue_ids)} venues")
        return pd.DataFrame()

    weather = pd.concat(frames, ignore_index=True)
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True)
    log.info(f"Loaded ERA5: {len(weather):,} hourly rows, "
             f"{weather['venue_id'].nunique()} venues, "
             f"{weather['timestamp'].dt.year.min()}–{weather['timestamp'].dt.year.max()}")
    return weather


# ══════════════════════════════════════════════════════════════════════════════
# Top-level orchestrator (called from build.py)
# ══════════════════════════════════════════════════════════════════════════════

def attach_weather_features(
    games: pd.DataFrame,
    source: str,
    artifacts_dir: Path,
) -> pd.DataFrame:
    """Full weather feature pipeline: load ERA5, calibrate, compute, attach.

    Caches azimuth calibration and climatology to artifacts_dir so subsequent
    builds (incremental) skip the expensive ERA5 load for those one-time computations.

    Parameters
    ----------
    games : DataFrame
        Game frame with game_pk, venue_id, game_datetime_utc, weather_wind
    source : str
        S3 URI or local path to raw data root (same as build_features source)
    artifacts_dir : Path
        Directory for cached weather artifacts (azimuths.json, climatology.parquet)

    Returns
    -------
    DataFrame with weather feature columns added
    """
    import json as _json

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    azimuth_path = artifacts_dir / "park_azimuths.json"
    climatology_path = artifacts_dir / "weather_climatology.parquet"

    # Get unique venue IDs from game frame
    venue_ids = games["venue_id"].dropna().astype(int).unique().tolist()
    seasons = sorted(games["game_date"].str[:4].dropna().astype(int).unique().tolist())

    # Load ERA5 for all venues/seasons
    log.info(f"Loading ERA5 weather for {len(venue_ids)} venues, seasons {min(seasons)}–{max(seasons)}...")
    era5 = load_era5_from_source(source, venue_ids, seasons)

    if era5.empty:
        log.warning("No ERA5 data available — weather features will be NaN")
        return games

    # ── Calibrate park azimuths (cached) ──────────────────────────────────────
    if azimuth_path.exists():
        with open(azimuth_path) as f:
            azimuths = {int(k): v for k, v in _json.load(f).items()}
        log.info(f"Loaded cached azimuths for {len(azimuths)} venues from {azimuth_path}")
    else:
        log.info("Calibrating park CF azimuths from GUMBO wind + ERA5...")
        azimuths = calibrate_park_azimuths(games, era5)
        # Convert numpy floats to Python floats for JSON serialization
        azimuths_json = {int(k): float(v) for k, v in azimuths.items()}
        with open(azimuth_path, "w") as f:
            _json.dump(azimuths_json, f, indent=2)
        log.info(f"Saved azimuths for {len(azimuths)} venues to {azimuth_path}")

    # ── Compute climatology (cached) ──────────────────────────────────────────
    if climatology_path.exists():
        climatology = pd.read_parquet(climatology_path)
        log.info(f"Loaded cached climatology: {len(climatology)} venue-month entries")
    else:
        log.info("Computing venue-month climatology from ERA5...")
        climatology = compute_climatology(era5)
        climatology.to_parquet(climatology_path, index=False)
        log.info(f"Saved climatology to {climatology_path}")

    # ── Join ERA5 at game hour ────────────────────────────────────────────────
    log.info("Joining ERA5 to game hours...")
    games = join_era5_to_games(games, era5)

    # Free ERA5 memory (can be several GB for 10+ years × 30+ venues)
    del era5

    # ── Engineer weather features ─────────────────────────────────────────────
    log.info("Computing weather features...")
    games = engineer_weather_features(games, climatology, azimuths)

    return games
