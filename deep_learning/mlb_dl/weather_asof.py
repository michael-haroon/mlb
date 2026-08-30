"""
As-of weather tensor assembly — the SINGLE shared implementation for training
builder and live inference (parity by construction).

The living-window tensor wx_asof[d, h, C] = [7, 7, 99]:
  d ∈ 0..6   decision hours after floor(game_datetime_utc, 1h)
  h ∈ -1..5  target hours relative to the same anchor
  C = 99     [fcst_z*fcst_mask(22) | fcst_mask(22) | obs_z*obs_mask(27) |
              obs_mask(27) | lead_norm(1)]

The obs vector is the 22-dim physics layout plus 5 obs-only dims derived from
METAR fields no forecast carries (D2b, user request 2026-08-30): thunder,
precip-intensity ordinal, frozen precip, obstruction (fog/haze/smoke), and
hour-peak wind gust. An empty wxcodes on a present report is a REAL "no
significant weather" observation, so the extras share the report-present mask.

Rules (each unit-tested in tests/test_weather_asof.py):
  - A source row is usable at decision d iff available_time_utc <= anchor + d.
  - Hour h is OBSERVED iff h < d (fully elapsed); hour -1 is observed from d=0,
    so the pregame row always carries a real current observation.
  - At observed hours the freshest pre-decision forecast stays populated —
    the model can read tonight's forecast error so far.
  - Standardize THEN mask: z-score with per-dim stats over populated train
    entries, then multiply by mask, so missing = 0 = the mean (raw-unit zeros
    like 0 hPa would be violent outliers).

Raw inputs arrive in their source units (METAR: °F/kt/inHg/mi/in from
fetch_asos_obs; HRRR GRIB: K/(m/s)/Pa/m/mm from fetch_nwp_asissued) and are
converted HERE into the era5-schema (°F/mph/hPa/mm/m/%) that
weather_context.compute_hour_features_vectorized consumes — the physics layer
is reused verbatim, including its empirically-anchored wind convention
(fetch_weather.py:568: u = speed*sin(from_dir), v = speed*cos(from_dir), the
NEGATION of true velocity components, validated against Gumbo wind labels).
"""

from __future__ import annotations

import io
import json
import logging
from typing import Optional

import numpy as np
import pandas as pd

from .weather_context import (
    WEATHER_TOKEN_DIM,
    compute_hour_features_vectorized,
)

log = logging.getLogger("mlb_dl.weather_asof")

# ── Window geometry ───────────────────────────────────────────────────────────
TARGET_HOURS = list(range(-1, 6))          # h ∈ -1..5
DECISION_HOURS = list(range(0, 7))         # d ∈ 0..6
N_TARGET_HOURS = len(TARGET_HOURS)         # 7
N_DECISIONS = len(DECISION_HOURS)          # 7

N_DIMS = WEATHER_TOKEN_DIM                 # 22 (fcst vector; obs dims 0-21)

# Obs-only extras (D2b): appended after the 22 physics dims in the obs vector.
OBS_EXTRA_NAMES = ["wx_thunder", "wx_precip_intensity", "wx_frozen_precip",
                   "wx_obstruction", "wx_peak_gust"]
N_OBS_DIMS = N_DIMS + len(OBS_EXTRA_NAMES)  # 27

# Channel layout offsets — single source of truth for builder, dataset, model.
OFF_FCST = 0
OFF_FCST_MASK = OFF_FCST + N_DIMS          # 22
OFF_OBS = OFF_FCST_MASK + N_DIMS           # 44
OFF_OBS_MASK = OFF_OBS + N_OBS_DIMS        # 71
OFF_LEAD = OFF_OBS_MASK + N_OBS_DIMS       # 98
ASOF_CHANNELS = OFF_LEAD + 1               # 99

# Dims 0-13 are constructible from a METAR report (density, park wind, gust,
# moisture, temp, cloud, visibility, precip, pressure). 14-21 (BLH, radiation,
# soil, AQI, lapse, shear) have no station observation — obs_mask is 0 forever.
OBS_OBSERVABLE_DIMS = list(range(14))

# v1 decision (2026-08-30): AQI dims 17-19 stay PERMANENTLY masked in both
# channels — CAMS coverage starts 2022-07 (measured 82% of population missing),
# so an honest mask would flip at a calendar date, handing the model a free era
# regressor; AQI's physical pathway for baseball (wildfire smoke) is already
# carried by visibility (dims 10-11, both channels) and wxcodes HZ/FU (obs).
# Builders must simply not pass air_quality_by_hour. Revisit when CAMS spans
# the full population.

# lead_norm = log1p(lead_hours) / log1p(MAX_LEAD): 18 h is the hourly-cycle
# HRRR horizon, the largest lead the extraction plans.
MAX_LEAD_HOURS = 18.0

# ── Unit conversions ──────────────────────────────────────────────────────────
KT_TO_MPH = 1.15078
MS_TO_MPH = 2.23694
MI_TO_M = 1609.34
IN_TO_MM = 25.4
INHG_TO_HPA = 33.8639

# METAR sky cover -> % (okta midpoints: FEW=1-2, SCT=3-4, BKN=5-7, OVC=8;
# WMO No.306 code table 2700; VV = sky obscured, treated as overcast)
SKY_COVER_PCT = {
    "CLR": 0.0, "SKC": 0.0, "NCD": 0.0, "NSC": 0.0, "CAVOK": 0.0,
    "FEW": 18.75, "SCT": 43.75, "BKN": 75.0, "OVC": 100.0, "VV": 100.0,
}


def _f_from_k(k):
    return (np.asarray(k, dtype=float) - 273.15) * 9.0 / 5.0 + 32.0


def _c_from_f(f):
    return (np.asarray(f, dtype=float) - 32.0) * 5.0 / 9.0


def _magnus_es_kpa(temp_c):
    """Saturation vapor pressure (kPa), Magnus form (Alduchov & Eskridge 1996)."""
    t = np.asarray(temp_c, dtype=float)
    return 0.61094 * np.exp(17.625 * t / (t + 243.04))


def relative_humidity_pct(temp_f, dew_f):
    """RH from T/Td via Magnus — used wherever the source has no RH field
    (2015-era HRRR lacks RH:2m; deriving it uniformly avoids a mid-era mask flip)."""
    rh = 100.0 * _magnus_es_kpa(_c_from_f(dew_f)) / _magnus_es_kpa(_c_from_f(temp_f))
    return np.clip(rh, 0.0, 100.0)


def vapour_pressure_deficit_kpa(temp_f, dew_f):
    return np.maximum(
        _magnus_es_kpa(_c_from_f(temp_f)) - _magnus_es_kpa(_c_from_f(dew_f)), 0.0
    )


def wet_bulb_f(temp_f, rh_pct):
    """Stull (2011, J. Appl. Meteor. Climatol. 50:2267) wet-bulb approximation,
    valid RH 5-99%, T -20..50°C — covers playable baseball weather."""
    t = _c_from_f(temp_f)
    rh = np.asarray(rh_pct, dtype=float)
    tw = (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
          + np.arctan(t + rh) - np.arctan(rh - 1.676331)
          + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
          - 4.686035)
    return tw * 9.0 / 5.0 + 32.0


def station_pressure_hpa(alti_inhg, station_elev_m):
    """True station pressure from the altimeter setting (ICAO standard
    atmosphere inversion). At sea level this equals the altimeter itself."""
    alti_hpa = np.asarray(alti_inhg, dtype=float) * INHG_TO_HPA
    elev = np.asarray(station_elev_m, dtype=float)
    return alti_hpa * (1.0 - 2.25577e-5 * elev) ** 5.25588


def _from_dir_uv_mph(speed_mph, from_dir_deg):
    """The repo's wind convention (fetch_weather.py:568): u = s*sin(θ_from),
    v = s*cos(θ_from). Nonstandard (points INTO the wind) but the park-azimuth
    rotation was validated against Gumbo labels with exactly this convention."""
    rad = np.radians(np.asarray(from_dir_deg, dtype=float))
    s = np.asarray(speed_mph, dtype=float)
    return s * np.sin(rad), s * np.cos(rad)


# ── wxcodes-derived obs extras (D2b) ─────────────────────────────────────────
# WMO 4678 present-weather groups. Precip codes cover liquid+frozen+unknown;
# intensity comes from the -/+ prefix ('' = moderate). Obstruction groups are
# the visibility-reducers (incl. FU smoke — the wildfire/AQI proxy).
_PRECIP_CODES = ("RA", "DZ", "SN", "SG", "IC", "PL", "GR", "GS", "UP")
_FROZEN_CODES = ("SN", "SG", "IC", "PL", "GR", "GS")
_OBSTRUCTION_CODES = ("FG", "BR", "HZ", "FU", "DU", "SA", "PO", "VA")


def wx_extra_features(wxcodes, peak_gust_kt, fallback_gust_mph) -> np.ndarray:
    """(wxcodes string, PK WND gust kt, snapshot gust mph) -> the 5 extras.

    A missing PK WND remark means "no notable peak" (ASOS only encodes peaks
    above the snapshot), so the snapshot gust is the honest fallback — same
    absence semantics as the METAR gust group itself.
    """
    s = "" if wxcodes is None or (isinstance(wxcodes, float) and np.isnan(wxcodes)) else str(wxcodes)
    thunder = 1.0 if "TS" in s else 0.0
    intensity = 0.0
    if any(c in s for c in _PRECIP_CODES):
        intensity = 3.0 if "+" in s else (1.0 if "-" in s else 2.0)
    frozen = 1.0 if any(c in s for c in _FROZEN_CODES) else 0.0
    obstruction = 1.0 if any(c in s for c in _OBSTRUCTION_CODES) else 0.0
    if peak_gust_kt is not None and not pd.isna(peak_gust_kt):
        peak = float(peak_gust_kt) * KT_TO_MPH
    elif fallback_gust_mph is not None and not pd.isna(fallback_gust_mph):
        peak = float(fallback_gust_mph)
    else:
        # Calm/'M'-wind METAR: both gust group and speed absent. `x or 0.0`
        # would pass NaN through (NaN is truthy) — found as 2,518 NaNs in the
        # real 2015 artifact.
        peak = 0.0
    return np.array([thunder, intensity, frozen, obstruction, peak], dtype=np.float32)


# ── Source-schema -> era5-schema conversion ──────────────────────────────────
def metar_to_era5(df: pd.DataFrame, station_elev_m: float) -> pd.DataFrame:
    """fetch_asos_obs rows (raw METAR units) -> era5-schema columns."""
    out = pd.DataFrame(index=df.index)
    out["valid_utc"] = df["valid_utc"]
    out["available_time_utc"] = df["available_time_utc"]
    out["temperature_2m"] = df["tmpf"].astype(float)
    out["dew_point_2m"] = df["dwpf"].astype(float)
    rh = df["relh"].astype(float) if "relh" in df else pd.Series(np.nan, index=df.index)
    out["relative_humidity_2m"] = rh.where(
        rh.notna(), relative_humidity_pct(out["temperature_2m"], out["dew_point_2m"]))
    out["vapour_pressure_deficit"] = vapour_pressure_deficit_kpa(
        out["temperature_2m"], out["dew_point_2m"])
    out["wet_bulb_temperature_2m"] = wet_bulb_f(
        out["temperature_2m"], out["relative_humidity_2m"])
    out["surface_pressure"] = station_pressure_hpa(df["alti"].astype(float), station_elev_m)

    speed_mph = df["sknt"].astype(float) * KT_TO_MPH
    u, v = _from_dir_uv_mph(speed_mph, df["drct"].astype(float))
    out["wind_speed_10m"] = speed_mph
    out["wind_u_10m"], out["wind_v_10m"] = u, v
    # A METAR omits the gust group when gusting is < 10 kt over the mean: an
    # absent gust is a REAL "no significant gust" report, not missing data.
    out["wind_gusts_10m"] = (df["gust"].astype(float) * KT_TO_MPH).fillna(speed_mph)

    covers = pd.concat(
        [df[c].map(SKY_COVER_PCT) for c in ("skyc1", "skyc2", "skyc3", "skyc4") if c in df],
        axis=1)
    out["cloud_cover"] = covers.max(axis=1)
    out["visibility"] = df["vsby"].astype(float) * MI_TO_M
    out["precipitation"] = df["p01i"].astype(float) * IN_TO_MM
    # Raw fields the D2b extras derive from — carried through, not converted.
    out["wxcodes"] = df["wxcodes"] if "wxcodes" in df else None
    out["peak_wind_gust_kt"] = (pd.to_numeric(df["peak_wind_gust"], errors="coerce")
                                if "peak_wind_gust" in df else np.nan)
    return out


def hrrr_to_era5(df: pd.DataFrame) -> pd.DataFrame:
    """fetch_nwp_asissued rows (raw GRIB SI units) -> era5-schema columns,
    including the pressure-level columns dims 20-21 need."""
    out = pd.DataFrame(index=df.index)
    for c in ("issue_time_utc", "available_time_utc", "valid_time_utc", "lead_hours"):
        out[c] = df[c]
    out["temperature_2m"] = _f_from_k(df["t2m_k"])
    out["dew_point_2m"] = _f_from_k(df["d2m_k"])
    out["relative_humidity_2m"] = relative_humidity_pct(
        out["temperature_2m"], out["dew_point_2m"])
    out["vapour_pressure_deficit"] = vapour_pressure_deficit_kpa(
        out["temperature_2m"], out["dew_point_2m"])
    out["wet_bulb_temperature_2m"] = wet_bulb_f(
        out["temperature_2m"], out["relative_humidity_2m"])
    out["surface_pressure"] = df["sp_pa"].astype(float) / 100.0

    # GRIB u/v are true velocity components (toward east/north); the repo
    # convention is their negation in mph (see module docstring).
    u_true, v_true = df["u10_ms"].astype(float), df["v10_ms"].astype(float)
    out["wind_u_10m"] = -u_true * MS_TO_MPH
    out["wind_v_10m"] = -v_true * MS_TO_MPH
    out["wind_speed_10m"] = np.hypot(u_true, v_true) * MS_TO_MPH
    out["wind_gusts_10m"] = df["gust_ms"].astype(float) * MS_TO_MPH

    out["cloud_cover"] = df["tcc_pct"].astype(float)
    out["visibility"] = df["vis_m"].astype(float)
    out["precipitation"] = df["apcp_mm"].astype(float)
    out["boundary_layer_height"] = df["hpbl_m"].astype(float)
    out["shortwave_radiation"] = df["dswrf_wm2"].astype(float)

    out["temperature_850hPa"] = _f_from_k(df["t850_k"])
    out["temperature_1000hPa"] = _f_from_k(df["t1000_k"])
    out["geopotential_height_850hPa"] = df["z850_m"].astype(float)
    out["geopotential_height_1000hPa"] = df["z1000_m"].astype(float)
    u850, v850 = df["u850_ms"].astype(float), df["v850_ms"].astype(float)
    out["wind_speed_850hPa"] = np.hypot(u850, v850) * MS_TO_MPH
    # FROM-direction (meteorological), matching Open-Meteo's wind_direction_850hPa
    out["wind_direction_850hPa"] = (np.degrees(np.arctan2(-u850, -v850))) % 360.0
    return out


# ── As-of selection ───────────────────────────────────────────────────────────
def select_asof_forecast(fcst: pd.DataFrame, decision_time: pd.Timestamp,
                         valid_time: pd.Timestamp) -> Optional[pd.Series]:
    """Freshest forecast row for one valid hour whose availability precedes the
    decision — THE leakage boundary of the forecast channel."""
    rows = fcst[(fcst["valid_time_utc"] == valid_time)
                & (fcst["available_time_utc"] <= decision_time)]
    if rows.empty:
        return None
    return rows.loc[rows["issue_time_utc"].idxmax()]


def select_asof_obs(obs: pd.DataFrame, decision_time: pd.Timestamp,
                    hour_start: pd.Timestamp) -> Optional[pd.Series]:
    """Latest available report representing hour [hour_start, hour_start+1h).

    The search window opens 1 h early: US ASOS reports at ~:53 and disseminates
    ~10 min later, so hour h's own report lands at (h+1):03 — 3 min AFTER the
    decision at h+1. The (h-1):53 report (7 min before the hour starts) is the
    freshest legal proxy at that decision; the hour's own report takes over one
    decision later via valid_utc.idxmax(). Staleness is bounded at < 2 h.

    Callers only invoke this for elapsed hours (h < d — the gate lives in
    assemble_asof_tensor), and the availability filter still applies: a
    late-disseminated METAR must not appear before it existed."""
    rows = obs[(obs["valid_utc"] >= hour_start - pd.Timedelta(hours=1))
               & (obs["valid_utc"] < hour_start + pd.Timedelta(hours=1))
               & (obs["available_time_utc"] <= decision_time)]
    if rows.empty:
        return None
    return rows.loc[rows["valid_utc"].idxmax()]


# ── Tensor assembly ───────────────────────────────────────────────────────────
def _features_22(era5_row: pd.Series, venue_id: int, cf_azimuth: float,
                 air_quality_row: Optional[pd.Series] = None) -> np.ndarray:
    """One era5-schema row -> the 22-dim vector, via the existing physics."""
    df = pd.DataFrame([era5_row])
    aq = pd.DataFrame([air_quality_row]) if air_quality_row is not None else None
    return compute_hour_features_vectorized(
        df, np.array([venue_id]), np.array([cf_azimuth]),
        air_quality_df=aq, pressure_df=df,
    )[0]


def _standardize_masked(vec22: np.ndarray, mask22: np.ndarray,
                        mean22: Optional[np.ndarray],
                        std22: Optional[np.ndarray]) -> np.ndarray:
    """z-score then mask: masked entries land exactly at 0 (= the mean)."""
    if mean22 is None:
        return vec22 * mask22
    z = (vec22 - mean22) / np.where(std22 > 1e-8, std22, 1.0)
    return z * mask22


def assemble_asof_tensor(
    obs_era5: Optional[pd.DataFrame],
    fcst_era5: Optional[pd.DataFrame],
    game_hour_utc: pd.Timestamp,
    venue_id: int,
    cf_azimuth: float,
    norm_stats: Optional[dict] = None,
    air_quality_by_hour: Optional[dict] = None,
) -> np.ndarray:
    """Build wx_asof[7, 7, 89] for one game.

    obs_era5 / fcst_era5 : era5-schema frames from metar_to_era5 / hrrr_to_era5
        (None or empty -> that channel is fully masked, meaning "unknown").
    norm_stats : {"fcst_mean","fcst_std","obs_mean","obs_std"} each (22,);
        None during the stats-computation pass (raw values, still masked).
    air_quality_by_hour : {target_hour_int: row} of 24h-lagged CAMS persistence
        (dims 17-19); soil persistence rides the fcst frame's soil column.
    """
    T = np.zeros((N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS), dtype=np.float32)
    fm, fs = (norm_stats.get("fcst_mean"), norm_stats.get("fcst_std")) if norm_stats else (None, None)
    om, os_ = (norm_stats.get("obs_mean"), norm_stats.get("obs_std")) if norm_stats else (None, None)
    # Observable obs dims: physics dims 0-13 plus all 5 D2b extras (22-26).
    obs_observable = np.zeros(N_OBS_DIMS, dtype=np.float32)
    obs_observable[OBS_OBSERVABLE_DIMS] = 1.0
    obs_observable[N_DIMS:] = 1.0

    for di, d in enumerate(DECISION_HOURS):
        decision_time = game_hour_utc + pd.Timedelta(hours=d)
        for hi, h in enumerate(TARGET_HOURS):
            hour_start = game_hour_utc + pd.Timedelta(hours=h)
            aq_row = air_quality_by_hour.get(h) if air_quality_by_hour else None

            fcst_vec = np.zeros(N_DIMS, dtype=np.float32)
            fcst_mask = np.zeros(N_DIMS, dtype=np.float32)
            lead_norm = 0.0
            if fcst_era5 is not None and len(fcst_era5):
                row = select_asof_forecast(fcst_era5, decision_time, hour_start)
                if row is not None:
                    fcst_vec = _features_22(row, venue_id, cf_azimuth, aq_row)
                    fcst_mask = np.ones(N_DIMS, dtype=np.float32)
                    # The mask must not claim dims the source cannot populate:
                    # soil (16) rides ERA5 persistence merged into the fcst row;
                    # AQI (17-19) rides CAMS persistence via air_quality_by_hour.
                    if "soil_moisture_0_to_7cm" not in row or pd.isna(row.get("soil_moisture_0_to_7cm")):
                        fcst_mask[16] = 0.0
                    if aq_row is None:
                        fcst_mask[17:20] = 0.0
                    lead_norm = float(np.log1p(row["lead_hours"]) / np.log1p(MAX_LEAD_HOURS))

            obs_vec = np.zeros(N_OBS_DIMS, dtype=np.float32)
            obs_mask = np.zeros(N_OBS_DIMS, dtype=np.float32)
            if h < d and obs_era5 is not None and len(obs_era5):
                row = select_asof_obs(obs_era5, decision_time, hour_start)
                if row is not None:
                    obs_vec[:N_DIMS] = _features_22(row, venue_id, cf_azimuth, aq_row)
                    obs_vec[N_DIMS:] = wx_extra_features(
                        row.get("wxcodes"), row.get("peak_wind_gust_kt"),
                        row.get("wind_gusts_10m"))
                    obs_mask = obs_observable.copy()

            T[di, hi, OFF_FCST:OFF_FCST_MASK] = _standardize_masked(fcst_vec, fcst_mask, fm, fs)
            T[di, hi, OFF_FCST_MASK:OFF_OBS] = fcst_mask
            T[di, hi, OFF_OBS:OFF_OBS_MASK] = _standardize_masked(obs_vec, obs_mask, om, os_)
            T[di, hi, OFF_OBS_MASK:OFF_LEAD] = obs_mask
            T[di, hi, OFF_LEAD] = lead_norm
    return T


# ── Live path ─────────────────────────────────────────────────────────────────
# The daemon (data_curation/scripts/live_daemon.py) fetches AWC obs and the
# freshest HRRR issue hourly and appends them to daily-accumulating S3 files;
# this reader assembles the same tensor the training builder wrote — one shared
# assemble_asof_tensor, so parity holds by construction. The inference box
# never touches GRIB or external weather APIs.
S3_BUCKET = "mlb-265753586044-us-east-1-an"
OBS_LIVE_PREFIX = "data/weather/source=asos_obs_live"
HRRR_LIVE_PREFIX = "data/weather/source=hrrr_asissued_live"
NORM_STATS_KEY = "deep_learning/feature_store/weather_asof_norm.json"
STATION_MAP_KEY = "data/weather/station_venue_map.json"

_s3_client = None


def _get_s3():
    global _s3_client
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client("s3", region_name="us-east-1")
    return _s3_client


def _read_parquet_s3(key: str) -> Optional[pd.DataFrame]:
    try:
        body = _get_s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        return pd.read_parquet(io.BytesIO(body))
    except Exception:
        return None


def load_norm_stats() -> Optional[dict]:
    try:
        raw = json.loads(_get_s3().get_object(Bucket=S3_BUCKET, Key=NORM_STATS_KEY)["Body"].read())
        return {k: np.asarray(raw[k], dtype=np.float32)
                for k in ("fcst_mean", "fcst_std", "obs_mean", "obs_std")}
    except Exception:
        log.warning("weather_asof norm stats unavailable — live tensor would be "
                    "unstandardized; refusing to assemble", exc_info=True)
        return None


def load_station_map() -> dict:
    return json.loads(_get_s3().get_object(Bucket=S3_BUCKET, Key=STATION_MAP_KEY)["Body"].read())


def fetch_live_asof(
    venue_id: int,
    game_hour_utc: pd.Timestamp,
    cf_azimuth: float,
    now: Optional[pd.Timestamp] = None,
    station_map: Optional[dict] = None,
    norm_stats: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """Assemble the live [7, 7, 99] as-of tensor from daemon-written S3 files.

    Returns None when norm stats are unavailable (an unstandardized tensor
    would silently feed the model raw units — worse than the zero-mask path).
    The availability filters inside assemble_asof_tensor still apply, so rows
    fetched seconds ago are only used at decisions that could have seen them.
    """
    now = now or pd.Timestamp.now(tz="UTC")
    norm_stats = norm_stats or load_norm_stats()
    if norm_stats is None:
        return None
    station_map = station_map or load_station_map()
    m = station_map.get(str(int(venue_id)))
    if m is None:
        log.warning(f"venue {venue_id} not in station map — no live weather")
        return None

    # Obs: today + yesterday (late-night games cross the UTC date line)
    obs_frames = []
    for st, elev in ((m["primary_station"], m.get("primary_elev_m")),
                     (m["backup_station"], m.get("backup_elev_m"))):
        for day in (now.normalize(), now.normalize() - pd.Timedelta(days=1)):
            raw = _read_parquet_s3(f"{OBS_LIVE_PREFIX}/station={st}/date={day:%Y-%m-%d}.parquet")
            if raw is not None and len(raw):
                obs_frames.append(metar_to_era5(raw, float(elev or 0.0)))
    obs = pd.concat(obs_frames, ignore_index=True) if obs_frames else None

    fcst_frames = []
    for day in (now.normalize(), now.normalize() - pd.Timedelta(days=1)):
        raw = _read_parquet_s3(f"{HRRR_LIVE_PREFIX}/date={day:%Y-%m-%d}.parquet")
        if raw is not None and len(raw):
            fcst_frames.append(raw[raw["venue_id"] == int(venue_id)])
    fcst = None
    if fcst_frames:
        fcst = hrrr_to_era5(pd.concat(fcst_frames, ignore_index=True))
        # Soil persistence (dim 16): ERA5 at -7d from the daily-refreshed archive
        soil = _read_parquet_s3(
            f"data/weather/source=era5/venue_id={int(venue_id)}/year={game_hour_utc.year}.parquet")
        if soil is not None and "soil_moisture_0_to_7cm" in soil.columns:
            soil = soil[["timestamp", "soil_moisture_0_to_7cm"]].dropna()
            soil["valid_time_utc"] = pd.to_datetime(soil["timestamp"], utc=True) + pd.Timedelta(days=7)
            fcst = fcst.merge(soil[["valid_time_utc", "soil_moisture_0_to_7cm"]],
                              on="valid_time_utc", how="left")

    if obs is None and fcst is None:
        log.warning(f"venue {venue_id}: no live obs and no live forecast on S3")
        return None
    return assemble_asof_tensor(obs, fcst, game_hour_utc, int(venue_id),
                                cf_azimuth, norm_stats=norm_stats)
