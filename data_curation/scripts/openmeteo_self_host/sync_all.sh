#!/bin/bash
# Sync all required weather variables from AWS open data into the local
# Open-Meteo data directory.
#
# Variable names here are SYNC names (stored primitives), not query names.
# The API derives: wind_speed/direction from u/v components, dew_point/apparent_temp
# from T+RH, is_day from coords+time. Do NOT sync derived variables.
#
# Dataset → fetch_weather.py source mapping:
#   copernicus_era5_land  → era5_land        (9km, 1950+)
#   copernicus_era5       → era5 + era5_pressure (31km, 1940+)
#   ecmwf_ifs             → ecmwf_ifs        (archive, 2017+)
#   ncep_hrrr_conus       → hrrr_forecast + hrrr_forecast_pressure (CONUS, 2018+)
#   ncep_gfs025           → gfs_forecast + gfs_forecast_pressure (global, 2021+)
#   copernicus_era5_ocean → marine           (ERA5-Ocean, 1940+)
#   ecmwf_wam025          → marine (alt)
#   ncep_gfswave025       → marine (alt)
#   cams_global           → air_quality      (2013+)
#   cams_europe           → air_quality (EU)
#
# NOT available via sync:
#   flood/GloFAS          → stays on hosted flood-api.open-meteo.com
#
# Usage:
#   export OPENMETEO_DATADIR=/data/openmeteo
#   bash sync_all.sh             # sync all datasets
#   bash sync_all.sh era5_land   # sync a single dataset by name
#
# Optional env:
#   SYNC_PAST_DAYS=9500  limit historical depth (days); omit for full history

set -euo pipefail

DATADIR="${OPENMETEO_DATADIR:-/data/openmeteo}"
IMAGE="ghcr.io/open-meteo/open-meteo:latest"
# Build --past-days flag once; empty string → no flag (sync all available history)
PAST_DAYS_FLAG=""
if [ -n "${SYNC_PAST_DAYS:-}" ]; then
    PAST_DAYS_FLAG="--past-days ${SYNC_PAST_DAYS}"
fi

_sync() {
    local dataset="$1"
    local vars="$2"
    echo ""
    echo "=== Syncing ${dataset} ${PAST_DAYS_FLAG:+(past-days=${SYNC_PAST_DAYS})} ==="
    echo "    Variables: ${vars:0:120}..."
    echo ""
    # shellcheck disable=SC2086  # PAST_DAYS_FLAG must word-split
    sudo docker run --rm \
        -v "${DATADIR}:/app/data" \
        "$IMAGE" \
        sync "$dataset" "$vars" ${PAST_DAYS_FLAG}
    echo "=== Done: ${dataset} ==="
}

# Generate pressure-level variable list: <base_var>_<level>hPa for all levels
# Wind stored as u/v components; API derives speed/direction at query time.
_pressure_vars() {
    local levels="$1"   # comma-separated level list
    local base="$2"     # comma-separated base variable list
    python3 -c "
levels = [int(x) for x in '$levels'.split(',')]
base   = '$base'.split(',')
print(','.join(f'{v}_{l}hPa' for l in levels for v in base))
"
}

DATASET_FILTER="${1:-all}"

# ── ERA5-Land surface (9km, 1950+) ───────────────────────────────────────────
ERA5_LAND_VARS="\
temperature_2m,\
relative_humidity_2m,\
wind_u_component_10m,\
wind_v_component_10m,\
surface_pressure,\
precipitation,\
rain,\
snowfall,\
snow_depth,\
et0_fao_evapotranspiration,\
shortwave_radiation,\
direct_radiation,\
diffuse_radiation,\
direct_normal_irradiance,\
terrestrial_radiation,\
soil_temperature_0_to_7cm,\
soil_temperature_7_to_28cm,\
soil_temperature_28_to_100cm,\
soil_temperature_100_to_255cm,\
soil_moisture_0_to_7cm,\
soil_moisture_7_to_28cm,\
soil_moisture_28_to_100cm,\
soil_moisture_100_to_255cm"

case "$DATASET_FILTER" in era5_land | all)
    _sync "copernicus_era5_land" "$ERA5_LAND_VARS"
    ;;
esac

# ── ERA5 surface (31km, 1940+) ────────────────────────────────────────────────
ERA5_SURF_VARS="\
temperature_2m,\
relative_humidity_2m,\
wind_u_component_10m,\
wind_v_component_10m,\
surface_pressure,\
pressure_msl,\
precipitation,\
rain,\
snowfall,\
snow_depth,\
cloud_cover,\
cloud_cover_low,\
cloud_cover_mid,\
cloud_cover_high,\
shortwave_radiation,\
direct_radiation,\
diffuse_radiation,\
direct_normal_irradiance,\
terrestrial_radiation,\
weather_code,\
boundary_layer_height,\
soil_temperature_0_to_7cm,\
soil_temperature_7_to_28cm,\
soil_temperature_28_to_100cm,\
soil_temperature_100_to_255cm,\
soil_moisture_0_to_7cm,\
soil_moisture_7_to_28cm,\
soil_moisture_28_to_100cm,\
soil_moisture_100_to_255cm,\
et0_fao_evapotranspiration"

case "$DATASET_FILTER" in era5 | all)
    _sync "copernicus_era5" "$ERA5_SURF_VARS"
    ;;
esac

# ── ERA5 pressure levels (19 levels, 1940+) ───────────────────────────────────
ERA5_PLEVELS="1000,975,950,925,900,850,800,700,600,500,400,300,250,200,150,100,70,50,30"
ERA5_PBASE="temperature,relative_humidity,cloud_cover,wind_u_component,wind_v_component,geopotential_height"
ERA5_PRESSURE_VARS=$(_pressure_vars "$ERA5_PLEVELS" "$ERA5_PBASE")

case "$DATASET_FILTER" in era5_pressure | all)
    _sync "copernicus_era5" "$ERA5_PRESSURE_VARS"
    ;;
esac

# ── ECMWF IFS archive surface (2017+) ────────────────────────────────────────
ECMWF_IFS_VARS="\
temperature_2m,\
relative_humidity_2m,\
wind_u_component_10m,\
wind_v_component_10m,\
wind_u_component_80m,\
wind_v_component_80m,\
wind_gusts_10m,\
surface_pressure,\
pressure_msl,\
precipitation,\
snowfall,\
cloud_cover,\
cloud_cover_low,\
cloud_cover_mid,\
cloud_cover_high,\
shortwave_radiation,\
direct_radiation,\
diffuse_radiation,\
direct_normal_irradiance,\
terrestrial_radiation,\
weather_code,\
boundary_layer_height,\
cape,\
visibility,\
soil_temperature_0_to_7cm,\
soil_temperature_7_to_28cm,\
soil_temperature_28_to_100cm,\
soil_temperature_100_to_255cm,\
soil_moisture_0_to_7cm,\
soil_moisture_7_to_28cm,\
soil_moisture_28_to_100cm,\
soil_moisture_100_to_255cm,\
et0_fao_evapotranspiration"

case "$DATASET_FILTER" in ecmwf_ifs | all)
    _sync "ecmwf_ifs" "$ECMWF_IFS_VARS"
    ;;
esac

# ── ECMWF IFS pressure levels (2017+) ────────────────────────────────────────
ECMWF_PLEVELS="1000,975,950,925,900,850,800,700,600,500,400,300,250,200,150,100,70,50,30"
ECMWF_PBASE="temperature,relative_humidity,cloud_cover,wind_u_component,wind_v_component,geopotential_height"
ECMWF_PRESSURE_VARS=$(_pressure_vars "$ECMWF_PLEVELS" "$ECMWF_PBASE")

case "$DATASET_FILTER" in ecmwf_ifs_pressure | all)
    _sync "ecmwf_ifs" "$ECMWF_PRESSURE_VARS"
    ;;
esac

# ── HRRR historical forecast surface (CONUS, 2018+) ──────────────────────────
HRRR_SURF_VARS="\
temperature_2m,\
relative_humidity_2m,\
wind_u_component_10m,\
wind_v_component_10m,\
wind_u_component_80m,\
wind_v_component_80m,\
wind_u_component_100m,\
wind_v_component_100m,\
wind_gusts_10m,\
surface_pressure,\
precipitation,\
rain,\
snowfall,\
snow_depth,\
cloud_cover,\
cloud_cover_low,\
cloud_cover_mid,\
cloud_cover_high,\
shortwave_radiation,\
direct_radiation,\
diffuse_radiation,\
direct_normal_irradiance,\
terrestrial_radiation,\
cape,\
weather_code,\
boundary_layer_height,\
convective_inhibition,\
freezing_level_height,\
snowfall_height,\
visibility,\
soil_temperature_0_to_10cm,\
soil_temperature_10_to_40cm,\
soil_temperature_40_to_100cm,\
soil_temperature_100_to_200cm,\
soil_moisture_0_to_10cm,\
soil_moisture_10_to_40cm,\
soil_moisture_40_to_100cm,\
soil_moisture_100_to_200cm,\
surface_temperature,\
total_column_integrated_water_vapour"

case "$DATASET_FILTER" in hrrr | all)
    _sync "ncep_hrrr_conus" "$HRRR_SURF_VARS"
    ;;
esac

# ── HRRR pressure levels (44 levels, CONUS, 2018+) ───────────────────────────
HRRR_PLEVELS="1000,975,950,925,900,875,850,825,800,775,750,725,700,675,650,625,600,575,550,525,500,475,450,425,400,375,350,325,300,275,250,225,200,175,150,125,100,70,50,40,30,20,15,10"
HRRR_PBASE="temperature,relative_humidity,dew_point,cloud_cover,wind_u_component,wind_v_component,vertical_velocity,geopotential_height"
HRRR_PRESSURE_VARS=$(_pressure_vars "$HRRR_PLEVELS" "$HRRR_PBASE")

case "$DATASET_FILTER" in hrrr_pressure | all)
    _sync "ncep_hrrr_conus" "$HRRR_PRESSURE_VARS"
    ;;
esac

# ── GFS historical forecast surface (global, 2021+) ──────────────────────────
GFS_SURF_VARS="\
temperature_2m,\
relative_humidity_2m,\
wind_u_component_10m,\
wind_v_component_10m,\
wind_u_component_80m,\
wind_v_component_80m,\
wind_u_component_100m,\
wind_v_component_100m,\
wind_gusts_10m,\
surface_pressure,\
pressure_msl,\
precipitation,\
snowfall,\
cloud_cover,\
shortwave_radiation,\
direct_radiation,\
cape,\
weather_code,\
boundary_layer_height,\
convective_inhibition,\
freezing_level_height,\
snowfall_height,\
visibility,\
soil_temperature_0_to_10cm,\
soil_temperature_10_to_40cm,\
soil_temperature_40_to_100cm,\
soil_temperature_100_to_200cm,\
soil_moisture_0_to_10cm,\
soil_moisture_10_to_40cm,\
soil_moisture_40_to_100cm,\
soil_moisture_100_to_200cm,\
surface_temperature,\
total_column_integrated_water_vapour,\
precipitation_probability,\
thunderstorm_probability,\
rain_probability,\
snowfall_probability,\
freezing_rain_probability"

case "$DATASET_FILTER" in gfs | all)
    _sync "ncep_gfs025" "$GFS_SURF_VARS"
    ;;
esac

# ── GFS pressure levels (44 levels, global, 2021+) ───────────────────────────
GFS_PLEVELS="1000,975,950,925,900,875,850,825,800,775,750,725,700,675,650,625,600,575,550,525,500,475,450,425,400,375,350,325,300,275,250,225,200,175,150,125,100,70,50,40,30,20,15,10"
GFS_PBASE="temperature,relative_humidity,dew_point,cloud_cover,wind_u_component,wind_v_component,vertical_velocity,geopotential_height"
GFS_PRESSURE_VARS=$(_pressure_vars "$GFS_PLEVELS" "$GFS_PBASE")

case "$DATASET_FILTER" in gfs_pressure | all)
    _sync "ncep_gfs025" "$GFS_PRESSURE_VARS"
    ;;
esac

# ── Marine / ocean ────────────────────────────────────────────────────────────
ERA5_OCEAN_VARS="\
sea_surface_temperature,\
sea_level_height_msl,\
ocean_current_velocity,\
ocean_current_direction,\
invert_barometer_height"

WAVE_VARS="\
wave_height,\
wave_direction,\
wave_period,\
wave_peak_period,\
wind_wave_height,\
wind_wave_direction,\
wind_wave_period,\
swell_wave_height,\
swell_wave_direction,\
swell_wave_period"

case "$DATASET_FILTER" in marine | all)
    _sync "copernicus_era5_ocean" "$ERA5_OCEAN_VARS"
    _sync "ecmwf_wam025"          "$WAVE_VARS"
    _sync "ncep_gfswave025"       "$WAVE_VARS"
    ;;
esac

# ── Air quality (CAMS, 2013+) ─────────────────────────────────────────────────
CAMS_VARS="\
pm10,\
pm2_5,\
carbon_monoxide,\
nitrogen_dioxide,\
sulphur_dioxide,\
ozone,\
dust,\
aerosol_optical_depth,\
ammonia"

case "$DATASET_FILTER" in air_quality | all)
    _sync "cams_global"  "$CAMS_VARS"
    _sync "cams_europe"  "$CAMS_VARS"
    ;;
esac

echo ""
echo "=== Sync complete ==="
echo "Total data directory size: $(du -sh "$DATADIR" | cut -f1)"
echo ""
echo "NOT synced (excluded from AWS open data):"
echo "  flood/GloFAS — stays on hosted flood-api.open-meteo.com"
