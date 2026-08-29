#!/bin/bash
# Probe actual disk usage for one variable before committing to a full sync.
# Syncs copernicus_era5_land/temperature_2m for 730 days, measures the delta,
# then extrapolates for our full variable list × 2015–2026 date range.
#
# Run this BEFORE provisioning final EBS storage.
# The reference claim "9 GiB per 80-year record" could not be verified — this
# probe will give the real number for your deployment.
#
# Usage:
#   export OPENMETEO_DATADIR=/data/openmeteo   # must match bootstrap.sh
#   bash probe_size.sh

set -euo pipefail

DATADIR="${OPENMETEO_DATADIR:-/data/openmeteo}"
PROBE_DAYS=730

echo "=== Probe: copernicus_era5_land / temperature_2m (${PROBE_DAYS} days) ==="
echo "Data dir: ${DATADIR}"
echo ""

BEFORE_KB=$(du -sk "$DATADIR" 2>/dev/null | cut -f1 || echo 0)

# Note: --past-days flag form — the docs show both --past-days and --past_days.
# If this fails, try replacing with --past_days.
sudo docker run --rm \
  -v "${DATADIR}:/app/data" \
  ghcr.io/open-meteo/open-meteo:latest \
  sync copernicus_era5_land temperature_2m --past-days "$PROBE_DAYS"

AFTER_KB=$(du -sk "$DATADIR" | cut -f1)
DELTA_KB=$((AFTER_KB - BEFORE_KB))
DELTA_GIB=$(echo "scale=3; $DELTA_KB / 1048576" | bc)
GIB_PER_YEAR=$(echo "scale=4; $DELTA_GIB * 365 / $PROBE_DAYS" | bc)

echo ""
echo "=== Results ==="
printf "  %-35s %s GiB\n" "temperature_2m @ ${PROBE_DAYS} days:" "$DELTA_GIB"
printf "  %-35s %s GiB/year\n" "Per-variable rate:" "$GIB_PER_YEAR"
echo ""

# ── Extrapolation ──────────────────────────────────────────────────────────────
# Variable counts are conservative — derived vars (apparent_temp, dew_point,
# wind_speed) are computed by the API from stored primitives and don't need syncing.
# Actual stored variable list per dataset must be confirmed at:
#   https://github.com/open-meteo/open-data
ERA5_LAND_STORED_VARS=18   # temperature, RH, wind_u/v, pressure, precip, snowfall,
                            # snow_depth, ET0, radiation×4, soil_temp×4, soil_moist×4
ERA5_SURF_STORED_VARS=15   # similar but coarser grid (31km vs 9km)
ECMWF_IFS_STORED_VARS=12   # fewer soil vars, shorter history (2017+)
HRRR_STORED_VARS=12        # CONUS only; 3km resolution — expect larger per-var

YEARS_ERA5=12    # 2015–2026
YEARS_ECMWF=9   # 2017–2026
YEARS_HRRR=8    # 2018–2026

ERA5_LAND_EST=$(echo "scale=1; $GIB_PER_YEAR * $YEARS_ERA5 * $ERA5_LAND_STORED_VARS" | bc)
ERA5_SURF_EST=$(echo "scale=1; $GIB_PER_YEAR * $YEARS_ERA5 * $ERA5_SURF_STORED_VARS * 0.6" | bc)  # 31km vs 9km ≈ 60% size
ECMWF_EST=$(echo  "scale=1; $GIB_PER_YEAR * $YEARS_ECMWF * $ECMWF_IFS_STORED_VARS * 0.8" | bc)
HRRR_EST=$(echo   "scale=1; $GIB_PER_YEAR * $YEARS_HRRR  * $HRRR_STORED_VARS        * 0.4" | bc)  # CONUS-only fraction
TOTAL=$(echo "scale=1; $ERA5_LAND_EST + $ERA5_SURF_EST + $ECMWF_EST + $HRRR_EST" | bc)

echo "=== Extrapolated storage for full sync (2015–2026) ==="
printf "  %-45s %s GiB\n" "copernicus_era5_land  (${ERA5_LAND_STORED_VARS} vars × ${YEARS_ERA5}yr):" "$ERA5_LAND_EST"
printf "  %-45s %s GiB\n" "copernicus_era5       (${ERA5_SURF_STORED_VARS} vars × ${YEARS_ERA5}yr, 31km):" "$ERA5_SURF_EST"
printf "  %-45s %s GiB\n" "ecmwf_ifs             (${ECMWF_IFS_STORED_VARS} vars × ${YEARS_ECMWF}yr):" "$ECMWF_EST"
printf "  %-45s %s GiB\n" "ncep_hrrr_conus       (${HRRR_STORED_VARS} vars × ${YEARS_HRRR}yr, CONUS):" "$HRRR_EST"
echo "  ──────────────────────────────────────────────"
printf "  %-45s %s GiB\n" "Total:" "$TOTAL"
echo ""
echo "Provision EBS gp3 volume at: \$(( (${TOTAL%.*} / 100 + 1) * 100 )) GiB (round up to next 100)"
echo ""
echo "NOTE: HRRR estimate is speculative — run a separate probe with ncep_hrrr_conus"
echo "      before provisioning if HRRR is required."
echo ""
echo "NOT available via self-hosted sync (excluded from AWS open data):"
echo "  - flood / GloFAS — stays on hosted flood-api.open-meteo.com"
echo ""
echo "Available and included in sync_all.sh:"
echo "  - marine: copernicus_era5_ocean, ecmwf_wam025, ncep_gfswave025"
echo "  - air quality: cams_global, cams_europe"
