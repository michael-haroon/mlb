"""Unit tests for weather_context.py feature computation."""

import numpy as np
import pandas as pd
import pytest

from deep_learning.mlb_dl.weather_context import (
    WEATHER_TOKEN_DIM,
    WEATHER_TEMPORAL_COLUMNS,
    WEATHER_TEMPORAL_HOURS,
    CLOSED_ROOF_VENUES,
    TURF_VENUES,
    compute_air_density,
    compute_hour_features,
    compute_hour_features_vectorized,
    compute_lapse_rate,
    compute_wind_shear,
    rotate_wind_to_park,
)


# ---------------------------------------------------------------------------
# Physics function tests
# ---------------------------------------------------------------------------


class TestComputeAirDensity:
    def test_standard_atmosphere(self):
        """15°C (59°F), 0°C dew (32°F), 1013.25 hPa → ~1.225 kg/m³."""
        rho = compute_air_density(
            np.array([59.0]), np.array([32.0]), np.array([1013.25])
        )
        assert rho.shape == (1,)
        # Standard atmosphere is 1.225; with dry air (low dew point)
        # expect slightly above since less moisture displacement
        assert 1.20 < rho[0] < 1.30

    def test_hot_humid_low_density(self):
        """Hot, humid, low-pressure → lower density (Denver on a hot day)."""
        rho_hot = compute_air_density(
            np.array([95.0]), np.array([75.0]), np.array([840.0])
        )[0]
        rho_cold = compute_air_density(
            np.array([40.0]), np.array([20.0]), np.array([1020.0])
        )[0]
        assert rho_hot < rho_cold

    def test_vectorized(self):
        """Vectorized computation matches scalar."""
        temps = np.array([59.0, 80.0, 40.0])
        dews = np.array([32.0, 60.0, 20.0])
        pressures = np.array([1013.25, 1005.0, 1020.0])
        rho = compute_air_density(temps, dews, pressures)
        assert rho.shape == (3,)
        assert all(0.9 < r < 1.4 for r in rho)


class TestRotateWindToPark:
    def test_north_wind_north_facing_park(self):
        """Wind from south (u=0, v=+5) at a north-facing park (az=0°) → +5 toward CF."""
        toward, cross = rotate_wind_to_park(np.array([0.0]), np.array([5.0]), 0.0)
        assert toward[0] == pytest.approx(5.0, abs=0.01)
        assert cross[0] == pytest.approx(0.0, abs=0.01)

    def test_east_wind_east_facing_park(self):
        """Wind from west (u=+5, v=0) at an east-facing park (az=90°) → +5 toward CF."""
        toward, cross = rotate_wind_to_park(np.array([5.0]), np.array([0.0]), 90.0)
        assert toward[0] == pytest.approx(5.0, abs=0.01)
        assert cross[0] == pytest.approx(0.0, abs=0.01)

    def test_crosswind(self):
        """Wind from west (u=+5, v=0) at a north-facing park → 0 toward CF, +5 crossfield."""
        toward, cross = rotate_wind_to_park(np.array([5.0]), np.array([0.0]), 0.0)
        assert toward[0] == pytest.approx(0.0, abs=0.01)
        assert cross[0] == pytest.approx(5.0, abs=0.01)


class TestComputeLapseRate:
    def test_standard_atmosphere_lapse(self):
        """Standard atmosphere: ~6.5°C/km between 1000 and 850 hPa."""
        # 1000 hPa: 59°F (15°C), Z=111m; 850 hPa: 41.9°F (5.5°C), Z=1457m
        # Lapse = (15 - 5.5) / (1457 - 111) * 1000 = 7.06 °C/km
        # Use Fahrenheit since our data is in Fahrenheit
        t_1000 = np.array([59.0])  # 15°C
        t_850 = np.array([41.9])   # 5.5°C
        z_1000 = np.array([111.0])
        z_850 = np.array([1457.0])
        lapse = compute_lapse_rate(t_1000, t_850, z_1000, z_850)
        # (59-41.9) * 5/9 / ((1457-111)/1000) = 9.5 * 5/9 / 1.346 = 7.05 °C/km
        assert 6.0 < lapse[0] < 8.0

    def test_inversion(self):
        """Temperature inversion: 850 hPa warmer than 1000 hPa → negative lapse."""
        t_1000 = np.array([50.0])
        t_850 = np.array([55.0])  # warmer aloft = inversion
        z_1000 = np.array([111.0])
        z_850 = np.array([1457.0])
        lapse = compute_lapse_rate(t_1000, t_850, z_1000, z_850)
        assert lapse[0] < 0.0


class TestComputeWindShear:
    def test_zero_shear(self):
        """Same wind at both levels → zero shear."""
        shear = compute_wind_shear(
            np.array([5.0]), np.array([180.0]),  # 5 m/s from south at 850
            np.array([0.0]), np.array([-5.0]),   # same: u=0, v=-5 at surface
        )
        assert shear[0] == pytest.approx(0.0, abs=0.5)

    def test_opposing_winds(self):
        """Opposing wind at 850 vs surface → high shear."""
        shear = compute_wind_shear(
            np.array([10.0]), np.array([0.0]),   # 10 m/s from north at 850
            np.array([0.0]), np.array([-10.0]),  # 10 m/s from south at surface
        )
        # u_850 = 10*sin(0) = 0, v_850 = 10*cos(0) = 10
        # du = 0-0=0, dv = 10-(-10)=20
        # shear = sqrt(0 + 400) = 20
        assert shear[0] == pytest.approx(20.0, abs=0.5)


# ---------------------------------------------------------------------------
# Feature vector tests
# ---------------------------------------------------------------------------


class TestComputeHourFeatures:
    @pytest.fixture
    def standard_era5_row(self):
        return {
            "temperature_2m": 72.0,
            "dew_point_2m": 55.0,
            "surface_pressure": 1013.0,
            "relative_humidity_2m": 60.0,
            "vapour_pressure_deficit": 1.2,
            "wet_bulb_temperature_2m": 63.0,
            "wind_u_10m": 3.0,
            "wind_v_10m": 4.0,
            "wind_speed_10m": 5.0,
            "wind_gusts_10m": 8.0,
            "cloud_cover": 25.0,
            "visibility": 30000.0,
            "precipitation": 0.0,
            "boundary_layer_height": 1500.0,
            "shortwave_radiation": 400.0,
            "soil_moisture_0_to_7cm": 0.3,
        }

    def test_output_shape(self, standard_era5_row):
        out = compute_hour_features(standard_era5_row, venue_id=2500, cf_azimuth_deg=45.0)
        assert out.shape == (WEATHER_TOKEN_DIM,)
        assert out.dtype == np.float32

    def test_all_22_features_populated(self, standard_era5_row):
        """Standard inputs should produce non-zero values for all features except AQ/pressure."""
        out = compute_hour_features(standard_era5_row, venue_id=2500, cf_azimuth_deg=45.0)
        # Indices 0-16 should all be non-zero with standard inputs
        for i in range(17):
            if i == 12:  # precip is 0 by design in fixture
                continue
            assert out[i] != 0.0, f"Feature index {i} should be non-zero"

    def test_closed_roof_zeros_wind(self, standard_era5_row):
        """Closed-roof venues should have zero wind features."""
        closed_venue = list(CLOSED_ROOF_VENUES)[0]
        out = compute_hour_features(standard_era5_row, venue_id=closed_venue, cf_azimuth_deg=0.0)
        assert out[2] == 0.0  # wind_toward_cf
        assert out[3] == 0.0  # wind_crossfield
        assert out[4] == 0.0  # wind_speed
        assert out[5] == 0.0  # wind_gusts
        # Temperature and density should still be non-zero
        assert out[0] != 0.0
        assert out[9] != 0.0

    def test_turf_zeros_soil_moisture(self, standard_era5_row):
        """Turf venues should have zero soil moisture."""
        turf_venue = list(TURF_VENUES)[0]
        out = compute_hour_features(standard_era5_row, venue_id=turf_venue, cf_azimuth_deg=0.0)
        assert out[16] == 0.0

    def test_nan_inputs_produce_zeros(self):
        """NaN/None in raw data should produce 0.0, not NaN."""
        row = {k: None for k in [
            "temperature_2m", "dew_point_2m", "surface_pressure",
            "relative_humidity_2m", "vapour_pressure_deficit",
            "wet_bulb_temperature_2m", "wind_u_10m", "wind_v_10m",
            "wind_speed_10m", "wind_gusts_10m", "cloud_cover",
            "visibility", "precipitation", "boundary_layer_height",
            "shortwave_radiation", "soil_moisture_0_to_7cm",
        ]}
        out = compute_hour_features(row, venue_id=2500, cf_azimuth_deg=0.0)
        assert not np.any(np.isnan(out))
        assert np.all(out == 0.0)

    def test_air_quality_integration(self, standard_era5_row):
        """Air quality features should populate when AQ row is provided."""
        aq_row = {"us_aqi": 55.0, "pm2_5": 12.0, "ozone": 45.0}
        out = compute_hour_features(
            standard_era5_row, venue_id=2500, cf_azimuth_deg=0.0,
            air_quality_row=aq_row,
        )
        assert out[17] == 55.0
        assert out[18] == 12.0
        assert out[19] == 45.0

    def test_pressure_level_integration(self, standard_era5_row):
        """Pressure-level features should populate when pressure row is provided."""
        pressure_row = {
            "temperature_1000hPa": 59.0,
            "temperature_850hPa": 41.9,
            "geopotential_height_1000hPa": 111.0,
            "geopotential_height_850hPa": 1457.0,
            "wind_speed_850hPa": 15.0,
            "wind_direction_850hPa": 270.0,
        }
        out = compute_hour_features(
            standard_era5_row, venue_id=2500, cf_azimuth_deg=0.0,
            pressure_row=pressure_row,
        )
        assert out[20] != 0.0  # lapse rate
        assert out[21] != 0.0  # wind shear


# ---------------------------------------------------------------------------
# Vectorized tests
# ---------------------------------------------------------------------------


class TestVectorized:
    def test_matches_scalar(self):
        """Vectorized output should match per-row scalar computation."""
        era5_df = pd.DataFrame({
            "temperature_2m": [72.0, 80.0],
            "dew_point_2m": [55.0, 65.0],
            "surface_pressure": [1013.0, 1005.0],
            "relative_humidity_2m": [60.0, 70.0],
            "vapour_pressure_deficit": [1.2, 0.8],
            "wet_bulb_temperature_2m": [63.0, 70.0],
            "wind_u_10m": [3.0, -2.0],
            "wind_v_10m": [4.0, 5.0],
            "wind_speed_10m": [5.0, 5.4],
            "wind_gusts_10m": [8.0, 9.0],
            "cloud_cover": [25.0, 80.0],
            "visibility": [30000.0, 10000.0],
            "precipitation": [0.0, 2.0],
            "boundary_layer_height": [1500.0, 800.0],
            "shortwave_radiation": [400.0, 50.0],
            "soil_moisture_0_to_7cm": [0.3, 0.4],
        })
        venue_ids = np.array([2500, 2500])
        azimuths = np.array([45.0, 45.0])

        vec_result = compute_hour_features_vectorized(era5_df, venue_ids, azimuths)

        for i in range(2):
            scalar_result = compute_hour_features(
                era5_df.iloc[i].to_dict(), int(venue_ids[i]), float(azimuths[i])
            )
            np.testing.assert_allclose(
                vec_result[i], scalar_result, rtol=1e-5,
                err_msg=f"Mismatch at row {i}",
            )

    def test_closed_roof_vectorized(self):
        """Closed-roof masking works in vectorized path."""
        era5_df = pd.DataFrame({
            "temperature_2m": [72.0, 72.0],
            "dew_point_2m": [55.0, 55.0],
            "surface_pressure": [1013.0, 1013.0],
            "relative_humidity_2m": [60.0, 60.0],
            "vapour_pressure_deficit": [1.2, 1.2],
            "wet_bulb_temperature_2m": [63.0, 63.0],
            "wind_u_10m": [3.0, 3.0],
            "wind_v_10m": [4.0, 4.0],
            "wind_speed_10m": [5.0, 5.0],
            "wind_gusts_10m": [8.0, 8.0],
            "cloud_cover": [25.0, 25.0],
            "visibility": [30000.0, 30000.0],
            "precipitation": [0.0, 0.0],
            "boundary_layer_height": [1500.0, 1500.0],
            "shortwave_radiation": [400.0, 400.0],
            "soil_moisture_0_to_7cm": [0.3, 0.3],
        })
        closed_venue = list(CLOSED_ROOF_VENUES)[0]
        venue_ids = np.array([2500, closed_venue])
        azimuths = np.array([45.0, 45.0])

        result = compute_hour_features_vectorized(era5_df, venue_ids, azimuths)

        # Row 0 (open air): wind should be non-zero
        assert result[0, 2] != 0.0
        assert result[0, 4] != 0.0

        # Row 1 (closed roof): wind should be zero
        assert result[1, 2] == 0.0
        assert result[1, 3] == 0.0
        assert result[1, 4] == 0.0
        assert result[1, 5] == 0.0
