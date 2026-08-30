"""Adversarial tests for the as-of weather tensor.

The leakage guarantee is structural: a forecast issued after the decision hour
or an observation from a non-elapsed hour appearing anywhere in wx_asof[d]
reproduces the exact defect this module exists to remove. Poison-sentinel rows
verify absence, not just presence.
"""

import numpy as np
import pandas as pd
import pytest

from mlb_dl.weather_asof import (
    ASOF_CHANNELS,
    N_DIMS,
    N_OBS_DIMS,
    N_DECISIONS,
    N_TARGET_HOURS,
    OBS_OBSERVABLE_DIMS,
    OFF_FCST,
    OFF_FCST_MASK,
    OFF_OBS,
    OFF_OBS_MASK,
    OFF_LEAD,
    assemble_asof_tensor,
    hrrr_to_era5,
    metar_to_era5,
    relative_humidity_pct,
    select_asof_forecast,
    station_pressure_hpa,
    vapour_pressure_deficit_kpa,
    wet_bulb_f,
    wx_extra_features,
)

GH = pd.Timestamp("2023-07-14 23:00", tz="UTC")
VENUE = 3313  # Fenway (open roof, natural grass — no venue masking in play)
CF_AZ = 30.0
POISON = 6666.0  # unmistakable sentinel °K/°F magnitude


# ── Fixtures ──────────────────────────────────────────────────────────────────
def _hrrr_frame():
    """Issues hourly 14:00-05:00Z; each row one (issue, valid) pair with
    availability = issue + 75 min. Values encode (issue_hour, valid_hour) so a
    test can identify exactly WHICH forecast landed in a slot."""
    rows = []
    for issue_h in range(14, 30):
        issue = GH.normalize() + pd.Timedelta(hours=issue_h)
        for fxx in range(1, 10):
            valid = issue + pd.Timedelta(hours=fxx)
            rows.append(dict(
                venue_id=VENUE, model="hrrr",
                issue_time_utc=issue,
                available_time_utc=issue + pd.Timedelta(minutes=75),
                valid_time_utc=valid, lead_hours=fxx,
                t2m_k=290.0 + issue_h * 0.01 + fxx * 0.0001,
                d2m_k=280.0, sp_pa=101000.0,
                u10_ms=2.0, v10_ms=1.0, gust_ms=5.0,
                tcc_pct=50.0, vis_m=16000.0, apcp_mm=0.0,
                hpbl_m=800.0, dswrf_wm2=100.0,
                t850_k=285.0, t1000_k=293.0, z850_m=1500.0, z1000_m=100.0,
                u850_ms=5.0, v850_ms=2.0,
            ))
    return hrrr_to_era5(pd.DataFrame(rows))


def _obs_frame():
    """METARs at :53 each hour 20:00Z..06:00Z, +10 min dissemination."""
    rows = []
    for hh in range(20, 31):
        valid = GH.normalize() + pd.Timedelta(hours=hh, minutes=53)
        rows.append(dict(
            station="BOS", valid_utc=valid,
            available_time_utc=valid + pd.Timedelta(minutes=10),
            tmpf=70.0 + hh * 0.01, dwpf=55.0, relh=np.nan, drct=180.0,
            sknt=10.0, gust=np.nan, alti=29.92, mslp=np.nan, vsby=10.0,
            skyc1="SCT", skyl1=5000.0, skyc2=None, skyl2=None,
            skyc3=None, skyl3=None, p01i=0.0,
        ))
    return metar_to_era5(pd.DataFrame(rows), station_elev_m=6.0)


@pytest.fixture(scope="module")
def tensor():
    return assemble_asof_tensor(_obs_frame(), _hrrr_frame(), GH, VENUE, CF_AZ)


# ── Shape and layout ──────────────────────────────────────────────────────────
def test_shape(tensor):
    assert tensor.shape == (N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS) == (7, 7, 99)


def test_pregame_row_has_hour_minus1_observed(tensor):
    """d=0 must carry a real current observation at h=-1 (hi=0)."""
    obs_mask = tensor[0, 0, OFF_OBS_MASK:OFF_LEAD]
    assert obs_mask[OBS_OBSERVABLE_DIMS].all()


def test_d0_has_no_future_observations(tensor):
    """At d=0 every hour h >= 0 (hi >= 1) must have a zero obs channel."""
    assert not tensor[0, 1:, OFF_OBS_MASK:OFF_LEAD].any()


def test_obs_rule_h_lt_d(tensor):
    """Hour h observed iff h < d, for every (d, h)."""
    for di in range(7):
        for hi, h in enumerate(range(-1, 6)):
            observed = tensor[di, hi, OFF_OBS_MASK:OFF_LEAD].any()
            assert observed == (h < di), (di, h)


def test_forecast_stays_populated_at_observed_hours(tensor):
    """D1: the last pre-observation forecast remains readable alongside obs."""
    assert tensor[6, 0, OFF_FCST_MASK:OFF_OBS].any()  # h=-1 fully observed at d=6
    assert tensor[6, 0, OFF_OBS_MASK:OFF_LEAD].any()


def test_unobservable_dims_never_observed(tensor):
    unobservable = [i for i in range(N_DIMS) if i not in OBS_OBSERVABLE_DIMS]
    assert not tensor[:, :, OFF_OBS_MASK:OFF_LEAD][:, :, unobservable].any()


def test_persistence_dims_masked_without_sources(tensor):
    """No soil column and no AQ rows were provided: fcst_mask dims 16-19 must
    admit that instead of claiming zeros are real measurements."""
    fcst_mask = tensor[:, :, OFF_FCST_MASK:OFF_OBS]
    assert not fcst_mask[:, :, 16:20].any()
    assert fcst_mask[:, :, 9].all() == (tensor[:, :, OFF_FCST_MASK + 9] == 1).all()


# ── Leakage (poison sentinels) ────────────────────────────────────────────────
def test_planted_future_issue_never_selected():
    """A forecast available 1 min after the LAST decision hour must never
    appear anywhere. (A forecast available at decision d + 1 min IS legitimate
    at decisions > d, so the sentinel must postdate every decision.)"""
    fcst = _hrrr_frame()
    last_decision = GH + pd.Timedelta(hours=6)
    poison_rows = [dict(
        issue_time_utc=last_decision,  # newest issue_time -> idxmax would pick it
        available_time_utc=last_decision + pd.Timedelta(minutes=1),
        valid_time_utc=GH + pd.Timedelta(hours=h),
        lead_hours=1, temperature_2m=POISON, dew_point_2m=POISON - 20,
        relative_humidity_2m=50.0, vapour_pressure_deficit=1.0,
        wet_bulb_temperature_2m=POISON, surface_pressure=1000.0,
        wind_u_10m=0.0, wind_v_10m=0.0, wind_speed_10m=0.0,
        wind_gusts_10m=0.0, cloud_cover=0.0, visibility=0.0,
        precipitation=0.0, boundary_layer_height=0.0,
        shortwave_radiation=0.0,
        temperature_850hPa=0.0, temperature_1000hPa=0.0,
        geopotential_height_850hPa=0.0, geopotential_height_1000hPa=0.0,
        wind_speed_850hPa=0.0, wind_direction_850hPa=0.0,
    ) for h in range(-1, 6)]
    fcst = pd.concat([fcst, pd.DataFrame(poison_rows)], ignore_index=True)
    T = assemble_asof_tensor(None, fcst, GH, VENUE, CF_AZ)
    temps = T[:, :, 9]  # dim 9 = temperature_f, unstandardized here
    assert np.abs(temps).max() < 200.0, "poison future forecast leaked into tensor"


def test_availability_boundary_is_inclusive():
    """A forecast available exactly AT the decision instant is usable (<=)."""
    fcst = _hrrr_frame()
    exact = fcst[fcst["available_time_utc"] <= GH]
    row = select_asof_forecast(exact, exact["available_time_utc"].max(),
                               GH + pd.Timedelta(hours=2))
    assert row is not None


def test_planted_future_obs_never_selected():
    """An observation available 1 min after the decision must not appear even
    for an elapsed hour."""
    obs = _obs_frame()
    late = obs.iloc[[0]].copy()  # hour 20:53 report (h=-3, always elapsed)
    late["valid_utc"] = GH - pd.Timedelta(minutes=7)  # inside hour -1
    late["available_time_utc"] = GH + pd.Timedelta(hours=7)  # after every decision
    late["temperature_2m"] = POISON
    obs = pd.concat([obs[obs["valid_utc"] < GH - pd.Timedelta(hours=1)], late],
                    ignore_index=True)
    T = assemble_asof_tensor(obs, None, GH, VENUE, CF_AZ)
    assert np.abs(T[:, 0, OFF_OBS + 9]).max() < 200.0


def test_lead_monotone_nonincreasing(tensor):
    """Living window: lead_norm for a fixed target hour never increases in d
    (while the forecast is populated)."""
    for hi in range(7):
        leads = tensor[:, hi, OFF_LEAD]
        populated = tensor[:, hi, OFF_FCST_MASK] == 1
        prev = np.inf
        for di in range(7):
            if populated[di] and leads[di] > 0:
                assert leads[di] <= prev + 1e-6
                prev = leads[di]


def test_freshest_issue_selected():
    """The value encodes its issue hour; at d=6 (05:00Z) the freshest available
    issue for h=5 (04:00 valid... already past) — use h=5, d=6, freshest issue
    is 03Z (avail 04:15 <= 05:00)."""
    fcst = _hrrr_frame()
    row = select_asof_forecast(fcst, GH + pd.Timedelta(hours=6), GH + pd.Timedelta(hours=5))
    assert row["issue_time_utc"] == pd.Timestamp("2023-07-15 03:00", tz="UTC")


# ── Standardization ───────────────────────────────────────────────────────────
def test_standardize_then_mask_missing_is_exact_zero():
    stats = {k: np.full(N_DIMS if k.startswith("fcst") else N_OBS_DIMS,
                        5.0 if "mean" in k else 2.0, dtype=np.float32)
             for k in ("fcst_mean", "fcst_std", "obs_mean", "obs_std")}
    T = assemble_asof_tensor(None, None, GH, VENUE, CF_AZ, norm_stats=stats)
    assert not T.any(), "fully-missing tensor must be exactly zero everywhere"


def test_standardization_applied_to_populated_entries():
    stats = {"fcst_mean": np.zeros(N_DIMS, np.float32),
             "fcst_std": np.full(N_DIMS, 2.0, np.float32),
             "obs_mean": np.zeros(N_OBS_DIMS, np.float32),
             "obs_std": np.full(N_OBS_DIMS, 2.0, np.float32)}
    raw = assemble_asof_tensor(None, _hrrr_frame(), GH, VENUE, CF_AZ)
    std = assemble_asof_tensor(None, _hrrr_frame(), GH, VENUE, CF_AZ, norm_stats=stats)
    np.testing.assert_allclose(std[:, :, 9], raw[:, :, 9] / 2.0, rtol=1e-5)


# ── Unit conversions / physics interface ─────────────────────────────────────
def test_hrrr_wind_convention_negates_true_components():
    """A pure-westerly true wind (u=+10 m/s eastward) is a 270° FROM wind; the
    repo convention gives u = s*sin(270°) = -s. Sign flip is load-bearing —
    park wind features were validated against Gumbo labels in this convention."""
    df = _hrrr_frame()
    src = pd.DataFrame([dict(
        issue_time_utc=GH, available_time_utc=GH, valid_time_utc=GH, lead_hours=1,
        t2m_k=290.0, d2m_k=280.0, sp_pa=101000.0,
        u10_ms=10.0, v10_ms=0.0, gust_ms=0.0, tcc_pct=0.0, vis_m=0.0,
        apcp_mm=0.0, hpbl_m=0.0, dswrf_wm2=0.0, t850_k=285.0, t1000_k=293.0,
        z850_m=1500.0, z1000_m=100.0, u850_ms=10.0, v850_ms=0.0)])
    out = hrrr_to_era5(src)
    assert out["wind_u_10m"].iloc[0] == pytest.approx(-22.3694, rel=1e-4)
    assert out["wind_v_10m"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert out["wind_direction_850hPa"].iloc[0] == pytest.approx(270.0)


def test_metar_wind_matches_repo_formula():
    """drct=180, 10 kt: u = s*sin(180°)=0, v = s*cos(180°)=-s (mph)."""
    out = _obs_frame()
    assert out["wind_u_10m"].iloc[0] == pytest.approx(0.0, abs=1e-4)
    assert out["wind_v_10m"].iloc[0] == pytest.approx(-11.5078, rel=1e-4)


def test_station_pressure_sea_level_equals_altimeter():
    assert station_pressure_hpa(29.92, 0.0) == pytest.approx(29.92 * 33.8639, rel=1e-6)


def test_station_pressure_denver_much_lower():
    p = station_pressure_hpa(30.00, 1609.0)
    assert 830 < p < 860  # ~840 hPa at Denver elevation


def test_moisture_derivations():
    assert relative_humidity_pct(70.0, 70.0) == pytest.approx(100.0, abs=0.1)
    assert vapour_pressure_deficit_kpa(70.0, 70.0) == pytest.approx(0.0, abs=1e-6)
    # Stull 2011 worked example: T=20°C (68°F), RH=50% -> Tw ~ 13.7°C (56.7°F)
    assert wet_bulb_f(68.0, 50.0) == pytest.approx(13.7 * 9 / 5 + 32, abs=1.0)


def test_metar_sky_cover_max_of_layers():
    df = pd.DataFrame([dict(
        station="X", valid_utc=GH, available_time_utc=GH,
        tmpf=70.0, dwpf=55.0, relh=60.0, drct=0.0, sknt=0.0, gust=np.nan,
        alti=29.92, mslp=np.nan, vsby=10.0,
        skyc1="FEW", skyl1=3000.0, skyc2="BKN", skyl2=8000.0,
        skyc3=None, skyl3=None, p01i=0.05)])
    out = metar_to_era5(df, 0.0)
    assert out["cloud_cover"].iloc[0] == 75.0
    assert out["precipitation"].iloc[0] == pytest.approx(1.27)  # 0.05 in -> mm
    assert out["visibility"].iloc[0] == pytest.approx(16093.4)


def test_metar_missing_gust_means_calm_not_missing():
    out = _obs_frame()
    assert out["wind_gusts_10m"].iloc[0] == pytest.approx(11.5078, rel=1e-4)


# ── D2b obs extras ────────────────────────────────────────────────────────────
def test_wx_extra_features_derivation():
    # heavy thunderstorm rain: thunder=1, intensity=3, liquid, no obstruction
    v = wx_extra_features("+TSRA", 40.0, 15.0)
    assert v.tolist()[:4] == [1.0, 3.0, 0.0, 0.0]
    assert v[4] == pytest.approx(40.0 * 1.15078)
    # light snow with mist: frozen + obstruction, intensity 1
    v = wx_extra_features("-SN BR", None, 18.0)
    assert v.tolist()[:4] == [0.0, 1.0, 1.0, 1.0]
    assert v[4] == pytest.approx(18.0)  # PK WND absent -> snapshot gust
    # haze only (wildfire-smoke proxy): obstruction without precip
    assert wx_extra_features("HZ", None, 0.0).tolist()[:4] == [0.0, 0.0, 0.0, 1.0]
    # empty codes on a present report = real "no significant weather"
    assert wx_extra_features(None, None, 12.0).tolist()[:4] == [0.0, 0.0, 0.0, 0.0]


def test_obs_extras_land_in_tensor_with_mask():
    """A thunderstorm METAR must set the thunder dim (22) at its hour, and the
    extras must be masked-observable at every observed hour."""
    obs = _obs_frame()
    df = obs.copy()
    df["wxcodes"] = "TSRA"  # storms all night — any selected report carries it
    T = assemble_asof_tensor(df, None, GH, VENUE, CF_AZ)
    thunder_dim = OFF_OBS + N_DIMS  # first extra
    assert T[:, :, thunder_dim].max() == 1.0
    extras_mask = T[:, :, OFF_OBS_MASK + N_DIMS:OFF_LEAD]
    phys_mask = T[:, :, OFF_OBS_MASK:OFF_OBS_MASK + 1]
    np.testing.assert_array_equal(extras_mask.any(axis=2), phys_mask.any(axis=2))


def test_peak_gust_nan_fallback_is_zero_not_nan():
    """Both PK WND and the snapshot gust missing (calm/'M'-wind METAR):
    the extra must be finite — found as 2,518 NaNs in the real 2015 artifact
    (channel 70), because float(nan or 0.0) passes NaN through."""
    v = wx_extra_features(None, None, np.nan)
    assert np.isfinite(v).all()
    assert v[4] == 0.0
