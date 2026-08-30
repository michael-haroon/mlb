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
    MI_TO_M,
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


# ── Physically impossible raw METAR values ────────────────────────────────────
# Swept over the whole raw asos_obs archive (2026-08-30, 1,079 station-season files,
# 13.6M reports). Every column the tensor consumes carries values that are not
# measurements:
#   tmpf  66 out of range, observed [-80, 149] F
#   dwpf   6 out of range, observed min -268.6 F
#   sknt   5 out of range, max 910 kt (1,047 mph)
#   gust   7 out of range, max 525 kt
#   alti 319 out of range, observed [0, 99.99] inHg -- 0 and 99.99 are missing-data
#         sentinels; 99.99 renders as 3,386 hPa and, unlike 0, is NOT caught by the
#         impossible-zero mask correction
#   p01i  44 out of range, max 24 in/h (world record hourly is ~12 in)
# relh, drct and peak_wind_gust swept clean. mslp and skyl* are out of bounds too
# (221 and 1,711) but neither is read anywhere in the as-of path, so they cannot reach
# a tensor and are deliberately not filtered.
#
# Why DROP the report rather than NaN the field. _safe() in weather_context renders NaN
# as 0.0 and there is no per-field obs mask, so NaN-ing one column silently fabricates a
# real reading in the dims where zero is legitimate -- a NaN dew point becomes 0% RH and
# a zero VPD (saturated air), both masked in as measured. Dropping the report is the only
# representation the existing mask machinery renders honestly: select_asof_obs falls
# through to the next report for that hour, or the hour is left genuinely unobserved with
# every dim masked 0. The cost is 447 reports of 13.6M (0.003%).
#
# vsby is deliberately handled the other way, by clamping: "10SM" means "10 or more", so
# the reading saturates and the excess carries no information. Dropping its 95,547
# out-of-ceiling reports (0.71%) would forfeit real coverage for nothing.
IMPOSSIBLE_RAW = [
    ("tmpf", 149.0), ("tmpf", -80.0), ("dwpf", -268.6), ("sknt", 910.0),
    ("gust", 525.0), ("alti", 99.99), ("alti", 0.0), ("p01i", 24.0),
]


def _one_metar(**overrides):
    row = dict(station="BOS", valid_utc=GH, available_time_utc=GH,
               tmpf=70.0, dwpf=55.0, relh=60.0, drct=180.0, sknt=10.0, gust=np.nan,
               alti=29.92, mslp=np.nan, vsby=10.0,
               skyc1="SCT", skyl1=5000.0, skyc2=None, skyl2=None,
               skyc3=None, skyl3=None, p01i=0.0)
    row.update(overrides)
    return pd.DataFrame([row])


@pytest.mark.parametrize("col,value", IMPOSSIBLE_RAW)
def test_a_report_with_an_impossible_value_is_dropped(col, value):
    """Measured values, not invented ones -- each pair is an extreme actually present
    in the raw archive."""
    out = metar_to_era5(_one_metar(**{col: value}), station_elev_m=6.0)
    assert len(out) == 0, f"{col}={value} survived into the era5 frame"


def test_a_clean_report_is_never_dropped():
    out = metar_to_era5(_one_metar(), station_elev_m=6.0)
    assert len(out) == 1


def test_a_merely_missing_field_does_not_drop_the_report():
    """A METAR omits the groups it has nothing to report; absence is normal and must
    not be confused with corruption, or coverage would collapse."""
    for col in ("tmpf", "dwpf", "relh", "drct", "sknt", "gust", "alti", "p01i", "vsby"):
        out = metar_to_era5(_one_metar(**{col: np.nan}), station_elev_m=6.0)
        assert len(out) == 1, f"{col}=NaN wrongly dropped the report"


def test_real_weather_extremes_are_never_dropped():
    """The filter must clear the record book. The coldest MLB game on record is ~18 F
    and the hottest ~115 F; Coors Field altimeter runs far below sea-level values;
    hurricane-edge gusts and a torrential inch of rain are all real."""
    for kw in (dict(tmpf=18.0, dwpf=10.0), dict(tmpf=115.0, dwpf=70.0),
               dict(alti=24.9 + 0.2), dict(alti=31.0), dict(sknt=60.0),
               dict(gust=95.0), dict(p01i=3.0), dict(drct=360.0), dict(relh=100.0)):
        out = metar_to_era5(_one_metar(**kw), station_elev_m=6.0)
        assert len(out) == 1, f"{kw} wrongly dropped"


def test_dropping_preserves_the_surviving_rows_intact():
    """A filter that reindexed or reordered would silently corrupt the as-of selection,
    which matches reports to hours by valid_utc."""
    df = pd.concat([_one_metar(tmpf=70.0), _one_metar(tmpf=149.0), _one_metar(tmpf=72.0)],
                   ignore_index=True)
    df["valid_utc"] = [GH, GH + pd.Timedelta(hours=1), GH + pd.Timedelta(hours=2)]
    out = metar_to_era5(df, station_elev_m=6.0)
    assert len(out) == 2
    np.testing.assert_allclose(out["temperature_2m"].to_numpy(), [70.0, 72.0])
    assert out["valid_utc"].tolist() == [GH, GH + pd.Timedelta(hours=2)]


def test_an_impossible_altimeter_never_reaches_the_pressure_dim():
    """alti=99.99 is the case the impossible-zero mask correction cannot catch: it
    renders 3,386 hPa, which is a number, not a zero, so the mask would call it
    measured. End-to-end through the tensor, not just the frame.
    """
    rows = []
    for hh in range(20, 31):
        valid = GH.normalize() + pd.Timedelta(hours=hh, minutes=53)
        rows.append(dict(station="BOS", valid_utc=valid,
                         available_time_utc=valid + pd.Timedelta(minutes=10),
                         tmpf=70.0, dwpf=55.0, relh=np.nan, drct=180.0, sknt=10.0,
                         gust=np.nan, alti=99.99, mslp=np.nan, vsby=10.0,
                         skyc1="SCT", skyl1=5000.0, skyc2=None, skyl2=None,
                         skyc3=None, skyl3=None, p01i=0.0))
    T = assemble_asof_tensor(metar_to_era5(pd.DataFrame(rows), 6.0), _hrrr_frame(),
                             GH, VENUE, CF_AZ)
    pressure = T[:, :, OFF_OBS + 13]
    mask = T[:, :, OFF_OBS_MASK + 13]
    assert not (mask == 1.0).any(), "claimed a measured pressure from a sentinel altimeter"
    assert np.abs(pressure).max() == 0.0


def test_metar_visibility_is_capped_at_the_reporting_ceiling():
    """Raw ASOS vsby carries values that are not measurements.

    Measured on the raw 2015 asos_obs feed (2026-08-30, 1,225,768 reports): 1.45% of
    reports exceed 10 SM, the 99.99th percentile is 70 SM, and the MAXIMUM is 34,006 SM
    -- 54,700 km, further than the circumference of the Earth. The offending stations are
    the non-US ones in the venue map: CYQG (Windsor, serving Comerica Park), MMMX/MMMY/
    MMTO (Mexico City, Monterrey, Toluca -- all real MLB series venues) and BJC near
    Coors. 70 SM reached the built 2015 tensor as 112,700 m.

    A "10SM" METAR means "10 statute miles OR MORE", so the sensor's information
    saturates at the ceiling and everything above it means the same thing: clear. Nothing
    meteorologically relevant to a baseball -- fog, haze, precipitation -- lives above
    10 SM, so clamping keeps every bit of real signal while removing an outlier that
    z-scores to tens of sigma against a ~15 km mean.

    Clamping rather than masking is the weaker, safer claim: a corrupt-high parse can
    only have come from a large raw token, so "at least 10 SM" is still true of it,
    whereas masking would discard a genuine clear-air observation.
    """
    df = pd.DataFrame([dict(
        station="CYQG", valid_utc=GH, available_time_utc=GH,
        tmpf=70.0, dwpf=55.0, relh=60.0, drct=0.0, sknt=0.0, gust=np.nan,
        alti=29.92, mslp=np.nan, vsby=v,
        skyc1="CLR", skyl1=np.nan, skyc2=None, skyl2=None, skyc3=None, skyl3=None,
        p01i=0.0) for v in (10.0, 15.0, 70.0, 34006.0)])
    out = metar_to_era5(df, 190.0)
    ceiling = 10.0 * MI_TO_M
    assert out["visibility"].max() == pytest.approx(ceiling)
    # The in-range report must be untouched, not merely under the cap.
    assert out["visibility"].iloc[0] == pytest.approx(ceiling)
    assert (out["visibility"] <= ceiling + 1e-6).all()


def test_metar_visibility_below_the_ceiling_is_preserved_exactly():
    """The clamp must not flatten the range that carries the signal: fog, mist and
    light-rain visibilities are the whole reason the dim exists."""
    df = pd.DataFrame([dict(
        station="BOS", valid_utc=GH, available_time_utc=GH,
        tmpf=70.0, dwpf=68.0, relh=90.0, drct=0.0, sknt=0.0, gust=np.nan,
        alti=29.92, mslp=np.nan, vsby=v,
        skyc1="OVC", skyl1=200.0, skyc2=None, skyl2=None, skyc3=None, skyl3=None,
        p01i=0.0) for v in (0.0, 0.25, 1.5, 7.0)])
    out = metar_to_era5(df, 6.0)
    np.testing.assert_allclose(out["visibility"].to_numpy(),
                               np.array([0.0, 0.25, 1.5, 7.0]) * MI_TO_M, rtol=1e-6)


def test_metar_missing_visibility_stays_missing_not_clamped():
    """A clamp implemented with np.minimum would turn NaN into NaN (fine) but one
    implemented with a comparison could turn it into the ceiling, inventing a clear-air
    reading for a report that had none."""
    df = pd.DataFrame([dict(
        station="BOS", valid_utc=GH, available_time_utc=GH,
        tmpf=70.0, dwpf=55.0, relh=60.0, drct=0.0, sknt=0.0, gust=np.nan,
        alti=29.92, mslp=np.nan, vsby=np.nan,
        skyc1="CLR", skyl1=np.nan, skyc2=None, skyl2=None, skyc3=None, skyl3=None,
        p01i=0.0)])
    out = metar_to_era5(df, 6.0)
    v = out["visibility"].iloc[0]
    assert not (v == pytest.approx(10.0 * MI_TO_M)), "invented a clear-air reading"


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


# ── Obs mask must not claim fields the report omitted ─────────────────────────
# Measured on weather_asof/season=2015.parquet (2026-08-30): 111 of 120,785 rows
# (0.092%), across 16 of 2,465 games, carry obs_mask=1 over a value of exactly 0
# in a dim where zero is physically impossible -- air_density (36), density_ratio
# (36), wet_bulb_f (103), temperature_f (27), surface_pressure (10). Every
# pressure-zero row is also a density-zero row (density is derived FROM pressure),
# which identifies the cause: a METAR that omits its altimeter group.
#
# The count is small but the consequence is not proportional to it. The artifact
# stores raw units and the loader z-scores later, so a masked-in 0 hPa against a
# mean near 1000 hPa becomes roughly a -50 sigma activation -- whereas an honestly
# masked entry is exactly 0 (the mean) and contributes nothing. Outliers that
# large distort gradients far beyond their frequency.
#
# assemble_asof_tensor already states this invariant for the forecast channel
# ("the mask must not claim dims the source cannot populate", dims 16-19); these
# tests extend the same rule to obs, which took no such correction.
IMPOSSIBLE_ZERO_DIMS = [0, 1, 8, 9, 13]   # density, density_ratio, wet_bulb, temp, pressure


def _obs_frame_with(**overrides):
    """_obs_frame's hourly METAR series with a field overridden in EVERY report.

    It has to be the full 20:00Z-06:00Z series, not one row: select_asof_obs looks
    for the report representing [hour_start, hour_start+1h) with the window opened
    an hour early, so a lone report outside that span selects nothing and every
    mask assertion would pass vacuously on an empty obs channel.
    """
    rows = []
    for hh in range(20, 31):
        valid = GH.normalize() + pd.Timedelta(hours=hh, minutes=53)
        row = dict(station="BOS", valid_utc=valid,
                   available_time_utc=valid + pd.Timedelta(minutes=10),
                   tmpf=70.0, dwpf=55.0, relh=np.nan, drct=180.0, sknt=10.0,
                   gust=np.nan, alti=29.92, mslp=np.nan, vsby=10.0,
                   skyc1="SCT", skyl1=5000.0, skyc2=None, skyl2=None,
                   skyc3=None, skyl3=None, p01i=0.0)
        row.update(overrides)
        rows.append(row)
    return metar_to_era5(pd.DataFrame(rows), station_elev_m=6.0)


def _obs_row(obs):
    """The (d=6, h=-1) cell: h < d, so obs is eligible and must be populated."""
    T = assemble_asof_tensor(obs, _hrrr_frame(), GH, VENUE, CF_AZ)
    return T[6, 0, OFF_OBS:OFF_OBS_MASK], T[6, 0, OFF_OBS_MASK:OFF_LEAD]


def test_metar_missing_altimeter_does_not_claim_a_measured_pressure():
    """The measured defect: no altimeter group -> pressure and the densities
    derived from it collapse to 0, but the mask still advertised them."""
    vec, mask = _obs_row(_obs_frame_with(alti=np.nan))
    for d in (0, 1, 13):
        assert mask[d] == 0.0, f"dim {d} claims measured, value={vec[d]}"
    # Fields the same report DID carry must survive: this must not mask the row.
    assert mask[9] == 1.0 and vec[9] != 0.0, "temperature was reported"
    assert mask[4] == 1.0, "wind speed was reported"


def test_no_impossible_zero_is_ever_masked_in():
    """Whatever the cause, a dim where zero cannot occur must never be presented
    as a real measurement -- that is the invariant, independent of which METAR
    field went missing."""
    for miss in ({"alti": np.nan}, {"tmpf": np.nan}, {"dwpf": np.nan},
                 {"tmpf": np.nan, "alti": np.nan}):
        vec, mask = _obs_row(_obs_frame_with(**miss))
        for d in IMPOSSIBLE_ZERO_DIMS:
            assert not (mask[d] == 1.0 and vec[d] == 0.0), (miss, d)


def test_calm_wind_and_clear_sky_stay_real_measurements():
    """The over-masking guard. Zero is a legitimate reading for wind, gusts,
    cloud and precipitation, so the fix must not sweep those into 'missing' --
    a calm, clear, dry evening is a fully observed evening."""
    vec, mask = _obs_row(_obs_frame_with(sknt=0.0, drct=0.0, skyc1="CLR", p01i=0.0))
    assert vec[4] == 0.0 and mask[4] == 1.0, "calm wind is measured, not missing"
    assert vec[10] == 0.0 and mask[10] == 1.0, "clear sky is measured, not missing"
    assert vec[12] == 0.0 and mask[12] == 1.0, "no precip is measured, not missing"


def test_a_fully_populated_report_masks_in_every_observable_dim():
    """Baseline: the fix must not cost coverage on a healthy report."""
    vec, mask = _obs_row(_obs_frame_with())
    assert mask[OBS_OBSERVABLE_DIMS].all(), "healthy METAR lost coverage"
    assert np.isfinite(vec).all()


def test_obs_channel_never_carries_nonfinite_values():
    """NaN would propagate through the whole network, and a mask could not rescue it
    because the channel is z*mask and NaN*0 is still NaN.

    This is a property tripwire, not a guard on defensive code: the feature layer
    renders every absent METAR group as 0.0 rather than NaN, so assemble_asof_tensor
    carries no non-finite branch (an explicit one was written, found unreachable by
    mutation testing, and removed rather than shipped untested). The loud gate for
    this property at season scale is the np.isfinite check in
    verify_weather_asof_artifact.py, which runs over all 99 channels.
    """
    for miss in ({"alti": np.nan}, {"tmpf": np.nan}, {"vsby": np.nan}, {"p01i": np.nan}):
        vec, _ = _obs_row(_obs_frame_with(**miss))
        assert np.isfinite(vec).all(), miss
