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
    METAR_VIS_CEILING_MI,
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
    select_asof_obs,
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
    hurricane-edge gusts and a torrential inch of rain are all real.

    The 60 kt case carries a gust because these fixtures assert one field at a time and
    inherit NaN for the rest: a 60 kt sustained wind reported with no gust group at all is
    a contradiction the archive shows to be corruption, not an extreme (see
    GUST_REPORT_FLOOR_KT), and it is dropped on purpose. The record-book claim being made
    here -- that 60 kt is not too fast to be real -- is unchanged.

    The 95 kt gust and the 3 in/h rain need partners for the same reason. The base fixture
    carries sknt=10, so a bare gust=95 is a factor of 9.5 -- the corrupt-report signature
    GUST_FACTOR_MAX exists to catch, not a windstorm; pairing it with a 40 kt mean keeps
    the claim (95 kt is not too fast to be real) while making the report self-consistent.
    Likewise the base carries vsby=10 SM and no wxcodes, so a bare p01i=3.0 asserts three
    inches of rain in an hour with unlimited visibility and no weather group; adding the
    -RA code and the visibility such rain actually produces preserves the claim (3 in/h is
    not too much rain to be real) without asserting a physically impossible pairing.
    """
    for kw in (dict(tmpf=18.0, dwpf=10.0), dict(tmpf=115.0, dwpf=70.0),
               dict(alti=24.9 + 0.2), dict(alti=31.0), dict(sknt=60.0, gust=75.0),
               dict(sknt=40.0, gust=95.0), dict(p01i=3.0, wxcodes="+RA", vsby=1.0),
               dict(drct=360.0), dict(relh=100.0)):
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


# ── Cross-channel visibility commensurability ─────────────────────────────────
# Measured on the rebuilt season=2015 artifact (118,869 populated fcst entries, 68,999
# obs). HRRR VIS is censored at 60,000 m while a METAR censors at 10 SM = 16,093 m, so
# before this fix the same clear sky produced very different numbers in the two channels:
#
#   channel   std        1 mi fog vs median   above 10 SM
#   fcst      11,330 m   1.23 sigma           47.98%
#   obs        3,310 m   4.38 sigma            0.00%
#   fcst clamped 3,206 m 4.36 sigma            0.00%
#
# Two consequences, both of which the clamp removes. First, nearly half the forecast
# channel's spread is consumed by a distinction with no baseball mechanism behind it --
# there is no way 37 miles of visibility plays differently from 10 -- which pushes a real
# fog event down to 1.23 sigma where the observation channel puts it at 4.38. Z-scoring
# cannot recover this: it is linear, so it leaves the ratio untouched. Second, the whole
# point of the as-of tensor is to let the model compare a forecast against what was
# actually observed, and that comparison is only meaningful if "clear" is the same number
# in both channels.
#
# Nothing informative is lost. Detail above 10 SM has no observational counterpart to be
# compared against (the obs channel is already censored there), and visibility as a proxy
# for dry/clear air is redundant with the dedicated humidity, VPD, cloud-cover and
# air-density dims, all of which measure it directly.
def test_hrrr_visibility_is_censored_at_the_same_ceiling_as_metar():
    src = pd.DataFrame([dict(
        venue_id=VENUE, model="hrrr",
        issue_time_utc=GH, available_time_utc=GH, valid_time_utc=GH, lead_hours=1,
        t2m_k=290.0, d2m_k=280.0, sp_pa=101000.0,
        u10_ms=2.0, v10_ms=1.0, gust_ms=5.0,
        tcc_pct=50.0, vis_m=60000.0, apcp_mm=0.0,
        hpbl_m=800.0, dswrf_wm2=100.0,
        t850_k=285.0, t1000_k=293.0, z850_m=1500.0, z1000_m=100.0,
        u850_ms=5.0, v850_ms=2.0,
    )])
    out = hrrr_to_era5(src)
    assert out["visibility"].iloc[0] == pytest.approx(METAR_VIS_CEILING_MI * MI_TO_M)


def test_hrrr_visibility_below_the_ceiling_is_preserved_exactly():
    """The fog-to-clear range is the whole reason the dim exists; the clamp must not
    touch it."""
    for raw in (0.0, 400.0, 1609.34, 8046.7, 16000.0):
        src = pd.DataFrame([dict(
            venue_id=VENUE, model="hrrr",
            issue_time_utc=GH, available_time_utc=GH, valid_time_utc=GH, lead_hours=1,
            t2m_k=290.0, d2m_k=280.0, sp_pa=101000.0,
            u10_ms=2.0, v10_ms=1.0, gust_ms=5.0,
            tcc_pct=50.0, vis_m=raw, apcp_mm=0.0,
            hpbl_m=800.0, dswrf_wm2=100.0,
            t850_k=285.0, t1000_k=293.0, z850_m=1500.0, z1000_m=100.0,
            u850_ms=5.0, v850_ms=2.0,
        )])
        assert hrrr_to_era5(src)["visibility"].iloc[0] == pytest.approx(raw)


def test_a_clear_sky_is_the_same_number_in_both_channels():
    """Commensurability is the property the model actually consumes: forecast-vs-observed
    is only a meaningful comparison if both censor at the same point. Asserted across the
    converters rather than on a constant, so a change to either one breaks this."""
    hrrr = pd.DataFrame([dict(
        venue_id=VENUE, model="hrrr",
        issue_time_utc=GH, available_time_utc=GH, valid_time_utc=GH, lead_hours=1,
        t2m_k=290.0, d2m_k=280.0, sp_pa=101000.0,
        u10_ms=2.0, v10_ms=1.0, gust_ms=5.0,
        tcc_pct=50.0, vis_m=45000.0, apcp_mm=0.0,
        hpbl_m=800.0, dswrf_wm2=100.0,
        t850_k=285.0, t1000_k=293.0, z850_m=1500.0, z1000_m=100.0,
        u850_ms=5.0, v850_ms=2.0,
    )])
    metar = _one_metar(vsby=25.0)   # "clear" reported past the 10SM ceiling
    assert (hrrr_to_era5(hrrr)["visibility"].iloc[0]
            == pytest.approx(
                metar_to_era5(metar, station_elev_m=6.0)["visibility"].iloc[0]))


# ── The obs mask must be derived from the source, not from the value ───────────
# The impossible-zero correction only rescues dims where 0 cannot occur (density, wet
# bulb, temperature, pressure). Measured over the 1,225,759 converted 2015 reports, these
# dims arrive NaN from an omitted METAR group and _safe() renders each as a legal-looking
# 0.0 that the mask then claims as observed:
#
#   dim(s)  source omitted            rate     what the masked-in 0 asserts   enters as
#   2,3     drct (wind direction)     4.54%    wind exactly perpendicular      ~0 sigma
#   10      sky cover                 3.52%    clear sky                      -1.01 sigma
#   11      vsby (visibility)         1.98%    DENSE FOG                      -4.56 sigma
#   4,5     sknt (wind speed)         1.68%    dead calm                      -1.44 sigma
#   6       dew point -> VPD          0.25%    saturated air                  -1.32 sigma
#   7       dew point -> RH           0.25%    0% relative humidity           -3.02 sigma
#   12      p01i                      0.14%    a dry hour                     ~-0.1 sigma
#
# The sigma column is why this matters: a masked-in 0 is not a harmless neutral value. It
# is z-scored against the dim's mean, so a fabricated visibility 0 reaches the model as a
# 4.56-sigma dense-fog observation. Visibility is the worst case by far -- the archive
# holds 24,237 omitted readings against just 101 genuine 0-visibility reports, so
# fabricated fog outnumbers real fog 240 to 1, in the exact tail the channel was just
# sharpened to detect.
#
# Wind direction is the subtlest: of the 55,686 reports missing drct only 2 are genuinely
# calm, 35,094 have a measured speed with an unreported (variable) direction, and 20,590
# have no wind data at all. The expected-value defence -- that variable winds are light
# (mean 3.90 kt) so 0 is close to the truth -- is exactly the reasoning that lets a
# market-making model price confidently on data nobody observed. Speed stays in dim 4;
# only the direction-dependent dims are disclaimed.
def _obs_frame_with_missing(**overrides):
    """One observed METAR hour, with chosen raw groups omitted."""
    base = dict(vsby=10.0, tmpf=75.0, dwpf=60.0, relh=60.0, drct=180.0, sknt=8.0,
                gust=np.nan, alti=29.92, p01i=0.0, skyc1="FEW", wxcodes=None,
                peak_wind_gust=np.nan)
    base.update(overrides)
    rows = []
    for k in range(8):
        rows.append(dict(station="TST",
                         valid_utc=GH - pd.Timedelta(hours=6) + pd.Timedelta(hours=k),
                         available_time_utc=GH - pd.Timedelta(hours=6)
                         + pd.Timedelta(hours=k), **base))
    return metar_to_era5(pd.DataFrame(rows), station_elev_m=6.0)


def _obs_mask_for(**overrides):
    """-> obs_mask (27,) at a decision/hour where an observation exists."""
    obs = _obs_frame_with_missing(**overrides)
    T = assemble_asof_tensor(obs, None, GH, VENUE, CF_AZ)
    live = T[:, :, OFF_OBS_MASK:OFF_LEAD].reshape(-1, N_OBS_DIMS)
    claimed = live[live.sum(axis=1) > 0]
    assert len(claimed), "fixture produced no observed hour"
    return claimed[0]


def test_a_complete_report_claims_every_observable_dim():
    """The baseline must not regress: a full METAR still populates dims 0-13."""
    m = _obs_mask_for()
    for d in range(14):
        assert m[d] == 1.0, f"dim {d} lost from a complete report"


def test_missing_wind_direction_disclaims_only_the_direction_dims():
    """Speed is still measured, so dim 4 must survive; only the rotated components,
    which need a bearing, are unknown."""
    m = _obs_mask_for(drct=np.nan)
    assert m[2] == 0.0 and m[3] == 0.0
    assert m[4] == 1.0, "wind speed was measured and must not be discarded with direction"


def test_missing_visibility_is_masked_not_reported_as_dense_fog():
    """The 240-to-1 case. A masked-in 0 here reaches the model as -4.56 sigma."""
    assert _obs_mask_for(vsby=np.nan)[11] == 0.0


def test_a_genuine_zero_visibility_report_is_kept():
    """Dense fog really happens (101 reports in the 2015 archive) and is exactly the
    signal the dim exists for, so it must not be masked away with the missing ones."""
    m = _obs_mask_for(vsby=0.0)
    assert m[11] == 1.0


def test_missing_sky_cover_is_masked_not_reported_as_clear():
    assert _obs_mask_for(skyc1=None)[10] == 0.0


def test_missing_wind_speed_is_masked_not_reported_as_calm():
    m = _obs_mask_for(sknt=np.nan)
    assert m[4] == 0.0 and m[5] == 0.0


def test_a_genuine_calm_report_is_kept():
    """00000KT is a real and common observation."""
    m = _obs_mask_for(sknt=0.0, drct=0.0)
    assert m[4] == 1.0


def test_missing_dew_point_disclaims_the_moisture_dims():
    """One omitted group reaches four dims: VPD, RH, wet bulb and both densities."""
    m = _obs_mask_for(dwpf=np.nan, relh=np.nan)
    for d in (0, 1, 6, 7, 8):
        assert m[d] == 0.0, f"dim {d} still claimed without a dew point"
    assert m[9] == 1.0, "temperature was measured and is independent of dew point"


def test_missing_precipitation_is_masked_not_reported_as_dry():
    assert _obs_mask_for(p01i=np.nan)[12] == 0.0


def test_missing_pressure_disclaims_pressure_and_density():
    m = _obs_mask_for(alti=np.nan)
    assert m[13] == 0.0 and m[0] == 0.0 and m[1] == 0.0
    assert m[9] == 1.0


def test_dims_with_no_station_observation_stay_masked():
    """Dims 14-21 have no METAR counterpart at all and must remain 0 regardless."""
    m = _obs_mask_for()
    for d in range(14, N_DIMS):
        assert m[d] == 0.0


def test_every_observable_dim_has_a_declared_source():
    """A dim missing from the table would silently keep the old fabricate-a-zero
    behaviour forever."""
    from mlb_dl.weather_asof import OBS_DIM_SOURCES
    assert set(OBS_DIM_SOURCES) == set(OBS_OBSERVABLE_DIMS)
    for d, cols in OBS_DIM_SOURCES.items():
        assert cols, f"dim {d} declares no source column"


def test_the_extras_are_not_disclaimed_by_absence():
    """Dims 22-26 are the one place where absence IS the reading, so the source rule must
    stop at dim 13. No wxcodes group means no thunder and no precipitation type, which is
    a real report of quiet weather -- masking it out would throw away 77% of the extras
    (only 22.7% of 2015 reports carry a wxcodes group) to guard against nothing."""
    m = _obs_mask_for(wxcodes=None, peak_wind_gust=np.nan)
    for d in range(N_DIMS, N_OBS_DIMS):
        assert m[d] == 1.0, f"extra dim {d} disclaimed although absence is its signal"


def test_an_omitted_gust_group_still_claims_the_gust_dim():
    """The converter fills a missing gust from the mean wind on purpose: a METAR omits the
    group when gusting is under 10 kt over the mean, so absence is a real "no significant
    gust" report. The source rule must not undo that -- it reads wind_gusts_10m after the
    conversion, not the raw gust column."""
    m = _obs_mask_for(gust=np.nan, sknt=8.0)
    assert m[5] == 1.0 and m[4] == 1.0


# --- the same honesty rule on the FORECAST channel -----------------------------
# The forecast branch set fcst_mask = ones and disclaimed only soil (16) and AQI (17-19),
# so an HRRR row with a missing field was claimed in full. Found by building 2021, where
# 2 of 120,834 forecast entries carried surface_pressure = 0 hPa with mask = 1 -- reaching
# the model near -50 sigma. Rare, but the fix costs nothing and the audit gate blocks the
# season build until it is honest.
def _fcst_frame_with_missing(**overrides):
    """One HRRR issue/valid pair, with chosen raw fields omitted."""
    base = dict(venue_id=VENUE, model="hrrr", t2m_k=290.0, d2m_k=280.0, sp_pa=101000.0,
                u10_ms=2.0, v10_ms=1.0, gust_ms=5.0, tcc_pct=50.0, vis_m=16000.0,
                apcp_mm=0.0, hpbl_m=800.0, dswrf_wm2=100.0, t850_k=285.0, t1000_k=293.0,
                z850_m=1500.0, z1000_m=100.0, u850_ms=5.0, v850_ms=2.0)
    base.update(overrides)
    rows = []
    for issue_h in range(14, 30):
        issue = GH.normalize() + pd.Timedelta(hours=issue_h)
        for fxx in range(1, 10):
            rows.append(dict(issue_time_utc=issue,
                             available_time_utc=issue + pd.Timedelta(minutes=75),
                             valid_time_utc=issue + pd.Timedelta(hours=fxx),
                             lead_hours=fxx, **base))
    return hrrr_to_era5(pd.DataFrame(rows))


def _fcst_mask_for(**overrides):
    """-> fcst_mask (22,) at a decision/hour where a forecast exists."""
    fcst = _fcst_frame_with_missing(**overrides)
    T = assemble_asof_tensor(None, fcst, GH, VENUE, CF_AZ)
    live = T[:, :, OFF_FCST_MASK:OFF_OBS].reshape(-1, N_DIMS)
    claimed = live[live.sum(axis=1) > 0]
    assert len(claimed), "fixture produced no forecast hour"
    return claimed[0]


def test_a_complete_forecast_claims_every_dim_it_owns():
    """Baseline: soil and AQI ride other frames and stay disclaimed, but every dim the
    HRRR row itself populates must survive."""
    m = _fcst_mask_for()
    for d in list(range(16)) + [20, 21]:
        assert m[d] == 1.0, f"fcst dim {d} lost from a complete forecast row"


def test_forecast_missing_pressure_disclaims_pressure_and_density():
    """The 2021 defect, reproduced: dims 0, 1 and 13 all need surface pressure, and
    temperature is unaffected."""
    m = _fcst_mask_for(sp_pa=np.nan)
    assert m[13] == 0.0 and m[0] == 0.0 and m[1] == 0.0
    assert m[9] == 1.0, "temperature was forecast and must not be discarded with pressure"


def test_forecast_missing_visibility_is_masked_not_reported_as_dense_fog():
    m = _fcst_mask_for(vis_m=np.nan)
    assert m[11] == 0.0


def test_forecast_missing_wind_disclaims_the_wind_dims():
    m = _fcst_mask_for(u10_ms=np.nan, v10_ms=np.nan)
    assert m[2] == 0.0 and m[3] == 0.0


def test_forecast_missing_pressure_levels_disclaim_lapse_and_shear():
    """Dims 20-21 already coerced an absent pressure level to 0, which is legal weather
    (an isothermal layer, no shear) and so invisible to a range check."""
    m = _fcst_mask_for(t850_k=np.nan, u850_ms=np.nan)
    assert m[20] == 0.0 and m[21] == 0.0


def test_forecast_missing_boundary_layer_and_radiation_are_masked():
    m = _fcst_mask_for(hpbl_m=np.nan, dswrf_wm2=np.nan)
    assert m[14] == 0.0 and m[15] == 0.0


def test_the_obs_table_is_the_forecast_table_restricted_to_observables():
    """One table, two channels: the obs and forecast schemas are both era5, so a dim must
    not be allowed to declare different sources depending on which channel reads it."""
    from mlb_dl.weather_asof import DIM_SOURCES, OBS_DIM_SOURCES
    for d, cols in OBS_DIM_SOURCES.items():
        assert DIM_SOURCES[d] == cols, f"dim {d} declares different sources per channel"


# --- wind reports that contradict themselves -----------------------------------
# METAR_PHYSICAL_LIMITS is deliberately "world records widened", one field at a time, so it
# admits a 150 kt sustained wind and cannot ever catch it. Building 2023 surfaced 15 such
# reports, which then failed the artifact's own dim-4 range gate. Measured over the whole
# 13,708,338-report archive, the entire sustained tail at or above 50 kt is 320 reports and
# 221 of them carry no gust group at all -- 69%, where a real windstorm would essentially
# always report one. Both rules below are cross-field, so neither needs a new threshold
# fitted to the data, and together they cost 0.002% of the archive.
def _wind_report(**overrides):
    base = dict(vsby=10.0, tmpf=75.0, dwpf=60.0, relh=60.0, drct=180.0, sknt=8.0,
                gust=np.nan, alti=29.92, p01i=0.0, skyc1="FEW", wxcodes=None,
                peak_wind_gust=np.nan)
    base.update(overrides)
    return pd.DataFrame([dict(station="TST", valid_utc=GH, available_time_utc=GH, **base)])


def test_a_gust_below_the_sustained_wind_is_dropped():
    """Definitional: a gust is the maximum over the averaging period, so it cannot be less
    than the mean. Real case from 2023 -- 145 kt sustained reported with a 20 kt gust."""
    assert len(metar_to_era5(_wind_report(sknt=145.0, gust=20.0), 6.0)) == 0


def test_a_gust_above_the_sustained_wind_is_kept():
    assert len(metar_to_era5(_wind_report(sknt=25.0, gust=40.0), 6.0)) == 1


def test_a_gust_equal_to_the_sustained_wind_is_kept():
    """Steady wind gusting exactly to the mean is legal, if unusual."""
    assert len(metar_to_era5(_wind_report(sknt=25.0, gust=25.0), 6.0)) == 1


def test_a_high_sustained_wind_with_no_gust_group_is_dropped():
    """A METAR omits the gust group only when the peak is under 10 kt above the mean. The
    3-second-peak-over-2-minute-mean gust factor over open airport terrain is at least
    ~1.2, so a 50 kt mean already implies a >=60 kt peak -- exactly the reporting
    threshold. Above that, silence about gusts contradicts the wind speed itself."""
    assert len(metar_to_era5(_wind_report(sknt=150.0, gust=np.nan), 6.0)) == 0


def test_a_high_sustained_wind_WITH_a_gust_group_is_kept():
    """The rule must not become a wind-speed ceiling: a severe but self-consistent
    windstorm is real weather and has to survive."""
    assert len(metar_to_era5(_wind_report(sknt=60.0, gust=85.0), 6.0)) == 1


def test_a_moderate_wind_with_no_gust_group_is_kept():
    """Below the floor, an absent gust group is the normal case and must not be touched --
    this is the branch that protects the other 13.7M reports."""
    assert len(metar_to_era5(_wind_report(sknt=30.0, gust=np.nan), 6.0)) == 1


def test_the_gust_floor_is_not_applied_to_a_missing_wind_speed():
    """A report with no wind group at all is absence, not contradiction."""
    assert len(metar_to_era5(_wind_report(sknt=np.nan, gust=np.nan), 6.0)) == 1


# --- gusts wildly above their own mean wind -----------------------------------
# The rule above catches a gust BELOW the mean; nothing caught a gust far above it, and
# that is what reached the artifact. GPM 2024-04-28 20:50Z reported sknt=11 with gust=202,
# a factor of 18.4, its neighbouring hours at 16-26 kt and wxcodes/p01i showing clear dry
# air. 202 kt is under the 220 kt Barrow Island world record, so the one-field-at-a-time
# (0,250) bound passes it; the artifact's dim-5 ceiling of 120 mph then failed on it, twice
# (dim 26 too, because an absent PK WND group falls back to the snapshot gust).
# Measured over the whole 13,708,338-report archive the rule costs 67 reports (0.00049%),
# 20 of them at GPM -- the same order as the two rules above.
def test_a_gust_wildly_above_the_sustained_wind_is_dropped():
    """The real GPM 2024-04-28 report: an 11 kt mean cannot peak at 202 kt."""
    assert len(metar_to_era5(_wind_report(sknt=11.0, gust=202.0), 6.0)) == 0


def test_a_severe_but_self_consistent_gust_is_kept():
    """The rule must not become a gust ceiling: a real windstorm gusts hard AND blows hard,
    so its factor stays low. Durst (1960) and ASCE 7-22 put the 3-second peak at ~1.53x
    the mean over open terrain and convective downbursts reach ~3x, all far below the
    cutoff."""
    assert len(metar_to_era5(_wind_report(sknt=55.0, gust=95.0), 6.0)) == 1


def test_the_gust_factor_rule_spares_light_and_variable_wind():
    """The branch that protects the archive. A METAR encodes a gust only at >=10 kt above
    the mean, so sknt=1/gust=11 is the SMALLEST gust it can report and carries a factor of
    11 -- judging light wind by ratio would delete valid reports wholesale. The absolute
    floor is what makes the ratio safe to apply, and 11 kt is harmless anyway: it is far
    inside the artifact's own 120 mph bound."""
    assert len(metar_to_era5(_wind_report(sknt=1.0, gust=11.0), 6.0)) == 1


def test_the_gust_factor_rule_spares_a_calm_mean_with_a_modest_gust():
    """sknt=0 makes the ratio undefined; a modest gust over calm air is a real, common
    report and must not be dropped by a divide-by-zero landing on infinity."""
    assert len(metar_to_era5(_wind_report(sknt=0.0, gust=15.0), 6.0)) == 1


def test_the_gust_factor_rule_is_not_applied_without_a_mean_wind():
    """No mean wind to compare against is absence, not contradiction -- same principle as
    the missing-gust-group case above."""
    assert len(metar_to_era5(_wind_report(sknt=np.nan, gust=70.0), 6.0)) == 1


# --- rain that nothing else in the report corroborates -------------------------
# MCF (MacDill AFB, the backup station for venue 12) suffered an episodic gauge fault in
# 2018/2019/2021: a cluster parked at exactly 0.80 in/h plus excursions to 24.00, reported
# across consecutive hours with wxcodes silent and vsby at 10 SM. Pooled over 12 seasons
# its p01i looks healthy (p99.9 = 0.85 in), so this is not a unit error and not a station
# to blocklist -- eight of its seasons are clean. The (0,12) in/h bound catches only the
# loudest excursions, so 6.78 survived and the 25.4 in->mm conversion turned it into
# 172 mm/h in the 2018 artifact, failing the dim-12 range gate.
def test_torrential_rain_with_no_weather_group_and_clear_visibility_is_dropped():
    """The real MCF 2018-09-24 report. 0.75 in/h is 19 mm/h, which the Marshall-Palmer
    rate/visibility relation puts near 0.6 SM -- 10 SM and a silent wxcodes field cannot
    both be true of it."""
    assert len(metar_to_era5(
        _wind_report(p01i=6.78, wxcodes=None, vsby=10.0), 6.0)) == 0


def test_torrential_rain_that_reports_a_weather_group_is_kept():
    """Real heavy rain codes itself. The rule needs BOTH corroborations absent, so a
    thunderstorm downpour survives however hard it rains."""
    assert len(metar_to_era5(
        _wind_report(p01i=6.78, wxcodes="+TSRA", vsby=1.0), 6.0)) == 1


def test_torrential_rain_that_cuts_visibility_is_kept():
    """Visibility alone is enough to corroborate: some automated sites report the rate
    without a present-weather group, and dropping those would lose real rain."""
    assert len(metar_to_era5(
        _wind_report(p01i=6.78, wxcodes=None, vsby=0.5), 6.0)) == 1


def test_trace_rain_with_no_weather_group_is_kept():
    """The branch that protects the archive: trace accumulation legitimately carries no
    code -- 0.01 in with a silent wxcodes field is the single most common nonzero reading
    at healthy stations, so the threshold has to sit far above it."""
    assert len(metar_to_era5(
        _wind_report(p01i=0.05, wxcodes=None, vsby=10.0), 6.0)) == 1


def test_the_precip_rule_is_not_applied_without_a_visibility_reading():
    """No visibility to contradict is absence, not contradiction."""
    assert len(metar_to_era5(
        _wind_report(p01i=6.78, wxcodes=None, vsby=np.nan), 6.0)) == 1


# ── Station-role preference ───────────────────────────────────────────────────
# The station map ranks by distance (primary_km < backup_km), so the primary is the more
# representative sensor for the venue. Concatenating primary+backup exists to degrade
# per-hour when the primary goes dark (the DMH 2021 case), NOT to let the backup preempt a
# healthy primary. But ASOS files at a FIXED minute past the hour, so a backup at :56
# outranked a primary at :53 in every hour of every season: measured 28.6% of all 2018
# selections, and 98.8% at Chase Field, where the winner was 24.5 km away and 123 m higher
# than the 5.3 km primary — straight into surface pressure and air density.

def _two_station_frame(primary_min=53, backup_min=56, primary_avail_min=10,
                       backup_avail_min=10, drop_primary=False):
    """One elapsed hour, both stations reporting, distinguishable by temperature."""
    rows = []
    base = GH.normalize() + pd.Timedelta(hours=20)
    specs = [] if drop_primary else [("primary", "SPG", primary_min, primary_avail_min, 70.0)]
    specs.append(("backup", "MCF", backup_min, backup_avail_min, 90.0))
    for role, st, minute, avail, tmpf in specs:
        valid = base + pd.Timedelta(minutes=minute)
        rows.append(dict(
            station=st, station_role=role, valid_utc=valid,
            available_time_utc=valid + pd.Timedelta(minutes=avail),
            tmpf=tmpf, dwpf=55.0, relh=np.nan, drct=180.0, sknt=10.0, gust=np.nan,
            alti=29.92, mslp=np.nan, vsby=10.0, skyc1="SCT", skyl1=5000.0,
            skyc2=None, skyl2=None, skyc3=None, skyl3=None, p01i=0.0,
        ))
    df = metar_to_era5(pd.DataFrame(rows), station_elev_m=6.0)
    df["station_role"] = [r[0] for r in specs]
    return df


def _select(df, decision_offset_h=3):
    return select_asof_obs(df, GH.normalize() + pd.Timedelta(hours=20 + decision_offset_h),
                           GH.normalize() + pd.Timedelta(hours=20))


def test_the_nearer_primary_station_wins_over_a_later_filing_backup():
    """The real Tropicana case: SPG at :53 (2.5 km) must beat MCF at :56 (15.9 km)."""
    row = _select(_two_station_frame())
    assert row is not None and row["station_role"] == "primary"


def test_a_dark_primary_still_degrades_to_the_backup():
    """The documented reason the frames are concatenated at all (DMH 2021)."""
    row = _select(_two_station_frame(drop_primary=True))
    assert row is not None and row["station_role"] == "backup"


def test_role_preference_applies_after_the_availability_filter_not_before():
    """A primary that has not disseminated yet is not a legal choice.

    Preferring the primary BEFORE the availability filter would resurrect exactly the
    leakage this module exists to remove: at the decision hour the primary's report does
    not yet exist, so the backup is the only lawful answer.
    """
    # Primary lands 10 h late; backup disseminates in 2 min, so at the 21:00Z decision the
    # backup exists and the primary does not.
    df = _two_station_frame(primary_avail_min=600, backup_avail_min=2)
    row = _select(df, decision_offset_h=1)
    assert row is not None and row["station_role"] == "backup"


def test_recency_still_decides_within_the_chosen_station():
    """Preferring the primary must not also discard the primary's freshest report."""
    df = _two_station_frame()
    extra = df[df["station_role"] == "primary"].copy()
    extra["valid_utc"] = extra["valid_utc"] + pd.Timedelta(minutes=-30)
    extra["temperature_2m"] = -99.0
    row = _select(pd.concat([df, extra], ignore_index=True))
    assert row["station_role"] == "primary" and row["temperature_2m"] != -99.0


def test_selection_is_unchanged_when_no_role_column_is_present():
    """Backwards compatibility: callers that never tagged roles keep pure-recency."""
    # metar_to_era5 does not carry `station` through, so identify the winner by its
    # reporting minute: :56 is the backup, and pure recency must still pick it.
    df = _two_station_frame().drop(columns=["station_role"])
    row = _select(df)
    assert row is not None and row["valid_utc"].minute == 56


def test_the_live_retrieval_path_tags_station_roles_too(monkeypatch):
    """Role preference is opt-in per frame, so tagging on one side alone is a silent skew.

    select_asof_obs prefers the primary only when the column exists. The training builder
    tags it in load_obs_for_venues; if fetch_live_asof does not, live keeps resolving by
    recency while train resolves by role, and the two paths disagree on which sensor
    represents the venue. Nothing else catches it: assemble_asof_tensor is shared, so every
    physics test still passes, and the parity replay needs a populated feature store.
    """
    import mlb_dl.weather_asof as wx

    raw_by_station = {}
    for st, minute, tmpf in (("SPG", 53, 70.0), ("MCF", 56, 90.0)):
        valid = GH.normalize() + pd.Timedelta(hours=20, minutes=minute)
        raw_by_station[st] = pd.DataFrame([dict(
            station=st, valid_utc=valid,
            available_time_utc=valid + pd.Timedelta(minutes=10),
            tmpf=tmpf, dwpf=55.0, relh=np.nan, drct=180.0, sknt=10.0, gust=np.nan,
            alti=29.92, mslp=np.nan, vsby=10.0, skyc1="SCT", skyl1=5000.0,
            skyc2=None, skyl2=None, skyc3=None, skyl3=None, p01i=0.0,
        )])

    def fake_read(key: str):
        # Only the obs prefix is served: no live HRRR and no soil, which is a legal live
        # state (fcst stays None) and keeps the test on the obs retrieval layer.
        for st, raw in raw_by_station.items():
            if f"station={st}/" in key and wx.OBS_LIVE_PREFIX in key:
                return raw
        return None

    seen = {}

    def spy_assemble(obs, fcst, *a, **k):
        seen["obs"] = obs
        return np.zeros((N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS), dtype=np.float32)

    monkeypatch.setattr(wx, "_read_parquet_s3", fake_read)
    monkeypatch.setattr(wx, "assemble_asof_tensor", spy_assemble)

    out = wx.fetch_live_asof(
        VENUE, GH, CF_AZ, now=GH + pd.Timedelta(hours=6),
        station_map={str(VENUE): dict(primary_station="SPG", backup_station="MCF",
                                      primary_elev_m=6.0, backup_elev_m=6.0)},
        norm_stats={"fcst_mean": np.zeros(N_DIMS, np.float32),
                    "fcst_std": np.ones(N_DIMS, np.float32),
                    "obs_mean": np.zeros(N_OBS_DIMS, np.float32),
                    "obs_std": np.ones(N_OBS_DIMS, np.float32)})

    assert out is not None
    obs = seen["obs"]
    assert "station_role" in obs.columns, "live obs frame reached the assembler untagged"
    assert set(obs["station_role"]) == {"primary", "backup"}
    # And the tag actually changes the outcome on the frame the live path built.
    row = select_asof_obs(obs, GH + pd.Timedelta(hours=3),
                          GH.normalize() + pd.Timedelta(hours=20))
    assert row is not None and row["station_role"] == "primary"
