"""The artifact gate must notice values that are not weather.

Why this check exists at all: the existing audit in verify_weather_asof_artifact.py is
entirely structural (mask geometry, leakage, monotonicity, coverage) plus a spot
recomputation. The recomputation cannot catch a unit error or a dishonest mask, because
it re-runs assemble_asof_tensor -- the same function that produced the artifact -- so it
is bit-identical BY CONSTRUCTION and would happily confirm a tensor full of Kelvin.
That blind spot is not hypothetical: the obs mask-honesty defect (a METAR missing its
altimeter group stored surface_pressure = 0 hPa with mask = 1) passed the full structural
audit on season=2015 and was only found by measuring per-dim ranges by hand.

What these bounds are, and are not. They are OUTER plausibility tripwires derived from
physics and from the record book, deliberately loose enough that no real playable-game
weather can trip them. They catch order-of-magnitude faults -- Kelvin for Fahrenheit,
Pascals for hectopascals, a mask claiming a dim the source never populated.

Two faults they deliberately do NOT catch, stated so the gate is not mistaken for more
than it is. First, a same-magnitude unit error: m/s served as mph is a factor of 2.24, so
a 30 mph wind reads 13.4 mph and sits inside any honest bound. Second, a fraction served
as a percent: catching an all-0.6 humidity channel would need a lower bound above 0.6,
but desert-venue games really do report single-digit RH, so such a bound would fail real
data -- and a check that fails real data gets switched off, which is worse than no check.
Both faults are the job of cross-source agreement (verify_weather_archives.py `cross`),
not of a range test on a single artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_curation" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deep_learning"))

import verify_weather_asof_artifact as v  # noqa: E402
from mlb_dl.weather_asof import (  # noqa: E402
    ASOF_CHANNELS,
    IMPOSSIBLE_ZERO_OBS_DIMS,
    N_DECISIONS,
    N_DIMS,
    N_OBS_DIMS,
    N_TARGET_HOURS,
    OFF_FCST,
    OFF_FCST_MASK,
    OFF_LEAD,
    OFF_OBS,
    OFF_OBS_MASK,
)

# A mid-range, entirely unremarkable summer evening in raw artifact units, indexed by
# the 22 physics dims then the 5 obs extras. Every value sits well inside its bound, so
# any failure reported against this vector is the check misfiring, not the data.
PLAUSIBLE = np.array([
    1.15,    # 0  air_density kg/m3
    0.94,    # 1  air_density_ratio
    6.0,     # 2  wind_toward_cf mph (signed)
    -3.0,    # 3  wind_crossfield mph (signed)
    8.0,     # 4  wind_speed mph
    14.0,    # 5  wind_gusts mph
    1.2,     # 6  vpd kPa
    62.0,    # 7  humidity %
    64.0,    # 8  wet_bulb_f
    78.0,    # 9  temperature_f
    40.0,    # 10 cloud_cover %
    16093.0, # 11 visibility m
    0.0,     # 12 precip mm (a dry hour is a real measurement)
    1013.0,  # 13 surface_pressure hPa
    900.0,   # 14 boundary_layer_height m
    250.0,   # 15 shortwave_radiation W/m2
    0.22,    # 16 soil_moisture m3/m3
    45.0,    # 17 us_aqi
    9.0,     # 18 pm2_5 ug/m3
    60.0,    # 19 ozone ug/m3
    6.5,     # 20 lapse_rate_1000_850 K/km
    12.0,    # 21 wind_shear_sfc_850
    0.0,     # 22 wx_thunder
    0.0,     # 23 wx_precip_intensity
    0.0,     # 24 wx_frozen_precip
    0.0,     # 25 wx_obstruction
    14.0,    # 26 wx_peak_gust mph
], dtype=np.float32)


def clean_tensor(n_games: int = 2) -> np.ndarray:
    """A fully populated raw-unit tensor. Mask geometry is irrelevant to a range check,
    so everything is masked in -- that is the strictest case, since a range check only
    ever inspects populated entries."""
    T = np.zeros((n_games, N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS), np.float32)
    T[..., OFF_FCST:OFF_FCST_MASK] = PLAUSIBLE[:N_DIMS]
    T[..., OFF_FCST_MASK:OFF_OBS] = 1.0
    T[..., OFF_OBS:OFF_OBS_MASK] = PLAUSIBLE
    T[..., OFF_OBS_MASK:OFF_LEAD] = 1.0
    T[..., OFF_LEAD] = 0.5
    return T


def _fails_for(T):
    return v.check_physical_ranges(T)


# --- the clean baseline must be silent ---------------------------------------
def test_ordinary_weather_reports_nothing():
    assert _fails_for(clean_tensor()) == []


def test_bounds_cover_every_dim_the_tensor_carries():
    """A missing entry would silently exempt that dim from the gate forever."""
    assert len(v.PHYSICAL_RANGES) == N_OBS_DIMS
    for i, (lo, hi) in enumerate(v.PHYSICAL_RANGES):
        assert lo < hi, i


# --- order-of-magnitude unit faults ------------------------------------------
def test_kelvin_served_as_fahrenheit_is_caught():
    """The single most likely regression: hrrr_to_era5 stops converting t2m_k."""
    T = clean_tensor()
    T[..., OFF_FCST + 9] = 298.0          # 298 K = 77 F, plausible-looking number
    fails = _fails_for(T)
    assert any("temperature_f" in f for f in fails), fails


def test_pascals_served_as_hectopascals_is_caught():
    T = clean_tensor()
    T[..., OFF_FCST + 13] = 101325.0
    assert any("surface_pressure" in f for f in _fails_for(T))


def test_kilopascals_served_as_hectopascals_is_caught():
    """The lower bounds carry real weight, not just symmetry. Pressure arrives from two
    converters (INHG_TO_HPA for METAR altimeter, Pa/100 for HRRR sp_pa), so an off-by-100
    lands at ~101 kPa rather than ~1013 hPa -- a fault that only a floor can see, since
    101 is a perfectly ordinary-looking number.
    """
    T = clean_tensor()
    T[..., OFF_FCST + 13] = 101.3
    assert any("surface_pressure" in f for f in _fails_for(T))


def test_density_computed_from_pascals_collapses_and_is_caught():
    """ρ = p/(R_d·T_v) with p left in Pa is 100x too small, so the density dims fall far
    below their floor while staying positive and finite."""
    T = clean_tensor()
    T[..., OFF_OBS + 0] = 0.0116
    T[..., OFF_OBS + 1] = 0.0095
    fails = _fails_for(T)
    assert any("air_density" in f for f in fails), fails


def test_humidity_above_100_percent_is_caught():
    """Relative humidity is definitionally bounded above; anything past it means the
    dim is not RH. The lower end is deliberately left at 0 -- see the module docstring
    on why a fraction-vs-percent floor cannot be set honestly."""
    T = clean_tensor()
    T[..., OFF_FCST + 7] = 140.0
    assert any("humidity" in f for f in _fails_for(T))


def test_cloud_cover_above_100_percent_is_caught():
    T = clean_tensor()
    T[..., OFF_OBS + 10] = 800.0
    assert any("cloud_cover" in f for f in _fails_for(T))


# --- the defect that motivated the check -------------------------------------
def test_masked_in_impossible_zero_is_caught():
    """The measured 2015 defect. A range check alone cannot see it: 0 is inside no
    bound we could set for pressure, but the entry is only wrong BECAUSE the mask
    claims it. This is checked against the mask, not the bounds.
    """
    T = clean_tensor()
    T[0, 6, 0, OFF_OBS + 13] = 0.0        # pressure absent, mask still 1
    fails = _fails_for(T)
    assert any("impossible" in f.lower() for f in fails), fails


def test_every_impossible_zero_dim_is_actually_watched():
    for d in IMPOSSIBLE_ZERO_OBS_DIMS:
        T = clean_tensor()
        T[0, 6, 0, OFF_OBS + d] = 0.0
        assert any("impossible" in f.lower() for f in _fails_for(T)), d


def test_an_honestly_masked_zero_passes():
    """The fix's own output must not be reported as a failure: value 0 with mask 0 is
    the correct representation of a METAR that omitted the group."""
    T = clean_tensor()
    T[0, 6, 0, OFF_OBS + 13] = 0.0
    T[0, 6, 0, OFF_OBS_MASK + 13] = 0.0
    assert _fails_for(T) == []


def test_the_forecast_channel_is_held_to_the_same_rule():
    """HRRR always carries surface pressure, so a masked-in 0 there means the field
    was dropped in the merge -- the fcst channel needs the check as much as obs."""
    T = clean_tensor()
    T[0, 3, 4, OFF_FCST + 13] = 0.0
    assert any("impossible" in f.lower() for f in _fails_for(T))


# --- the check must not fire on genuine zeros or genuine negatives -----------
def test_calm_clear_dry_weather_is_not_flagged():
    """Zero is a real reading for wind, gusts, cloud, visibility and precipitation. A
    check that swept these into 'suspicious' would fail every calm night and get
    switched off, which is worse than not having it."""
    T = clean_tensor()
    for d in (2, 3, 4, 5, 10, 11, 12):
        T[..., OFF_FCST + d] = 0.0
        T[..., OFF_OBS + d] = 0.0
    assert _fails_for(T) == []


def test_saturated_air_is_not_flagged():
    """RH 100% with VPD 0 and wet bulb equal to dry bulb: a rain-delay evening."""
    T = clean_tensor()
    T[..., OFF_FCST + 6] = 0.0
    T[..., OFF_FCST + 7] = 100.0
    T[..., OFF_FCST + 8] = T[..., OFF_FCST + 9]
    assert _fails_for(T) == []


def test_temperature_inversion_keeps_its_negative_lapse_rate():
    """Lapse rate is signed and routinely negative in an evening inversion; a >= 0
    bound here would fail a large share of real night games."""
    T = clean_tensor()
    T[..., OFF_FCST + 20] = -8.0
    assert _fails_for(T) == []


def test_signed_wind_components_may_be_negative():
    T = clean_tensor()
    T[..., OFF_FCST + 2] = -25.0
    T[..., OFF_FCST + 3] = -18.0
    assert _fails_for(T) == []


def test_record_extremes_stay_inside_the_bounds():
    """The bounds must clear the record book, or the gate fails on real games.
    Coors Field station pressure, the coldest and hottest MLB games on record, and a
    gale-force gust are all legal weather.
    """
    T = clean_tensor()
    T[..., OFF_FCST + 13] = 826.0    # measured 2015 minimum, Coors Field
    T[..., OFF_FCST + 9] = 18.0      # coldest MLB game on record
    T[..., OFF_FCST + 8] = 16.0      # wet bulb <= dry bulb
    T[..., OFF_FCST + 5] = 70.0      # gust, well past any playable sustained wind
    assert _fails_for(T) == []
    T[..., OFF_FCST + 9] = 115.0     # hottest MLB game on record
    T[..., OFF_FCST + 8] = 80.0
    assert _fails_for(T) == []


def test_obs_extras_are_bounded_to_their_codebook():
    """wx_precip_intensity is an ordinal 0-3 and the three flags are indicators; a
    value outside that is a wx_extra_features regression, not weather."""
    T = clean_tensor()
    T[..., OFF_OBS + 23] = 7.0
    assert any("precip_intensity" in f for f in _fails_for(T))
    T = clean_tensor()
    T[..., OFF_OBS + 22] = 5.0
    assert any("thunder" in f for f in _fails_for(T))


def test_a_fully_masked_tensor_reports_nothing():
    """Not-yet-populated (d, h) cells are exactly zero everywhere by design, and a
    range check that inspected them would flag every single one."""
    T = np.zeros((2, N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS), np.float32)
    assert _fails_for(T) == []


def test_failure_messages_name_the_dim_and_show_a_value():
    """A gate that reports 'range violation' costs an hour of bisecting a 27 GB
    artifact; it has to say which channel and how bad."""
    T = clean_tensor()
    T[..., OFF_FCST + 9] = 298.0
    msg = _fails_for(T)[0]
    assert "fcst" in msg and "temperature_f" in msg
    assert "298" in msg
