"""The season build must refuse to write an artifact from an incomplete archive.

Proven defect (2026-08-30): weather_asof/season=2015.parquet was written at 06:39:20Z
while 77 of the season's 200 HRRR date files did not yet exist -- the extraction was
still running. load_hrrr_for_dates logged one warning per missing date and the build
wrote anyway, producing an artifact that looked entirely normal:

    ok    2015: artifact covers all 2465 population games
    FAIL  2015: NaN/Inf present
    FAIL  2015: fcst coverage 59.3% ... fcst below 90%
    FAIL  2015: game 415601 recomputation mismatch (1882 entries)

123/200 present is 61.5%, matching the 59.3% measured forecast coverage. Nothing in the
pipeline objected; only the dedicated artifact audit caught it, and only because someone
ran it. A stale artifact is especially dangerous here because it is silently CORRECT in
shape and coverage-of-games -- it trains fine and simply carries less weather signal
than the A/B arm it is being compared against, which would quietly invalidate the
experiment rather than fail it.

The chain script now gates on coverage before building, which covers chain-driven
builds. This guard covers the case that actually caused the damage: a build invoked
directly, with no gate in front of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deep_learning"))

from mlb_dl import build_weather_asof as bwa  # noqa: E402


def _dates(n: int, start: str = "2015-04-06") -> list[pd.Timestamp]:
    return [pd.Timestamp(start) + pd.Timedelta(days=i) for i in range(n)]


def test_guard_rejects_the_real_2015_shortfall():
    """The exact case that shipped: 77 of 200 dates absent."""
    dates = _dates(200)
    with pytest.raises(SystemExit) as e:
        bwa.assert_fcst_dates_complete(2015, dates, dates[:77])
    msg = str(e.value)
    assert "2015" in msg
    assert "77" in msg


def test_guard_allows_the_known_legitimate_absences():
    """A complete season still misses a few dates for reasons no rerun can fix:

      * genuine era gaps -- the GRIB was never published (1 date in 2015, 2 in 2017)
      * out-of-domain venues -- the Tokyo and Seoul openers, which HRRR cannot cover
        at any lead (2 dates each in 2019, 2024, 2025)

    Worst observed case is 4 of ~200 dates, so the guard must tolerate that or it would
    block every affected season permanently.
    """
    dates = _dates(200)
    bwa.assert_fcst_dates_complete(2019, dates, dates[:4])  # must not raise


def test_guard_floor_has_headroom_over_the_observed_legitimate_gap():
    """Pins the constant against both failure directions.

    The floor must sit above the worst legitimate absence rate (4/200 = 2%) and far
    below the shortfall that shipped (77/200 = 38.5%).
    """
    assert 0.02 < (1.0 - bwa.MIN_FCST_DATE_COVERAGE) < 0.385


def test_guard_reports_which_dates_are_missing():
    """The operator needs the dates, not just a percentage -- the fix is a targeted
    rerun of exactly those ranges."""
    dates = _dates(200)
    with pytest.raises(SystemExit) as e:
        bwa.assert_fcst_dates_complete(2015, dates, dates[:77])
    assert "2015-04-06" in str(e.value)


def test_guard_rejects_a_totally_empty_archive():
    dates = _dates(50)
    with pytest.raises(SystemExit):
        bwa.assert_fcst_dates_complete(2020, dates, list(dates))


def test_empty_date_list_does_not_divide_by_zero():
    bwa.assert_fcst_dates_complete(2015, [], [])  # must not raise


def test_loader_reports_missing_dates_through_the_out_param(monkeypatch):
    """load_hrrr_for_dates must surface absences, not just log them.

    Logging was the whole problem: 77 warnings scrolled past and the build continued.
    """
    dates = _dates(3)

    def fake_read(key, columns=None):
        if "2015-04-07" in key:
            raise FileNotFoundError(key)
        return pd.DataFrame({"venue_id": [1]})

    monkeypatch.setattr(bwa, "_read_parquet", fake_read)
    monkeypatch.setattr(bwa, "hrrr_to_era5_with_soil_placeholder", lambda df: df)
    missing: list = []
    bwa.load_hrrr_for_dates(dates, missing_out=missing)
    assert [f"{d:%Y-%m-%d}" for d in missing] == ["2015-04-07"]


def test_loader_signature_stays_backward_compatible(monkeypatch):
    """The two verifier scripts call this positionally with one argument; changing the
    return type would break them."""
    monkeypatch.setattr(bwa, "_read_parquet",
                        lambda key, columns=None: pd.DataFrame({"venue_id": [1]}))
    monkeypatch.setattr(bwa, "hrrr_to_era5_with_soil_placeholder", lambda df: df)
    out = bwa.load_hrrr_for_dates(_dates(2))
    assert isinstance(out, pd.DataFrame), "callers expect a DataFrame, not a tuple"


def test_build_season_actually_invokes_the_guard():
    """A guard that exists but is never called is worse than none."""
    import inspect
    src = inspect.getsource(bwa.build_season)
    assert "assert_fcst_dates_complete" in src, (
        "build_season does not call the completeness guard"
    )
    # and it must be called BEFORE the write, not after
    assert (src.index("assert_fcst_dates_complete")
            < src.index("_write_parquet")), "guard runs after the artifact is written"
