"""Adversarial checks on the derived-market pricer in scripts/kalshi_multi_target_backtest.py.

The pricer replaces a truncated 2-D joint grid with closed-form NegBin CDF identities, so the thing
worth testing is that those identities agree with brute force *including* in the regimes where a
grid would fail: near-zero dispersion (alpha=0.1 gives variance mu + 10*mu^2), strikes already
settled by the score on the board, and strikes far outside the support.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import nbinom

SPEC = importlib.util.spec_from_file_location(
    "bt", Path(__file__).resolve().parents[1] / "kalshi_multi_target_backtest.py")
bt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bt)

# (mu_h, alpha_h, mu_a, alpha_a) spanning the observed readout range: mu in [0.01, 13.2],
# alpha in [0.1, 20.1]. The alpha=0.1 rows are the heavy-tail stress cases.
PARAMS = [
    (4.5, 2.0, 4.2, 1.8),      # typical pregame
    (0.01, 0.1, 0.01, 0.1),    # degenerate: game effectively over
    (13.2, 0.1, 12.0, 0.1),    # heaviest tail seen in the readout
    (2.4, 20.1, 2.3, 11.3),    # near-Poisson
    (0.5, 0.9, 8.0, 3.0),      # asymmetric
]
BRUTE = 4000  # brute-force support cap; P(R > 4000) is < 1e-300 for every row in PARAMS


@pytest.fixture(scope="module")
def tables():
    mu_h, a_h, mu_a, a_a = (np.array(x) for x in zip(*PARAMS))
    return bt.RunTables(mu_h, a_h, mu_a, a_a)


def _marg(mu, alpha, k):
    return nbinom.pmf(k, alpha, alpha / (alpha + mu))


def _brute_team(mu, alpha, base, strike):
    k = np.arange(BRUTE + 1)
    return _marg(mu, alpha, k)[base + k > strike].sum()


def _brute_pair(row, base, strike, op):
    """P(base + op(Rh, Ra) > strike) by explicit 2-D summation."""
    mu_h, a_h, mu_a, a_a = PARAMS[row]
    k = np.arange(BRUTE + 1)
    ph, pa = _marg(mu_h, a_h, k), _marg(mu_a, a_a, k)
    joint = np.outer(ph, pa)
    val = base + op(k[:, None], k[None, :])
    return joint[val > strike].sum()


@pytest.mark.parametrize("row", range(len(PARAMS)))
@pytest.mark.parametrize("base,strike", [(0, 4.5), (3, 4.5), (7, 4.5), (0, 0.5), (0, 29.5)])
def test_team_total_matches_brute_force(tables, row, base, strike):
    for is_home, (mu, alpha) in ((True, PARAMS[row][:2]), (False, PARAMS[row][2:])):
        got, err = tables.team_total_over(np.array([row]), np.array([float(base)]),
                                          np.array([strike]), np.array([is_home]))
        assert err[0] == 0.0, "closed-form branch must claim zero truncation"
        assert got[0] == pytest.approx(_brute_team(mu, alpha, base, strike), abs=1e-12)


@pytest.mark.parametrize("row", range(len(PARAMS)))
@pytest.mark.parametrize("base,strike", [(0, 8.5), (5, 8.5), (12, 8.5), (0, 0.5), (0, 29.5)])
def test_total_matches_brute_force(tables, row, base, strike):
    """The total identity is claimed EXACT, so it is held to float tolerance, not to a bound."""
    got, err = tables.total_over(np.array([row]), np.array([float(base)]), np.array([strike]))
    assert err[0] == 0.0
    assert got[0] == pytest.approx(_brute_pair(row, base, strike, lambda h, a: h + a), abs=1e-10)


@pytest.mark.parametrize("row", range(len(PARAMS)))
@pytest.mark.parametrize("margin,strike", [(0, 1.5), (-3, 1.5), (4, 1.5), (0, 7.5), (0, 29.5)])
@pytest.mark.parametrize("is_home", [True, False])
def test_spread_matches_brute_force_within_its_own_bound(tables, row, margin, strike, is_home):
    """Spread truncates, so the contract is: |price - truth| <= the bound the pricer reports.

    Asserting against the reported bound rather than a hand-picked tolerance is what makes the
    bound trustworthy -- if it were ever optimistic, this test fails.
    """
    got, err = tables.spread_over(np.array([row]), np.array([float(margin)]),
                                  np.array([strike]), np.array([is_home]))
    op = (lambda h, a: h - a) if is_home else (lambda h, a: a - h)
    truth = _brute_pair(row, margin, strike, op)
    assert got[0] <= truth + 1e-12, "truncated sum must be a lower bound on the truth"
    assert truth - got[0] <= err[0] + 1e-12, f"residual {truth - got[0]:.3e} exceeds bound {err[0]:.3e}"


def test_settled_strikes_are_exactly_certain(tables):
    """A strike the score on the board has already cleared must price 1.0, not 1-epsilon.

    Two ways this breaks: floor(strike - base) < 0 must short-circuit to 1 instead of gathering
    SF[0], and the total branch must add back SF_a(MAX_REM) -- without that term the price was
    1 - P(Ra > 80) = 0.953 on the heaviest-tail row, which is how the truncation bug surfaced.
    """
    n = len(PARAMS)
    rows = np.arange(n)
    assert tables.team_total_over(rows, np.full(n, 10.0), np.full(n, 4.5),
                                  np.ones(n, bool))[0] == pytest.approx(1.0)
    assert tables.total_over(rows, np.full(n, 20.0), np.full(n, 8.5))[0] == pytest.approx(1.0)
    # A spread is NEVER settled by the margin alone -- the opponent can always erase it -- so the
    # only certain spread is one where no runs remain. PARAMS[1] is that degenerate row.
    dead = np.array([1])
    assert tables.spread_over(dead, np.array([9.0]), np.array([1.5]),
                              np.array([True]))[0][0] == pytest.approx(1.0, abs=1e-6)
    assert tables.spread_over(rows, np.full(n, 9.0), np.full(n, 1.5),
                              np.ones(n, bool))[0].max() <= 1.0
    # and the dead-game row must give ~0 for an unreachable strike
    assert tables.total_over(dead, np.array([0.0]), np.array([8.5]))[0][0] < 1e-9


def test_probabilities_stay_in_unit_interval_and_are_monotone(tables):
    """Higher strike must never be more likely. Guards a sign error in the marginalization."""
    n = len(PARAMS)
    rows, base = np.arange(n), np.zeros(n)
    prev_t = prev_s = np.ones(n)
    for strike in np.arange(0.5, 29.5, 1.0):
        tot, _ = tables.total_over(rows, base, np.full(n, strike))
        spr, _ = tables.spread_over(rows, base, np.full(n, strike), np.ones(n, bool))
        assert ((tot >= -1e-12) & (tot <= 1 + 1e-12)).all()
        assert ((spr >= -1e-12) & (spr <= 1 + 1e-12)).all()
        assert (tot <= prev_t + 1e-12).all()
        assert (spr <= prev_s + 1e-12).all()
        prev_t, prev_s = tot, spr


def test_spread_complement_is_consistent(tables):
    """P(Rh >= Ra) + P(Ra >= Rh) - P(tie) == 1.

    "margin > -0.5" on an integer margin means margin >= 0, so BOTH sides include the tie and the
    identity subtracts it once. Getting this backwards was the first version of this test.
    """
    n = len(PARAMS)
    rows, base = np.arange(n), np.zeros(n)
    k = np.arange(4001)
    h, eh = tables.spread_over(rows, base, np.full(n, -0.5), np.ones(n, bool))
    a, ea = tables.spread_over(rows, base, np.full(n, -0.5), np.zeros(n, bool))
    ties = np.array([
        (_marg(PARAMS[r][0], PARAMS[r][1], k) * _marg(PARAMS[r][2], PARAMS[r][3], k)).sum()
        for r in rows])
    assert (h + a - ties) == pytest.approx(1.0, abs=(eh + ea).max() + 1e-9)


TICKERS = {
    # 2025 form: no HHMM, strike suffix is the YES team
    "KXMLBGAME-25APR16ATHCWS-CWS": ("2025-04-16", "ATHCWS", 1, "CWS"),
    # doubleheader game 2 carries a trailing digit on the pair
    "KXMLBGAME-25APR18ATHMIL2-ATH": ("2025-04-18", "ATHMIL", 2, "ATH"),
    # 2026 form: HHMM inserted before the pair
    "KXMLBTOTAL-26AUG311940MILCHC-18": ("2026-08-31", "MILCHC", 1, "18"),
    "KXMLBSPREAD-26AUG312138NYYLAA-LAA8": ("2026-08-31", "NYYLAA", 1, "LAA8"),
    "KXMLBTEAMTOTAL-26AUG311805SFATL-SF8": ("2026-08-31", "SFATL", 1, "SF8"),
    # KXMLBRFI has no strike suffix at all
    "KXMLBRFI-25OCT17TORSEA": ("2025-10-17", "TORSEA", 1, None),
    # legacy Arizona code must normalize to the feature store's AZ
    "KXMLBTOTAL-26JUL041940ARISD-9": ("2026-07-04", "AZSD", 1, "9"),
}


def test_ticker_parsing_covers_every_observed_form():
    import pandas as pd
    s = pd.Series(list(TICKERS))
    got = bt.parse_tickers(s)
    for i, (tk, (d, pair, gn, suf)) in enumerate(TICKERS.items()):
        assert got.tkd.iloc[i] == pd.Timestamp(d), tk
        assert got.pair.iloc[i] == pair, tk
        assert got.game_number.iloc[i] == gn, tk
        assert (got.suffix.iloc[i] is None or pd.isna(got.suffix.iloc[i])) if suf is None \
            else got.suffix.iloc[i] == suf, tk

