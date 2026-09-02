"""Adversarial checks on scripts/kalshi_topbook_accuracy.py.

The pricer itself is covered by test_kalshi_multi_pricer.py; both scripts call the same functions, so
duplicating those cases here would only prove that imports work. What is new -- and what can silently
produce a fake edge -- is everything between the candle file and the three-way split:

  1. the as-of join must never see a candle that closed AFTER the model state (that is the leak),
  2. an absent book side reads 0.0000 in the feed and must not be treated as a 0c price,
  3. `take_yes` / `take_no` must be strict and disjoint, and paid/won/gap must be the arithmetic of a
     real Kalshi order (buy YES at the ask, buy NO at 1-bid) rather than of a mid,
  4. the shared orientation helpers must flip BOTH the fair and the settlement together -- flipping
     one without the other looks like enormous skill.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bt = _load("kalshi_multi_target_backtest")
tb = _load("kalshi_topbook_accuracy")


def _obs(ts, f, y=1.0, ticker="T"):
    ts = np.atleast_1d(ts).astype(float)
    return pd.DataFrame({"ticker": ticker, "ts": ts,
                         "f": np.broadcast_to(np.atleast_1d(f).astype(float), ts.shape),
                         "y": np.broadcast_to(np.atleast_1d(y).astype(float), ts.shape)})


def _q(qts, bid, ask, ticker="T"):
    return pd.DataFrame({"ticker": ticker, "qts": np.atleast_1d(qts).astype(float),
                         "bid": np.atleast_1d(bid).astype(float),
                         "ask": np.atleast_1d(ask).astype(float)})


# ----------------------------------------------------------------------------------------------
# 1. the as-of join
# ----------------------------------------------------------------------------------------------
def test_asof_never_uses_a_candle_that_closed_after_the_state():
    """This is the leak test. end_period_ts is the candle CLOSE, so <= T is the only legal match."""
    q = _q([100.0, 160.0, 220.0], [0.40, 0.60, 0.80], [0.42, 0.62, 0.82])
    obs = tb.attach_book(_obs([159.0, 160.0, 161.0, 500.0], 0.5), q, max_stale=1e9)
    assert obs.qts.tolist() == [100.0, 160.0, 160.0, 220.0]
    assert (obs.qts <= obs.ts).all()
    assert obs.stale.tolist() == [59.0, 0.0, 1.0, 280.0]


def test_state_before_the_first_quote_is_dropped_not_backfilled():
    """No quote existed yet, so there is nothing to be more accurate than. Forward-filling from a
    later candle would be the leak in its most direct form."""
    obs = tb.attach_book(_obs([50.0, 150.0], 0.5), _q([100.0], [0.4], [0.5]), max_stale=1e9)
    assert obs.ts.tolist() == [150.0]


def test_max_stale_drops_quotes_nobody_was_making():
    q = _q([0.0], [0.30], [0.70])
    assert len(tb.attach_book(_obs([1700.0], 0.5), q, max_stale=1800)) == 1
    assert tb.attach_book(_obs([1900.0], 0.5), q, max_stale=1800).empty


def test_the_join_does_not_cross_tickers():
    """by=ticker must isolate contracts; a strike must never inherit another strike's book."""
    q = pd.concat([_q([100.0], [0.10], [0.12], "A"), _q([100.0], [0.80], [0.82], "B")])
    obs = pd.concat([_obs([200.0], 0.5, ticker="A"), _obs([200.0], 0.5, ticker="B")])
    got = tb.attach_book(obs, q, max_stale=1e9).set_index("ticker")
    assert got.loc["A", "bid"] == 0.10 and got.loc["B", "bid"] == 0.80


# ----------------------------------------------------------------------------------------------
# 2. what counts as a book
# ----------------------------------------------------------------------------------------------
def test_empty_and_crossed_book_sides_are_rejected(tmp_path):
    """A 0.0000 bid means NO BID. Pricing against it would hand the model a ~50c fake edge on every
    illiquid contract, which is exactly the sort of thing that would be reported as alpha."""
    rows = [
        ("T", 100, "0.4000", "0.4200"),   # good
        ("T", 160, "0.0000", "0.4200"),   # no bid
        ("T", 220, "0.4000", "0.0000"),   # no ask
        ("T", 280, "0.5000", "0.4500"),   # crossed (OHLC artifact within the minute)
        ("T", 340, "0.4000", "1.0100"),   # impossible ask
        ("OTHER", 100, "0.4000", "0.4200"),   # a different game: outside the window
    ]
    d = tmp_path / "KXTEST"
    d.mkdir()
    pd.DataFrame(rows, columns=["market_ticker", "end_period_ts",
                                "yes_bid_close_dollars", "yes_ask_close_dollars"]).to_parquet(
        d / "candlesticks_batch_0.parquet")
    q, stats = tb.load_quotes(dict(dirs=[d]), keep={"T"}, listed={"T", "OTHER"})
    assert q.qts.tolist() == [100.0]
    assert stats == dict(files=1, unreadable=0, candles=6, in_window=5, one_sided=2, crossed=1,
                         two_sided=1, tickers_seen=2, tickers_unlisted=0)


def test_a_quoted_ticker_absent_from_the_markets_files_is_counted(tmp_path):
    """No markets row means no strike and no `result`, so the ticker cannot be priced or
    cross-checked. It must still be COUNTED separately from the out-of-window majority, otherwise a
    systematic hole in the markets archive looks identical to a date filter working correctly."""
    d = tmp_path / "KXTEST"
    d.mkdir()
    pd.DataFrame({"market_ticker": ["T", "GHOST"], "end_period_ts": [100, 100],
                  "yes_bid_close_dollars": ["0.40", "0.40"],
                  "yes_ask_close_dollars": ["0.42", "0.42"]}).to_parquet(
        d / "candlesticks_batch_0.parquet")
    _, stats = tb.load_quotes(dict(dirs=[d]), keep={"T"}, listed={"T"})
    assert stats["tickers_seen"] == 2 and stats["tickers_unlisted"] == 1


def test_both_candle_writer_generations_are_read(tmp_path):
    """kalshi_history interleaves a bare-named generation with a `_dollars`-suffixed one, and 931 of
    938 KXMLBGAME candle files are the BARE kind -- reading only the suffixed name would drop the one
    series with a full season and report "no quotes" instead of failing. Both already hold dollars in
    [0,1], so the loader must alias the names WITHOUT rescaling."""
    d = tmp_path / "KXTEST"
    d.mkdir()
    pd.DataFrame({"market_ticker": ["T"], "end_period_ts": [100],
                  "yes_bid_close": ["0.40"], "yes_ask_close": ["0.42"],
                  "price_close": ["0.41"]}).to_parquet(d / "candlesticks_batch_0.parquet")
    pd.DataFrame({"market_ticker": ["T"], "end_period_ts": [200],
                  "yes_bid_close_dollars": ["0.60"], "yes_ask_close_dollars": ["0.62"],
                  "price_close_dollars": ["0.61"]}).to_parquet(d / "candlesticks_batch_1.parquet")
    q, stats = tb.load_quotes(dict(dirs=[d]), keep={"T"}, listed={"T"})
    assert stats["unreadable"] == 0 and stats["two_sided"] == 2
    assert q.sort_values("qts").bid.tolist() == [0.40, 0.60]


def test_a_candle_file_with_no_recognized_bid_ask_column_is_counted_not_skipped_silently(tmp_path):
    d = tmp_path / "KXTEST"
    d.mkdir()
    pd.DataFrame({"market_ticker": ["T"], "end_period_ts": [100],
                  "price_close": ["0.41"]}).to_parquet(d / "candlesticks_batch_0.parquet")
    q, stats = tb.load_quotes(dict(dirs=[d]), keep={"T"}, listed={"T"})
    assert q.empty and stats["unreadable"] == 1 and stats["candles"] == 0


def test_no_candle_files_is_reported_not_crashed():
    q, stats = tb.load_quotes(dict(dirs=[Path("/nonexistent")]), keep=set(), listed=set())
    assert q.empty and stats["files"] == 0 and stats["candles"] == 0


# ----------------------------------------------------------------------------------------------
# 3. the three-way split -- the user's actual question
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("f,expect", [
    (0.70, "yes"),      # above the ask: YES is cheap
    (0.62, "none"),     # AT the ask: not cheaper, so no view (strict inequality)
    (0.50, "none"),     # inside
    (0.40, "none"),     # AT the bid
    (0.20, "no"),       # below the bid: NO is cheap
])
def test_split_is_strict_and_disjoint(f, expect):
    obs = tb.attach_book(_obs([200.0], f), _q([100.0], [0.40], [0.62]), max_stale=1e9)
    r = obs.iloc[0]
    assert [r.take_yes, r.take_no, r.no_view].count(True) == 1
    assert {"yes": r.take_yes, "no": r.take_no, "none": r.no_view}[expect]


def test_buying_yes_pays_the_ask_and_buying_no_pays_one_minus_bid():
    """Kalshi has no short. Selling YES is buying NO, and its cost is 1-bid, not the bid."""
    q = _q([100.0], [0.40], [0.62])
    hit = tb.attach_book(_obs([200.0], 0.90, y=1.0), q, max_stale=1e9).iloc[0]
    assert hit.take_yes and hit.paid == pytest.approx(0.62)
    assert hit.won == 1.0 and hit.gap == pytest.approx(1.0 - 0.62)

    lift = tb.attach_book(_obs([200.0], 0.10, y=1.0), q, max_stale=1e9).iloc[0]
    assert lift.take_no and lift.paid == pytest.approx(1.0 - 0.40)
    assert lift.won == 0.0 and lift.gap == pytest.approx(0.40 - 1.0)


def test_gap_is_the_settlement_minus_the_price_actually_payable():
    """gap must be computed off the executable side, never off the mid. On a 40/62 book a mid-based
    gap would be 11c kinder to the model on a YES than the truth."""
    rng = np.random.default_rng(0)
    n = 500
    ts = 200.0 + np.arange(n)
    bid = rng.uniform(0.01, 0.90, n).round(2)
    ask = (bid + rng.uniform(0.01, 0.09, n)).round(2)
    q = _q(ts - 10, bid, ask)
    obs = tb.attach_book(_obs(ts, rng.uniform(0, 1, n), rng.integers(0, 2, n).astype(float)),
                         q, max_stale=1e9)
    y, g = obs.y.to_numpy(), obs.gap.to_numpy()
    yes, no = obs.take_yes.to_numpy(), obs.take_no.to_numpy()
    assert np.allclose(g[yes], y[yes] - obs.ask.to_numpy()[yes])
    assert np.allclose(g[no], obs.bid.to_numpy()[no] - y[no])
    assert np.isnan(g[obs.no_view.to_numpy()]).all()
    # the gap is bounded by a contract's payoff regardless of the model
    assert np.nanmin(g) >= -1.0 and np.nanmax(g) <= 1.0


def test_mid_and_width_are_the_book_not_the_trade():
    obs = tb.attach_book(_obs([200.0], 0.5), _q([100.0], [0.15], [0.79]), max_stale=1e9).iloc[0]
    assert obs.mid == pytest.approx(0.47) and obs.width == pytest.approx(0.64)


# ----------------------------------------------------------------------------------------------
# 4. shared orientation helpers (used by BOTH scripts, so a break here is a double break)
# ----------------------------------------------------------------------------------------------
HEAD_CFG = dict(prob="p_home_win", y="y_home_win", side_from_suffix=True)


def _head(suffix, home="NYY", away="BOS", p=0.70, y=1.0):
    n = len(suffix)
    return pd.DataFrame({"p_home_win": [p] * n, "y_home_win": [y] * n, "suffix": suffix,
                         "home_team_abbr": [home] * n, "away_team_abbr": [away] * n})


def test_price_head_flips_fair_and_settlement_together():
    """An away-side ticker's YES is the away team. Flipping f without y (or vice versa) turns a
    calibrated model into either a perfect one or an anti-model -- both loud, neither detectable if
    the two flips are not tested jointly."""
    f, y, keep = bt.price_head(HEAD_CFG, _head(["NYY", "BOS"]))
    assert keep.all()
    assert f == pytest.approx([0.70, 0.30]) and y.tolist() == [1.0, 0.0]
    # a series without a side suffix must not be flipped at all
    f2, y2, keep2 = bt.price_head(dict(HEAD_CFG, side_from_suffix=False), _head(["NYY", "BOS"]))
    assert keep2.all() and f2 == pytest.approx([0.70, 0.70]) and y2.tolist() == [1.0, 1.0]


def test_price_head_normalizes_the_legacy_arizona_code():
    """Kalshi ticker suffixes say ARI; the feature store says AZ. Before CODE_FIX was applied to the
    SUFFIX (it was only applied to the team PAIR), an away-ARI ticker matched neither abbreviation
    and fell through to the home orientation -- so its fair AND its settlement were both inverted.
    Caught in production by the Kalshi `result` cross-check: 22 of 4,264 KXMLBGAME tickers, every one
    of them ARI."""
    f, y, keep = bt.price_head(HEAD_CFG, _head(["ARI"], home="SD", away="AZ"))
    assert keep.all(), "an away-ARI ticker must resolve, not be dropped"
    assert f == pytest.approx([0.30]) and y.tolist() == [0.0], "away ticker must be flipped"
    f2, _, _ = bt.price_head(HEAD_CFG, _head(["ARI"], home="AZ", away="SD"))
    assert f2 == pytest.approx([0.70]), "home-ARI ticker must NOT be flipped"


def test_price_head_refuses_to_guess_a_suffix_that_names_neither_team():
    """Silently defaulting to `home` is how the ARI bug survived. An unresolvable suffix must be
    reported as dropped, not priced against an arbitrary side."""
    _, _, keep = bt.price_head(HEAD_CFG, _head(["NYY", "XXX", "BOS"]))
    assert keep.tolist() == [True, False, True]


def test_strike_side_drops_suffixes_that_name_neither_team():
    """A suffix we cannot resolve is a parse failure. Defaulting it to `away` would price half the
    unparseable rows correctly by luck and the other half backwards."""
    t = pd.DataFrame({"suffix": ["NYY9", "BOS9", "XXX9"],
                      "home_team_abbr": ["NYY"] * 3, "away_team_abbr": ["BOS"] * 3})
    is_home, keep = bt.strike_side(t)
    assert keep.tolist() == [True, True, False]
    assert is_home.tolist() == [True, False]


# ----------------------------------------------------------------------------------------------
# 5. the settlement cross-check must be able to fail
# ----------------------------------------------------------------------------------------------
def test_settlement_check_counts_tickers_not_observations():
    """One busy ticker must not outvote fifty quiet ones, and a real disagreement must show up."""
    obs = pd.DataFrame({
        "ticker": ["A"] * 50 + ["B"],
        "y": [1.0] * 50 + [1.0],
        "result": ["yes"] * 50 + ["no"],
    })
    n, agree, bad = tb.settlement_check(obs)
    assert (n, agree, bad) == (2, 0.5, ["B"])


def test_settlement_check_ignores_unsettled_and_missing_results():
    obs = pd.DataFrame({"ticker": ["A", "B", "C"], "y": [1.0, 0.0, 1.0],
                        "result": ["yes", "", None]})
    assert tb.settlement_check(obs) == (1, 1.0, [])
    assert tb.settlement_check(obs.drop(columns=["result"])) is None
