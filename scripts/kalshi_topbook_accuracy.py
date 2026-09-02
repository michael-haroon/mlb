#!/usr/bin/env python3
"""Is the DL model more accurate than the Kalshi TOP OF BOOK? No fill model, no PnL.

WHY THIS INSTEAD OF A PnL. A maker's PnL is decided by queue position at the top of book, and the
book cannot be reconstructed from trade prints and 1-minute candles. Any fill simulation therefore
invents its own answer. What the archive CAN support is an accuracy question with a hard, executable
interpretation, so that is all this script asks.

WHY THE BOOK, NOT THE MID. A mid is a number nobody can transact at, and on these series it is often
a fiction: derived-market books run tens of cents wide. The candles carry the real thing --
yes_bid_close and yes_ask_close, the best bid and best offer as of each minute's close. Keeping the
two sides apart instead of averaging them is what makes readout (B) executable: lifting the ask and
hitting the bid are both immediate fills, so no queue assumption is needed.

    a Kalshi YES ask at A  ==  a NO bid at 1-A        (there is no short; every order is a buy)

so "sell YES at the bid B" below always means "buy NO at 1-B", which is a real postable order.

UNIT OF OBSERVATION is one (model state, market ticker) pair -- ~15 prefixes per game times the
tickers listed on that game. NOT one trade. A trade-weighted sample over-samples whichever contracts
happened to be busy, which is exactly where the informed flow is; one row per state per contract
weights every market's every state equally and leaves the game as the natural bootstrap cluster.

LEAKAGE. `end_period_ts` is a candle's CLOSE. Taking the last candle with end_period_ts <= T uses
only information that existed at T, and is at most 60s stale when the market is quoting. Staleness is
then measured and reported, because a quote that has not moved in 20 minutes is not a price the model
beat -- it is a price nobody was making. Readout (D) exists to catch exactly that: the last time this
project found a late-game candle edge it was an artifact of pace-interpolated timestamps and
theoretical mids, and both are gone here (real pitch clocks, real BBO).

SETTLEMENT IS CROSS-CHECKED, not assumed. The markets files carry Kalshi's own `result` per ticker.
Our computed y is compared against it and the agreement rate is printed. That is an independent check
on the strike parse, the away/home orientation and the derived settlement rule -- a disagreement means
the pricer is settling a different contract than Kalshi did.

Readouts, all with a game-clustered bootstrap CI:
  (A) accuracy vs the book mid, by game-state bucket. Side-agnostic sanity check.
  (B) "was YES cheaper?" -- the three-way split every observation falls into: p > ask (model says buy
      YES), p < bid (model says buy NO), or bid <= p <= ask (the model holds NO actionable view,
      however accurate it is). For each disagreement set: what we would pay, what actually settled,
      and the gap. This is the number that matters. Gross of fees, and NOT a strategy -- it ignores
      depth, capacity and our own impact, so read it as an accuracy statement in price units.
  (C) calibration by model-p decile against realized settlement and against where the book sat.
  (D) the (A) edge restricted by quote staleness and by book width, to prove it is not an artifact.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kalshi_multi_target_backtest as bt   # noqa: E402  (one pricer, two consumers)

MIN_OBS = 30                                # below this a bootstrap CI is uninformative
STALE_CAPS = [60, 300, 900]                 # seconds; readout (D)
WIDTH_CAPS = [0.02, 0.05, 0.10]             # dollars; readout (D)
DECILES = np.arange(0.0, 1.01, 0.1)


# ----------------------------------------------------------------------------------------------
# top of book
# ----------------------------------------------------------------------------------------------
# Two writer generations exist in kalshi_history and they are interleaved within a series: an older
# one with bare names and a newer one that suffixes `_dollars` / `_fp`. Verified 2026-09-01 that BOTH
# already hold dollars in [0, 1] on the 1c tick -- the suffix is documentation, not a unit change, so
# no scaling is applied. 931 of 938 KXMLBGAME candle files are the bare generation, i.e. reading only
# the `_dollars` name would silently discard the one series that has a full season.
QCOLS = {"ticker": ["market_ticker"],
         "qts": ["end_period_ts"],
         "bid": ["yes_bid_close_dollars", "yes_bid_close"],
         "ask": ["yes_ask_close_dollars", "yes_ask_close"]}


def load_quotes(cfg, keep, listed=frozenset()) -> tuple[pd.DataFrame, dict]:
    """Best bid / best ask time series per ticker, from the candlesticks.

    Filtered to `keep` FILE BY FILE, before any type conversion. KXMLBGAME alone is ~940 candle
    files covering every minute of every market Kalshi ever listed, with prices stored as strings;
    concatenating that whole thing first peaks at many GB for a frame of which <1% is in the
    prediction window.

    A zero bid or zero ask means that side of the book was EMPTY -- the feed reports an absent side
    as 0.0000, not as a 0c price. Treating it as a price would manufacture a ~50c fake edge on every
    illiquid contract, so both sides present is required. Crossed books (bid >= ask) are dropped and
    counted: those are artifacts of aggregating OHLC over a minute (a bid high and an ask low from
    different instants), not arbitrage that existed at one moment.
    """
    files = bt.series_files(cfg["dirs"], "candlesticks")
    parts, seen = [], set()
    st = dict(files=len(files), unreadable=0, candles=0, in_window=0,
              one_sided=0, crossed=0, two_sided=0)
    for p in files:
        have = set(pq.ParquetFile(p).schema_arrow.names)
        pick = {next((a for a in alts if a in have), None): out for out, alts in QCOLS.items()}
        if None in pick:
            # Do NOT read a partial frame: a file with no bid/ask column is not a wide book, it is a
            # file this loader does not understand, and quietly skipping it would shrink the sample
            # invisibly. Counted and reported instead.
            st["unreadable"] += 1
            continue
        c = pd.read_parquet(p, columns=list(pick)).rename(columns=pick)
        st["candles"] += len(c)
        seen.update(c.ticker.unique())
        c = c[c.ticker.isin(keep)]
        st["in_window"] += len(c)
        if c.empty:
            continue
        for col in ("qts", "bid", "ask"):
            c[col] = pd.to_numeric(c[col], errors="coerce")
        c = c.dropna(subset=["ticker", "qts", "bid", "ask"])
        st["one_sided"] += int(((c.bid <= 0) | (c.ask <= 0)).sum())
        c = c[(c.bid > 0) & (c.ask > 0) & (c.ask <= 1.0)]
        st["crossed"] += int((c.bid >= c.ask).sum())
        c = c[c.bid < c.ask]
        if not c.empty:
            parts.append(c[["ticker", "qts", "bid", "ask"]])
    st["tickers_seen"] = len(seen)
    # quoted but absent from the markets files entirely -- no strike and no `result`, so unusable.
    # This is NOT the same as "out of the prediction window", which is the bulk of what `keep` drops.
    st["tickers_unlisted"] = len(seen - set(listed))
    if not parts:
        return pd.DataFrame(columns=["ticker", "qts", "bid", "ask"]), st
    q = pd.concat(parts, ignore_index=True)
    st["two_sided"] = len(q)
    return (q.sort_values(["ticker", "qts"]).drop_duplicates(["ticker", "qts"], keep="last"),
            st)


# ----------------------------------------------------------------------------------------------
# the (model state x ticker) grid
# ----------------------------------------------------------------------------------------------
def market_universe(cfg, preds) -> pd.DataFrame:
    """Every LISTED ticker that maps to a game in the prediction window, with its strike + result.

    The universe comes from the markets files, not from the trades: conditioning on trading activity
    would drop precisely the quiet contracts, which is the opposite of the sample we care about. The
    markets files are also tiny (tens of KB each), so this is what makes the candle filter cheap.
    """
    mk = bt.load_markets(cfg)
    if mk.empty:
        return mk, frozenset()
    listed = frozenset(mk.ticker)
    uni = mk.join(bt.parse_tickers(mk.ticker)).dropna(subset=["tkd", "pair"])
    uni = bt.attach_games(uni, preds).dropna(subset=["game_pk"])
    if not uni.empty:
        uni["game_pk"] = uni.game_pk.astype(int)
    return uni, listed


def observation_grid(cfg, preds, uni) -> pd.DataFrame:
    """Cross every listed ticker with every model state on the game it belongs to."""
    carry = ["prefix_length", "ts", "base_h", "base_a", "final_h", "final_a", "y_total_runs",
             "game_type_code", "home_team_abbr", "away_team_abbr", "split", "game_date"]
    if cfg["kind"] == "head":
        carry += [cfg["prob"], cfg["y"]] + ([cfg["mask"]] if cfg["mask"] else [])
    p = preds.reset_index(names="row")[["row", "game_pk"] + carry]
    return uni.merge(p, on="game_pk", how="inner")


def price_observations(name, cfg, preds, obs):
    """Attach the model fair `f` and the settlement `y`, reusing the backtest's pricer verbatim."""
    if cfg["kind"] == "head":
        if cfg["mask"]:
            obs = obs[obs[cfg["mask"]].astype(bool)]
            if obs.empty:
                return None, f"{name}: no model state where {cfg['prob']} is defined"
        f, y, keep = bt.price_head(cfg, obs)
        if not keep.all():
            print(f"  {name}: dropped {int((~keep).sum()):,} observations whose ticker suffix "
                  f"matched neither team abbreviation")
        obs = obs.loc[keep].copy()
        if obs.empty:
            return None, f"{name}: no ticker suffix resolved to either team"
        obs["f"], obs["y"] = f[keep], y[keep]
        obs["trunc_err"] = 0.0
        return obs, None

    obs = obs.copy()
    obs["K"] = pd.to_numeric(obs.get("floor_strike"), errors="coerce")
    obs = obs.dropna(subset=["K"])
    if obs.empty:
        return None, f"{name}: no strike metadata on any quoted market"
    if obs.K.max() > bt.MAX_STRIKE:
        raise SystemExit(f"{name}: strike {obs.K.max()} exceeds MAX_STRIKE={bt.MAX_STRIKE}")
    is_home = np.zeros(len(obs), bool)
    if cfg["kind"] != "total":
        is_home, keep = bt.strike_side(obs)
        obs = obs.loc[keep].copy()
        if obs.empty:
            return None, f"{name}: strike side never matched either team abbreviation"
    obs["f"], obs["y"], obs["trunc_err"] = bt.price_derived(cfg["kind"], preds, obs, is_home)
    return obs, None


def settlement_check(obs):
    """Per-TICKER agreement between our computed y and Kalshi's `result`.

    y is one fact per (game, strike), so a ticker contributes a single comparison however many model
    states it carries -- otherwise busy markets would dominate the agreement rate.
    """
    if "result" not in obs:
        return None
    per = obs.groupby("ticker").agg(y=("y", "first"), result=("result", "first"))
    res = per.result.astype("string").str.lower()
    per = per[res.isin(["yes", "no"])]
    if per.empty:
        return None
    kal = (res.loc[per.index] == "yes").to_numpy().astype(float)
    bad = per.index[kal != per.y.to_numpy()]
    return len(per), float((kal == per.y.to_numpy()).mean()), list(bad[:5])


# ----------------------------------------------------------------------------------------------
# assembly
# ----------------------------------------------------------------------------------------------
def attach_book(obs, q, max_stale):
    """As-of join each model state to the last candle that CLOSED at or before it."""
    q = q[q.ticker.isin(obs.ticker.unique())].copy()
    if q.empty:
        return q
    # merge_asof consumes the right frame's join key, so carry a copy to measure staleness with.
    # It also refuses an int64/float64 key pair, and end_period_ts arrives as int64 in some files.
    q["qts"] = q.qts.astype(float)
    q["ts"] = q.qts
    obs = obs.astype({"ts": float})
    obs = pd.merge_asof(obs.sort_values("ts"), q.sort_values("ts"),
                        on="ts", by="ticker", direction="backward")
    obs = obs.dropna(subset=["bid", "ask"])
    if obs.empty:
        return obs
    obs["stale"] = obs.ts - obs.qts
    obs = obs[obs.stale <= max_stale]
    if obs.empty:
        return obs

    obs["mid"] = (obs.bid + obs.ask) / 2.0
    obs["width"] = obs.ask - obs.bid
    # The model's view, in the only two forms that are immediately executable.
    obs["take_yes"] = obs.f > obs.ask       # buy YES at the ask
    obs["take_no"] = obs.f < obs.bid        # buy NO at 1 - bid
    obs["no_view"] = ~(obs.take_yes | obs.take_no)
    nan = np.full(len(obs), np.nan)
    ask, bid, y = obs.ask.to_numpy(), obs.bid.to_numpy(), obs.y.to_numpy()
    obs["paid"] = np.where(obs.take_yes, ask, np.where(obs.take_no, 1.0 - bid, nan))
    obs["won"] = np.where(obs.take_yes, y, np.where(obs.take_no, 1.0 - y, nan))
    # buy YES at ask: gap = y - ask.   buy NO at 1-bid: gap = (1-y) - (1-bid) = bid - y.
    obs["gap"] = np.where(obs.take_yes, y - ask, np.where(obs.take_no, bid - y, nan))
    return obs


def build(name, cfg, preds, max_stale):
    uni, listed = market_universe(cfg, preds)
    if uni.empty:
        return None, (f"{name}: no listed market maps to a game in the prediction window "
                      f"({preds.game_date.min().date()}..{preds.game_date.max().date()})"), None, {}
    q, qstats = load_quotes(cfg, set(uni.ticker), listed)
    if q.empty:
        return None, (f"{name}: {uni.ticker.nunique():,} listed tickers in the window but no "
                      f"two-sided top-of-book quote in any candle file"), None, qstats

    obs, why = price_observations(name, cfg, preds, observation_grid(cfg, preds, uni))
    if obs is None:
        return None, why, None, qstats
    chk = settlement_check(obs)

    obs = attach_book(obs, q, max_stale)
    if obs.empty:
        return None, f"{name}: no model state has a top-of-book quote within {max_stale:.0f}s", None, qstats
    return obs, None, chk, qstats


# ----------------------------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------------------------
def band(obs, lo, hi):
    return obs[(obs.prefix_length >= lo) & (obs.prefix_length <= hi)]


def brier_edge(b, rng):
    """Brier(model) minus Brier(mid), with a game-clustered CI. Unweighted: one row, one vote."""
    bd = float(((b.f - b.y) ** 2).mean())
    bp = float(((b.mid - b.y) ** 2).mean())
    edge, lo, hi = bt.ratio_ci(b.assign(d=(b.mid - b.y) ** 2 - (b.f - b.y) ** 2, w=1.0),
                               "d", "w", rng)
    return bd, bp, edge, lo, hi


def report(name, obs, rng, chk, qstats):
    print(f"\n{'=' * 118}\n{name}")
    if qstats["unreadable"]:
        print(f"  WARNING {qstats['unreadable']} of {qstats['files']} candle files carry no "
              f"recognized bid/ask column and were skipped")
    print(f"  {qstats['candles']:,} candles in {qstats['files']} files -> "
          f"{qstats['in_window']:,} on a listed in-window ticker -> {qstats['two_sided']:,} "
          f"two-sided ({qstats['one_sided']:,} had an empty side, {qstats['crossed']:,} crossed)")
    if qstats["tickers_unlisted"]:
        print(f"  NOTE {qstats['tickers_unlisted']:,} of {qstats['tickers_seen']:,} quoted tickers "
              f"are absent from the markets files (no strike/result available; excluded)")
    print(f"  {len(obs):,} (state x ticker) observations, {obs.ticker.nunique():,} tickers, "
          f"{obs.game_pk.nunique()} games ({obs.game_date.min().date()} .. "
          f"{obs.game_date.max().date()})  game types "
          f"{obs.groupby('game_type_code').game_pk.nunique().to_dict()}  "
          f"splits {obs.split.value_counts().to_dict()}")
    print(f"  book width: median {obs.width.median() * 100:.1f}c, mean {obs.width.mean() * 100:.1f}c"
          f"   quote age: median {obs.stale.median():.0f}s, p90 {obs.stale.quantile(0.9):.0f}s")
    if obs.trunc_err.max() > 0:
        print(f"  worst NegBin truncation bound on any priced strike: {obs.trunc_err.max():.2e}")
    if chk:
        n, agree, bad = chk
        flag = "" if agree > 0.999 else f"   <-- INVESTIGATE, e.g. {bad}"
        print(f"  settlement cross-check vs Kalshi `result`: {agree:.3%} of {n:,} tickers agree{flag}")

    print("\n  (A) ACCURACY VS THE BOOK MID   (unweighted Brier; game-clustered CI on the difference)")
    print(f"  {'bucket':>10} {'obs':>8} {'Brier_DL':>9} {'Brier_mid':>10} {'edge':>9} {'edge_95CI':>20}")
    for lo, hi in bt.BUCKETS:
        b = band(obs, lo, hi)
        if len(b) < MIN_OBS:
            continue
        bd, bp, e, l, h = brier_edge(b, rng)
        print(f"  {bt.bucket_label(lo, hi):>10} {len(b):>8} {bd:>9.5f} {bp:>10.5f} {e:>+9.5f} "
              f"[{l:>+8.5f},{h:>+8.5f}]")

    print("\n  (B) WAS YES CHEAPER?   p>ask => buy YES at ask; p<bid => buy NO at 1-bid; "
          "gap = settled - paid, in cents")
    print(f"  {'bucket':>10} {'obs':>8} {'no_view':>8} | {'buyYES':>7} {'pay':>6} {'settled':>8} "
          f"{'gap_c':>7} {'gap_95CI_c':>19} | {'buyNO':>7} {'pay':>6} {'settled':>8} {'gap_c':>7} "
          f"{'gap_95CI_c':>19}")
    for lo, hi in bt.BUCKETS:
        b = band(obs, lo, hi)
        if len(b) < MIN_OBS:
            continue
        row = f"  {bt.bucket_label(lo, hi):>10} {len(b):>8} {b.no_view.mean():>7.1%} |"
        for nm in ("yes", "no"):
            d = b[b[f"take_{nm}"]]
            if len(d) < MIN_OBS:
                row += f" {len(d):>7} {'-':>6} {'-':>8} {'-':>7} {'-':>19} |"
                continue
            g, l, h = bt.ratio_ci(d.assign(w=1.0), "gap", "w", rng)
            row += (f" {len(d):>7} {d.paid.mean() * 100:>6.1f} {d.won.mean():>8.1%} "
                    f"{g * 100:>+7.2f} [{l * 100:>+7.2f},{h * 100:>+7.2f}] |")
        print(row.rstrip(" |"))

    print("\n  (C) CALIBRATION BY MODEL DECILE   (does p mean what it says, and where was the book?)")
    print(f"  {'p_bin':>12} {'obs':>8} {'mean_p':>8} {'realized':>9} {'mean_bid':>9} "
          f"{'mean_mid':>9} {'mean_ask':>9}")
    for iv, g in obs.groupby(pd.cut(obs.f, DECILES, include_lowest=True), observed=True):
        if len(g) < MIN_OBS:
            continue
        print(f"  {f'{iv.left:.1f}-{iv.right:.1f}':>12} {len(g):>8} {g.f.mean():>8.3f} "
              f"{g.y.mean():>9.3f} {g.bid.mean():>9.3f} {g.mid.mean():>9.3f} {g.ask.mean():>9.3f}")

    print("\n  (D) IS THE EDGE AN ARTIFACT OF STALE OR WIDE QUOTES?   (readout (A), restricted)")
    print(f"  {'restriction':>16} {'obs':>8} {'Brier_DL':>9} {'Brier_mid':>10} {'edge':>9} "
          f"{'edge_95CI':>20}")
    subsets = [(f"stale<={c}s", obs.stale <= c) for c in STALE_CAPS]
    subsets += [(f"width<={c * 100:.0f}c", obs.width <= c) for c in WIDTH_CAPS]
    for label, m in subsets:
        b = obs[m]
        if len(b) < MIN_OBS:
            continue
        bd, bp, e, l, h = brier_edge(b, rng)
        print(f"  {label:>16} {len(b):>8} {bd:>9.5f} {bp:>10.5f} {e:>+9.5f} "
              f"[{l:>+8.5f},{h:>+8.5f}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="*", default=list(bt.SERIES))
    ap.add_argument("--max-stale", type=float, default=1800,
                    help="drop states whose newest top-of-book quote is older than this (seconds)")
    args = ap.parse_args()

    preds = bt.load_predictions()
    print(f"predictions: {len(preds):,} prefix rows on {preds.game_pk.nunique()} games, "
          f"{preds.game_date.min().date()} .. {preds.game_date.max().date()}")

    rng = np.random.default_rng(0)
    skipped = []
    for name in args.series:
        obs, why, chk, qstats = build(name, bt.SERIES[name], preds, args.max_stale)
        if obs is None:
            skipped.append(why)
            continue
        report(name, obs, rng, chk, qstats)

    if skipped:
        print(f"\n{'=' * 118}\nNO USABLE OVERLAP")
        for s in skipped:
            print(f"  - {s}")
    print(f"\nreal pitch timestamps (raw feed); real best bid / best ask, never a mid, for the "
          f"executable test; leak-free as-of join on candle CLOSE with quotes older than "
          f"{args.max_stale:.0f}s dropped.\nAccuracy only -- no fill model, no queue assumption, no "
          f"PnL. Readout (B) is gross of fees and ignores depth and impact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
