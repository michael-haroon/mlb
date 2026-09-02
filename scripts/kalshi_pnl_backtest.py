#!/usr/bin/env python3
"""Fee-aware PnL backtest for the DL home_win signal against the 2025 Kalshi market.

The prefix-curve backtest (kalshi_prefix_backtest.py) answered "is the model more ACCURATE than
the market" in Brier units. That is necessary but not sufficient: a Brier improvement is not
dollars. This script converts the same aligned data into a realized-PnL curve net of Kalshi's
actual fee schedule, so the tradeable question is answered in money, not squared error.

Cost model (TAKER, the conservative case):
  - Enter at the real quote: buy YES at ask when p_dl > mid, buy NO at (1-bid) when p_dl < mid.
  - Kalshi taker fee = 0.07 * P * (1-P) dollars/contract (maker is 0.0175, 4x cheaper, and also
    EARNS the spread; a maker strategy therefore dominates this and is not simulated here because
    it needs a fill model). Fee is charged on the entry price P.
  - Settle at the realized outcome y (YES pays $1 if home win). PnL is per 1-contract trade.

HARD LIMITATION — this is quote-based, NOT fill-verified. 57.6% of candle-minutes have zero
volume (data-topology-inspector, 2026-09-01), so a quoted bid/ask does not guarantee a
counterparty. The late-game buckets where the edge concentrates are the thinnest. Confirming the
+5.9c/trade 301+ edge is real requires the actual trades feed, not these candles. Also inherits
the LINEAR prefix->wall-clock pace approximation from the prefix backtest: late-game edge may
partly be a stale mid rather than skill. Both caveats inflate late buckets specifically.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "data/backtest"
CANDLES = ROOT / "data/backtest/kalshi_game"
META = ROOT / "deep_learning/feature_store/game_meta.parquet"
PITCHES = ROOT / "deep_learning/feature_store/pitch_sequences.parquet"

TICKER = re.compile(r"^KXMLBGAME-(\d{2}[A-Z]{3}\d{2})([A-Z0-9]+)-([A-Z0-9]+)$")
MON = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
BUCKETS = [(0, 0), (1, 50), (51, 100), (101, 150), (151, 200), (201, 300), (301, 10000)]
TAKER_RATE = 0.07  # Kalshi taker fee coefficient; maker is 0.0175


def norm(c): return "AZ" if c == "ARI" else c
def ticker_date(t): return pd.Timestamp(year=2000 + int(t[:2]), month=MON[t[2:5]], day=int(t[5:7]))


def to_unix_s(x):
    # game_datetime_utc is datetime64[us]; .astype("int64")//1e9 undershoots 1000x. Resolution-agnostic:
    s = pd.to_datetime(x, utc=True)
    return ((s - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")).astype("int64")


def taker_fee(price):
    return TAKER_RATE * price * (1.0 - price)


def main() -> int:
    preds = pd.concat(
        [pd.read_parquet(PRED / f"readout_fix_{s}.parquet",
                         columns=["game_pk", "prefix_length", "p_home_win", "y_home_win"])
         for s in ("test", "val")], ignore_index=True)
    meta = pd.read_parquet(META, columns=[
        "game_pk", "game_date", "game_datetime_utc", "home_team_abbr",
        "game_type_code", "game_duration_minutes"])
    preds = preds.merge(meta, on="game_pk", how="left")
    preds = preds[preds.game_type_code == "R"].dropna(subset=["game_datetime_utc"])
    preds["start_ts"] = to_unix_s(preds.game_datetime_utc)
    preds["dur_s"] = preds.game_duration_minutes * 60.0

    gpks = preds.game_pk.unique().tolist()
    tot = ds.dataset(PITCHES).to_table(
        filter=pc.field("game_pk").isin(gpks), columns=["game_pk"]).to_pandas() \
        .groupby("game_pk").size().rename("total_seq")
    preds = preds.merge(tot, on="game_pk", how="left").dropna(subset=["total_seq"])
    frac = np.clip(preds.prefix_length / preds.total_seq, 0.0, 1.0)
    preds["est_ts"] = preds.start_ts + frac * preds.dur_s

    cols = ["market_ticker", "end_period_ts", "yes_bid_close", "yes_ask_close"]
    cd = pd.concat([pd.read_parquet(p, columns=cols) for p in glob.glob(str(CANDLES / "*.parquet"))],
                   ignore_index=True)
    for c in ("yes_bid_close", "yes_ask_close"):
        cd[c] = pd.to_numeric(cd[c], errors="coerce")
    cd = cd.dropna(subset=["yes_bid_close", "yes_ask_close"])
    m = cd.market_ticker.str.extract(TICKER)
    cd["tkd"] = m[0].map(ticker_date, na_action="ignore")
    cd["yes"] = m[2].map(norm, na_action="ignore")
    cd = cd.dropna(subset=["tkd", "yes"]).sort_values("end_period_ts")
    by = {k: g.reset_index(drop=True) for k, g in cd.groupby([cd.tkd, cd.yes])}

    bid = np.full(len(preds), np.nan)
    ask = np.full(len(preds), np.nan)
    preds = preds.reset_index(drop=True)
    for gpk, idx in preds.groupby("game_pk").groups.items():
        r0 = preds.loc[idx[0]]
        g = by.get((pd.Timestamp(r0.game_date).normalize(), r0.home_team_abbr))
        if g is None:
            continue
        ts = g.end_period_ts.to_numpy()
        pos = np.clip(np.searchsorted(ts, preds.loc[idx, "est_ts"].to_numpy(), side="right") - 1,
                      0, len(g) - 1)
        bid[idx] = g.yes_bid_close.to_numpy()[pos]
        ask[idx] = g.yes_ask_close.to_numpy()[pos]
    preds["bid"], preds["ask"] = bid, ask
    j = preds.dropna(subset=["bid", "ask"]).copy()
    j["mid"] = (j.bid + j.ask) / 2.0

    p = j.p_home_win.to_numpy(float)
    y = j.y_home_win.to_numpy(float)
    a, b = j.ask.to_numpy(float), j.bid.to_numpy(float)
    buy_yes, buy_no = p > j.mid.to_numpy(), p < j.mid.to_numpy()
    # YES: pay ask, collect y. NO: pay (1-bid), collect (1-y) -> PnL simplifies to bid - y - fee.
    pnl = np.where(buy_yes, y - a - taker_fee(a),
                   np.where(buy_no, b - y - taker_fee(1 - b), 0.0))
    j["pnl"], j["traded"] = pnl, (buy_yes | buy_no)

    print(f"aligned samples: {len(j)} across {j.game_pk.nunique()} games (taker, 1 contract/trade)\n")
    print(f"{'bucket':>10} {'trades':>7} {'pnl/trade_c':>12} {'total_pnl_$':>12} {'winrate':>8}")
    for lo, hi in BUCKETS:
        t = j[(j.prefix_length >= lo) & (j.prefix_length <= hi) & j.traded]
        if len(t) < 20:
            continue
        label = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10000 else f"{lo}+")
        print(f"{label:>10} {len(t):>7} {t.pnl.mean()*100:>12.3f} {t.pnl.sum():>12.2f} "
              f"{(t.pnl>0).mean():>8.3f}")
    print("\nNOT fill-verified (quote-based; 57.6% of candle-minutes are zero-volume). "
          "Maker economics dominate this and are not simulated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
