#!/usr/bin/env python3
"""Trade-based, fill-honest backtest of the DL home_win signal vs the 2025 Kalshi market.

WHY trades, not candles. Candle mids are a theoretical BBO midpoint you cannot necessarily
transact at, and 1-min candles miss most activity. A TRADE print is ground truth: it means a
taker crossed AND a resting order existed at that price, so real executable liquidity was there.
The aggressor side (taker_book_side) tells us which resting order existed:
  - print on the ASK at P  -> a resting SELLER at P; we could have BOUGHT YES at P.
  - print on the BID at P  -> a resting BUYER  at P; we could have SOLD  YES at P.
So we only ever claim edge on the side where a counterparty is *proven* to have existed.

WHY real timestamps. Each DL prediction is indexed by prefix_length (cumulative game sequence
index), not wall-clock. Stage 1 attaches the ACTUAL pitch time by joining the feature store's
sequence rows to the raw pitch rows on (game_pk, play_index, at_bat_index, pitch_sequence_index)
and reading pitch_start_time (raw MLB feed). This replaces the linear pace interpolation the
candle backtest used, removing the stale-mid artifact risk in the late buckets.

Two readouts per game-state bucket (bucket = prefix_length active at the trade), both with a
game-clustered bootstrap CI:
  (A) trade-price accuracy: Brier(model fair as-of trade) vs Brier(executed yes_price) vs outcome.
  (B) executable PnL: side-matched trades only, net of Kalshi taker fee 0.07*P*(1-P), realized at
      the outcome, per contract and size-weighted (size capped by the print).
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
TRADES = ROOT / "data/backtest/kalshi_game_trades"
RAW = ROOT / "data/backtest/raw_pitches_2025"
META = ROOT / "deep_learning/feature_store/game_meta.parquet"
SEQ = ROOT / "deep_learning/feature_store/pitch_sequences.parquet"

TICKER = re.compile(r"^KXMLBGAME-(\d{2}[A-Z]{3}\d{2})([A-Z0-9]+)-([A-Z0-9]+)$")
MON = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
BUCKETS = [(0, 0), (1, 50), (51, 100), (101, 150), (151, 200), (201, 300), (301, 10000)]
TAKER_RATE = 0.07


def norm(c): return "AZ" if c == "ARI" else c
def ticker_date(t): return pd.Timestamp(year=2000 + int(t[:2]), month=MON[t[2:5]], day=int(t[5:7]))


def to_unix_s(x):
    s = pd.to_datetime(x, utc=True, errors="coerce")
    return ((s - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")).astype("float64")


def taker_fee(price):
    return TAKER_RATE * price * (1.0 - price)


def build_seq_time(gpks):
    """Per-game map: sequence_index -> unix seconds of the real pitch, floored at first pitch."""
    fs = ds.dataset(SEQ).to_table(
        filter=pc.field("game_pk").isin(gpks),
        columns=["game_pk", "sequence_index", "play_index", "at_bat_index",
                 "pitch_sequence_index"]).to_pandas()
    raw = pd.concat([pd.read_parquet(p, columns=[
        "game_pk", "play_index", "at_bat_index", "pitch_sequence_index",
        "pitch_start_time", "at_bat_start_time", "game_datetime_utc"])
        for p in glob.glob(str(RAW / "*.parquet"))], ignore_index=True)
    raw = raw[raw.game_pk.isin(gpks)]
    keys = ["game_pk", "play_index", "at_bat_index", "pitch_sequence_index"]
    j = fs.merge(raw, on=keys, how="left")
    # pitch time; fall back to the at-bat clock for non-pitch events / gaps
    ts = to_unix_s(j.pitch_start_time.fillna(j.at_bat_start_time))
    first_pitch = to_unix_s(j.game_datetime_utc)
    # Pre-game sequence rows carry warmup timestamps hours before first pitch (verified); floor
    # them at game start so they cannot leak a bogus early wall-clock into the join.
    j["ts"] = np.maximum(ts, first_pitch)
    j = j.dropna(subset=["ts"]).sort_values(["game_pk", "sequence_index"])
    # enforce monotone non-decreasing time within a game (raw feed has rare out-of-order stamps)
    j["ts"] = j.groupby("game_pk").ts.cummax()
    return j[["game_pk", "sequence_index", "ts"]]


def main() -> int:
    preds = pd.concat(
        [pd.read_parquet(PRED / f"readout_fix_{s}.parquet",
                         columns=["game_pk", "prefix_length", "p_home_win", "y_home_win"])
         for s in ("test", "val")], ignore_index=True)
    meta = pd.read_parquet(META, columns=[
        "game_pk", "game_date", "home_team_abbr", "game_type_code"])
    preds = preds.merge(meta, on="game_pk", how="left")
    preds = preds[preds.game_type_code == "R"]

    # --- Stage 1: real time per (game_pk, prefix_length) --------------------------------------
    gpks = preds.game_pk.unique().tolist()
    seqt = build_seq_time(gpks)
    preds = preds.merge(
        seqt.rename(columns={"sequence_index": "prefix_length"}),
        on=["game_pk", "prefix_length"], how="left")
    preds = preds.dropna(subset=["ts"])   # drop prefixes with no matching raw pitch (rare)

    # --- Stage 2: trades, home-side ticker -----------------------------------------------------
    cols = ["ticker", "created_time", "yes_price_dollars", "count_fp",
            "taker_book_side", "is_block_trade"]
    td = pd.concat([pd.read_parquet(p, columns=cols) for p in glob.glob(str(TRADES / "*.parquet"))],
                   ignore_index=True)
    td = td[~td.is_block_trade]                        # lit book only (OTC/RFQ excluded)
    td["P"] = pd.to_numeric(td.yes_price_dollars, errors="coerce")
    td["size"] = pd.to_numeric(td.count_fp, errors="coerce")
    td["ts"] = to_unix_s(td.created_time)
    m = td.ticker.str.extract(TICKER)
    td["tkd"] = m[0].map(ticker_date, na_action="ignore")
    td["yes"] = m[2].map(norm, na_action="ignore")
    td = td.dropna(subset=["P", "size", "ts", "tkd", "yes"])

    # keep only home-side tickers that map to a backtest game (yes-team == home, date == game_date)
    home_key = {(pd.Timestamp(d).normalize(), h): g
                for g, (d, h) in preds.groupby("game_pk")[["game_date", "home_team_abbr"]]
                .first().iterrows()}
    td["game_pk"] = [home_key.get((d, y)) for d, y in zip(td.tkd, td.yes)]
    td = td.dropna(subset=["game_pk"])
    td["game_pk"] = td.game_pk.astype(int)

    # --- as-of join: each trade gets the model's latest fair value at or before the trade time --
    preds = preds.sort_values(["game_pk", "ts"])
    td = td.sort_values(["game_pk", "ts"])
    parts = []
    for gpk, tg in td.groupby("game_pk"):
        pg = preds[preds.game_pk == gpk]
        if pg.empty:
            continue
        pos = np.searchsorted(pg.ts.to_numpy(), tg.ts.to_numpy(), side="right") - 1
        ok = pos >= 0
        tg = tg.loc[ok].copy()
        pos = pos[ok]
        tg["f"] = pg.p_home_win.to_numpy()[pos]        # model fair as-of the trade
        tg["y"] = pg.y_home_win.to_numpy()[pos]
        tg["prefix_length"] = pg.prefix_length.to_numpy()[pos]
        parts.append(tg)
    t = pd.concat(parts, ignore_index=True)
    print(f"trades matched to a model state: {len(t):,} across {t.game_pk.nunique()} games\n")

    # side-matched executable trade: ask-print -> buy YES if f>P; bid-print -> sell YES if f<P
    ask = t.taker_book_side == "ask"
    bid = t.taker_book_side == "bid"
    buy = ask & (t.f > t.P)
    sell = bid & (t.f < t.P)
    t["exec"] = buy | sell
    # per-contract realized PnL net of taker fee (fee coefficient is symmetric in P(1-P))
    pnl = np.where(buy, t.y - t.P - taker_fee(t.P),
                   np.where(sell, t.P - t.y - taker_fee(1 - t.P), 0.0))
    t["pnl"] = pnl
    # trade-price accuracy pieces
    t["se_f"] = (t.f - t.y) ** 2
    t["se_p"] = (t.P - t.y) ** 2

    rng = np.random.default_rng(0)

    def ratio_ci(b, num_col, den_col, n=2000):
        # game-clustered, size-weighted bootstrap of the AGGREGATE ratio sum(num)/sum(den).
        # Unit of resampling = game, so within-game correlation and trade size are both respected
        # and the point estimate (full-sample ratio) is the mean of the bootstrap by construction.
        per = b.groupby("game_pk").apply(
            lambda g: pd.Series({"num": (g[num_col] * g[den_col]).sum(), "den": g[den_col].sum()}))
        num, den = per.num.to_numpy(), per.den.to_numpy()
        point = num.sum() / den.sum()
        idx = np.arange(len(num))
        bs = np.empty(n)
        for k in range(n):
            s = rng.choice(idx, len(idx), replace=True)
            bs[k] = num[s].sum() / den[s].sum()
        return point, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    print("(A) TRADE-PRICE ACCURACY  (Brier vs executed yes-price; size-weighted, game-clustered CI)")
    print(f"{'bucket':>10} {'trades':>8} {'Brier_DL':>9} {'Brier_px':>9} {'edge':>9} {'edge_95CI':>18}")
    for lo, hi in BUCKETS:
        b = t[(t.prefix_length >= lo) & (t.prefix_length <= hi)]
        if len(b) < 50:
            continue
        w = b["size"].to_numpy()
        bd = np.average(b.se_f, weights=w)
        bp = np.average(b.se_p, weights=w)
        b = b.assign(d=b.se_p - b.se_f)          # >0 => model beats the executed price
        edge, lo_ci, hi_ci = ratio_ci(b, "d", "size")
        lbl = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10000 else f"{lo}+")
        print(f"{lbl:>10} {len(b):>8} {bd:>9.5f} {bp:>9.5f} {edge:>+9.5f} [{lo_ci:>+7.4f},{hi_ci:>+7.4f}]")

    print("\n(B) EXECUTABLE PnL  (side-matched trades only, net of taker fee; size-weighted, clustered CI)")
    print(f"{'bucket':>10} {'trades':>8} {'contracts':>12} {'pnl/ct_c':>9} {'total_$':>11} "
          f"{'pnl/ct_95CI_c':>20}")
    for lo, hi in BUCKETS:
        b = t[(t.prefix_length >= lo) & (t.prefix_length <= hi) & t.exec]
        if len(b) < 50:
            continue
        ct = b["size"].sum()
        pc_, lo_ci, hi_ci = ratio_ci(b, "pnl", "size")
        tot = float((b.pnl * b["size"]).sum())
        lbl = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10000 else f"{lo}+")
        print(f"{lbl:>10} {len(b):>8} {ct:>12,.0f} {pc_*100:>9.3f} {tot:>11,.0f} "
              f"[{lo_ci*100:>+7.3f},{hi_ci*100:>+7.3f}]")
    print("\nreal pitch timestamps (raw feed); lit-book trades only; taker join (maker dominates).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
