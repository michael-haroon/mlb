#!/usr/bin/env python3
"""Full prefix-curve DL-vs-market backtest for home_win on the 2025 season.

The question serving hinges on: at each game state, is the model's home_win probability closer
to the outcome than the market's price? Answered by bucketing on prefix_length (game progress)
and comparing DL Brier vs market Brier vs a constant, per bucket. The bucket where DL beats the
market AND the edge clears the half-spread is the window worth serving.

ALIGNMENT CAVEAT (v1). The feature store has no per-pitch wall-clock timestamp, so each DL
sample's game state (prefix_length == cumulative sequence index) is mapped to wall-clock by
LINEAR pace interpolation between two real per-game anchors: game_datetime_utc (pitch-1 time) and
game_duration_minutes (end). This is exact at the endpoints and wrong mid-game to the extent pace
is non-uniform (pitching changes, extra innings, replay). Market prob is smooth except at scoring
events, so coarse buckets survive the approximation; a fill-grade backtest would need real Gumbo
playEvent timestamps. Flagged, not hidden.
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


def norm(c): return "AZ" if c == "ARI" else c
def ticker_date(t): return pd.Timestamp(year=2000 + int(t[:2]), month=MON[t[2:5]], day=int(t[5:7]))
def brier(p, y): return float(np.mean((p - y) ** 2))


def to_unix_s(x):
    # game_datetime_utc parses to datetime64[us], so .astype("int64") yields MICROSECONDS;
    # dividing by 1e9 undershoots by 1000x (unix seconds land ~1.7e6 instead of ~1.7e9),
    # which silently maps every prefix's est_ts before the first candle. Subtract-epoch /
    # floor-divide by 1s is resolution-agnostic and immune to that bug.
    s = pd.to_datetime(x, utc=True)
    return ((s - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")).astype("int64")


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

    # total sequence length per game (pace-model denominator) from pitch_sequences
    gpks = preds.game_pk.unique().tolist()
    tbl = ds.dataset(PITCHES).to_table(
        filter=pc.field("game_pk").isin(gpks), columns=["game_pk"])
    total_seq = tbl.to_pandas().groupby("game_pk").size().rename("total_seq")
    preds = preds.merge(total_seq, on="game_pk", how="left").dropna(subset=["total_seq"])
    # frac in [0,1]; clamp so a prefix beyond the recorded sequence maps to game end
    frac = np.clip(preds.prefix_length / preds.total_seq, 0.0, 1.0)
    preds["est_ts"] = preds.start_ts + frac * preds.dur_s

    # candle lookup: home-side ticker mid time series, keyed by (date, home_abbr)
    cols = ["market_ticker", "end_period_ts", "yes_bid_close", "yes_ask_close"]
    cd = pd.concat([pd.read_parquet(p, columns=cols) for p in glob.glob(str(CANDLES / "*.parquet"))],
                   ignore_index=True)
    for c in ("yes_bid_close", "yes_ask_close"):
        cd[c] = pd.to_numeric(cd[c], errors="coerce")
    cd = cd.dropna(subset=["yes_bid_close", "yes_ask_close"])
    cd["mid"] = (cd.yes_bid_close + cd.yes_ask_close) / 2.0
    cd["spread"] = cd.yes_ask_close - cd.yes_bid_close
    m = cd.market_ticker.str.extract(TICKER)
    cd["tk_date"] = m[0].map(ticker_date, na_action="ignore")
    cd["yes_team"] = m[2].map(norm, na_action="ignore")
    cd = cd.dropna(subset=["tk_date", "yes_team"]).sort_values("end_period_ts")
    cd_by_key = {k: g.reset_index(drop=True) for k, g in
                 cd.groupby([cd.tk_date, cd.yes_team])}

    # asof: last candle at or before est_ts, per game's home-side series
    p_mkt = np.full(len(preds), np.nan)
    spread = np.full(len(preds), np.nan)
    preds = preds.reset_index(drop=True)
    for gpk, idx in preds.groupby("game_pk").groups.items():
        r0 = preds.loc[idx[0]]
        g = cd_by_key.get((pd.Timestamp(r0.game_date).normalize(), r0.home_team_abbr))
        if g is None:
            continue
        ts = g.end_period_ts.to_numpy()
        pos = np.searchsorted(ts, preds.loc[idx, "est_ts"].to_numpy(), side="right") - 1
        pos = np.clip(pos, 0, len(g) - 1)
        p_mkt[idx] = g.mid.to_numpy()[pos]
        spread[idx] = g.spread.to_numpy()[pos]
    preds["p_mkt"] = p_mkt
    preds["spread"] = spread
    j = preds.dropna(subset=["p_mkt"])
    print(f"aligned samples: {len(j)} across {j.game_pk.nunique()} games\n")

    # Paired edge CI must cluster on game: samples share a game's outcome and near-duplicate
    # probs, so i.i.d. bootstrap over rows understates variance. Resample GAMES with replacement.
    rng = np.random.default_rng(0)
    j = j.copy()
    j["se_dl"] = (j.p_home_win.to_numpy(float) - j.y_home_win.to_numpy(float)) ** 2
    j["se_mkt"] = (j.p_mkt.to_numpy(float) - j.y_home_win.to_numpy(float)) ** 2

    def edge_ci(b, n_boot=2000):
        # per-game mean paired diff, then bootstrap over games
        per = b.groupby("game_pk").apply(
            lambda g: pd.Series({"d": (g.se_mkt - g.se_dl).mean(), "w": len(g)}))
        d, w = per.d.to_numpy(), per.w.to_numpy()
        boot = np.empty(n_boot)
        idx = np.arange(len(d))
        for k in range(n_boot):
            s = rng.choice(idx, size=len(idx), replace=True)
            boot[k] = np.average(d[s], weights=w[s])
        return np.percentile(boot, 2.5), np.percentile(boot, 97.5)

    print(f"{'bucket':>10} {'n':>6} {'base':>6} {'Brier_DL':>9} {'Brier_mkt':>10} "
          f"{'BSS_vs_mkt':>11} {'edge':>8} {'edge_95CI':>18} {'spread':>7} {'tradeable':>9}")
    for lo, hi in BUCKETS:
        b = j[(j.prefix_length >= lo) & (j.prefix_length <= hi)]
        if len(b) < 20:
            continue
        y = b.y_home_win.to_numpy(float)
        base = y.mean()
        bd = brier(b.p_home_win.to_numpy(float), y)
        bm = brier(b.p_mkt.to_numpy(float), y)
        edge = bm - bd          # >0 means DL is more accurate than the market
        lo_ci, hi_ci = edge_ci(b)
        sp = b.spread.mean()
        label = f"{lo}" if lo == hi else f"{lo}-{hi if hi < 10000 else ''}+".replace("-+", "+")
        # tradeable requires the edge to (a) clear the half-spread AND (b) be significantly >0
        trad = "yes" if (edge > sp / 2 and lo_ci > 0) else "no"
        print(f"{label:>10} {len(b):>6} {base:>6.3f} {bd:>9.5f} {bm:>10.5f} "
              f"{1-bd/bm:>11.4f} {edge:>+8.5f} [{lo_ci:>+7.4f},{hi_ci:>+7.4f}] {sp:>7.4f} {trad:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
