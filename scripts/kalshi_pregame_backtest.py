#!/usr/bin/env python3
"""Pregame DL-vs-market backtest for home_win on the 2025 season.

Every DL number to date is BSS vs a CONSTANT base rate, which is not edge. This script
replaces that baseline with the Kalshi market's own pregame implied probability and asks the
only question that gates serving: at first pitch, is the model's home_win probability closer to
the realized outcome than the market's is?

Scope is deliberately the pregame bucket (prefix_length == 0): it needs only each game's start
time, not a per-pitch timestamp map, so it validates the ticker<->game join and the yes-side
logic before the full prefix curve is built on top of it. The EXPECTED result here is that DL
LOSES to the market pregame -- the model sits at a ~+0.011 BSS information ceiling vs a constant
while the market prices confirmed lineups it never sees (~+0.056). A negative pregame edge is the
control that proves the join is faithful; the interesting edge, if any, lives later in the game.

Join contract:
  - Kalshi ticker KXMLBGAME-{YYMMMDD}{AWAY}{HOME}-{YESTEAM}: one market per side, yes-team explicit.
  - Kalshi codes == MLB game_meta abbrs except ARI->AZ (Arizona appears as both in Kalshi).
  - Market pregame price = mid (yes_bid_close+yes_ask_close)/2 of the LAST candle at or before
    game_datetime_utc, from the HOME-side ticker so mid == P(home) with no inversion.
  - historical/ candle columns have NO _dollars/_fp suffix and are stored as strings.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "data/backtest"
CANDLES = ROOT / "data/backtest/kalshi_game"
META = ROOT / "deep_learning/feature_store/game_meta.parquet"

TICKER = re.compile(r"^KXMLBGAME-(\d{2}[A-Z]{3}\d{2})([A-Z0-9]+)-([A-Z0-9]+)$")
MON = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def norm(code: str) -> str:
    return "AZ" if code == "ARI" else code


def ticker_date(tok: str) -> pd.Timestamp:
    return pd.Timestamp(year=2000 + int(tok[:2]), month=MON[tok[2:5]], day=int(tok[5:7]))


def to_unix_s(x):
    # game_datetime_utc is datetime64[us]; .astype("int64") is MICROSECONDS, so //1e9 lands
    # 1000x too small and every "pregame" candle filter (end_period_ts <= start_ts) fell through
    # to the day's first candle by luck. Resolution-agnostic conversion below.
    s = pd.to_datetime(x, utc=True)
    return ((s - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")).astype("int64")


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def bss(p, y, base):
    return 1.0 - brier(p, y) / brier(np.full_like(y, base, dtype=float), y)


def main() -> int:
    # --- DL pregame predictions (test + val; 2025 is split across both) --------------------
    preds = pd.concat(
        [pd.read_parquet(PRED / f"readout_fix_{s}.parquet") for s in ("test", "val")],
        ignore_index=True)
    pre = preds[preds.prefix_length == 0][["game_pk", "p_home_win", "y_home_win"]].copy()
    print(f"pregame DL rows: {len(pre)} ({pre.game_pk.nunique()} games)")

    meta = pd.read_parquet(META, columns=[
        "game_pk", "game_date", "game_datetime_utc", "home_team_abbr", "away_team_abbr",
        "game_type_code", "season"])
    pre = pre.merge(meta, on="game_pk", how="left")
    pre["start_ts"] = to_unix_s(pre.game_datetime_utc)

    # --- Kalshi candles -> per-ticker pregame mid ------------------------------------------
    cols = ["market_ticker", "end_period_ts", "yes_bid_close", "yes_ask_close", "volume"]
    cd = pd.concat([pd.read_parquet(p, columns=cols) for p in glob.glob(str(CANDLES / "*.parquet"))],
                   ignore_index=True)
    for c in ("yes_bid_close", "yes_ask_close", "volume"):
        cd[c] = pd.to_numeric(cd[c], errors="coerce")
    cd = cd.dropna(subset=["yes_bid_close", "yes_ask_close"])
    cd["mid"] = (cd.yes_bid_close + cd.yes_ask_close) / 2.0
    cd["spread"] = cd.yes_ask_close - cd.yes_bid_close

    m = cd.market_ticker.str.extract(TICKER)
    cd["tk_date"] = m[0].map(ticker_date, na_action="ignore")
    cd["yes_team"] = m[2].map(norm, na_action="ignore")
    cd = cd.dropna(subset=["tk_date", "yes_team"])

    # HOME-side ticker per game: yes_team == home_abbr AND date matches. Build the lookup by
    # (date, home) since the yes-team of the home-side market IS the home team.
    home_ticker = pre.assign(
        key=list(zip(pre.game_date.dt.normalize(), pre.home_team_abbr)))
    cd["key"] = list(zip(cd.tk_date, cd.yes_team))

    rows = []
    cd_by_key = {k: g for k, g in cd.groupby("key")}
    for _, r in home_ticker.iterrows():
        g = cd_by_key.get(r["key"])
        if g is None:
            continue
        pg = g[g.end_period_ts <= r.start_ts]
        use = pg if len(pg) else g            # fall back to first candle if none pre-start
        c = use.sort_values("end_period_ts").iloc[-1 if len(pg) else 0]
        rows.append({"game_pk": r.game_pk, "p_mkt": c.mid, "spread": c.spread,
                     "p_dl": r.p_home_win, "y": r.y_home_win,
                     "game_type_code": r.game_type_code})

    j = pd.DataFrame(rows)
    # Regular season only (game_type_code 'R'): playoff/spring dynamics differ and Kalshi's
    # postseason markets are a separate slice.
    j = j[j.game_type_code == "R"].dropna(subset=["p_mkt", "p_dl", "y"])
    print(f"joined regular-season games with a pregame market: {len(j)}")
    if not len(j):
        print("NO JOIN — check team codes / dates")
        return 1

    y = j.y.to_numpy(float)
    base = y.mean()
    print(f"\nbase rate P(home win) = {base:.4f}  (n={len(j)})")
    print(f"mean bid/ask spread   = {j.spread.mean():.4f}  (edge below half this = {j.spread.mean()/2:.4f} is not tradeable)")
    print(f"\n{'model':<10} {'Brier':>9} {'BSS_vs_const':>13} {'BSS_vs_mkt':>11}")
    b_const = brier(np.full(len(y), base), y)
    b_mkt = brier(j.p_mkt.to_numpy(float), y)
    b_dl = brier(j.p_dl.to_numpy(float), y)
    print(f"{'constant':<10} {b_const:>9.5f} {0.0:>13.4f} {'--':>11}")
    print(f"{'market':<10} {b_mkt:>9.5f} {1-b_mkt/b_const:>13.4f} {0.0:>11.4f}")
    print(f"{'DL':<10} {b_dl:>9.5f} {1-b_dl/b_const:>13.4f} {1-b_dl/b_mkt:>11.4f}")
    print(f"\nVERDICT (pregame): DL beats market = {b_dl < b_mkt}  "
          f"(DL Brier {b_dl:.5f} vs market {b_mkt:.5f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
