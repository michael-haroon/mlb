#!/usr/bin/env python3
"""Trade-based, fill-honest backtest of the DL model against EVERY Kalshi MLB series in S3.

Generalizes scripts/kalshi_trades_backtest.py (home_win only) to the five other series under
s3://<bucket>/kalshi_history/: KXMLBRFI, KXMLBEXTRAS, KXMLBTOTAL, KXMLBTEAMTOTAL, KXMLBSPREAD.

WHY trades, not candles. A candle carries BOTH trade OHLC (price_*) and best-bid/best-ask OHLC
(yes_bid_*, yes_ask_*) for a one-minute window, so it does describe how the top of book moved --
but its mid is a *quote* you may never have been able to transact at, and price_* is NaN in every
minute with no trade (the common case). A TRADE print is ground truth: a taker crossed AND a
resting order existed at that price. `is_block_trade` (OTC/RFQ) prints are dropped because they
never touched the lit book.

NO PnL HERE, BY DECISION (2026-09-01). Earlier versions of this file simulated fills -- first taker,
then a resting two-sided maker quote. Both were withdrawn. You cannot reconstruct a Kalshi order
book from trade prints and 1-minute candles, so any fill model has to invent the one thing that
decides a maker's PnL: queue position at the top of book. The resting-quote version posted at a
fixed offset from fair regardless of where the book was, which meant it only ever filled on prints
that traded THROUGH the quote -- exactly the informed ones -- and structurally excluded the benign
fills a top-of-book maker gets from uninformed flow. Its answer was therefore biased by an unknown
amount in a known direction, which is not a measurement.
What survives is the readout below, which needs no fill model: it asks only whether the model's
number is a better forecast than the price the market actually transacted at. For the same question
against the top of book (best bid / best ask, not a mid), see kalshi_topbook_accuracy.py.

WHY real timestamps. Each DL prediction is indexed by prefix_length (cumulative game sequence
index), not wall-clock. Stage 1 attaches the ACTUAL pitch time by joining the feature store's
sequence rows to the raw pitch rows and reading pitch_start_time (raw MLB feed).

DERIVED PRICING. The model emits three probability heads (home_win, yrfi, extra_innings) plus a
NegBin pair over runs REMAINING (mu/alpha per side). Total / team-total / spread strikes are priced
by integrating that pair on top of the score already on the board:

    current_score = final_score - y_runs_remaining

That base is the score as-of the prefix. It is NOT leakage: `final` and `remaining` are two halves
of the same known decomposition and their difference is a pre-state quantity. It is deliberately
computed this way rather than read from pitch_sequences.score_home because score_home is POST-play
while the runs-remaining target is PRE-play, and the two disagree on ~1.5% of rows (rows where a
run scored on the current play). Subtracting the target guarantees the base the pricer adds is
exactly the base the target was defined against.

Tail handling. For a strike K and score-on-board c, with SF_x(k) = P(R_x > k) the closed-form
NegBin survival function and the a-marginal summed over a in [0, MAX_REM]:
    team total : P(c + R_T > K)      = SF_T(floor(K - c))                             -- exact
    total      : P(c + Rh + Ra > K)  = sum_a pmf_a(a) SF_h(floor(K-c-a)) + SF_a(MAX_REM)
    spread     : P(c + Rh - Ra > K)  = sum_a pmf_a(a) SF_h(floor(K-c+a))
The total form is EXACT, not truncated: the threshold K-c-a falls below zero for every a > K-c, so
every dropped term equals 1 and their total mass is exactly SF_a(MAX_REM). That identity holds for
any MAX_REM >= MAX_STRIKE, which is asserted at call time.

The spread form is the one place truncation bites, because there the threshold K-c+a *grows* with a
so the dropped terms are neither 0 nor 1. They are bracketed: the residual lies in
[0, SF_a(MAX_REM) * SF_h(K-c+MAX_REM)], so the pricer returns that product alongside the price and
the caller reports the worst bound it saw. This matters: at alpha=0.1, mu=13.2 (the readout's
heaviest tail) P(R > 80) is 4.7%, so a naive 80-run grid would have been wrong in the third decimal
of a probability -- an 80-run cap was the first version of this file and the unit tests caught it.
On the actual readout the bound is negligible: only 0.02% of rows have P(R > 200) > 1e-6.

INDEPENDENCE. Rh and Ra are convolved as independent. That is not an extra assumption layered on
the model -- it IS the model's implied joint, since the head emits two marginals and nothing else
(see distributions.joint_pmf_independent). Any home/away run covariance the true process has is
mispriced identically by the model in production, so the backtest measures the deployed object.

Readout: per series, per game-state bucket (bucket = prefix_length active at the trade), the
size-weighted Brier of the model fair as-of the trade against the Brier of the executed yes_price,
with a game-clustered bootstrap CI on the difference. Side-agnostic and fill-model-free.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.stats import nbinom

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "data/backtest"
MULTI = ROOT / "data/backtest/kalshi_multi"
GAME_TRADES = ROOT / "data/backtest/kalshi_game_trades"
RAW = ROOT / "data/backtest/raw_pitches_2025"
META = ROOT / "deep_learning/feature_store/game_meta.parquet"
SEQ = ROOT / "deep_learning/feature_store/pitch_sequences.parquet"

BUCKETS = [(0, 0), (1, 50), (51, 100), (101, 150), (151, 200), (201, 300), (301, 10000)]
MIN_TRADES = 30          # below this a bucket's bootstrap CI is uninformative; suppress the row
MAX_REM = 400            # runs-remaining support cap for the a-marginal sum (see module docstring)
MAX_STRIKE = 30          # widest total/team-total/spread line Kalshi lists

MON = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
# KXMLBTOTAL-26AUG311940MILCHC-18 / KXMLBGAME-25APR16ATHCWS-CWS / KXMLBRFI-25OCT17TORSEA
# HHMM is present only on 2026 tickers; the trailing digit on the team pair is the doubleheader
# game number; the strike suffix is absent entirely on KXMLBRFI.
TICKER = re.compile(r"^KX[A-Z]+-(\d{2})([A-Z]{3})(\d{2})(\d{4})?([A-Z]+)(\d)?(?:-(.+))?$")
# Kalshi used ARI before switching to MLB's AZ; every other code already matches game_meta.
CODE_FIX = {"ARI": "AZ"}


# ----------------------------------------------------------------------------------------------
# series configuration
# ----------------------------------------------------------------------------------------------
# kind drives both the pricer and the settlement rule:
#   "head"       -> a probability head read straight off the readout
#   "team_total" -> P(one side's final score > K)
#   "total"      -> P(combined final score > K)
#   "spread"     -> P(one side's final margin > K)
SERIES = {
    "KXMLBGAME": dict(kind="head", prob="p_home_win", y="y_home_win", mask=None,
                      dirs=[GAME_TRADES, MULTI / "KXMLBGAME"], side_from_suffix=True),
    "KXMLBRFI": dict(kind="head", prob="p_yrfi", y="y_yrfi", mask="yrfi_mask",
                     dirs=[MULTI / "KXMLBRFI"], side_from_suffix=False),
    "KXMLBEXTRAS": dict(kind="head", prob="p_extra_innings", y="y_extra_innings", mask=None,
                        dirs=[MULTI / "KXMLBEXTRAS"], side_from_suffix=False),
    "KXMLBTOTAL": dict(kind="total", dirs=[MULTI / "KXMLBTOTAL"], side_from_suffix=False),
    "KXMLBTEAMTOTAL": dict(kind="team_total", dirs=[MULTI / "KXMLBTEAMTOTAL"],
                           side_from_suffix=True),
    "KXMLBSPREAD": dict(kind="spread", dirs=[MULTI / "KXMLBSPREAD"], side_from_suffix=True),
}


def to_unix_s(x):
    s = pd.to_datetime(x, utc=True, errors="coerce")
    return ((s - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")).astype("float64")


def read_many(paths, want):
    """Concat parquet files, keeping only the columns each file actually has (schemas drift)."""
    parts = []
    for p in paths:
        have = set(pq.ParquetFile(p).schema_arrow.names)
        parts.append(pd.read_parquet(p, columns=[c for c in want if c in have]))
    if not parts:
        return pd.DataFrame(columns=list(want))
    return pd.concat(parts, ignore_index=True)


def series_files(dirs, kind):
    out = []
    for d in dirs:
        out += sorted(glob.glob(str(Path(d) / "**" / f"{kind}_batch_*.parquet"), recursive=True))
    return out


# ----------------------------------------------------------------------------------------------
# stage 1: real wall-clock per (game_pk, prefix_length)
# ----------------------------------------------------------------------------------------------
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


def build_final_scores(gpks):
    """Final home/away runs per game. score_home/score_away are monotone, so max == final."""
    s = ds.dataset(SEQ).to_table(
        filter=pc.field("game_pk").isin(gpks),
        columns=["game_pk", "score_home", "score_away"]).to_pandas()
    return s.groupby("game_pk")[["score_home", "score_away"]].max().rename(
        columns={"score_home": "final_h", "score_away": "final_a"}).reset_index()


# ----------------------------------------------------------------------------------------------
# stage 2: predictions
# ----------------------------------------------------------------------------------------------
def load_predictions():
    cols = ["game_pk", "prefix_length", "p_home_win", "p_yrfi", "p_extra_innings",
            "mu_home", "mu_away", "alpha_home", "alpha_away",
            "y_home_win", "y_yrfi", "y_extra_innings", "y_total_runs",
            "y_home_runs_remaining", "y_away_runs_remaining", "yrfi_mask"]
    preds = pd.concat(
        [pd.read_parquet(PRED / f"readout_fix_{s}.parquet", columns=cols).assign(split=s)
         for s in ("val", "test")], ignore_index=True)
    meta = pd.read_parquet(META, columns=[
        "game_pk", "game_date", "game_number", "game_type_code",
        "home_team_abbr", "away_team_abbr"])
    n0, g0 = len(preds), preds.game_pk.nunique()
    preds = preds.merge(meta, on="game_pk", how="inner")
    if len(preds) < n0:
        print(f"  dropped {n0 - len(preds):,} rows / {g0 - preds.game_pk.nunique()} games absent "
              f"from game_meta.parquet (local store is behind the readout; unjoinable to Kalshi "
              f"regardless, since the ticker key is date + team abbrs)")
    preds["game_date"] = pd.to_datetime(preds.game_date).dt.normalize()

    gpks = preds.game_pk.unique().tolist()
    fin = build_final_scores(gpks)
    preds = preds.merge(fin, on="game_pk", how="left")

    # Contract check: the readout's total target must equal the feature store's running score. Where
    # it does not, `final - remaining` is not the score on the board and every derived strike would
    # be priced off a fictional base. Quarantine those games instead of pricing them.
    #
    # KNOWN CAUSE (2026-09-01): the runs targets are contaminated on doubleheaders -- 43 of the 53
    # offending games have a sibling game with the same (date, home, away), vs 45 of 4,318 clean
    # games, and y_total_runs comes out at exactly 2x (or 3x) the game's real total. The target
    # builder is evidently keyed on date+teams somewhere instead of game_pk. Bug is in the feature
    # store, not here; this only refuses to launder it into a PnL number.
    bad = preds.final_h + preds.final_a != preds.y_total_runs
    if bad.any():
        gb = preds.loc[bad, "game_pk"].nunique()
        print(f"  QUARANTINED {int(bad.sum()):,} rows / {gb} games "
              f"({gb / preds.game_pk.nunique():.1%}) where y_total_runs != final score "
              f"(doubleheader target contamination -- see comment)")
        preds = preds[~bad]

    preds["base_h"] = preds.final_h - preds.y_home_runs_remaining
    preds["base_a"] = preds.final_a - preds.y_away_runs_remaining

    seqt = build_seq_time(gpks)
    preds = preds.merge(seqt.rename(columns={"sequence_index": "prefix_length"}),
                        on=["game_pk", "prefix_length"], how="left")
    preds = preds.dropna(subset=["ts"])   # prefixes with no matching raw pitch (rare)
    return preds.sort_values(["game_pk", "ts"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# stage 3: NegBin tables (one row per prediction row)
# ----------------------------------------------------------------------------------------------
class RunTables:
    """NegBin runs-remaining marginals for a compact set of prediction rows.

    Built per series over only the rows that trades actually reference, because the survival grid is
    (n_rows, KMAX+1) and KMAX has to reach MAX_STRIKE + MAX_REM for the spread sum. Over all 81k
    readout rows that would be gigabytes; over the rows a series touches it is megabytes.

    Every pricing method returns (probability, error_bound) so the caller can report exactly how
    much of the answer is truncation rather than model.
    """

    KMAX = MAX_STRIKE + MAX_REM

    def __init__(self, mu_h, alpha_h, mu_a, alpha_a):
        k = np.arange(self.KMAX + 1)
        rem = np.arange(MAX_REM + 1)
        self.sf_h, self.pmf_h = self._grids(mu_h, alpha_h, k, rem)
        self.sf_a, self.pmf_a = self._grids(mu_a, alpha_a, k, rem)

    @staticmethod
    def _grids(mu, alpha, k, rem):
        # clamps mirror negbin_nll's, so the pricer sees exactly the parameters the loss saw
        mu = np.clip(np.asarray(mu, float), 1e-6, None)[:, None]
        alpha = np.clip(np.asarray(alpha, float), 1e-3, None)[:, None]
        p = alpha / (alpha + mu)
        return nbinom.sf(k[None, :], alpha, p), nbinom.pmf(rem[None, :], alpha, p)

    def _sf_at(self, sf, rows, thresh):
        """P(R > thresh) for integer R; thresh may be fractional, negative, or past the grid."""
        idx = np.floor(thresh).astype(np.int64)
        if idx.max(initial=-1) > self.KMAX:
            raise AssertionError(f"threshold {idx.max()} exceeds KMAX={self.KMAX}")
        return np.where(idx < 0, 1.0, sf[rows, np.clip(idx, 0, self.KMAX)])

    def team_total_over(self, rows, base, strike, is_home):
        """P(base + R_side > strike). Closed form, so the bound is identically zero."""
        out = np.empty(len(rows))
        for flag, sf in ((True, self.sf_h), (False, self.sf_a)):
            m = is_home == flag
            if m.any():
                out[m] = self._sf_at(sf, rows[m], (strike - base)[m])
        return out, np.zeros(len(rows))

    def _sum_over_other(self, rows, thresh0, sign, pmf_other, sf_other, sf_target):
        """sum_o pmf_other(o) * P(R_target > thresh0 + sign*o) over o in [0, MAX_REM].

        sign=-1 (total): dropped terms are all exactly 1 once thresh0 - o < 0, so their mass
        SF_other(MAX_REM) is added back and the result is exact.
        sign=+1 (spread): dropped terms lie in [0, SF_target(thresh0+MAX_REM)], so the result is a
        lower bound and the residual is bounded by SF_other(MAX_REM)*SF_target(thresh0+MAX_REM).
        """
        o = np.arange(MAX_REM + 1)
        th = thresh0[:, None] + sign * o[None, :]
        idx = np.floor(th).astype(np.int64)
        inside = idx >= 0
        if idx.max(initial=-1) > self.KMAX:
            raise AssertionError(f"threshold {idx.max()} exceeds KMAX={self.KMAX}")
        tail = np.where(inside, sf_target[rows[:, None], np.clip(idx, 0, self.KMAX)], 1.0)
        p = (pmf_other[rows] * tail).sum(axis=1)
        other_tail = sf_other[rows, MAX_REM]
        if sign < 0:
            if (thresh0 > MAX_REM).any():
                raise AssertionError("MAX_REM < strike - base; the total identity needs MAX_REM >= it")
            return p + other_tail, np.zeros(len(rows))
        edge = self._sf_at(sf_target, rows, thresh0 + MAX_REM)
        return p, other_tail * edge

    def total_over(self, rows, base_total, strike):
        """P(base + Rh + Ra > strike)."""
        return self._sum_over_other(rows, strike - base_total, -1.0,
                                    self.pmf_a, self.sf_a, self.sf_h)

    def spread_over(self, rows, base_margin, strike, is_home):
        """P(base_margin + R_side - R_opp > strike), base_margin from the strike side's view."""
        out = np.empty(len(rows))
        err = np.empty(len(rows))
        for flag, pmf_o, sf_o, sf_t in ((True, self.pmf_a, self.sf_a, self.sf_h),
                                        (False, self.pmf_h, self.sf_h, self.sf_a)):
            m = is_home == flag
            if m.any():
                out[m], err[m] = self._sum_over_other(
                    rows[m], (strike - base_margin)[m], +1.0, pmf_o, sf_o, sf_t)
        return out, err


# ----------------------------------------------------------------------------------------------
# stage 4: ticker parsing and the trade -> (game, prefix) join
# ----------------------------------------------------------------------------------------------
def parse_tickers(tk: pd.Series) -> pd.DataFrame:
    ex = tk.str.extract(TICKER)
    ok = ex[0].notna()
    out = pd.DataFrame(index=tk.index)
    out["tkd"] = pd.NaT
    out.loc[ok, "tkd"] = pd.to_datetime(dict(
        year=2000 + ex.loc[ok, 0].astype(int),
        month=ex.loc[ok, 1].map(MON),
        day=ex.loc[ok, 2].astype(int)))
    out["pair"] = ex[4].replace(CODE_FIX, regex=False)
    for bad, good in CODE_FIX.items():
        out["pair"] = out.pair.str.replace(bad, good, regex=False)
    out["game_number"] = pd.to_numeric(ex[5], errors="coerce").fillna(1).astype(int)
    out["suffix"] = ex[6]
    return out


def attach_games(td: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Map each trade's ticker to a game_pk via (date, away_abbr+home_abbr, game_number).

    Orientation is away-then-home; verified against 479/479 parseable 2025 KXMLBGAME tickers.
    """
    g = preds.groupby("game_pk")[["game_date", "game_number", "home_team_abbr",
                                  "away_team_abbr"]].first().reset_index()
    key = {(d, f"{a}{h}", int(n)): int(gp)
           for gp, d, n, h, a in zip(g.game_pk, g.game_date, g.game_number,
                                     g.home_team_abbr, g.away_team_abbr)}
    td["game_pk"] = [key.get((d, p, n)) for d, p, n in zip(td.tkd, td.pair, td.game_number)]
    return td


def asof_join(td: pd.DataFrame, preds: pd.DataFrame, carry: list[str]) -> pd.DataFrame:
    """Give each trade the model's latest state at or before the trade time, per game."""
    parts = []
    by_game = {g: d for g, d in preds.groupby("game_pk")}
    for gpk, tg in td.groupby("game_pk"):
        pg = by_game.get(gpk)
        if pg is None or pg.empty:
            continue
        pos = np.searchsorted(pg.ts.to_numpy(), tg.ts.to_numpy(), side="right") - 1
        ok = pos >= 0
        if not ok.any():
            continue
        tg = tg.loc[ok].copy()
        pos = pos[ok]
        tg["row"] = pg.index.to_numpy()[pos]
        for c in carry:
            tg[c] = pg[c].to_numpy()[pos]
        parts.append(tg)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_trades(cfg) -> pd.DataFrame:
    files = series_files(cfg["dirs"], "trades")
    if not files:
        return pd.DataFrame()
    want = ["ticker", "created_time", "yes_price_dollars", "count_fp",
            "taker_book_side", "is_block_trade"]
    td = read_many(files, want)
    if "is_block_trade" in td:
        td = td[~td.is_block_trade.fillna(False)]   # lit book only (OTC/RFQ excluded)
    td["P"] = pd.to_numeric(td.yes_price_dollars, errors="coerce")
    td["size"] = pd.to_numeric(td.count_fp, errors="coerce")
    td["ts"] = to_unix_s(td.created_time)
    td = td.join(parse_tickers(td.ticker))
    return td.dropna(subset=["P", "size", "ts", "tkd", "pair"])


def load_markets(cfg) -> pd.DataFrame:
    """Strike + settlement per market ticker. Authoritative over anything parsed from the suffix."""
    files = series_files(cfg["dirs"], "markets")
    if not files:
        return pd.DataFrame(columns=["ticker", "floor_strike", "strike_type", "result"])
    m = read_many(files, ["ticker", "floor_strike", "strike_type", "result"])
    return m.drop_duplicates("ticker", keep="last")


# ----------------------------------------------------------------------------------------------
# stage 5: pricing + settlement per series
# ----------------------------------------------------------------------------------------------
def suffix_team(t):
    """The suffix's leading team code, normalized to the feature store's abbreviations.

    CODE_FIX has to be applied HERE and not only to the ticker's team pair. It was originally applied
    to the pair alone, so an away-ARI ticker matched neither `home_team_abbr` (AZ) nor
    `away_team_abbr` and silently kept the home orientation -- inverting both its fair and its
    settlement. 22 of 4,264 KXMLBGAME tickers; found by the Kalshi `result` cross-check in
    kalshi_topbook_accuracy.py, which is the only reason it was visible at all.
    """
    return t.suffix.str.extract(r"^([A-Z]+)")[0].replace(CODE_FIX)


def price_head(cfg, t):
    """Model fair + settlement for a probability head, oriented to the ticker's YES side.

    Returns (f, y, keep). `keep` is all-True unless the side comes from the suffix and the suffix
    resolves to neither team; those rows must be dropped rather than defaulted to a side, because a
    silent default is exactly what hid the ARI bug (see suffix_team).
    """
    f = t[cfg["prob"]].astype(float).to_numpy().copy()
    y = t[cfg["y"]].astype(float).to_numpy().copy()
    keep = np.ones(len(t), bool)
    if cfg["side_from_suffix"]:
        # A YES on KXMLBGAME is the named team, which may be the away side.
        team = suffix_team(t)
        away = (team == t.away_team_abbr).to_numpy()
        keep = away | (team == t.home_team_abbr).to_numpy()
        f[away], y[away] = 1.0 - f[away], 1.0 - y[away]
    return f, y, keep


def strike_side(t):
    """is_home per row from the ticker suffix's leading team code, plus a keep mask.

    Rows whose suffix names neither team are dropped: that is a parse failure, not a market, and
    pricing it against an arbitrary side would be silent garbage.
    """
    team = suffix_team(t)
    is_home = (team == t.home_team_abbr).to_numpy()
    keep = is_home | (team == t.away_team_abbr).to_numpy()
    return is_home[keep], keep


def price_derived(kind, preds, t, is_home):
    """Model fair + settlement for total / team_total / spread strikes.

    Shared by the trade-price readout and kalshi_topbook_accuracy.py -- one pricer, two consumers,
    because two copies of a NegBin marginalization will drift and only one of them will be tested.
    `t` must carry row / K / base_h / base_a / final_h / final_a / y_total_runs.
    """
    used = np.sort(t.row.unique())
    tables = RunTables(*(preds.loc[used, c].to_numpy()
                         for c in ("mu_home", "alpha_home", "mu_away", "alpha_away")))
    rows = np.searchsorted(used, t.row.to_numpy())
    K = t.K.to_numpy(float)
    bh, ba = t.base_h.to_numpy(float), t.base_a.to_numpy(float)
    fh, fa = t.final_h.to_numpy(float), t.final_a.to_numpy(float)
    if kind == "total":
        f, err = tables.total_over(rows, bh + ba, K)
        y = t.y_total_runs.to_numpy(float) > K
    elif kind == "team_total":
        f, err = tables.team_total_over(rows, np.where(is_home, bh, ba), K, is_home)
        y = np.where(is_home, fh, fa) > K
    else:
        margin = np.where(is_home, bh - ba, ba - bh)
        f, err = tables.spread_over(rows, margin, K, is_home)
        y = np.where(is_home, fh - fa, fa - fh) > K
    return f, y.astype(float), err


def price_series(name, cfg, preds):
    td = load_trades(cfg)
    if td.empty:
        return None, f"{name}: no trade files on disk"
    n_raw = len(td)
    td = attach_games(td, preds)
    td = td.dropna(subset=["game_pk"])
    if td.empty:
        return None, (f"{name}: {n_raw:,} lit trades, none on a game in the prediction window "
                      f"({preds.game_date.min().date()}..{preds.game_date.max().date()})")
    td["game_pk"] = td.game_pk.astype(int)

    carry = ["prefix_length", "base_h", "base_a", "final_h", "final_a", "y_total_runs",
             "game_type_code", "home_team_abbr", "away_team_abbr", "split"]
    if cfg["kind"] == "head":
        carry += [cfg["prob"], cfg["y"]] + ([cfg["mask"]] if cfg["mask"] else [])
    t = asof_join(td.sort_values(["game_pk", "ts"]), preds, carry)
    if t.empty:
        return None, f"{name}: {n_raw:,} lit trades matched games but none had a prior model state"

    if cfg["kind"] == "head":
        if cfg["mask"]:
            t = t[t[cfg["mask"]].astype(bool)]      # head is only defined on masked prefixes
            if t.empty:
                return None, f"{name}: no trades land on a prefix where {cfg['prob']} is defined"
        f, y, keep = price_head(cfg, t)
        if not keep.all():
            print(f"  {name}: dropped {int((~keep).sum()):,} trades whose ticker suffix matched "
                  f"neither team abbreviation")
        t = t.loc[keep].copy()
        t["f"], t["y"] = f[keep], y[keep]
        if t.empty:
            return None, f"{name}: no trade's ticker suffix resolved to either team"
    else:
        mk = load_markets(cfg)
        t = t.merge(mk, on="ticker", how="left")
        t["K"] = pd.to_numeric(t.floor_strike, errors="coerce")
        t = t.dropna(subset=["K"])
        if t.empty:
            return None, f"{name}: no strike metadata for any matched trade"
        if t.K.max() > MAX_STRIKE:
            raise SystemExit(f"{name}: strike {t.K.max()} exceeds MAX_STRIKE={MAX_STRIKE}")

        # a total has no side; price_derived ignores is_home for that kind but still takes the arg
        is_home = np.zeros(len(t), bool)
        if cfg["kind"] != "total":
            is_home, keep = strike_side(t)
            t = t.loc[keep].copy()
            if t.empty:
                return None, f"{name}: strike side never matched either team abbreviation"

        t["f"], t["y"], t["trunc_err"] = price_derived(cfg["kind"], preds, t, is_home)

    t["se_f"] = (t.f - t.y) ** 2
    t["se_p"] = (t.P - t.y) ** 2
    return t, None


# ----------------------------------------------------------------------------------------------
# stage 7: reporting
# ----------------------------------------------------------------------------------------------
def ratio_ci(b, num_col, den_col, rng, n=2000):
    """Game-clustered, size-weighted bootstrap of the AGGREGATE ratio sum(num*den)/sum(den).

    Unit of resampling = game, so within-game correlation and trade size are both respected and
    the point estimate (full-sample ratio) is the bootstrap mean by construction.
    """
    per = b.groupby("game_pk").apply(
        lambda g: pd.Series({"num": (g[num_col] * g[den_col]).sum(), "den": g[den_col].sum()}),
        include_groups=False)
    num, den = per.num.to_numpy(), per.den.to_numpy()
    point = num.sum() / den.sum()
    idx = np.arange(len(num))
    bs = np.empty(n)
    for k in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        bs[k] = num[s].sum() / den[s].sum()
    return point, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def bucket_label(lo, hi):
    return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10000 else f"{lo}+")


def report(name, t, rng):
    gt = t.groupby("game_type_code").game_pk.nunique().to_dict()
    print(f"\n{'=' * 104}\n{name}")
    print(f"  {len(t):,} lit trades on {t.game_pk.nunique()} games "
          f"({t.game_date_min} .. {t.game_date_max})  game types {gt}  "
          f"splits {t.split.value_counts().to_dict()}")
    print(f"  {int(t['size'].sum()):,} contracts printed")
    if "trunc_err" in t:
        print(f"  worst NegBin truncation bound on any priced strike: {t.trunc_err.max():.2e}")

    print("\n  TRADE-PRICE ACCURACY  (Brier vs executed yes-price; size-weighted, clustered CI)")
    print(f"  {'bucket':>10} {'trades':>8} {'Brier_DL':>9} {'Brier_px':>9} {'edge':>9} {'edge_95CI':>20}")
    for lo, hi in BUCKETS:
        b = t[(t.prefix_length >= lo) & (t.prefix_length <= hi)]
        if len(b) < MIN_TRADES:
            continue
        w = b["size"].to_numpy()
        bd, bp = np.average(b.se_f, weights=w), np.average(b.se_p, weights=w)
        edge, lo_ci, hi_ci = ratio_ci(b.assign(d=b.se_p - b.se_f), "d", "size", rng)
        print(f"  {bucket_label(lo, hi):>10} {len(b):>8} {bd:>9.5f} {bp:>9.5f} {edge:>+9.5f} "
              f"[{lo_ci:>+8.5f},{hi_ci:>+8.5f}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="*", default=list(SERIES),
                    help="subset of series to run (default: all)")
    args = ap.parse_args()

    preds = load_predictions()
    print(f"predictions: {len(preds):,} prefix rows on {preds.game_pk.nunique()} games, "
          f"{preds.game_date.min().date()} .. {preds.game_date.max().date()}")

    rng = np.random.default_rng(0)
    skipped = []
    for name in args.series:
        cfg = SERIES[name]
        t, why = price_series(name, cfg, preds)
        if t is None:
            skipped.append(why)
            continue
        d = preds.set_index("game_pk").game_date
        t.game_date_min = d.loc[t.game_pk].min().date()
        t.game_date_max = d.loc[t.game_pk].max().date()
        report(name, t, rng)

    if skipped:
        print(f"\n{'=' * 104}\nNO OVERLAP")
        for s in skipped:
            print(f"  - {s}")
    print("\nreal pitch timestamps (raw feed); lit-book trades only (is_block_trade dropped, so no "
          "OTC/RFQ). Accuracy only -- no fill model, no PnL. Positive edge = the model's fair beat "
          "the price the market actually transacted at.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
