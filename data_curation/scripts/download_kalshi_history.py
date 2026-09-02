"""
download_kalshi_history.py
---------------------------
Historical Kalshi MLB market archive: per-game series metadata + either 1-minute
candlesticks or time & sales (trades), from market open through settlement, for
every settled market across the 6 real per-game MLB series (see
trading/market_map.MODEL_TO_SERIES).

Storage mirrors data_curation/scripts/download_history.py: S3 by default,
--local for disk. Checkpointed by market ticker so a killed/restarted run
resumes instead of re-pulling. Candles and trades use separate checkpoint
namespaces (--data-type candles|trades) so running one doesn't mark tickers
"done" for the other.

Dry run (default, no --live) only does discovery: paginate /markets per
series, log market counts + open/close time ranges + one sample payload for
the selected --data-type. Nothing is written.

Run:
    conda run -n pred python data_curation/scripts/download_kalshi_history.py                       # dry run, candles
    conda run -n pred python data_curation/scripts/download_kalshi_history.py --live                 # full candles pull
    conda run -n pred python data_curation/scripts/download_kalshi_history.py --live --data-type trades
    conda run -n pred python data_curation/scripts/download_kalshi_history.py --live --retry
    conda run -n pred python data_curation/scripts/download_kalshi_history.py --local                # write to disk instead of S3
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from trading.kalshi_client import make_client, RateLimitError  # noqa: E402
from trading.market_map import MODEL_TO_SERIES  # noqa: E402

# --- INGESTION TARGET CONSTANTS ---
SERIES_TICKERS = sorted({s for s in MODEL_TO_SERIES.values() if s})  # the 6 real per-game series
MAX_WORKERS = 12
# Starting point only — RateLimiter below adapts from here based on observed 429s.
# The on-paper Basic-tier ceiling (200 read tokens/sec / 10-token default cost = 20/s,
# confirmed via GET /account/endpoint_costs) measured ~35-40% 429 rejections at 12.5/s
# in practice, so the account's real available budget is lower than that ceiling
# (likely shared with another consumer of the same KALSHI_READ_KEY). This starting
# value is backed into roughly where that measurement implied the real limit sits.
RATE_LIMIT_DELAY = 0.15
CANDLE_PERIOD_INTERVAL_MIN = 1  # 1-minute candles; fine enough to later align to per-pitch DL predictions
MARKET_FLUSH_THRESHOLD = 300
DATA_FLUSH_THRESHOLD = 8000  # candles or trades, whichever --data-type is active
SUBMIT_BATCH_SIZE = 150  # cap in-flight futures so results awaiting flush don't pin unbounded memory

# Storage config — S3 is default; --local overrides to local disk
S3_BUCKET = "mlb-265753586044-us-east-1-an"
S3_PREFIX = "kalshi_history"
S3_REGION = "us-east-1"
USE_S3 = True
_s3_client = None

DATA_DIR = "data"
LOG_DIR = os.path.join(DATA_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("KALSHI_HISTORY")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "kalshi_history.log"))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s"))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("[KALSHI HISTORY] %(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"))

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ---------------------------------------------------------------------------
# STORAGE LAYER — unified S3 / local abstraction (mirrors download_history.py)
# ---------------------------------------------------------------------------
def _get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=S3_REGION)
    return _s3_client


def _s3_key(rel_path: str) -> str:
    return f"{S3_PREFIX}/{rel_path}"


def _read_json_store(rel_path: str):
    if USE_S3:
        try:
            obj = _get_s3().get_object(Bucket=S3_BUCKET, Key=_s3_key(rel_path))
            return json.loads(obj["Body"].read().decode("utf-8"))
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise
    else:
        local = os.path.join(DATA_DIR, rel_path)
        if not os.path.exists(local):
            return None
        with open(local) as f:
            return json.load(f)


def _write_json_store(rel_path: str, data: Any):
    if USE_S3:
        _get_s3().put_object(
            Bucket=S3_BUCKET, Key=_s3_key(rel_path),
            Body=json.dumps(data, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        logger.debug(f"[store] Written s3://{S3_BUCKET}/{_s3_key(rel_path)}")
    else:
        local = os.path.join(DATA_DIR, rel_path)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        tmp = local + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, local)
        logger.debug(f"[store] Written {local}")


def _save(df: pd.DataFrame, rel_path: str):
    if df.empty:
        return
    if USE_S3:
        key = _s3_key(rel_path)
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
        buf.seek(0)
        _get_s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
        logger.debug(f"[save] {len(df)} rows -> s3://{S3_BUCKET}/{key}")
    else:
        full = os.path.join(DATA_DIR, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        df.to_parquet(full, engine="pyarrow", compression="snappy", index=False)
        logger.debug(f"[save] {len(df)} rows -> {full}")


def save_markets(records: List[Dict[str, Any]], series_ticker: str):
    if not records:
        return
    _save(pd.DataFrame(records), f"{series_ticker}/markets_batch_{int(time.time() * 1000)}.parquet")


def save_candlesticks(records: List[Dict[str, Any]], series_ticker: str):
    if not records:
        return
    # json_normalize flattens Kalshi's nested price/yes_bid/yes_ask candle sub-objects
    _save(pd.json_normalize(records, sep="_"), f"{series_ticker}/candlesticks_batch_{int(time.time() * 1000)}.parquet")


def save_trades(records: List[Dict[str, Any]], series_ticker: str):
    if not records:
        return
    # Trade records are flat (count_fp, created_time, taker_side, yes_price_dollars, ...)
    _save(pd.DataFrame(records), f"{series_ticker}/trades_batch_{int(time.time() * 1000)}.parquet")


# ---------------------------------------------------------------------------
# CHECKPOINT MANAGER — keyed by market ticker (mirrors download_history.py)
# candles and trades use separate file prefixes so one doesn't mark tickers
# "done" for the other.
# ---------------------------------------------------------------------------
class CheckpointManager:
    def __init__(self, prefix: str = ""):
        self.checkpoint_rel = f"{prefix}checkpoint.json"
        self.retry_rel = f"{prefix}retry_queue.json"
        self._lock = threading.Lock()
        self.completed: Set[str] = set()
        self.retry_queue: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        data = _read_json_store(self.checkpoint_rel)
        if data:
            self.completed = set(data.get("completed", []))
            logger.info(f"[checkpoint] Loaded {len(self.completed)} completed tickers from {self.checkpoint_rel}.")
        else:
            logger.info(f"[checkpoint] No existing {self.checkpoint_rel} — starting fresh.")

        retry_data = _read_json_store(self.retry_rel)
        if retry_data:
            self.retry_queue = retry_data
            logger.info(f"[checkpoint] Loaded {len(self.retry_queue)} tickers in {self.retry_rel}.")

    def is_completed(self, ticker: str) -> bool:
        return ticker in self.completed

    def mark_completed(self, ticker: str):
        with self._lock:
            self.completed.add(ticker)
            self._flush_checkpoint()

    def mark_failed(self, ticker: str, series_ticker: str, reason: str, error: str):
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            existing = {e["ticker"]: e for e in self.retry_queue}
            if ticker in existing:
                e = existing[ticker]
                e["attempts"] += 1; e["last_error"] = error
                e["reason"] = reason; e["last_failed"] = now
            else:
                self.retry_queue.append({
                    "ticker": ticker, "series_ticker": series_ticker, "reason": reason,
                    "attempts": 1, "last_error": error,
                    "first_failed": now, "last_failed": now,
                })
            self._flush_retry()

    def clear_retry_entry(self, ticker: str):
        with self._lock:
            self.retry_queue = [e for e in self.retry_queue if e["ticker"] != ticker]
            self.completed.add(ticker)
            self._flush_checkpoint()
            self._flush_retry()

    def get_retry_markets(self) -> List[Dict[str, Any]]:
        return [{"ticker": e["ticker"], "series_ticker": e["series_ticker"]} for e in self.retry_queue]

    def discard_retry_entry(self, ticker: str):
        """Drop a ticker that's confirmed permanently unresolvable (e.g. a settled-market
        listing whose individual market 404s even on direct lookup — Kalshi-side data
        inconsistency, not a transient failure) — leaving it would retry forever."""
        with self._lock:
            self.retry_queue = [e for e in self.retry_queue if e["ticker"] != ticker]
            self._flush_retry()

    def _flush_checkpoint(self):
        _write_json_store(self.checkpoint_rel, {
            "completed": sorted(self.completed),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        })

    def _flush_retry(self):
        _write_json_store(self.retry_rel, self.retry_queue)


# ---------------------------------------------------------------------------
# RATE LIMITER — shared across worker threads so wall-clock request spacing
# stays under Kalshi's cap regardless of thread count.
#
# Adaptive: a fixed guessed delay (12.5 req/s, derived from Basic tier's stated
# 200 read-tokens/sec / 10-token default cost) measured ~35-40% 429 rejections
# in practice — the account's real available budget doesn't match that on-paper
# ceiling (likely shared with another consumer of the same key). Rather than
# guess a second fixed constant, this backs off multiplicatively on 429 and
# eases up gradually after a run of successes, so it tracks the real limit.
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, initial_delay: float, min_delay: float = 0.02, max_delay: float = 1.0):
        self.delay = initial_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self._last = 0.0
        self._lock = threading.Lock()
        self._success_streak = 0

    def wait(self):
        with self._lock:
            now = time.time()
            wait = max(0.0, self._last + self.delay - now)
            self._last = now + wait
        if wait:
            time.sleep(wait)

    def on_429(self):
        with self._lock:
            self.delay = min(self.max_delay, self.delay * 1.5)
            self._success_streak = 0
            logger.debug(f"[ratelimit] 429 -> delay={self.delay:.3f}s ({1/self.delay:.1f}/s)")

    def on_success(self):
        with self._lock:
            self._success_streak += 1
            if self._success_streak >= 50:
                self._success_streak = 0
                new_delay = max(self.min_delay, self.delay * 0.9)
                if new_delay != self.delay:
                    self.delay = new_delay
                    logger.debug(f"[ratelimit] 50 successes -> delay={self.delay:.3f}s ({1/self.delay:.1f}/s)")


_rate_limiter = RateLimiter(RATE_LIMIT_DELAY)


# ---------------------------------------------------------------------------
# Shared retry/backoff wrapper — every client call must go through this rather
# than call _rate_limiter.wait() directly. discover_markets originally called
# the API bare (no try/except) and an uncaught 429 killed the whole process
# mid-run — this closes that gap for every remaining unwrapped call site too.
# ---------------------------------------------------------------------------
def _call_with_backoff(label: str, fn, *args, **kwargs):
    max_retries = 5
    backoff_factor = 2.0
    for attempt in range(max_retries):
        try:
            _rate_limiter.wait()
            resp = fn(*args, **kwargs)
            _rate_limiter.on_success()
            return resp
        except RateLimitError as e:
            _rate_limiter.on_429()
            sleep_time = backoff_factor ** attempt
            logger.warning(f"[{label}] rate limited, backing off {sleep_time}s ({e})")
            time.sleep(sleep_time)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"[{label}] failed after {max_retries} attempts: {e}", exc_info=True)
                raise
            sleep_time = backoff_factor ** attempt
            logger.warning(f"[{label}] error: {e}. retry in {sleep_time}s")
            time.sleep(sleep_time)
    raise RuntimeError(f"[{label}] exhausted retries")


# ---------------------------------------------------------------------------
# DISCOVERY
# ---------------------------------------------------------------------------
def discover_markets(client, series_ticker: str) -> List[Dict[str, Any]]:
    """Paginate /markets?series_ticker=X&status=settled until cursor exhausts."""
    markets: List[Dict[str, Any]] = []
    cursor = None
    page = 0
    while True:
        resp = _call_with_backoff(
            f"discover:{series_ticker}", client.get_markets,
            series_ticker=series_ticker, status="settled", limit=200, cursor=cursor,
        )
        batch = resp.get("markets", [])
        markets.extend(batch)
        page += 1
        cursor = resp.get("cursor")
        logger.debug(f"[discover:{series_ticker}] page={page} batch={len(batch)} total={len(markets)} cursor={cursor!r}")
        if not cursor or not batch:
            break
    return markets


def _ts(iso_str: Optional[str]) -> Optional[int]:
    """Kalshi timestamps are ISO8601; the candlestick endpoint wants unix seconds."""
    if not iso_str:
        return None
    return int(datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp())


# ---------------------------------------------------------------------------
# CANDLESTICK FETCH (threaded, retried)
# ---------------------------------------------------------------------------
# Kalshi caps a single candlestick request at 5000 periods (confirmed via the API's
# own error body: "requested time range with candlesticks: 5894.55, max candlesticks:
# 5000"). Markets that stay open unusually long (postponed/rescheduled games) exceed
# that at 1-minute resolution, so long spans are split into sequential sub-requests
# instead of dropped or degraded to coarser granularity.
MAX_CANDLES_PER_REQUEST = 4900  # margin under Kalshi's 5000 cap for boundary rounding


def _fetch_candlestick_window(client, series_ticker: str, ticker: str, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
    """One candlestick request (<= MAX_CANDLES_PER_REQUEST periods), retried on transient errors."""
    max_retries = 5
    backoff_factor = 2.0
    for attempt in range(max_retries):
        try:
            _rate_limiter.wait()
            resp = client.get_candlesticks(
                series_ticker, ticker, start_ts, end_ts,
                period_interval=CANDLE_PERIOD_INTERVAL_MIN,
            )
            _rate_limiter.on_success()
            return resp.get("candlesticks", [])
        except RateLimitError as e:
            _rate_limiter.on_429()
            sleep_time = backoff_factor ** attempt
            logger.warning(f"[candles:{ticker}] rate limited, backing off {sleep_time}s ({e})")
            time.sleep(sleep_time)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"[candles:{ticker}] failed after {max_retries} attempts: {e}", exc_info=True)
                raise
            sleep_time = backoff_factor ** attempt
            logger.warning(f"[candles:{ticker}] error: {e}. retry in {sleep_time}s")
            time.sleep(sleep_time)
    return []


def fetch_candlesticks_for_market(client, series_ticker: str, market: Dict[str, Any]) -> List[Dict[str, Any]]:
    ticker = market["ticker"]
    start_ts = _ts(market.get("open_time"))
    end_ts = _ts(market.get("close_time")) or _ts(market.get("expiration_time"))
    if start_ts is None or end_ts is None:
        logger.warning(f"[candles:{ticker}] missing open_time/close_time — skipping")
        return []

    step_seconds = MAX_CANDLES_PER_REQUEST * 60 * CANDLE_PERIOD_INTERVAL_MIN
    windows = []
    cur = start_ts
    while cur < end_ts:
        nxt = min(cur + step_seconds, end_ts)
        windows.append((cur, nxt))
        cur = nxt + 60 * CANDLE_PERIOD_INTERVAL_MIN  # skip one period to avoid a duplicate boundary candle

    all_candles: List[Dict[str, Any]] = []
    for s, e in windows:
        all_candles.extend(_fetch_candlestick_window(client, series_ticker, ticker, s, e))

    for c in all_candles:
        c["market_ticker"] = ticker
        c["series_ticker"] = series_ticker
    return all_candles


# ---------------------------------------------------------------------------
# TRADES FETCH (threaded, retried) — time & sales / individual fills.
# Filtering by ticker alone bounds results to that market's lifetime; no
# time-range cap like /candlesticks has, so this is plain cursor pagination.
# ---------------------------------------------------------------------------
def fetch_trades_for_market(client, series_ticker: str, market: Dict[str, Any]) -> List[Dict[str, Any]]:
    ticker = market["ticker"]
    all_trades: List[Dict[str, Any]] = []
    cursor = None
    max_retries = 5
    backoff_factor = 2.0
    while True:
        for attempt in range(max_retries):
            try:
                _rate_limiter.wait()
                resp = client.get_trades(ticker=ticker, limit=200, cursor=cursor)
                _rate_limiter.on_success()
                break
            except RateLimitError as e:
                _rate_limiter.on_429()
                sleep_time = backoff_factor ** attempt
                logger.warning(f"[trades:{ticker}] rate limited, backing off {sleep_time}s ({e})")
                time.sleep(sleep_time)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"[trades:{ticker}] failed after {max_retries} attempts: {e}", exc_info=True)
                    raise
                sleep_time = backoff_factor ** attempt
                logger.warning(f"[trades:{ticker}] error: {e}. retry in {sleep_time}s")
                time.sleep(sleep_time)
        else:
            break  # exhausted retries without breaking out — resp undefined, stop paginating

        batch = resp.get("trades", [])
        all_trades.extend(batch)
        cursor = resp.get("cursor")
        if not cursor or not batch:
            break

    for t in all_trades:
        t["series_ticker"] = series_ticker
    return all_trades


# ---------------------------------------------------------------------------
# MAIN CRAWL — generalized over --data-type (candles vs trades) via fetch_fn/save_fn.
# ---------------------------------------------------------------------------
def run_series(client, series_ticker: str, checkpoint: CheckpointManager, fetch_fn, save_fn,
               data_label: str, is_retry: bool = False):
    if is_retry:
        targets = [m for m in checkpoint.get_retry_markets() if m["series_ticker"] == series_ticker]
        if not targets:
            return
        # Retry queue only stores the ticker, not the full market object (which
        # carries open_time/close_time needed for the candlestick call) — refetch it.
        markets = []
        for t in targets:
            try:
                resp = _call_with_backoff(f"retry-refetch:{t['ticker']}", client.get_market, t["ticker"])
            except Exception as e:
                # A settled-market listing can reference a ticker that 404s even on a
                # direct lookup (confirmed on ~210 KXMLBTEAMTOTAL markets) — that's a
                # permanent, not transient, failure. Drop it rather than retry forever.
                logger.warning(f"[retry-refetch:{series_ticker}] {t['ticker']} unresolvable, dropping: {e}")
                checkpoint.discard_retry_entry(t["ticker"])
                continue
            m = resp.get("market")
            if m:
                markets.append(m)
    else:
        markets = discover_markets(client, series_ticker)
        if not markets:
            logger.info(f"[{series_ticker}] no settled markets found")
            return
        close_times = sorted(m.get("close_time") for m in markets if m.get("close_time"))
        logger.info(
            f"[{series_ticker}] {len(markets)} settled markets | "
            f"close_time range: {close_times[0]} .. {close_times[-1]}"
        )
        markets = [m for m in markets if not checkpoint.is_completed(m["ticker"])]
        logger.info(f"[{series_ticker}] {len(markets)} pending after checkpoint filter")

    if not markets:
        return

    market_buf: List[Dict[str, Any]] = []
    data_buf: List[Dict[str, Any]] = []
    pending: List[str] = []

    def _flush(label: str):
        if not pending:
            return
        try:
            save_markets(market_buf, series_ticker)
            save_fn(data_buf, series_ticker)
            for ticker in pending:
                if is_retry:
                    checkpoint.clear_retry_entry(ticker)
                else:
                    checkpoint.mark_completed(ticker)
            logger.debug(f"[flush:{series_ticker}:{label}] checkpointed {len(pending)} markets")
        except Exception as e:
            logger.error(f"[flush:{series_ticker}:{label}] save failed: {e}", exc_info=True)
            for ticker in pending:
                checkpoint.mark_failed(ticker, series_ticker, type(e).__name__, str(e))
            raise
        finally:
            market_buf.clear(); data_buf.clear(); pending.clear()

    pbar = tqdm(total=len(markets), desc=f"{series_ticker} {data_label} ({'retry' if is_retry else 'live'})")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix=f"Kalshi-{series_ticker}") as executor:
        # Submit in a rolling window capped at SUBMIT_BATCH_SIZE in-flight futures —
        # submitting the whole series at once (12k+ for KXMLBTEAMTOTAL) let unconsumed
        # results pile up in memory on a 2GB box with no swap and locked it out of SSH.
        market_iter = iter(markets)
        in_flight: Dict = {}

        def _fill_queue():
            while len(in_flight) < SUBMIT_BATCH_SIZE:
                m = next(market_iter, None)
                if m is None:
                    break
                f = executor.submit(fetch_fn, client, series_ticker, m)
                in_flight[f] = m

        _fill_queue()
        while in_flight:
            future = next(as_completed(in_flight))
            m = in_flight.pop(future)
            ticker = m["ticker"]
            try:
                records = future.result()
                market_buf.append(m)
                data_buf.extend(records)
                pending.append(ticker)
                pbar.update(1)
                pbar.set_postfix({data_label: len(data_buf), "pending": len(pending)})
                if len(data_buf) >= DATA_FLUSH_THRESHOLD or len(market_buf) >= MARKET_FLUSH_THRESHOLD:
                    _flush("threshold")
            except Exception as e:
                logger.error(f"[{series_ticker}] worker failed for {ticker}: {e}")
                checkpoint.mark_failed(ticker, series_ticker, type(e).__name__, str(e))
                pbar.update(1)
            _fill_queue()
    pbar.close()
    _flush("final")


def dry_run(client, fetch_fn, data_label: str):
    logger.info(f"=== DRY RUN ({data_label}): discovery only, no writes ===")
    for series_ticker in SERIES_TICKERS:
        markets = discover_markets(client, series_ticker)
        if not markets:
            logger.info(f"[{series_ticker}] 0 settled markets found")
            continue
        close_times = sorted(m.get("close_time") for m in markets if m.get("close_time"))
        logger.info(
            f"[{series_ticker}] {len(markets)} settled markets | "
            f"close_time range: {close_times[0]} .. {close_times[-1]}"
        )
        sample = markets[0]
        logger.info(f"[{series_ticker}] sample market fields: {list(sample.keys())}")
        logger.info(f"[{series_ticker}] sample market: {json.dumps(sample, indent=2, default=str)}")
        records = fetch_fn(client, series_ticker, sample)
        logger.info(f"[{series_ticker}] sample {data_label} count: {len(records)}")
        if records:
            logger.info(f"[{series_ticker}] sample {data_label} fields: {list(records[0].keys())}")
            logger.info(f"[{series_ticker}] sample {data_label} record: {json.dumps(records[0], indent=2, default=str)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Historical Kalshi MLB market archive.")
    parser.add_argument("--live", action="store_true", help="Disable dry run and pull + persist all settled markets.")
    parser.add_argument("--retry", action="store_true", help="Process only markets in the retry queue.")
    parser.add_argument("--local", action="store_true", help="Write to local disk instead of S3.")
    parser.add_argument("--series", type=str, default=None, help="Comma-separated series tickers to restrict to (for testing).")
    parser.add_argument("--env", type=str, default="prod", choices=["prod", "demo"], help="Kalshi environment.")
    parser.add_argument("--data-type", type=str, default="candles", choices=["candles", "trades"],
                        help="candles = 1-min OHLC price history; trades = individual fills (time & sales).")
    args = parser.parse_args()

    USE_S3 = not args.local
    if args.series:
        SERIES_TICKERS = [s.strip() for s in args.series.split(",")]

    if args.data_type == "trades":
        fetch_fn, save_fn, data_label, checkpoint_prefix = fetch_trades_for_market, save_trades, "trades", "trades_"
    else:
        fetch_fn, save_fn, data_label, checkpoint_prefix = fetch_candlesticks_for_market, save_candlesticks, "candles", ""

    dest = f"s3://{S3_BUCKET}/{S3_PREFIX}/" if USE_S3 else f"{DATA_DIR}/"
    client = make_client(env=args.env)

    if not args.live and not args.retry:
        logger.info(f"=== KALSHI HISTORY DRY RUN | data_type={args.data_type} | series={SERIES_TICKERS} ===")
        dry_run(client, fetch_fn, data_label)
    else:
        checkpoint = CheckpointManager(prefix=checkpoint_prefix)
        if args.retry:
            logger.info(f"=== RETRY MODE | data_type={args.data_type} | dest={dest} ===")
            for s in SERIES_TICKERS:
                run_series(client, s, checkpoint, fetch_fn, save_fn, data_label, is_retry=True)
        else:
            logger.info(f"=== FULL HISTORICAL PULL | data_type={args.data_type} | dest={dest} | series={SERIES_TICKERS} ===")
            for s in SERIES_TICKERS:
                run_series(client, s, checkpoint, fetch_fn, save_fn, data_label, is_retry=False)
        logger.info("=== DONE ===")
