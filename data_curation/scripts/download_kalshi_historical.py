"""
download_kalshi_historical.py
------------------------------
Pre-cutoff Kalshi MLB archive (2025 season and any other pre-cutoff dates).

The live /markets and /events?with_nested_markets endpoints only surface a
recent rolling window (see GET /historical/cutoff) — a market that settled
before the cutoff comes back with zero attached markets on those, even though
it still resolves individually via /historical/markets/{ticker}. The way in is
event-driven: /events (no status filter) lists event shells for ALL dates,
then /historical/markets?event_ticker=X returns that event's real market(s).

Coverage differs sharply by series — checked empirically before running this:
KXMLBGAME has 2,203 real 2025 events (~the whole season); SPREAD/TOTAL have
~40 each; RFI has 10; TEAMTOTAL/EXTRAS have 0 (didn't exist yet in 2025).

Mirrors download_kalshi_history.py's storage/checkpoint/rate-limiter pattern
(deliberately not imported from it — importing would re-run that module's
logging setup against the same log file as the concurrently-running live-window
jobs). Own log file, own checkpoint namespace, own S3 subpath.

Run:
    conda run -n pred python data_curation/scripts/download_kalshi_historical.py            # dry run
    conda run -n pred python data_curation/scripts/download_kalshi_historical.py --live      # full pull
    conda run -n pred python data_curation/scripts/download_kalshi_historical.py --live --retry
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

SERIES_TICKERS = sorted({s for s in MODEL_TO_SERIES.values() if s})
YEAR_PREFIX = "25"  # 2025 season; the pre-cutoff year we've confirmed has real markets
MAX_WORKERS = 8  # lighter than the live-window pull — this runs concurrently alongside it
RATE_LIMIT_DELAY = 0.2  # starting point; same adaptive scheme, shares the account budget with the other jobs
CANDLE_PERIOD_INTERVAL_MIN = 1
MAX_CANDLES_PER_REQUEST = 4900
MARKET_FLUSH_THRESHOLD = 200
DATA_FLUSH_THRESHOLD = 5000
SUBMIT_BATCH_SIZE = 100

S3_BUCKET = "mlb-265753586044-us-east-1-an"
S3_PREFIX = "kalshi_history"
S3_REGION = "us-east-1"
_s3_client = None

DATA_DIR = "data"
LOG_DIR = os.path.join(DATA_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("KALSHI_HISTORICAL")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(os.path.join(LOG_DIR, "kalshi_historical.log"))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s"))
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("[KALSHI HISTORICAL] %(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"))
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ---------------------------------------------------------------------------
# STORAGE — same shape as download_kalshi_history.py, nested under historical/
# so it can never collide with the live-window pull's batch files.
# ---------------------------------------------------------------------------
def _get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=S3_REGION)
    return _s3_client


def _s3_key(rel_path: str) -> str:
    return f"{S3_PREFIX}/{rel_path}"


def _read_json_store(rel_path: str):
    try:
        obj = _get_s3().get_object(Bucket=S3_BUCKET, Key=_s3_key(rel_path))
        return json.loads(obj["Body"].read().decode("utf-8"))
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def _write_json_store(rel_path: str, data: Any):
    _get_s3().put_object(
        Bucket=S3_BUCKET, Key=_s3_key(rel_path),
        Body=json.dumps(data, indent=2).encode("utf-8"), ContentType="application/json",
    )


def _save(df: pd.DataFrame, rel_path: str):
    if df.empty:
        return
    key = _s3_key(rel_path)
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
    buf.seek(0)
    _get_s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    logger.debug(f"[save] {len(df)} rows -> s3://{S3_BUCKET}/{key}")


def save_markets(records: List[Dict[str, Any]], series_ticker: str):
    if records:
        _save(pd.DataFrame(records), f"{series_ticker}/historical/markets_batch_{int(time.time()*1000)}.parquet")


def save_candlesticks(records: List[Dict[str, Any]], series_ticker: str):
    if records:
        _save(pd.json_normalize(records, sep="_"), f"{series_ticker}/historical/candlesticks_batch_{int(time.time()*1000)}.parquet")


def save_trades(records: List[Dict[str, Any]], series_ticker: str):
    if records:
        _save(pd.DataFrame(records), f"{series_ticker}/historical/trades_batch_{int(time.time()*1000)}.parquet")


# ---------------------------------------------------------------------------
# CHECKPOINT — own namespace ("historical_*"), keyed by ticker.
# ---------------------------------------------------------------------------
class CheckpointManager:
    def __init__(self):
        self.checkpoint_rel = "historical_checkpoint.json"
        self.retry_rel = "historical_retry_queue.json"
        self._lock = threading.Lock()
        self.completed: Set[str] = set()
        self.retry_queue: List[Dict[str, Any]] = []
        data = _read_json_store(self.checkpoint_rel)
        if data:
            self.completed = set(data.get("completed", []))
            logger.info(f"[checkpoint] Loaded {len(self.completed)} completed tickers.")
        retry_data = _read_json_store(self.retry_rel)
        if retry_data:
            self.retry_queue = retry_data
            logger.info(f"[checkpoint] Loaded {len(self.retry_queue)} tickers in retry queue.")

    def is_completed(self, ticker: str) -> bool:
        return ticker in self.completed

    def mark_completed(self, ticker: str):
        with self._lock:
            self.completed.add(ticker)
            _write_json_store(self.checkpoint_rel, {"completed": sorted(self.completed),
                                                      "last_updated": datetime.now(timezone.utc).isoformat()})

    def mark_failed(self, ticker: str, series_ticker: str, reason: str, error: str):
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            existing = {e["ticker"]: e for e in self.retry_queue}
            if ticker in existing:
                e = existing[ticker]; e["attempts"] += 1; e["last_error"] = error; e["last_failed"] = now
            else:
                self.retry_queue.append({"ticker": ticker, "series_ticker": series_ticker, "reason": reason,
                                          "attempts": 1, "last_error": error, "first_failed": now, "last_failed": now})
            _write_json_store(self.retry_rel, self.retry_queue)

    def discard_retry_entry(self, ticker: str):
        with self._lock:
            self.retry_queue = [e for e in self.retry_queue if e["ticker"] != ticker]
            _write_json_store(self.retry_rel, self.retry_queue)


# ---------------------------------------------------------------------------
# RATE LIMITER — independent instance from the other two processes; fair
# sharing of the account budget happens via each process backing off on the
# 429s it individually observes (same mechanism as download_kalshi_history.py).
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, initial_delay: float, min_delay: float = 0.03, max_delay: float = 1.0):
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

    def on_success(self):
        with self._lock:
            self._success_streak += 1
            if self._success_streak >= 50:
                self._success_streak = 0
                self.delay = max(self.min_delay, self.delay * 0.9)


_rate_limiter = RateLimiter(RATE_LIMIT_DELAY)


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
                logger.error(f"[{label}] failed after {max_retries} attempts: {e}")
                raise
            sleep_time = backoff_factor ** attempt
            logger.warning(f"[{label}] error: {e}. retry in {sleep_time}s")
            time.sleep(sleep_time)
    raise RuntimeError(f"[{label}] exhausted retries")


def _ts(iso_str: Optional[str]) -> Optional[int]:
    if not iso_str:
        return None
    return int(datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp())


# ---------------------------------------------------------------------------
# DISCOVERY — event-driven, not the live /markets listing.
# ---------------------------------------------------------------------------
def discover_year_events(client, series_ticker: str, year_prefix: str) -> List[str]:
    events: List[Dict[str, Any]] = []
    cursor = None
    page = 0
    while True:
        resp = _call_with_backoff(
            f"events:{series_ticker}", client.get_events,
            series_ticker=series_ticker, status=None, limit=200, cursor=cursor,
        )
        batch = resp.get("events", [])
        events.extend(batch)
        cursor = resp.get("cursor")
        page += 1
        logger.debug(f"[events:{series_ticker}] page={page} total={len(events)}")
        if not cursor or not batch:
            break
    prefix = f"{series_ticker}-{year_prefix}"
    return sorted(e["event_ticker"] for e in events if e.get("event_ticker", "").startswith(prefix))


def discover_historical_markets(client, series_ticker: str, event_tickers: List[str]) -> List[Dict[str, Any]]:
    markets: List[Dict[str, Any]] = []
    for et in event_tickers:
        resp = _call_with_backoff(f"hist-markets:{et}", client.get_historical_markets, event_ticker=et)
        markets.extend(resp.get("markets", []))
    return markets


# ---------------------------------------------------------------------------
# FETCH — candles (chunked, same 5000-period cap as the live endpoint) + trades.
# ---------------------------------------------------------------------------
def fetch_candlesticks(client, ticker: str, market: Dict[str, Any]) -> List[Dict[str, Any]]:
    start_ts = _ts(market.get("open_time"))
    end_ts = _ts(market.get("close_time")) or _ts(market.get("expiration_time"))
    if start_ts is None or end_ts is None:
        return []
    step = MAX_CANDLES_PER_REQUEST * 60 * CANDLE_PERIOD_INTERVAL_MIN
    windows, cur = [], start_ts
    while cur < end_ts:
        nxt = min(cur + step, end_ts)
        windows.append((cur, nxt))
        cur = nxt + 60 * CANDLE_PERIOD_INTERVAL_MIN
    out = []
    for s, e in windows:
        resp = _call_with_backoff(f"candles:{ticker}", client.get_historical_candlesticks, ticker, s, e, CANDLE_PERIOD_INTERVAL_MIN)
        out.extend(resp.get("candlesticks", []))
    for c in out:
        c["market_ticker"] = ticker
    return out


def fetch_trades(client, ticker: str) -> List[Dict[str, Any]]:
    out, cursor = [], None
    while True:
        resp = _call_with_backoff(f"trades:{ticker}", client.get_historical_trades, ticker=ticker, limit=200, cursor=cursor)
        batch = resp.get("trades", [])
        out.extend(batch)
        cursor = resp.get("cursor")
        if not cursor or not batch:
            break
    return out


def fetch_all_for_market(client, series_ticker: str, market: Dict[str, Any]):
    ticker = market["ticker"]
    candles = fetch_candlesticks(client, ticker, market)
    trades = fetch_trades(client, ticker)
    return candles, trades


# ---------------------------------------------------------------------------
# MAIN CRAWL
# ---------------------------------------------------------------------------
def run_series(client, series_ticker: str, checkpoint: CheckpointManager, is_retry: bool = False):
    if is_retry:
        targets = [e for e in checkpoint.retry_queue if e["series_ticker"] == series_ticker]
        markets = []
        for t in targets:
            try:
                resp = _call_with_backoff(f"retry-refetch:{t['ticker']}", client.get_historical_market, t["ticker"])
            except Exception as e:
                logger.warning(f"[retry-refetch:{series_ticker}] {t['ticker']} unresolvable, dropping: {e}")
                checkpoint.discard_retry_entry(t["ticker"])
                continue
            m = resp.get("market")
            if m:
                markets.append(m)
    else:
        year_events = discover_year_events(client, series_ticker, YEAR_PREFIX)
        logger.info(f"[{series_ticker}] {len(year_events)} {YEAR_PREFIX}-prefixed events found")
        if not year_events:
            return
        markets = discover_historical_markets(client, series_ticker, year_events)
        logger.info(f"[{series_ticker}] {len(markets)} historical markets across those events")
        markets = [m for m in markets if not checkpoint.is_completed(m["ticker"])]
        logger.info(f"[{series_ticker}] {len(markets)} pending after checkpoint filter")

    if not markets:
        return

    market_buf, candle_buf, trade_buf, pending = [], [], [], []

    def _flush(label: str):
        if not pending:
            return
        try:
            save_markets(market_buf, series_ticker)
            save_candlesticks(candle_buf, series_ticker)
            save_trades(trade_buf, series_ticker)
            for ticker in pending:
                checkpoint.mark_completed(ticker)
            logger.debug(f"[flush:{series_ticker}:{label}] checkpointed {len(pending)}")
        except Exception as e:
            logger.error(f"[flush:{series_ticker}:{label}] save failed: {e}")
            for ticker in pending:
                checkpoint.mark_failed(ticker, series_ticker, type(e).__name__, str(e))
            raise
        finally:
            market_buf.clear(); candle_buf.clear(); trade_buf.clear(); pending.clear()

    pbar = tqdm(total=len(markets), desc=f"{series_ticker} historical")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix=f"KHist-{series_ticker}") as executor:
        market_iter = iter(markets)
        in_flight: Dict = {}

        def _fill_queue():
            while len(in_flight) < SUBMIT_BATCH_SIZE:
                m = next(market_iter, None)
                if m is None:
                    break
                f = executor.submit(fetch_all_for_market, client, series_ticker, m)
                in_flight[f] = m

        _fill_queue()
        while in_flight:
            future = next(as_completed(in_flight))
            m = in_flight.pop(future)
            ticker = m["ticker"]
            try:
                candles, trades = future.result()
                market_buf.append(m)
                candle_buf.extend(candles)
                trade_buf.extend(trades)
                pending.append(ticker)
                pbar.update(1)
                pbar.set_postfix({"candles": len(candle_buf), "trades": len(trade_buf), "pending": len(pending)})
                if len(candle_buf) + len(trade_buf) >= DATA_FLUSH_THRESHOLD or len(market_buf) >= MARKET_FLUSH_THRESHOLD:
                    _flush("threshold")
            except Exception as e:
                logger.error(f"[{series_ticker}] worker failed for {ticker}: {e}")
                checkpoint.mark_failed(ticker, series_ticker, type(e).__name__, str(e))
                pbar.update(1)
            _fill_queue()
    pbar.close()
    _flush("final")


def dry_run(client):
    logger.info(f"=== DRY RUN: {YEAR_PREFIX}-season discovery only, no writes ===")
    for series_ticker in SERIES_TICKERS:
        year_events = discover_year_events(client, series_ticker, YEAR_PREFIX)
        logger.info(f"[{series_ticker}] {len(year_events)} {YEAR_PREFIX}-prefixed events")
        if not year_events:
            continue
        sample_markets = discover_historical_markets(client, series_ticker, year_events[:1])
        logger.info(f"[{series_ticker}] sample event -> {len(sample_markets)} markets")
        if sample_markets:
            m = sample_markets[0]
            logger.info(f"[{series_ticker}] sample market: {json.dumps(m, indent=2, default=str)[:800]}")
            candles, trades = fetch_all_for_market(client, series_ticker, m)
            logger.info(f"[{series_ticker}] sample candles={len(candles)} trades={len(trades)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-cutoff (2025-season) Kalshi MLB archive.")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--series", type=str, default=None)
    parser.add_argument("--env", type=str, default="prod", choices=["prod", "demo"])
    args = parser.parse_args()

    if args.series:
        SERIES_TICKERS = [s.strip() for s in args.series.split(",")]

    client = make_client(env=args.env)

    if not args.live and not args.retry:
        logger.info(f"=== KALSHI HISTORICAL DRY RUN | series={SERIES_TICKERS} ===")
        dry_run(client)
    else:
        checkpoint = CheckpointManager()
        logger.info(f"=== {'RETRY' if args.retry else 'FULL'} HISTORICAL PULL | series={SERIES_TICKERS} ===")
        for s in SERIES_TICKERS:
            run_series(client, s, checkpoint, is_retry=args.retry)
        logger.info("=== DONE ===")
