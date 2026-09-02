"""
pregame/trading/kalshi_client.py
--------------------------------
Kalshi REST API client with RSA-PSS authentication.

Self-contained: no dependency on the NBA project's backtest module.
Supports both production and demo environments.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from .config import KALSHI_REST_BASE, KALSHI_DEMO_REST_BASE

logger = logging.getLogger(__name__)


def _load_private_key(path: str | Path):
    """Load RSA private key from PEM file."""
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def _sign_request(private_key, timestamp_ms: int, method: str, path: str) -> str:
    """Sign a Kalshi API request with RSA-PSS-SHA256.

    Kalshi's auth scheme: sign "{timestamp_ms}{METHOD}{path}" where path
    includes the leading slash but NOT the host.
    """
    message = f"{timestamp_ms}{method}{path}".encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


class KalshiClient:
    """REST client for the Kalshi Trade API v2."""

    def __init__(self, api_key: str, private_key, env: str = "prod"):
        self._api_key = api_key
        self._private_key = private_key
        self._base = KALSHI_REST_BASE if env == "prod" else KALSHI_DEMO_REST_BASE
        self._session = requests.Session()

    def _headers(self, method: str, path: str) -> dict:
        ts_ms = int(time.time() * 1000)
        sig = _sign_request(self._private_key, ts_ms, method, path)
        return {
            "KALSHI-ACCESS-KEY": self._api_key,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": str(ts_ms),
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self._base}{path}"
        headers = self._headers(method, path)
        resp = self._session.request(method, url, headers=headers, **kwargs)
        if resp.status_code == 429:
            raise RateLimitError(f"429 Too Many Requests on {path}")
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    # ── Market discovery ─────────────────────────────────────────────────────

    def get_markets(
        self,
        series_ticker: Optional[str] = None,
        status: str = "open",
        limit: int = 200,
        cursor: Optional[str] = None,
    ) -> dict:
        """List markets, optionally filtered by series and status."""
        params = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/markets", params=params)

    def get_market(self, ticker: str) -> dict:
        """Get a single market by ticker."""
        return self._request("GET", f"/markets/{ticker}")

    def get_trades(
        self,
        ticker: Optional[str] = None,
        limit: int = 200,
        cursor: Optional[str] = None,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
    ) -> dict:
        """Time & sales — individual fills. Filtering by ticker alone bounds results
        to that market's lifetime; no time-range cap like /candlesticks has."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts
        return self._request("GET", "/markets/trades", params=params)

    def get_event(self, event_ticker: str) -> dict:
        """Get event details (contains mutually exclusive market group info)."""
        return self._request("GET", f"/events/{event_ticker}")

    # ── Historical archive — markets that have aged out of the live /markets
    # index (confirmed cutoff via GET /historical/cutoff). event_ticker is the
    # reliable way in: /markets and /events?with_nested_markets both come back
    # empty for pre-cutoff events even though the markets themselves still
    # resolve individually. ──────────────────────────────────────────────────
    def get_historical_markets(
        self,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        limit: int = 200,
        cursor: Optional[str] = None,
    ) -> dict:
        params = {"limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/historical/markets", params=params)

    def get_historical_market(self, ticker: str) -> dict:
        return self._request("GET", f"/historical/markets/{ticker}")

    def get_historical_candlesticks(self, ticker: str, start_ts: int, end_ts: int, period_interval: int = 1) -> dict:
        """No series_ticker in the path — unlike the live /series/{s}/markets/{t}/candlesticks."""
        return self._request(
            "GET", f"/historical/markets/{ticker}/candlesticks",
            params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval},
        )

    def get_historical_trades(
        self,
        ticker: Optional[str] = None,
        limit: int = 200,
        cursor: Optional[str] = None,
    ) -> dict:
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/historical/trades", params=params)

    def get_events(
        self,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 200,
        cursor: Optional[str] = None,
    ) -> dict:
        """List events, optionally filtered by series and status. Paginated via cursor."""
        params = {"limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/events", params=params)

    def get_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
    ) -> dict:
        """OHLC price candlesticks for a market's lifetime.

        period_interval is in minutes — Kalshi accepts 1, 60, or 1440. Default
        1-minute so price ticks are fine-grained enough to eventually align
        against per-pitch DL predictions.
        """
        return self._request(
            "GET",
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            params={
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
            },
        )

    # ── Orderbook ────────────────────────────────────────────────────────────

    def get_orderbook(self, ticker: str, depth: int = 5) -> dict:
        """Get orderbook snapshot for a market."""
        return self._request("GET", f"/orderbook/{ticker}", params={"depth": depth})

    # ── Orders ───────────────────────────────────────────────────────────────

    def create_order(
        self,
        ticker: str,
        side: str,
        action: str,
        count: int,
        price: int,
        order_type: str = "limit",
        client_order_id: Optional[str] = None,
    ) -> dict:
        """Place an order.

        Args:
            side: "yes" or "no"
            action: "buy" or "sell"
            count: number of contracts
            price: price in cents (1-99)
            order_type: "limit" or "market"
        """
        body = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": order_type,
        }
        if order_type == "limit":
            body["yes_price"] = price if side == "yes" else None
            body["no_price"] = price if side == "no" else None
        if client_order_id:
            body["client_order_id"] = client_order_id

        return self._request("POST", "/portfolio/orders", json=body)

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a resting order."""
        return self._request("DELETE", f"/portfolio/orders/{order_id}")

    def get_orders(self, status: Optional[str] = None, ticker: Optional[str] = None) -> dict:
        """List orders, optionally filtered."""
        params = {}
        if status:
            params["status"] = status
        if ticker:
            params["ticker"] = ticker
        return self._request("GET", "/portfolio/orders", params=params)

    # ── Portfolio ────────────────────────────────────────────────────────────

    def get_positions(self, **kwargs) -> dict:
        """Get current portfolio positions."""
        return self._request("GET", "/portfolio/positions", params=kwargs)

    def get_balance(self) -> dict:
        """Get account balance."""
        return self._request("GET", "/portfolio/balance")

    def get_fills(self, **kwargs) -> dict:
        """Get recent fills."""
        return self._request("GET", "/portfolio/fills", params=kwargs)

    # ── Account ──────────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        """Get account info (includes netting/collateral settings)."""
        return self._request("GET", "/account")

    def upgrade_rate_limit(self) -> dict:
        """Upgrade API usage level from Basic (200r/100w) to Advanced (300r/300w).

        Idempotent: calling when already at Advanced or higher is a no-op on
        Kalshi's side. Higher tiers (expert+) require trading volume thresholds.
        """
        return self._request("POST", "/account/api_usage_level/upgrade")


class RateLimitError(Exception):
    """Raised on 429 — caller should retry with backoff."""
    pass


def make_client(env: str = "prod", key_path: Optional[str | Path] = None) -> KalshiClient:
    """Factory: build a read-only client from environment variables.

    Expects in .env:
      KALSHI_READ_KEY — read-only API key ID
      KALSHI_READ_RSA_PATH — path to RSA private key PEM (or pass key_path)
      KALSHI_DEMO_KEY — demo API key (used when env="demo")
    """
    if env == "demo":
        api_key = os.environ.get("KALSHI_DEMO_KEY", "")
        if not api_key:
            raise EnvironmentError("KALSHI_DEMO_KEY not set")
    else:
        api_key = os.environ.get("KALSHI_READ_KEY", "")
        if not api_key:
            raise EnvironmentError("KALSHI_READ_KEY not set")

    if key_path is None:
        key_path = os.environ.get(
            "KALSHI_READ_RSA_PATH",
            str(Path(__file__).resolve().parent / "keys" / "read.pem"),
        )
    private_key = _load_private_key(key_path)
    return KalshiClient(api_key, private_key, env=env)


def make_write_client(env: str = "prod", key_path: Optional[str | Path] = None) -> KalshiClient:
    """Factory: build a write-capable client for live trading.

    Expects in .env:
      KALSHI_WRITE_KEY — write-capable API key ID
      KALSHI_WRITE_RSA_PATH — path to trade RSA private key PEM
    """
    api_key = os.environ.get("KALSHI_WRITE_KEY", "")
    if not api_key:
        raise EnvironmentError("KALSHI_WRITE_KEY not set")

    if key_path is None:
        key_path = os.environ.get(
            "KALSHI_WRITE_RSA_PATH",
            str(Path(__file__).resolve().parent / "keys" / "trade.pem"),
        )
    private_key = _load_private_key(key_path)
    return KalshiClient(api_key, private_key, env=env)
