"""
probe_lifecycle.py — Dump raw WS lifecycle payloads to see if settled/determined events carry result.

Run on EC2:
    python3.11 -m pregame.trading.probe_lifecycle

Connects to Kalshi WS, subscribes to market_lifecycle_v2, and prints every
lifecycle message verbatim (JSON-indented). Let it run until a game settles
(or look at already-determined markets via the lifecycle stream).

Press Ctrl+C to stop.
"""

import base64
import json
import os
import sys
import time
from pathlib import Path

import websocket
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from dotenv import load_dotenv
load_dotenv()

WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

def _load_private_key(path):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def _auth_headers(api_key, private_key):
    ts_ms = int(time.time() * 1000)
    msg = f"{ts_ms}GET/trade-api/ws/v2".encode("utf-8")
    sig = private_key.sign(
        msg,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": api_key,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("utf-8"),
        "KALSHI-ACCESS-TIMESTAMP": str(ts_ms),
    }


def on_open(ws):
    print("[CONNECTED] Subscribing to market_lifecycle_v2...")
    ws.send(json.dumps({
        "id": 1,
        "cmd": "subscribe",
        "params": {"channels": ["market_lifecycle_v2"]},
    }))


def on_message(ws, raw):
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[RAW] {raw[:200]}")
        return

    msg_type = msg.get("type", "")

    # Print ALL lifecycle messages in full
    if msg_type == "market_lifecycle_v2":
        data = msg.get("msg", {})
        event_type = data.get("event_type", "")
        ticker = data.get("market_ticker", "")

        # Only show MLB markets to reduce noise
        if ticker.startswith("KXMLB"):
            print(f"\n{'='*60}")
            print(f"EVENT: {event_type} | TICKER: {ticker}")
            print(f"FULL PAYLOAD:")
            print(json.dumps(msg, indent=2))
            print(f"{'='*60}")
    elif msg_type == "error":
        print(f"[ERROR] {json.dumps(msg, indent=2)}")
    # Silently ignore other message types (subscription confirmations, etc.)


def on_error(ws, error):
    print(f"[ERROR] {error}")


def on_close(ws, status_code, close_msg):
    print(f"[CLOSED] {status_code} {close_msg}")


def main():
    api_key = os.environ.get("KALSHI_READ_KEY", "")
    rsa_path = os.environ.get("KALSHI_READ_RSA_PATH", "")

    if not api_key or not rsa_path:
        print("ERROR: Set KALSHI_READ_KEY and KALSHI_READ_RSA_PATH in .env")
        sys.exit(1)

    private_key = _load_private_key(rsa_path)
    headers = _auth_headers(api_key, private_key)
    header_list = [f"{k}: {v}" for k, v in headers.items()]

    print(f"Connecting to {WS_URL}...")
    print("Waiting for MLB lifecycle events (settled/determined)...")
    print("Press Ctrl+C to stop.\n")

    ws = websocket.WebSocketApp(
        WS_URL,
        header=header_list,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.run_forever(ping_interval=20, ping_timeout=10)


if __name__ == "__main__":
    main()
