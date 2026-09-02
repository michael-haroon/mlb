"""
probe_kalshi_g1g2.py
---------------------
Find doubleheader game tickers (ending in G1/G2) in the KXMLBGAMES series.
READ-ONLY. Sleeps between calls to avoid 429s.

Run on EC2:
    python3.11 scripts/probe_kalshi_g1g2.py
"""

from __future__ import annotations
import base64, json, os, re, time
from pathlib import Path
from dotenv import load_dotenv

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

API_KEY  = os.environ["KALSHI_READ_KEY"]
RSA_PATH = os.environ["KALSHI_READ_RSA_PATH"]
BASE     = "https://api.elections.kalshi.com/trade-api/v2"

def _load_key(path):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def _sign(key, ts_ms, method, path):
    msg = f"{ts_ms}{method}{path}".encode()
    return base64.b64encode(
        key.sign(msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256())
    ).decode()

def _headers(key, method, path):
    ts = int(time.time() * 1000)
    return {
        "KALSHI-ACCESS-KEY":       API_KEY,
        "KALSHI-ACCESS-SIGNATURE": _sign(key, ts, method, path),
        "KALSHI-ACCESS-TIMESTAMP": str(ts),
        "Content-Type":            "application/json",
    }

def get(key, path, params=None, sleep=0.5):
    time.sleep(sleep)
    r = requests.get(f"{BASE}{path}", headers=_headers(key, "GET", path),
                     params=params or {}, timeout=20)
    if not r.ok:
        print(f"  !! HTTP {r.status_code} → {r.text[:200]}")
        return {}
    return r.json() if r.content else {}

def paginate(key, path, list_key, params=None):
    params = dict(params or {})
    params.setdefault("limit", 200)
    results = []
    while True:
        data = get(key, path, params)
        if not data:
            break
        batch = data.get(list_key, [])
        results.extend(batch)
        cursor = data.get("cursor")
        if not cursor or not batch:
            break
        params["cursor"] = cursor
    return results

def main():
    key = _load_key(RSA_PATH)

    SERIES = "KXMLBGAMES"

    # ── 1. Confirm the series exists ──────────────────────────────────────────
    print(f"Checking series {SERIES} ...")
    r = get(key, f"/series/{SERIES}")
    if "series" in r:
        s = r["series"]
        print(f"  title={s.get('title')}  freq={s.get('frequency')}  category={s.get('category')}")
    else:
        print(f"  Series not found: {r}")

    # ── 2. Pull all events (settled + open) ───────────────────────────────────
    all_events = []
    for status in ("open", "settled", "closed"):
        print(f"\nFetching events  series={SERIES}  status={status} ...")
        batch = paginate(key, "/events", "events",
                         {"series_ticker": SERIES, "status": status})
        print(f"  → {len(batch)} events")
        all_events.extend(batch)

    # ── 3. Pull all markets (settled + open) ──────────────────────────────────
    all_markets = []
    for status in ("open", "settled", "closed"):
        print(f"\nFetching markets  series={SERIES}  status={status} ...")
        batch = paginate(key, "/markets", "markets",
                         {"series_ticker": SERIES, "status": status})
        print(f"  → {len(batch)} markets")
        all_markets.extend(batch)

    # ── 4. Find G1/G2 tickers ─────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("DOUBLEHEADER MARKETS (ticker ends with G1 or G2)")
    print("─" * 70)
    dh = [m for m in all_markets if re.search(r'[_-]G[12]$', m.get("ticker", ""), re.I)]
    if dh:
        for m in dh:
            print(f"\nTICKER: {m['ticker']}")
            print(json.dumps(m, indent=2))
    else:
        print("  None found.")
        print("\n  All market tickers (sorted):")
        for t in sorted(set(m["ticker"] for m in all_markets)):
            print(f"    {t}")

if __name__ == "__main__":
    main()
