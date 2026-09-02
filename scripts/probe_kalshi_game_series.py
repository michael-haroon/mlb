"""
probe_kalshi_game_series.py
----------------------------
KXMLB is the World Series futures series — wrong for individual games.
This script finds the correct per-game MLB series and inspects doubleheader naming.

READ-ONLY — all GET requests, no writes.

Run on EC2:
    python3.11 scripts/probe_kalshi_game_series.py
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
SEP      = "─" * 80

# ── Auth ──────────────────────────────────────────────────────────────────────

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

def get(key, path, params=None):
    r = requests.get(f"{BASE}{path}", headers=_headers(key, "GET", path),
                     params=params or {}, timeout=20)
    if not r.ok:
        print(f"  !! HTTP {r.status_code} on GET {path} {params} → {r.text[:200]}")
        return {}
    return r.json() if r.content else {}

def paginate(key, path, list_key, params=None, max_pages=20):
    params = dict(params or {})
    params.setdefault("limit", 200)
    results = []
    for _ in range(max_pages):
        data = get(key, path, params)
        if not data:
            break
        batch = data.get(list_key, [])
        results.extend(batch)
        cursor = data.get("cursor")
        if not cursor or not batch:
            break
        params["cursor"] = cursor
        time.sleep(0.2)
    return results

DH_RE = re.compile(r'(G1|G2|GM1|GM2|GAME[12]|DH\b|DOUBLEHEADER)', re.I)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    key = _load_key(RSA_PATH)

    # ── 1. List ALL series (category=Sports) ─────────────────────────────────
    print(SEP)
    print("1. ALL SERIES  category=Sports")
    print(SEP)
    series_list = paginate(key, "/series", "series", {"category": "Sports"})
    print(f"  → {len(series_list)} series")
    for s in series_list:
        ticker   = s.get("ticker", "?")
        title    = s.get("title", "")
        freq     = s.get("frequency", "")
        category = s.get("category", "")
        print(f"  {ticker:<30}  {freq:<12}  {title[:50]}")

    # Look for anything baseball-flavored
    baseball_series = [
        s for s in series_list
        if any(kw in (s.get("ticker","") + s.get("title","")).upper()
               for kw in ("MLB","BASEBALL","KXMLB","KXNFL"))  # KXNFL just to gauge naming
    ]
    print(f"\n  Baseball-related series: {[s.get('ticker') for s in baseball_series]}")

    # ── 2. Also try listing series with no filter ─────────────────────────────
    print(f"\n{SEP}")
    print("2. ALL SERIES  (no filter) — first 50 tickers")
    print(SEP)
    all_series = paginate(key, "/series", "series", {}, max_pages=5)
    print(f"  → {len(all_series)} series")
    for s in all_series[:50]:
        print(f"  {s.get('ticker','?'):<30}  {s.get('title','')[:50]}")

    # ── 3. Probe plausible per-game series tickers directly ───────────────────
    print(f"\n{SEP}")
    print("3. DIRECT SERIES PROBES — guessing per-game MLB series tickers")
    print(SEP)
    candidates = [
        # Common Kalshi MLB game-level series patterns
        "KXMLBGAME", "KXMLBW",   "KXMLBWIN",
        "MLBGAME",   "MLBW",     "MLB",
        "KXMLB-WIN", "KXBASEBL",
        # Possibly split by market type
        "KXMLBYRFI", "KXMLBTOT", "KXMLBRL",
        # Maybe year-scoped
        "KXMLB25",   "KXMLB26",
        # Kalshi sometimes uses event-level tickers differently
        "KXMLBWINNER", "KXBASEBALLW",
    ]
    found_series = []
    for ticker in candidates:
        r = get(key, f"/series/{ticker}")
        if r and "series" in r:
            s = r["series"]
            print(f"  FOUND: {ticker} → title={s.get('title','?')}  freq={s.get('frequency','?')}")
            found_series.append(ticker)
        else:
            print(f"  miss:  {ticker}")
        time.sleep(0.1)

    # ── 4. Fetch the 5 'DH' markets from the previous probe ──────────────────
    # Section 9 of the last script returned 5 hits for ticker='DH', series_ticker='KXMLB'
    # Let's re-run that and print them fully
    print(f"\n{SEP}")
    print("4. MARKETS  ticker='DH'  series_ticker='KXMLB'  (the 5 mystery hits)")
    print(SEP)
    r = get(key, "/markets", {"ticker": "DH", "series_ticker": "KXMLB", "limit": 10})
    dh_markets = r.get("markets", [])
    print(f"  → {len(dh_markets)} markets")
    for m in dh_markets:
        print(json.dumps(m, indent=2))

    # ── 5. Markets without series filter — search for 'game' type ────────────
    print(f"\n{SEP}")
    print("5. MARKETS  no series filter  ticker='KXMLB'  (prefix search)")
    print(SEP)
    r = get(key, "/markets", {"ticker": "KXMLB", "limit": 200})
    markets = r.get("markets", [])
    print(f"  → {len(markets)} markets")
    for m in markets[:60]:
        t = m.get("ticker","")
        print(f"  {t:<60}  [{m.get('status','')}]  {m.get('title','')[:35]}")
    if len(markets) > 60:
        print(f"  ... {len(markets)-60} more")

    # ── 6. Search /events with no series filter for 'baseball' keyword ────────
    print(f"\n{SEP}")
    print("6. EVENTS  no series filter  (all statuses, looking for game-level events)")
    print(SEP)
    for status in ("open", "settled"):
        batch = paginate(key, "/events", "events", {"status": status, "limit": 200}, max_pages=3)
        # Filter to baseball-looking ones
        mlb = [e for e in batch
               if any(kw in (e.get("event_ticker","") + e.get("title","")).upper()
                      for kw in ("MLB","BASEBALL","KXMLB","KXBASE"))]
        print(f"  status={status}: {len(batch)} total events, {len(mlb)} baseball-related")
        for e in mlb:
            print(f"    {e.get('event_ticker',''):<50}  {e.get('title','')[:45]}")

    # ── 7. If we found any game-level series, pull their events/markets ───────
    for sticker in found_series:
        print(f"\n{SEP}")
        print(f"7. DEEP DIVE: {sticker}")
        print(SEP)
        for status in ("open", "settled", "closed"):
            batch = paginate(key, "/events", "events",
                             {"series_ticker": sticker, "status": status, "limit": 100})
            if batch:
                print(f"  Events [{status}]: {len(batch)}")
                for e in batch[:20]:
                    print(f"    {e.get('event_ticker',''):<55}  {e.get('title','')[:40]}")
                dh = [e for e in batch if DH_RE.search(e.get("event_ticker","")) or DH_RE.search(e.get("title",""))]
                if dh:
                    print(f"  !! DOUBLEHEADER EVENTS FOUND: {len(dh)}")
                    for e in dh:
                        print(json.dumps(e, indent=4))


if __name__ == "__main__":
    main()
