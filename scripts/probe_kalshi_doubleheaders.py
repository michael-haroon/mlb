"""
probe_kalshi_doubleheaders.py
------------------------------
Comprehensive READ-ONLY probe of every Kalshi endpoint that might reveal
how doubleheader MLB games are named. Touches nothing — all GET requests.

Run on EC2:
    python3.11 scripts/probe_kalshi_doubleheaders.py

Requires:
    - KALSHI_READ_KEY in env (or .env file)
    - KALSHI_READ_RSA_PATH pointing to your PEM file

Strategy: Kalshi uses events-based hierarchy.
  Series  →  Events  →  Markets
  KXMLB       KXMLB-25-NYY-BOS        KXMLB-25-NYY-BOS-WIN

For doubleheaders the hypothesis is one of:
  (a) Two separate events:  KXMLB-25-NYY-BOS-G1  /  KXMLB-25-NYY-BOS-G2
  (b) One event, two markets suffixed G1/G2 within it
  (c) Entirely different naming convention

We probe: series info, events (all statuses, paginated), markets (all
statuses, paginated), and a direct ticker-pattern search via /markets
with various keyword searches.
"""

from __future__ import annotations
import base64, json, os, re, sys, time
from pathlib import Path
from dotenv import load_dotenv

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# ── Load .env from repo root ──────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

API_KEY  = os.environ["KALSHI_READ_KEY"]
RSA_PATH = os.environ["KALSHI_READ_RSA_PATH"]
BASE     = "https://api.elections.kalshi.com/trade-api/v2"

SEP = "─" * 80

# ── Auth helpers ──────────────────────────────────────────────────────────────

def _load_key(path: str):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def _sign(key, ts_ms: int, method: str, path: str) -> str:
    msg = f"{ts_ms}{method}{path}".encode()
    return base64.b64encode(
        key.sign(msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256())
    ).decode()

def _headers(key, method: str, path: str) -> dict:
    ts = int(time.time() * 1000)
    return {
        "KALSHI-ACCESS-KEY":       API_KEY,
        "KALSHI-ACCESS-SIGNATURE": _sign(key, ts, method, path),
        "KALSHI-ACCESS-TIMESTAMP": str(ts),
        "Content-Type":            "application/json",
    }

def get(key, path: str, params: dict | None = None) -> dict:
    """Single authenticated GET. Raises on non-2xx."""
    # path used for signing must NOT include query string
    r = requests.get(
        f"{BASE}{path}",
        headers=_headers(key, "GET", path),
        params=params or {},
        timeout=20,
    )
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f"  !! HTTP {r.status_code} on GET {path} → {r.text[:200]}")
        return {}
    return r.json() if r.content else {}

def paginate(key, path: str, list_key: str, params: dict | None = None,
             max_pages: int = 20) -> list:
    """Paginate through cursor-based results, return flat list."""
    params = dict(params or {})
    params.setdefault("limit", 200)
    results = []
    for page in range(max_pages):
        data = get(key, path, params)
        if not data:
            break
        batch = data.get(list_key, [])
        results.extend(batch)
        cursor = data.get("cursor")
        if not cursor or not batch:
            break
        params["cursor"] = cursor
        time.sleep(0.15)   # be polite
    return results

# ── DH detection helpers ──────────────────────────────────────────────────────

DH_RE = re.compile(r'(G1|G2|GM1|GM2|GAME[12]|DH|DOUBLEHEADER)', re.I)

def is_dh(ticker: str, title: str = "") -> bool:
    return bool(DH_RE.search(ticker) or DH_RE.search(title))

def print_event(e: dict, label: str = ""):
    ticker = e.get("event_ticker", e.get("ticker", "?"))
    title  = e.get("title", "")
    status = e.get("status", "")
    print(f"  {label}{ticker:<50}  [{status}]  {title[:45]}")

def print_market(m: dict, indent: int = 4):
    ticker = m.get("ticker", "?")
    title  = m.get("title", "")
    status = m.get("status", "")
    print(f"  {' '*indent}{ticker:<55}  [{status}]  {title[:35]}")

# ══════════════════════════════════════════════════════════════════════════════
def main():
    key = _load_key(RSA_PATH)

    # ── 1. Series metadata ────────────────────────────────────────────────────
    print(SEP)
    print("1. SERIES: KXMLB")
    print(SEP)
    series = get(key, "/series/KXMLB")
    print(json.dumps(series, indent=2)[:1500])

    # ── 2. Events — all statuses, paginated ──────────────────────────────────
    all_events: list[dict] = []
    for status in ("open", "closed", "settled", "active"):
        print(f"\n{SEP}")
        print(f"2. EVENTS  series=KXMLB  status={status}")
        print(SEP)
        batch = paginate(key, "/events",
                         list_key="events",
                         params={"series_ticker": "KXMLB", "status": status})
        print(f"  → {len(batch)} events")
        for e in batch:
            print_event(e)
        all_events.extend(batch)

    # ── 3. Events — no status filter (returns everything) ────────────────────
    print(f"\n{SEP}")
    print("3. EVENTS  series=KXMLB  (no status filter)")
    print(SEP)
    batch = paginate(key, "/events",
                     list_key="events",
                     params={"series_ticker": "KXMLB"})
    # merge deduplicated
    seen = {e["event_ticker"] for e in all_events}
    new = [e for e in batch if e["event_ticker"] not in seen]
    print(f"  → {len(batch)} total, {len(new)} not seen before")
    for e in new:
        print_event(e, "NEW  ")
    all_events.extend(new)

    # ── 4. Markets — all statuses, paginated ─────────────────────────────────
    all_markets: list[dict] = []
    for status in ("open", "closed", "settled"):
        print(f"\n{SEP}")
        print(f"4. MARKETS  series=KXMLB  status={status}")
        print(SEP)
        batch = paginate(key, "/markets",
                         list_key="markets",
                         params={"series_ticker": "KXMLB", "status": status})
        print(f"  → {len(batch)} markets")
        for m in batch[:40]:   # cap display
            print_market(m)
        if len(batch) > 40:
            print(f"  ... {len(batch)-40} more not shown")
        all_markets.extend(batch)

    # ── 5. Markets — no status filter ────────────────────────────────────────
    print(f"\n{SEP}")
    print("5. MARKETS  series=KXMLB  (no status filter)")
    print(SEP)
    batch = paginate(key, "/markets",
                     list_key="markets",
                     params={"series_ticker": "KXMLB"})
    seen_m = {m["ticker"] for m in all_markets}
    new_m = [m for m in batch if m["ticker"] not in seen_m]
    print(f"  → {len(batch)} total, {len(new_m)} not seen before")
    for m in new_m[:40]:
        print_market(m, indent=4)
    all_markets.extend(new_m)

    # ── 6. Doubleheader spotlight ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("6. DOUBLEHEADER SPOTLIGHT — any event/market matching G1/G2/DH")
    print(SEP)
    dh_events  = [e for e in all_events  if is_dh(e.get("event_ticker",""), e.get("title",""))]
    dh_markets = [m for m in all_markets if is_dh(m.get("ticker",""),       m.get("title",""))]
    print(f"  DH events:  {len(dh_events)}")
    print(f"  DH markets: {len(dh_markets)}")

    if dh_events:
        print("\n  ── DH Events ──")
        for e in dh_events:
            print(json.dumps(e, indent=4))

    if dh_markets:
        print("\n  ── DH Markets ──")
        for m in dh_markets:
            print(json.dumps(m, indent=4))

    # ── 7. Drill into one event to see full market list ───────────────────────
    # Pick an event that likely corresponds to a game (has team codes in ticker)
    candidate = next(
        (e for e in all_events if re.search(r'KXMLB-\d{2}-[A-Z]{2,3}-[A-Z]{2,3}', e.get("event_ticker",""))),
        all_events[0] if all_events else None
    )
    if candidate:
        eticker = candidate["event_ticker"]
        print(f"\n{SEP}")
        print(f"7. EVENT DETAIL (sample): {eticker}")
        print(SEP)
        detail = get(key, f"/events/{eticker}")
        print(json.dumps(detail, indent=2)[:3000])

    # ── 8. Try common doubleheader ticker patterns directly ──────────────────
    print(f"\n{SEP}")
    print("8. DIRECT TICKER PROBES — guessing plausible DH patterns")
    print(SEP)
    # Build some plausible guesses based on recent dates + common matchups
    # Format appears to be KXMLB-YY-AWAY-HOME-MARKETTYPE or KXMLB-25-NYY-BOS-G1
    guesses = [
        # Year prefix variants
        "KXMLB-25", "KXMLB-2025", "KXMLB-26", "KXMLB-2026",
    ]
    # Also try fetching events with a broader series search
    for prefix in ("KXMLB",):
        print(f"\n  Probing /series/{prefix} …")
        s = get(key, f"/series/{prefix}")
        if s:
            print(json.dumps(s, indent=2)[:800])

    # ── 9. Search via /markets with ticker keyword (if supported) ─────────────
    print(f"\n{SEP}")
    print("9. /markets with ticker_symbol (some Kalshi versions support this)")
    print(SEP)
    for kw in ("G1", "G2", "DH"):
        r = get(key, "/markets", params={"ticker": kw, "series_ticker": "KXMLB", "limit": 5})
        markets = r.get("markets", [])
        print(f"  keyword={kw!r}: {len(markets)} hits")

    # ── 10. Summary ──────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("SUMMARY")
    print(SEP)
    print(f"  Total KXMLB events seen:  {len(all_events)}")
    print(f"  Total KXMLB markets seen: {len(all_markets)}")
    print(f"  Doubleheader events:      {len(dh_events)}")
    print(f"  Doubleheader markets:     {len(dh_markets)}")

    if all_events:
        print("\n  ALL event tickers (sorted):")
        for t in sorted(e["event_ticker"] for e in all_events):
            print(f"    {t}")
    else:
        print("\n  !! No events found at all — check API key & RSA path")

    if all_markets:
        print("\n  First 20 market tickers (sorted):")
        for t in sorted(m["ticker"] for m in all_markets)[:20]:
            print(f"    {t}")


if __name__ == "__main__":
    main()
