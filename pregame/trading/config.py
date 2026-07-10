"""
pregame/trading/config.py
-------------------------
All tunable parameters for the MLB pregame market-making system.

Values are calibrated for MLB's market structure:
- ~15 games/day → more diversification, smaller per-bet sizing
- Thinner books than NBA → wider natural spreads, more repricing caution
- Zero maker fees → spread capture is pure profit minus adverse selection
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # mlb/
TRADING_DIR = Path(__file__).resolve().parent
LOGS_DIR = TRADING_DIR / "logs"
STATE_DIR = TRADING_DIR / "state"
ARTIFACTS_DIR = PROJECT_ROOT / "pregame" / "artifacts"

# ── Execution mode ───────────────────────────────────────────────────────────
DRY_RUN = True

# ── Position sizing ──────────────────────────────────────────────────────────
KELLY_FRACTION = 0.15              # Lower than NBA (0.25): more daily opportunities
MAX_POSITION_PCT = 3.0             # Lower than NBA: thinner MLB books
MAX_DAILY_EXPOSURE_PCT = 50.0      # Higher than NBA: more diversification (15 games/day)
MAX_CONCURRENT_POSITIONS = 50
MAX_CONTRACTS_PER_MARKET = 12      # Thinner books than NBA

# ── Edge thresholds ──────────────────────────────────────────────────────────
# Taker fee = 0.07 * P * (1-P); maker fee = $0. Edge must exceed fee * buffer.
MIN_EDGE_BUFFER_TAKER = 2.0        # Higher bar for paying taker fee in thin books
MIN_EDGE_BUFFER_MAKER = 1.5        # Maker is free but still need real signal
TAKER_EDGE_THRESHOLD = 2.0         # edge >= taker_breakeven * 2 → aggress

# ── Quoting spread ───────────────────────────────────────────────────────────
# Half-spread in cents by confidence tier. Wider = less fill rate but less adverse selection.
HALF_SPREAD_CENTS = {
    "HIGH": 2,      # Tight: we're confident in fair value
    "MEDIUM": 3,
    "LOW": 4,       # Wide: uncertainty reflected in spread
}

# ── Confidence shading ───────────────────────────────────────────────────────
# Sigma units to shade fair value toward less favorable direction.
# Higher = more conservative = fewer fills but higher EV per fill.
SHADE_SIGMA = {
    "HIGH": 0.5,
    "MEDIUM": 1.0,
    "LOW": 1.5,
}

# ── Timing ───────────────────────────────────────────────────────────────────
MIN_HOURS_TO_FIRST_PITCH = 0.5     # MLB lineups confirmed ~1h before; trade from 30min out
CANCEL_BEFORE_FIRST_PITCH_MIN = 10 # Cancel all resting orders this many min before first pitch
EXIT_BUFFER_MINUTES = 15           # Stop posting new quotes this many min before first pitch

# ── Risk limits ──────────────────────────────────────────────────────────────
DAILY_LOSS_LIMIT_PCT = 4.0         # Circuit breaker: halt all trading

# ── Price filters ────────────────────────────────────────────────────────────
PRICE_FLOOR = 0.12                 # YRFI can trade in tails
PRICE_CEILING = 0.88               # Avoid extreme favorites

# ── Repricing (top-of-book fighting) ────────────────────────────────────────
REPRICE_MIN_TICK_MOVE = 2          # Cents; only reprice if target differs by >= this
MAX_REPRICES_PER_ORDER = 5         # Prevent infinite repricing wars
MIN_REPRICE_INTERVAL_SEC = 30      # Rate limit: don't churn faster than this

# ── Scanning ─────────────────────────────────────────────────────────────────
SCAN_INTERVAL_SEC = 60             # Main loop cadence

# ── Cluster caps (correlated position limits) ────────────────────────────────
# Prevents concentration in correlated markets within a single game.
CLUSTER_MAX_CONTRACTS = {
    "winner": 10,        # MLBGAME moneyline
    "spread": 8,         # MLBSPREAD (correlated with winner)
    "total": 10,         # MLBTOTAL (weakly correlated with spread)
    "team_total": 8,     # MLBTEAMTOTAL (correlated with total AND winner)
    "first_inning": 6,   # MLBRFI (low correlation with full-game)
    "extra_innings": 4,  # MLBEXTRAINNINGS (tail event)
}

# ── Prediction health ───────────────────────────────────────────────────────
# OOF prediction std from training (empirical). If live predictions collapse
# below MIN_SHARPNESS_RATIO * expected, feature distributions have shifted and
# the model is outputting uninformative probs clustered near 0.5.
EXPECTED_PRED_STD = {
    "home_win": 0.14,
    "yrfi": 0.02,
    "extra_innings": 0.04,
    "first_5_home_win": 0.06,
}
MIN_SHARPNESS_RATIO = 0.40  # halt target if live_std < 40% of expected

# ── Feature refresh ─────────────────────────────────────────────────────────
S3_DATA_URI = "s3://mlb-265753586044-us-east-1-an/data"
FEATURES_MAX_AGE_HOURS = 6         # Rebuild if parquet older than this
FEATURES_PERIODIC_REFRESH_HOURS = 2

# ── Kalshi connection ────────────────────────────────────────────────────────
KALSHI_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
KALSHI_REST_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_DEMO_WS_URL = "wss://demo-api.kalshi.co/trade-api/ws/v2"
KALSHI_DEMO_REST_BASE = "https://demo-api.kalshi.co/trade-api/v2"

# ── MLB market series we trade ───────────────────────────────────────────────
TRADEABLE_SERIES = [
    "KXMLBGAME",    # Game winner
    "KXMLBRFI",     # Run first inning (YRFI/NRFI)
    "KXMLBTOTAL",   # Game total runs (over/under)
    "KXMLBSPREAD",  # Run line / spread
    "KXMLBTEAMTOTAL",  # Team total runs
    "KXMLBEXTRAS",  # Extra innings
]
