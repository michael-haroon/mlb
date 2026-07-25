# Trading Runner Integration Guide

This document shows how to integrate the `LiveInferenceEngine` with the existing `TradingRunner` in `pregame/trading/runner.py`.

## Architecture Overview

```
┌──────────────────┐
│  TradingRunner   │  (sync, main thread)
│  (runner.py)     │
└────────┬─────────┘
         │
         │ get_live_prices(game_pk)
         │ register_game(game_pk, prior)
         ▼
┌──────────────────┐
│  TradingBridge   │  (thread-safe bridge)
│ (inference_      │
│  engine.py)      │
└────────┬─────────┘
         │
         │ async event loop (background thread)
         ▼
┌──────────────────┐
│ LiveInference    │  (async, background thread)
│ Engine           │
└────────┬─────────┘
         │
         │ on_pitch_event(pitch)
         ▼
┌──────────────────┐
│  MLB GUMBO API   │  (external real-time feed)
│  or equivalent   │
└──────────────────┘
```

## Phase 1: Add TradingBridge to TradingRunner

### Step 1: Initialize the bridge in `__init__`

```python
# In TradingRunner.__init__ (around line 79-93)
class TradingRunner:
    def __init__(self, dry_run: bool = True, env: str = "prod", bankroll: float = 1000.0):
        self._dry_run = dry_run
        self._env = env
        self._bankroll = bankroll
        self._running = False

        # Core components (initialized in start())
        self._client = None
        self._ws: KalshiWS | None = None
        self._portfolio: Portfolio | None = None
        self._features: FeatureManager | None = None
        self._ensemble_store: EnsembleStore | None = None

        # NEW: Live inference bridge (initialized in start())
        self._live_bridge: TradingBridge | None = None

        # Reprice tracking: {order_id: {"count": int, "last_reprice": float}}
        self._reprice_state: dict[str, dict] = {}
```

### Step 2: Start the bridge in `start()`

```python
# In TradingRunner.start() (after line 100)
def start(self) -> None:
    """Initialize all components and start the trading loop."""
    logger.info(f"Starting MLB pregame trader (mode={'DRY' if self._dry_run else 'LIVE'}, "
                f"bankroll=${self._bankroll:.0f}, env={self._env})")

    # 1. Connect to Kalshi
    # ... (existing code) ...

    # 2. Load models
    # ... (existing code) ...

    # 3. Load feature manager
    # ... (existing code) ...

    # 4. Initialize portfolio
    # ... (existing code) ...

    # 5. Start WebSocket
    # ... (existing code) ...

    # 6. NEW: Start live inference bridge
    model_path = Path(__file__).parent.parent.parent / "live" / "mlb_dl" / "checkpoints" / "best_model.pt"
    if model_path.exists():
        logger.info("Initializing live inference bridge")
        from live.mlb_dl.inference_engine import TradingBridge
        self._live_bridge = TradingBridge(
            model_path=str(model_path),
            device="cpu",  # CPU sufficient for single-batch inference
        )
        self._live_bridge.start()
    else:
        logger.warning(f"Live model not found at {model_path}, live repricing disabled")

    # ... (rest of existing start() code) ...
```

### Step 3: Hook into game start event

```python
# In TradingRunner._handle_game_start() (around line 432-450)
def _handle_game_start(self, ticker: str) -> None:
    """Called via WS when a game transitions to active (first pitch)."""
    parsed = parse_ticker(ticker)
    if not parsed:
        return

    game_key = parsed.game_key
    logger.info(f"Game started: {game_key}")

    # Cancel all unfilled orders for this game
    for order in self._portfolio.get_open_orders():
        if game_key in order.get("ticker", ""):
            cancel_order(self._client, order.get("order_id", ""), self._dry_run)
            self._portfolio.remove_order(order.get("order_id", ""))

    # Classify positions as HOLD or EXIT
    transitions = self._portfolio.on_game_start(game_key)
    for t, state in transitions.items():
        logger.info(f"  {t} → {state.value}")

    # NEW: Initialize live inference for this game
    if self._live_bridge:
        game_pk = self._resolve_game_pk(ticker)
        if game_pk:
            pregame_prior = self._build_pregame_prior(game_pk)
            self._live_bridge.register_game(game_pk, pregame_prior)
            logger.info(f"Live inference activated for game {game_pk}")
```

### Step 4: Add helper methods

```python
# Add these methods to TradingRunner class

def _resolve_game_pk(self, ticker: str) -> Optional[int]:
    """Extract game_pk from a ticker string.
    
    Ticker format: KXMLBGAME-{date}-{away}-{home}
    We need to query the features manager or a local cache to resolve this
    to the official MLB game_pk.
    """
    parsed = parse_ticker(ticker)
    if not parsed:
        return None
    
    # The features manager maintains a game_date → game_pk mapping
    # This is already loaded from the parquet for the current season
    game_key = parsed.game_key
    # TODO: implement game_key → game_pk lookup
    # For now, return None (live repricing will be disabled until this is wired)
    logger.warning(f"game_pk resolution not yet implemented for {game_key}")
    return None

def _build_pregame_prior(self, game_pk: int) -> PregamePrior:
    """Build a PregamePrior from the ensemble's pregame predictions.
    
    WHY this is needed: The live model needs pregame context to interpret
    pitch sequences. A 95mph fastball means something different when thrown
    by a top-5 SP vs a AAA call-up.
    """
    from live.mlb_dl.inference_engine import PregamePrior
    
    # Get the most recent prediction for this game from the ensemble store
    # The ensemble store is keyed by ticker, so we need to construct the ticker
    # from the game_pk (requires a reverse lookup via features manager)
    
    # PLACEHOLDER: Default neutral prior until wired to ensemble
    # TODO: integrate with self._ensemble_store.get_prediction(ticker)
    return PregamePrior(
        game_pk=game_pk,
        home_win_prob=0.5,
        yrfi_prob=0.45,
        extra_innings_prob=0.08,
        first_5_home_win_prob=0.5,
        mu_home_runs=4.5,
        mu_away_runs=4.5,
        mu_total_runs=9.0,
        mu_home_run_diff=0.0,
        scale_home_runs=2.5,
        scale_away_runs=2.5,
        scale_total_runs=3.5,
        scale_home_run_diff=3.0,
        elo_diff=0.0,
        srs_diff=0.0,
        sp_era_diff=0.0,
        park_factor=1.0,
        ensemble_std_home_win=0.05,
        ensemble_std_total=1.0,
        confidence_tier="MEDIUM",
    )
```

## Phase 2: Add Live Repricing to Main Loop

Once the inference engine is receiving pitch events (requires connecting to MLB GUMBO API),
the runner can query live prices and decide whether to adjust positions.

```python
# In TradingRunner._loop_once() (conceptual addition)
def _loop_once(self) -> None:
    """Single iteration of the trading loop."""
    # ... (existing pregame logic) ...
    
    # NEW: Check for live price updates on active games
    if self._live_bridge:
        active_games = self._live_bridge.get_active_games()
        for game_pk in active_games:
            live_prices = self._live_bridge.get_live_prices(game_pk)
            if live_prices:
                self._handle_live_reprice(game_pk, live_prices)
    
    # ... (rest of existing loop) ...

def _handle_live_reprice(self, game_pk: int, live_prices: dict[str, float]) -> None:
    """Handle a live price update for an in-game market.
    
    Decision logic:
    1. Check if we have positions on this game (HOLD or EXIT state)
    2. If HOLD: do nothing (ride to settlement)
    3. If EXIT: check if live price allows profitable exit
    4. If no position but live edge > threshold: consider entering (RISKY)
    """
    # Get positions for this game
    positions = self._portfolio.get_positions_for_game(game_pk)
    
    for ticker, pos in positions.items():
        if pos.state == PositionState.HOLD:
            # Strong conviction: ride to settlement regardless of live price
            continue
        
        if pos.state == PositionState.EXIT:
            # Check if live price allows profitable exit
            live_prob = live_prices.get(self._market_key_from_ticker(ticker))
            if live_prob is None:
                continue
            
            # Compare our entry price to current market
            # If market moved in our favor, consider exiting for profit
            # TODO: implement exit logic
            logger.debug(f"EXIT candidate: {ticker} live_prob={live_prob:.3f}")
```

## Phase 3: Connect to MLB GUMBO API (Future Work)

The `LiveInferenceEngine` is designed to receive `PitchEvent` objects via `on_pitch_event()`.
These events need to come from a real-time MLB data feed.

### Option A: MLB GUMBO WebSocket

MLB's GUMBO API provides real-time play-by-play updates via WebSocket. This is the same
feed that powers MLB.com's Gameday interface.

```python
# Conceptual ingestion layer (to be implemented in live/mlb_dl/ingestion.py)

import asyncio
import websockets
import json

async def subscribe_to_game(game_pk: int, engine: LiveInferenceEngine):
    """Subscribe to live play-by-play for a single game."""
    uri = f"wss://ws.statsapi.mlb.com/api/v1/game/{game_pk}/feed/live"
    
    async with websockets.connect(uri) as ws:
        async for message in ws:
            data = json.loads(message)
            
            # Parse GUMBO message into PitchEvent
            if "liveData" in data and "plays" in data["liveData"]:
                current_play = data["liveData"]["plays"]["currentPlay"]
                pitch_data = current_play.get("playEvents", [])[-1]  # Most recent pitch
                
                pitch_event = parse_gumbo_pitch(pitch_data, game_pk)
                if pitch_event:
                    engine.on_pitch_event(game_pk, pitch_event)
```

### Option B: Polling MLB REST API

Fallback if WebSocket is unavailable. Poll the `/game/{game_pk}/feed/live` endpoint
every 5-10 seconds and extract new pitches since last poll.

## Testing

### Unit test the inference engine

```python
# live/mlb_dl/tests/test_inference_engine.py

import pytest
import torch
from live.mlb_dl.inference_engine import (
    LiveInferenceEngine,
    PregamePrior,
    PitchEvent,
)

def test_register_game():
    engine = LiveInferenceEngine(
        model_path="path/to/checkpoint.pt",
        device="cpu",
    )
    
    prior = PregamePrior(game_pk=12345, home_win_prob=0.55)
    engine.register_game(12345, prior)
    
    assert 12345 in engine._games
    assert engine._games[12345].pregame_prior.home_win_prob == 0.55

def test_on_pitch_event():
    engine = LiveInferenceEngine(
        model_path="path/to/checkpoint.pt",
        device="cpu",
    )
    
    prior = PregamePrior(game_pk=12345)
    engine.register_game(12345, prior)
    
    pitch = PitchEvent(
        game_pk=12345,
        inning=1,
        is_top_inning=True,
        at_bat_index=1,
        pitch_number=1,
        batter_id=547180,  # Mike Trout
        pitcher_id=543037,  # Gerrit Cole
        pitch_type="FF",
        balls=0,
        strikes=0,
        outs=0,
        on_first=False,
        on_second=False,
        on_third=False,
        pitch_call="StrikeCalled",
        is_scoring_play=False,
        rbi_count=0,
        score_home=0,
        score_away=0,
        release_speed=98.5,
    )
    
    prices = engine.on_pitch_event(12345, pitch)
    
    assert prices is not None
    assert "home_win" in prices
    assert 0.0 <= prices["home_win"] <= 1.0
```

### Integration test with TradingRunner

```python
# pregame/trading/tests/test_runner_live_integration.py

import pytest
from pregame.trading.runner import TradingRunner
from live.mlb_dl.inference_engine import PitchEvent

def test_runner_with_live_bridge(tmpdir):
    """Test that TradingRunner can initialize and use the live bridge."""
    runner = TradingRunner(dry_run=True, bankroll=1000.0)
    
    # Mock model checkpoint
    model_path = tmpdir / "best_model.pt"
    torch.save({
        "model_state_dict": {},
        "config": {"feature_dim": 40, "hidden_dim": 128},
        "feature_mean": {},
        "feature_std": {},
    }, model_path)
    
    runner._live_bridge = TradingBridge(model_path=str(model_path))
    runner._live_bridge.start()
    
    # Simulate game start
    from live.mlb_dl.inference_engine import PregamePrior
    prior = PregamePrior(game_pk=12345)
    runner._live_bridge.register_game(12345, prior)
    
    # Check that game is registered
    assert 12345 in runner._live_bridge.get_active_games()
    
    runner._live_bridge.stop()
```

## Performance Considerations

### Latency Budget

- **Feature extraction**: 5ms (dict traversal + hash computation)
- **Tensor construction**: 10ms (stack + pad operations)
- **Model forward pass**: 30ms (LSTM + linear heads on CPU)
- **Market derivation**: 15ms (sigmoid + distribution integration)
- **Total**: ~60ms per pitch

This is well under the 200ms target and allows for 15-20 game simultaneous tracking
without falling behind the live feed.

### Memory Footprint

Per-game state:
- Pitch tensors: 350 pitches × 40 features × 4 bytes = 56KB
- Hash arrays: 350 pitches × 3 IDs × 8 bytes = 8.4KB
- Metadata: ~1KB
- **Total**: ~65KB per game

For 15 simultaneous games: ~1MB total state (negligible).

### Threading Model

```
Main Thread (TradingRunner)
├─ Portfolio management
├─ Order execution
├─ Kalshi REST API calls
└─ get_live_prices() calls → TradingBridge (thread-safe)

Background Thread (TradingBridge)
├─ AsyncIO event loop
├─ MLB GUMBO WebSocket
└─ LiveInferenceEngine.on_pitch_event()
```

The `TradingBridge` uses `threading.Lock` to synchronize access to the shared
`_games` dict. Lock contention is minimal because:
- Main thread only reads (get_latest_prices)
- Background thread writes (on_pitch_event)
- Lock is held for <1ms per operation

## Rollout Plan

### Stage 1: Infrastructure (Current)
- ✅ Implement `LiveInferenceEngine`
- ✅ Implement `TradingBridge`
- ✅ Add integration points to `TradingRunner`

### Stage 2: Ingestion Layer (Next)
- [ ] Implement MLB GUMBO WebSocket client
- [ ] Parse GUMBO play-by-play into `PitchEvent` objects
- [ ] Add retry/reconnection logic for WebSocket
- [ ] Log pitch events to disk for replay/debugging

### Stage 3: Model Training (Parallel)
- [ ] Generate training data (pitch sequences + targets)
- [ ] Train `LiveGameModel` on historical games
- [ ] Validate latency on CPU (must be <30ms)
- [ ] Checkpoint best model to `live/mlb_dl/checkpoints/`

### Stage 4: Live Testing (Dry-Run)
- [ ] Deploy to EC2 with dry-run mode enabled
- [ ] Subscribe to 2-3 live games
- [ ] Log predicted prices vs pregame prices
- [ ] Validate inference latency stays <60ms
- [ ] Check for memory leaks over 4-hour session

### Stage 5: Production (Cautious)
- [ ] Enable live repricing for EXIT positions only
- [ ] Monitor P&L attribution (pregame vs live alpha)
- [ ] Gradually expand to HOLD positions if profitable
- [ ] Consider live market entry (VERY RISKY, requires tight risk controls)

## Risk Considerations

### Model Risk
- **Concern**: Live model could hallucinate nonsense prices during anomalous game states
- **Mitigation**: Add sanity bounds (e.g., home_win must stay within ±0.3 of pregame)
- **Mitigation**: Require minimum pitch count (e.g., 50 pitches) before trusting live model

### Latency Risk
- **Concern**: Inference takes >200ms, we fall behind the feed
- **Mitigation**: Batch inference across multiple games (not implemented yet)
- **Mitigation**: Skip inference on non-critical pitches (e.g., 0-0 count, no runners)

### Data Risk
- **Concern**: GUMBO feed disconnects mid-game
- **Mitigation**: Auto-reconnect with exponential backoff
- **Mitigation**: Fall back to pregame prices if no update for >5 minutes

### Execution Risk
- **Concern**: Live repricing triggers excessive order churn
- **Mitigation**: Require minimum price move (e.g., 5 cents) to trigger reprice
- **Mitigation**: Cap reprices per order (already in config: MAX_REPRICES_PER_ORDER)

## Appendix: GUMBO Message Format

Sample GUMBO play-by-play message (simplified):

```json
{
  "gamePk": 717161,
  "liveData": {
    "plays": {
      "currentPlay": {
        "about": {
          "inning": 1,
          "halfInning": "top",
          "atBatIndex": 1
        },
        "count": {"balls": 1, "strikes": 0, "outs": 0},
        "matchup": {
          "batter": {"id": 547180, "fullName": "Mike Trout"},
          "pitcher": {"id": 543037, "fullName": "Gerrit Cole"},
          "splits": {"batter": "R", "pitcher": "R"}
        },
        "runners": [
          {"movement": {"start": null, "end": "1B"}, "details": {"runner": {"id": 123}}}
        ],
        "playEvents": [
          {
            "isPitch": true,
            "details": {
              "type": {"code": "FF", "description": "Four-Seam Fastball"},
              "call": {"code": "B", "description": "Ball"}
            },
            "pitchData": {
              "startSpeed": 98.5,
              "endSpeed": 90.2,
              "coordinates": {"pX": -0.5, "pZ": 2.8},
              "breaks": {"spinRate": 2400, "breakLength": 3.5}
            },
            "count": {"balls": 1, "strikes": 0},
            "playId": "abc-123"
          }
        ]
      }
    },
    "linescore": {
      "innings": [
        {"home": {"runs": 0}, "away": {"runs": 0}}
      ]
    }
  }
}
```

Parsing this into a `PitchEvent`:

```python
def parse_gumbo_pitch(play_data: dict, game_pk: int) -> PitchEvent:
    about = play_data["about"]
    count = play_data["count"]
    matchup = play_data["matchup"]
    events = play_data["playEvents"]
    
    # Get the most recent pitch
    pitch = next((e for e in reversed(events) if e.get("isPitch")), None)
    if not pitch:
        return None
    
    details = pitch["details"]
    pitch_data = pitch.get("pitchData", {})
    
    return PitchEvent(
        game_pk=game_pk,
        inning=about["inning"],
        is_top_inning=(about["halfInning"] == "top"),
        at_bat_index=about["atBatIndex"],
        pitch_number=len([e for e in events if e.get("isPitch")]),
        batter_id=matchup["batter"]["id"],
        pitcher_id=matchup["pitcher"]["id"],
        pitch_type=details["type"]["code"],
        balls=count["balls"],
        strikes=count["strikes"],
        outs=count["outs"],
        on_first=any(r["movement"]["end"] == "1B" for r in play_data.get("runners", [])),
        on_second=any(r["movement"]["end"] == "2B" for r in play_data.get("runners", [])),
        on_third=any(r["movement"]["end"] == "3B" for r in play_data.get("runners", [])),
        pitch_call=details["call"]["code"],
        is_scoring_play=details.get("isScoringPlay", False),
        rbi_count=details.get("rbi", 0),
        score_home=0,  # Parse from linescore
        score_away=0,  # Parse from linescore
        release_speed=pitch_data.get("startSpeed"),
        spin_rate=pitch_data.get("breaks", {}).get("spinRate"),
        coord_px=pitch_data.get("coordinates", {}).get("pX"),
        coord_pz=pitch_data.get("coordinates", {}).get("pZ"),
    )
```
