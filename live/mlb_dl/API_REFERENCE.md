# Live Inference Engine — API Reference

Quick reference for the `LiveInferenceEngine` and `TradingBridge` APIs.

---

## LiveInferenceEngine

**Import**:
```python
from live.mlb_dl.inference_engine import (
    LiveInferenceEngine,
    PregamePrior,
    PitchEvent,
)
```

### Constructor

```python
engine = LiveInferenceEngine(
    model_path: str | Path,          # Path to trained model checkpoint (.pt file)
    device: str = "cpu",             # "cpu" or "cuda" (cpu recommended)
    on_reprice: Callable | None = None  # Optional callback(game_pk, prices)
)
```

**Example**:
```python
def my_callback(game_pk: int, prices: dict[str, float]):
    print(f"Game {game_pk}: home_win={prices['home_win']:.3f}")

engine = LiveInferenceEngine(
    model_path="checkpoints/best_model.pt",
    on_reprice=my_callback,
)
```

### Methods

#### `register_game(game_pk, pregame_prior)`

Initialize inference state for a new live game.

**Parameters**:
- `game_pk` (int): MLB game primary key
- `pregame_prior` (PregamePrior): Pregame model output for this game

**Example**:
```python
prior = PregamePrior(
    game_pk=12345,
    home_win_prob=0.55,
    yrfi_prob=0.48,
    mu_total_runs=9.0,
    mu_home_run_diff=0.5,
)
engine.register_game(12345, prior)
```

**WHY**: The engine needs pregame context to interpret pitch sequences. A 95mph fastball from Gerrit Cole means something different than from a AAA call-up.

---

#### `on_pitch_event(game_pk, pitch_event)`

Process a new pitch event and return updated market prices.

**Parameters**:
- `game_pk` (int): MLB game primary key
- `pitch_event` (PitchEvent): Pitch data from live feed

**Returns**:
- `dict[str, float] | None`: Market prices, or None if game not registered

**Example**:
```python
pitch = PitchEvent(
    game_pk=12345,
    inning=1,
    is_top_inning=True,
    at_bat_index=1,
    pitch_number=1,
    batter_id=547180,  # Mike Trout
    pitcher_id=543037,  # Gerrit Cole
    pitch_type="FF",    # Four-seam fastball
    balls=0, strikes=0, outs=0,
    on_first=False, on_second=False, on_third=False,
    pitch_call="StrikeCalled",
    is_scoring_play=False,
    rbi_count=0,
    score_home=0,
    score_away=0,
    release_speed=98.5,
)

prices = engine.on_pitch_event(12345, pitch)
# Returns: {"home_win": 0.56, "yrfi": 0.49, "total_runs_mu": 8.8, ...}
```

**Latency**: <60ms on CPU

---

#### `get_latest_prices(game_pk)`

Get the most recent market prices for a game (cached from last inference).

**Parameters**:
- `game_pk` (int): MLB game primary key

**Returns**:
- `dict[str, float] | None`: Cached prices, or None if no inference yet

**Example**:
```python
prices = engine.get_latest_prices(12345)
if prices:
    print(f"home_win: {prices['home_win']:.3f}")
```

**WHY**: Avoids redundant inference when multiple components need the same prices.

---

#### `unregister_game(game_pk)`

Clean up state for a completed game.

**Parameters**:
- `game_pk` (int): MLB game primary key

**Example**:
```python
engine.unregister_game(12345)
```

**WHY**: Frees memory (65KB per game). Call when game reaches final status.

---

## TradingBridge

**Import**:
```python
from live.mlb_dl.inference_engine import TradingBridge, PregamePrior
```

### Constructor

```python
bridge = TradingBridge(
    model_path: str | Path,  # Path to trained model checkpoint
    device: str = "cpu",     # "cpu" or "cuda"
)
```

**Example**:
```python
bridge = TradingBridge(model_path="checkpoints/best_model.pt")
```

### Methods

#### `start()`

Start the background event loop + inference engine.

**Example**:
```python
bridge.start()
# Blocks until engine initialized (max 5 seconds)
```

**WHY**: Runs async event loop in background thread so trading runner (sync) can use the engine.

---

#### `register_game(game_pk, pregame_prior)`

Thread-safe: register a new game for live tracking.

**Parameters**:
- `game_pk` (int): MLB game primary key
- `pregame_prior` (PregamePrior): Pregame model output

**Example**:
```python
prior = PregamePrior(game_pk=12345, home_win_prob=0.55)
bridge.register_game(12345, prior)
```

**Thread-Safety**: Can be called from any thread (e.g., main trading loop).

---

#### `get_live_prices(game_pk)`

Thread-safe: get latest market prices for a game.

**Parameters**:
- `game_pk` (int): MLB game primary key

**Returns**:
- `dict[str, float] | None`: Prices, or None if no inference yet

**Example**:
```python
prices = bridge.get_live_prices(12345)
if prices:
    print(f"home_win: {prices['home_win']:.3f}")
```

**Thread-Safety**: Uses `threading.Lock` internally (<1ms lock contention).

---

#### `get_active_games()`

Thread-safe: list of games currently being tracked.

**Returns**:
- `list[int]`: List of game_pks

**Example**:
```python
active = bridge.get_active_games()
for game_pk in active:
    prices = bridge.get_live_prices(game_pk)
    # ... decide whether to reprice ...
```

---

#### `stop()`

Graceful shutdown.

**Example**:
```python
bridge.stop()
```

**WHY**: Stops background thread and event loop. Call during runner shutdown.

---

## Data Classes

### PregamePrior

Pregame model output for a single game.

**Fields**:
```python
@dataclass
class PregamePrior:
    game_pk: int
    # Classification probabilities
    home_win_prob: float = 0.5
    yrfi_prob: float = 0.45
    extra_innings_prob: float = 0.08
    first_5_home_win_prob: float = 0.5
    # Regression parameters (NegBin)
    mu_home_runs: float = 4.5
    mu_away_runs: float = 4.5
    mu_total_runs: float = 9.0
    mu_home_run_diff: float = 0.0
    scale_home_runs: float = 2.5
    scale_away_runs: float = 2.5
    scale_total_runs: float = 3.5
    scale_home_run_diff: float = 3.0
    # Key pregame features
    elo_diff: float = 0.0
    srs_diff: float = 0.0
    sp_era_diff: float = 0.0
    park_factor: float = 1.0
    # Model uncertainty
    ensemble_std_home_win: float = 0.05
    ensemble_std_total: float = 1.0
    confidence_tier: str = "MEDIUM"  # HIGH/MEDIUM/LOW
```

**Example**:
```python
prior = PregamePrior(
    game_pk=12345,
    home_win_prob=0.55,  # Home team favored
    yrfi_prob=0.48,
    mu_total_runs=9.0,
    elo_diff=40.0,       # Home team +40 Elo
    sp_era_diff=-0.5,    # Home SP has lower ERA
    park_factor=1.05,    # Hitter-friendly park
    confidence_tier="HIGH",
)
```

---

### PitchEvent

A single pitch from the live feed (MLB GUMBO API).

**Fields**:
```python
@dataclass
class PitchEvent:
    game_pk: int
    inning: int
    is_top_inning: bool
    at_bat_index: int
    pitch_number: int
    # Identities (hash-bucketed)
    batter_id: int
    pitcher_id: int
    pitch_type: str
    # Count state
    balls: int
    strikes: int
    outs: int
    # Runners
    on_first: bool
    on_second: bool
    on_third: bool
    # Outcome
    pitch_call: str  # "InPlay", "StrikeCalled", "BallCalled", etc.
    is_scoring_play: bool
    rbi_count: int
    # Score (current state after this pitch)
    score_home: int
    score_away: int
    # Statcast (optional)
    release_speed: float | None = None
    spin_rate: float | None = None
    break_length: float | None = None
    coord_px: float | None = None  # Horizontal plate location
    coord_pz: float | None = None  # Vertical plate location
    hit_launch_speed: float | None = None
    hit_launch_angle: float | None = None
```

**Example**:
```python
pitch = PitchEvent(
    game_pk=12345,
    inning=1,
    is_top_inning=True,
    at_bat_index=1,
    pitch_number=1,
    batter_id=547180,  # Mike Trout
    pitcher_id=543037,  # Gerrit Cole
    pitch_type="FF",    # Four-seam fastball
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
    spin_rate=2450,
    coord_px=0.1,
    coord_pz=2.5,
)
```

---

## Market Prices Output

The `on_pitch_event()` and `get_latest_prices()` methods return a dict with the following keys:

```python
{
    # Classification markets
    "home_win": 0.56,        # P(home team wins)
    "away_win": 0.44,        # P(away team wins) = 1 - home_win
    "yrfi": 0.49,            # P(run scored in 1st inning)
    "nrfi": 0.51,            # P(no run in 1st inning) = 1 - yrfi
    
    # Regression parameters (for deriving totals/spreads)
    "total_runs_mu": 8.8,       # Mean total runs
    "total_runs_sigma": 3.2,    # Std dev total runs
    "home_run_diff_mu": 0.5,    # Mean run differential (home - away)
    "home_run_diff_sigma": 2.8, # Std dev run differential
}
```

**Note**: Full market derivation (totals at specific lines, spreads, team totals) requires integrating `distributions.py`. This is a future enhancement (see `TODO_LIVE_INTEGRATION.md`).

---

## Callbacks

### on_reprice Callback

Fired after each inference. Use this to trigger repricing decisions.

**Signature**:
```python
def on_reprice(game_pk: int, prices: dict[str, float]) -> None:
    pass
```

**Example**:
```python
def handle_reprice(game_pk: int, prices: dict[str, float]):
    logger.info(f"Game {game_pk} repriced")
    
    # Check if we have positions on this game
    positions = portfolio.get_positions_for_game(game_pk)
    
    for ticker, pos in positions.items():
        if pos.state == PositionState.EXIT:
            # Check if live price allows profitable exit
            market_key = parse_ticker(ticker).market_key
            live_prob = prices.get(market_key)
            if live_prob and abs(live_prob - pos.entry_prob) > 0.05:
                # Consider exiting
                logger.info(f"EXIT candidate: {ticker}")

engine = LiveInferenceEngine(
    model_path="checkpoints/best_model.pt",
    on_reprice=handle_reprice,
)
```

---

## Usage Patterns

### Pattern 1: Standalone Inference (No Trading)

```python
from live.mlb_dl.inference_engine import LiveInferenceEngine, PregamePrior, PitchEvent

# Initialize
engine = LiveInferenceEngine(model_path="checkpoints/best_model.pt")

# Register game
prior = PregamePrior(game_pk=12345, home_win_prob=0.55)
engine.register_game(12345, prior)

# Process pitches
for pitch in pitch_stream:  # From MLB GUMBO WebSocket
    prices = engine.on_pitch_event(12345, pitch)
    print(f"home_win: {prices['home_win']:.3f}")

# Cleanup
engine.unregister_game(12345)
```

---

### Pattern 2: Integration with TradingRunner (Recommended)

```python
from live.mlb_dl.inference_engine import TradingBridge, PregamePrior

class TradingRunner:
    def __init__(self, ...):
        self._live_bridge = None
    
    def start(self):
        # ... (existing startup) ...
        
        # Start live inference bridge
        self._live_bridge = TradingBridge(model_path="checkpoints/best_model.pt")
        self._live_bridge.start()
    
    def _handle_game_start(self, ticker: str):
        # ... (cancel unfilled orders) ...
        
        # Register game with live inference
        game_pk = self._resolve_game_pk(ticker)
        prior = self._build_pregame_prior(game_pk)
        self._live_bridge.register_game(game_pk, prior)
    
    def _loop_once(self):
        # ... (existing pregame logic) ...
        
        # Check for live price updates
        for game_pk in self._live_bridge.get_active_games():
            prices = self._live_bridge.get_live_prices(game_pk)
            if prices:
                self._handle_live_reprice(game_pk, prices)
```

---

### Pattern 3: Multi-Game Monitoring

```python
from live.mlb_dl.inference_engine import TradingBridge

bridge = TradingBridge(model_path="checkpoints/best_model.pt")
bridge.start()

# Register multiple games
for game_pk in [12345, 12346, 12347]:
    prior = PregamePrior(game_pk=game_pk)
    bridge.register_game(game_pk, prior)

# Monitor all games
while True:
    for game_pk in bridge.get_active_games():
        prices = bridge.get_live_prices(game_pk)
        if prices:
            print(f"Game {game_pk}: home_win={prices['home_win']:.3f}")
    
    time.sleep(1)  # Poll every second
```

---

## Error Handling

### Common Errors

**FileNotFoundError**: Model checkpoint not found
```python
try:
    engine = LiveInferenceEngine(model_path="checkpoints/best_model.pt")
except FileNotFoundError:
    print("Model not found. Train first with: python -m live.mlb_dl.train")
```

**RuntimeError**: TradingBridge failed to initialize
```python
bridge = TradingBridge(model_path="checkpoints/best_model.pt")
try:
    bridge.start()
except RuntimeError as e:
    print(f"Bridge initialization failed: {e}")
```

**None returned**: Game not registered
```python
prices = engine.on_pitch_event(12345, pitch)
if prices is None:
    print("Warning: Game 12345 not registered")
```

---

## Performance Tips

### Latency

- **Use CPU, not GPU**: Single-batch inference is latency-bound by memory bandwidth, not FLOPS. GPU adds ~10ms kernel launch overhead.
- **Skip non-critical pitches**: Only run inference on critical state changes (score, RISP, 3-2 count).
- **Batch across games**: For 15+ simultaneous games, batch inference (future enhancement).

### Memory

- **Unregister completed games**: Call `unregister_game()` when game reaches final status to free 65KB per game.
- **Limit max sequence length**: 350 pitches covers 99th percentile. Longer sequences waste memory.

### Thread Safety

- **Use TradingBridge for multi-threaded access**: The bridge uses `threading.Lock` internally (<1ms contention).
- **Avoid direct engine access from multiple threads**: `LiveInferenceEngine` is not thread-safe on its own.

---

## See Also

- **`INFERENCE_ENGINE_README.md`**: Comprehensive overview + architecture
- **`runner_integration_guide.md`**: Step-by-step integration with TradingRunner
- **`example_inference.py`**: Standalone example script
- **`tests/test_inference_engine.py`**: Unit test suite
- **`TODO_LIVE_INTEGRATION.md`**: Remaining work (model training, ingestion layer, etc.)

---

**Last Updated**: 2026-07-09
