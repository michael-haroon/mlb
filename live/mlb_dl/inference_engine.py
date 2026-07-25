"""
Live inference engine — maintains per-game state and triggers model repricing.

Latency budget:
  Feature extraction:  5ms
  Tensor construction: 10ms
  Model forward pass:  30ms
  Market derivation:   15ms
  Total:              ~60ms (well under 200ms target)

WHY this architecture:
- Append-only tensor buffer: New pitches append to cached sequence. No recomputation
  of previous embeddings needed because the LSTM processes the full sequence each time
  (not autoregressive). Memory: 350 pitches × 128 dims × 4 bytes = 175KB per game.
- Pregame prior injection: Pregame module's output is projected and passed to model's
  cross-attention, ensuring live predictions are conditioned on team quality.
- Selective repricing: Not every pitch warrants the 30ms inference cost. Repricing
  triggers are defined by the ingestion layer (critical count/score changes).
- Smooth transition: A learned sigmoid gate blends live vs pregame predictions.
  At game start, gate ~0 (trust pregame). By mid-game, gate ~1 (trust live model).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .models import LiveGameModel

log = logging.getLogger(__name__)


@dataclass
class PregamePrior:
    """Pregame model's output for a single game, used to condition the live model.

    WHY this exists: The live model needs to know team quality, SP matchup,
    and park effects to properly interpret pitch sequences. A 95mph fastball
    means something different from Gerrit Cole than from a AAA call-up.
    """
    game_pk: int
    # Classification probabilities
    home_win_prob: float = 0.5
    yrfi_prob: float = 0.45
    extra_innings_prob: float = 0.08
    first_5_home_win_prob: float = 0.5
    # Regression parameters (from NegBin pregame model)
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

    def to_tensor(self) -> torch.Tensor:
        """Convert to a flat tensor for model input (d_pregame = 19 dims)."""
        values = [
            self.home_win_prob, self.yrfi_prob, self.extra_innings_prob,
            self.first_5_home_win_prob,
            self.mu_home_runs, self.mu_away_runs, self.mu_total_runs, self.mu_home_run_diff,
            self.scale_home_runs, self.scale_away_runs, self.scale_total_runs, self.scale_home_run_diff,
            self.elo_diff / 200.0,  # Normalize to ~[-1, 1]
            self.srs_diff / 5.0,
            self.sp_era_diff / 3.0,
            self.park_factor - 1.0,  # Center at 0
            self.ensemble_std_home_win,
            self.ensemble_std_total / 3.0,
            # Confidence tier as ordinal
            {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.0}.get(self.confidence_tier, 0.5),
        ]
        return torch.tensor(values, dtype=torch.float32)


@dataclass
class PitchEvent:
    """A single pitch event from the live ingestion stream.

    This dataclass mirrors the structure of the PITCHES parquet table but is
    populated from a real-time feed (MLB GUMBO API or equivalent).
    """
    game_pk: int
    inning: int
    is_top_inning: bool
    at_bat_index: int
    pitch_number: int
    # Identities (will be hash-bucketed)
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
    # Statcast (optional, may be None for non-batted balls)
    release_speed: Optional[float] = None
    spin_rate: Optional[float] = None
    break_length: Optional[float] = None
    coord_px: Optional[float] = None  # horizontal plate location
    coord_pz: Optional[float] = None  # vertical plate location
    hit_launch_speed: Optional[float] = None
    hit_launch_angle: Optional[float] = None


@dataclass
class GameInferenceState:
    """Per-game tensor state for the inference engine."""
    game_pk: int
    pregame_prior: PregamePrior
    # Growing list of pitch feature tensors
    pitch_tensors: list[torch.Tensor] = field(default_factory=list)
    batter_hashes: list[int] = field(default_factory=list)
    pitcher_hashes: list[int] = field(default_factory=list)
    pitch_type_hashes: list[int] = field(default_factory=list)
    pitch_count: int = 0
    # Hierarchy tracking (for positional encoding when/if we upgrade to HAN)
    current_inning: int = 1
    current_ab_in_inning: int = 0
    current_pitch_in_ab: int = 0
    ab_index_global: int = 0
    # Last inference result (for caching/comparison)
    last_result: Optional[dict[str, float]] = None
    last_inference_time: float = 0.0
    # Game start time (for elapsed time feature)
    game_start_time: float = 0.0
    # Score tracking (redundant with pitch events but fast access)
    score_home: int = 0
    score_away: int = 0
    innings_completed: float = 0.0


class LiveInferenceEngine:
    """Maintains per-game state and triggers model inference on new events.

    Lifecycle:
      1. register_game(game_pk, pregame_prior) — called at first pitch
      2. on_pitch_event(game_pk, pitch_event) — called by ingestion layer
      3. get_latest_prices(game_pk) → dict — called by trading runner
      4. unregister_game(game_pk) — called at game final
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        on_reprice: Optional[Callable[[int, dict[str, float]], None]] = None,
    ):
        """
        Parameters
        ----------
        model_path : path to saved model checkpoint (.pt file)
        device : "cpu" or "cuda" — cpu is sufficient for single-game inference
        on_reprice : callback(game_pk, market_prices) fired after each inference
        """
        self.device = torch.device(device)
        self.on_reprice = on_reprice

        # Load model checkpoint
        log.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Extract model config from checkpoint
        config = checkpoint.get("config", {})
        self.model = LiveGameModel(
            feature_dim=config.get("feature_dim", 40),  # TODO: validate — placeholder
            hidden_dim=config.get("hidden_dim", 128),
            dropout=0.0,  # No dropout during inference
            batter_buckets=config.get("batter_buckets", 50000),
            pitcher_buckets=config.get("pitcher_buckets", 50000),
            pitch_type_buckets=config.get("pitch_type_buckets", 256),
            embed_dim=config.get("embed_dim", 16),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Feature standardization (mean/std from training)
        self.feature_mean = checkpoint.get("feature_mean", {})
        self.feature_std = checkpoint.get("feature_std", {})

        # Per-game state
        self._games: dict[int, GameInferenceState] = {}
        self._lock = threading.Lock()

        log.info(f"Inference engine initialized (device={device}, hidden_dim={config.get('hidden_dim', 128)})")

    def register_game(self, game_pk: int, pregame_prior: PregamePrior) -> None:
        """Initialize inference state for a new live game."""
        with self._lock:
            if game_pk in self._games:
                log.warning(f"Game {game_pk} already registered, overwriting")

            self._games[game_pk] = GameInferenceState(
                game_pk=game_pk,
                pregame_prior=pregame_prior,
                game_start_time=time.time(),
            )
            log.info(f"Registered game {game_pk} for live inference")

    def on_pitch_event(self, game_pk: int, pitch_event: PitchEvent) -> Optional[dict[str, float]]:
        """Process a new pitch event. Returns updated market prices or None.

        Steps:
          1. Encode pitch_event into tensor features
          2. Append to game's growing sequence
          3. Update hierarchy indices (inning/AB/pitch tracking)
          4. Run model forward pass
          5. Derive market prices from model output
          6. Fire on_reprice callback
          7. Return market prices dict
        """
        with self._lock:
            state = self._games.get(game_pk)
            if state is None:
                log.warning(f"Received pitch for unregistered game {game_pk}")
                return None

            # 1. Encode pitch event
            pitch_tensor, batter_hash, pitcher_hash, ptype_hash = self._encode_pitch(pitch_event, state)

            # 2. Append to sequence
            state.pitch_tensors.append(pitch_tensor)
            state.batter_hashes.append(batter_hash)
            state.pitcher_hashes.append(pitcher_hash)
            state.pitch_type_hashes.append(ptype_hash)
            state.pitch_count += 1

            # 3. Update hierarchy tracking
            self._update_hierarchy(state, pitch_event)

            # 4. Update score tracking
            state.score_home = pitch_event.score_home
            state.score_away = pitch_event.score_away

            # 5. Compute innings completed (for time-based features)
            state.innings_completed = pitch_event.inning - 1 + (0.5 if pitch_event.is_top_inning else 1.0)

        # 6. Run inference (release lock during computation)
        start = time.time()
        market_prices = self._run_inference(state)
        elapsed_ms = (time.time() - start) * 1000

        with self._lock:
            state.last_result = market_prices
            state.last_inference_time = time.time()

        log.debug(f"Game {game_pk} pitch #{state.pitch_count}: inference took {elapsed_ms:.1f}ms")

        # 7. Fire callback
        if self.on_reprice and market_prices:
            self.on_reprice(game_pk, market_prices)

        return market_prices

    def get_latest_prices(self, game_pk: int) -> Optional[dict[str, float]]:
        """Return the most recent market prices for a game (cached from last inference)."""
        with self._lock:
            state = self._games.get(game_pk)
            return state.last_result if state else None

    def unregister_game(self, game_pk: int) -> None:
        """Clean up state for a completed game."""
        with self._lock:
            state = self._games.pop(game_pk, None)
            if state:
                log.info(f"Unregistered game {game_pk} (processed {state.pitch_count} pitches)")
            else:
                log.warning(f"Attempted to unregister unknown game {game_pk}")

    def _encode_pitch(
        self,
        event: PitchEvent,
        state: GameInferenceState
    ) -> tuple[torch.Tensor, int, int, int]:
        """Convert a PitchEvent into the tensor format expected by the model.

        Returns (pitch_feature_tensor, batter_hash, pitcher_hash, pitch_type_hash).

        WHY these features: Each pitch is characterized by:
        - Count state (balls, strikes, outs) — game-theoretic context
        - Base runners — run expectancy context
        - Inning/score — leverage and urgency
        - Statcast metrics — pitch quality and outcome
        - Elapsed time — fatigue proxy
        """
        # Normalize count state
        balls_norm = event.balls / 3.0
        strikes_norm = event.strikes / 2.0
        outs_norm = event.outs / 2.0

        # Base state encoding
        runners = float(event.on_first) + float(event.on_second) + float(event.on_third)
        runners_norm = runners / 3.0

        # Score differential (home perspective)
        score_diff = event.score_home - event.score_away
        score_diff_norm = np.tanh(score_diff / 5.0)  # Squash to [-1, 1]

        # Inning features
        inning_norm = event.inning / 9.0
        inning_phase = 1.0 if event.inning >= 7 else 0.0  # Late-game flag

        # Elapsed time (minutes since game start)
        elapsed_min = (time.time() - state.game_start_time) / 60.0
        elapsed_norm = np.tanh(elapsed_min / 180.0)  # Normalize by ~3hr game

        # Statcast features (with null handling)
        release_speed_norm = self._standardize("release_speed", event.release_speed)
        spin_rate_norm = self._standardize("spin_rate", event.spin_rate)
        break_length_norm = self._standardize("break_length", event.break_length)
        px_norm = self._standardize("coord_px", event.coord_px)
        pz_norm = self._standardize("coord_pz", event.coord_pz)
        launch_speed_norm = self._standardize("hit_launch_speed", event.hit_launch_speed)
        launch_angle_norm = self._standardize("hit_launch_angle", event.hit_launch_angle)

        # Outcome flags
        is_in_play = 1.0 if "InPlay" in event.pitch_call else 0.0
        is_strike = 1.0 if "Strike" in event.pitch_call else 0.0
        is_ball = 1.0 if "Ball" in event.pitch_call else 0.0
        is_scoring = 1.0 if event.is_scoring_play else 0.0

        # Assemble feature vector (40 dims)
        # TODO: validate — placeholder feature set
        features = [
            balls_norm, strikes_norm, outs_norm,
            runners_norm, float(event.on_first), float(event.on_second), float(event.on_third),
            score_diff_norm, float(event.score_home), float(event.score_away),
            inning_norm, inning_phase, float(event.is_top_inning),
            elapsed_norm,
            release_speed_norm, spin_rate_norm, break_length_norm,
            px_norm, pz_norm,
            launch_speed_norm, launch_angle_norm,
            is_in_play, is_strike, is_ball, is_scoring,
            float(event.rbi_count),
            # Pregame context (repeated at each pitch for cross-attention)
            state.pregame_prior.home_win_prob,
            state.pregame_prior.yrfi_prob,
            state.pregame_prior.mu_total_runs / 10.0,
            state.pregame_prior.park_factor - 1.0,
            # Pitch sequence position
            float(event.pitch_number) / 10.0,
            float(event.at_bat_index) / 50.0,
            # Count milestone flags
            1.0 if event.balls == 3 else 0.0,  # Full count pressure
            1.0 if event.strikes == 2 else 0.0,  # Two-strike approach
            1.0 if event.outs == 2 else 0.0,  # Two-out tension
            # Dummy padding to reach 40 dims
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]

        pitch_tensor = torch.tensor(features[:40], dtype=torch.float32)

        # Hash identities
        batter_hash = self._hash_player_id(event.batter_id)
        pitcher_hash = self._hash_player_id(event.pitcher_id)
        ptype_hash = self._hash_string(event.pitch_type)

        return pitch_tensor, batter_hash, pitcher_hash, ptype_hash

    def _standardize(self, feature_name: str, value: Optional[float]) -> float:
        """Standardize a feature using training set statistics.

        Returns 0.0 for null values (which are common for Statcast metrics when
        the ball is not put in play).
        """
        if value is None or not np.isfinite(value):
            return 0.0

        mean = self.feature_mean.get(feature_name, 0.0)
        std = self.feature_std.get(feature_name, 1.0)
        return float((value - mean) / std) if std > 1e-6 else 0.0

    def _build_batch(self, state: GameInferenceState) -> dict[str, torch.Tensor]:
        """Construct a full model batch from the accumulated pitch tensors.

        Handles:
        - Padding to max_seq_len (350)
        - Attention mask construction
        - Pregame prior tensor
        """
        seq_len = len(state.pitch_tensors)
        max_seq_len = 350  # TODO: validate — placeholder

        # Stack pitch features
        if seq_len == 0:
            # Edge case: no pitches yet (shouldn't happen, but defensive)
            values = torch.zeros((1, max_seq_len, 40), dtype=torch.float32)
            padding = torch.zeros((1, max_seq_len), dtype=torch.float32)
            batter_hashes = torch.zeros((1, max_seq_len), dtype=torch.long)
            pitcher_hashes = torch.zeros((1, max_seq_len), dtype=torch.long)
            ptype_hashes = torch.zeros((1, max_seq_len), dtype=torch.long)
        else:
            # Stack tensors
            stacked = torch.stack(state.pitch_tensors[-max_seq_len:], dim=0)  # Take last N pitches
            actual_len = stacked.shape[0]

            # Pad if necessary
            if actual_len < max_seq_len:
                pad_len = max_seq_len - actual_len
                values = F.pad(stacked, (0, 0, 0, pad_len), value=0.0).unsqueeze(0)
                padding = torch.cat([
                    torch.ones(actual_len, dtype=torch.float32),
                    torch.zeros(pad_len, dtype=torch.float32)
                ]).unsqueeze(0)
            else:
                values = stacked.unsqueeze(0)
                padding = torch.ones((1, max_seq_len), dtype=torch.float32)

            # Identity hashes (take last N, pad if needed)
            batter_arr = state.batter_hashes[-max_seq_len:]
            pitcher_arr = state.pitcher_hashes[-max_seq_len:]
            ptype_arr = state.pitch_type_hashes[-max_seq_len:]

            if len(batter_arr) < max_seq_len:
                pad_len = max_seq_len - len(batter_arr)
                # Right-pad to match feature tensor padding direction
                batter_arr = batter_arr + [0] * pad_len
                pitcher_arr = pitcher_arr + [0] * pad_len
                ptype_arr = ptype_arr + [0] * pad_len

            batter_hashes = torch.tensor([batter_arr], dtype=torch.long)
            pitcher_hashes = torch.tensor([pitcher_arr], dtype=torch.long)
            ptype_hashes = torch.tensor([ptype_arr], dtype=torch.long)

        # Mask (all features are valid — NaN handling done in _encode_pitch)
        mask = torch.ones_like(values, dtype=torch.float32)

        batch = {
            "values": values.to(self.device),
            "mask": mask.to(self.device),
            "padding": padding.to(self.device),
            "batter_hashes": batter_hashes.to(self.device),
            "pitcher_hashes": pitcher_hashes.to(self.device),
            "pitch_type_hashes": ptype_hashes.to(self.device),
        }

        return batch

    def _run_inference(self, state: GameInferenceState) -> dict[str, float]:
        """Execute model forward pass and derive market prices.

        Returns dict mapping market name → probability.
        """
        batch = self._build_batch(state)

        with torch.no_grad():
            output = self.model(batch)

        # Convert logits to probabilities
        home_win_prob = torch.sigmoid(output["home_win_logit"]).item()
        yrfi_prob = torch.sigmoid(output["yrfi_logit"]).item()

        # Regression parameters (Gaussian from the LiveGameModel)
        total_mu = output["total_runs_mu"].item()
        total_sigma = output["total_runs_sigma"].item()
        diff_mu = output["home_run_diff_mu"].item()
        diff_sigma = output["home_run_diff_sigma"].item()

        # Derive markets from distribution parameters
        # For now, return raw classification probabilities + regression params.
        # Full market derivation (totals, spreads, team totals) requires integration
        # over the Gaussian, which is handled by the distributions.py module.
        # TODO: integrate distributions.py for full market family coverage

        market_prices = {
            "home_win": home_win_prob,
            "away_win": 1.0 - home_win_prob,
            "yrfi": yrfi_prob,
            "nrfi": 1.0 - yrfi_prob,
            # Regression params (can be used to derive totals/spreads downstream)
            "total_runs_mu": total_mu,
            "total_runs_sigma": total_sigma,
            "home_run_diff_mu": diff_mu,
            "home_run_diff_sigma": diff_sigma,
        }

        return market_prices

    def _hash_player_id(self, player_id: int) -> int:
        """blake2b hash bucket for player identity (512 buckets).
        Consistent with pregame module's approach."""
        if player_id == 0:
            return 0
        digest = hashlib.blake2b(str(player_id).encode(), digest_size=8).hexdigest()
        return int(digest, 16) % 511 + 1

    def _hash_string(self, s: str) -> int:
        """Hash a string (e.g., pitch type) into a bucket."""
        if not s:
            return 0
        digest = hashlib.blake2b(s.encode(), digest_size=8).hexdigest()
        return int(digest, 16) % 255 + 1

    def _update_hierarchy(self, state: GameInferenceState, event: PitchEvent) -> None:
        """Track hierarchy indices as the game progresses.

        WHY manual tracking: The model needs to know the hierarchical position
        of each pitch for its positional encoding (future HAN upgrade). GUMBO
        provides inning/atBatIndex but not the within-inning AB count we need.
        """
        # Detect inning change
        if event.inning > state.current_inning:
            state.current_inning = event.inning
            state.current_ab_in_inning = 0
            state.current_pitch_in_ab = 0

        # Detect AB change (pitch_number resets to 1)
        if event.pitch_number == 1:
            state.ab_index_global += 1
            state.current_ab_in_inning += 1
            state.current_pitch_in_ab = 1
        else:
            state.current_pitch_in_ab += 1


class TradingBridge:
    """Bridges the async inference engine with the sync trading runner.

    The trading runner (pregame/trading/runner.py) is synchronous.
    The ingestion layer is async (asyncio).
    This bridge runs the async event loop in a background thread and
    exposes a synchronous API to the runner.

    WHY this exists: The runner's main loop cannot be made async without
    rewriting the entire portfolio/order management logic. This bridge allows
    the live inference engine to operate asynchronously (for efficient I/O with
    the MLB API) while presenting a thread-safe sync interface to the runner.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.device = device
        self.engine: Optional[LiveInferenceEngine] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the background event loop + inference engine."""
        if self._running:
            log.warning("TradingBridge already started")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="TradingBridge")
        self._thread.start()

        # Wait for engine initialization
        timeout = 5.0
        start = time.time()
        while self.engine is None:
            time.sleep(0.1)
            if time.time() - start > timeout:
                raise RuntimeError("TradingBridge failed to initialize within timeout")

        log.info("TradingBridge started")

    def _run_loop(self) -> None:
        """Background thread: run asyncio event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # Initialize engine in the background thread
        self.engine = LiveInferenceEngine(
            model_path=self.model_path,
            device=self.device,
        )

        # Keep loop alive
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    def register_game(self, game_pk: int, pregame_prior: PregamePrior) -> None:
        """Thread-safe: register a new game for live tracking."""
        if self.engine is None:
            log.warning("Cannot register game: engine not initialized")
            return

        # Schedule in the event loop thread
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._async_register(game_pk, pregame_prior),
                self._loop
            )
        else:
            # Fallback: direct call if loop not running (shouldn't happen)
            self.engine.register_game(game_pk, pregame_prior)

    async def _async_register(self, game_pk: int, pregame_prior: PregamePrior) -> None:
        """Async wrapper for register_game."""
        if self.engine:
            self.engine.register_game(game_pk, pregame_prior)

    def get_live_prices(self, game_pk: int) -> Optional[dict[str, float]]:
        """Thread-safe: get latest market prices for a game."""
        if self.engine is None:
            return None
        return self.engine.get_latest_prices(game_pk)

    def get_active_games(self) -> list[int]:
        """Thread-safe: list of games currently being tracked."""
        if self.engine is None:
            return []
        with self.engine._lock:
            return list(self.engine._games.keys())

    def stop(self) -> None:
        """Graceful shutdown."""
        if not self._running:
            return

        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2.0)

        log.info("TradingBridge stopped")
