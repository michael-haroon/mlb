"""
Example script demonstrating the LiveInferenceEngine in isolation.

This shows how to:
1. Initialize the engine with a trained model
2. Register a game with pregame context
3. Feed pitch events one by one
4. Get updated market prices after each pitch

Usage:
    conda run -n pred python -m live.mlb_dl.example_inference
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from live.mlb_dl.inference_engine import (
    LiveInferenceEngine,
    PregamePrior,
    PitchEvent,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)


def on_reprice_callback(game_pk: int, prices: dict[str, float]) -> None:
    """Called after each inference. This is where the trading runner would
    decide whether to adjust positions."""
    log.info(f"Game {game_pk} repriced:")
    log.info(f"  home_win: {prices['home_win']:.3f}  (pregame context)")
    log.info(f"  yrfi:     {prices['yrfi']:.3f}")
    log.info(f"  total_runs_mu: {prices['total_runs_mu']:.2f} ± {prices['total_runs_sigma']:.2f}")


def simulate_game_sequence():
    """Simulate a realistic pitch sequence for one game."""

    # Step 1: Load the trained model
    # For this example, we'll use a dummy checkpoint path. In production, this
    # would point to the actual trained model in live/mlb_dl/checkpoints/.
    model_path = Path(__file__).parent / "checkpoints" / "best_model.pt"

    if not model_path.exists():
        log.error(f"Model not found at {model_path}")
        log.error("Train a model first using: conda run -n pred python -m live.mlb_dl.train")
        return

    engine = LiveInferenceEngine(
        model_path=str(model_path),
        device="cpu",
        on_reprice=on_reprice_callback,
    )

    # Step 2: Register a game with pregame context
    # These values would come from the pregame ensemble's predictions.
    pregame_prior = PregamePrior(
        game_pk=717161,
        home_win_prob=0.55,  # Home team favored (e.g., Dodgers -150)
        yrfi_prob=0.48,
        extra_innings_prob=0.09,
        first_5_home_win_prob=0.54,
        mu_home_runs=4.8,
        mu_away_runs=4.2,
        mu_total_runs=9.0,
        mu_home_run_diff=0.6,
        scale_home_runs=2.4,
        scale_away_runs=2.3,
        scale_total_runs=3.2,
        scale_home_run_diff=2.8,
        elo_diff=40.0,  # Home team +40 Elo
        srs_diff=0.8,   # Home team +0.8 runs/game SRS
        sp_era_diff=-0.5,  # Home SP has -0.5 lower ERA
        park_factor=1.05,  # Slightly hitter-friendly park
        ensemble_std_home_win=0.04,
        ensemble_std_total=0.9,
        confidence_tier="HIGH",  # Strong pregame consensus
    )

    engine.register_game(717161, pregame_prior)
    log.info("Game 717161 registered with pregame prior")

    # Step 3: Simulate a realistic pitch sequence
    # Top of the 1st inning, Mookie Betts (batter) vs. Gerrit Cole (pitcher)

    log.info("\n=== Top 1st, Mookie Betts at bat ===")

    # Pitch 1: Called strike (fastball middle-middle)
    pitch_1 = PitchEvent(
        game_pk=717161,
        inning=1,
        is_top_inning=True,
        at_bat_index=1,
        pitch_number=1,
        batter_id=605141,  # Mookie Betts
        pitcher_id=543037,  # Gerrit Cole
        pitch_type="FF",   # Four-seam fastball
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
        release_speed=98.2,
        spin_rate=2450,
        break_length=3.8,
        coord_px=0.1,  # Middle-middle
        coord_pz=2.5,
    )
    engine.on_pitch_event(717161, pitch_1)

    # Pitch 2: Ball outside (slider)
    pitch_2 = PitchEvent(
        game_pk=717161,
        inning=1,
        is_top_inning=True,
        at_bat_index=1,
        pitch_number=2,
        batter_id=605141,
        pitcher_id=543037,
        pitch_type="SL",  # Slider
        balls=1,
        strikes=1,
        outs=0,
        on_first=False,
        on_second=False,
        on_third=False,
        pitch_call="BallCalled",
        is_scoring_play=False,
        rbi_count=0,
        score_home=0,
        score_away=0,
        release_speed=85.3,
        spin_rate=2650,
        coord_px=-1.2,  # Off the plate outside
        coord_pz=2.0,
    )
    engine.on_pitch_event(717161, pitch_2)

    # Pitch 3: Swinging strike (curveball)
    pitch_3 = PitchEvent(
        game_pk=717161,
        inning=1,
        is_top_inning=True,
        at_bat_index=1,
        pitch_number=3,
        batter_id=605141,
        pitcher_id=543037,
        pitch_type="CU",  # Curveball
        balls=1,
        strikes=2,
        outs=0,
        on_first=False,
        on_second=False,
        on_third=False,
        pitch_call="StrikeSwinging",
        is_scoring_play=False,
        rbi_count=0,
        score_home=0,
        score_away=0,
        release_speed=81.5,
        spin_rate=2850,
        coord_px=0.3,
        coord_pz=1.5,
    )
    engine.on_pitch_event(717161, pitch_3)

    # Pitch 4: Foul ball (fastball)
    pitch_4 = PitchEvent(
        game_pk=717161,
        inning=1,
        is_top_inning=True,
        at_bat_index=1,
        pitch_number=4,
        batter_id=605141,
        pitcher_id=543037,
        pitch_type="FF",
        balls=1,
        strikes=2,  # Foul doesn't add a strike at 2 strikes
        outs=0,
        on_first=False,
        on_second=False,
        on_third=False,
        pitch_call="FoulBall",
        is_scoring_play=False,
        rbi_count=0,
        score_home=0,
        score_away=0,
        release_speed=97.8,
        coord_px=-0.2,
        coord_pz=2.8,
    )
    engine.on_pitch_event(717161, pitch_4)

    # Pitch 5: Single to right field
    pitch_5 = PitchEvent(
        game_pk=717161,
        inning=1,
        is_top_inning=True,
        at_bat_index=1,
        pitch_number=5,
        batter_id=605141,
        pitcher_id=543037,
        pitch_type="SL",
        balls=1,
        strikes=2,
        outs=0,
        on_first=True,  # Betts now on first
        on_second=False,
        on_third=False,
        pitch_call="InPlay",
        is_scoring_play=False,
        rbi_count=0,
        score_home=0,
        score_away=0,
        release_speed=86.1,
        coord_px=0.5,
        coord_pz=2.2,
        hit_launch_speed=95.3,
        hit_launch_angle=12.0,
    )
    engine.on_pitch_event(717161, pitch_5)

    log.info("\n=== Second batter, Freddie Freeman ===")

    # AB 2, Pitch 1: Ball
    pitch_6 = PitchEvent(
        game_pk=717161,
        inning=1,
        is_top_inning=True,
        at_bat_index=2,
        pitch_number=1,
        batter_id=518692,  # Freddie Freeman
        pitcher_id=543037,
        pitch_type="FF",
        balls=1,
        strikes=0,
        outs=0,
        on_first=True,
        on_second=False,
        on_third=False,
        pitch_call="BallCalled",
        is_scoring_play=False,
        rbi_count=0,
        score_home=0,
        score_away=0,
        release_speed=98.5,
    )
    engine.on_pitch_event(717161, pitch_6)

    # AB 2, Pitch 2: Double, Betts scores (YRFI!)
    pitch_7 = PitchEvent(
        game_pk=717161,
        inning=1,
        is_top_inning=True,
        at_bat_index=2,
        pitch_number=2,
        batter_id=518692,
        pitcher_id=543037,
        pitch_type="SL",
        balls=1,
        strikes=0,
        outs=0,
        on_first=False,
        on_second=True,  # Freeman on second
        on_third=False,
        pitch_call="InPlay",
        is_scoring_play=True,  # Betts scored!
        rbi_count=1,
        score_home=0,
        score_away=1,  # Away team (Dodgers) scored
        release_speed=84.8,
        hit_launch_speed=102.4,
        hit_launch_angle=18.0,
    )
    engine.on_pitch_event(717161, pitch_7)

    log.info("\n=== YRFI achieved! Away team scored in top 1st ===")
    log.info(f"Total pitches processed: 7")

    # Check final prices
    final_prices = engine.get_latest_prices(717161)
    if final_prices:
        log.info("\nFinal market prices:")
        log.info(f"  home_win:  {final_prices['home_win']:.3f}")
        log.info(f"  yrfi:      {final_prices['yrfi']:.3f}  (should be near 1.0 now)")
        log.info(f"  total_mu:  {final_prices['total_runs_mu']:.2f}")

    # Cleanup
    engine.unregister_game(717161)
    log.info("\nGame unregistered. Inference engine test complete.")


if __name__ == "__main__":
    log.info("Starting LiveInferenceEngine example")
    log.info("=" * 60)

    try:
        simulate_game_sequence()
    except FileNotFoundError as e:
        log.error(f"Model checkpoint not found: {e}")
        log.error("\nTo generate a model checkpoint:")
        log.error("  1. Run: conda run -n pred python -m live.mlb_dl.train")
        log.error("  2. Wait for training to complete")
        log.error("  3. Checkpoint will be saved to live/mlb_dl/checkpoints/best_model.pt")
    except Exception as e:
        log.exception(f"Error during simulation: {e}")

    log.info("=" * 60)
    log.info("Example complete")
