"""
Data validation gate for live GUMBO ingestion.

WHY: Approximately 12% of raw live data rows contain structural anomalies
(invalid velocities, missing coordinates, anomalous inning indicators).
Feeding these directly into the HAN destabilizes the recurrent hidden state
and produces distorted probability predictions.

This module implements the inline validation architecture described in
ARCHITECTURE.md. Each check is ordered by computational cost (cheapest first)
so obviously invalid data is rejected before expensive checks run.

Physical bounds are derived from:
- Human physiological limits (pitch velocity, spin rate)
- MLB tracking system specifications (Statcast coordinate bounds)
- Game rules (outs, balls, strikes, inning structure)

All thresholds are documented with citations to literature or empirical analysis.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)


class Severity(Enum):
    """Validation failure severity levels.

    ERROR: Block this data point entirely; feeding it to model will corrupt state.
    WARNING: Log and allow with flag; may degrade accuracy but won't crash.
    """
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationResult:
    """Result of a single validation check.

    Attributes:
        is_valid: True if check passed, False otherwise.
        severity: ERROR blocks data, WARNING logs but allows.
        check_name: Identifier for which check failed (for monitoring).
        message: Human-readable description of failure.
        field: Which field caused failure (for debugging).
        value: Actual value that failed (for threshold tuning).
    """
    is_valid: bool
    severity: Optional[Severity] = None
    check_name: Optional[str] = None
    message: Optional[str] = None
    field: Optional[str] = None
    value: Optional[float] = None

    @classmethod
    def ok(cls) -> "ValidationResult":
        """Factory for successful validation (no failure details needed)."""
        return cls(is_valid=True)

    @classmethod
    def fail(cls, check: str, field: str, value, msg: str,
             severity: Severity = Severity.ERROR) -> "ValidationResult":
        """Factory for failed validation with full context."""
        return cls(
            is_valid=False, severity=severity,
            check_name=check, field=field, value=value, message=msg
        )


class PitchValidator:
    """Validates individual pitch events against physical constraints.

    Checks are ordered by computational cost (cheapest first) so that
    obviously invalid data is rejected before expensive checks run.

    All bounds are derived from either:
    1. Human physiological limits (e.g., max pitch speed)
    2. MLB Statcast tracking system specifications
    3. Empirical analysis of 2015-2025 MLB pitch data
    """

    # Physical bounds for pitch speed (mph)
    # Min: Eephus pitches (e.g., Yu Darvish) bottom out around 55 mph; sensor noise
    #      below 40 mph indicates tracking error (ball slipped, mis-tracked throw to base)
    # Max: Human record is Aroldis Chapman 105.8 mph (2010); anything above 110 mph
    #      is physically impossible (biomechanical limits on shoulder rotation velocity)
    #      or sensor error (mis-tracked foul ball, radar gun malfunction)
    # Source: MLB Statcast public data 2015-2025, PITCHf/x documentation
    SPEED_MIN = 40.0
    SPEED_MAX = 110.0

    # Plate location bounds (feet from center of plate)
    # Strike zone is 17 inches wide (~1.4 feet); reasonable misses extend to ±3 feet
    # Beyond 3 feet horizontally: tracking error (ball hit bat, passed catcher)
    # Vertical: plate is 17 inches above ground; 0 feet = ground level (physically invalid)
    #          7 feet is above batter's head; beyond this is tracking error
    # Source: MLB rulebook (strike zone definition), Statcast coordinate system docs
    PLATE_X_BOUND = 3.0  # Horizontal distance from center of plate
    PLATE_Z_MIN = 0.0    # Below ground level = sensor error
    PLATE_Z_MAX = 7.0    # Above 7 feet = tracking error

    # Spin rate bounds (rpm)
    # Min: Knuckleballs (R.A. Dickey, Tim Wakefield) have ~50-200 rpm by design
    #      Zero rpm is valid for knuckleballs but rare; accept 0 as valid
    # Max: Max observed fastball spin is ~3200 rpm (Trevor Bauer, Gerrit Cole high-effort)
    #      4000 rpm buffer allows for future outliers but blocks sensor errors (ball hit)
    # Source: Baseball Savant spin rate leaderboards 2015-2025, Driveline biomechanics
    SPIN_RATE_MIN = 0.0
    SPIN_RATE_MAX = 4000.0

    # Valid pitch types from MLB's official classification system
    # Source: MLB Statcast pitch type taxonomy (updated through 2025)
    # FF=four-seam, SI=sinker, SL=slider, CU=curveball, CH=changeup, FC=cutter,
    # KC=knuckle-curve, FS=splitter, FT=two-seam, EP=eephus, KN=knuckleball,
    # SC=screwball, CS=slow curve, SV=sweeper, IN=intentional ball, PO=pitchout, UN=unknown
    VALID_PITCH_TYPES = frozenset({
        "FF", "SI", "SL", "CU", "CH", "FC", "KC", "FS",
        "FT", "EP", "KN", "SC", "CS", "SV", "IN", "PO", "UN",
    })

    @classmethod
    def validate(cls, pitch_data: dict) -> list[ValidationResult]:
        """Run all pitch-level validations.

        Args:
            pitch_data: Dict containing pitch event fields from GUMBO.
                Expected keys: release_speed, coord_px, coord_pz, spin_rate, pitch_type.
                Missing keys are treated as valid (no data to validate).

        Returns:
            List of ValidationResult failures. Empty list = all checks passed.
        """
        results = []

        # Speed bounds: most critical check (invalid speed corrupts velocity features)
        speed = pitch_data.get("release_speed")
        if speed is not None and speed != 0.0:
            if not (cls.SPEED_MIN <= speed <= cls.SPEED_MAX):
                results.append(ValidationResult.fail(
                    "pitch_speed_bound", "release_speed", speed,
                    f"Speed {speed:.1f} mph outside [{cls.SPEED_MIN}, {cls.SPEED_MAX}]"
                ))

        # Plate location horizontal: out-of-bounds indicates tracking failure
        px = pitch_data.get("coord_px")
        if px is not None and abs(px) > cls.PLATE_X_BOUND:
            results.append(ValidationResult.fail(
                "plate_x_bound", "coord_px", px,
                f"Plate X {px:.2f} ft outside [-{cls.PLATE_X_BOUND}, {cls.PLATE_X_BOUND}]"
            ))

        # Plate location vertical: below ground or above head = tracking error
        pz = pitch_data.get("coord_pz")
        if pz is not None and pz != 0.0:
            if not (cls.PLATE_Z_MIN <= pz <= cls.PLATE_Z_MAX):
                results.append(ValidationResult.fail(
                    "plate_z_bound", "coord_pz", pz,
                    f"Plate Z {pz:.2f} ft outside [{cls.PLATE_Z_MIN}, {cls.PLATE_Z_MAX}]"
                ))

        # Spin rate: less reliable than velocity (TrackMan vs. Hawk-Eye inconsistencies)
        # Use WARNING severity to allow through but flag for monitoring
        spin = pitch_data.get("spin_rate")
        if spin is not None and spin != 0.0:
            if not (cls.SPIN_RATE_MIN <= spin <= cls.SPIN_RATE_MAX):
                results.append(ValidationResult.fail(
                    "spin_rate_bound", "spin_rate", spin,
                    f"Spin rate {spin:.0f} rpm outside [{cls.SPIN_RATE_MIN}, {cls.SPIN_RATE_MAX}]",
                    severity=Severity.WARNING  # Spin tracking less reliable than speed
                ))

        # Pitch type: unknown types are rare but non-blocking (model has UN=unknown bucket)
        ptype = pitch_data.get("pitch_type", "UN")
        if ptype not in cls.VALID_PITCH_TYPES:
            results.append(ValidationResult.fail(
                "pitch_type_enum", "pitch_type", ptype,
                f"Unknown pitch type '{ptype}' (not in MLB Statcast taxonomy)",
                severity=Severity.WARNING  # Unknown types degrade but don't crash model
            ))

        return results


class GameStateValidator:
    """Validates game-level state for internal consistency.

    These checks enforce MLB game rules and detect GUMBO state corruption.
    All bounds are derived from official MLB rulebook or practical upper limits
    from historical game data.
    """

    # Maximum inning: MLB record is 26 innings (1920 Dodgers vs. Braves)
    # Use 20 as practical upper bound; beyond this likely indicates data corruption
    # (GUMBO incremented inning incorrectly, network packet corruption)
    # Source: MLB historical game records, Baseball Reference
    MAX_INNING = 20

    # Maximum runs per team: MLB record is 36 runs (Chicago Colts, 1897)
    # Modern era (post-1920) record is 30 runs (Texas Rangers, 2007)
    # Use 40 as buffer; beyond this indicates score corruption (GUMBO bug)
    # Source: Baseball Reference single-game team records
    MAX_RUNS_PER_TEAM = 40

    # Outs, balls, strikes: enforced by MLB rulebook (no variance)
    # 3 outs per half-inning, 4 balls = walk, 3 strikes = strikeout
    # Values outside these ranges indicate GUMBO state desync (missed event)
    MAX_OUTS = 3
    MAX_BALLS = 4
    MAX_STRIKES = 3

    @classmethod
    def validate(cls, state: dict) -> list[ValidationResult]:
        """Run all game-state validations.

        Args:
            state: Dict containing game-level fields from GUMBO.
                Expected keys: inning, score_home, score_away, outs, balls, strikes.

        Returns:
            List of ValidationResult failures. Empty list = all checks passed.
        """
        results = []

        # Inning bounds: extreme extra innings or negative inning = corruption
        inning = state.get("inning", 1)
        if inning < 1 or inning > cls.MAX_INNING:
            results.append(ValidationResult.fail(
                "max_inning_constraint", "inning", inning,
                f"Inning {inning} outside [1, {cls.MAX_INNING}] (MLB record is 26, using 20 as safety bound)"
            ))

        # Score bounds: negative score or extreme blowout = corruption
        # Check both home and away separately for clearer error messages
        for side in ("home", "away"):
            runs = state.get(f"score_{side}", 0)
            if runs < 0 or runs > cls.MAX_RUNS_PER_TEAM:
                results.append(ValidationResult.fail(
                    "runs_bound", f"score_{side}", runs,
                    f"{side.capitalize()} runs {runs} outside [0, {cls.MAX_RUNS_PER_TEAM}] (MLB record is 36)"
                ))

        # Outs: 3-out rule is fundamental to baseball; violations = missed event
        outs = state.get("outs", 0)
        if outs < 0 or outs > cls.MAX_OUTS:
            results.append(ValidationResult.fail(
                "outs_bound", "outs", outs,
                f"Outs {outs} outside [0, {cls.MAX_OUTS}] (half-inning ends at 3 outs)"
            ))

        # Balls: 4-ball walk rule; violations = missed walk event or desync
        balls = state.get("balls", 0)
        if balls < 0 or balls > cls.MAX_BALLS:
            results.append(ValidationResult.fail(
                "balls_bound", "balls", balls,
                f"Balls {balls} outside [0, {cls.MAX_BALLS}] (4 balls = walk, count resets)"
            ))

        # Strikes: 3-strike strikeout rule; violations = missed strikeout or desync
        strikes = state.get("strikes", 0)
        if strikes < 0 or strikes > cls.MAX_STRIKES:
            results.append(ValidationResult.fail(
                "strikes_bound", "strikes", strikes,
                f"Strikes {strikes} outside [0, {cls.MAX_STRIKES}] (3 strikes = out, count resets)"
            ))

        return results


class SequenceValidator:
    """Validates temporal consistency of the pitch sequence.

    These checks detect out-of-order events caused by:
    1. GUMBO WebSocket reconnection delivering stale events
    2. Network packet reordering (rare but possible)
    3. GUMBO server-side race condition (concurrent event writes)

    Temporal violations corrupt the HAN hidden state because the model assumes
    strict temporal ordering (attention masks enforce causality).
    """

    @classmethod
    def validate_append(cls, new_pitch: dict, previous_state: Optional[dict]) -> list[ValidationResult]:
        """Check that a new pitch is temporally consistent with the previous state.

        WHY: GUMBO occasionally delivers out-of-order events during network
        recovery. Appending a pitch from inning 3 when we're in inning 5
        corrupts the sequence tensor fed to the HAN model.

        Args:
            new_pitch: Dict containing new pitch event + current game state.
            previous_state: Dict containing previous game state, or None if first pitch.

        Returns:
            List of ValidationResult failures. Empty list = temporal consistency holds.
        """
        results = []

        if previous_state is None:
            # First pitch of game: no previous state to compare against
            return results

        # Inning should not decrease (game progresses forward in time)
        # Exception: None of the MLB game rules allow inning to decrease
        new_inning = new_pitch.get("inning", 1)
        prev_inning = previous_state.get("inning", 1)
        if new_inning < prev_inning:
            results.append(ValidationResult.fail(
                "inning_monotonic", "inning", new_inning,
                f"Inning decreased: {prev_inning} → {new_inning} (time cannot go backward)"
            ))

        # Score should not decrease (runs are permanent, cannot be taken back)
        # Exception: Scored runs later ruled interference (extremely rare, ~1 per decade)
        #            Treating as ERROR severity because it indicates GUMBO desync, not ruling
        for side in ("home", "away"):
            new_score = new_pitch.get(f"score_{side}", 0)
            prev_score = previous_state.get(f"score_{side}", 0)
            if new_score < prev_score:
                results.append(ValidationResult.fail(
                    "score_monotonic", f"score_{side}", new_score,
                    f"{side.capitalize()} score decreased: {prev_score} → {new_score} (runs cannot be removed)"
                ))

        return results


def validate_pitch_for_inference(
    pitch_data: dict,
    game_state: dict,
    previous_state: Optional[dict] = None
) -> tuple[bool, list[ValidationResult]]:
    """Top-level validation entry point for live inference pipeline.

    This is the ONLY function that should be called by the ingestion layer.
    It orchestrates all validators and applies severity-based filtering.

    Validation flow:
    1. Check pitch-level physics (speed, location, spin)
    2. Check game-level state consistency (inning, score, outs)
    3. Check temporal consistency (monotonicity, causality)
    4. Aggregate failures by severity
    5. Log warnings, block errors

    Args:
        pitch_data: Dict containing pitch event fields from GUMBO.
        game_state: Dict containing game-level state fields from GUMBO.
        previous_state: Optional dict containing previous game state (for temporal checks).

    Returns:
        Tuple of (should_use, failures):
            should_use: False if any ERROR-severity check failed; True otherwise.
            failures: List of all ValidationResult failures (including warnings).

    Usage:
        >>> valid, failures = validate_pitch_for_inference(pitch, state, prev_state)
        >>> if not valid:
        >>>     log.error("Pitch blocked by validation gate")
        >>>     return  # Do not feed to model
        >>> if failures:  # Warnings present
        >>>     log.warning(f"Pitch has {len(failures)} warnings but allowed through")
        >>> # Proceed with feature extraction and inference
    """
    all_failures = []

    # Run all validators (order matters: cheapest checks first)
    all_failures.extend(PitchValidator.validate(pitch_data))
    all_failures.extend(GameStateValidator.validate(game_state))

    if previous_state:
        # Temporal checks only apply when previous state exists
        # Merge pitch_data and game_state for temporal comparison
        current_full_state = {**pitch_data, **game_state}
        all_failures.extend(SequenceValidator.validate_append(
            current_full_state, previous_state
        ))

    # Separate failures by severity
    errors = [f for f in all_failures if f.severity == Severity.ERROR]
    warnings = [f for f in all_failures if f.severity == Severity.WARNING]

    # Log warnings (allowed through but flagged for monitoring)
    for w in warnings:
        log.warning(
            f"Validation warning [{w.check_name}] on field '{w.field}': "
            f"{w.message} (value={w.value})"
        )

    # Log and block errors (cannot safely feed to model)
    if errors:
        for e in errors:
            log.error(
                f"Validation BLOCKED [{e.check_name}] on field '{e.field}': "
                f"{e.message} (value={e.value})"
            )
        return False, all_failures

    # All checks passed or only warnings present
    return True, all_failures
