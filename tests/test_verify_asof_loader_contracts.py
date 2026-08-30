"""Each check must fire on the corruption it claims to catch.

A verifier that passes everything is worse than no verifier: it converts an unchecked
assumption into a documented guarantee that isn't one. Every test here starts from a
frame shaped exactly like the real artifact, breaks one thing, and asserts the specific
check catches it -- and asserts the clean frame stays clean, so the checks can't be
passing by being unconditionally noisy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_curation" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deep_learning"))

import verify_asof_loader_contracts as v  # noqa: E402
from mlb_dl.weather_asof import DECISION_HOURS, TARGET_HOURS  # noqa: E402


def asof_frame(pks=(1, 2)) -> pd.DataFrame:
    rows = [(pk, d, h) for pk in pks for d in DECISION_HOURS for h in TARGET_HOURS]
    return pd.DataFrame(rows, columns=["game_pk", "decision_hour", "target_hour"])


def offset_frame(counts={1: 5, 2: 4}) -> pd.DataFrame:
    rows = []
    for pk, n in counts.items():
        for i in range(n):
            rows.append((pk, i, min(i // 2, DECISION_HOURS[-1])))
    return pd.DataFrame(rows, columns=["game_pk", "sequence_index", "wx_hour_offset"])


# --- the clean baseline must be silent ---------------------------------------
def test_clean_asof_frame_passes():
    assert v.check_asof_blocks(asof_frame()) == []


def test_clean_offset_frame_passes():
    assert v.check_offsets(offset_frame()) == []


def test_clean_cross_passes():
    assert v.check_cross(asof_frame((1, 2)), offset_frame({1: 3, 2: 3})) == []


def test_rows_per_game_matches_the_loader_constant():
    """If this drifts from train_unified's per_game the audit is checking nothing."""
    assert v.ROWS_PER_GAME == len(DECISION_HOURS) * len(TARGET_HOURS) == 49


# --- weather_asof block contract ---------------------------------------------
def test_short_block_is_caught():
    df = asof_frame().iloc[:-1]  # game 2 has 48 rows -> reshape would raise
    fails = v.check_asof_blocks(df)
    assert any("multiple of 49" in f for f in fails)


def test_interleaved_games_are_caught():
    """The failure the block loader cannot survive: it keys off the first row of each
    49-row window, so interleaving assigns game 1's weather to game 2."""
    df = asof_frame()
    df.loc[0, "game_pk"] = 2      # game 2 now appears in two separate runs
    df.loc[len(df) - 1, "game_pk"] = 1
    fails = v.check_asof_blocks(df)
    assert any("more than one run" in f or "contiguous" in f for f in fails)


def test_interleaving_is_caught_even_when_every_run_is_exactly_49_rows():
    """Isolates the duplicate-run check from the row-count check.

    Mutation testing showed test_interleaved_games_are_caught was passing only via the
    "contiguous rows" branch, leaving the run-duplication branch unverified. Here game 1
    appears in two separate 49-row runs, so every run length is legal and the total is a
    clean multiple of 49 -- only run duplication reveals it. The loader would key the
    third block to game 1 again and silently overwrite its first block.
    """
    df = pd.concat([asof_frame((1,)), asof_frame((2,)), asof_frame((1,))],
                   ignore_index=True)
    assert len(df) % v.ROWS_PER_GAME == 0
    fails = v.check_asof_blocks(df)
    assert any("more than one run" in f for f in fails), fails


def test_wrong_row_count_per_game_is_caught():
    extra = pd.DataFrame([(1, 0, 0)], columns=["game_pk", "decision_hour", "target_hour"])
    df = pd.concat([asof_frame((1,)), extra, asof_frame((2,))], ignore_index=True)
    fails = v.check_asof_blocks(df)
    assert any("contiguous rows" in f for f in fails)


def test_transposed_block_order_is_caught():
    """reshape(7,7,C) encodes meaning positionally, so target-major order silently
    transposes decision hour and target hour."""
    rows = [(pk, d, h) for pk in (1, 2) for h in TARGET_HOURS for d in DECISION_HOURS]
    df = pd.DataFrame(rows, columns=["game_pk", "decision_hour", "target_hour"])
    fails = v.check_asof_blocks(df)
    assert any("decision-major" in f for f in fails)


def test_empty_asof_is_caught():
    assert v.check_asof_blocks(asof_frame().iloc[:0]) != []


# --- wx_hour_offset positional contract --------------------------------------
def test_sequence_index_gap_is_caught():
    df = offset_frame({1: 6})
    df.loc[3, "sequence_index"] = 99   # gap: positions 3.. now misaligned
    fails = v.check_offsets(df)
    assert any("gaps/duplicates" in f for f in fails)


def test_duplicate_sequence_index_is_caught():
    df = offset_frame({1: 6})
    df.loc[3, "sequence_index"] = 2
    fails = v.check_offsets(df)
    assert any("duplicate" in f or "gaps/duplicates" in f for f in fails)


def test_game_not_starting_at_zero_is_caught():
    df = offset_frame({1: 4})
    df["sequence_index"] = df["sequence_index"] + 1
    fails = v.check_offsets(df)
    assert any("sequence_index 0" in f for f in fails)


def test_out_of_range_offset_is_caught():
    df = offset_frame({1: 4})
    df.loc[2, "wx_hour_offset"] = DECISION_HOURS[-1] + 3
    fails = v.check_offsets(df)
    assert any("outside" in f for f in fails)


def test_negative_offset_is_caught():
    df = offset_frame({1: 4})
    df.loc[1, "wx_hour_offset"] = -2
    assert any("outside" in f for f in v.check_offsets(df))


def test_isolated_backwards_step_is_a_warning_not_a_failure():
    """SPEC CHANGE 2026-08-30, driven by measurement, not by a failing test.

    This originally asserted that ANY backwards step fails. Running it against the real
    artifacts showed the assertion is unsatisfiable for reasons outside this pipeline:
    raw pitch_start_time is non-monotonic (game 413650 reports 23:02:07 then 22:55:46
    for two pitches of the same play, every pitch timed). 205 of 32,193 games carry a
    backwards step; only 252 pitches in 10.3M (2.4e-05) land in the leak direction.

    The builder sorts by the same keys the dataset does, so nothing is misaligned. A
    blanket failure would therefore have been permanently red -- and a permanently red
    check is one nobody reads, which would hide the reversal this exists to catch.
    """
    df = offset_frame({1: 20})
    df.loc[10, "wx_hour_offset"] = 0     # one dip inside an otherwise ordered game
    assert v.check_offsets(df) == []


def test_reversed_game_still_fails():
    """The failure the check exists for: a sort or key mismatch reverses the sequence, so
    nearly every step runs backwards. This must never be excused as jitter."""
    df = offset_frame({1: 20})
    df["wx_hour_offset"] = list(range(6, -1, -1)) + [0] * 13
    fails = v.check_offsets(df)
    assert any("reversal/interleaving" in f for f in fails)


def test_interleaved_two_game_pattern_fails():
    """Two games' pitches merged under one key alternate between time regimes, driving
    about half the steps negative."""
    df = offset_frame({1: 20})
    df["wx_hour_offset"] = [0, 5] * 10
    assert any("reversal/interleaving" in f for f in v.check_offsets(df))


def test_game_boundaries_are_not_counted_as_backwards_steps():
    """Regression: the season-rate test first diffed the offset column globally, so every
    game boundary (offset resets to 0) scored as a backwards step. That made the measured
    rate ~1/pitches-per-game for any input and failed a perfectly clean 12,000-pitch
    frame."""
    df = offset_frame({pk: 30 for pk in range(1, 401)})
    assert (df.sort_values(["game_pk", "sequence_index"])["wx_hour_offset"]
            .diff() < 0).sum() > 0, "fixture must contain boundary resets to be meaningful"
    assert v.check_offsets(df) == []


def test_thresholds_bracket_the_measured_jitter_and_a_real_reversal():
    """Pins both constants against the data that justified them."""
    # worst observed real game: 16 backwards steps in 426 pitches
    assert 16 / 426 < v.MAX_GAME_DECREASING_FRACTION < 0.5
    # measured season floor ~5.1e-05, with headroom before a regression trips
    assert 5.1e-5 < v.MAX_SEASON_DECREASING_RATE < 1e-2


def test_systemic_backwards_rate_fails_even_when_no_single_game_is_reversed():
    """A regression that sprinkles one bad step into most games would slip under the
    per-game fraction test, so the season-wide rate is checked independently."""
    counts = {pk: 30 for pk in range(1, 401)}   # 12,000 pitches: past MIN_PITCHES_FOR_RATE
    df = offset_frame(counts)
    # one dip per game: ~3% of each game's steps, far under the per-game threshold
    for pk in counts:
        idx = df.index[(df.game_pk == pk) & (df.sequence_index == 15)][0]
        df.loc[idx, "wx_hour_offset"] = 0
    per_game = v.check_offsets(df)
    assert not any("reversal/interleaving" in f for f in per_game), (
        "per-game test should not fire; this fixture exists to exercise the rate test"
    )
    assert any("systemic regression" in f for f in per_game)


def test_nan_offset_is_caught():
    df = offset_frame({1: 4}).astype({"wx_hour_offset": "float64"})
    df.loc[2, "wx_hour_offset"] = float("nan")
    assert any("NaN" in f for f in v.check_offsets(df))


def test_empty_offsets_is_caught():
    assert v.check_offsets(offset_frame().iloc[:0]) != []


# --- cross-artifact ----------------------------------------------------------
def test_game_with_weather_but_no_offsets_is_caught():
    """This one degrades silently rather than crashing: the dataset falls back to the
    pregame decision row for every pitch, so the game trains with no live weather."""
    fails = v.check_cross(asof_frame((1, 2)), offset_frame({1: 3}))
    assert any("no wx_hour_offset" in f for f in fails)


def test_extra_offsets_without_weather_is_not_an_error():
    """The reverse is harmless: the dataset filters offsets to games it has targets for."""
    assert v.check_cross(asof_frame((1,)), offset_frame({1: 3, 2: 3})) == []
