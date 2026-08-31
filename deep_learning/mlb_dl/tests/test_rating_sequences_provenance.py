"""The rating_sequences sidecar must record the cut its means/stds were fit on.

Why this is worth a test rather than a comment. `build_rating_sequences` standardizes every
sequence with means/stds fit on `game_date < train_end`, and when `train_end is None` it fits on
the ENTIRE population -- val and test included. That is a real leak, and until 2026-08-31 the
saved metadata recorded no cut at all, so the two cases were indistinguishable in the artifact.

Three cutoffs coexist in this stack and only one of them is the truth:
    rating_sequences CLI default      2024-04-01
    build_weather_asof TRAIN_END_DATE 2024-01-01
    temporal_split_dates(game_targets) 80% quantile over distinct dates -- MOVES with the
                                       population (2024-05-14 void cache, 2024-08-03 corrected)
Measured 2026-08-31: both hardcoded dates are earlier than the real 2024-08-03, so they are
conservative (dropping 1,597 and 2,119 train games from their fits) rather than leaky. Recording
the cut is what makes that checkable at all.

Usage:
    conda run -n pred python -m pytest deep_learning/mlb_dl/tests/test_rating_sequences_provenance.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "deep_learning")

from mlb_dl.rating_sequences import (
    RATING_SEQ_STEPS,
    build_rating_sequences,
    save_rating_sequences,
)


def _meta(tmp_path, **kwargs):
    out = tmp_path / "rating_sequences"
    save_rating_sequences(
        sequences={(1, "home"): np.zeros((RATING_SEQ_STEPS, 1), dtype=np.float32)},
        rating_cols=["elo_diff"],
        means={"elo_diff": 0.0},
        stds={"elo_diff": 1.0},
        output_path=str(out),
        **kwargs,
    )
    return json.loads(out.with_suffix(".json").read_text())


def test_train_end_is_recorded(tmp_path):
    assert _meta(tmp_path, train_end="2024-08-03")["train_end"] == "2024-08-03"


def test_absent_train_end_is_recorded_as_null_not_omitted(tmp_path):
    """None means "fit on everything", i.e. leaked. It must be visible, not merely missing."""
    meta = _meta(tmp_path)
    assert "train_end" in meta
    assert meta["train_end"] is None


@pytest.mark.parametrize("train_end,expect_leak", [("2024-01-01", False), (None, True)])
def test_norm_stats_respect_the_cut(train_end, expect_leak):
    """Adversarial: post-cut games carry an extreme value the fit must not see.

    Constructed so leakage is unambiguous rather than a small numeric drift -- pre-cut ratings
    are 1.0 and post-cut 1000.0.

    Assert on STD, not mean. `elo_diff` is a diff column, so the perspective map negates it for
    the away view, and every game contributes exactly one home (+v) and one away (-v) row --
    which drives the mean to 0.0 for ANY symmetric fixture, leak or not. That also happens to
    equal the empty-fit fallback value, so a mean-based assertion here would be doubly blind.
    Std is what actually separates the two populations: ~1 with only pre-cut rows, ~hundreds
    once 1000.0 rows enter.
    """
    rows = []
    for i, (d, val) in enumerate(
        [("2023-05-01", 1.0)] * 6 + [("2024-06-01", 1000.0)] * 6
    ):
        rows.append({
            "game_pk": 1000 + i,
            "game_date": d,
            "home_team_id": 1,
            "away_team_id": 2,
            "elo_diff": val,
        })
    gf = pd.DataFrame(rows)

    _, cols, _, stds = build_rating_sequences(gf, k_steps=2, train_end=train_end)
    assert cols == ["elo_diff"]

    if expect_leak:
        assert stds["elo_diff"] > 100.0, (
            "train_end=None is expected to fit on all games; if this ever stops being true the "
            f"module docstring is wrong (std={stds['elo_diff']})"
        )
    else:
        assert stds["elo_diff"] == pytest.approx(1.0), (
            f"post-cut value leaked into the fit: std={stds['elo_diff']}"
        )
