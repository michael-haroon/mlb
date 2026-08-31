"""OOF predictions and their game_pk key array must be written as one atomic, aligned pair.

THE BUG THIS PINS: `train.py` wrote the OOF array unconditionally but guarded the key array
with `if not gpk_path.exists()`. A key array from an earlier run therefore survived a population
change while the OOF array beneath it was replaced, silently re-pointing every prediction at the
wrong game. That is the same failure class as the CalibrationBundle isotonic misalignment
(memory: calibration-oof-alignment-bug), and it is invisible downstream because both files load
fine and only their *pairing* is wrong.

Structured against a small helper rather than the full training loop on purpose: the failure is
in the persistence step, and driving it through Optuna + CV folds would make a fast unit test
into a slow integration one without testing anything more.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.strategy.train import _save_oof_with_keys  # noqa: E402


def _read(output_dir: Path, target: str, family: str, tier: str):
    oof = np.load(output_dir / f"oof_{target}_{family}_{tier}.npy")
    keys = np.load(output_dir / f"oof_game_pks_{target}_{tier}.npy")
    return oof, keys


class TestOOFKeyArrayIsAlwaysWritten:
    def test_writes_both_arrays_aligned(self, tmp_path):
        oof = np.array([0.1, 0.2, 0.3])
        gpks = pd.Series([101, 102, 103], name="game_pk")
        _save_oof_with_keys(tmp_path, "total_runs", "xgboost", "A", oof, gpks)

        got_oof, got_keys = _read(tmp_path, "total_runs", "xgboost", "A")
        assert got_oof.shape == got_keys.shape == (3,)
        np.testing.assert_array_equal(got_keys, [101, 102, 103])
        np.testing.assert_allclose(got_oof, oof)

    def test_stale_key_array_is_overwritten_not_preserved(self, tmp_path):
        """THE REGRESSION. A shorter key array from a previous population must not survive."""
        stale = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
        np.save(tmp_path / "oof_game_pks_total_runs_A.npy", stale)

        oof = np.array([0.1, 0.2, 0.3])
        gpks = pd.Series([101, 102, 103])
        _save_oof_with_keys(tmp_path, "total_runs", "xgboost", "A", oof, gpks)

        _, got_keys = _read(tmp_path, "total_runs", "xgboost", "A")
        assert got_keys.shape == (3,), (
            f"key array still has {got_keys.shape[0]} rows from the previous run; every OOF "
            "prediction is now attributed to the wrong game"
        )
        np.testing.assert_array_equal(got_keys, [101, 102, 103])

    def test_second_family_may_share_the_key_array(self, tmp_path):
        """Both families of one target share a key array, so rewriting it must be a no-op here.

        This is why the `if not exists` guard looked reasonable: within a single target the
        second family writes identical keys. The guard only bites ACROSS runs.
        """
        gpks = pd.Series([101, 102, 103])
        _save_oof_with_keys(tmp_path, "total_runs", "xgboost", "A", np.zeros(3), gpks)
        _save_oof_with_keys(tmp_path, "total_runs", "lightgbm", "A", np.ones(3), gpks)

        oof_x, keys = _read(tmp_path, "total_runs", "xgboost", "A")
        oof_l, keys2 = _read(tmp_path, "total_runs", "lightgbm", "A")
        np.testing.assert_array_equal(keys, keys2)
        np.testing.assert_allclose(oof_x, np.zeros(3))
        np.testing.assert_allclose(oof_l, np.ones(3))


class TestMisalignmentIsRefusedNotPersisted:
    def test_length_mismatch_raises(self, tmp_path):
        with pytest.raises(ValueError, match="aligned"):
            _save_oof_with_keys(
                tmp_path, "total_runs", "xgboost", "A",
                np.zeros(5), pd.Series([101, 102, 103]),
            )

    def test_length_mismatch_writes_nothing(self, tmp_path):
        """Fail before the first np.save: a half-written pair on disk is worse than no pair.

        A stale OOF array with no key array is exactly the state 10 of 11 classical targets are
        in today, and it is unrecoverable — the population that produced it is not knowable
        after the fact.
        """
        with pytest.raises(ValueError):
            _save_oof_with_keys(
                tmp_path, "total_runs", "xgboost", "A",
                np.zeros(5), pd.Series([101, 102, 103]),
            )
        assert list(tmp_path.iterdir()) == [], (
            f"wrote {[p.name for p in tmp_path.iterdir()]} despite refusing the pair"
        )


class TestKeyInputForms:
    def test_accepts_a_bare_ndarray(self, tmp_path):
        """`game_pks` is a Series at the call site, but a caller passing .values must not
        silently produce a 0-d array via a missing `.values` attribute."""
        _save_oof_with_keys(
            tmp_path, "yrfi", "xgboost", "A",
            np.zeros(2), np.array([7, 8], dtype=np.int64),
        )
        _, keys = _read(tmp_path, "yrfi", "xgboost", "A")
        np.testing.assert_array_equal(keys, [7, 8])

    def test_keys_are_not_stored_as_float(self, tmp_path):
        """game_pk is an integer id; a float key array breaks exact-match joins."""
        _save_oof_with_keys(
            tmp_path, "yrfi", "xgboost", "A",
            np.zeros(2), pd.Series([777001, 777002]),
        )
        _, keys = _read(tmp_path, "yrfi", "xgboost", "A")
        assert np.issubdtype(keys.dtype, np.integer), f"keys stored as {keys.dtype}"
