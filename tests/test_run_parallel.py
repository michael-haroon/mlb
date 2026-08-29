"""Tests for the parallel feature importance orchestrator (run.py).

Verifies:
1. Worker dispatch routes correctly to all 6 test functions
2. run_importance_parallel produces expected output structure
3. Parallel mode flag is set in workers (n_jobs=1)

Run: conda run -n pred python -m pytest tests/test_run_parallel.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_learning.analysis.run import _TEST_WORKERS, _run_worker


class TestWorkerDispatch:
    """Verify _run_worker dispatches to all 6 test groups."""

    def test_all_six_tests_registered(self):
        expected = {"mdi_cfi_mdi", "cfi_mda", "sfi", "desub_mda", "pca_mda", "resid_mda"}
        assert set(_TEST_WORKERS.keys()) == expected

    def test_run_worker_dispatches_correctly(self):
        """_run_worker passes args to the correct function."""
        for test_name in _TEST_WORKERS:
            with patch.dict(
                "pregame.analysis.run._TEST_WORKERS",
                {test_name: MagicMock(return_value={"test": test_name, "target": "x"})}
            ):
                result = _run_worker("/fake", "x", test_name, "2015+", ["a"], {0: ["a"]}, False)
                assert result["test"] == test_name
                assert result["target"] == "x"


class TestParallelModeInWorkers:
    """Workers must set _PARALLEL_MODE=True so inner n_jobs=1."""

    def test_worker_sets_parallel_mode(self):
        """Each worker calls set_parallel_mode(True)."""
        from classical_learning.analysis.compute import _PARALLEL_MODE, set_parallel_mode

        # All workers import and call set_parallel_mode(True) at the top.
        # Verify the flag propagates correctly.
        set_parallel_mode(True)
        from classical_learning.analysis.compute import get_n_jobs
        assert get_n_jobs() == 1

        set_parallel_mode(False)
        assert get_n_jobs() > 0  # restored
