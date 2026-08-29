"""The cache build's log must capture train_unified's records, not just its own.

`build_and_save` does almost nothing itself: the per-artifact load lines, the read-time prune
report and the population-filter counts are all emitted by `mlb_dl.train_unified`. Those are
the only audit trail for which rows a cache was built from, and a cache whose population is
wrong is indistinguishable from a correct one without them (see the 1950-2024 train split).

The first fix attached handlers to `logging.getLogger(__name__.rsplit(".", 1)[0])`. That is
correct on import but wrong under `python -m mlb_dl.dataset_cache`, where `__name__` is
`"__main__"` and the expression yields `"__main__"` — a logger in a different tree from
`mlb_dl.train_unified`, so propagation never reaches the file. Observed on the real build:
dataset_cache.log contained five `INFO __main__` lines and nothing from train_unified.
"""

import logging

import pytest

from mlb_dl import dataset_cache


@pytest.fixture
def clean_logging():
    """Snapshot and restore the loggers the setup touches.

    logging state is process-global; without this a test that attaches a FileHandler to
    `mlb_dl` leaks it into every later test in the session.
    """
    names = ("mlb_dl", "__main__", "mlb_dl.dataset_cache")
    saved = {}
    for n in names:
        lg = logging.getLogger(n)
        saved[n] = (list(lg.handlers), lg.level, lg.propagate)
        lg.handlers = []
    yield
    for n, (handlers, level, propagate) in saved.items():
        lg = logging.getLogger(n)
        for h in lg.handlers:
            h.close()
        lg.handlers = handlers
        lg.level = level
        lg.propagate = propagate


def _emitted_by_train_unified(tmp_path, monkeypatch, module_name):
    """Run _setup_logging with a given __name__ and return the file's contents."""
    monkeypatch.chdir(tmp_path)  # _setup_logging writes to the relative path data/logs
    monkeypatch.setattr(dataset_cache, "__name__", module_name)

    dataset_cache._setup_logging()
    logging.getLogger("mlb_dl.train_unified").info("pruned 29184236 of 39512838 rows")

    for h in logging.getLogger("mlb_dl").handlers:
        h.flush()
    log_file = tmp_path / "data" / "logs" / "dataset_cache.log"
    return log_file.read_text() if log_file.exists() else ""


def test_train_unified_records_reach_the_log_when_run_as_main(tmp_path, monkeypatch, clean_logging):
    """The regression: `python -m mlb_dl.dataset_cache` gives __name__ == "__main__"."""
    text = _emitted_by_train_unified(tmp_path, monkeypatch, "__main__")
    assert "pruned 29184236 of 39512838 rows" in text, (
        "train_unified's records were dropped — handlers did not land on the `mlb_dl` package"
    )


def test_train_unified_records_reach_the_log_when_imported(tmp_path, monkeypatch, clean_logging):
    """The already-working case must keep working: normal import path."""
    text = _emitted_by_train_unified(tmp_path, monkeypatch, "mlb_dl.dataset_cache")
    assert "pruned 29184236 of 39512838 rows" in text


def test_setup_is_idempotent(tmp_path, monkeypatch, clean_logging):
    """build_and_save may be called more than once in a process; handlers must not stack.

    Duplicated handlers would double every line in the file, which silently breaks any
    grep-based count of how many rows a build kept.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dataset_cache, "__name__", "__main__")
    dataset_cache._setup_logging()
    n_after_first = len(logging.getLogger("mlb_dl").handlers)
    dataset_cache._setup_logging()
    assert len(logging.getLogger("mlb_dl").handlers) == n_after_first == 2


def test_both_handlers_present_at_required_levels(tmp_path, monkeypatch, clean_logging):
    """CLAUDE.md contract: file at DEBUG (granular), stdout at INFO (milestones)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dataset_cache, "__name__", "__main__")
    dataset_cache._setup_logging()

    handlers = logging.getLogger("mlb_dl").handlers
    file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
    # StreamHandler is FileHandler's base class, so identify stdout by exact type.
    stream_handlers = [h for h in handlers if type(h) is logging.StreamHandler]

    assert len(file_handlers) == 1 and file_handlers[0].level == logging.DEBUG
    assert len(stream_handlers) == 1 and stream_handlers[0].level == logging.INFO
    assert logging.getLogger("mlb_dl").level == logging.DEBUG


def test_debug_records_reach_the_file_but_not_stdout(tmp_path, monkeypatch, clean_logging, capsys):
    """The split-level contract is what makes the file useful for forensics."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dataset_cache, "__name__", "__main__")
    dataset_cache._setup_logging()

    logging.getLogger("mlb_dl.train_unified").debug("per-column dtype downcast detail")
    for h in logging.getLogger("mlb_dl").handlers:
        h.flush()

    text = (tmp_path / "data" / "logs" / "dataset_cache.log").read_text()
    assert "per-column dtype downcast detail" in text
    assert "per-column dtype downcast detail" not in capsys.readouterr().out
