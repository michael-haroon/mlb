"""prepare_all must hold at most ONE cached split in memory at a time.

Reproduces the 2026-08-31 OOM. `prepare_all` did
`train_ds, val_ds, test_ds = load_cached_datasets(cache_dir)` and only then looped over the
three, so all three splits were resident for the whole run. The kernel's own accounting from
the cgroup kill:

    Killed process 29356 (python) total-vm:68500264kB, anon-rss:27104444kB, file-rss:5372kB
    constraint=CONSTRAINT_MEMCG

25.8 GB anonymous RSS with file-rss ~= 0, i.e. real allocations rather than memmap/page
cache -- `load_dataset` calls np.load(..., allow_pickle=True) with no mmap_mode, so object
arrays plus a 481 MB player_game_stats.json inflate the 8.8 GB on-disk cache 2.93x. Splits
are 5.4 / 1.7 / 1.7 GB on disk, so holding all three needs ~26 GB while the largest alone
needs ~15.8 GB. On a 33 GB box the first attempt livelocked the instance hard enough that
sshd could not fork; the second died cleanly under a 26 GB cgroup cap.

The invariant under test is structural, not a byte count: a split must be loaded, consumed,
and released before the next is loaded. That is what keeps peak memory at max(split) instead
of sum(splits), and it holds regardless of how the cache grows later.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlb_dl import precollate  # noqa: E402
from mlb_dl import dataset_cache  # noqa: E402


def test_prepare_all_streams_one_split_at_a_time(tmp_path, monkeypatch):
    events: list[tuple[str, str]] = []

    def fake_load_dataset(cache_dir, split_name):
        events.append(("load", split_name))
        return f"ds::{split_name}"

    def fake_load_cached_datasets(cache_dir):
        # Loading all three up front is the defect. Fail loudly rather than let the test
        # pass by silently tolerating the old shape.
        raise AssertionError(
            "prepare_all called load_cached_datasets, which materialises train+val+test "
            "simultaneously (~26 GB resident). It must call load_dataset per split.")

    def fake_prepare_split(ds, output_path, name, num_workers):
        events.append(("prepare", name))
        assert ds == f"ds::{name}", f"prepare_split got {ds!r} while preparing {name}"
        return {"n_samples": 0, "n_games": 0}

    monkeypatch.setattr(dataset_cache, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(dataset_cache, "load_cached_datasets", fake_load_cached_datasets)
    monkeypatch.setattr(precollate, "prepare_split", fake_prepare_split)

    precollate.prepare_all(str(tmp_path / "cache"), str(tmp_path / "out"), num_workers=2)

    # Strict interleaving. [load,load,load,prepare,prepare,prepare] would mean all three are
    # alive at once even though each prepare only needs its own.
    assert events == [
        ("load", "train"), ("prepare", "train"),
        ("load", "val"), ("prepare", "val"),
        ("load", "test"), ("prepare", "test"),
    ], f"splits were not streamed one at a time: {events}"


def test_prepare_all_still_writes_a_complete_top_manifest(tmp_path, monkeypatch):
    """Streaming must not drop a split from the manifest.

    The obvious way to implement the fix -- rebind one `ds` variable in a loop -- also makes
    it easy to overwrite manifest entries or lose one. A precollate output whose manifest is
    missing a split fails much later, inside training.
    """
    monkeypatch.setattr(dataset_cache, "load_dataset",
                        lambda cache_dir, split_name: f"ds::{split_name}")
    monkeypatch.setattr(
        precollate, "prepare_split",
        lambda ds, output_path, name, num_workers: {"n_samples": 7, "tag": name})

    out = tmp_path / "out"
    precollate.prepare_all(str(tmp_path / "cache"), str(out), num_workers=1)

    import json
    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest["splits"]) == {"train", "val", "test"}
    for name in ("train", "val", "test"):
        assert manifest["splits"][name]["tag"] == name, \
            f"manifest entry for {name} holds another split's result"
