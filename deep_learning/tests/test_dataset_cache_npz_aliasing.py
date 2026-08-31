"""Reading a per-game array out of an .npz must touch each member exactly once.

Reproduces the 2026-08-31 precollate OOM. `CachedGameTransformerDataset.__init__` loaded the
as-of weather and the per-pitch decision offsets like this:

    z = np.load(asof_file)
    self._weather_asof_by_pk = {int(p): z["tensors"][i] for i, p in enumerate(z["pks"])}

`np.load` on an .npz returns a LAZY NpzFile, so `z["tensors"]` re-reads and re-materialises
the entire member on every iteration. Worse, `arr[i]` is a numpy *view* that keeps its parent
alive through `.base`, so each of those full arrays stays resident for as long as its row is
in the dict. Both costs are quadratic.

Measured on the real train split before the fix:

    tensors = (21384, 7, 7, 99) float32 = 414.9 MB, compress_type=0, one read = 0.45s
    21384 reads  -> 2.7 hours of I/O
    21384 live views x 414.9 MB base -> 8.9 TB retained
    np.load(...)["tensors"][0].base is np.load(...)["tensors"][1].base  ->  False

and what the kernel actually recorded when it killed it:

    Killed process 30286 (python) total-vm:68500300kB, anon-rss:27117324kB, file-rss:6696kB
    [30286] ... rss_anon 6779331  swapents 8388512   (27.1 GB resident + exactly 32.0 GiB swap)

59 GB of anonymous memory is reached after ~142 games, ~1.1 min in; the process died at
4.2 min. file-rss ~= 0 confirms these were real allocations, not the memmapped pitch arrays.

The invariants under test are both structural, so they keep holding as the cache grows:
one read per member, and one shared base for every row.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlb_dl.dataset_cache import _npz_rows_by_pk  # noqa: E402

# Real per-game shape: 7 decision hours x 7 target hours x 99 channels.
ROW_SHAPE = (7, 7, 99)
N_GAMES = 64


@pytest.fixture
def asof_npz(tmp_path):
    pks = np.arange(700000, 700000 + N_GAMES, dtype=np.int64)
    rng = np.random.default_rng(0)
    tensors = rng.standard_normal((N_GAMES, *ROW_SHAPE)).astype(np.float32)
    path = tmp_path / "weather_asof.npz"
    np.savez(path, pks=pks, tensors=tensors)
    return path, pks, tensors


def _count_member_reads(monkeypatch, path):
    """Record every NpzFile subscript. Returns the list the calls append to."""
    probe = np.load(path)
    cls = type(probe)
    probe.close()
    original = cls.__getitem__
    reads: list[str] = []

    def counting_getitem(self, key):
        reads.append(key)
        return original(self, key)

    monkeypatch.setattr(cls, "__getitem__", counting_getitem)
    return reads


def test_each_npz_member_is_read_exactly_once(asof_npz, monkeypatch):
    path, pks, _ = asof_npz
    reads = _count_member_reads(monkeypatch, path)

    _npz_rows_by_pk(path, "tensors")

    # One read of the keys, one of the values. The defect showed up here as N_GAMES + 1
    # reads of "tensors", which is what made a 415 MB member cost 2.7 hours.
    assert Counter(reads) == {"pks": 1, "tensors": 1}, (
        f"members were re-read from the archive: {Counter(reads)}")


def test_all_rows_share_one_base_array(asof_npz):
    path, pks, tensors = asof_npz
    rows = _npz_rows_by_pk(path, "tensors")

    bases = {id(r.base if r.base is not None else r) for r in rows.values()}
    assert len(bases) == 1, (
        f"{len(bases)} distinct base arrays retained for {len(rows)} rows; each view pins a "
        f"full copy of the member, so peak memory is O(n_games x member_size)")

    # State the footprint as a bound rather than trusting the base check alone: the whole
    # point is that total retained memory is the member, not a multiple of it.
    unique = {id(r.base): r.base for r in rows.values()}
    retained = sum(b.nbytes for b in unique.values())
    assert retained == tensors.nbytes, (
        f"retained {retained} bytes for a {tensors.nbytes}-byte member")


def test_rows_are_correct_and_keyed_by_int_pk(asof_npz):
    """Neither hoisting the read nor sharing a base may permute or mistype the mapping."""
    path, pks, tensors = asof_npz
    rows = _npz_rows_by_pk(path, "tensors")

    assert len(rows) == N_GAMES
    assert all(isinstance(k, int) for k in rows), "keys must be plain ints, not np.int64"
    for i, pk in enumerate(pks):
        np.testing.assert_array_equal(rows[int(pk)], tensors[i])


def test_object_arrays_need_allow_pickle(tmp_path):
    """wx_offsets.npz holds ragged per-pitch offsets, so it is an object array.

    Its call site passed allow_pickle=True; dropping that on the shared path would turn a
    working load into a ValueError only on the offsets artifact.
    """
    pks = np.array([1, 2, 3], dtype=np.int64)
    offsets = np.empty(3, dtype=object)
    for i in range(3):
        offsets[i] = np.arange(i + 1, dtype=np.int16)
    path = tmp_path / "wx_offsets.npz"
    np.savez(path, pks=pks, offsets=offsets)

    with pytest.raises(ValueError):
        _npz_rows_by_pk(path, "offsets")

    rows = _npz_rows_by_pk(path, "offsets", allow_pickle=True)
    assert len(rows) == 3
    np.testing.assert_array_equal(rows[3], np.arange(3, dtype=np.int16))
