"""Seed control, without which the weather A/B cannot be read.

Measured on the training box: three phase-1 runs at the same nominal architecture and
lr=4e-04 reached best_val 4.9330 (6 epochs, interrupted), 5.0289 (1 epoch, crashed) and
4.9521 (12 epochs, early-stopped). Within the 12-epoch run the val series after epoch 1
spans 4.9521-5.0250. So the run-to-run and epoch-to-epoch spread is a few hundredths of a
nat, and the trainer called no seeding function at all -- random init and shuffle order
differed every time.

An unseeded single-run A/B therefore cannot attribute a few-hundredths change to the
weather channels rather than to initialization. Seeding makes the arms a paired
comparison: identical batch order, and an identical init draw sequence up to the point
the weather encoder's differing shapes consume different numbers of random values.
"""

import subprocess
import sys

import numpy as np
import pytest
import torch

from mlb_dl.train_unified import _seed_everything


def test_seeding_makes_torch_draws_reproducible():
    _seed_everything(1234)
    a = torch.randn(64)
    _seed_everything(1234)
    b = torch.randn(64)
    assert torch.equal(a, b)


def test_seeding_covers_numpy_and_python_random():
    """The dataset and collate paths draw from numpy, and the player-hash bucketing and
    any subsampling from python's random; seeding torch alone would leave those free."""
    import random

    _seed_everything(7)
    a = (np.random.rand(8).tolist(), random.random())
    _seed_everything(7)
    b = (np.random.rand(8).tolist(), random.random())
    assert a == b


def test_different_seeds_actually_differ():
    """Guard against a no-op implementation that returns without seeding."""
    _seed_everything(1)
    a = torch.randn(32)
    _seed_everything(2)
    b = torch.randn(32)
    assert not torch.equal(a, b)


def test_seeded_shuffle_order_is_identical_across_runs():
    """The A/B's real requirement: the two arms must see the same batches in the same
    order. Shuffling is what a bare torch.manual_seed can still leave adrift when the
    loader builds its own generator."""
    from torch.utils.data import DataLoader

    ds = list(range(500))

    def order(seed):
        _seed_everything(seed)
        g = torch.Generator()
        g.manual_seed(seed)
        return [int(x) for b in DataLoader(ds, batch_size=25, shuffle=True, generator=g)
                for x in b]

    assert order(99) == order(99)
    assert order(99) != order(100)


@pytest.mark.parametrize("cmd", ["fit-unified", "evaluate"])
def test_seed_is_exposed_on_the_cli(cmd):
    out = subprocess.run([sys.executable, "-m", "mlb_dl.train_unified", cmd, "--help"],
                         capture_output=True, text=True)
    assert "--seed" in out.stdout, f"{cmd} has no --seed:\n{out.stdout}"
