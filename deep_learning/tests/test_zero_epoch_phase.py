"""A phase requested with 0 epochs must be skipped, not crash the run.

`--phase2-epochs 0 --phase3-epochs 0` is the natural way to ask for a phase-1-only run,
which is what the weather A/B needs: the comparison is decided on phase-1 best_val, and
running phases 2 and 3 as well would roughly double each arm's ~5h.

The phases are invoked unconditionally, and _train_phase ends by restoring the best
checkpoint with an unguarded `torch.load(checkpoint_dir / "best.pt")`. With max_epochs=0
the epoch loop never runs, so best.pt is never written and that load raises
FileNotFoundError — at the START of phase 2, i.e. after phase 1 has already spent its five
hours and saved a perfectly good checkpoint. Both A/B arms would die that way, and the
failure surfaces only after ten hours of GPU time.
"""

import logging

import pytest
import torch

from mlb_dl.train_unified import _train_phase


class _Model(torch.nn.Module):
    """Minimal stand-in: _train_phase only ever calls state_dict/load_state_dict on it
    when no epochs run."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)


def _call(tmp_path, max_epochs):
    return _train_phase(
        model=_Model(), loss_fn=None, train_loader=[], val_loader=[],
        optimizer=None, scheduler=None, device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "phase2", max_epochs=max_epochs,
        patience=5, grad_clip=1.0, phase_name="phase2",
    )


def test_zero_epoch_phase_returns_instead_of_loading_a_checkpoint(tmp_path):
    """The bug: no epochs ran, so there is no best.pt to restore."""
    hist = _call(tmp_path, 0)
    assert hist["epochs_trained"] == 0
    assert hist["history"] == []


def test_zero_epoch_phase_does_not_invent_a_best_loss(tmp_path):
    """best_val_loss must not come back as a real-looking number. inf also breaks strict
    JSON, and training_history.json is written with json.dump — None is the honest value
    for a phase that never evaluated anything."""
    hist = _call(tmp_path, 0)
    assert hist["best_val_loss"] is None


def test_zero_epoch_phase_leaves_the_previous_phase_weights_untouched(tmp_path):
    """Phase 1's restored weights are the ones the run must carry forward. A skipped phase
    that reset or reloaded the model would silently discard them."""
    model = _Model()
    before = model.lin.weight.detach().clone()
    _train_phase(
        model=model, loss_fn=None, train_loader=[], val_loader=[],
        optimizer=None, scheduler=None, device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "phase3", max_epochs=0,
        patience=5, grad_clip=1.0, phase_name="phase3",
    )
    assert torch.equal(model.lin.weight, before)


def test_zero_epoch_phase_is_logged(caplog, tmp_path):
    """Never weaken logging: a silently skipped phase in a 10h run is indistinguishable
    from one that ran and did nothing useful."""
    with caplog.at_level(logging.INFO):
        _call(tmp_path, 0)
    assert any("phase2" in r.message and "skip" in r.message.lower()
               for r in caplog.records), [r.message for r in caplog.records]


def test_negative_epochs_are_treated_as_zero(tmp_path):
    """argparse accepts any int; -1 must not fall through to the same unguarded load."""
    hist = _call(tmp_path, -1)
    assert hist["epochs_trained"] == 0
