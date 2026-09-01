"""The pregame readout must have its OWN head parameters, not share the live head.

WHY. `ab619fd` fixed *which representation* a pregame row reads (per-row `ctx_pool`
instead of batch-level last-token padding). It did not fix *what reads it*: one
`head_home_win` served both `ctx_pool` (pregame) and `backbone_out[:, -1, :]` (live).
Those two inputs have different geometry and different label difficulty, and live rows
are 93.2% of samples (293,646 of 315,791), so the shared head is fitted almost entirely
to the live distribution — where sharpness is rewarded because the outcome is nearly
determined late in a game.

The measured consequence, on the `readout_fix` e6 checkpoint over the test split:
the checkpoint's own head emits pregame home_win BSS **-0.0054**, while a linear probe
refitted on that same frozen `ctx_pool` reaches **+0.0082** (test) / +0.0113 (pooled
val+test). The representation carries more than the shared head is able to say, and the
gap is attributable to the head rather than the trunk because the probe read the trunk
unchanged. See [[project-pregame-probe-verdict-2026-08-31]] and
[[project-dl-skill-by-prefix-2026-09-01]].

These tests pin parameter-level separation, which is the property that lets the pregame
readout be fitted to pregame rows without live gradient overwriting it. They are
deliberately about *gradient reachability*, not about skill: skill is a training
outcome, separation is a code invariant.
"""

from __future__ import annotations

import torch

from deep_learning.mlb_dl.game_transformer import GameTransformer
from deep_learning.mlb_dl.train_unified import _prepare_model_input

from deep_learning.tests.test_pregame_readout_invariance import (  # noqa: F401 — fixtures
    D_MODEL,
    PREFIX_LEN,
    _collated,
    _model,
    _price,
)

# The five team-level heads. The player head is intentionally NOT untied: player targets
# at prefix_length=0 are a separate question (scratched slots settle last-fair) and
# bundling them would make a retrain's result unattributable.
TEAM_HEADS = (
    "head_home_win",
    "head_yrfi",
    "head_extra_innings",
    "head_negbin_home",
    "head_negbin_away",
)

LOGIT_KEYS = ("home_win_logit", "yrfi_logit", "extra_innings_logit", "mu_home", "mu_away")


def _perturb(module: torch.nn.Module) -> None:
    """Push every parameter of `module` far off its trained value.

    Large and dense on purpose: a subtle nudge could vanish into fp tolerance and let a
    shared head pass as separated.
    """
    with torch.no_grad():
        for p in module.parameters():
            p.add_(torch.full_like(p, 0.5))


def test_the_five_team_heads_all_have_a_pregame_counterpart():
    """Structural: the untied parameters exist and are distinct objects."""
    model = _model()
    for name in TEAM_HEADS:
        pregame_name = f"{name}_pregame"
        assert hasattr(model, pregame_name), (
            f"{pregame_name} does not exist — the pregame readout still shares "
            f"{name} with the live path"
        )
        live_params = {id(p) for p in getattr(model, name).parameters()}
        pregame_params = {id(p) for p in getattr(model, pregame_name).parameters()}
        assert live_params.isdisjoint(pregame_params), (
            f"{pregame_name} aliases {name}'s parameters — untying must not be a "
            f"second reference to the same tensors"
        )


def test_perturbing_the_pregame_head_leaves_live_prices_bit_identical():
    """Live rows must not see the pregame head at all.

    This is the half that protects the working task: the live path currently reaches
    +0.77 BSS by 300 pitches and must be bit-for-bit unaffected by anything the pregame
    readout does.
    """
    model = _model()
    batch = _collated([0, 9], seed=23)

    before = _price(model, batch, row=1)  # row 1 has 9 real pitches -> live path
    for name in TEAM_HEADS:
        _perturb(getattr(model, f"{name}_pregame"))
    after = _price(model, batch, row=1)

    drift = {k: abs(before[k] - after[k]) for k in LOGIT_KEYS}
    worst = max(drift, key=drift.get)
    assert drift[worst] == 0.0, (
        f"the live price moved when only the PREGAME head changed: {worst} "
        f"{before[worst]:.6f} -> {after[worst]:.6f} (delta {drift[worst]:.6g}). "
        f"The two readouts are still coupled."
    )


def test_perturbing_the_live_head_leaves_pregame_prices_bit_identical():
    """Pregame rows must not see the live head.

    This is the half that buys the skill: it is what stops 93.2%-live gradient from
    dictating the pregame readout's weights.
    """
    model = _model()
    batch = _collated([0, 9], seed=23)

    before = _price(model, batch, row=0)  # row 0 is pregame
    for name in TEAM_HEADS:
        _perturb(getattr(model, name))
    after = _price(model, batch, row=0)

    drift = {k: abs(before[k] - after[k]) for k in LOGIT_KEYS}
    worst = max(drift, key=drift.get)
    assert drift[worst] == 0.0, (
        f"the pregame price moved when only the LIVE head changed: {worst} "
        f"{before[worst]:.6f} -> {after[worst]:.6f} (delta {drift[worst]:.6g}). "
        f"The pregame readout is still fitted by live gradient."
    )


def test_only_the_pregame_head_receives_gradient_from_a_pregame_row():
    """Gradient reachability, which is the property that actually trains the head.

    A price-level test can pass while gradient still leaks (e.g. a `where` on the wrong
    axis), so assert on `.grad` directly: a loss built only from pregame rows must leave
    every live head parameter with no gradient, and vice versa.
    """
    model = _model()
    model.train()
    batch = _prepare_model_input(_collated([0, 0, 9, 9], seed=31),
                                 player_context_dim=2 * D_MODEL)

    out = model(batch)
    # Rows 0-1 are pregame; build a loss that touches nothing else.
    out["home_win_logit"][:2].sum().backward()

    live_grad = [
        f"{name}.{pname}"
        for name in TEAM_HEADS
        for pname, p in getattr(model, name).named_parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    ]
    assert not live_grad, (
        f"a pregame-only loss produced gradient in live head parameters: {live_grad}. "
        f"Pregame rows are still fitting the live head."
    )

    pregame_grad = [
        f"{name}_pregame.{pname}"
        for name in ("head_home_win",)
        for pname, p in getattr(model, f"{name}_pregame").named_parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    ]
    assert pregame_grad, (
        "a pregame-only loss produced NO gradient in head_home_win_pregame — the "
        "pregame head is not reachable, which is the bug ab619fd was about, one layer up"
    )


def test_only_the_live_head_receives_gradient_from_a_live_row():
    model = _model()
    model.train()
    batch = _prepare_model_input(_collated([0, 0, 9, 9], seed=31),
                                 player_context_dim=2 * D_MODEL)

    out = model(batch)
    out["home_win_logit"][2:].sum().backward()  # rows 2-3 are live

    pregame_grad = [
        f"{name}_pregame.{pname}"
        for name in TEAM_HEADS
        for pname, p in getattr(model, f"{name}_pregame").named_parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    ]
    assert not pregame_grad, (
        f"a live-only loss produced gradient in pregame head parameters: "
        f"{pregame_grad}"
    )


def test_pregame_row_still_prices_identically_alone_and_beside_a_live_row():
    """Regression guard on the `ab619fd` invariant, now with untied heads.

    Untying introduces a per-row branch on top of the per-row representation choice.
    A batch-level implementation of THAT branch would reintroduce exactly the bug
    ab619fd fixed, one layer up, so re-assert the serving property here rather than
    trusting the readout test to cover it.
    """
    model = _model()

    both_pregame = _collated([0, 0], seed=7)
    mixed = _collated([0, 0], seed=7)
    live_rows = _collated([0, 9], seed=7)
    for key, val in live_rows.items():
        if isinstance(val, torch.Tensor):
            mixed[key][1] = val[1]

    alone = _price(model, both_pregame, row=0)
    beside_live = _price(model, mixed, row=0)

    drift = {k: abs(alone[k] - beside_live[k]) for k in alone}
    worst = max(drift, key=drift.get)
    assert drift[worst] < 1e-5, (
        f"with untied heads the pregame price again depends on batch composition: "
        f"{worst} moved {alone[worst]:.4f} -> {beside_live[worst]:.4f}"
    )


def test_a_pre_untying_checkpoint_trips_the_five_percent_load_bail():
    """An old checkpoint must FAIL to load, not load with random pregame heads.

    `score_test_predictions.py:105` and `extract_pregame_repr.py` refuse to score when more than
    5% of *state_dict keys* are unmatched. The five pregame heads are 20 of 207 keys = 9.66%, so
    the bail fires — but only by arithmetic margin, and it is measured on key count while the
    parameter share is just 4.35%, which would NOT have fired. That is the same silent-load class
    as [[reference-n-heads-invisible-to-state-dict]]: a plausible-looking pregame price produced
    by an untrained head.

    Pinned because the margin is fragile: adding heads, or raising the key count elsewhere,
    could drop the pregame heads back under 5% and re-open the silent path.
    """
    model = _model()
    keys = list(model.state_dict())
    pregame_keys = [k for k in keys if "_pregame" in k]

    assert len(pregame_keys) == 20, (
        f"expected 20 pregame head keys (5 heads x 2 layers x weight+bias), got "
        f"{len(pregame_keys)}: {pregame_keys}"
    )
    assert len(pregame_keys) > 0.05 * len(keys), (
        f"the pregame heads are {len(pregame_keys)}/{len(keys)} = "
        f"{100 * len(pregame_keys) / len(keys):.2f}% of state_dict keys, which is UNDER the 5% "
        f"load bail. A pre-untying checkpoint would load silently and be priced pregame by "
        f"randomly initialised heads. Tighten the bail or add an explicit pregame-head check."
    )


def test_an_all_live_batch_without_live_lengths_uses_the_live_head():
    """The legacy hand-built-batch path (kv-cache decode, smoke tests).

    `_team_readout` treats `live_lengths is None` as "every row is live". That contract
    must keep routing to the LIVE head, or callers that never pass `live_lengths` would
    silently be priced by an untrained pregame head.
    """
    model = _model()
    collated = _collated([9, 9], seed=41)
    batch = _prepare_model_input(collated, player_context_dim=2 * D_MODEL)
    batch.pop("live_lengths", None)

    with torch.no_grad():
        before = model(batch)["home_win_logit"].clone()
        for name in TEAM_HEADS:
            _perturb(getattr(model, f"{name}_pregame"))
        after = model(batch)["home_win_logit"]

    assert torch.equal(before, after), (
        "a batch with no live_lengths was routed through the pregame head; the "
        "all-live legacy contract is broken"
    )
