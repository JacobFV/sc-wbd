"""The regional tensors must have MOVED, measured on the weights themselves.

`test_card_patterns_reach_the_model.py` is the mechanism half: it asserts that
some enabled card's glob matches every module's parameter names. This file is
the measurement half: it asserts that the tensors are no longer bit-identical to
their initialisation.

They are independent, and run 2 failed both without either being checked. The
cards granted `local.*` against modules called `family_local`, so 88.8% of the
parameters could not receive a gradient -- and nothing measured whether they
had, because an unmatched glob is an empty permission set rather than an error
and the loss fell anyway.

Two properties this checks that a loss curve cannot:

* a module that is entirely bit-identical to its initialisation while sitting on
  the forward pass;
* `residual_ratio == 0.0` **exactly**. The residual output projections ship
  zero-initialised, so an exact zero is not a small effect -- it is the
  signature of a residual that never trained. A tolerance would hide it.

The comparison is a bit comparison via sha256, on purpose. The question is "did
this tensor receive a gradient at all", and any answer involving a threshold
invites the threshold to be tuned until the answer is yes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUN3 = REPO / "checkpoints/scwbd-003"
SMOKE = REPO / "checkpoints/scwbd-003-smoke"

#: Modules that carry the regional model. These are the ones run 2 froze.
REGIONAL = ("family_local", "family_residual", "family_readout")

#: Modules added in run 3 that were unreachable before it, so a regression to
#: "declared but frozen" is the specific thing worth catching.
RUN3_NEW = ("behaviour", "eeg_montages", "tms_drive")


def _checkpoints() -> list[Path]:
    out: list[Path] = []
    for d in (RUN3, SMOKE):
        if d.is_dir():
            out += sorted(d.glob("stage_*.pt")) or sorted(d.glob("last.pt"))
    return out


def _load(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _moved(ck: dict) -> dict:
    # An architecture-only checkpoint is written before any step, purely to give
    # `test_card_patterns_reach_the_model` a set of parameter names to match the
    # cards against. Nothing in it has moved and nothing should have; asking
    # this question of it is a category error, not a finding. It says so in its
    # own `stage` field rather than being recognised by filename.
    if ck.get("stage") == "architecture_only":
        pytest.skip("architecture-only checkpoint: written before any training step")
    rep = (ck.get("extra") or {}).get("moved_since_init")
    if not rep:
        pytest.skip(
            f"checkpoint carries no `moved_since_init`; it predates the "
            "fingerprint and cannot answer this question"
        )
    return rep


@pytest.mark.parametrize("ckpt", _checkpoints(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_the_regional_modules_are_not_at_their_initialisation(ckpt: Path) -> None:
    """The check HANDOFF-003 makes a launch precondition."""
    rep = _moved(_load(ckpt))
    by = rep["by_module"]
    dead = []
    for m in REGIONAL:
        e = by.get(m)
        if e is None:
            continue  # the pooled arm names them without the prefix
        if e["moved"] == 0:
            dead.append(f"{m} ({e['frozen']} tensors, all bit-identical to init)")
    assert not dead, (
        f"{ckpt.name}: the regional model did not train: {dead}. This is run 2's "
        "defect measured directly on the weights. Check which card is expected to "
        "grant these modules and whether its glob matches the names they actually "
        "have -- `fnmatch('family_local.ports.out_proj.weight', 'local.*')` is False."
    )


@pytest.mark.parametrize("ckpt", _checkpoints(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_the_attachment_kinds_added_in_run3_received_gradient(ckpt: Path) -> None:
    """`behaviour`, the montage heads and the TMS drive are new and reachable.

    Each existed on the forward path before it had a source. The point of run 3
    is that each now has one, so "present and frozen" is exactly the regression
    to catch.
    """
    rep = _moved(_load(ckpt))
    by = rep["by_module"]
    dead = [
        f"{m} ({by[m]['frozen']} tensors)"
        for m in RUN3_NEW
        if m in by and by[m]["moved"] == 0
    ]
    assert not dead, (
        f"{ckpt.name}: {dead} are on the forward path and bit-identical to their "
        "initialisation. Either the source that should reach them is not admitted "
        "by this stage, or its grant pattern names nothing."
    )


@pytest.mark.parametrize("ckpt", _checkpoints(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_most_of_the_model_moved(ckpt: Path) -> None:
    """The headline number run 2 got wrong: 88.8% could not receive a gradient.

    A generous floor. It is not a quality bar -- a stage may legitimately freeze
    a head whose source it does not admit -- but a run in which most of the model
    is at its initialisation is the failure this whole file exists for.
    """
    rep = _moved(_load(ckpt))
    frac = rep["n_moved"] / max(rep["n_parameters"], 1)
    assert frac > 0.5, (
        f"{ckpt.name}: only {rep['n_moved']}/{rep['n_parameters']} tensors "
        f"({frac:.1%}) differ from their initialisation. Run 2 shipped at 11.2%. "
        f"Frozen by module: "
        f"{ {k: v['frozen'] for k, v in rep['by_module'].items() if v['frozen']} }"
    )


@pytest.mark.parametrize("ckpt", _checkpoints(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_the_residual_ratio_is_not_exactly_zero(ckpt: Path) -> None:
    """`residual_ratio == 0.0` exactly is a frozen residual, not a small effect.

    The residual output projections are zero-initialised, so the ratio starts at
    exactly zero and stays there if the module never trains. Any real training
    moves it off zero; the assertion is therefore against the exact value and
    needs no tolerance to choose.
    """
    ck = _load(ckpt)
    metrics = ck.get("metrics") or {}
    extra = ck.get("extra") or {}
    ratio = metrics.get("residual_ratio", extra.get("residual_ratio"))
    if ratio is None:
        pytest.skip("this checkpoint records no residual_ratio")
    assert float(ratio) != 0.0, (
        f"{ckpt.name}: residual_ratio is exactly 0.0. The residual output "
        "projections ship zero-initialised, so this is the signature of a "
        "residual that never received a gradient -- not a residual with a small "
        "effect."
    )


@pytest.mark.parametrize("ckpt", _checkpoints(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_parameter_has_a_recorded_initialisation(ckpt: Path) -> None:
    """A module built after the fingerprint has no baseline, and reads as moved.

    ``tms_drive`` is constructed by ``build_data`` -- it only exists when the
    perturbation corpus is on disk -- so it is absent from the fingerprint taken
    in ``__init__``. ``moved_since_init`` then compared a hash against ``None``,
    which is unequal, so the drive reported as MOVED on a checkpoint that had
    never taken a step. Measured exactly that: ``tms_drive moved 4/4`` with
    every other module frozen, on the architecture-only checkpoint.

    That is a false pass in the guard run 3 exists for, on the module carrying
    its novel claim. The trainer now registers late-built modules; this asserts
    the set stays empty so the same thing cannot come back through another one.
    """
    rep = _moved(_load(ckpt))
    unf = rep.get("unfingerprinted")
    if unf is None:
        pytest.skip("checkpoint predates the unfingerprinted field")
    assert unf == [], (
        f"{ckpt.name}: {unf} have no recorded initialisation, so their "
        "moved/frozen status is not evidence either way. Register the module "
        "with `_fingerprint_late_module` where it is built."
    )


def test_there_is_something_to_check() -> None:
    """Guards the guard: a parametrisation over an empty list passes vacuously.

    ``reports/decorative_guards.md`` names exactly this shape. With no run-3
    checkpoint on disk every test above collects zero cases and the file reports
    green while measuring nothing.
    """
    if not _checkpoints():
        pytest.skip("no run-3 checkpoint on disk yet")
    assert _checkpoints()
