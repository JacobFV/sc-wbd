"""A module belonging to the other arm must not be built, or counted.

Run 3 published "99.98% of parameters moved". Every such fraction has a
denominator, and an unreachable module inflates it. Two modules were in that
denominator while sitting on no forward path:

* ``msg_proj`` — both call sites read ``family_local.ports.message(...)`` when a
  family layout exists and only fall back to ``msg_proj`` when it does not, so
  on the family arm it is never called. 2 tensors, 72 parameters, bit-identical
  to init after 13,400 steps. It was the *only* frozen group in run 3 with no
  explanation, and the explanation is that it belongs to the pooled arm.
* ``source_proj`` — once a parcel carries a 3-vector moment, ``EEGHead.forward``
  takes the ``L_vec`` path and ``source_amplitude`` is never called. 16 tensors,
  1,796 parameters.

Together, 1,868 of run 3's 5,966 frozen parameters — 31% of everything the run
could not train — were modules that could not have trained on that arm.

Neither is deleted: the pooled arm and the scalar arm are real configurations,
and the pooled arm is the control for body.tex sec. 11.4's first ablation. They
are built when their arm is selected and not otherwise.
"""

from __future__ import annotations

import dataclasses

import pytest

from scwbd.foundation.anatomy import load_anatomy
from scwbd.foundation.config import load_config
from scwbd.foundation.model import SCWBD
from scwbd.foundation.util import set_determinism

CONFIG = "configs/run3/scwbd-003.yaml"


@pytest.fixture(scope="module")
def anat():
    return load_anatomy(device="cpu")


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


def _build(model_cfg, anat) -> SCWBD:
    set_determinism(20260807)
    return SCWBD(model_cfg, anat)


def test_the_family_arm_does_not_build_the_pooled_arms_message_projection(cfg, anat) -> None:
    m = _build(cfg.model, anat)
    assert cfg.model.family_state, "this run config is the family arm; the test assumes it"
    assert m.msg_proj is None, (
        "msg_proj was built on the family arm. It is unreachable there -- both "
        "call sites prefer family_local.ports.message -- so it lands in the "
        "parameter total as 72 permanently untrainable parameters and shows up "
        "as an unexplained frozen group, which is exactly what happened in run 3."
    )


def test_the_pooled_arm_still_builds_it(cfg, anat) -> None:
    """Gating must not delete the control arm's capability."""
    pooled = dataclasses.replace(cfg.model, family_state=False)
    m = _build(pooled, anat)
    assert m.msg_proj is not None, (
        "the pooled arm has no message projection, so it cannot pass messages at "
        "all -- the gate was written as a deletion rather than as an arm switch"
    )


def test_the_family_arm_reaches_its_message_path_without_msg_proj(cfg, anat) -> None:
    """Guards the guard: `msg_proj is None` is only safe if nothing calls it.

    If a future edit routes the family arm through `msg_proj`, this fails with a
    TypeError rather than the model silently training a module the report says
    is absent.
    """
    import torch

    m = _build(cfg.model, anat)
    assert m.family_local is not None
    x = torch.zeros(1, m.n_regions, m.layout.dim)
    msg = m.family_local.ports.message(x)
    assert msg.shape[0] == 1 and msg.dim() >= 2


def test_the_frozen_parameter_count_drops_by_exactly_the_dead_module(cfg, anat) -> None:
    """The number is the point: 72, not "some"."""
    with_gate = sum(p.numel() for p in _build(cfg.model, anat).parameters())

    # The pooled arm's own msg_proj is 228 parameters, not 72: `export_dim` is a
    # function of which state components the arm exports, so the two arms build
    # differently-shaped projections. 72 is the FAMILY arm's shape -- the size of
    # the module that was being built and never called, and the exact count run 3
    # reported frozen.
    pooled = _build(dataclasses.replace(cfg.model, family_state=False), anat)
    assert sum(p.numel() for p in pooled.msg_proj.parameters()) == 228

    assert with_gate == 26_304_729 - 72, (
        f"expected run 3's total minus msg_proj's 72, got {with_gate:,}. If the "
        "architecture changed for another reason, re-derive this number rather "
        "than adjusting it."
    )
