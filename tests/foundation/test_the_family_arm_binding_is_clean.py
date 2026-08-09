"""The FAMILY arm's compiler binding must name tensors the family arm builds.

`test_compiler_binding.py` builds its model from `configs/scwbd_001_beta.yaml`,
which is `family_state: false`. Every one of its fourteen assertions therefore
runs on the POOLED arm, and `FOUNDATION_FAMILY_BINDING` -- the table that
REPLACES those entries whenever a family layout exists -- is exercised by none
of them.

That is how this shipped: `port:*.message_out` was bound to
`("msg_proj.*", "family_local.ports.out_proj.*")`, and when `SCWBD.msg_proj` was
gated on `not cfg.family_state` the first glob stopped naming anything on the
arm the table is FOR. `audit_binding` reports it as a decorative permission and
`FoundationTrainer.__init__` raises `BindingDriftError` rather than training
through it -- so the run-4 trainer would not construct at all. Found by trying
to build one.

The lesson is the smoke's own, in `FoundationTrainer.smoke`: "a fix verified
against the control is not verified". Both arms are live; a table that exists
only for one of them has to be checked on that one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scwbd.foundation import compiler_bridge as cb
from scwbd.foundation.anatomy import load_anatomy
from scwbd.foundation.config import load_config
from scwbd.foundation.mixture import SourceSpec
from scwbd.foundation.model import SCWBD
from scwbd.foundation.util import set_determinism

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not cb.compiler_available(), reason="scwbd.compiler unavailable"
)


@pytest.fixture(scope="module")
def anat():
    return load_anatomy(
        device="cpu", n_cortex=40, n_subcortex=12, n_cerebellum=8, density=0.15, seed=7
    )


@pytest.fixture(scope="module")
def family_model(anat):
    """The treatment arm: heterogeneous region-indexed state, per-family operators."""
    cfg = load_config(REPO / "configs" / "scwbd_001_beta.yaml")
    cfg.model.n_regions = 60
    cfg.model.family_state = True
    set_determinism(0)
    m = SCWBD(cfg.model, anat)
    assert m.family_local is not None, "fixture did not build the family arm"
    assert m.msg_proj is None, (
        "the pooled arm's message projection is built on the family arm, so this "
        "test cannot distinguish the two tables"
    )
    return m


@pytest.fixture(scope="module")
def compiled(anat):
    cfg = load_config(REPO / "configs" / "scwbd_001_beta.yaml")
    specs = SourceSpec.load_dir(cfg.mixture_cards)
    assert specs, "no source cards found; the production config ships them"
    probe = [
        SourceSpec(**{**s.as_dict(), "gradient_permission": s.compiler_permission})
        for s in specs.values()
    ]
    return cb.compile_foundation(anat, probe)


def test_the_family_arm_audit_reports_no_problems(family_model, compiled) -> None:
    """The assertion `FoundationTrainer.__init__` makes before it will train."""
    audit = cb.audit_binding(family_model, compiled)
    assert audit["problems"] == [], (
        "the family arm's compiler->torch binding is incomplete:\n  "
        + "\n  ".join(audit["problems"])
        + "\n\nThe trainer raises BindingDriftError on this, so a run cannot even "
        "construct. Fix FOUNDATION_FAMILY_BINDING so every declared group names "
        "tensors this arm builds."
    )


def test_no_family_binding_glob_names_a_pooled_arm_module(family_model) -> None:
    """The direct form: the family table may not name the other arm's modules.

    `FOUNDATION_FAMILY_BINDING` *replaces* the corresponding
    `FOUNDATION_BINDING` entries rather than being unioned with them, so a
    pooled-arm glob left in it is not a harmless duplicate -- it is a pattern
    matching nothing on the only arm that reads the table.
    """
    import fnmatch

    names = [n for n, p in family_model.named_parameters() if p.requires_grad]
    dead: dict[str, list[str]] = {}
    for group, pats in cb.FOUNDATION_FAMILY_BINDING.items():
        empty = [p for p in pats if not any(fnmatch.fnmatch(n, p) for n in names)]
        if empty:
            dead[group] = empty
    assert dead == {}, (
        f"these FOUNDATION_FAMILY_BINDING globs match no parameter of the family "
        f"arm: {dead}. `msg_proj.*` was one of them, for a whole cycle."
    )


def test_both_arms_bind_the_same_declared_groups(family_model, anat, compiled) -> None:
    """A source card must not gain or lose a permission by which arm was built.

    `F_local` is the same declared thing whichever operator implements it. If
    the two tables cover different GROUP names, a card's `A_k` means something
    different in the two arms and the 11.4 comparison is between two different
    permission regimes as well as two operators.
    """
    cfg = load_config(REPO / "configs" / "scwbd_001_beta.yaml")
    cfg.model.n_regions = 60
    set_determinism(0)
    pooled = SCWBD(cfg.model, anat)
    assert pooled.family_local is None

    fam = {k for k, v in cb.bind_masks(family_model, compiled).items() if v}
    poo = {k for k, v in cb.bind_masks(pooled, compiled).items() if v}
    assert fam == poo, (
        f"sources bound on one arm only: family-only={sorted(fam - poo)}, "
        f"pooled-only={sorted(poo - fam)}"
    )
