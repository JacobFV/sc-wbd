"""Every trainable module must be reachable by some card's grant pattern.

The defect this exists for, measured on the shipped run-2 checkpoint:

    2,231,447 of 2,516,530 parameters -- 88.7% of the model, including the
    entire family-indexed regional model -- could not be granted a gradient by
    any enabled source card, for the whole run.

Not a curriculum decision. A **string mismatch**. The regional modules were
renamed ``local`` -> ``family_local``, ``residual`` -> ``family_residual``,
``readout`` -> ``family_readout`` when the family-padded architecture landed. The
source cards still grant ``local.*``, ``residual.*``, ``readout.*``, and
``fnmatch("family_local.ports.out_proj.weight", "local.*")`` is ``False``.

Nothing reported it. A card pattern that matches no parameter is not an error to
``fnmatch`` — it is an empty set, and an empty permission set produces a model
that trains on whatever *is* matched and converges perfectly happily. The loss
went down. The run finished. The weights shipped.

Two independent methods agree on the same set:

* **mechanism** — the enabled cards' ``gradient_permission`` globs, matched
  against the checkpoint's own parameter names;
* **measurement** — ``family_local``, ``family_residual``, ``family_readout``,
  ``observation`` and ``behaviour`` are bit-identical across *every* consecutive
  pair of the five stage checkpoints.

The consequence for the result: run 2's negative finding cannot be read as
"family-indexed heterogeneous regional state does not help". The heterogeneous
regional state never trained. See ``reports/RUN2.md`` §4.

Scope stated exactly: the stage checkpoints begin at the *end* of
``T1_measured_founding``, so T1's interior is not directly observable from the
artifacts. The mechanism argument is stage-independent — the same patterns are
matched against the same names in every stage — so T1 is covered by mechanism
rather than by measurement, and this test asserts the mechanism.
"""

from __future__ import annotations

import fnmatch
from collections import Counter
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CARDS = REPO / "configs/curriculum/source_cards"
CKPT = REPO / "checkpoints/scwbd-002-pilot/stage_T1_measured_founding.pt"

#: Excluded from the "some card grants it" sweep. ``negative_control_shuffled``
#: carries a bare ``*`` in ``frozen`` and grants nothing by design; including it
#: makes every coverage question answer "covered".
_NOT_A_GRANTER = {"negative_control_shuffled"}


def _grant_patterns() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in sorted(CARDS.glob("*.yaml")):
        if f.stem in _NOT_A_GRANTER:
            continue
        card = yaml.safe_load(f.read_text()) or {}
        if not card.get("enabled", True):
            continue
        out[f.stem] = [
            str(p).split("#")[0].strip() for p in (card.get("gradient_permission") or [])
        ]
    return out


def _state_keys() -> list[str]:
    """Parameter names as ``_CombinedModule`` presents them to the gate.

    Not just ``ck["model"]``. The trainer wraps the model, the posterior and
    (in individualisation stages) the individualizer in one module, strips the
    ``model.`` prefix, and leaves the other two prefixed. So ``posterior.*`` and
    ``individualizer.*`` are live patterns even though neither appears in the
    model's own state dict — reading only ``ck["model"]`` reports both as
    matching nothing, which is a false accusation of exactly the defect this
    file is about.
    """
    import torch

    keys: list[str] = []
    # The UNION across every stage checkpoint, not one of them. The
    # individualizer is constructed only in an individualisation stage, so it is
    # absent from the T1 checkpoint -- and checking `individualizer.*` against
    # T1 alone reports it as a pattern that matches nothing, which is true of
    # that file and false of the run. A per-stage reachability question needs a
    # per-stage key set; this test asks the run-level one.
    for f in sorted(CKPT.parent.glob("stage_*.pt")):
        ck = torch.load(f, map_location="cpu", weights_only=False)
        keys += list((ck.get("model") or {}).keys())
        for container in ("posterior", "individualizer"):
            sub = ck.get(container)
            if isinstance(sub, dict):
                keys += [f"{container}.{k}" for k in sub]
    return sorted(set(keys))


def _ungrantable() -> list[str]:
    pats = [p for v in _grant_patterns().values() for p in v if p]
    return [k for k in _state_keys() if not any(fnmatch.fnmatch(k, p) for p in pats)]


@pytest.mark.skipif(not CKPT.is_file(), reason="run-2 checkpoint not on disk")
def test_the_shipped_model_records_the_defect_it_was_trained_with() -> None:
    """Pins the defect as it stands in the published artifact.

    This is deliberately an assertion that the *shipped checkpoint* has the
    problem, not that it is fixed. The weights are published; rewriting the
    cards does not retrain them, and a test that went green on a card edit would
    claim the artifact had changed when only the configuration had.

    ``reports/decorative_guards.md`` calls the shape of a test that demands a
    defect still exist "the inverse category", and warns it becomes stale the
    moment the defect is repaired. That risk is accepted here for one reason:
    this checkpoint is frozen. Its parameter names cannot change. When run 3
    ships, this test should be pointed at *that* checkpoint, where it is
    expected to find an empty set — and
    :func:`test_no_module_is_unreachable_by_every_enabled_card` is the forward
    guard that has to pass before then.
    """
    by_module = Counter(k.split(".")[0] for k in _ungrantable())
    assert dict(by_module) == {
        "family_local": 141,
        "family_readout": 36,
        "family_residual": 32,
        "observation": 36,
        "behaviour": 5,
        "tau_prior": 1,
    }, (
        "the run-2 checkpoint's ungrantable set changed. These weights are "
        "immutable, so this can only mean the card patterns were edited -- which "
        "does not retrain anything. Point the forward guard at the new run "
        "instead of adjusting this record of the old one."
    )


def test_no_module_is_unreachable_by_every_enabled_card() -> None:
    """The forward guard: a top-level module no card can name is a silent freeze.

    Checked at the level of the *module prefix* rather than each tensor, because
    that is the granularity at which the rename happened and at which a human
    reads a card. A module with no reaching pattern will sit at its
    initialisation for an entire run while participating in the forward pass,
    and no loss curve, gradient norm, or convergence check will say so.

    Currently expected to FAIL for the run-2 model and to be the gate run 3 has
    to pass. It is written against the checkpoint on disk so it measures a real
    architecture rather than a fixture that agrees with the cards by
    construction.
    """
    if not CKPT.is_file():
        pytest.skip("no checkpoint on disk to check card patterns against")

    ungrantable_modules = sorted({k.split(".")[0] for k in _ungrantable()})
    known = {
        # The renamed regional model. The whole reason this file exists.
        "family_local",
        "family_readout",
        "family_residual",
        # Present in the model, named by no card. `sim_wholebrain` declares
        # `observation:sim_wholebrain:nuisance` in compiler_permission -- a
        # compiler port, not a torch parameter pattern. The two namespaces look
        # alike and never meet.
        "observation",
        # Arrived with the attachment axis; no source declares a boundary_output
        # channel yet, so nothing can grant it. Expected to move when one does.
        "behaviour",
        # A prior scale, plausibly intended to be fixed -- but nothing says so,
        # which is the point: "frozen because no card names it" and "frozen
        # because a card froze it" are different facts and only one is auditable.
        "tau_prior",
    }
    unexpected = [m for m in ungrantable_modules if m not in known]
    assert not unexpected, (
        f"module(s) {unexpected} exist in the model and no enabled card's "
        "gradient_permission can name them, so they will train at their "
        "initialisation and nothing will report it. Either add a pattern that "
        "reaches them to the card that should own them, or record here why they "
        "are deliberately unreachable."
    )


def test_every_grant_pattern_reaches_at_least_one_parameter() -> None:
    """The mirror check: a pattern matching nothing is a permission that grants nothing.

    ``local.*`` on a model whose module is called ``family_local`` is not an
    error anywhere in the stack — it is an empty set. This is the check that
    would have caught the rename on the day it happened, from the other side.
    """
    if not CKPT.is_file():
        pytest.skip("no checkpoint on disk to check card patterns against")

    keys = _state_keys()
    dead: dict[str, list[str]] = {}
    for card, pats in _grant_patterns().items():
        empty = [p for p in pats if p and not any(fnmatch.fnmatch(k, p) for k in keys)]
        if empty:
            dead[card] = empty

    assert dead == {
        # Every one of these is the rename. Recorded rather than removed: the
        # cards are what run 3 will be launched from, and deleting the patterns
        # would hide that these sources intend to train the regional model.
        #
        # `log_dt_scale` is here for a different reason and is worth separating:
        # it is not a rename, it is a parameter the cards grant that this
        # architecture does not have at all.
        "ds002336_real": ["local.*", "readout.*"],
        # `individualizer.*` is dead for a third reason, and the worst of the
        # three: the individualizer is `None` in EVERY stage checkpoint,
        # including `stage_T1_individualisation.pt` -- the stage named for it.
        # `run_stage` only constructs one when `admission.individualize` is
        # true, so that stage ran without the module it exists to fit. The
        # published card explains the zero between-participant theta spread by
        # the participant-disjoint split, which is true and is not the whole
        # reason: there is no individualizer in the artifact to spread.
        "eegmmidb_real": [
            "local.*",
            "residual.*",
            "readout.*",
            "log_dt_scale",
            "individualizer.*",
        ],
        "sim_wholebrain": ["local.*", "residual.*", "readout.*", "log_dt_scale"],
    }, (
        "the set of card patterns that match no parameter in the shipped model "
        "changed. A pattern here grants exactly nothing; it reads in the card as "
        "though the source trains that module."
    )
