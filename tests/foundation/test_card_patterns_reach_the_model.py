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

#: The grant patterns **as they stood during run 2**, before the repair. Frozen
#: here rather than read from the cards, because the cards have since been fixed
#: and rewriting them does not retrain the published weights. Reading live cards
#: to characterise a finished run would report the defect as gone the moment the
#: configuration changed -- which is the same "artifact vs intention" confusion
#: that put a LEGACY card directory on the published model card.
_RUN2_GRANTS: dict[str, tuple[str, ...]] = {
    "anatomical_prior": ("coupling.gain_*", "coupling.global_scale"),
    "ds002336_real": ("bold.*", "coupling.*", "local.*", "readout.*"),
    "eegmmidb_real": (
        "local.*", "residual.*", "coupling.*", "msg_proj.*", "msg_readin.*",
        "assimilate.*", "context.*", "readout.*", "eeg.*", "log_dt_scale",
        "individualizer.*",
    ),
    "montage_calibration": ("eeg.log_gain", "eeg.offset", "eeg.log_noise", "eeg.nuisance*"),
    "sim_wholebrain": (
        "local.*", "residual.*", "coupling.*", "msg_proj.*", "msg_readin.*",
        "assimilate.*", "context.*", "readout.*", "log_dt_scale", "posterior.*",
    ),
}

#: Every live architecture. The pooled/run-1 arm names its regional modules
#: ``local``/``residual``/``readout``; the family-padded arm prefixes them
#: ``family_*``. A pattern dead in one arm may be intended; a pattern dead in
#: EVERY arm is the defect.
#:
#: Run 3's architecture is here because it is the only one that builds the
#: modules run 3's cards grant: ``eeg_montages.*`` (one observation head per
#: montage) and ``tms_drive.*`` (the measured pulse). Checked against run 2's
#: checkpoint alone, those patterns correctly report as matching nothing --
#: they name modules that did not exist when those weights were written. The
#: fix is to check against the architecture the cards describe, which is what
#: this file's own docstring said to do when run 3 shipped.
_ARCHITECTURES = ("checkpoints/scwbd-002-pilot/stage_T1_measured_founding.pt",
                  "checkpoints/ci-smoke/last.pt",
                  "checkpoints/scwbd-003-smoke/last.pt",
                  "checkpoints/scwbd-003/last.pt")



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
        for container in ("posterior", "individualizer", "tms_drive"):
            sub = ck.get(container)
            if isinstance(sub, dict):
                keys += [f"{container}.{k}" for k in sub]
    return sorted(set(keys))


def _all_architecture_keys() -> list[str]:
    """Parameter names across every architecture this repo builds.

    Both arms are live. Checking the cards against one of them reports the
    other's patterns as matching nothing -- a false accusation of exactly the
    defect under test.
    """
    import torch

    keys: list[str] = []
    for rel in _ARCHITECTURES:
        f = REPO / rel
        if not f.is_file():
            continue
        ck = torch.load(f, map_location="cpu", weights_only=False)
        keys += list((ck.get("model") or {}).keys())
        for container in ("posterior", "individualizer", "tms_drive"):
            sub = ck.get(container)
            if isinstance(sub, dict):
                keys += [f"{container}.{k}" for k in sub]
    return sorted(set(keys))


def _ungrantable() -> list[str]:
    pats = [p for v in _grant_patterns().values() for p in v if p]
    return [k for k in _all_architecture_keys() if not any(fnmatch.fnmatch(k, p) for p in pats)]


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
    pats = [p for v in _RUN2_GRANTS.values() for p in v]
    keys = _state_keys()
    dead = [k for k in keys if not any(fnmatch.fnmatch(k, p) for p in pats)]
    by_module = Counter(k.split(".")[0] for k in dead)
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
        # `family_local`, `family_readout`, `family_residual` and `observation`
        # were here and are NOT any more -- the cards now grant both the pooled
        # and the family-padded namings, and `observation.*` is granted to the
        # sources whose likelihood it serves. They are deliberately absent from
        # this allowance so that a regression fails rather than being tolerated
        # by a list that still names them.
        #
        # Arrived with the attachment axis; no source declares a boundary_output
        # channel yet, so nothing has evidence about it. Unreachable is the
        # honest state, not an oversight -- and it is recorded rather than
        # granted, which is the difference this file is about.
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

    keys = _all_architecture_keys()
    dead: dict[str, list[str]] = {}
    for card, pats in _grant_patterns().items():
        empty = [p for p in pats if p and not any(fnmatch.fnmatch(k, p) for k in keys)]
        if empty:
            dead[card] = empty

    assert dead == {}, (
        f"these grant patterns name no parameter in ANY architecture this repo "
        f"builds: {dead}. A pattern here grants exactly nothing while reading in "
        "the card as though the source trains that module -- which is how run 2 "
        "trained 11.3% of its parameters and shipped. Fix the pattern, or delete "
        "it if the source genuinely should not train that module."
    )
