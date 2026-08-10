"""Check #1, one layer up: the STAGE allowlist must name real parameters too.

``test_card_patterns_reach_the_model`` validates each source card's
``gradient_permission``. It does not look at a run config's per-stage
``extra.curriculum.tier_permissions``, and the effective permission is the
**intersection** of the two:

    FoundationTrainer.stage_sources  ->  narrower of (card pattern, stage glob)

So a stage glob that matches nothing narrows the intersection to nothing for
whatever it was meant to cover, and no card-level check can see it. That is
run 2's defect with a second name: an unmatched glob is an empty permission set,
not an error, and the loss still falls.

Found in ``configs/run3/scwbd-003.yaml`` before it mattered: every stage granted
``local.*``, ``residual.*``, ``readout.*`` and ``log_dt_scale``, which name
nothing in the family-padded arm this run builds. They were harmless there --
the ``family_*`` globs were granted beside them -- but a run config carrying
four dead globs teaches a reader that dead globs are normal, and that is how
run 2 shipped.

A **card** may legitimately grant both arms' namings, because one card is shared
across runs and arms. A **run config** is one run with one arm, so every glob in
it should mean something.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import pytest

from ._runs import (
    known_parameter_names,
    parametrize_runs,
    raw_stages as _raw_stages,
    stages as _stages,
)

REPO = Path(__file__).resolve().parents[2]

#: Modules the regional model lives in. A stage that trains the dynamics at all
#: must reach these, or it is run 2.
_REGIONAL = ("family_local", "family_residual", "family_readout")


def _freezes_population(run, stage_name: str) -> bool:
    """Does this stage declare that it fits a person effect over frozen weights?

    Reads ``extra.curriculum.individualize``, which ``FoundationTrainer`` uses to
    decide whether to construct an individualizer at all. Keyed on a field the
    trainer consumes so the exemption cannot drift away from what the run does.
    """
    for s in _raw_stages(run):
        if s.get("name") != stage_name:
            continue
        return bool(((s.get("extra") or {}).get("curriculum") or {}).get("individualize"))
    return False


def _keys(run) -> list[str]:
    """Parameter names of the architecture THIS run's config builds.

    Was a fixed list of run-3 checkpoints. A run that has not launched has no
    checkpoint, so that spelling could only ever grade a finished run -- and it
    graded run 4 against run 3's architecture, in which ``msg_proj`` exists and
    in run 4's it does not.
    """
    return list(known_parameter_names(run))


@parametrize_runs
def test_no_stage_permission_glob_is_dead(run) -> None:
    keys = _keys(run)
    assert keys, f"{run.run_id}: could not enumerate an architecture to match globs against"
    dead: dict[str, list[str]] = {}
    for name, tp in _stages(run):
        for tier, globs in tp.items():
            for g in globs:
                g = str(g).split("#")[0].strip()
                if g and not any(fnmatch.fnmatch(k, g) for k in keys):
                    dead.setdefault(name, []).append(f"tier{tier}:{g}")
    assert dead == {}, (
        f"{run.run_id}: these stage globs name no parameter in the architecture "
        f"{run.config.name} builds: {dead}. "
        "The effective permission is the intersection of the card pattern and "
        "the stage glob, so a dead stage glob empties the permission for "
        "whatever it was meant to cover, and no card-level check can see it. "
        "Delete it, or fix the spelling to the arm this run actually builds."
    )


@parametrize_runs
def test_every_stage_reaches_the_regional_model(run) -> None:
    """A stage that trains the dynamics must be able to name them.

    "Every stage" was true of run 3, whose five stages all trained the dynamics,
    and the assertion inherited that as if it were an invariant. Run 4 adds
    ``T6_individual``, which freezes the population weights on purpose: the
    question it asks is whether a person effect over theta predicts a held-out
    night GIVEN the population model, so letting the dynamics move would answer
    a different question and make the comparison against
    ``stage_T5_measured_return.pt`` incoherent.

    The exemption is keyed on ``individualize``, which the trainer READS to
    construct the individualizer -- not on a flag added here for the test to
    find, which would be a decorative config key (reports/decorative_guards.md).
    A stage that reaches nothing regional for any *other* reason still fails,
    which is the run-2 defect this guard exists for.
    """
    keys = _keys(run)
    assert keys, f"{run.run_id}: could not enumerate an architecture"
    missing: dict[str, list[str]] = {}
    exempt: list[str] = []
    for name, tp in _stages(run):
        if _freezes_population(run, name):
            exempt.append(name)
            continue
        allow = [str(g).split("#")[0].strip() for gs in tp.values() for g in gs]
        for mod in _REGIONAL:
            mod_keys = [k for k in keys if k.startswith(mod + ".")]
            if not mod_keys:
                continue
            if not any(fnmatch.fnmatch(k, g) for g in allow for k in mod_keys):
                missing.setdefault(name, []).append(mod)
    assert missing == {}, (
        f"{run.run_id}: stage(s) cannot reach the regional model: {missing}. A stage that "
        "trains the dynamics and cannot name family_local, family_residual or "
        "family_readout leaves them at their initialisation while they sit on the "
        "forward path -- that is run 2, in which 88.8% of parameters went untrained "
        "while the loss fell. If the stage freezes the population on purpose, it must "
        f"declare `individualize: true` and be scored accordingly. Exempt here: {exempt}."
    )


@parametrize_runs
def test_a_frozen_population_stage_is_declared_not_inferred(run) -> None:
    """The exemption above must not be reachable by omission.

    A stage is excused from reaching the regional model only when it declares
    ``individualize: true``. This asserts the converse of what the exemption
    grants: every exempt stage really does grant something -- the person effect
    it exists to fit -- so "exempt" can never come to mean "grants nothing at
    all", which is the state the guard is looking for in the first place.
    """
    for name, tp in _stages(run):
        if not _freezes_population(run, name):
            continue
        allow = [str(g).split("#")[0].strip() for gs in tp.values() for g in gs]
        assert allow, f"{run.run_id}/{name}: declares individualize but grants nothing at all"
        assert any(g.startswith("individualizer") for g in allow), (
            f"{run.run_id}/{name}: declares `individualize: true` and is therefore excused "
            f"from reaching the regional model, but grants {allow} -- nothing matching "
            "`individualizer.*`. A stage that freezes the population and also cannot name "
            "the person effect trains nothing, and would report a fitted individualizer "
            "and an absent one identically."
        )


@parametrize_runs
def test_the_measured_return_stage_owns_the_pulse(run) -> None:
    """`tms_drive.*` is granted in T5 and nowhere else, on purpose.

    The drive's amplitude and its profile over motor parcels are learned against
    the measured evoked response. A simulator stage reaching them would fit the
    pulse to synthetic dynamics, which is the one thing the measured-perturbation
    source exists to avoid.

    Run 2 predates the measured-perturbation source and grants ``tms_drive.*``
    nowhere, which is consistent rather than a defect -- so the invariant is
    "at most one stage, and if any, the measured-return one", not "exactly T5".
    Asserting the latter across runs made the guard fail on a run that never had
    a pulse to own.
    """
    granted = [
        name
        for name, tp in _stages(run)
        if any("tms_drive" in str(g) for gs in tp.values() for g in gs)
    ]
    assert granted in ([], ["T5_measured_return"]), (
        f"{run.run_id}: tms_drive.* is granted by stages {granted}; expected either "
        "no stage (a run with no measured-perturbation source) or exactly "
        "['T5_measured_return']. A simulator stage reaching the drive would fit the "
        "pulse to synthetic dynamics."
    )
