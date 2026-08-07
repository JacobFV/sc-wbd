"""Stage admission must come from the config, not from the stage's name.

Every test here was **watched to fail** against ``scwbd/foundation/train.py`` at
``3400cee`` before the patch in ``configs/run2/patches/`` was written, and the
ones that did *not* fail are marked, because a test that passes in both worlds
measures nothing (``reports/decorative_guards.md``: three of eight tests in the
anatomy fix passed pre-fix because they asserted a property the synthetic prior
also satisfied).

Measured outcome, ``pytest tests/foundation/test_curriculum_admission.py -q``:
**7 failed, 4 passed** on master ``3400cee``/``508ae5e``; **11 passed** with
``configs/run2/patches/0001-*.patch`` and ``f816f2a`` applied.

===================================================== ========= ==========
test                                                   pre-patch post-patch
===================================================== ========= ==========
test_run_stage_has_no_stage_name_gates                 FAIL      pass
test_run_stage_consults_stage_admission                FAIL      pass
test_stage_sources_takes_an_admission                  FAIL      pass
test_stage_sources_excludes_unadmitted_sources         FAIL      pass
test_sim_losses_takes_an_admission                     FAIL      pass
test_anatomical_prior_is_not_gated_on_the_sim_batch    FAIL      pass
test_run2_config_admission_matches_its_declaration     FAIL†     pass
test_legacy_stage_sources_still_returns_every_source   pass      pass   <- regression guard
test_undeclared_new_stage_raises_instead_of_widening   pass*     pass   <- new module only
test_assert_region_count_refuses_a_mismatch            pass*     pass   <- new module only
test_name_gate_inventory_is_documented                 pass*     pass   <- inventory guard
===================================================== ========= ==========

``pass*`` = exercises only ``scwbd.foundation.curriculum_admission``, which is
new, so it cannot discriminate the patch.  Kept because it discriminates
*something* -- whether the replacement decision is correct -- but it is not
evidence that the trainer uses it.

``FAIL†`` = **failed for a different reason than predicted**, and that is worth
recording rather than counting as a win.  It was expected to fail on the
admission decision; it actually failed at ``load_config``, with
``KeyError: unknown config key 'anatomy_force_fallback' for TrainConfig`` --
because ``f816f2a`` is not on master either.  So this row discriminates the
*anatomy* precondition, not the admission patch, and it would keep passing if
the admission patch were reverted while ``f816f2a`` stayed.  Six rows, not
seven, are evidence about this patch.

**Stated limit, the same one ``scwbd.curriculum.legacy`` declares:** the source
assertions establish what ``run_stage`` *says*, not what a running process did.
Only ``test_stage_sources_excludes_unadmitted_sources`` executes production code,
and it executes one method.  A full behavioural check needs a corpus and a GPU.
"""

from __future__ import annotations

import inspect
import types

import pytest

from scwbd.foundation.config import StageConfig, load_config
from scwbd.foundation.curriculum_admission import (
    NAME_GATES,
    StageAdmission,
    UndeclaredStage,
    assert_region_count,
    stage_admission,
)
from scwbd.foundation.mixture import SourceSpec
from scwbd.foundation.train import STAGE_PERMISSIONS, FoundationTrainer

RUN2 = "configs/run2/scwbd-001.yaml"
CARDS = "configs/curriculum/source_cards"

#: These tests are RED ON PURPOSE until run 2's preconditions land, and they are
#: marked so that fact is deselectable rather than ambient::
#:
#:     pytest tests/foundation -m "not run2_pending"    # today's green suite
#:     pytest tests/foundation -m run2_pending          # 7 failed, 4 passed
#:
#: Deselecting is for other people's CI, not for run 2's owner. The marker comes
#: off when `configs/run2/patches/0001-*.patch` and `f816f2a` are applied, and
#: the removal is what proves the patch did something -- a suite that goes green
#: because a marker was left on is the register's absence variant with extra
#: steps.
pytestmark = pytest.mark.run2_pending


# ======================================================================
# the discriminating tests: these fail against the unpatched trainer
# ======================================================================
def test_run_stage_has_no_stage_name_gates() -> None:
    """``run_stage`` must not decide admission by comparing ``stage.name``.

    Pre-patch this finds ``stage.name != "V_individual"`` (simulated sources)
    and ``stage.name in ("III_sliced", "IV_assembly", "V_individual")``
    (measured sources), which is why no config edit can reorder the curriculum.
    """
    src = inspect.getsource(FoundationTrainer.run_stage)
    offenders = [
        ln.strip()
        for ln in src.splitlines()
        if "stage.name" in ln
        and any(op in ln for op in ("==", "!=", " in "))
        and not ln.strip().startswith("#")
    ]
    assert not offenders, (
        "FoundationTrainer.run_stage still decides behaviour from the stage's NAME:\n  "
        + "\n  ".join(offenders)
        + "\nAdmission must come from the config (extra.curriculum) via "
        "scwbd.foundation.curriculum_admission.stage_admission."
    )


def test_run_stage_consults_stage_admission() -> None:
    src = inspect.getsource(FoundationTrainer.run_stage)
    assert "stage_admission(" in src, (
        "run_stage does not call stage_admission(); removing the name gates without "
        "putting the config in their place would leave admission undefined."
    )


def test_stage_sources_takes_an_admission() -> None:
    """The stage allowlist must be config-driven, not ``STAGE_PERMISSIONS[name]``.

    The pre-patch default is the dangerous one: ``.get(name, ("*",))`` grants
    **everything** to a stage name it does not recognise, so run 2's renamed
    stages would each have silently widened to the union of the cards' own
    permissions.
    """
    params = inspect.signature(FoundationTrainer.stage_sources).parameters
    assert "admission" in params, (
        "stage_sources still resolves its allowlist from STAGE_PERMISSIONS.get(stage.name, "
        f'("*",)). For a run-2 stage name that default is {STAGE_PERMISSIONS.get("T1_measured_founding", ("*",))!r} '
        "-- no restriction at all."
    )


def test_stage_sources_excludes_unadmitted_sources() -> None:
    """Production code, executed: an unadmitted source must not reach the mixture."""
    sources = {
        "eegmmidb_real": SourceSpec(id="eegmmidb_real", role="likelihood", gradient_permission=("local.*", "eeg.*")),
        "sim_wholebrain": SourceSpec(id="sim_wholebrain", role="prior", is_simulated=True, losses=("prior",), gradient_permission=("local.*",)),
        "anatomical_prior": SourceSpec(id="anatomical_prior", role="prior", losses=("prior",), gradient_permission=("coupling.gain_*",)),
    }
    trainer = types.SimpleNamespace(sources=sources)
    stage = StageConfig(name="T1_measured_founding", steps=1)
    admission = StageAdmission(
        stage="T1_measured_founding",
        admits=(1,),
        source_ids=("eegmmidb_real",),
        tier_permissions={1: ("local.*", "eeg.*")},
    )
    out = FoundationTrainer.stage_sources(trainer, stage, admission)
    assert set(out) == {"eegmmidb_real"}, (
        "a stage admitting tier 1 alone still handed the simulated and prior sources to the "
        f"mixture: got {sorted(out)}"
    )


def test_sim_losses_takes_an_admission() -> None:
    """Boundary randomisation and ``with_hemo`` were both keyed on the stage name."""
    params = inspect.signature(FoundationTrainer.sim_losses).parameters
    assert "admission" in params
    src = inspect.getsource(FoundationTrainer.sim_losses)
    bad = [
        ln.strip()
        for ln in src.splitlines()
        if "stage.name" in ln and "admission" not in ln and not ln.strip().startswith("#")
    ]
    assert not bad, (
        "sim_losses still branches on the stage NAME with no config fallback:\n  " + "\n  ".join(bad)
    )


def test_anatomical_prior_is_not_gated_on_the_sim_batch() -> None:
    """Tier 3 must be able to contribute in a stage that does not admit tier 4.

    Pre-patch the ``anatomical_prior`` loss is composed inside ``sim_losses``, so
    it is only emitted on a step where the *simulated* loader ran.  Run 2's
    ``T3_population_prior`` admits tiers 1, 2 and 3 and not 4 -- on the unpatched
    trainer that stage would emit no tier-3 loss whatsoever, i.e. it would admit
    the population prior in name only.
    """
    assert hasattr(FoundationTrainer, "anat_losses"), (
        "no FoundationTrainer.anat_losses: the tier-3 term is still composed inside sim_losses"
    )
    assert "anatomical_prior" not in inspect.getsource(FoundationTrainer.sim_losses)
    assert "anat_losses(" in inspect.getsource(FoundationTrainer.run_stage)


# ======================================================================
# regression guard: the legacy path must keep working while run 1 resumes
# ======================================================================
def test_legacy_stage_sources_still_returns_every_source() -> None:
    """Passes in both worlds by design -- it guards what must NOT change.

    001-beta's config declares no ``extra.curriculum``, and its checkpoint may
    still be resumed.  With no admission the old behaviour must be untouched.
    """
    sources = {
        "eegmmidb_real": SourceSpec(id="eegmmidb_real", role="likelihood", gradient_permission=("local.*",)),
        "sim_wholebrain": SourceSpec(id="sim_wholebrain", role="prior", is_simulated=True, losses=("prior",), gradient_permission=("local.*",)),
    }
    trainer = types.SimpleNamespace(sources=sources)
    out = FoundationTrainer.stage_sources(trainer, StageConfig(name="I_regional", steps=1))
    assert set(out) == set(sources)


# ======================================================================
# the replacement decision itself (new module only -- cannot discriminate)
# ======================================================================
def test_undeclared_new_stage_raises_instead_of_widening() -> None:
    with pytest.raises(UndeclaredStage):
        stage_admission(StageConfig(name="T9_brand_new", steps=10), cards_dir=CARDS)
    # ...and a run-1 name still resolves, non-strictly, to run 1's behaviour
    a = stage_admission(StageConfig(name="V_individual", steps=10), cards_dir=CARDS, strict=False)
    assert a.individualize is True
    # `legacy:` or `frozen:run1@...` -- the property is that an INHERITED
    # admission is distinguishable from a declared one (`config:`), not which of
    # the two inherited spellings it uses. The fallback now serves the frozen
    # run-1 record, which also carries the sources the three behaviour booleans
    # never had.
    assert a.provenance.startswith(("legacy:", "frozen:run1@")), a.provenance
    assert not a.provenance.startswith("config:")


def test_run2_config_admission_matches_its_declaration() -> None:
    cfg = load_config(RUN2)
    got = {
        s.name: stage_admission(s, cards_dir=CARDS).source_ids
        for s in cfg.train.stages
        if s.enabled and s.steps > 0
    }
    # ds002336_real joined tier 1 on 2026-08-06, once the parcel-space BOLD path
    # was complete end to end (registration run, 55/55 runs cached, coverage
    # invariant holding on every artifact). The expectations are widened rather
    # than loosened: each still names the exact source set, so the next addition
    # is refused the same way this one was.
    MEASURED = ("ds002336_real", "eegmmidb_real")
    assert got["T1_measured_founding"] == MEASURED
    assert got["T2_boundary_calibration"] == (*MEASURED, "montage_calibration")
    assert got["T3_population_prior"] == ("anatomical_prior", *MEASURED, "montage_calibration")
    assert "sim_wholebrain" in got["T4_simulator_extension"]
    assert got["T1_individualisation"] == MEASURED
    # the simulator is admitted LAST among the training tiers, and only there
    assert [n for n, ids in got.items() if "sim_wholebrain" in ids] == ["T4_simulator_extension"]
    # ...and the individualisation flag survives the rename
    stages = {s.name: s for s in cfg.train.stages}
    assert stage_admission(stages["T1_individualisation"], cards_dir=CARDS).individualize is True
    assert stage_admission(stages["T1_measured_founding"], cards_dir=CARDS).boundary_randomisation is True
    assert stage_admission(stages["T4_simulator_extension"], cards_dir=CARDS).with_hemo is False


def test_assert_region_count_refuses_a_mismatch() -> None:
    anat = types.SimpleNamespace(n_regions=454, provenance="synthetic_fallback", is_biological=lambda: False)
    with pytest.raises(ValueError, match="414"):
        assert_region_count(414, anat)
    assert_region_count(454, anat)  # agreeing is silent


def test_name_gate_inventory_is_documented() -> None:
    """Every gate the register records is either gone or config-backed.

    Not a discriminator -- it passes pre-patch too, because pre-patch every gate
    in :data:`NAME_GATES` is present and the inventory is correct.  Its job is to
    stop the inventory going stale.
    """
    assert len(NAME_GATES) == 6
    perm = [g for g in NAME_GATES if "STAGE_PERMISSIONS" in g[1]]
    assert perm and "PERMISSIVE" in perm[0][3], (
        "the register must keep recording that the stage-allowlist default is the permissive one"
    )
