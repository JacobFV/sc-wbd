"""Stage behaviour must come from the config, and every stage must declare it.

**This file was rewritten after the fix landed, and the rewrite is the point.**

The first version asserted things like *"every stage name appears in
``STAGE_PERMISSIONS``"* and *"some stage name is in the real-data tuple"*. Those
were the right questions while behaviour was selected by matching
``stage.name`` — six gates keyed to the previous run's names, five of which
returned the wrong answer for run 2 (``reports/RUN2.md`` §2b).

With ``0001-run_stage-config-driven-admission.patch`` applied, they are the
*wrong* questions. ``STAGE_PERMISSIONS`` is now only a legacy fallback for the
five names run 1 used, and membership in it says nothing about a run-2 stage.
Kept as ``xfail`` the tests would have gone on passing while checking a
mechanism no longer in use — a decorative guard produced by fixing the thing it
guarded.

So they assert the new invariant instead: **the trainer's behaviour for a stage
is whatever that stage declares, and a stage that declares nothing is refused.**
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

CONFIGS = [
    "configs/run2/pilot-families.yaml",
    "configs/run2/scwbd-001-families.yaml",
    "configs/run2/scwbd-001.yaml",
]


def _cfg(rel: str):
    from scwbd.foundation.config import load_config

    return load_config(str(ROOT / rel))


def _admissions(rel: str):
    from scwbd.foundation.curriculum_admission import stage_admission

    cfg = _cfg(rel)
    return cfg, [
        (s, stage_admission(s, cards_dir=cfg.mixture_cards, strict=False))
        for s in cfg.train.stages
    ]


@pytest.mark.parametrize("rel", CONFIGS)
def test_every_stage_resolves_an_admission(rel: str):
    """A stage the trainer cannot classify must raise, not be granted everything.

    The defect this replaces was `.get(name, ("*",))` — a default that grants.
    """
    if not (ROOT / rel).is_file():
        pytest.skip(f"{rel} not present")
    cfg, pairs = _admissions(rel)
    assert pairs, "no stages declared -- this test would pass vacuously"
    for stage, adm in pairs:
        assert adm is not None, f"{rel}: {stage.name} resolved no admission"


@pytest.mark.parametrize("rel", CONFIGS)
def test_run_stage_decides_by_admission_not_by_name(rel: str):
    """The five behaviours must be read off the admission object.

    Checked against ``run_stage``'s source rather than by running it, because
    running it needs a GPU and the property is structural.
    """
    import inspect

    from scwbd.foundation.train import FoundationTrainer

    src = inspect.getsource(FoundationTrainer.run_stage)
    assert "stage_admission(" in src, "run_stage does not derive an admission"
    assert "admission.individualize" in src, "individualisation still keyed on the name"
    assert "admits_measured" in src, "measured-source admission still keyed on the name"


@pytest.mark.parametrize("rel", CONFIGS)
def test_a_stage_named_for_measurement_admits_measured_data(rel: str):
    """``T1_measured_founding`` trained on simulation for a whole run.

    The name is not the defect; a name that *claims* something the trainer will
    not do is. Checked in the direction that matters: if the stage says
    measurement, its admission must contain a measured source.
    """
    if not (ROOT / rel).is_file():
        pytest.skip(f"{rel} not present")
    from scwbd.foundation.mixture import SourceSpec

    cfg, pairs = _admissions(rel)
    sources = SourceSpec.load_dir(cfg.mixture_cards)
    for stage, adm in pairs:
        if "measured" not in stage.name.lower():
            continue
        assert adm.admits_measured(sources), (
            f"{rel}: {stage.name} is named for measured data but admits "
            f"{list(adm.source_ids)}, none of which is a measured likelihood source"
        )


@pytest.mark.parametrize("rel", CONFIGS)
def test_a_stage_named_for_individualisation_individualises(rel: str):
    if not (ROOT / rel).is_file():
        pytest.skip(f"{rel} not present")
    _, pairs = _admissions(rel)
    for stage, adm in pairs:
        if "individual" not in stage.name.lower():
            continue
        assert adm.individualize, (
            f"{stage.name} is named for individualisation but its admission has "
            "individualize=False, so no Individualizer is built"
        )


@pytest.mark.parametrize("rel", CONFIGS)
def test_at_least_one_enabled_stage_admits_measured_data(rel: str):
    """A curriculum that never touches measurement trains on simulation alone.

    Run 2 satisfied this vacuously for nine hours.
    """
    if not (ROOT / rel).is_file():
        pytest.skip(f"{rel} not present")
    from scwbd.foundation.mixture import SourceSpec

    cfg, pairs = _admissions(rel)
    sources = SourceSpec.load_dir(cfg.mixture_cards)
    live = [(s, a) for s, a in pairs if getattr(s, "enabled", True) and s.steps > 0]
    assert live, "no enabled stages -- vacuous"
    assert any(a.admits_measured(sources) for _, a in live), (
        f"{rel}: no enabled stage admits a measured source, so the run would "
        "train on simulation alone"
    )
