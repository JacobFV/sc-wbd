"""`configs/run4/scwbd-004.yaml` must declare what run 4 is for.

Run 4 exists to do two things run 3 could not: integrate the haemodynamic ODE on
the measured path (ISSUE-008) and fit a person effect (run 3 shipped
`individualization: {"applied": false}`). Both are config-level facts before
they are code-level ones -- `bold_predict_frames` and `bold_every` bound the
first, and `individualize` on a stage is the only thing that constructs the
second.

The stage-glob checks mirror `test_stage_permissions_reach_the_model`, which is
pinned to run 3. A dead glob is an empty permission set rather than an error,
so the check has to be repeated for each run config rather than assumed to
travel with the code.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
RUN4 = REPO / "configs/run4/scwbd-004.yaml"

pytestmark = pytest.mark.skipif(not RUN4.is_file(), reason="run-4 config absent")

#: Every architecture this repo builds. `ci-smoke` is here because it is the
#: ONLY checkpoint on disk carrying an `individualizer` payload -- no full run
#: has ever constructed one, which is the fact run 4 exists to change. Without
#: it the dead-glob check reports `individualizer.*` as naming nothing, which is
#: true of run 3's files and false of the module.
_ARCHITECTURES = (
    "checkpoints/scwbd-003/last.pt",
    "checkpoints/scwbd-003-smoke/last.pt",
    "checkpoints/scwbd-002-pilot/stage_T1_measured_founding.pt",
    "checkpoints/ci-smoke/last.pt",
)
_REGIONAL = ("family_local", "family_residual", "family_readout")

#: The one stage that may not reach the regional model, and why. Named rather
#: than inferred from a property of the stage, so adding a second frozen stage
#: is a visible edit.
_FROZEN_POPULATION_STAGE = "T6_individual"


def _cfg():
    from scwbd.foundation.config import load_config

    return load_config(RUN4)


def _stages() -> list[tuple[str, dict]]:
    cfg = _cfg()
    return [
        (s.name, ((s.extra or {}).get("curriculum") or {}).get("tier_permissions") or {})
        for s in cfg.train.stages
    ]


def _keys() -> list[str]:
    import torch

    out: list[str] = []
    for rel in _ARCHITECTURES:
        f = REPO / rel
        if not f.is_file():
            continue
        ck = torch.load(f, map_location="cpu", weights_only=False)
        out += list((ck.get("model") or {}).keys())
        for c in ("posterior", "individualizer", "tms_drive"):
            sub = ck.get(c)
            if isinstance(sub, dict):
                out += [f"{c}.{k}" for k in sub]
    return sorted(set(out))


# ======================================================================
# ISSUE-008's two cost fields
# ======================================================================
def test_the_bold_horizon_and_duty_cycle_are_declared() -> None:
    """Both are REDUCTIONS in what the fMRI likelihood sees, so both are stated.

    Left at their dataclass defaults they would be a decision nobody made. The
    values here are what the measured step time and peak reserve in
    reports/RUN4.md admit.
    """
    raw = yaml.safe_load(RUN4.read_text())
    assert "bold_predict_frames" in raw["model"]
    assert "bold_every" in raw["model"]
    m = _cfg().model
    assert m.bold_predict_frames >= 1
    assert m.bold_every >= 1
    # 8 target frames per window; predicting fewer is the compromise ISSUE-008
    # records, and predicting more than the window holds is a config error.
    assert m.bold_predict_frames <= 8


def test_the_measured_cost_is_written_into_the_config() -> None:
    """A step time and a peak reserve, not an estimate.

    Run 3's `batch: 8` carries its measurement in a comment beside it and that
    is why nobody re-derived it. Run 4's BOLD rollout is 500 neural steps where
    run 3's was 8, so the number changed and the comment has to have changed
    with it.
    """
    text = RUN4.read_text()
    assert "s/step" in text, "no measured step time in the config"
    assert "GB" in text and "reserved" in text, "no measured peak reserve in the config"
    assert "MEASURED_BLOCK" not in text and "BATCH_BLOCK" not in text, (
        "a placeholder survived into the config: the cost block was never filled "
        "in with the measurement"
    )


# ======================================================================
# individualisation
# ======================================================================
def test_exactly_one_stage_fits_a_person_effect() -> None:
    from scwbd.foundation.curriculum_admission import stage_admission

    cfg = _cfg()
    named = [
        s.name
        for s in cfg.train.stages
        if stage_admission(s, cards_dir=cfg.mixture_cards, strict=False).individualize
    ]
    assert named == [_FROZEN_POPULATION_STAGE], (
        f"stages declaring `individualize`: {named}. Run 3 had none, which is why "
        "`evaluation_run3.json` records 'no individualizer on the trainer' after a "
        "completed run."
    )


def test_the_individualisation_stage_grants_the_person_effect() -> None:
    for name, tp in _stages():
        if name != _FROZEN_POPULATION_STAGE:
            continue
        globs = [str(g).split("#")[0].strip() for gs in tp.values() for g in gs]
        assert "individualizer.*" in globs, (
            "the effective permission is the intersection of the card pattern and "
            "the stage glob. Without this the cards' grant is intersected with "
            "nothing and the person effect cannot receive a gradient."
        )
        assert not any(g.startswith(m) for g in globs for m in _REGIONAL), (
            "the individualisation stage trains the person effect against FROZEN "
            "population weights. Letting the dynamics move here answers a "
            "different question and makes the held-out comparison against the "
            "previous stage's checkpoint incoherent."
        )


def test_the_person_effect_is_granted_nowhere_else() -> None:
    granted = [
        name
        for name, tp in _stages()
        if any("individualizer" in str(g) for gs in tp.values() for g in gs)
    ]
    assert granted == [_FROZEN_POPULATION_STAGE], (
        f"`individualizer.*` is granted by {granted}. A stage that fits person "
        "effects while the population weights are still moving cannot be read as "
        "either a population result or an individualisation one."
    )


# ======================================================================
# the run-3 checks, repeated against run 4's own globs
# ======================================================================
def test_no_stage_permission_glob_is_dead() -> None:
    keys = _keys()
    if not keys:
        pytest.skip("no architecture checkpoint on disk to match globs against")
    dead: dict[str, list[str]] = {}
    for name, tp in _stages():
        for tier, globs in tp.items():
            for g in globs:
                g = str(g).split("#")[0].strip()
                if g and not any(fnmatch.fnmatch(k, g) for k in keys):
                    dead.setdefault(name, []).append(f"tier{tier}:{g}")
    assert dead == {}, (
        f"these run-4 stage globs name no parameter in any architecture this repo "
        f"builds: {dead}. An unmatched glob is an empty permission set, not an "
        "error, and the loss still falls."
    )


def test_every_stage_but_the_individualisation_one_reaches_the_regional_model() -> None:
    keys = _keys()
    if not keys:
        pytest.skip("no architecture checkpoint on disk")
    missing: dict[str, list[str]] = {}
    for name, tp in _stages():
        if name == _FROZEN_POPULATION_STAGE:
            continue
        allow = [str(g).split("#")[0].strip() for gs in tp.values() for g in gs]
        for mod in _REGIONAL:
            mod_keys = [k for k in keys if k.startswith(mod + ".")]
            if not mod_keys:
                continue
            if not any(fnmatch.fnmatch(k, g) for g in allow for k in mod_keys):
                missing.setdefault(name, []).append(mod)
    assert missing == {}, f"stage(s) cannot reach the regional model: {missing}"


def test_the_measured_return_stage_still_owns_the_pulse() -> None:
    granted = [
        name
        for name, tp in _stages()
        if any("tms_drive" in str(g) for gs in tp.values() for g in gs)
    ]
    assert granted == ["T5_measured_return"]


def test_the_run_writes_to_its_own_name_and_directory() -> None:
    """`--out` moves checkpoints, not logs; logs are keyed by `train.run_name`."""
    t = _cfg().train
    assert t.run_name == "scwbd-004"
    assert t.out_dir == "checkpoints/scwbd-004"
    assert t.out_dir != "checkpoints/scwbd-003"
