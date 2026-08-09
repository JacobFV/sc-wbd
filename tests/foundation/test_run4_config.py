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


def test_the_config_is_sized_by_a_measurement_that_fits() -> None:
    """The config's numbers must match an arm that was measured and did not OOM.

    Written this way after the first version failed its own mutation test. That
    version grepped the file for "s/step" and "GB", and deleting a row of the
    cost table left both behind -- a guard that cannot be made to fail on the
    regression it names is decorative, which is the category
    `reports/decorative_guards.md` exists for.

    So it checks the property instead: whatever `bold_predict_frames` and
    `batch` the config declares, some recorded arm ran that configuration, and
    that arm's peak reserve is under this config's own cap. Run 3's `batch: 8`
    was a measured maximum on a BOLD path that rolled 8 neural steps; run 4's
    rolls 250, and the 2-frame arm exceeded the cap on its first step.
    """
    import json

    m, d, t = _cfg().model, _cfg().data, _cfg().train
    arms = sorted((REPO / "reports/run4_cost").glob("cost_*.json"))
    assert arms, (
        "no cost measurement on disk. The BOLD rollout is 250 neural steps per "
        "frame against run 3's 8 for the whole window, so run 3's batch cannot "
        "be carried over on an estimate."
    )
    recorded = [json.loads(p.read_text()) for p in arms]
    match = [
        r
        for r in recorded
        if r["bold_predict_frames"] == m.bold_predict_frames
        and r["batch"] == d.batch
        and r["bold_every"] == m.bold_every
    ]
    assert match, (
        f"no recorded arm ran bold_predict_frames={m.bold_predict_frames}, "
        f"batch={d.batch}, bold_every={m.bold_every}. Measured arms: "
        + str([(r["bold_predict_frames"], r["batch"], r["bold_every"]) for r in recorded])
    )
    worst = max(r["peak_cuda_reserved_gb"] for r in match)
    assert worst < t.cuda_reserve_gb, (
        f"the measured peak reserve for this configuration is {worst:.2f} GB "
        f"against a {t.cuda_reserve_gb} GB cap. Raising the cap does not buy "
        "room -- it lets the caching allocator grow toward the OOM that has "
        "taken this machine down twice. Reduce bold_predict_frames or batch."
    )
    # The measurement has to be IN the config too, not only on disk: run 3's
    # batch survived three runs unquestioned because its number sat beside it.
    text = RUN4.read_text()
    assert f"{worst:.2f} GB" in text, (
        f"the measured peak reserve {worst:.2f} GB does not appear in the config. "
        "A reader deciding whether to change `batch` must not have to go find it."
    )
    assert "OUT OF MEMORY" in text, (
        "the config does not record that a larger horizon was tried and failed, "
        "so the next reader will try it again"
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
