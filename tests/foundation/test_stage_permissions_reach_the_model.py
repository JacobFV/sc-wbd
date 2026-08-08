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
import yaml

REPO = Path(__file__).resolve().parents[2]
RUN3 = REPO / "configs/run3/scwbd-003.yaml"

#: Architectures a run config may legitimately be written against. A glob is
#: dead only if it names nothing in ANY of them.
_ARCHITECTURES = (
    "checkpoints/scwbd-003/last.pt",
    "checkpoints/scwbd-003-smoke/last.pt",
)

#: Modules the regional model lives in. A stage that trains the dynamics at all
#: must reach these, or it is run 2.
_REGIONAL = ("family_local", "family_residual", "family_readout")


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


def _stages() -> list[tuple[str, dict]]:
    cfg = yaml.safe_load(RUN3.read_text())
    out = []
    for s in cfg["train"]["stages"]:
        tp = ((s.get("extra") or {}).get("curriculum") or {}).get("tier_permissions") or {}
        out.append((s["name"], tp))
    return out


@pytest.mark.skipif(not RUN3.is_file(), reason="run-3 config absent")
def test_no_stage_permission_glob_is_dead() -> None:
    keys = _keys()
    if not keys:
        pytest.skip("no run-3 architecture checkpoint on disk to match globs against")
    dead: dict[str, list[str]] = {}
    for name, tp in _stages():
        for tier, globs in tp.items():
            for g in globs:
                g = str(g).split("#")[0].strip()
                if g and not any(fnmatch.fnmatch(k, g) for k in keys):
                    dead.setdefault(name, []).append(f"tier{tier}:{g}")
    assert dead == {}, (
        f"these stage globs name no parameter in any run-3 architecture: {dead}. "
        "The effective permission is the intersection of the card pattern and "
        "the stage glob, so a dead stage glob empties the permission for "
        "whatever it was meant to cover, and no card-level check can see it. "
        "Delete it, or fix the spelling to the arm this run actually builds."
    )


@pytest.mark.skipif(not RUN3.is_file(), reason="run-3 config absent")
def test_every_stage_reaches_the_regional_model() -> None:
    """The forward guard: a stage that trains nothing regional is run 2."""
    keys = _keys()
    if not keys:
        pytest.skip("no run-3 architecture checkpoint on disk")
    missing: dict[str, list[str]] = {}
    for name, tp in _stages():
        allow = [str(g).split("#")[0].strip() for gs in tp.values() for g in gs]
        for mod in _REGIONAL:
            mod_keys = [k for k in keys if k.startswith(mod + ".")]
            if not mod_keys:
                continue
            if not any(fnmatch.fnmatch(k, g) for g in allow for k in mod_keys):
                missing.setdefault(name, []).append(mod)
    assert missing == {}, (
        f"stage(s) cannot reach the regional model: {missing}. Every stage in "
        "this run trains the dynamics; one that cannot name family_local, "
        "family_residual or family_readout would leave them at their "
        "initialisation while they sit on the forward path."
    )


@pytest.mark.skipif(not RUN3.is_file(), reason="run-3 config absent")
def test_the_measured_return_stage_owns_the_pulse() -> None:
    """`tms_drive.*` is granted in T5 and nowhere else, on purpose.

    The drive's amplitude and its profile over motor parcels are learned against
    the measured evoked response. A simulator stage reaching them would fit the
    pulse to synthetic dynamics, which is the one thing the measured-perturbation
    source exists to avoid.
    """
    granted = [
        name
        for name, tp in _stages()
        if any("tms_drive" in str(g) for gs in tp.values() for g in gs)
    ]
    assert granted == ["T5_measured_return"], (
        f"tms_drive.* is granted by stages {granted}; expected exactly "
        "['T5_measured_return']"
    )
