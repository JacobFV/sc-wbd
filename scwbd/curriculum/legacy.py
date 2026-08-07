"""Reconstruct what a *pre-curriculum* config actually admits, from the trainer.

``configs/scwbd_001_beta.yaml`` names five stages and says nothing about which
sources any of them sees.  A validator that guessed the admission from stage
names would be checking its own guess.  So this module reads the admission out of
:func:`scwbd.foundation.train.FoundationTrainer.run_stage` --- the function that
actually decides it --- with :func:`inspect.getsource`, and **refuses** rather
than defaults if the gates it expects are not there.

That refusal matters more than the parse.  ``reports/decorative_guards.md``
records three separate cases of a check that exercised a different path from
production and passed while production failed.  A hard-coded copy of the gates
would be exactly that: correct on the day it was written and silently stale
afterwards.  If someone rewrites ``run_stage``, this raises
:class:`GateNotFound` and the validator reports "I could not establish what this
config admits", which is the honest answer and is not the same as "it is fine".

Static reading has a stated limit: it establishes what the source says, not what
a running process did.  It is the strongest check available without launching a
job, and the report says so where it is used.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any

__all__ = ["GateNotFound", "LegacyAdmission", "StageAdmission", "reconstruct_stage_admission"]


class GateNotFound(RuntimeError):
    """The trainer no longer contains a gate this reconstruction depends on."""


#: The loss keys ``run_stage`` composes, and which loader gates each one.  These
#: are the *source ids* the mixture is keyed by, read from the same functions.
_SIM_LOSS_KEYS = ("sim_wholebrain", "anatomical_prior")
_REAL_LOSS_KEYS = ("eegmmidb_real",)

_SIM_GATE = re.compile(r"if\s+stage\.name\s*!=\s*(['\"])(?P<excluded>[A-Za-z0-9_]+)\1\s*:")
_REAL_GATE = re.compile(
    r"if\s+self\.real_train\s+is\s+not\s+None\s+and\s+stage\.name\s+in\s*\((?P<names>[^)]*)\)"
)
_STRINGS = re.compile(r"['\"]([A-Za-z0-9_]+)['\"]")


@dataclass
class StageAdmission:
    admits: tuple[int, ...]
    tier_permissions: dict[int, tuple[str, ...]]
    objective: tuple[str, ...]
    absence: tuple[dict[str, Any], ...]
    provenance: str
    source_ids: tuple[str, ...] = ()


@dataclass
class LegacyAdmission:
    """Admission for every stage name the trainer knows about."""

    by_stage: dict[str, StageAdmission]
    sim_excluded_stage: str
    real_admitted_stages: tuple[str, ...]
    stage_permissions: dict[str, tuple[str, ...]]
    trainer_sha_note: str = ""
    unknown: StageAdmission | None = None

    def for_stage(self, name: str) -> StageAdmission:
        if name in self.by_stage:
            return self.by_stage[name]
        return StageAdmission(
            admits=(),
            tier_permissions={},
            objective=(),
            absence=(
                {
                    "tier": None,
                    "reason": (
                        f"stage {name!r} is not named by any gate in FoundationTrainer.run_stage, "
                        "so no admission can be established for it from the trainer's source"
                    ),
                },
            ),
            provenance="reconstructed:unknown_stage",
        )


def reconstruct_stage_admission(*, card_tiers: dict[str, int] | None = None) -> LegacyAdmission:
    """Read admission gates out of the trainer and project them onto tiers.

    ``card_tiers`` maps source id -> tier.  When omitted it is derived from
    ``configs/source_cards`` --- the directory ``configs/scwbd_001_beta.yaml``
    points ``mixture_cards`` at.
    """
    from scwbd.foundation.train import STAGE_PERMISSIONS, FoundationTrainer

    src = inspect.getsource(FoundationTrainer.run_stage)

    m_sim = _SIM_GATE.search(src)
    if m_sim is None:
        raise GateNotFound(
            "FoundationTrainer.run_stage no longer contains the simulated-source gate "
            "`if stage.name != '<stage>':`. The admission of a config without an explicit "
            "curriculum block cannot be established; refusing to assume it.\n\n"
            "This is expected as of 2026-08-06 and is not a regression to repair here. "
            "run_stage now reads each stage's DECLARED admission instead of matching its "
            "name (ARCHITECTURE.md RL-14), so the gates this module reads are gone on "
            "purpose. Run 1's behaviour survives only as the hardcoded _LEGACY_FLAGS "
            "table in scwbd.foundation.curriculum_admission.\n\n"
            "What that costs, stated because this module exists to prevent exactly it: "
            "the table is a transcription of gates that no longer exist, so nothing in "
            "this repository can now falsify it. The check that could have -- reading "
            "the gates out of the function that runs -- is what this refusal reports as "
            "unavailable. The cost is accepted because run 1 is finished and its configs "
            "are frozen: a LIVE curriculum must declare its admission, and only a dead "
            "one may inherit it from a table."
        )
    sim_excluded = m_sim.group("excluded")

    m_real = _REAL_GATE.search(src)
    if m_real is None:
        raise GateNotFound(
            "FoundationTrainer.run_stage no longer contains the measured-source gate "
            "`if self.real_train is not None and stage.name in (...)`. Refusing to assume "
            "which stages see measured data."
        )
    real_stages = tuple(_STRINGS.findall(m_real.group("names")))
    if not real_stages:
        raise GateNotFound("the measured-source gate matched but names no stages")

    if card_tiers is None:
        from .tiers import load_mixture_cards, tier_of

        cards = load_mixture_cards("configs/source_cards")
        card_tiers = {}
        for sid, c in cards.items():
            a = tier_of(c)
            if a.tier is not None:
                card_tiers[sid] = a.tier

    by_stage: dict[str, StageAdmission] = {}
    all_stages = sorted(set(STAGE_PERMISSIONS) | set(real_stages) | {sim_excluded})
    for name in all_stages:
        source_ids: list[str] = []
        if name != sim_excluded:
            source_ids += [k for k in _SIM_LOSS_KEYS if k in card_tiers]
        if name in real_stages:
            source_ids += [k for k in _REAL_LOSS_KEYS if k in card_tiers]
        tiers = tuple(sorted({card_tiers[s] for s in source_ids}))
        perms = STAGE_PERMISSIONS.get(name, ("*",))
        absence: list[dict[str, Any]] = []
        if not source_ids:
            absence.append(
                {
                    "tier": None,
                    "reason": f"no loss key in run_stage is gated on for stage {name!r}",
                }
            )
        by_stage[name] = StageAdmission(
            admits=tiers,
            tier_permissions={t: perms for t in tiers},
            objective=("composite: see StageConfig lambda_* weights",),
            absence=tuple(absence),
            provenance=(
                "reconstructed from inspect.getsource(FoundationTrainer.run_stage) + "
                "STAGE_PERMISSIONS"
            ),
            source_ids=tuple(source_ids),
        )

    return LegacyAdmission(
        by_stage=by_stage,
        sim_excluded_stage=sim_excluded,
        real_admitted_stages=real_stages,
        stage_permissions=dict(STAGE_PERMISSIONS),
        trainer_sha_note=(
            "gates read statically from the installed scwbd.foundation.train; this establishes "
            "what the source says, not what any particular process did"
        ),
    )
