"""A curriculum expressed as *(admitted tiers, gradient permissions, objective)*.

The 001-beta configuration expresses its curriculum as a **fixed stage list**:
five names, five step counts, five learning rates.  Which data each stage sees is
not in the file at all --- it is hard-coded in
``scwbd.foundation.train.FoundationTrainer.run_stage`` as a membership test on
the stage's *name*.  That is why the ordering error is not visible in the config
and could not be caught by reading it.

This module makes the three things a stage actually decides explicit and
machine-checkable:

``admits``
    which integrity tiers may contribute a gradient during the stage.
``tier_permissions``
    per tier, the globs that tier may update *in this stage*.  Intersected with
    (never added to) the source card's own ``A_k``, exactly as
    ``STAGE_PERMISSIONS`` already is --- a stage may restrict, never grant.
``objective``
    which loss families are summed.  Kept alongside admission because a tier
    that is admitted but contributes no loss family it is licensed for is
    admitted in name only.

``absence``
    what the stage *would* have admitted and could not.  A stage that declares a
    tier for which no live source exists writes that here.  Without it, "this
    tier contributed nothing because there is nothing on disk" and "this tier was
    never wired in" are the same silence (``reports/decorative_guards.md``, the
    absence variant).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

__all__ = [
    "StageCurriculum",
    "Curriculum",
    "FoundingExemption",
    "TierPolicy",
    "load_tier_policy",
    "admitted_source_ids",
    "stage_gradient_permission",
]


def admitted_source_ids(stage: "StageCurriculum", cards: Mapping[str, Any]) -> tuple[str, ...]:
    """Which sources a stage admits --- the replacement for the hard-coded gates.

    ``scwbd.foundation.train.FoundationTrainer.run_stage`` currently decides
    admission with two membership tests on the stage's *name*::

        if stage.name != "V_individual":                      # simulated sources
        if ... and stage.name in ("III_sliced", "IV_assembly", "V_individual"):

    That is why the integrity ordering cannot be corrected by editing a config,
    and why refusal ``X06`` fires on the corrected one.  This function is the
    mechanism that replaces both tests: it reads the stage's declared tiers and
    resolves them against the cards, so admission lives in the config where it
    can be checked.  The trainer change is one line per gate.

    Not called from ``scwbd.foundation`` yet --- that module is owned elsewhere
    and was mid-run when this was written.
    """
    from .tiers import tier_of

    out = []
    for sid, card in sorted(cards.items()):
        if not card.spec.enabled:
            continue
        a = tier_of(card)
        if a.tier and a.tier in stage.admits:
            out.append(sid)
    return tuple(out)


def stage_gradient_permission(
    stage: "StageCurriculum", card: Any, tier: int
) -> tuple[str, ...]:
    """The stage's mask for a tier, intersected with (never added to) the card's.

    Mirrors ``FoundationTrainer.stage_sources``: a stage may restrict what a
    source updates and may never grant a permission the card withheld.
    """
    import fnmatch

    declared = stage.permits(tier)
    if not declared:
        return tuple(card.spec.gradient_permission)
    return tuple(
        p
        for p in card.spec.gradient_permission
        if p == "*" or any(fnmatch.fnmatch(p.rstrip("*").rstrip("."), d.rstrip("*").rstrip(".")) or
                           fnmatch.fnmatch(d.rstrip("*").rstrip("."), p.rstrip("*").rstrip(".")) or
                           p.split(".")[0] == d.split(".")[0]
                           for d in declared)
    )


@dataclass
class FoundingExemption:
    """A named, recorded permission for a lower tier to found a parameter.

    Some parameters cannot be founded by measured evidence *in principle*: the
    amortised posterior maps observations to ground-truth ``theta``, and no
    measured recording carries a ``theta`` label.  Refusing such a parameter
    outright would delete a real capability; permitting it silently would let
    simulation found part of the representation with nothing in the artifact
    saying so.

    So it is permitted **and written down**.  The exemption is the record.
    """

    globs: tuple[str, ...]
    granted_to_tier: int
    reason: str
    #: What would remove the need for the exemption.
    dischargeable_by: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "globs": list(self.globs),
            "granted_to_tier": self.granted_to_tier,
            "reason": self.reason,
            "dischargeable_by": self.dischargeable_by,
        }


@dataclass
class TierPolicy:
    """Project-level ordering rules, held apart from any one run's config.

    Kept in ``configs/curriculum/tiers.yaml`` rather than inside a run config on
    purpose: a run may not redefine the ordering it is being checked against.
    """

    founding_tier: int = 1
    exemptions: tuple[FoundingExemption, ...] = ()
    #: Tiers whose absence from a run is acceptable without an ``absence`` record.
    #: Empty by design --- every absence writes something.
    silently_optional_tiers: tuple[int, ...] = ()
    notes: str = ""

    def exemption_for(self, glob: str, tier: int) -> FoundingExemption | None:
        for e in self.exemptions:
            if glob in e.globs and tier >= e.granted_to_tier:
                return e
        return None


def load_tier_policy(path: str | Path = "configs/curriculum/tiers.yaml") -> TierPolicy:
    d = yaml.safe_load(Path(path).read_text()) or {}
    pol = d.get("policy", {}) or {}
    return TierPolicy(
        founding_tier=int(pol.get("founding_tier", 1)),
        exemptions=tuple(
            FoundingExemption(
                globs=tuple(e["globs"]),
                granted_to_tier=int(e["granted_to_tier"]),
                reason=str(e["reason"]),
                dischargeable_by=str(e.get("dischargeable_by", "")),
            )
            for e in pol.get("founding_exemptions", []) or []
        ),
        silently_optional_tiers=tuple(pol.get("silently_optional_tiers", []) or []),
        notes=str(d.get("notes", "")),
    )


@dataclass
class StageCurriculum:
    name: str
    order: int
    steps: int
    #: thesis §6 model-scope stage this corresponds to (I_regional … V_individual).
    #: Scope and integrity are **orthogonal axes**; 001-beta conflated them.
    scope: str = ""
    admits: tuple[int, ...] = ()
    tier_permissions: dict[int, tuple[str, ...]] = field(default_factory=dict)
    objective: tuple[str, ...] = ()
    absence: tuple[dict[str, Any], ...] = ()
    #: False when the admission had to be reconstructed from the trainer's source
    #: rather than read from the config (see :mod:`scwbd.curriculum.legacy`).
    declared: bool = True
    provenance: str = "config"
    enabled: bool = True

    def permits(self, tier: int) -> tuple[str, ...]:
        return self.tier_permissions.get(tier, ())

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "order": self.order,
            "steps": self.steps,
            "scope": self.scope,
            "admits": list(self.admits),
            "tier_permissions": {k: list(v) for k, v in sorted(self.tier_permissions.items())},
            "objective": list(self.objective),
            "absence": [dict(a) for a in self.absence],
            "declared": self.declared,
            "provenance": self.provenance,
            "enabled": self.enabled,
        }


@dataclass
class Curriculum:
    run_name: str
    config_path: Path
    stages: tuple[StageCurriculum, ...]
    mixture_cards: str
    #: True when *every* stage declared its own admission.
    fully_declared: bool
    notes: str = ""

    # -- ordering queries -------------------------------------------------
    def first_admission(self, tier: int) -> StageCurriculum | None:
        for s in self.stages:
            if s.enabled and s.steps > 0 and tier in s.admits:
                return s
        return None

    def admitted_tiers(self) -> tuple[int, ...]:
        return tuple(sorted({t for s in self.stages if s.enabled and s.steps > 0 for t in s.admits}))

    def total_steps(self) -> int:
        return sum(s.steps for s in self.stages if s.enabled)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "config": str(self.config_path),
            "mixture_cards": self.mixture_cards,
            "fully_declared": self.fully_declared,
            "total_steps": self.total_steps(),
            "stages": [s.as_dict() for s in self.stages],
        }

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_config(cls, path: str | Path, *, legacy_ok: bool = True) -> "Curriculum":
        """Build from a foundation run config.

        Stages that carry ``extra.curriculum`` are read directly.  Stages that do
        not are reconstructed by :mod:`scwbd.curriculum.legacy` from the
        trainer's own source, and flagged ``declared=False`` --- a reconstruction
        is evidence about the code as it stands today, not a commitment the
        config made.
        """
        p = Path(path)
        payload = yaml.safe_load(p.read_text()) or {}
        if "base" in payload:
            base = yaml.safe_load((p.parent / payload["base"]).read_text()) or {}
            merged = dict(base)
            for k, v in payload.items():
                if k == "base":
                    continue
                merged[k] = v
            payload = merged

        train = payload.get("train", {}) or {}
        raw_stages: Sequence[Mapping[str, Any]] = train.get("stages", []) or []
        stages: list[StageCurriculum] = []
        need_legacy: list[int] = []
        for i, sd in enumerate(raw_stages):
            cur = ((sd.get("extra") or {}).get("curriculum")) or None
            name = str(sd.get("name", f"stage_{i}"))
            steps = int(sd.get("steps", 0))
            enabled = bool(sd.get("enabled", True))
            if cur is None:
                need_legacy.append(i)
                stages.append(
                    StageCurriculum(
                        name=name,
                        order=i,
                        steps=steps,
                        declared=False,
                        provenance="undeclared",
                        enabled=enabled,
                    )
                )
                continue
            stages.append(
                StageCurriculum(
                    name=name,
                    order=i,
                    steps=steps,
                    scope=str(cur.get("scope", "")),
                    admits=tuple(int(t) for t in (cur.get("admits") or [])),
                    tier_permissions={
                        int(k): tuple(v) for k, v in (cur.get("tier_permissions") or {}).items()
                    },
                    objective=tuple(cur.get("objective") or []),
                    absence=tuple(cur.get("absence") or []),
                    declared=True,
                    provenance="config",
                    enabled=enabled,
                )
            )

        if need_legacy:
            if not legacy_ok:
                raise ValueError(
                    f"{p}: stages {[stages[i].name for i in need_legacy]} declare no "
                    "extra.curriculum block and legacy reconstruction was disabled"
                )
            from .legacy import GateNotFound, _frozen_admission, reconstruct_stage_admission

            # The live read first: if the gates ever come back, they win over any
            # stored copy. They will not -- 217b01f removed them on purpose -- so
            # in practice this takes the frozen branch, and the distinction is
            # kept because a stored copy silently standing in for a live read is
            # the exact defect `legacy.py` was written to prevent.
            try:
                recon = reconstruct_stage_admission()
            except GateNotFound:
                recon = _frozen_admission()
                if recon is None:
                    raise
                # Every stage built below carries r.provenance, which for this
                # branch reads `frozen:run1@b2b5f7b` -- so a downstream consumer
                # can tell a captured admission from a reconstructed one without
                # asking this function which branch it took.
            for i in need_legacy:
                s = stages[i]
                r = recon.for_stage(s.name)
                stages[i] = StageCurriculum(
                    name=s.name,
                    order=s.order,
                    steps=s.steps,
                    scope=s.name,
                    admits=r.admits,
                    tier_permissions=r.tier_permissions,
                    objective=r.objective,
                    absence=r.absence,
                    declared=False,
                    provenance=r.provenance,
                    enabled=s.enabled,
                )

        return cls(
            run_name=str(train.get("run_name", p.stem)),
            config_path=p,
            stages=tuple(stages),
            mixture_cards=str(payload.get("mixture_cards", "configs/source_cards")),
            fully_declared=not need_legacy,
            notes=str(payload.get("notes", "")),
        )
