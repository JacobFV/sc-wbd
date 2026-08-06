"""``CompiledModel``: the artifact ``compile()`` returns.

Exposes exactly what ARCHITECTURE.md sec. 2 promises: ``.state_layout``,
``.adjacency``, ``.dispatch``, ``.schedule``, ``.gradient_masks``,
``.frame_graph``, ``.clock_graph``, ``.ledger`` - plus ``.provenance``, which
is where an override's cost is recorded and stays visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schema.base import canonical_json, content_hash_of
from ..schema.claims import ClaimManifest
from ..schema.refusals import RefusalRecord
from ..schema.schema import BrainSchema
from .adjacency import Adjacency
from .dispatch import Dispatch
from .graphs import CompiledClockGraph, CompiledFrameGraph
from .layout import StateLayout
from .ledger import CompiledLedger
from .masks import GradientMasks
from .schedule import MultirateSchedule

__all__ = ["CompiledModel", "Provenance", "COMPILER_VERSION"]

COMPILER_VERSION = "scwbd-compiler/1.0.0"


@dataclass(frozen=True)
class Provenance:
    """What this artifact is, and what it is allowed to claim.

    ``effective_claim_class`` is the field an override mutates.  It is derived,
    not copied: an artifact compiled with an override can never report the
    claim class its author originally asked for.
    """

    schema_id: str
    schema_version: str
    schema_hash: str
    claim_id: str
    #: Claim class the manifest requested.
    requested_claim_class: str
    #: Claim class the artifact actually carries after overrides.
    effective_claim_class: str
    posterior_class: str
    compiler_version: str
    #: Every refusal that fired, with its override state.
    refusals: tuple[RefusalRecord, ...] = ()
    #: Refusal codes that were checked and passed.
    checks_passed: tuple[str, ...] = ()
    #: Non-fatal observations (unreachable parameter groups, bfloat16 in state,
    #: variance keys outside the declared taxonomy, ...).
    warnings: tuple[str, ...] = ()
    target_gates: tuple[str, ...] = ()
    disabling_evidence: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def was_overridden(self) -> bool:
        return any(r.overridden for r in self.refusals)

    @property
    def overridden_codes(self) -> tuple[str, ...]:
        return tuple(sorted({r.code for r in self.refusals if r.overridden}))

    @property
    def claim_was_demoted(self) -> bool:
        return self.effective_claim_class != self.requested_claim_class

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "schema_hash": self.schema_hash,
            "claim_id": self.claim_id,
            "requested_claim_class": self.requested_claim_class,
            "effective_claim_class": self.effective_claim_class,
            "posterior_class": self.posterior_class,
            "compiler_version": self.compiler_version,
            "refusals": [r.model_dump(mode="json") for r in self.refusals],
            "checks_passed": list(self.checks_passed),
            "warnings": list(self.warnings),
            "target_gates": list(self.target_gates),
            "disabling_evidence": self.disabling_evidence,
            "extra": self.extra,
        }

    def to_json(self) -> str:
        return canonical_json(self.as_dict())

    def summary(self) -> str:
        line = (
            f"claim={self.effective_claim_class} "
            f"(requested {self.requested_claim_class}), "
            f"posterior={self.posterior_class}"
        )
        if self.was_overridden:
            line += f", overrides={list(self.overridden_codes)}"
        return line


@dataclass(frozen=True)
class CompiledModel:
    """A schema that survived every refusal, resolved into executable structure."""

    schema: BrainSchema
    claim: ClaimManifest
    state_layout: StateLayout
    adjacency: Adjacency
    dispatch: Dispatch
    schedule: MultirateSchedule
    gradient_masks: GradientMasks
    frame_graph: CompiledFrameGraph
    clock_graph: CompiledClockGraph
    ledger: CompiledLedger
    provenance: Provenance

    # -- identity ----------------------------------------------------------
    def content_hash(self) -> str:
        """Hash over the schema, the claim and the emitted layout ABI."""
        return content_hash_of(
            {
                "schema": self.schema.content_hash(),
                "claim": self.claim.content_hash(),
                "abi": self.state_layout.abi_digest(),
                "compiler": COMPILER_VERSION,
            }
        )

    @property
    def claim_class(self) -> str:
        """The claim this artifact carries - never the one that was requested."""
        return self.provenance.effective_claim_class

    def summary(self) -> str:
        return "\n".join(
            [
                f"CompiledModel[{self.schema.id}] {self.provenance.summary()}",
                f"  {self.state_layout.summary()}",
                f"  {self.adjacency.summary()}",
                f"  Dispatch({len(self.dispatch)} operators, "
                f"max_delay={self.dispatch.max_delay():.4g}s)",
                f"  {self.schedule.summary()}",
                f"  {self.gradient_masks.summary()}",
                f"  {self.frame_graph.summary()}",
                f"  {self.clock_graph.summary()}",
                f"  {self.ledger.summary()}",
            ]
        )

    def __repr__(self) -> str:
        return (
            f"CompiledModel(schema={self.schema.id!r}, "
            f"claim_class={self.claim_class!r}, "
            f"regions={len(self.schema.regions)}, "
            f"operators={len(self.dispatch)})"
        )
