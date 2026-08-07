"""Claim manifests and overrides.

"An override is possible only via ``ClaimManifest.overrides`` and **changes the
claim class recorded in the artifact's provenance**" (ARCHITECTURE.md sec. 2);
the thesis says the same: "An override changes the claim carried by the
resulting artifact and remains visible in its provenance."

An override is therefore not a suppression switch.  It must *demote* the claim
class, and the compiler records the demotion in ``CompiledModel.provenance``.
A no-op override (same or stronger class) is itself refused, because that would
be exactly the silent override the thesis forbids.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from .authorization import (
    AuthorizationRecord,
    AuthorizationVerdict,
    validate_authorization,
)
from .base import SchemaModel
from .refusals import REFUSAL_CODES, RefusalCode

__all__ = [
    "ClaimClass",
    "CLAIM_STRENGTH",
    "PosteriorClass",
    "ClaimOverride",
    "ClaimManifest",
    "is_demotion",
]

#: Ordered from strongest to weakest.  "Mechanistic language is earned only by
#: predictions that an equal-capacity surrogate misses" (thesis sec. 0).
ClaimClass = Literal[
    "mechanistic", "effective", "functional", "surrogate", "provenance_only"
]

CLAIM_STRENGTH: dict[str, int] = {
    "mechanistic": 4,
    "effective": 3,
    "functional": 2,
    "surrogate": 1,
    "provenance_only": 0,
}

#: What the artifact's posterior may be *called*.  R09 turns on this field.
PosteriorClass = Literal["calibrated_bayesian", "generalized", "pseudo", "point_estimate"]


def is_demotion(before: str, after: str) -> bool:
    return CLAIM_STRENGTH[after] < CLAIM_STRENGTH[before]


class ClaimOverride(SchemaModel):
    """An explicit, visible weakening of the claim in exchange for a refusal."""

    code: RefusalCode
    justification: str
    #: The claim class the artifact carries once this override is applied.
    resulting_claim_class: ClaimClass
    approved_by: str
    #: What empirical finding would disable the override (ARCHITECTURE sec. 4).
    disabling_evidence: str = ""

    @model_validator(mode="after")
    def _check(self) -> "ClaimOverride":
        if not self.justification.strip():
            raise ValueError(f"override of {self.code} requires a justification")
        if not self.approved_by.strip():
            raise ValueError(f"override of {self.code} requires an approver")
        return self


class ClaimManifest(SchemaModel):
    """What the compiled artifact is allowed to claim, and what it requests."""

    id: str
    #: The claim the artifact intends to carry before any override.
    claim_class: ClaimClass
    posterior_class: PosteriorClass = "generalized"
    #: Request a single materialized global cross-scale section (drives R03).
    requires_global_section: bool = False
    #: Request optimization over intervention parameters (drives R11).
    optimizes_intervention: bool = False
    #: Whether prospective human stimulation is intended.  R11 refuses this
    #: *unless* a validated :class:`~scwbd.schema.authorization.\
    #: AuthorizationRecord` below admits the requested intervention class at
    #: :attr:`request_time_s` -- and continues to refuse for every other reason
    #: (no A_safe, uncalibrated dose, uncertainty dominating the benefit, no
    #: trained model, unresolved field solve).
    prospective_human: bool = False
    #: The recorded claim of ethics/consent/regulatory authorization this
    #: artifact is compiled under, if any.  Its content hash and the resulting
    #: claim scope are written into ``CompiledModel.provenance``: an artifact
    #: compiled under a protocol can never be mistaken for one that was not.
    authorization: AuthorizationRecord | None = None
    #: The time the request applies to, on the ``wall`` clock.  Required for a
    #: prospective request, because approvals expire and an undated request
    #: cannot be checked against a validity window.
    request_time_s: float | None = None
    #: Claim gates (G1..G5) this artifact is meant to be evaluated against.
    target_gates: tuple[str, ...] = ()
    #: Which refusals are overridden, and at what cost.
    overrides: tuple[ClaimOverride, ...] = ()
    #: Statement of what empirical finding would disable the claim.
    disabling_evidence: str = ""
    notes: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "ClaimManifest":
        seen: set[str] = set()
        for o in self.overrides:
            if o.code in seen:
                raise ValueError(f"duplicate override for {o.code}")
            seen.add(o.code)
            if not is_demotion(self.claim_class, o.resulting_claim_class):
                raise ValueError(
                    f"override of {o.code} sets claim class "
                    f"{o.resulting_claim_class!r}, which does not weaken "
                    f"{self.claim_class!r}; an override must change the claim "
                    "carried by the artifact"
                )
        for g in self.target_gates:
            if g not in ("G1", "G2", "G3", "G4", "G5"):
                raise ValueError(f"unknown claim gate {g!r}")
        return self

    # -- queries ----------------------------------------------------------
    def override_for(self, code: str) -> ClaimOverride | None:
        for o in self.overrides:
            if o.code == code:
                return o
        return None

    def overridden_codes(self) -> tuple[str, ...]:
        return tuple(sorted(o.code for o in self.overrides))

    def authorization_for(
        self,
        intervention_class: str,
        *,
        a_safe_id: str | None = None,
        required_a_safe_axes: tuple[str, ...] = (),
        what: str = "claim manifest",
    ) -> "AuthorizationVerdict":
        """Check this manifest's declared authorization against one request.

        Never raises for an invalid record; returns a verdict naming every
        specific failure.  A manifest carrying no record yields a verdict whose
        single failure is ``AUTH_ABSENT`` -- still a refusal, but one that
        names a missing, supplyable artifact instead of asserting a belief
        about the world.
        """
        return validate_authorization(
            self.authorization,
            intervention_class=intervention_class,
            at_time_s=self.request_time_s,
            a_safe_id=a_safe_id,
            required_a_safe_axes=required_a_safe_axes,
            what=what,
        )

    def effective_claim_class(self) -> str:
        """Claim class after every override is applied (the weakest wins)."""
        cls = self.claim_class
        for o in self.overrides:
            if is_demotion(cls, o.resulting_claim_class):
                cls = o.resulting_claim_class
        return cls

    @property
    def claims_causality(self) -> bool:
        return self.claim_class in ("mechanistic", "effective")

    @classmethod
    def strict(cls, id: str = "strict", claim_class: ClaimClass = "effective") -> "ClaimManifest":
        """A manifest with no overrides: every refusal is fatal."""
        return cls(id=id, claim_class=claim_class)


assert set(REFUSAL_CODES), "refusal codes must be non-empty"
