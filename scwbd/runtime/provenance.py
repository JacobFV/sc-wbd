"""Version/provenance handshake between SC-WBD and any downstream consumer.

The point of this module is one sentence: **a consumer must be able to assert
exactly which model, schema, and claim it is talking to, and fail if that is
not what it expected.**  ``tms-robotics`` is a repository whose whole discipline
is keeping evidence classes apart; handing it an unlabelled prediction would
destroy that discipline from the outside.

Two failure modes it exists to prevent:

1. a randomly-initialised or analytic-fallback model being mistaken for the
   trained ``SC-WBD-001-beta`` artifact;
2. a consumer pinned to one schema silently receiving another.

Claim limits
------------
A successful handshake asserts *identity*, never *validity*.  It says "you are
talking to the artifact you named"; it says nothing about whether that
artifact's predictions are accurate, and nothing whatsoever about whether any
intervention is safe, approved, or applicable to a person.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

from ._compat import SIMULATION_ONLY_NOTICE

__all__ = [
    "MODEL_DESIGNATION",
    "SCHEMA_VERSION",
    "RUNTIME_API_VERSION",
    "WeightsStatus",
    "BackendClass",
    "ModelProvenance",
    "ProvenanceExpectation",
    "ProvenanceMismatch",
]

#: The one model this release serves.
MODEL_DESIGNATION = "SC-WBD-001-beta"
#: ``ARCHITECTURE.md`` header, "Schema version".
SCHEMA_VERSION = "scwbd-schema/1.0.0"
#: The shape of :class:`~scwbd.runtime.targeting.PoseEvaluation` and friends.
#: Bumped whenever a consumer would have to change code.
RUNTIME_API_VERSION = "scwbd-runtime/1.0.0"

WeightsStatus = Literal[
    "trained",  # a checkpoint with a ClaimManifest was loaded
    "randomly_initialized",  # a network exists but was never trained
    "analytic_backend",  # no network at all; closed-form physics + surrogates
    "absent",  # nothing was loaded; the service refuses to evaluate
]

BackendClass = Literal[
    "analytic", "numerical_bem", "numerical_fem", "learned", "unknown"
]


class ProvenanceMismatch(RuntimeError):
    """Raised when the served artifact is not what the consumer asserted.

    Carries ``mismatches`` (a tuple of human-readable field-level differences)
    so the consumer can log exactly which assertion failed rather than a
    generic version error.
    """

    def __init__(self, mismatches: tuple[str, ...], *, served: "ModelProvenance") -> None:
        super().__init__(
            "SC-WBD provenance handshake failed:\n  - " + "\n  - ".join(mismatches)
        )
        self.mismatches = tuple(mismatches)
        self.served = served


@dataclass(frozen=True)
class ModelProvenance:
    """What is actually being served, stated in full.

    ``human_use_authorized`` is hard-wired ``False`` and refuses to be
    constructed ``True``.  It exists so that a consumer that forgets to branch
    on ``decision`` still carries the flag through its own evidence graph.
    """

    model_designation: str = MODEL_DESIGNATION
    schema_version: str = SCHEMA_VERSION
    runtime_api_version: str = RUNTIME_API_VERSION

    #: Directory name under ``checkpoints/`` this was loaded from, or ``""``.
    checkpoint_id: str = ""
    #: sha256 of the checkpoint weights file, or ``None`` when there is none.
    checkpoint_sha256: str | None = None
    weights_status: WeightsStatus = "analytic_backend"

    #: From the ``ClaimManifest`` beside the checkpoint (agent A's type).
    claim_manifest_id: str = "runtime_default"
    claim_class: str = "surrogate"
    posterior_class: str = "pseudo"
    #: Digest of the port contract the checkpoint declares
    #: (:meth:`scwbd.runtime.ports.PortContract.digest`), or ``""`` when it
    #: declares none.  A consumer pins this instead of learning a state width:
    #: run 1 is uniform ``(B,T,454,28)`` and run 2 is per-family with
    #: heterogeneous widths, so a pinned digest fails loudly at load where a
    #: pinned ``28`` would have failed silently at the first read.
    port_contract_digest: str = ""
    #: ``"family.name"`` of every port the checkpoint declares **and** exports.
    #: Empty means the artifact declared no consumable ports, which is a
    #: different statement from "it exports nothing you asked for".
    exported_ports: tuple[str, ...] = ()
    #: ``thesis_contract.tex`` Sec. 0.6 build-order item this artifact reaches.
    build_order_item: int = 4
    #: Whether this service is being served for prospective human application.
    #:
    #: This used to raise unconditionally, with the reason "out of scope for
    #: SC-WBD-001-beta (build-order item 6)".  That was the mirror of the stale
    #: strings ``scwbd.intervene`` used to carry: a restriction asserted in
    #: source that no record could ever satisfy, which made the authorization
    #: mechanism structurally unreachable from the consumer -- the gate could
    #: only ever be closed, so it could not discriminate, so it was not a gate.
    #: It is now decided by
    #: :func:`~scwbd.intervene.deployment.authorize_live_application` via
    #: :attr:`live_application`, so the refusal has a reason that can be
    #: satisfied.  It still refuses by default and on every incomplete record.
    prospective_human: bool = False
    #: ``LiveApplicationVerdict.as_provenance()`` for this service, or ``None``.
    #: Required to be present and admitting whenever
    #: :attr:`prospective_human` is ``True``.
    live_application: Mapping[str, Any] | None = None

    efield_backend: str = "analytic_spherical_primary"
    efield_backend_class: BackendClass = "analytic"
    #: Numerical gates the field solve has passed, by name.  Empty means the
    #: field computation carries no independent numerical validation, which is
    #: a different statement from "the field is wrong" and from "the head model
    #: is right".
    efield_gate_evidence: tuple[str, ...] = ()
    response_models: tuple[str, ...] = ()
    dynamics_model_classes: tuple[str, ...] = ()
    a_safe_source: str = ""
    a_safe_citations: tuple[str, ...] = ()

    device: str = "cpu"
    torch_version: str = ""
    notes: Mapping[str, Any] = field(default_factory=dict)

    # -- governance --------------------------------------------------------
    #: ``"simulation_only"`` or ``"protocol:<id>@<version>"``.  An evaluation
    #: served under a validated ``AuthorizationRecord`` is a different artifact
    #: from one that was not, and this is where a consumer sees which.
    claim_scope: str = "simulation_only"
    #: ``AuthorizationVerdict.as_provenance()`` for the record this service was
    #: constructed with, or ``None``.  Pins the record's content hash.
    authorization: Mapping[str, Any] | None = None

    #: Whether *this software* authorises human use.  Permanently ``False``,
    #: and unrelated to ``claim_scope``: SC-WBD records a declaration made by
    #: the people responsible for a protocol; it never issues one, and a
    #: recorded protocol scope is not this repository granting permission.
    human_use_authorized: bool = False
    notice: str = SIMULATION_ONLY_NOTICE

    def __post_init__(self) -> None:
        if self.human_use_authorized:
            # This one stays unconditional, and the asymmetry is deliberate.
            # `prospective_human` is a statement about what is being attempted,
            # which a record can answer.  `human_use_authorized` is a statement
            # that *this repository* issued an authorization, which is never
            # true whatever any record says -- SC-WBD records declarations made
            # by the people responsible for a protocol; it does not make them.
            raise ValueError(
                "ModelProvenance cannot be marked authorized for human use; "
                "this software does not issue authorization. Governance is "
                "carried by an AuthorizationRecord declared by the people "
                "responsible for a protocol and recorded in `claim_scope` and "
                "`authorization` (scwbd.schema.authorization); a recorded "
                "protocol scope is not this repository granting permission"
            )
        if self.prospective_human:
            verdict = self.live_application
            admitted = bool(verdict) and bool(verdict.get("admitted"))
            live = bool(verdict) and verdict.get("application_mode") == "live"
            if not (admitted and live):
                why = (
                    "no live-application verdict was recorded"
                    if not verdict
                    else "; ".join(verdict.get("failure_codes") or ())
                    or f"application_mode={verdict.get('application_mode')!r}"
                )
                raise ValueError(
                    "prospective human application requires an admitting "
                    "live-application verdict in `live_application`, produced by "
                    "scwbd.intervene.deployment.authorize_live_application with "
                    f"mode='live'. Refused: {why}. This refusal is satisfiable: "
                    "supply a PreliminaryReviewRecord recording that the review "
                    "occurred with an approving outcome, together with a "
                    "validated AuthorizationRecord covering the intervention "
                    "class -- the authorization alone is necessary and not "
                    "sufficient"
                )

    # -- identity ----------------------------------------------------------
    @property
    def is_trained_artifact(self) -> bool:
        """``True`` only for a loaded, hashed, trained checkpoint."""
        return self.weights_status == "trained"

    @property
    def is_protocol_bound(self) -> bool:
        """True when this service carries a validated authorization record."""
        return self.claim_scope != "simulation_only"

    @property
    def authorization_hash(self) -> str:
        if not self.authorization:
            return ""
        return str(self.authorization.get("record_hash", ""))

    @property
    def label(self) -> str:
        scope = "" if not self.is_protocol_bound else f" under {self.claim_scope}"
        return (
            f"{self.model_designation}[{self.weights_status}] "
            f"{self.schema_version} {self.runtime_api_version}{scope}"
        )

    def canonical(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = dict(self.notes)
        payload["authorization"] = (
            dict(self.authorization) if self.authorization else None
        )
        payload.pop("notice", None)
        return payload

    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class ProvenanceExpectation:
    """What a consumer asserts before it will use a prediction.

    Every field is optional; ``None`` means "do not assert this".  Asserting
    nothing is legal but is itself recorded, because a consumer that checked
    nothing should not later claim it verified the model.
    """

    model_designation: str | None = MODEL_DESIGNATION
    schema_version: str | None = SCHEMA_VERSION
    runtime_api_version: str | None = RUNTIME_API_VERSION
    #: Acceptable weight states.  A consumer that wants a trained artifact
    #: passes ``("trained",)`` and gets a hard failure on a fallback.
    accept_weights_status: tuple[WeightsStatus, ...] | None = None
    accept_efield_backend_class: tuple[BackendClass, ...] | None = None
    #: Gate names the consumer requires the field solve to have passed, e.g.
    #: ``("N8_induced_efield_contact",)`` for a lane that positions a coil
    #: against the scalp.
    require_efield_gates: tuple[str, ...] | None = None
    checkpoint_sha256: str | None = None
    #: The exact port contract the consumer was written against.  Pin it: a
    #: mismatch here is the model's state layout having changed underneath the
    #: consumer, which is the failure this whole handshake exists to make loud.
    port_contract_digest: str | None = None
    #: Port names (``"family.name"``) the consumer requires to be **declared
    #: and exported**.  Asserting the digest pins the whole layout; asserting
    #: names pins only what is actually read, which is what a consumer that
    #: wants to survive an unrelated layout change should do.
    require_exported_ports: tuple[str, ...] | None = None
    #: Minimum number of distinct candidate response models required before the
    #: consumer will look at an engagement distribution at all.
    min_response_models: int | None = None
    min_dynamics_model_classes: int | None = None
    require_a_safe_citations: bool = False
    #: Assert the governance scope of what is being served.  A consumer that
    #: must never touch a protocol-bound artifact passes
    #: ``"simulation_only"``; one that requires a named protocol passes
    #: ``"protocol:<id>@<version>"``.
    require_claim_scope: str | None = None

    def mismatches(self, served: ModelProvenance) -> tuple[str, ...]:
        out: list[str] = []

        def eq(name: str, expected: Any, actual: Any) -> None:
            if expected is not None and expected != actual:
                out.append(f"{name}: expected {expected!r}, served {actual!r}")

        eq("model_designation", self.model_designation, served.model_designation)
        eq("schema_version", self.schema_version, served.schema_version)
        eq("runtime_api_version", self.runtime_api_version, served.runtime_api_version)
        eq("checkpoint_sha256", self.checkpoint_sha256, served.checkpoint_sha256)

        if (
            self.port_contract_digest is not None
            and self.port_contract_digest != served.port_contract_digest
        ):
            out.append(
                f"port_contract_digest: expected "
                f"{self.port_contract_digest!r}, served "
                f"{served.port_contract_digest!r} -- the model's declared state "
                "layout is not the one this consumer was written against. Do "
                "not re-pin this without re-reading which ports moved: a state "
                "layout changes between runs (run 1 is uniform 28-wide, run 2 "
                "is per-family), and a stale pin is the only thing standing "
                "between a consumer and a silently different quantity"
            )
        if self.require_exported_ports is not None:
            absent = [
                p for p in self.require_exported_ports if p not in served.exported_ports
            ]
            if absent:
                out.append(
                    f"exported_ports: required {absent} to be declared and "
                    f"exported; served exports {list(served.exported_ports)} -- "
                    "an undeclared port is not an empty read, it is a read this "
                    "artifact cannot support"
                )

        if (
            self.accept_weights_status is not None
            and served.weights_status not in self.accept_weights_status
        ):
            out.append(
                f"weights_status: expected one of "
                f"{list(self.accept_weights_status)}, served "
                f"{served.weights_status!r} -- this artifact is not a trained "
                "SC-WBD-001-beta checkpoint"
            )
        if (
            self.accept_efield_backend_class is not None
            and served.efield_backend_class not in self.accept_efield_backend_class
        ):
            out.append(
                f"efield_backend_class: expected one of "
                f"{list(self.accept_efield_backend_class)}, served "
                f"{served.efield_backend_class!r} ({served.efield_backend})"
            )
        if self.require_efield_gates is not None:
            missing = [
                g for g in self.require_efield_gates if g not in served.efield_gate_evidence
            ]
            if missing:
                out.append(
                    f"efield_gate_evidence: expected {missing} to have passed, "
                    f"served {list(served.efield_gate_evidence)} -- the field "
                    "computation carries no such numerical validation"
                )
        if (
            self.min_response_models is not None
            and len(served.response_models) < self.min_response_models
        ):
            out.append(
                f"response_models: expected at least {self.min_response_models}, "
                f"served {len(served.response_models)} {list(served.response_models)}"
            )
        if (
            self.min_dynamics_model_classes is not None
            and len(served.dynamics_model_classes) < self.min_dynamics_model_classes
        ):
            out.append(
                f"dynamics_model_classes: expected at least "
                f"{self.min_dynamics_model_classes}, served "
                f"{len(served.dynamics_model_classes)}"
            )
        if (
            self.require_claim_scope is not None
            and self.require_claim_scope != served.claim_scope
        ):
            out.append(
                f"claim_scope: expected {self.require_claim_scope!r}, served "
                f"{served.claim_scope!r} -- an artifact produced under a named "
                "protocol is not the same artifact as one that was not"
            )
        if self.require_a_safe_citations and not served.a_safe_citations:
            out.append(
                "a_safe_citations: expected a non-empty citation list; an "
                "uncited feasible set is not an independently validated one"
            )
        return tuple(out)

    def check(self, served: ModelProvenance) -> ModelProvenance:
        """Return ``served`` or raise :class:`ProvenanceMismatch`."""
        bad = self.mismatches(served)
        if bad:
            raise ProvenanceMismatch(bad, served=served)
        return served
