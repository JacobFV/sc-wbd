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
    """What is actually being served, stated in full."""

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
    #: ``thesis_contract.tex`` Sec. 0.6 build-order item this artifact reaches.
    build_order_item: int = 4

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

    notice: str = SIMULATION_ONLY_NOTICE

    # -- identity ----------------------------------------------------------
    @property
    def is_trained_artifact(self) -> bool:
        """``True`` only for a loaded, hashed, trained checkpoint."""
        return self.weights_status == "trained"

    @property
    def label(self) -> str:
        return (
            f"{self.model_designation}[{self.weights_status}] "
            f"{self.schema_version} {self.runtime_api_version}"
        )

    def canonical(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = dict(self.notes)
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
    #: Minimum number of distinct candidate response models required before the
    #: consumer will look at an engagement distribution at all.
    min_response_models: int | None = None
    min_dynamics_model_classes: int | None = None
    require_a_safe_citations: bool = False

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
