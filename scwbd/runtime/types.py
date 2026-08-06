"""The types a downstream consumer sees.

Everything here obeys three rules taken from ``ARCHITECTURE.md`` Sec. 6 and
``thesis_contract.tex`` Sec. 0.5:

1. **Target engagement is a distribution, never a point.**  The types that
   carry it refuse ``float()``.
2. **Field accuracy, target engagement, network effect and utility are four
   separate reported quantities.**  Nothing here fuses them, and
   :class:`UtilityStatus` refuses to carry a value at all in this release.
3. **A read that cannot be supported returns ``Unresolved(reason=...)``, and a
   pose outside :math:`\\mathcal A_{\\rm safe}` returns ``Refuse(code="R11")``.**
   Refusal is a return type, not an exception, wherever the consumer is
   expected to branch on it.

Claim limits for the whole module: these objects describe a *simulation*.  None
of them is a device driver, a dosing protocol, a trajectory, a joint command,
or a recommendation about a person.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import torch
from torch import Tensor

from ._compat import (
    SIMULATION_ONLY_NOTICE,
    VARIANCE_COMPONENTS,
    CompilerRefusal,
    Defer,
    InterventionRefusal,
    NetworkEffect,
    Pose,
    TargetEngagement,
    UncertaintyLedger,
    Unresolved,
)
from .provenance import ModelProvenance

__all__ = [
    "SIMULATION_ONLY_NOTICE",
    "Unresolved",
    "Defer",
    "Recommend",
    "Refuse",
    "Decision",
    "UnderspecifiedPose",
    "UndeclaredTransform",
    "PoseRequest",
    "EFieldPrediction",
    "FieldAccuracy",
    "EngagementDistribution",
    "NetworkResponse",
    "UtilityStatus",
    "PoseEvaluation",
    "full_ledger",
    "LedgerIncomplete",
]

_DT = torch.float64


# ---------------------------------------------------------------------------
# refusals raised at the door
# ---------------------------------------------------------------------------


class UnderspecifiedPose(CompilerRefusal):
    """R01: a pose that names no frame, or no orientation, is not a pose.

    ``thesis_contract.tex`` Sec. 0.5 step 2: *"A pose expressed only as '5 cm
    anterior' is rejected."*  This is raised rather than returned because there
    is nothing to evaluate: without a declared frame there is no field to
    solve, so there is no evaluation object to attach a decision to.
    """

    def __init__(self, message: str, *, offending_object: Any = None) -> None:
        super().__init__(
            "R01",
            message,
            remedy=(
                "state the full 6-DoF coil pose, the frame it is expressed in, "
                "its length unit, its handedness, and the session epoch; a "
                "scalp label or a scalar offset is not a pose "
                "(body.tex Sec. 2.8)"
            ),
            offending_object=offending_object,
        )


class UndeclaredTransform(CompilerRefusal):
    """R01: the transform chain to the head model was never declared.

    The runtime will not insert an assumed identity between two frames.  The
    consumer's registration convention and SC-WBD's frame graph are related by
    *declared, checked* edges or by nothing at all.
    """

    def __init__(self, message: str, *, offending_object: Any = None) -> None:
        super().__init__(
            "R01",
            message,
            remedy=(
                "declare the missing edge with its geometry, provenance, "
                "method and residual; an identity between two differently "
                "named frames is a claim, not a default"
            ),
            offending_object=offending_object,
        )


class LedgerIncomplete(CompilerRefusal):
    """R08: an evaluation was assembled without a full variance decomposition."""

    def __init__(self, message: str, *, offending_object: Any = None) -> None:
        super().__init__(
            "R08",
            message,
            remedy=(
                "populate every variance component and state the bias status "
                "with its estimator or external bound; an unknown term stays "
                "unknown and is never imputed as zero"
            ),
            offending_object=offending_object,
        )


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recommend:
    """The simulated comparison distinguished this candidate.

    A ``Recommend`` is a statement about a *simulation*: this pose is predicted
    to separate from the comparator by more than the epistemic uncertainty of
    the prediction.  It is not clearance to move a robot, energise a coil, or
    stimulate anyone.  ``human_use_authorized`` refuses to be ``True``.
    """

    label: str
    rationale: str
    #: Predicted benefit difference against the declared comparator, in the
    #: units of the network readout.
    benefit_margin: float
    #: Reducible model disagreement + transform-uncertainty contribution.
    epistemic_uncertainty: float
    comparator: str = "no_stimulation"
    basis: tuple[str, ...] = ()
    human_use_authorized: bool = False
    notice: str = SIMULATION_ONLY_NOTICE

    def __post_init__(self) -> None:
        if self.human_use_authorized:  # pragma: no cover - guard
            raise CompilerRefusal(
                "R11",
                "a Recommend cannot be marked authorized for human use",
                offending_object=self.label,
            )

    def __str__(self) -> str:
        return f"Recommend({self.label}): margin {self.benefit_margin:.4g}"


@dataclass(frozen=True)
class Refuse:
    """A refusal returned to a consumer that is expected to branch on it."""

    code: str
    reason: str
    remedy: str = ""
    offending: str = ""
    violations: tuple[str, ...] = ()
    notice: str = SIMULATION_ONLY_NOTICE

    def __bool__(self) -> bool:  # pragma: no cover - guard against `if decision:`
        return False

    def __str__(self) -> str:
        return f"Refuse({self.code}): {self.reason}"


#: What ``evaluate_pose`` may answer.  ``Defer`` is agent G's type from
#: :mod:`scwbd.intervene.safety`; it already carries the *recommended next
#: measurement* (``suggested_action``) that Sec. 0.5 step 6 asks for.
Decision = Recommend | Defer | Refuse


# ---------------------------------------------------------------------------
# pose requests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseRequest:
    """What a caller asks for -- which may be underspecified, and often is.

    This type exists so that underspecification is *representable* and can
    therefore be *rejected* rather than being a type error the caller never
    sees.  ``"5 cm anterior"`` is a legal :class:`PoseRequest` and an illegal
    pose; :meth:`resolve` is where that distinction is enforced.
    """

    #: A fully specified rigid pose, parent = the frame it is expressed in,
    #: child = the coil frame.
    pose: Pose | None = None
    #: The frame the caller believes the pose is expressed in.  Checked against
    #: ``pose.parent``; a disagreement is a refusal, not a warning.
    frame: str | None = None
    #: Free text such as ``"5 cm anterior of the motor hotspot"``.  Recorded,
    #: never parsed into geometry.
    description: str | None = None
    #: Session identity.  ``head<-tracker`` on Tuesday is not the same edge as
    #: ``head<-tracker`` on Wednesday (body.tex Sec. 2.8).
    epoch: str | None = None
    label: str = "candidate"
    #: The declared 6-DoF uncertainty of *this* pose measurement (tracker
    #: jitter, coil-holder compliance, digitisation).  Distinct from the
    #: registration chain's uncertainty, which the head model carries.
    #: ``None`` means "not declared", which is a refusal downstream, not zero.
    uncertainty: Any = None
    #: Which variance bucket this pose's own error belongs in.
    uncertainty_scope: Literal[
        "measurement", "within_session", "between_session"
    ] = "within_session"
    notes: Mapping[str, Any] = field(default_factory=dict)

    def resolve(self) -> Pose:
        """Return the full pose or raise :class:`UnderspecifiedPose` (R01)."""
        if self.pose is None:
            detail = (
                f"description {self.description!r}"
                if self.description
                else "no geometry at all"
            )
            raise UnderspecifiedPose(
                f"pose request {self.label!r} carries {detail} but no 6-DoF "
                "rigid transform; a scalar offset or a scalp label does not "
                "determine coil orientation, and orientation relative to the "
                "local cortical normal is what the induced field depends on",
                offending_object=self.label,
            )
        if self.frame is not None and self.frame != self.pose.parent:
            raise UnderspecifiedPose(
                f"pose request {self.label!r} declares frame {self.frame!r} but "
                f"its transform is expressed as {self.pose.label!r}; the runtime "
                "will not guess which declaration is correct",
                offending_object=self.label,
            )
        if self.epoch is not None and self.pose.epoch not in (None, self.epoch):
            raise UnderspecifiedPose(
                f"pose request {self.label!r} declares epoch {self.epoch!r} but "
                f"its transform carries epoch {self.pose.epoch!r}",
                offending_object=self.label,
            )
        return self.pose

    @staticmethod
    def coerce(obj: "PoseRequest | Pose | Mapping[str, Any]") -> "PoseRequest":
        """Accept a :class:`PoseRequest`, a :class:`Pose`, or a mapping.

        A bare mapping is the shape an external caller most easily produces and
        most easily under-fills, so it is routed through the same refusal.
        """
        if isinstance(obj, PoseRequest):
            return obj
        if isinstance(obj, Pose):
            return PoseRequest(pose=obj, frame=obj.parent, epoch=obj.epoch)
        if isinstance(obj, Mapping):
            known = {
                "pose",
                "frame",
                "description",
                "epoch",
                "label",
                "notes",
                "uncertainty",
                "uncertainty_scope",
            }
            unknown = set(obj) - known
            if unknown:
                raise UnderspecifiedPose(
                    f"pose mapping carries unrecognised keys {sorted(unknown)}; "
                    "the runtime will not guess their meaning",
                    offending_object=sorted(unknown),
                )
            return PoseRequest(**dict(obj))
        raise UnderspecifiedPose(
            f"cannot interpret {type(obj).__name__} as a coil pose",
            offending_object=repr(obj)[:120],
        )


# ---------------------------------------------------------------------------
# ledger helper
# ---------------------------------------------------------------------------


def full_ledger(
    *,
    units: str,
    measurement: float,
    within_session: float,
    between_session: float,
    parameter: float,
    model_class: float,
    numerical: float,
    bias_interval: tuple[float, float],
    bias_status: str = "prior_specified_sensitivity",
    bias_estimator: str | None = None,
    external_bound_source: str | None = None,
    model_discrepancy: float | None = None,
    validity_domain: Mapping[str, Any] | None = None,
    notes: str = "",
) -> UncertaintyLedger:
    """Build a ledger with **every** variance component present.

    ``ARCHITECTURE.md`` Sec. 6 requires the ledger on a ``PoseEvaluation`` to be
    "always" populated.  This constructor makes a partially-filled ledger
    impossible to produce by accident: every component of
    :data:`~scwbd.schema.ledger.VARIANCE_COMPONENTS` is a required keyword.

    A component that is genuinely not estimable should be passed as its
    prior-specified upper bound, not as ``0.0``; ``0.0`` asserts "this term is
    absent", which is a claim.
    """
    ledger = UncertaintyLedger(
        variance={
            "measurement": float(measurement),
            "within_session": float(within_session),
            "between_session": float(between_session),
            "parameter": float(parameter),
            "model_class": float(model_class),
            "numerical": float(numerical),
        },
        bias_interval=(float(bias_interval[0]), float(bias_interval[1])),
        bias_status=bias_status,  # type: ignore[arg-type]
        bias_estimator=bias_estimator,
        external_bound_source=external_bound_source,
        model_discrepancy=model_discrepancy,
        validity_domain=dict(validity_domain or {}),
        units=units,
        notes=notes,
    )
    check_ledger(ledger, what="full_ledger")
    return ledger


def check_ledger(ledger: UncertaintyLedger, *, what: str) -> UncertaintyLedger:
    """Refuse (R08) a ledger that is missing a component or an estimator."""
    missing = [c for c in VARIANCE_COMPONENTS if c not in ledger.variance]
    if missing:
        raise LedgerIncomplete(
            f"{what}: variance decomposition is missing {missing}",
            offending_object=what,
        )
    if not ledger.has_estimator():
        raise LedgerIncomplete(
            f"{what}: bias_status={ledger.bias_status!r} is not backed by the "
            "evidence it names (an estimator, an external bound, or a "
            "non-degenerate prior-specified interval)",
            offending_object=what,
        )
    return ledger


# ---------------------------------------------------------------------------
# level 1: the field
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EFieldPrediction:
    """Induced electric field on the cortical surface, with its covariance.

    ``vector`` is ``[N, 3]`` in ``units`` on the support named by ``frame`` and
    ``support``.  ``covariance`` is ``[N, 3, 3]``: the first-order propagation
    of the *declared* pose-chain covariance and device-gain prior through the
    field solve (body.tex Sec. 2.8).  It is not a fitted residual and is not a
    validation of the field model itself -- see :class:`FieldAccuracy`.
    """

    frame: str
    support: str
    vector: Tensor
    covariance: Tensor
    ledger: UncertaintyLedger
    backend: str
    backend_class: str
    units: str = "V/m"
    is_trained_artifact: bool = False
    notice: str = SIMULATION_ONLY_NOTICE

    def __post_init__(self) -> None:
        v = torch.as_tensor(self.vector, dtype=_DT)
        c = torch.as_tensor(self.covariance, dtype=_DT)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"E-field vector must be [N,3], got {tuple(v.shape)}")
        if c.shape != (v.shape[0], 3, 3):
            raise ValueError(
                f"E-field covariance must be [N,3,3], got {tuple(c.shape)}"
            )
        object.__setattr__(self, "vector", v)
        object.__setattr__(self, "covariance", c)
        check_ledger(self.ledger, what="EFieldPrediction.ledger")

    @property
    def n_elements(self) -> int:
        return int(self.vector.shape[0])

    @property
    def magnitude(self) -> Tensor:
        return torch.linalg.norm(self.vector, dim=-1)

    @property
    def magnitude_sd(self) -> Tensor:
        """Delta-method sd of ``|E|``: ``sqrt(u^T Sigma u)`` with ``u = E/|E|``."""
        mag = self.magnitude.clamp_min(1e-30)
        u = self.vector / mag.unsqueeze(-1)
        var = torch.einsum("ni,nij,nj->n", u, self.covariance, u)
        return var.clamp_min(0.0).sqrt()

    def peak(self) -> float:
        return float(self.magnitude.max())

    def peak_sd(self) -> float:
        return float(self.magnitude_sd[int(torch.argmax(self.magnitude))])

    def as_physical_dose(self) -> Any:
        """Return agent G's :class:`~scwbd.intervene.base.PhysicalDose`.

        Note what that type refuses: ``PhysicalDose.as_neural_effect()`` raises
        R04.  A field is not an effect.
        """
        from ._compat import Ledger, PhysicalDose

        return PhysicalDose(
            modality="tms",
            quantity="E_field",
            units=self.units,
            value=self.vector,
            support=f"{self.support}@{self.frame}",
            ledger=Ledger(
                variance=dict(self.ledger.variance),
                bias_interval=tuple(self.ledger.bias_interval),
                bias_status=self.ledger.bias_status,
                validity_domain=dict(self.ledger.validity_domain),
            ),
        )


@dataclass(frozen=True)
class FieldAccuracy:
    """Level 1 of the thesis Sec. 7.2 validation ladder, reported separately.

    "Pose accuracy, field accuracy, target engagement, network change, symptom
    change, and comparative clinical utility are separate validation levels."
    This object reports how well the *field* is known and against what it has
    been checked -- which for SC-WBD-001-beta is, honestly, nothing external.
    """

    peak_v_per_m: float
    peak_sd_v_per_m: float
    #: Contribution of the declared transform chain alone to the peak sd.
    transform_sd_v_per_m: float
    #: Names of anything the field solve was checked against.  Empty means the
    #: field is unvalidated, and ``validation_status`` says so.
    validated_against: tuple[str, ...] = ()
    validation_status: Literal[
        "unvalidated", "solver_refinement_only", "phantom", "cross_solver", "in_vivo"
    ] = "unvalidated"
    notice: str = SIMULATION_ONLY_NOTICE

    @property
    def relative_sd(self) -> float:
        if self.peak_v_per_m <= 0.0:
            return float("inf")
        return self.peak_sd_v_per_m / self.peak_v_per_m


# ---------------------------------------------------------------------------
# level 2: target engagement -- a distribution, never a point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngagementDistribution:
    """Modelled drive delivered to a named target population.

    ``samples`` is ``[n_response_models, n_draws]``: the predictive draws of
    each *named candidate response operator*, kept separate so that model-class
    disagreement never disappears into a pooled variance.  ``log_weights`` are
    the unnormalised posterior model weights.

    This type deliberately refuses ``float()``.  Collapsing a distribution over
    unresolved mechanisms into one engagement number is exactly the move
    ``thesis_contract.tex`` Sec. 0.5 step 5 forbids.
    """

    target: str
    units: str
    response_models: tuple[str, ...]
    mechanistic_status: tuple[str, ...]
    log_weights: Tensor
    samples: Tensor
    ledger: UncertaintyLedger
    notice: str = SIMULATION_ONLY_NOTICE

    def __post_init__(self) -> None:
        s = torch.as_tensor(self.samples, dtype=_DT)
        w = torch.as_tensor(self.log_weights, dtype=_DT).reshape(-1)
        if s.ndim != 2:
            raise ValueError(f"samples must be [M,S], got {tuple(s.shape)}")
        if s.shape[0] != w.numel() or s.shape[0] != len(self.response_models):
            raise ValueError(
                "samples, log_weights and response_models disagree on the "
                f"number of candidate response operators: {s.shape[0]}, "
                f"{w.numel()}, {len(self.response_models)}"
            )
        if len(self.mechanistic_status) != len(self.response_models):
            raise ValueError("every response model must declare a mechanistic status")
        object.__setattr__(self, "samples", s)
        object.__setattr__(self, "log_weights", w)
        check_ledger(self.ledger, what="EngagementDistribution.ledger")

    def __float__(self) -> float:  # pragma: no cover - guard
        raise InterventionRefusal(
            "R09",
            "target engagement is a distribution over unresolved candidate "
            "response operators and does not have a scalar value; ask for "
            "mean(), quantile(q), or model_disagreement() and carry the "
            "spread with you",
            remedy="report the distribution, not a point",
            offending_object=self.target,
        )

    def posterior_weights(self) -> Tensor:
        return torch.softmax(self.log_weights, dim=0)

    def mean(self) -> float:
        w = self.posterior_weights().unsqueeze(-1)
        return float((w * self.samples).sum() / self.samples.shape[1])

    def sd(self) -> float:
        """Total predictive sd, mixture over response models."""
        w = self.posterior_weights().unsqueeze(-1)
        m = self.mean()
        return float(((w * (self.samples - m) ** 2).sum() / self.samples.shape[1]).sqrt())

    def model_disagreement(self) -> float:
        """Between-model sd of the per-model means: the *reducible* part."""
        per_model = self.samples.mean(dim=1)
        w = self.posterior_weights()
        m = float((w * per_model).sum())
        return float(((w * (per_model - m) ** 2).sum()).sqrt())

    def quantile(self, q: float) -> float:
        w = self.posterior_weights()
        n = max(1, int(round(float(self.samples.shape[1]))))
        counts = torch.clamp((w * n * self.samples.shape[0]).round().long(), min=1)
        pooled = torch.cat(
            [self.samples[i].repeat(int(counts[i])) for i in range(self.samples.shape[0])]
        )
        return float(torch.quantile(pooled, torch.tensor(q, dtype=_DT)))

    def as_target_engagements(self) -> tuple[Any, ...]:
        """One agent-G :class:`TargetEngagement` per named response operator."""
        from ._compat import Ledger

        led = Ledger(
            variance=dict(self.ledger.variance),
            bias_interval=tuple(self.ledger.bias_interval),
            bias_status=self.ledger.bias_status,
            validity_domain=dict(self.ledger.validity_domain),
        )
        return tuple(
            TargetEngagement(
                target=self.target,
                response_model=name,
                mechanistic_status=status,  # type: ignore[arg-type]
                units=self.units,
                value=self.samples[i],
                ledger=led,
            )
            for i, (name, status) in enumerate(
                zip(self.response_models, self.mechanistic_status)
            )
        )


# ---------------------------------------------------------------------------
# level 3: network response
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NetworkResponse:
    """Predicted propagation, with model-class disagreement kept explicit.

    ``per_model`` is ``[K, P]``: the predicted change on each of ``P`` readout
    elements under each of ``K`` *distinct dynamics model classes*.  Model
    comparison over backends is a first-class output of SC-WBD
    (``ARCHITECTURE.md`` Sec. 5), so the disagreement is reported, never
    averaged away.
    """

    readout: str
    units: str
    horizon_s: float
    model_classes: tuple[str, ...]
    log_weights: Tensor
    per_model: Tensor
    elements: tuple[str, ...]
    ledger: UncertaintyLedger
    notice: str = SIMULATION_ONLY_NOTICE

    def __post_init__(self) -> None:
        p = torch.as_tensor(self.per_model, dtype=_DT)
        w = torch.as_tensor(self.log_weights, dtype=_DT).reshape(-1)
        if p.ndim != 2:
            raise ValueError(f"per_model must be [K,P], got {tuple(p.shape)}")
        if p.shape[0] != len(self.model_classes) or p.shape[0] != w.numel():
            raise ValueError("per_model, model_classes and log_weights disagree")
        if p.shape[1] != len(self.elements):
            raise ValueError("per_model columns must match the readout elements")
        object.__setattr__(self, "per_model", p)
        object.__setattr__(self, "log_weights", w)
        check_ledger(self.ledger, what="NetworkResponse.ledger")

    def __float__(self) -> float:  # pragma: no cover - guard
        raise InterventionRefusal(
            "R09",
            "a network response is a profile over readout elements under "
            "several retained model classes and has no scalar value",
            remedy="ask for expected(), or for a named element",
            offending_object=self.readout,
        )

    def posterior_weights(self) -> Tensor:
        return torch.softmax(self.log_weights, dim=0)

    def expected(self) -> Tensor:
        return (self.posterior_weights().unsqueeze(-1) * self.per_model).sum(0)

    def disagreement_profile(self) -> Tensor:
        w = self.posterior_weights().unsqueeze(-1)
        mean = self.expected().unsqueeze(0)
        return ((w * (self.per_model - mean) ** 2).sum(0)).clamp_min(0.0).sqrt()

    def model_class_disagreement(self) -> float:
        return float(self.disagreement_profile().max())

    def at(self, element: str) -> float | Unresolved:
        """Value at a named readout element, or ``Unresolved`` if not modelled."""
        if element not in self.elements:
            return Unresolved(
                reason=(
                    f"{element!r} is not in the readout support of this "
                    "prediction; the runtime does not extrapolate a value onto "
                    "an element the model does not carry"
                ),
                missing=(element,),
            )
        return float(self.expected()[self.elements.index(element)])

    def as_network_effects(self) -> tuple[Any, ...]:
        from ._compat import Ledger

        led = Ledger(
            variance=dict(self.ledger.variance),
            bias_interval=tuple(self.ledger.bias_interval),
            bias_status=self.ledger.bias_status,
            validity_domain=dict(self.ledger.validity_domain),
        )
        return tuple(
            NetworkEffect(
                readout=f"{self.readout}[{name}]",
                units=self.units,
                value=self.per_model[i],
                horizon_s=self.horizon_s,
                ledger=led,
            )
            for i, name in enumerate(self.model_classes)
        )


# ---------------------------------------------------------------------------
# level 4: utility -- reported, and reported as not estimable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UtilityStatus:
    """Level 4, kept as a separate reported quantity that carries no number.

    Agent G's :class:`~scwbd.intervene.base.ClinicalUtility` refuses to be
    constructed at all.  That is correct, but it leaves a consumer with three
    quantities and a hole, and a hole is easy to fill with the third quantity.
    This object occupies the fourth slot explicitly and refuses to yield a
    value, so "utility" stays visible as *missing* rather than becoming a
    synonym for "network effect".
    """

    estimable: bool = False
    reason: str = (
        "comparative utility requires a prospective, causally identified, "
        "ethically approved comparison in people; SC-WBD-001-beta has none "
        "(thesis_contract.tex Sec. 0.6 item 6)"
    )
    value: None = None
    notice: str = SIMULATION_ONLY_NOTICE

    def __post_init__(self) -> None:
        if self.estimable:  # pragma: no cover - guard
            raise InterventionRefusal(
                "R11",
                "utility is not estimable in SC-WBD-001-beta",
                remedy="report the network effect and stop",
                offending_object="UtilityStatus",
            )

    def require_value(self) -> float:  # pragma: no cover - guard
        raise InterventionRefusal(
            "R11", self.reason, remedy="do not label a simulation clinical"
        )

    def as_unresolved(self) -> Unresolved:
        return Unresolved(reason=self.reason, missing=("clinical_utility",))


# ---------------------------------------------------------------------------
# the evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseEvaluation:
    """The object ``ARCHITECTURE.md`` Sec. 6 specifies, returned in full.

    Claim limits: this is the *neuro* half of target selection for one
    candidate coil pose in a simulation.  It contains no joint command, no
    trajectory, no actuation, no stimulation authority and no dosing protocol,
    and it is not evidence that any admitted operator is neurally realized.
    """

    label: str
    #: The resolved coil pose, expressed in the head model's working frame.
    pose: Pose
    #: The frame the caller declared, before the transform chain was applied.
    requested_frame: str
    #: Labels of the declared edges that were composed to get here.  An empty
    #: tuple would mean an assumed identity and is refused upstream.
    transform_chain: tuple[str, ...]

    # the four separate reported quantities, in ladder order
    field_accuracy: FieldAccuracy
    target_engagement: EngagementDistribution
    network_response: NetworkResponse
    utility: UtilityStatus

    efield: EFieldPrediction
    ledger: UncertaintyLedger
    decision: Decision
    provenance: ModelProvenance
    #: Everything the A_safe filter checked, and what it could not check.
    safety_axes_checked: tuple[str, ...] = ()
    safety_axes_unchecked: tuple[str, ...] = ()
    notice: str = SIMULATION_ONLY_NOTICE

    def __post_init__(self) -> None:
        check_ledger(self.ledger, what="PoseEvaluation.ledger")
        if not self.transform_chain:
            raise UndeclaredTransform(
                f"evaluation {self.label!r} records no transform chain; a pose "
                "that reached the head model through no declared edge reached "
                "it through an assumed identity",
                offending_object=self.label,
            )

    # -- the separation the thesis requires --------------------------------
    def four_quantities(self) -> dict[str, Any]:
        """The four validation levels, as four distinct objects.

        There is deliberately no method that combines them.  ``thesis
        Sec. 7.2``: "Pose accuracy, field accuracy, target engagement, network
        change, symptom change, and comparative clinical utility are separate
        validation levels."
        """
        return {
            "field_accuracy": self.field_accuracy,
            "target_engagement": self.target_engagement,
            "network_effect": self.network_response,
            "utility": self.utility,
        }

    @property
    def recommended(self) -> bool:
        return isinstance(self.decision, Recommend)

    @property
    def refused(self) -> bool:
        return isinstance(self.decision, Refuse)

    @property
    def deferred(self) -> bool:
        return isinstance(self.decision, Defer)

    def summary(self) -> dict[str, Any]:
        """A flat, JSON-ready record. Still carries the notice and the ledger."""
        return {
            "label": self.label,
            "pose": self.pose.label,
            "requested_frame": self.requested_frame,
            "transform_chain": list(self.transform_chain),
            "decision": type(self.decision).__name__,
            "decision_detail": str(self.decision),
            "field_peak_v_per_m": self.field_accuracy.peak_v_per_m,
            "field_peak_sd_v_per_m": self.field_accuracy.peak_sd_v_per_m,
            "field_validation_status": self.field_accuracy.validation_status,
            "engagement_mean": self.target_engagement.mean(),
            "engagement_sd": self.target_engagement.sd(),
            "engagement_model_disagreement": (
                self.target_engagement.model_disagreement()
            ),
            "network_model_class_disagreement": (
                self.network_response.model_class_disagreement()
            ),
            "utility_estimable": self.utility.estimable,
            "ledger": self.ledger.canonical(),
            "provenance": self.provenance.canonical(),
            "safety_axes_checked": list(self.safety_axes_checked),
            "safety_axes_unchecked": list(self.safety_axes_unchecked),
            "notice": self.notice,
        }

