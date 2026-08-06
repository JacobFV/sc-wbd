"""Source cards: ``D_k = (I,G,P,S,T,C,O,U,M,B,V,Delta,A,R)`` (Appendix B).

"A named dataset is not a training instruction."  Every dataset snapshot is
compiled from a source card that declares what it measures, what was done to
it, and where its gradients are permitted to flow.  The compiler converts these
declarations into dataloaders, transform paths, loss factors, gradient masks
and group-aware splits.

Refusals that read off this object: R01 (units/clock/support/frame/calibration),
R07 (hierarchical effects without centering or shrinkage), R08 (bias status),
R09 (pseudo-likelihood as calibrated likelihood), R10 (lineage-crossing split),
R11 (intervention outside a validated safe set).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from .base import SchemaModel
from .frames import CalibrationManifest
from .ledger import UncertaintyLedger
from .lineage import Identity, LineageUnit
from .priors import Prior
from .supports import Support, TemporalSupport
from .units import Unit

__all__ = [
    "Governance",
    "HierarchicalEffect",
    "PopulationStructure",
    "SignalSpec",
    "ObservationModel",
    "SafeSet",
    "InterventionModel",
    "Missingness",
    "GradientPermission",
    "TemporalHoldout",
    "SplitPolicy",
    "SourceCard",
    "SourceRole",
    "LikelihoodKind",
    "GENERATIVE_ROLES",
]

#: The seven non-equivalent roles of Appendix B.  A role is fixed in the run
#: manifest *before* final evaluation; changing it after seeing test results is
#: a rejection condition.
SourceRole = Literal[
    "prior",
    "likelihood",
    "boundary_target",
    "distillation",
    "calibration",
    "negative_control",
    "evaluation_only",
]

#: Roles that may contribute a factor claiming to be a generative likelihood.
GENERATIVE_ROLES: frozenset[str] = frozenset({"likelihood"})

LikelihoodKind = Literal[
    "generative",
    "pseudo",
    "agreement_penalty",
    "score_matching",
    "none",
]


class Governance(SchemaModel):
    """License, consent scope, redistribution and retention (Appendix B row 2)."""

    license: str
    dua: str | None = None
    purpose_limits: tuple[str, ...] = ()
    consent_scope: str = ""
    redistribution: Literal["none", "derived_only", "full", "unknown"] = "unknown"
    withdrawal_status: Literal["none_pending", "pending", "honored", "unknown"] = "unknown"
    privacy_class: Literal["public", "deidentified", "pseudonymized", "identifiable"] = (
        "deidentified"
    )
    retention: str = ""
    may_release_weights: bool = False
    may_release_examples: bool = False
    contains_biometric: bool = False
    safeguards: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> "Governance":
        if self.contains_biometric and not self.safeguards:
            raise ValueError(
                "biometric data requires declared safeguards (Appendix B row 2)"
            )
        return self

    @property
    def is_resolved(self) -> bool:
        return (
            self.withdrawal_status not in ("pending", "unknown")
            and self.redistribution != "unknown"
            and bool(self.license.strip())
        )


class HierarchicalEffect(SchemaModel):
    """A population/subject/session effect and how it is identified.

    "The additive decomposition is not identified" without centering or
    shrinkage - refusal R07.
    """

    name: str
    level: Literal[
        "population", "family", "site", "device", "participant", "session", "run", "trial"
    ]
    parameterization: Literal[
        "centered", "sum_to_zero", "noncentered_hierarchical", "unconstrained"
    ]
    shrinkage_prior: Prior | None = None
    #: Whether recovery of this effect was tested in simulation.
    recovery_tested: bool = False
    recovery_report: str | None = None
    units: Unit = Unit("dimensionless")

    @property
    def is_identified(self) -> bool:
        constrained = self.parameterization in (
            "centered",
            "sum_to_zero",
            "noncentered_hierarchical",
        )
        return (constrained or self.shrinkage_prior is not None) and self.recovery_tested

    def deficiency(self) -> str | None:
        if self.parameterization == "unconstrained" and self.shrinkage_prior is None:
            return "unconstrained parameterization with no shrinkage prior"
        if not self.recovery_tested:
            return "no simulation recovery test"
        return None


class PopulationStructure(SchemaModel):
    """Participants, families, sites, devices, sessions, runs, trials."""

    levels: tuple[str, ...] = ()
    units: tuple[LineageUnit, ...] = ()
    n_participants: int | None = Field(default=None, ge=0)
    demographics: dict[str, JsonValue] = Field(default_factory=dict)
    selection_mechanism: str = ""
    effects: tuple[HierarchicalEffect, ...] = ()
    #: Smallest subgroup for which calibration is claimed.
    min_subgroup_n: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check(self) -> "PopulationStructure":
        ids = [u.id for u in self.units]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate lineage units: {sorted(dupes)}")
        return self

    def unidentified_effects(self) -> tuple[HierarchicalEffect, ...]:
        return tuple(e for e in self.effects if not e.is_identified)


class SignalSpec(SchemaModel):
    """Native signal and units (Appendix B row 4)."""

    name: str
    units: Unit
    #: Recording reference (e.g. "average", "linked_mastoid", "none").
    reference: str = ""
    dynamic_range: tuple[float, float] | None = None
    saturation: float | None = None
    quantization: float | None = None
    n_channels: int | None = Field(default=None, gt=0)
    #: Whether an amplitude/gain calibration exists.  Required when ``units``
    #: are arbitrary, otherwise the number has no physical meaning (R01).
    amplitude_calibration: CalibrationManifest | None = None
    #: True when preprocessing irreversibly removed the target signal.
    irreversible_aggregation: bool = False

    @model_validator(mode="after")
    def _check(self) -> "SignalSpec":
        if self.dynamic_range is not None and self.dynamic_range[1] <= self.dynamic_range[0]:
            raise ValueError(f"signal {self.name!r} dynamic range must be ordered")
        return self

    @property
    def units_are_meaningful(self) -> bool:
        return (not self.units.is_arbitrary) or self.amplitude_calibration is not None


class ObservationModel(SchemaModel):
    """Forward physics and what kind of factor this source contributes."""

    name: str
    forward_physics: str
    observed_variables: tuple[str, ...] = ()
    preprocessing: tuple[str, ...] = ()
    #: A generative likelihood defines a noise model. An agreement penalty does
    #: not - treating one as the other is refusal R09.
    likelihood_kind: LikelihoodKind = "generative"
    noise_model: Prior | None = None
    calibration_status: Literal[
        "calibrated_empirically", "uncalibrated", "promoted_generative"
    ] = "uncalibrated"
    #: Whether preprocessing was checked for label leakage.
    leakage_checked: bool = False
    #: Port names this observation model writes into.
    target_ports: tuple[str, ...] = ()
    ledger: UncertaintyLedger | None = None

    @property
    def is_generative(self) -> bool:
        return self.likelihood_kind == "generative" or (
            self.calibration_status == "promoted_generative"
        )

    @property
    def is_calibrated_pseudo(self) -> bool:
        return self.calibration_status == "calibrated_empirically"


class SafeSet(SchemaModel):
    """``A_safe``: the independently validated feasible set for interventions.

    "A learned objective cannot guarantee device, exposure, or protocol
    safety" - refusal R11.
    """

    id: str
    #: name -> (min, max) in ``constraint_units[name]``.
    constraints: dict[str, tuple[float, float]] = Field(default_factory=dict)
    constraint_units: dict[str, Unit] = Field(default_factory=dict)
    #: Validated by a party independent of the optimizer.
    independently_validated: bool = False
    validator: str | None = None
    #: Reference to the independent validation of this envelope.
    review_reference: str | None = None
    #: What the system does when the optimum lies outside the set.
    deferral_policy: Literal["defer", "clip", "refuse", "none"] = "refuse"

    @model_validator(mode="after")
    def _check(self) -> "SafeSet":
        for name, (lo, hi) in self.constraints.items():
            if hi < lo:
                raise ValueError(f"safe-set constraint {name!r} is inverted")
            if name not in self.constraint_units:
                raise ValueError(f"safe-set constraint {name!r} has no declared units")
        return self

    @property
    def is_usable(self) -> bool:
        return (
            bool(self.constraints)
            and self.independently_validated
            and self.validator is not None
            and self.deferral_policy in ("defer", "refuse")
        )

    def contains(self, u: dict[str, float]) -> bool:
        for name, value in u.items():
            bounds = self.constraints.get(name)
            if bounds is None:
                return False
            if not (bounds[0] <= value <= bounds[1]):
                return False
        return True


class InterventionModel(SchemaModel):
    """Write operator: dose, pose, field solution and safety envelope."""

    name: str
    modality: Literal[
        "tms", "tfus", "tes", "sensory", "cognitive_task", "neurofeedback",
        "pharmacological", "body_state", "simulated_impulse",
    ]
    waveform: str = ""
    dose: dict[str, float] = Field(default_factory=dict)
    dose_units: dict[str, Unit] = Field(default_factory=dict)
    pose_frame: str | None = None
    field_solver: str | None = None
    target_ports: tuple[str, ...] = ()
    sham: str | None = None
    randomized: bool = False
    a_safe: SafeSet | None = None
    #: Whether dose and pose were calibrated independently of the neural model.
    dose_independently_calibrated: bool = False
    ledger: UncertaintyLedger | None = None

    @model_validator(mode="after")
    def _check(self) -> "InterventionModel":
        for name in self.dose:
            if name not in self.dose_units:
                raise ValueError(f"intervention dose {name!r} has no declared units")
        return self

    @property
    def is_physical_stimulation(self) -> bool:
        return self.modality in ("tms", "tfus", "tes")


class Missingness(SchemaModel):
    """Missing/censored data policy (Appendix B row 9).

    ARCHITECTURE.md rule 1 - "missing data is never imputed as zero or as an
    average-brain label" - is enforced by the *type*: zero/mean imputation is
    not a representable value of ``handling``.
    """

    mechanism: Literal["mcar", "mar", "mnar", "planned", "unknown"] = "unknown"
    handling: Literal[
        "marginalize",
        "mask",
        "inverse_probability_weight",
        "design_factor",
        "missingness_model",
        "drop_episode",
    ] = "mask"
    planned_missing_channels: tuple[str, ...] = ()
    unplanned_missing_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    artifact_rejection: str = ""
    dropout: float | None = Field(default=None, ge=0.0, le=1.0)
    attrition: float | None = Field(default=None, ge=0.0, le=1.0)
    #: True when selection depends on the outcome (post-outcome selection).
    post_outcome_selection: bool = False
    #: True when censoring is correlated with unmodeled state.
    censoring_correlated_with_state: bool = False

    @model_validator(mode="after")
    def _check(self) -> "Missingness":
        if self.mechanism == "mnar" and self.handling in ("mask", "drop_episode"):
            raise ValueError(
                "MNAR missingness cannot be handled by masking alone; use a "
                "missingness model, weighting, or a design factor"
            )
        return self


class GradientPermission(SchemaModel):
    """``A_k``: named modules, ports, scales and parameter groups a source may update."""

    modules: tuple[str, ...] = ()
    ports: tuple[str, ...] = ()
    scales: tuple[str, ...] = ()
    parameter_groups: tuple[str, ...] = ()
    frozen: tuple[str, ...] = ()
    evaluation_only: bool = False
    #: Adapters must be declared; a loss reaching an unobserved module through
    #: an undeclared adapter is an Appendix B rejection condition.
    adapters: tuple[str, ...] = ()
    #: Maximum loss weight this source may contribute (evidence, not row count).
    max_weight: float | None = Field(default=None, gt=0.0)
    notes: str = ""

    @model_validator(mode="after")
    def _check(self) -> "GradientPermission":
        overlap = set(self.frozen) & (
            set(self.modules) | set(self.parameter_groups) | set(self.ports)
        )
        if overlap:
            raise ValueError(
                f"gradient permission both grants and freezes {sorted(overlap)}"
            )
        if self.evaluation_only and (
            self.modules or self.ports or self.parameter_groups or self.scales
        ):
            raise ValueError(
                "an evaluation-only source must not name updatable targets"
            )
        return self

    @property
    def grants_anything(self) -> bool:
        return bool(self.modules or self.ports or self.scales or self.parameter_groups)


class TemporalHoldout(SchemaModel):
    """A time-based barrier: later data may not train on earlier predictions."""

    key: str = "session_start"
    #: Boundary in seconds relative to the source epoch.
    boundary: float
    direction: Literal["before_is_train", "after_is_train"] = "before_is_train"


class SplitPolicy(SchemaModel):
    """``R_k``: grouping keys, temporal holdout and fold assignment."""

    grouping_keys: tuple[str, ...] = ()
    temporal_holdout: TemporalHoldout | None = None
    #: lineage unit id -> fold.
    fold_assignments: dict[str, Literal["train", "val", "test", "holdout"]] = Field(
        default_factory=dict
    )
    #: The leakage barrier used before splitting.  Only ``parent_lineage``
    #: satisfies rule 6 / R10.
    leakage_barrier: Literal["parent_lineage", "random", "temporal_only", "none"] = (
        "parent_lineage"
    )
    #: Roles must be fixed before final evaluation.
    role_locked: bool = False
    #: Whether stimuli are repeated across held-out identity tests.
    stimuli_shared_across_folds: bool = False


class SourceCard(SchemaModel):
    """Appendix B: D_k = (I,G,P,S,T,C,O,U,M,B,V,Delta,A,R)."""

    identity: Identity
    governance: Governance
    population: PopulationStructure
    spatial: Support
    temporal: TemporalSupport
    calibration: CalibrationManifest
    observation: ObservationModel | None = None
    intervention: InterventionModel | None = None
    missingness: Missingness
    ledger: UncertaintyLedger
    gradient_permission: GradientPermission  # A_k : named modules/ports/scales
    role: SourceRole
    split_policy: SplitPolicy  # R_k : grouping keys, temporal holdout

    # -- additions used by the compiler --------------------------------------
    #: Native signal declaration (Appendix B row 4).
    signal: SignalSpec | None = None
    label: str = ""
    #: Effective independent evidence, used for loss weighting.  "Weights depend
    #: on effective independent evidence and declared reliability, not row
    #: count" (thesis eq. T6 discussion).
    effective_n: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> "SourceCard":
        if self.observation is None and self.intervention is None and self.role in (
            "likelihood",
            "boundary_target",
        ):
            raise ValueError(
                f"source {self.id!r} has role {self.role!r} but declares neither an "
                "observation nor an intervention model"
            )
        if self.role == "evaluation_only" and not self.gradient_permission.evaluation_only:
            raise ValueError(
                f"source {self.id!r} has role 'evaluation_only' but its gradient "
                "permission does not"
            )
        return self

    @property
    def id(self) -> str:
        return self.identity.persistent_id

    @property
    def receives_gradients(self) -> bool:
        return (
            self.role not in ("evaluation_only", "negative_control")
            and not self.gradient_permission.evaluation_only
            and self.gradient_permission.grants_anything
        )
