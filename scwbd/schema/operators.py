"""Typed operators: the edges of the simulator graph (thesis sec. 2.1-2.2).

An edge is **not a scalar weight**.  It is a typed map between declared boundary
spaces that records its operator family, evidence class, mechanistic status,
identification basis, delay prior, free parameters, residual policy, semigroup
behaviour and bias-variance ledger.

Three refusals read directly off this object:

* **R04** - an operator whose ``mechanistic_status`` makes a causal claim while
  its ``identification.basis`` is passive correlation alone.
* **R05** - a learned residual permitted to dominate a named mechanistic term
  without a preregistered ``rho_max`` on a declared validity set.
* **R06** - a learned propagator that permits adaptive stepping without
  semigroup testing at every permitted step pair.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .base import SchemaModel
from .ids import ClockId
from .ledger import UncertaintyLedger
from .priors import Prior
from .units import Unit

__all__ = [
    "OperatorFamily",
    "EvidenceClass",
    "MechanisticStatus",
    "IdentificationBasis",
    "Identification",
    "ResidualPolicy",
    "SemigroupTest",
    "OperatorSpec",
]

OperatorFamily = Literal[
    "flow_ode",
    "flow_pde",
    "field_kernel",
    "convolution",
    "delayed_ssm",
    "spectral_transfer",
    "attention",
    "point_process",
    "surrogate",
    "composition",
]

#: Thesis sec. 2.2 compilation classes. A zero mask creates exact independence
#: across a *direct edge inside the compiled model*; it is not a claim of
#: biological independence.
EvidenceClass = Literal["hard", "soft", "proposed"]

#: What kind of claim the edge carries. "mechanistic" and "effective" are
#: causal claims and are therefore subject to R04.
MechanisticStatus = Literal["mechanistic", "effective", "functional", "surrogate"]

IdentificationBasis = Literal[
    "passive_correlation",
    "intervention",
    "randomization",
    "assumption_set",
    "anatomical_prior",
    "simulation_ground_truth",
    "instrumental_variable",
]

#: Statuses that assert a causal/generative claim about the edge.
CAUSAL_STATUSES: frozenset[str] = frozenset({"mechanistic", "effective"})

#: Bases that can, on their own, support a causal claim.
CAUSAL_BASES: frozenset[str] = frozenset(
    {"intervention", "randomization", "instrumental_variable", "simulation_ground_truth"}
)


class Identification(SchemaModel):
    """How the operator's value is identified from evidence.

    "Common input, observation mixing, and omitted state remain alternatives"
    is the reason a correlation-only edge cannot be called effective or
    mechanistic (thesis Table tab:compiler-refusals, row 4).
    """

    basis: tuple[IdentificationBasis, ...] = ()
    #: Named assumption set sufficient for the causal claim (e.g. a DAG plus
    #: no-unmeasured-confounding, or a stated instrument validity condition).
    assumption_set: str | None = None
    #: Source cards that supply an identified intervention for this edge.
    intervention_source_ids: tuple[str, ...] = ()
    #: Source cards supplying the estimate at all (provenance).
    evidence_source_ids: tuple[str, ...] = ()
    notes: str = ""

    @property
    def supports_causal_claim(self) -> bool:
        if set(self.basis) & CAUSAL_BASES:
            return True
        if self.intervention_source_ids:
            return True
        if self.assumption_set:
            return True
        return False

    @property
    def is_passive_only(self) -> bool:
        return bool(self.basis) and set(self.basis).issubset({"passive_correlation"})


class ResidualPolicy(SchemaModel):
    """Preregistered gain ratio for a learned residual on a mechanistic term.

    ``||R_theta|| / ||F_mech|| <= rho_max`` on the declared validity set
    (thesis Table tab:compiler-refusals, row 5).  Without ``rho_max`` the named
    mechanism becomes unfalsifiable, so the compiler refuses (R05).
    """

    #: Preregistered maximum energy/gain ratio.
    rho_max: float | None = Field(default=None, gt=0.0)
    #: Measured ratio on ``validity_set``; None means "not measured yet".
    measured_ratio: float | None = Field(default=None, ge=0.0)
    #: Named set on which the ratio is enforced (a dataset/regime identifier).
    validity_set: str | None = None
    #: Whether violations are reported rather than silently clipped.
    report_violations: bool = True
    #: Whether the module has been reclassified as a surrogate after violation.
    reclassified_as_surrogate: bool = False
    norm: Literal["l2_energy", "operator_gain", "spectral"] = "l2_energy"

    @property
    def violates(self) -> bool:
        if self.rho_max is None or self.measured_ratio is None:
            return False
        return self.measured_ratio > self.rho_max

    @property
    def is_preregistered(self) -> bool:
        return self.rho_max is not None and self.validity_set is not None


class SemigroupTest(SchemaModel):
    """Semigroup residual record for a learned propagator (thesis sec. 4.5).

    ``eps_sg(d1,d2;x) = || Phi^{d1+d2}(x) - Phi^{d2}(Phi^{d1}(x)) ||_W``
    measured at every permitted step pair.  Adaptive stepping above tolerance
    is disabled (R06); below tolerance the empirical distribution enters the
    numerical and model-discrepancy budget.
    """

    #: Step pairs (d1, d2) in seconds at which the residual was measured.
    tested_step_pairs: tuple[tuple[float, float], ...] = ()
    #: Worst residual over the tested pairs, in ``units``.
    max_residual: float | None = Field(default=None, ge=0.0)
    tolerance: float | None = Field(default=None, gt=0.0)
    units: Unit = Unit("dimensionless")
    #: Step pairs the scheduler is allowed to use.
    permitted_step_pairs: tuple[tuple[float, float], ...] = ()
    #: Whether the residual distribution is folded into prediction intervals.
    folded_into_intervals: bool = False

    @property
    def is_tested(self) -> bool:
        return bool(self.tested_step_pairs) and self.max_residual is not None and self.tolerance is not None

    @property
    def within_tolerance(self) -> bool:
        if self.max_residual is None or self.tolerance is None:
            return False
        return self.max_residual <= self.tolerance

    def untested_permitted_pairs(self) -> tuple[tuple[float, float], ...]:
        tested = set(self.tested_step_pairs)
        return tuple(p for p in self.permitted_step_pairs if p not in tested)


class OperatorSpec(SchemaModel):
    """A typed edge. NOT a scalar weight (thesis sec. 2.2)."""

    src: str
    dst: str
    family: OperatorFamily
    evidence_class: EvidenceClass
    mechanistic_status: MechanisticStatus
    delay_prior: Prior
    params: dict[str, Prior] = Field(default_factory=dict)
    ledger: UncertaintyLedger

    # -- additions used by the compiler --------------------------------------
    id: str = ""
    label: str = ""
    #: Boundary components read on the source side / written on the dest side.
    src_port: str | None = None
    dst_port: str | None = None
    #: Clock this operator is dispatched on; defaults to the destination
    #: region's fastest component clock when omitted.
    clock: ClockId | None = None
    #: True for operators with learned parameters (surrogates, attention,
    #: neural-operator kernels).  Drives R06.
    is_learned: bool = False
    identification: Identification = Field(default_factory=Identification)
    residual: ResidualPolicy | None = None
    semigroup: SemigroupTest | None = None
    #: Euclidean/tract distance in metres; proposed edges pay a distance and
    #: provenance penalty in model comparison (thesis sec. 2.2).
    distance_m: float | None = Field(default=None, ge=0.0)
    #: Named gate (phase, neuromodulator, task context) that may deactivate the
    #: edge; "a permitted edge does not imply it is causally active".
    gating: str | None = None
    differentiable: bool = True
    solver: str | None = None
    units_in: Unit | None = None
    units_out: Unit | None = None

    @model_validator(mode="after")
    def _check(self) -> "OperatorSpec":
        # A delay prior with an explicitly negative bounded support is a type
        # error (acausal conduction).  Unbounded families are left alone: they
        # are a truncation question for the inference layer, not a schema one.
        if self.delay_prior.kind in ("uniform", "dirac"):
            if self.delay_prior.support()[0] < 0.0:
                raise ValueError(
                    f"operator {self.key} declares a negative conduction delay "
                    f"({self.delay_prior.support()[0]} s)"
                )
        if self.delay_prior.units.dimension != Unit("s").dimension:
            raise ValueError(
                f"operator {self.key} delay prior must be in units of time, got "
                f"{str(self.delay_prior.units)!r}"
            )
        if self.residual is not None and not self.is_learned:
            raise ValueError(
                f"operator {self.key} declares a learned residual but is_learned=False"
            )
        return self

    @property
    def key(self) -> str:
        return self.id or f"{self.src}->{self.dst}:{self.family}"

    @property
    def is_self_edge(self) -> bool:
        return self.src == self.dst

    @property
    def claims_causality(self) -> bool:
        return self.mechanistic_status in CAUSAL_STATUSES
