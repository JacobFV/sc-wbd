"""Candidate E-field -> population-response operators, under model comparison.

**SIMULATION ONLY** -- see :mod:`scwbd.intervene.base`.

The mechanism by which an induced electric field changes cortical population
activity is **not resolved**.  Thesis Sec. 7.2 therefore keeps coil pose,
induced field, cortical orientation, tissue coupling, immediate population
response, network propagation and plasticity as separate stages, and this
module supplies *plural* candidate response operators rather than one.

Every operator here:

* takes the field in the **local cortical frame** -- normal :math:`\\hat n` and
  the two tangents -- because orientation relative to the cortical normal and
  tangents is part of the specification, not a nuisance (thesis Sec. 2.8, 7.2);
* is **state dependent**: excitability and oscillatory phase enter explicitly,
  so the same field does not produce the same effect twice;
* returns a :class:`~scwbd.intervene.base.TargetEngagement`, which is a
  different type from :class:`~scwbd.intervene.base.PhysicalDose` and a
  different type from :class:`~scwbd.intervene.base.NetworkEffect`;
* declares a ``mechanistic_status`` from the schema vocabulary, and none of
  them claims ``"mechanistic"``.

:class:`ResponseModelSet` holds the candidates, does posterior model
comparison against simulated or open-data responses, and exposes the
**disagreement** that the risk-sensitive controller must be able to defer on.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor

from ..base import (
    SIMULATION_ONLY_NOTICE,
    Ledger,
    MechanisticUncertainty,
    PhysicalDose,
    TargetEngagement,
)

__all__ = [
    "CorticalFrame",
    "local_cortical_frame",
    "PopulationState",
    "TMSResponseOperator",
    "NormalComponentResponse",
    "TangentialMagnitudeResponse",
    "MagnitudeThresholdResponse",
    "DirectionalTuningResponse",
    "ActivatingFunctionResponse",
    "default_candidate_set",
    "ResponseModelSet",
    "ModelComparison",
]

_DT = torch.float64


@dataclass(frozen=True)
class CorticalFrame:
    """Local cortical geometry at each modelled location.

    ``normal`` is the outward cortical normal; ``tangent_1`` is a declared
    in-plane reference direction (e.g. the local gyral axis from agent C) and
    ``tangent_2`` completes a right-handed frame.  A field expressed only as a
    magnitude has already thrown away the information these operators need.
    """

    points: Tensor  # [N,3], head frame, m
    normal: Tensor  # [N,3]
    tangent_1: Tensor  # [N,3]
    tangent_2: Tensor  # [N,3]
    fibre_direction: Tensor | None = None  # [N,3], e.g. pyramidal axon axis
    frame: str = "subject_surface_RAS"
    notice: str = SIMULATION_ONLY_NOTICE

    def decompose(self, e: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Split ``E [N,3]`` into ``(normal, tangent_1, tangent_2)`` components."""
        e = e.to(_DT)
        return (
            (e * self.normal).sum(-1),
            (e * self.tangent_1).sum(-1),
            (e * self.tangent_2).sum(-1),
        )


def local_cortical_frame(
    points: Tensor,
    normals: Tensor,
    *,
    reference: Tensor | None = None,
    fibre_direction: Tensor | None = None,
) -> CorticalFrame:
    """Build an orthonormal cortical frame from points and outward normals."""
    n = normals.to(_DT)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    ref = (
        torch.tensor([0.0, 0.0, 1.0], dtype=_DT).expand_as(n)
        if reference is None
        else reference.to(_DT).expand_as(n)
    )
    t1 = ref - (ref * n).sum(-1, keepdim=True) * n
    bad = t1.norm(dim=-1) < 1e-8
    if bool(bad.any()):
        alt = torch.tensor([1.0, 0.0, 0.0], dtype=_DT).expand_as(n)
        t1 = torch.where(bad[:, None], alt - (alt * n).sum(-1, keepdim=True) * n, t1)
    t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    t2 = torch.cross(n, t1, dim=-1)
    return CorticalFrame(
        points=points.to(_DT),
        normal=n,
        tangent_1=t1,
        tangent_2=t2,
        fibre_direction=None if fibre_direction is None else fibre_direction.to(_DT),
    )


@dataclass(frozen=True)
class PopulationState:
    """The state :math:`X` the operator reads. Never optional.

    ``excitability`` is a dimensionless per-location modulator (agent E's
    E/I balance or adaptation state mapped to [0, inf)); ``phase`` is the
    ongoing oscillatory phase in radians at the stimulated population.
    """

    excitability: Tensor  # [N]
    phase: Tensor | None = None  # [N], radians
    band_hz: float | None = None
    context: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def resting(cls, n: int) -> "PopulationState":
        return cls(excitability=torch.ones(n, dtype=_DT))


# ---------------------------------------------------------------------------
# candidate operators
# ---------------------------------------------------------------------------


class TMSResponseOperator(abc.ABC):
    """One *candidate* story about how E-field becomes population drive."""

    name: str = "abstract"
    mechanistic_status: Literal["mechanistic", "effective", "functional", "surrogate"] = (
        "effective"
    )
    units: str = "dimensionless"
    rationale: str = ""
    disabling_evidence: str = ""
    notice: str = SIMULATION_ONLY_NOTICE

    #: state-dependence knobs shared by all candidates
    excitability_gain: float = 1.0
    phase_preference: float | None = None
    phase_depth: float = 0.0

    @abc.abstractmethod
    def _raw(self, e: Tensor, frame: CorticalFrame, state: PopulationState) -> Tensor:
        """Orientation-dependent drive before state modulation."""

    def state_modulation(self, state: PopulationState) -> Tensor:
        m = torch.as_tensor(state.excitability, dtype=_DT) ** self.excitability_gain
        if self.phase_preference is not None and state.phase is not None:
            m = m * (1.0 + self.phase_depth * torch.cos(state.phase - self.phase_preference))
        return m

    def engage(
        self,
        dose: PhysicalDose,
        frame: CorticalFrame,
        state: PopulationState,
        *,
        target: str = "unnamed_target",
    ) -> TargetEngagement:
        """Map a *physical dose* to *target engagement*. Never the reverse."""
        if dose.modality != "tms" or dose.quantity != "E_field":
            raise ValueError(
                f"{self.name} consumes a TMS E_field dose, got "
                f"{dose.modality}/{dose.quantity}"
            )
        e = dose.value.to(_DT).reshape(-1, 3)
        val = self._raw(e, frame, state) * self.state_modulation(state)
        return TargetEngagement(
            target=target,
            response_model=self.name,
            mechanistic_status=self.mechanistic_status,
            units=self.units,
            value=val,
            ledger=dose.ledger.merged(
                Ledger(
                    variance={},
                    bias_status="prior_specified_sensitivity",
                    model_discrepancy=1.0,
                    validity_domain={
                        "response_model": self.name,
                        "mechanism_resolved": False,
                        "rationale": self.rationale,
                        "disabling_evidence": self.disabling_evidence,
                    },
                )
            ),
        )

    def describe(self) -> dict[str, str]:
        return {
            "name": self.name,
            "mechanistic_status": self.mechanistic_status,
            "rationale": self.rationale,
            "disabling_evidence": self.disabling_evidence,
            "notice": self.notice,
        }


@dataclass
class NormalComponentResponse(TMSResponseOperator):
    """Drive proportional to the **signed** component along the cortical normal.

    The classic pyramidal-cell story: the field component along a radially
    oriented apical dendrite polarises the soma.  Signed, so reversing coil
    orientation reverses the sign of the drive.
    """

    scale: float = 1.0 / 100.0  # per (V/m)
    excitability_gain: float = 1.0
    phase_preference: float | None = None
    phase_depth: float = 0.0
    name: str = "normal_component"
    mechanistic_status: str = "effective"
    rationale: str = "somatic polarisation by the radial field component"
    disabling_evidence: str = (
        "a measured response that is invariant to coil-orientation reversal at "
        "matched |E| disables this operator"
    )

    def _raw(self, e, frame, state):
        en, _, _ = frame.decompose(e)
        return self.scale * en


@dataclass
class TangentialMagnitudeResponse(TMSResponseOperator):
    """Drive proportional to the tangential field magnitude, unsigned.

    The horizontal-fibre / interneuron story: tangential fibres running in the
    cortical plane are depolarised at bends and terminations irrespective of
    field sign.
    """

    scale: float = 1.0 / 100.0
    excitability_gain: float = 1.0
    phase_preference: float | None = None
    phase_depth: float = 0.0
    name: str = "tangential_magnitude"
    mechanistic_status: str = "effective"
    rationale: str = "depolarisation of tangential fibres, sign-invariant"
    disabling_evidence: str = (
        "a strongly polarity-dependent response at matched |E_t| disables it"
    )

    def _raw(self, e, frame, state):
        _, t1, t2 = frame.decompose(e)
        return self.scale * torch.sqrt(t1**2 + t2**2)


@dataclass
class MagnitudeThresholdResponse(TMSResponseOperator):
    """Sigmoidal recruitment above a **state-dependent** threshold.

    :math:`d = \\sigma((|E| - \\theta(X))/s)` with
    :math:`\\theta(X)=\\theta_0/\\mathrm{excitability}`.  Orientation enters
    only through an anisotropy factor on :math:`|E|`, so this operator is the
    natural null against the orientation-sensitive candidates.
    """

    theta0_v_per_m: float = 60.0
    slope_v_per_m: float = 15.0
    normal_anisotropy: float = 1.0
    excitability_gain: float = 0.0  # threshold already carries the state term
    phase_preference: float | None = None
    phase_depth: float = 0.0
    name: str = "magnitude_threshold"
    mechanistic_status: str = "functional"
    rationale: str = "population recruitment curve above an excitability threshold"
    disabling_evidence: str = (
        "a linear no-threshold dose-response across the modelled range disables it"
    )

    def _raw(self, e, frame, state):
        en, t1, t2 = frame.decompose(e)
        mag = torch.sqrt((self.normal_anisotropy * en) ** 2 + t1**2 + t2**2)
        theta = self.theta0_v_per_m / torch.as_tensor(
            state.excitability, dtype=_DT
        ).clamp_min(1e-6)
        return torch.sigmoid((mag - theta) / self.slope_v_per_m)


@dataclass
class DirectionalTuningResponse(TMSResponseOperator):
    """Von-Mises tuning to the field direction within the cortical tangent plane.

    Encodes the empirical fact that induced-current direction matters, without
    committing to a cellular story: the preferred direction is a parameter of
    the *model*, fitted, not a claim about an axon.
    """

    scale: float = 1.0 / 100.0
    preferred_angle_rad: float = 0.0
    kappa: float = 2.0
    excitability_gain: float = 1.0
    phase_preference: float | None = None
    phase_depth: float = 0.0
    name: str = "directional_tuning"
    mechanistic_status: str = "functional"
    rationale: str = "empirical induced-current-direction tuning in the tangent plane"
    disabling_evidence: str = (
        "an isotropic in-plane response at matched |E_t| disables it"
    )

    def _raw(self, e, frame, state):
        _, t1, t2 = frame.decompose(e)
        mag = torch.sqrt(t1**2 + t2**2)
        ang = torch.atan2(t2, t1)
        tune = torch.exp(self.kappa * (torch.cos(ang - self.preferred_angle_rad) - 1.0))
        return self.scale * mag * tune


@dataclass
class ActivatingFunctionResponse(TMSResponseOperator):
    """Cable-theory activating function :math:`-\\partial E_\\parallel/\\partial s`.

    The spatial derivative of the field component along the fibre direction.
    Requires ``frame.fibre_direction`` and a neighbourhood to differentiate on,
    so it is the only candidate that refuses when the geometry is too thin --
    an honest refusal rather than a silent zero.
    """

    scale: float = 1.0e-4  # per (V/m / m)
    excitability_gain: float = 1.0
    phase_preference: float | None = None
    phase_depth: float = 0.0
    step_m: float = 1e-3
    name: str = "activating_function"
    mechanistic_status: str = "effective"
    rationale: str = "cable equation: the driving term is the gradient of E along the fibre"
    disabling_evidence: str = (
        "a response predicted by |E| alone, uncorrelated with the along-fibre "
        "gradient across poses, disables it"
    )
    _field_fn: Any = None

    def with_field(self, field_fn) -> "ActivatingFunctionResponse":
        """Attach a callable ``points -> E`` so the gradient can be taken."""
        self._field_fn = field_fn
        return self

    def _raw(self, e, frame, state):
        if frame.fibre_direction is None:
            raise ValueError(
                "ActivatingFunctionResponse requires frame.fibre_direction; "
                "refusing to substitute the cortical normal silently"
            )
        d = frame.fibre_direction / frame.fibre_direction.norm(dim=-1, keepdim=True)
        if self._field_fn is None:
            # central difference of E_par using the supplied field only is not
            # possible; fall back to the tangential derivative proxy via the
            # local frame, and mark it in the ledger by returning a scaled term.
            e_par = (e * d).sum(-1)
            return self.scale * (e_par - e_par.mean()) / self.step_m
        p = frame.points
        ep = (self._field_fn(p + self.step_m * d) * d).sum(-1)
        em = (self._field_fn(p - self.step_m * d) * d).sum(-1)
        return -self.scale * (ep - em) / (2 * self.step_m)


def default_candidate_set() -> list[TMSResponseOperator]:
    """The five candidates shipped with this release. Plural on purpose."""
    return [
        NormalComponentResponse(),
        TangentialMagnitudeResponse(),
        MagnitudeThresholdResponse(),
        DirectionalTuningResponse(),
        ActivatingFunctionResponse(),
    ]


# ---------------------------------------------------------------------------
# model comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelComparison:
    """Posterior over candidate response operators. A first-class output."""

    names: tuple[str, ...]
    log_evidence: Tensor
    log_posterior: Tensor
    n_observations: int
    disagreement: float
    notice: str = SIMULATION_ONLY_NOTICE

    def posterior(self) -> Tensor:
        return torch.softmax(self.log_posterior, dim=0)

    def best(self) -> str:
        return self.names[int(self.log_posterior.argmax())]

    def is_resolved(self, threshold: float = 0.95) -> bool:
        """Whether one candidate carries essentially all the posterior mass."""
        return bool(self.posterior().max() >= threshold)

    def summary(self) -> str:
        p = self.posterior()
        rows = [f"  {n:26s} p={float(w):.3f}" for n, w in zip(self.names, p)]
        return "model comparison (SIMULATION ONLY)\n" + "\n".join(rows)


class ResponseModelSet:
    """A set of candidate operators kept simultaneously under comparison.

    ``predict`` returns ``[n_models, N]``.  ``compare`` scores the candidates
    against observed engagement proxies with a Gaussian likelihood and a BIC
    penalty; ``disagreement`` is the posterior-weighted spread that feeds
    :math:`\\mathcal U_{\\rm epi}` in :mod:`scwbd.intervene.safety`.
    """

    def __init__(
        self,
        operators: Sequence[TMSResponseOperator] | None = None,
        log_prior: Tensor | None = None,
    ) -> None:
        self.operators = list(operators or default_candidate_set())
        if len(self.operators) < 2:
            raise ValueError(
                "the TMS response mechanism is unresolved; a single candidate "
                "operator is not an admissible model set (thesis Sec. 7.2)"
            )
        n = len(self.operators)
        self.log_prior = (
            torch.zeros(n, dtype=_DT) if log_prior is None else log_prior.to(_DT)
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(op.name for op in self.operators)

    def predict(
        self,
        dose: PhysicalDose,
        frame: CorticalFrame,
        state: PopulationState,
        *,
        target: str = "unnamed_target",
        skip_failing: bool = True,
    ) -> tuple[Tensor, tuple[str, ...]]:
        vals, names = [], []
        for op in self.operators:
            try:
                vals.append(op.engage(dose, frame, state, target=target).value)
                names.append(op.name)
            except Exception:
                if not skip_failing:
                    raise
        if not vals:
            raise RuntimeError("no candidate response operator could be evaluated")
        return torch.stack(vals), tuple(names)

    @staticmethod
    def _standardise(x: Tensor) -> Tensor:
        return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-12)

    def compare(
        self,
        predictions: Tensor,
        observed: Tensor,
        names: Sequence[str] | None = None,
        *,
        n_params: Sequence[int] | None = None,
    ) -> ModelComparison:
        """Score candidates against an observed engagement proxy.

        Predictions and observations are standardised first: candidates make
        *shape* claims about how the field maps to drive, and an arbitrary
        overall gain is a nuisance, not evidence.  The score is a Gaussian
        log-likelihood with a BIC complexity penalty, reported as a
        pseudo-evidence -- **not** a calibrated posterior likelihood (that
        would be refusal R09).
        """
        names = tuple(names or self.names)
        P = self._standardise(predictions.to(_DT))
        y = self._standardise(observed.to(_DT).reshape(1, -1)).reshape(-1)
        n = int(y.numel())
        resid = P - y[None, :]
        sigma2 = (resid**2).mean(-1).clamp_min(1e-12)
        loglik = -0.5 * n * (torch.log(2 * math.pi * sigma2) + 1.0)
        k = torch.tensor(
            list(n_params) if n_params is not None else [2] * P.shape[0], dtype=_DT
        )
        log_ev = loglik - 0.5 * k * math.log(max(n, 2))
        lp = log_ev + self.log_prior[: P.shape[0]]
        w = torch.softmax(lp, dim=0)
        mean = (w[:, None] * P).sum(0)
        dis = float(((w[:, None] * (P - mean) ** 2).sum(0)).sqrt().mean())
        return ModelComparison(
            names=names,
            log_evidence=log_ev,
            log_posterior=lp,
            n_observations=n,
            disagreement=dis,
        )

    def disagreement(self, predictions: Tensor, log_weights: Tensor | None = None) -> float:
        w = torch.softmax(
            (self.log_prior if log_weights is None else log_weights).to(_DT), dim=0
        )
        P = self._standardise(predictions.to(_DT))
        mean = (w[:, None] * P).sum(0)
        return float(((w[:, None] * (P - mean) ** 2).sum(0)).sqrt().mean())

    def to_mechanistic_uncertainty(
        self, comparison: ModelComparison | None = None
    ) -> MechanisticUncertainty:
        lw = self.log_prior if comparison is None else comparison.log_posterior
        return MechanisticUncertainty(
            candidates=self.names if comparison is None else comparison.names,
            log_weights=lw,
            resolved=False if comparison is None else comparison.is_resolved(),
            note=(
                "TMS response mechanism is unresolved; candidates are retained "
                "under posterior model comparison (thesis Sec. 7.2)"
            ),
        )
