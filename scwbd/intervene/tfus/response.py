"""Candidate tFUS tissue-coupling operators, under posterior model comparison.

**SIMULATION ONLY** -- see :mod:`scwbd.intervene.base`.

Thesis Sec. 7.2: *"Because low-intensity neuromodulatory mechanisms remain
incompletely resolved, multiple tissue operators should coexist under posterior
model comparison."*  This module therefore ships five mutually incompatible
stories about how acoustic pressure becomes population drive, including a
**null** operator in which the direct effect is zero and the observed response
is an auditory confound.  Omitting that candidate would be the single most
consequential modelling error available here.

None of the candidates claims ``mechanistic_status="mechanistic"``.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor

from ..base import (
    SIMULATION_ONLY_NOTICE,
    Ledger,
    MechanisticUncertainty,
    PhysicalDose,
    TargetEngagement,
)
from .acoustics import BRAIN, AcousticMedium

__all__ = [
    "TFUSResponseOperator",
    "IntramembraneCavitationResponse",
    "MechanosensitiveChannelResponse",
    "RadiationForceResponse",
    "ThermalResponse",
    "AuditoryConfoundNullResponse",
    "default_tfus_candidate_set",
    "TFUSResponseModelSet",
]

_DT = torch.float64


@dataclass(frozen=True)
class TissueContext:
    """Local tissue and state the coupling operator reads."""

    medium: AcousticMedium = BRAIN
    frequency_hz: float = 500e3
    duty_cycle: float = 0.5
    excitability: Tensor | float = 1.0
    temperature_rise_c: Tensor | float = 0.0
    audible: bool = True  # tFUS pulsing is audible; the confound is on by default


class TFUSResponseOperator(abc.ABC):
    """One candidate story about pressure -> population drive."""

    name: str = "abstract"
    mechanistic_status: Literal["mechanistic", "effective", "functional", "surrogate"] = (
        "effective"
    )
    units: str = "dimensionless"
    rationale: str = ""
    disabling_evidence: str = ""
    notice: str = SIMULATION_ONLY_NOTICE

    @abc.abstractmethod
    def _raw(self, p: Tensor, ctx: TissueContext) -> Tensor:
        ...

    def engage(
        self, dose: PhysicalDose, ctx: TissueContext, *, target: str = "unnamed_target"
    ) -> TargetEngagement:
        if dose.modality != "tfus":
            raise ValueError(f"{self.name} consumes a tFUS dose, got {dose.modality}")
        val = self._raw(dose.value.to(_DT), ctx)
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
class IntramembraneCavitationResponse(TFUSResponseOperator):
    """Reduced intramembrane-cavitation (NICE-like) coupling.

    Membrane capacitance is modulated by bilayer deflection, which scales
    superlinearly with acoustic pressure amplitude and depends on the pulsing
    duty cycle.  Reduced to a power law with a duty-cycle factor; the full NICE
    model is a different object and is not claimed here.
    """

    p_ref_pa: float = 3e5
    exponent: float = 2.0
    duty_gain: float = 1.0
    name: str = "intramembrane_cavitation"
    mechanistic_status: str = "effective"
    rationale: str = "bilayer capacitance modulation, superlinear in pressure amplitude"
    disabling_evidence: str = (
        "a linear pressure dose-response, or a response independent of duty "
        "cycle at matched I_SPTA, disables it"
    )

    def _raw(self, p, ctx):
        return (
            (p / self.p_ref_pa) ** self.exponent
            * (1.0 + self.duty_gain * ctx.duty_cycle)
            * torch.as_tensor(ctx.excitability, dtype=_DT)
        )


@dataclass
class MechanosensitiveChannelResponse(TFUSResponseOperator):
    """Direct gating of mechanosensitive channels above a pressure threshold."""

    threshold_pa: float = 2.5e5
    slope_pa: float = 5e4
    name: str = "mechanosensitive_channel"
    mechanistic_status: str = "effective"
    rationale: str = "pressure-gated channel opening with a recruitment threshold"
    disabling_evidence: str = (
        "a graded sub-threshold response, or abolition by channel blockade that "
        "leaves the response intact, disables it"
    )

    def _raw(self, p, ctx):
        return torch.sigmoid((p - self.threshold_pa) / self.slope_pa) * torch.as_tensor(
            ctx.excitability, dtype=_DT
        )


@dataclass
class RadiationForceResponse(TFUSResponseOperator):
    """Acoustic radiation force: drive proportional to absorbed intensity.

    :math:`F = 2\\alpha I/c`, so the drive follows :math:`p^2` but -- unlike
    cavitation -- scales with the **absorption coefficient**, giving a
    frequency dependence the other candidates do not share.
    """

    scale: float = 1.0
    name: str = "radiation_force"
    mechanistic_status: str = "effective"
    rationale: str = "momentum transfer from absorbed acoustic intensity"
    disabling_evidence: str = (
        "a response that does not scale with tissue absorption across "
        "frequencies at matched pressure disables it"
    )

    def _raw(self, p, ctx):
        I = p**2 / (2 * ctx.medium.impedance)
        a = ctx.medium.alpha_np_per_m(ctx.frequency_hz)
        F = 2 * a * I / ctx.medium.sound_speed_m_per_s
        return self.scale * F / F.max().clamp_min(1e-30)


@dataclass
class ThermalResponse(TFUSResponseOperator):
    """Drive proportional to local temperature rise.

    For low-intensity tFUS the modelled rise is small, so this candidate
    usually predicts a near-null effect -- which is precisely why it must be in
    the set: if it fits, the "neuromodulation" was heating.
    """

    per_degree: float = 1.0
    name: str = "thermal"
    mechanistic_status: str = "effective"
    rationale: str = "temperature dependence of channel kinetics and excitability"
    disabling_evidence: str = (
        "a response present at matched temperature rise but different pressure, "
        "or absent at matched temperature rise, disables it"
    )

    def _raw(self, p, ctx):
        dT = torch.as_tensor(ctx.temperature_rise_c, dtype=_DT)
        return (self.per_degree * dT).expand_as(p).clone()


@dataclass
class AuditoryConfoundNullResponse(TFUSResponseOperator):
    """**Null**: no direct tissue effect; the response is the audible pulsing.

    Depends on the pulse *envelope* (through the duty cycle), not on the focal
    pressure, and vanishes when the stimulus is inaudible.  A candidate set
    without this operator cannot distinguish neuromodulation from a startle
    response, and any comparison that omits it is not a comparison.
    """

    gain: float = 1.0
    name: str = "auditory_confound_null"
    mechanistic_status: str = "surrogate"
    rationale: str = "indirect auditory/startle response to the audible pulse envelope"
    disabling_evidence: str = (
        "an equal response under a matched-audibility sham, or a preserved "
        "response with the auditory pathway controlled, disables it"
    )

    def _raw(self, p, ctx):
        if not ctx.audible:
            return torch.zeros_like(p)
        # spatially flat: an auditory response does not follow the acoustic focus
        return torch.full_like(p, self.gain * ctx.duty_cycle)


def default_tfus_candidate_set() -> list[TFUSResponseOperator]:
    return [
        IntramembraneCavitationResponse(),
        MechanosensitiveChannelResponse(),
        RadiationForceResponse(),
        ThermalResponse(),
        AuditoryConfoundNullResponse(),
    ]


class TFUSResponseModelSet:
    """Candidates held simultaneously, compared, and never silently collapsed."""

    def __init__(
        self,
        operators: Sequence[TFUSResponseOperator] | None = None,
        log_prior: Tensor | None = None,
    ) -> None:
        self.operators = list(operators or default_tfus_candidate_set())
        if len(self.operators) < 2:
            raise ValueError(
                "low-intensity tFUS mechanism is unresolved; a single candidate "
                "tissue operator is not an admissible model set (thesis Sec. 7.2)"
            )
        if not any(
            isinstance(o, AuditoryConfoundNullResponse) for o in self.operators
        ):
            raise ValueError(
                "the tFUS candidate set must contain a null/auditory-confound "
                "operator; without it a fitted 'neuromodulatory' effect is not "
                "distinguishable from a startle response"
            )
        self.log_prior = (
            torch.zeros(len(self.operators), dtype=_DT)
            if log_prior is None
            else log_prior.to(_DT)
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.operators)

    def predict(
        self, dose: PhysicalDose, ctx: TissueContext, *, target: str = "unnamed_target"
    ) -> Tensor:
        return torch.stack([o.engage(dose, ctx, target=target).value for o in self.operators])

    @staticmethod
    def _standardise(x: Tensor) -> Tensor:
        return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-12)

    def compare(self, predictions: Tensor, observed: Tensor) -> Tensor:
        """Pseudo-evidence log weights. Not a calibrated posterior (cf. R09)."""
        P = self._standardise(predictions.to(_DT))
        y = self._standardise(observed.to(_DT).reshape(1, -1)).reshape(-1)
        n = int(y.numel())
        sigma2 = ((P - y[None, :]) ** 2).mean(-1).clamp_min(1e-12)
        loglik = -0.5 * n * (torch.log(2 * math.pi * sigma2) + 1.0)
        return loglik - math.log(max(n, 2)) + self.log_prior

    def disagreement(self, predictions: Tensor, log_weights: Tensor | None = None) -> float:
        w = torch.softmax(
            (self.log_prior if log_weights is None else log_weights).to(_DT), dim=0
        )
        P = self._standardise(predictions.to(_DT))
        mean = (w[:, None] * P).sum(0)
        return float(((w[:, None] * (P - mean) ** 2).sum(0)).sqrt().mean())

    def to_mechanistic_uncertainty(
        self, log_weights: Tensor | None = None
    ) -> MechanisticUncertainty:
        lw = self.log_prior if log_weights is None else log_weights
        p = torch.softmax(lw.to(_DT), dim=0)
        return MechanisticUncertainty(
            candidates=self.names,
            log_weights=lw.to(_DT),
            resolved=bool(p.max() >= 0.95),
            note=(
                "low-intensity tFUS mechanism is unresolved; the candidate set "
                "includes an auditory-confound null (thesis Sec. 7.2)"
            ),
        )
