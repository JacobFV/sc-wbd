"""In-situ exposure: planned focus and realized exposure as distinct variables.

**SIMULATION ONLY** -- see :mod:`scwbd.intervene.base`.

Thesis Sec. 7.2: *"The planned focus and realized exposure are distinct random
variables; tracking, registration, array steering, skull modeling, and coupling
each contribute bias and variance."*  This module refuses to let them share a
name:

* :class:`PlannedFocus` is what the steering command asks for.  It is a
  **command**, not a measurement.
* :class:`RealizedExposure` is what a simulated propagation through a simulated
  skull actually produces.
* :class:`FocalDivergence` is their difference, reported as a distribution over
  the contributing error sources -- never as a single "targeting accuracy".

Exposure metrics follow the ITRUSST standardised-reporting quantities (Aubry
et al. 2023): peak positive/negative pressure, :math:`I_{\\rm SPPA}`,
:math:`I_{\\rm SPTA}`, mechanical index, temperature rise, and CEM43 thermal
dose (Sapareto & Dewey 1984), which **accumulates across the whole session and
is never reset between bursts**.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import torch
from torch import Tensor

from ..base import SIMULATION_ONLY_NOTICE, Ledger, PhysicalDose, ThermalHistory
from .acoustics import BRAIN, AcousticMedium
from .transducer import PulseSequence

__all__ = [
    "PlannedFocus",
    "RealizedExposure",
    "FocalDivergence",
    "ExposureMetrics",
    "exposure_metrics",
    "bioheat_temperature",
    "accumulate_thermal_dose",
]

_DT = torch.float64


# ---------------------------------------------------------------------------
# planned vs realized
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedFocus:
    """Where the steering command intends to put the focus. A command."""

    target_head_frame_m: Tensor  # [3]
    frame: str
    steering_target_transducer_frame_m: Tensor  # [3]
    sound_speed_assumed_m_per_s: float
    pulse: PulseSequence
    provenance: Mapping[str, object] = field(default_factory=dict)
    notice: str = SIMULATION_ONLY_NOTICE

    def as_realized(self):  # pragma: no cover - refusal by type
        raise TypeError(
            "a planned focus is not a realized exposure; propagate the field "
            "through the skull model and construct RealizedExposure from it"
        )


@dataclass(frozen=True)
class RealizedExposure:
    """What a simulated propagation actually delivered."""

    grid_points_m: Tensor  # [N,3] or [nz,ny,nx,3]
    pressure_pa: Tensor  # magnitude, same leading shape
    peak_location_m: Tensor  # [3]
    medium: AcousticMedium
    pulse: PulseSequence
    solver: str
    ledger: Ledger = field(default_factory=Ledger)
    notice: str = SIMULATION_ONLY_NOTICE

    @classmethod
    def from_field(
        cls,
        grid_points_m: Tensor,
        pressure: Tensor,
        *,
        medium: AcousticMedium,
        pulse: PulseSequence,
        solver: str,
        ledger: Ledger | None = None,
    ) -> "RealizedExposure":
        mag = pressure.abs().reshape(-1)
        pts = grid_points_m.reshape(-1, 3)
        return cls(
            grid_points_m=pts,
            pressure_pa=mag,
            peak_location_m=pts[int(mag.argmax())],
            medium=medium,
            pulse=pulse,
            solver=solver,
            ledger=ledger or Ledger(),
        )

    def as_dose(self) -> PhysicalDose:
        return PhysicalDose(
            modality="tfus",
            quantity="peak_pressure",
            units="Pa",
            value=self.pressure_pa,
            support=f"acoustic_grid[{tuple(self.pressure_pa.shape)}]/{self.solver}",
            ledger=self.ledger,
        )


@dataclass(frozen=True)
class FocalDivergence:
    """Planned minus realized, with the error sources kept apart.

    ``per_source_mm`` attributes displacement to named contributions (tracking,
    registration, steering, skull model, coupling).  Summing them into one
    number would be exactly the collapse the thesis forbids.
    """

    displacement_mm: float
    lateral_mm: float
    axial_mm: float
    pressure_ratio: float
    per_source_mm: Mapping[str, float]
    n_samples: int
    seed: int
    displacement_sd_mm: float = 0.0
    ledger: Ledger = field(default_factory=Ledger)
    notice: str = SIMULATION_ONLY_NOTICE

    def summary(self) -> str:
        rows = [
            "planned vs realized focus (SIMULATION ONLY)",
            f"  displacement {self.displacement_mm:.2f} mm "
            f"(lateral {self.lateral_mm:.2f}, axial {self.axial_mm:.2f})"
            f" +/- {self.displacement_sd_mm:.2f} mm",
            f"  realized/planned peak pressure ratio {self.pressure_ratio:.3f}",
        ]
        rows += [f"    {k}: {v:.2f} mm" for k, v in sorted(self.per_source_mm.items())]
        return "\n".join(rows)


def focal_divergence(
    planned: PlannedFocus,
    realized: RealizedExposure,
    *,
    planned_peak_pa: float,
    per_source_mm: Mapping[str, float] | None = None,
    n_samples: int = 1,
    seed: int = 0,
    displacement_sd_mm: float = 0.0,
) -> FocalDivergence:
    """Compare a planned focus against a realized exposure."""
    d = realized.peak_location_m.to(_DT) - planned.target_head_frame_m.to(_DT)
    axis = planned.target_head_frame_m.to(_DT)
    axis = axis / axis.norm().clamp_min(1e-12)
    axial = float(d @ axis)
    lateral = float((d - axial * axis).norm())
    return FocalDivergence(
        displacement_mm=float(d.norm()) * 1e3,
        lateral_mm=lateral * 1e3,
        axial_mm=axial * 1e3,
        pressure_ratio=float(realized.pressure_pa.max()) / max(planned_peak_pa, 1e-12),
        per_source_mm=dict(per_source_mm or {}),
        n_samples=int(n_samples),
        seed=int(seed),
        displacement_sd_mm=displacement_sd_mm,
        ledger=Ledger(
            variance={"targeting": (displacement_sd_mm * 1e-3) ** 2},
            bias_interval=(0.0, float(d.norm())),
            bias_status="externally_bounded",
            validity_domain={
                "note": "planned focus and realized exposure are distinct "
                "random variables (thesis Sec. 7.2)",
                "solver": realized.solver,
            },
        ),
    )


# ---------------------------------------------------------------------------
# exposure metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExposureMetrics:
    """ITRUSST-style standardised in-situ exposure quantities.

    All values are properties of a **simulated** field.  They exist to be
    compared against :mod:`scwbd.intervene.safety` limits so an optimizer can
    be blocked, not to characterise a real exposure.
    """

    peak_positive_pressure_pa: float
    peak_negative_pressure_pa: float
    isppa_w_per_cm2: float
    ispta_mw_per_cm2: float
    mechanical_index: float
    frequency_hz: float
    duty_cycle: float
    focal_volume_mm3: float
    ledger: Ledger = field(default_factory=Ledger)
    notice: str = SIMULATION_ONLY_NOTICE

    def as_safety_axes(self) -> dict[str, float]:
        """Map onto the declared ``A_safe`` axis names for a feasibility check."""
        return {
            "tfus.mechanical_index": self.mechanical_index,
            "tfus.isppa_w_per_cm2": self.isppa_w_per_cm2,
            "tfus.ispta_mw_per_cm2": self.ispta_mw_per_cm2,
            "tfus.duty_cycle": self.duty_cycle,
        }


def exposure_metrics(
    pressure_pa: Tensor,
    *,
    medium: AcousticMedium = BRAIN,
    pulse: PulseSequence,
    voxel_volume_m3: float = 1e-9,
    derating: float = 1.0,
) -> ExposureMetrics:
    """Compute in-situ exposure metrics from a simulated pressure magnitude field.

    :math:`I_{\\rm SPPA} = p^2/(2\\rho c)`;
    :math:`I_{\\rm SPTA} = I_{\\rm SPPA}\\times` duty cycle;
    :math:`\\mathrm{MI} = p_-[\\mathrm{MPa}]/\\sqrt{f[\\mathrm{MHz}]}`.

    ``derating`` is applied explicitly and recorded; an un-derated number and a
    derated number are not the same quantity.
    """
    p = torch.as_tensor(pressure_pa, dtype=_DT).abs() * derating
    p_peak = float(p.max())
    isppa = p_peak**2 / (2 * medium.impedance)  # W/m^2
    isppa_cm2 = isppa / 1e4
    ispta_cm2 = isppa_cm2 * pulse.duty_cycle
    mi = (p_peak / 1e6) / math.sqrt(pulse.frequency_hz / 1e6)
    fwhm = p >= (p_peak / math.sqrt(2.0))
    vol_mm3 = float(fwhm.sum()) * voxel_volume_m3 * 1e9
    return ExposureMetrics(
        peak_positive_pressure_pa=p_peak,
        peak_negative_pressure_pa=p_peak,  # magnitude field: symmetric assumption
        isppa_w_per_cm2=isppa_cm2,
        ispta_mw_per_cm2=ispta_cm2 * 1e3,
        mechanical_index=mi,
        frequency_hz=pulse.frequency_hz,
        duty_cycle=pulse.duty_cycle,
        focal_volume_mm3=vol_mm3,
        ledger=Ledger(
            variance={},
            bias_status="externally_bounded",
            validity_domain={
                "derating": derating,
                "symmetric_pressure_assumption": True,
                "note": "linear propagation; nonlinear p+/p- asymmetry not modelled",
            },
        ),
    )


# ---------------------------------------------------------------------------
# thermal
# ---------------------------------------------------------------------------


def bioheat_temperature(
    intensity_w_per_m2: float | Tensor,
    *,
    medium: AcousticMedium = BRAIN,
    frequency_hz: float = 500e3,
    duration_s: float,
    dt_s: float = 0.01,
    baseline_c: float = 37.0,
    duty_cycle: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Lumped Pennes bioheat integration at the focus.

    :math:`\\rho C \\,\\mathrm dT/\\mathrm dt = 2\\alpha I -
    \\rho C (T-T_0)/\\tau_{\\rm perf}` with the duty cycle applied to the
    time-averaged absorbed power.  Returns ``(times_s, temperature_c)``.

    A lumped model: it deliberately omits conduction away from a small focus
    and therefore **over**-estimates heating, which is the correct direction
    for a quantity used only to block an optimizer.
    """
    a = medium.alpha_np_per_m(frequency_hz)
    rhoC = medium.density_kg_per_m3 * medium.heat_capacity_j_per_kg_k
    tau = medium.perfusion_time_constant_s
    I = float(torch.as_tensor(intensity_w_per_m2, dtype=_DT).max()) * duty_cycle
    n = max(1, int(round(duration_s / dt_s)))
    t = torch.linspace(0.0, duration_s, n + 1, dtype=_DT)
    T = torch.empty(n + 1, dtype=_DT)
    T[0] = baseline_c
    for i in range(n):
        dT = (2 * a * I / rhoC - (float(T[i]) - baseline_c) / tau) * dt_s
        T[i + 1] = T[i] + dT
    return t, T


def accumulate_thermal_dose(
    history: ThermalHistory, times_s: Tensor, temperature_c: Tensor
) -> ThermalHistory:
    """Accumulate a temperature trace into a :class:`ThermalHistory` (CEM43).

    Uses the Sapareto--Dewey rule with the trapezoid midpoint temperature of
    each interval.  Accumulation is *monotone* and never reset between bursts:
    a session's thermal dose is the sum over its whole history.
    """
    t = torch.as_tensor(times_s, dtype=_DT)
    T = torch.as_tensor(temperature_c, dtype=_DT)
    h = history
    for i in range(int(t.numel()) - 1):
        dt = float(t[i + 1] - t[i])
        h = h.accumulate(0.5 * (float(T[i]) + float(T[i + 1])), dt)
    return h
