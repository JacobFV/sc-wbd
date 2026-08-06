"""tFUS transducer geometry, element layout, phasing, and pulse sequence.

**SIMULATION ONLY** -- see :mod:`scwbd.intervene.base`.  This module describes
a transducer as geometry plus a drive schedule.  It emits no device settings
and drives no hardware.

Kept separate, as thesis Sec. 7.2 requires: *tracked transducer pose*,
*skull acoustics*, *steering commands*, *in situ pressure and thermal
estimates*, *uncertain cellular coupling*, and *downstream network dynamics*
are six different objects.  This file owns only the first and third.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from ..base import SIMULATION_ONLY_NOTICE, BurstSequence, DeviceGeometry

__all__ = [
    "TransducerArray",
    "SingleElementBowl",
    "AnnularArray",
    "PlanarGridArray",
    "PulseSequence",
]

_DT = torch.float64


@dataclass(frozen=True)
class PulseSequence:
    """Tone-burst schedule. Duty cycle is a *declared* quantity, never inferred."""

    frequency_hz: float
    burst_duration_s: float
    pulse_repetition_frequency_hz: float
    n_bursts: int
    units: str = "s"
    notice: str = SIMULATION_ONLY_NOTICE

    @property
    def duty_cycle(self) -> float:
        return float(self.burst_duration_s * self.pulse_repetition_frequency_hz)

    @property
    def total_duration_s(self) -> float:
        return self.n_bursts / self.pulse_repetition_frequency_hz

    @property
    def total_on_time_s(self) -> float:
        return self.n_bursts * self.burst_duration_s

    @property
    def cycles_per_burst(self) -> float:
        return self.burst_duration_s * self.frequency_hz

    def to_burst_sequence(self) -> BurstSequence:
        return BurstSequence(
            pulse_period=1.0 / self.frequency_hz,
            n_pulses_per_burst=max(1, int(self.cycles_per_burst)),
            burst_period=1.0 / self.pulse_repetition_frequency_hz,
            n_bursts=self.n_bursts,
        )

    def __post_init__(self) -> None:
        if self.duty_cycle > 1.0:
            raise ValueError(
                f"duty cycle {self.duty_cycle:.3f} > 1: burst duration exceeds "
                "the pulse repetition interval"
            )


@dataclass(frozen=True)
class TransducerArray(DeviceGeometry):
    """Base: a planar source aperture in the **transducer frame**.

    The transducer frame has its origin at the aperture centre with ``+z``
    along the acoustic axis pointing into the head.  Every propagator in
    :mod:`scwbd.intervene.tfus.acoustics` consumes the source-plane normal
    velocity this class produces, so the geometry and the propagation stay
    separable.
    """

    frequency_hz: float = 500e3
    aperture_radius_m: float = 0.015
    focal_length_m: float = 0.060
    surface_velocity_m_per_s: float = 0.05
    grid_n: int = 256
    grid_extent_m: float = 0.120

    # -- grid ---------------------------------------------------------------

    def source_grid(self) -> tuple[Tensor, Tensor, float]:
        """Return ``(X, Y, dx)`` for the source plane, metres."""
        n = self.grid_n
        dx = self.grid_extent_m / n
        ax = (torch.arange(n, dtype=_DT) - n // 2) * dx
        X, Y = torch.meshgrid(ax, ax, indexing="ij")
        return X, Y, dx

    def aperture_mask(self) -> Tensor:
        X, Y, _ = self.source_grid()
        return (X**2 + Y**2) <= self.aperture_radius_m**2

    def element_positions(self) -> Tensor:
        X, Y, _ = self.source_grid()
        m = self.aperture_mask()
        return torch.stack([X[m], Y[m], torch.zeros_like(X[m])], dim=-1)

    # -- phasing ------------------------------------------------------------

    def geometric_focus_phase(self, sound_speed: float = 1500.0) -> Tensor:
        """Phase that focuses the aperture at ``focal_length_m`` on axis."""
        X, Y, _ = self.source_grid()
        k = 2 * math.pi * self.frequency_hz / sound_speed
        F = self.focal_length_m
        # +k with the e^{-ikR} propagation convention of
        # acoustics.angular_spectrum_propagate: source phase must
        # CANCEL the extra path length, not add to it.
        return k * (torch.sqrt(X**2 + Y**2 + F**2) - F)

    def steering_phase(
        self, target_m: Sequence[float] | Tensor, sound_speed: float = 1500.0
    ) -> Tensor:
        """Phase steering the focus to ``target_m = (x, y, z)`` in the transducer frame.

        A steering *command*: it says where the array is trying to put energy.
        Where the energy actually goes is a different random variable
        (:mod:`scwbd.intervene.tfus.exposure`).
        """
        X, Y, _ = self.source_grid()
        t = torch.as_tensor(target_m, dtype=_DT).reshape(3)
        k = 2 * math.pi * self.frequency_hz / sound_speed
        d = torch.sqrt((X - t[0]) ** 2 + (Y - t[1]) ** 2 + t[2] ** 2)
        return k * (d - d[self.grid_n // 2, self.grid_n // 2])

    def apodization(self) -> Tensor:
        return self.aperture_mask().to(_DT)

    def source_velocity(
        self,
        *,
        phase: Tensor | None = None,
        sound_speed: float = 1500.0,
        focused: bool = True,
    ) -> Tensor:
        """Complex normal surface velocity on the source plane, m/s."""
        ph = phase
        if ph is None:
            ph = self.geometric_focus_phase(sound_speed) if focused else torch.zeros_like(
                self.apodization()
            )
        return (
            self.surface_velocity_m_per_s
            * self.apodization()
            * torch.exp(1j * ph.to(_DT))
        ).to(torch.complex128)

    @property
    def wavelength_m(self) -> float:
        return 1500.0 / self.frequency_hz

    @property
    def f_number(self) -> float:
        return self.focal_length_m / (2 * self.aperture_radius_m)

    @property
    def notice(self) -> str:
        return SIMULATION_ONLY_NOTICE


@dataclass(frozen=True)
class SingleElementBowl(TransducerArray):
    """Fixed-geometry focused bowl: no steering, focus set by the shell radius."""

    device_id: str = "single_element_bowl"
    frame: str = "transducer"

    def steering_phase(self, target_m, sound_speed: float = 1500.0) -> Tensor:
        raise ValueError(
            "a single-element bowl cannot be electronically steered; its focus "
            "is fixed by the shell geometry and moves only with the pose"
        )


@dataclass(frozen=True)
class AnnularArray(TransducerArray):
    """Concentric rings: axial steering only, no lateral steering."""

    device_id: str = "annular_array"
    frame: str = "transducer"
    n_rings: int = 8

    def ring_index(self) -> Tensor:
        X, Y, _ = self.source_grid()
        rho = torch.sqrt(X**2 + Y**2)
        # equal-area rings
        idx = torch.floor(self.n_rings * (rho / self.aperture_radius_m) ** 2)
        return idx.clamp(0, self.n_rings - 1).to(torch.long)

    def steering_phase(self, target_m, sound_speed: float = 1500.0) -> Tensor:
        t = torch.as_tensor(target_m, dtype=_DT).reshape(3)
        if float(t[:2].norm()) > 1e-9:
            raise ValueError(
                "an annular array steers only along the acoustic axis; lateral "
                "steering to "
                f"({float(t[0]):.4f}, {float(t[1]):.4f}) m is not physically available"
            )
        full = super().steering_phase(t, sound_speed)
        # quantise to rings: each ring gets one delay
        ring = self.ring_index()
        out = torch.zeros_like(full)
        for r in range(self.n_rings):
            m = ring == r
            if bool(m.any()):
                out[m] = full[m].mean()
        return out


@dataclass(frozen=True)
class PlanarGridArray(TransducerArray):
    """2-D element grid: full 3-D electronic steering within a grating limit."""

    device_id: str = "planar_grid_array"
    frame: str = "transducer"
    n_elements_side: int = 16

    def element_pitch_m(self) -> float:
        return 2 * self.aperture_radius_m / self.n_elements_side

    def grating_lobe_free_angle_deg(self, sound_speed: float = 1500.0) -> float:
        """Steering angle beyond which grating lobes appear: ``sin th = lam/d - 1``."""
        lam = sound_speed / self.frequency_hz
        s = lam / self.element_pitch_m() - 1.0
        if s >= 1.0:
            return 90.0
        if s <= 0.0:
            return 0.0
        return math.degrees(math.asin(s))

    def steering_phase(self, target_m, sound_speed: float = 1500.0) -> Tensor:
        t = torch.as_tensor(target_m, dtype=_DT).reshape(3)
        ang = math.degrees(math.atan2(float(t[:2].norm()), float(t[2])))
        lim = self.grating_lobe_free_angle_deg(sound_speed)
        if ang > lim:
            raise ValueError(
                f"steering angle {ang:.1f} deg exceeds the grating-lobe-free "
                f"limit {lim:.1f} deg for pitch {self.element_pitch_m()*1e3:.2f} mm; "
                "the simulated field would contain unmodelled secondary maxima"
            )
        full = super().steering_phase(t, sound_speed)
        # quantise to square elements
        X, Y, _ = self.source_grid()
        p = self.element_pitch_m()
        ix = torch.floor((X + self.aperture_radius_m) / p).clamp(0, self.n_elements_side - 1)
        iy = torch.floor((Y + self.aperture_radius_m) / p).clamp(0, self.n_elements_side - 1)
        eid = (ix * self.n_elements_side + iy).to(torch.long)
        out = torch.zeros_like(full)
        for e in eid.unique():
            m = eid == e
            out[m] = full[m].mean()
        return out
