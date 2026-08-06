"""TMS coil geometry, winding discretisation, and current waveform.

**SIMULATION ONLY** -- see :mod:`scwbd.intervene.base`.  This module models a
coil as a geometric object and a circuit waveform.  It is not a device driver
and emits no stimulator settings.

Two independent discretisations of the same winding are provided, because
agreement between them is a *test* rather than an assumption:

``segments()``
    the winding as a polyline of current elements
    :math:`I\\,\\mathrm d\\boldsymbol\\ell`, giving the free-space vector
    potential :math:`A(r) = \\frac{\\mu_0 I}{4\\pi}\\oint \\frac{\\mathrm
    d\\boldsymbol\\ell}{|r-r'|}`.

``dipole_elements()``
    the winding as an equivalent magnetisation sheet
    :math:`\\mathrm dm = I\\,N(\\rho)\\,\\mathrm dA\\,\\hat z` (the ``.ccd``
    representation used by SimNIBS).  Required by the analytic spherical-head
    solution, which is written for magnetic dipole sources.

Coil frame convention (a *device* frame -- never a scalp label): origin at the
centre of the coil's head-facing face, :math:`+\\hat z` pointing **away** from
the head along the coil axis, :math:`+\\hat y` along the handle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from ..base import SIMULATION_ONLY_NOTICE, DeviceGeometry, WaveformSpec

__all__ = [
    "MU0",
    "CoilGeometry",
    "CircularCoil",
    "FigureEightCoil",
    "TMSPulse",
    "monophasic",
    "biphasic",
    "halfsine",
    "pulse_waveform_spec",
]

MU0 = 4.0e-7 * math.pi  # H/m, exact by the pre-2019 definition; retained as the
#                          conventional value used by every TMS field code.

_DT = torch.float64


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoilGeometry(DeviceGeometry):
    """Base for coil windings expressed in the coil frame (metres)."""

    inner_radius: float = 0.026
    outer_radius: float = 0.044
    n_turns: int = 9
    winding_height: float = 0.005  # m, above the head-facing face
    n_azimuth: int = 128
    n_radial: int = 12

    # -- discretisation 1: current elements ---------------------------------

    def segments(self) -> tuple[Tensor, Tensor]:
        """Winding as closed current loops. ``(midpoints [S,3], dl [S,3])``, metres.

        ``dl`` is the oriented segment vector; the free-space vector potential
        for unit current is ``MU0/(4 pi) * sum(dl / |r - mid|)``.
        """
        raise NotImplementedError

    # -- discretisation 2: equivalent magnetisation sheet --------------------

    def dipole_elements(self) -> tuple[Tensor, Tensor]:
        """Returns ``(positions [D,3], moment_per_amp [D,3])`` in metres, m^2.

        The magnetic moment of element *i* at coil current :math:`I` is
        ``I * moment_per_amp[i]``.
        """
        raise NotImplementedError

    def element_positions(self) -> Tensor:  # DeviceGeometry contract
        return self.dipole_elements()[0]

    # -- shared helpers ------------------------------------------------------

    def loop_radii(self) -> Tensor:
        """Radii of the ``n_turns`` closed windings, inner to outer."""
        if self.n_turns <= 1:
            return torch.tensor([0.5 * (self.inner_radius + self.outer_radius)], dtype=_DT)
        return torch.linspace(
            self.inner_radius, self.outer_radius, self.n_turns, dtype=_DT
        )

    def _turns_enclosing(self, rho: Tensor) -> Tensor:
        """Number of windings enclosing radius ``rho``.

        The **exact** step function over :meth:`loop_radii`, so the
        magnetisation-sheet and current-loop discretisations describe the same
        coil rather than two similar ones: their total moments agree exactly,
        :math:`\\int s(\\rho)\\,2\\pi\\rho\\,\\mathrm d\\rho = \\sum_i \\pi r_i^2`.
        """
        return (rho[..., None] < self.loop_radii()).sum(-1).to(_DT)

    def _disc_sheet(self, centre: Tensor, sign: float) -> tuple[Tensor, Tensor]:
        """Equal-area polar tiling of one wing's magnetisation sheet."""
        nr, na = self.n_radial, self.n_azimuth
        edges = torch.linspace(0.0, self.outer_radius, nr + 1, dtype=_DT)
        rho = 0.5 * (edges[:-1] + edges[1:])
        ring_area = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2) / na
        phi = (torch.arange(na, dtype=_DT) + 0.5) * (2 * math.pi / na)
        rr, pp = torch.meshgrid(rho, phi, indexing="ij")
        aa = ring_area[:, None].expand_as(rr)
        pos = torch.stack(
            [
                centre[0] + rr * torch.cos(pp),
                centre[1] + rr * torch.sin(pp),
                torch.full_like(rr, self.winding_height),
            ],
            dim=-1,
        ).reshape(-1, 3)
        dens = self._turns_enclosing(rr).reshape(-1) * aa.reshape(-1)
        mom = torch.zeros(pos.shape[0], 3, dtype=_DT)
        mom[:, 2] = sign * dens
        keep = dens > 0
        return pos[keep], mom[keep]

    def _loops(self, centre: Tensor, sign: float) -> tuple[Tensor, Tensor]:
        """``n_turns`` **closed** concentric current loops.

        Closed on purpose: an open spiral polyline has
        :math:`\\oint\\mathrm d\\boldsymbol\\ell \\neq 0` and therefore radiates a
        spurious dipole term that the magnetisation-sheet representation does
        not have.  A real winding's leads are a separate modelling decision,
        not an artefact of the discretisation.
        """
        mids, dls = [], []
        edge = torch.arange(self.n_azimuth + 1, dtype=_DT) * (
            2 * math.pi / self.n_azimuth
        )
        for r in self.loop_radii():
            pts = torch.stack(
                [
                    centre[0] + r * torch.cos(sign * edge),
                    centre[1] + r * torch.sin(sign * edge),
                    torch.full_like(edge, self.winding_height),
                ],
                dim=-1,
            )
            dls.append(pts[1:] - pts[:-1])
            mids.append(0.5 * (pts[1:] + pts[:-1]))
        return torch.cat(mids), torch.cat(dls)

    @property
    def notice(self) -> str:
        return SIMULATION_ONLY_NOTICE


@dataclass(frozen=True)
class CircularCoil(CoilGeometry):
    """Single round coil centred on the coil-frame origin."""

    device_id: str = "circular_70mm"
    frame: str = "coil"

    def segments(self) -> tuple[Tensor, Tensor]:
        return self._loops(torch.zeros(3, dtype=_DT), +1.0)

    def dipole_elements(self) -> tuple[Tensor, Tensor]:
        return self._disc_sheet(torch.zeros(3, dtype=_DT), +1.0)

    def total_moment_per_amp(self) -> float:
        return float(self.dipole_elements()[1][:, 2].sum())


@dataclass(frozen=True)
class FigureEightCoil(CoilGeometry):
    """Two wings with opposed winding sense; the standard focal TMS coil.

    ``wing_separation`` is the centre-to-centre distance between wings.  The
    wings lie along :math:`\\pm\\hat x`; the handle is along :math:`+\\hat y`,
    so the *induced-field direction* under the coil centre is along
    :math:`\\pm\\hat y` -- the quantity a neuronavigation record must state and
    a scalp label cannot.
    """

    device_id: str = "figure8_70mm"
    frame: str = "coil"
    wing_separation: float = 0.088

    def segments(self) -> tuple[Tensor, Tensor]:
        d = self.wing_separation / 2.0
        left = torch.tensor([-d, 0.0, 0.0], dtype=_DT)
        right = torch.tensor([+d, 0.0, 0.0], dtype=_DT)
        m1, l1 = self._loops(left, +1.0)
        m2, l2 = self._loops(right, -1.0)
        return torch.cat([m1, m2]), torch.cat([l1, l2])

    def dipole_elements(self) -> tuple[Tensor, Tensor]:
        d = self.wing_separation / 2.0
        left = torch.tensor([-d, 0.0, 0.0], dtype=_DT)
        right = torch.tensor([+d, 0.0, 0.0], dtype=_DT)
        p1, q1 = self._disc_sheet(left, +1.0)
        p2, q2 = self._disc_sheet(right, -1.0)
        return torch.cat([p1, p2]), torch.cat([q1, q2])

    def net_moment_per_amp(self) -> float:
        """Zero by construction: the wings cancel. A figure-8 has no net moment."""
        return float(self.dipole_elements()[1].sum())


# ---------------------------------------------------------------------------
# waveform
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TMSPulse:
    """Coil current :math:`I(t)` and its derivative :math:`\\mathrm dI/\\mathrm dt`.

    The E-field is driven by :math:`\\mathrm dI/\\mathrm dt`, **not** by
    :math:`I`; the two are kept as separate methods so no caller can confuse
    "stimulator output" with "induced field".

    Models (underdamped RLC / double-exponential circuit forms):

    ``biphasic``
        :math:`I(t)=I_0 e^{-\\alpha t}\\sin(\\omega t)` -- the current
        completes a full oscillation, so :math:`\\mathrm dI/\\mathrm dt` is
        near-symmetric about zero.
    ``monophasic``
        :math:`I(t)=I_0(e^{-t/\\tau_2}-e^{-t/\\tau_1})`, :math:`\\tau_1\\ll
        \\tau_2` -- a fast unidirectional rise and a long low-amplitude decay,
        so :math:`\\max \\mathrm dI/\\mathrm dt \\gg |\\min \\mathrm
        dI/\\mathrm dt|`.
    ``halfsine``
        :math:`I(t)=I_0\\sin(\\pi t/T)`.
    """

    kind: Literal["monophasic", "biphasic", "halfsine"]
    period: float  # s, nominal pulse duration
    peak_didt: float = 1.0e8  # A/s; typical single-pulse TMS is ~1e8 A/s
    damping: float = 0.35  # dimensionless, alpha * period for biphasic
    rise_fraction: float = 0.10  # tau1 / period for monophasic
    notice: str = SIMULATION_ONLY_NOTICE

    # -- shapes (unit amplitude) --------------------------------------------

    def _shape(self, t: Tensor) -> Tensor:
        t = torch.as_tensor(t, dtype=_DT)
        T = self.period
        if self.kind == "biphasic":
            a = self.damping / T
            w = 2 * math.pi / T
            return torch.exp(-a * t) * torch.sin(w * t)
        if self.kind == "monophasic":
            t1 = self.rise_fraction * T
            t2 = T
            return torch.exp(-t / t2) - torch.exp(-t / t1)
        if self.kind == "halfsine":
            return torch.sin(math.pi * t / T)
        raise ValueError(f"unknown pulse kind {self.kind!r}")

    def _dshape(self, t: Tensor) -> Tensor:
        t = torch.as_tensor(t, dtype=_DT)
        T = self.period
        if self.kind == "biphasic":
            a = self.damping / T
            w = 2 * math.pi / T
            e = torch.exp(-a * t)
            return e * (w * torch.cos(w * t) - a * torch.sin(w * t))
        if self.kind == "monophasic":
            t1 = self.rise_fraction * T
            t2 = T
            return -torch.exp(-t / t2) / t2 + torch.exp(-t / t1) / t1
        if self.kind == "halfsine":
            return (math.pi / T) * torch.cos(math.pi * t / T)
        raise ValueError(f"unknown pulse kind {self.kind!r}")

    @property
    def duration(self) -> float:
        """Modelled pulse support; monophasic tails are truncated at 5 tau_2."""
        return self.period if self.kind != "monophasic" else 5.0 * self.period

    def _norm(self) -> float:
        t = torch.linspace(0.0, self.duration, 200001, dtype=_DT)
        return float(self._dshape(t).abs().max())

    # -- physical quantities -------------------------------------------------

    def didt(self, t: Tensor | float) -> Tensor:
        """:math:`\\mathrm dI/\\mathrm dt` in A/s, peak normalised to ``peak_didt``."""
        tt = torch.as_tensor(t, dtype=_DT)
        val = self.peak_didt * self._dshape(tt) / self._norm()
        return torch.where(
            (tt >= 0.0) & (tt <= self.duration), val, torch.zeros_like(val)
        )

    def current(self, t: Tensor | float) -> Tensor:
        """Coil current in A, consistent with :meth:`didt` by construction."""
        tt = torch.as_tensor(t, dtype=_DT)
        val = self.peak_didt * self._shape(tt) / self._norm()
        return torch.where(
            (tt >= 0.0) & (tt <= self.duration), val, torch.zeros_like(val)
        )

    def peak_current(self) -> float:
        t = torch.linspace(0.0, self.duration, 20001, dtype=_DT)
        return float(self.current(t).abs().max())

    def didt_asymmetry(self) -> float:
        """``max(dI/dt) / |min(dI/dt)|``. Large for monophasic, ~1 for biphasic."""
        t = torch.linspace(0.0, self.duration, 200001, dtype=_DT)
        d = self.didt(t)
        neg = float(d.min().abs())
        return float(d.max()) / neg if neg > 0 else float("inf")


def monophasic(period: float = 80e-6, peak_didt: float = 1.0e8) -> TMSPulse:
    return TMSPulse("monophasic", period=period, peak_didt=peak_didt)


def biphasic(period: float = 280e-6, peak_didt: float = 1.0e8) -> TMSPulse:
    return TMSPulse("biphasic", period=period, peak_didt=peak_didt)


def halfsine(period: float = 160e-6, peak_didt: float = 1.0e8) -> TMSPulse:
    return TMSPulse("halfsine", period=period, peak_didt=peak_didt)


def pulse_waveform_spec(pulse: TMSPulse, burst=None) -> WaveformSpec:
    """Adapt a :class:`TMSPulse` to the generic :class:`~scwbd.intervene.base.WaveformSpec`.

    The drive :math:`u_k(t)` carried through the Sec. 2.4 SDE is
    :math:`\\mathrm dI/\\mathrm dt` in A/s, because that -- not the current --
    is what sources the induced electric field.
    """
    return WaveformSpec(
        name=f"tms_{pulse.kind}",
        units="A/s",
        period=pulse.duration,
        sample_fn=lambda t: pulse.didt(t),
        burst=burst,
        amplitude=1.0,
    )
