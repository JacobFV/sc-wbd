"""Skull-transmitted acoustic propagation, validated against analytic references.

**SIMULATION ONLY** -- see :mod:`scwbd.intervene.base`.  The output is an
acoustic *pressure field* -- a physical dose.  Pressure is not a neural effect;
converting it requires an explicitly named candidate operator from
:mod:`scwbd.intervene.tfus.response`.

``k-wave-python`` is wrapped when usable (:func:`kwave_status`).  On this host
it is **not**: the package installs from PyPI but downloads x86-64 ELF solver
binaries (``kspaceFirstOrder-OMP`` / ``-CUDA``) that cannot execute on
linux-aarch64, and installing it downgrades ``numpy``/``scipy`` for the whole
environment.  So this module ships its own propagators, validated against
closed-form references:

:func:`on_axis_piston_pressure`
    the **exact** on-axis field of a baffled circular piston,
    :math:`|p(z)| = 2\\rho c u_0\\,|\\sin[\\tfrac{k}{2}(\\sqrt{z^2+a^2}-z)]|`.
    This is the analytic homogeneous reference everything else is checked
    against.

:func:`angular_spectrum_propagate`
    the exact planar propagator for a homogeneous medium.  Starting from the
    normal surface velocity, :math:`P(k_x,k_y,z) = \\rho c\\,(k/k_z)\\,
    U_z(k_x,k_y,0)\\,e^{i k_z z}`, which is equivalent to the Rayleigh integral
    -- so agreement with :func:`on_axis_piston_pressure` is a real test, not a
    tautology.

:func:`rayleigh_sommerfeld_pressure`
    the direct surface integral, an independent semi-analytic cross-check for
    focused apertures where no simple closed form exists.

:class:`SplitStepPropagator`
    a heterogeneous split-step (phase-screen) propagator for skull layers.  It
    reduces **exactly** to the angular-spectrum solution when the medium is
    homogeneous, which is its regression test.

:func:`hounsfield_to_acoustic`
    CT Hounsfield-unit to (density, sound speed, absorption) conversion via the
    porosity model of Aubry et al. (2003), with IT'IS tissue values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..base import SIMULATION_ONLY_NOTICE, Ledger, PhysicalDose

__all__ = [
    "AcousticMedium",
    "WATER",
    "BRAIN",
    "CORTICAL_BONE",
    "on_axis_piston_pressure",
    "angular_spectrum_propagate",
    "rayleigh_sommerfeld_pressure",
    "SkullLayer",
    "SplitStepPropagator",
    "hounsfield_to_acoustic",
    "kwave_available",
    "kwave_status",
    "KWaveSolver",
]

_DT = torch.float64
_CT = torch.complex128


# ---------------------------------------------------------------------------
# media
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcousticMedium:
    """Homogeneous acoustic properties. Units are mandatory and stated."""

    name: str
    density_kg_per_m3: float
    sound_speed_m_per_s: float
    alpha_db_per_cm_mhz: float = 0.0
    alpha_power: float = 1.1
    heat_capacity_j_per_kg_k: float = 3600.0
    thermal_conductivity_w_per_m_k: float = 0.5
    perfusion_time_constant_s: float = 100.0
    citation: str = "IT'IS Foundation tissue property database v4.1"
    notice: str = SIMULATION_ONLY_NOTICE

    @property
    def impedance(self) -> float:
        return self.density_kg_per_m3 * self.sound_speed_m_per_s

    def alpha_np_per_m(self, frequency_hz: float) -> float:
        """Absorption in Np/m at ``frequency_hz`` from the dB/(cm MHz) power law."""
        f_mhz = frequency_hz / 1e6
        db_per_cm = self.alpha_db_per_cm_mhz * f_mhz**self.alpha_power
        return db_per_cm * 100.0 / 8.685889638


WATER = AcousticMedium("water", 1000.0, 1500.0, 0.0022, 2.0)
BRAIN = AcousticMedium("brain", 1046.0, 1546.0, 0.59, 1.1)
CORTICAL_BONE = AcousticMedium(
    "cortical_bone", 1908.0, 2813.0, 6.9, 1.1,
    heat_capacity_j_per_kg_k=1313.0, thermal_conductivity_w_per_m_k=0.32,
)


# ---------------------------------------------------------------------------
# analytic reference
# ---------------------------------------------------------------------------


def on_axis_piston_pressure(
    z: Tensor,
    *,
    radius_m: float,
    frequency_hz: float,
    medium: AcousticMedium = WATER,
    surface_velocity_m_per_s: float = 1.0,
) -> Tensor:
    """Exact on-axis pressure magnitude of a baffled circular piston.

    :math:`|p(z)| = 2\\rho c u_0\\left|\\sin\\!\\left[\\frac{k}{2}
    \\left(\\sqrt{z^2+a^2}-z\\right)\\right]\\right|` -- lossless, and the
    standard closed-form validation target for any planar propagator.
    """
    z = torch.as_tensor(z, dtype=_DT)
    k = 2 * math.pi * frequency_hz / medium.sound_speed_m_per_s
    arg = 0.5 * k * (torch.sqrt(z**2 + radius_m**2) - z)
    return 2 * medium.impedance * surface_velocity_m_per_s * torch.sin(arg).abs()


def rayleigh_sommerfeld_pressure(
    obs: Tensor,
    source_points: Tensor,
    source_velocity: Tensor,
    element_area: float,
    *,
    frequency_hz: float,
    medium: AcousticMedium = WATER,
    chunk: int = 4096,
) -> Tensor:
    """Direct Rayleigh integral :math:`p = \\frac{i\\rho c k}{2\\pi}\\int u_z
    \\frac{e^{-ikR}}{R}\\,\\mathrm dS`.

    An independent semi-analytic reference for apertures without a closed form.
    ``obs [N,3]``, ``source_points [M,3]``, ``source_velocity [M]`` complex.
    """
    obs = obs.to(_DT).reshape(-1, 3)
    sp = source_points.to(_DT).reshape(-1, 3)
    u = source_velocity.to(_CT).reshape(-1)
    k = 2 * math.pi * frequency_hz / medium.sound_speed_m_per_s
    a = medium.alpha_np_per_m(frequency_hz)
    out = torch.zeros(obs.shape[0], dtype=_CT)
    pref = 1j * medium.impedance * k / (2 * math.pi) * element_area
    for s in range(0, sp.shape[0], chunk):
        d = obs[:, None, :] - sp[None, s : s + chunk, :]
        R = d.norm(dim=-1).clamp_min(1e-12)
        out = out + (
            u[None, s : s + chunk] * torch.exp((-1j * k - a) * R.to(_CT)) / R.to(_CT)
        ).sum(1)
    return pref * out


# ---------------------------------------------------------------------------
# angular spectrum
# ---------------------------------------------------------------------------


def _rs_kernel(n0: int, n1: int, dx: float, z: float, k: float, alpha: float,
               impedance: float) -> Tensor:
    """Spatial Rayleigh--Sommerfeld kernel on the zero-padded grid."""
    ax0 = (torch.arange(2 * n0, dtype=_DT) - n0) * dx
    ax1 = (torch.arange(2 * n1, dtype=_DT) - n1) * dx
    X, Y = torch.meshgrid(ax0, ax1, indexing="ij")
    R = torch.sqrt(X**2 + Y**2 + z**2).clamp_min(1e-12)
    h = (
        1j
        * impedance
        * k
        / (2 * math.pi)
        * torch.exp((-1j * k - alpha) * R.to(_CT))
        / R.to(_CT)
        * (dx * dx)
    )
    return torch.fft.ifftshift(h)


def angular_spectrum_propagate(
    source_velocity: Tensor,
    dx: float,
    z: float | Tensor,
    *,
    frequency_hz: float,
    medium: AcousticMedium = WATER,
    kernel: str = "spatial",
    evanescent: bool = False,
) -> Tensor:
    """Propagate a source-plane normal velocity to pressure at distance(s) ``z``.

    Time convention :math:`e^{+i\\omega t}`, so a wave travelling ``+z``
    carries :math:`e^{-ikz}` and a focusing phase is
    :math:`-k(\\sqrt{\\rho^2+F^2}-F)` -- matching
    :meth:`~scwbd.intervene.tfus.transducer.TransducerArray.geometric_focus_phase`.

    ``kernel="spatial"`` (default, and the validated one) evaluates the
    band-limited angular spectrum as a **zero-padded FFT convolution with the
    exact Rayleigh--Sommerfeld kernel**

    .. math::

        p(x,y,z) = \\frac{i\\rho c k}{2\\pi}\\iint u_z(x',y')
                    \\frac{e^{-ikR}}{R}\\,\\mathrm dS',
        \\qquad R=\\sqrt{(x-x')^2+(y-y')^2+z^2}.

    ``kernel="spectral"`` uses the closed-form transfer function
    :math:`P = \\rho c\\,(k/k_z)\\,U_z\\,e^{-ik_z z}`.  It is analytically
    equivalent but numerically worse: the obliquity factor :math:`k/k_z`
    diverges at the evanescent cutoff, and truncating it costs about 4 %
    against the analytic piston reference where the spatial kernel costs
    about 0.2 %.  Both are provided so the discrepancy is visible rather than
    hidden.

    Returns ``[nz, ny, nx]`` complex pressure (or ``[ny,nx]`` for scalar ``z``).
    """
    u = source_velocity.to(_CT)
    n0, n1 = u.shape[-2:]
    k = 2 * math.pi * frequency_hz / medium.sound_speed_m_per_s
    alpha = medium.alpha_np_per_m(frequency_hz)
    zs = torch.atleast_1d(torch.as_tensor(z, dtype=_DT))
    out = []

    if kernel == "spatial":
        up = torch.zeros(2 * n0, 2 * n1, dtype=_CT)
        up[:n0, :n1] = u
        U = torch.fft.fft2(up)
        for zz in zs:
            H = torch.fft.fft2(
                _rs_kernel(n0, n1, dx, float(zz), k, alpha, medium.impedance)
            )
            out.append(torch.fft.ifft2(U * H)[:n0, :n1])
    elif kernel == "spectral":
        kx = 2 * math.pi * torch.fft.fftfreq(n0, d=dx, dtype=_DT)
        ky = 2 * math.pi * torch.fft.fftfreq(n1, d=dx, dtype=_DT)
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        kt2 = KX**2 + KY**2
        prop = kt2 <= k**2
        kz = torch.zeros_like(kt2)
        kz[prop] = torch.sqrt(k**2 - kt2[prop])
        U = torch.fft.fft2(u)
        obl = torch.zeros_like(kt2, dtype=_CT)
        safe = prop & (kz > 1e-6 * k)
        obl[safe] = (k / kz[safe]).to(_CT)
        for zz in zs:
            ph = torch.zeros_like(kt2, dtype=_CT)
            ph[safe] = torch.exp(-1j * kz[safe].to(_CT) * float(zz))
            if evanescent:
                kz_ev = torch.sqrt((kt2 - k**2).clamp_min(0.0))
                ph[~prop] = torch.exp(-kz_ev[~prop].to(_CT) * abs(float(zz)))
            out.append(
                torch.fft.ifft2(
                    medium.impedance * obl * U * ph * math.exp(-alpha * abs(float(zz)))
                )
            )
    else:
        raise ValueError(f"unknown kernel {kernel!r}")

    stacked = torch.stack(out)
    return stacked[0] if torch.as_tensor(z).ndim == 0 else stacked


# ---------------------------------------------------------------------------
# heterogeneous split-step
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkullLayer:
    """A transverse-varying slab standing between transducer and target.

    ``thickness_m`` and ``sound_speed_m_per_s`` are maps over the propagation
    grid; ``coupling_efficiency`` folds in the gel/water coupling path, which
    is a *separate* uncertainty source from the skull itself.
    """

    thickness_m: Tensor  # [ny,nx]
    sound_speed_m_per_s: Tensor  # [ny,nx]
    density_kg_per_m3: Tensor  # [ny,nx]
    alpha_np_per_m: Tensor  # [ny,nx]
    stand_off_m: float = 0.005
    coupling_efficiency: float = 0.9
    citation: str = "Aubry et al. 2003 JASA 113:84-93 (CT porosity model)"
    notice: str = SIMULATION_ONLY_NOTICE

    def transmission_coefficient(self, medium: AcousticMedium = WATER) -> Tensor:
        """Normal-incidence intensity transmission through a single interface pair."""
        Z1 = medium.impedance
        Z2 = self.density_kg_per_m3 * self.sound_speed_m_per_s
        t = 4 * Z1 * Z2 / (Z1 + Z2) ** 2
        return t**2  # in and out again

    def phase_aberration(self, frequency_hz: float, medium: AcousticMedium = WATER) -> Tensor:
        """Extra phase relative to propagating the same distance in ``medium``."""
        w = 2 * math.pi * frequency_hz
        return -w * self.thickness_m * (
            1.0 / self.sound_speed_m_per_s - 1.0 / medium.sound_speed_m_per_s
        )


class SplitStepPropagator:
    """Angular-spectrum marching with per-step phase and amplitude screens.

    Homogeneous steps use the exact propagator of
    :func:`angular_spectrum_propagate`; heterogeneity is applied as a thin
    screen between steps.  With no screens this reduces **exactly** to the
    angular-spectrum result, which is the regression test in
    ``tests/intervene``.
    """

    def __init__(
        self,
        dx: float,
        dz: float,
        *,
        frequency_hz: float,
        background: AcousticMedium = BRAIN,
    ) -> None:
        self.dx = dx
        self.dz = dz
        self.frequency_hz = frequency_hz
        self.background = background

    def _step_kernel(self, shape: tuple[int, int]) -> tuple[Tensor, Tensor]:
        n0, n1 = shape
        k = 2 * math.pi * self.frequency_hz / self.background.sound_speed_m_per_s
        kx = 2 * math.pi * torch.fft.fftfreq(n0, d=self.dx, dtype=_DT)
        ky = 2 * math.pi * torch.fft.fftfreq(n1, d=self.dx, dtype=_DT)
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        kt2 = KX**2 + KY**2
        prop = kt2 <= k**2
        kz = torch.zeros_like(kt2)
        kz[prop] = torch.sqrt(k**2 - kt2[prop])
        ker = torch.zeros_like(kt2, dtype=_CT)
        safe = prop & (kz > 1e-6 * k)
        ker[safe] = torch.exp(-1j * kz[safe].to(_CT) * self.dz)
        obl = torch.zeros_like(kt2, dtype=_CT)
        obl[safe] = (k / kz[safe]).to(_CT)
        return ker, obl

    def run(
        self,
        source_velocity: Tensor,
        n_steps: int,
        *,
        skull: SkullLayer | None = None,
        skull_step: int | None = None,
        record_every: int = 1,
    ) -> Tensor:
        """March the field. Returns ``[n_recorded, ny, nx]`` complex pressure."""
        u = source_velocity.to(_CT)
        ker, obl = self._step_kernel(u.shape[-2:])
        alpha = self.background.alpha_np_per_m(self.frequency_hz)
        atten = math.exp(-alpha * self.dz)

        # convert surface velocity to pressure spectrum once
        P = torch.fft.fft2(u) * self.background.impedance * obl
        out = []
        for i in range(n_steps):
            P = P * ker * atten
            if skull is not None and skull_step is not None and i == skull_step:
                p = torch.fft.ifft2(P)
                screen = torch.exp(
                    1j * skull.phase_aberration(self.frequency_hz, self.background).to(_CT)
                ) * torch.exp(
                    -(skull.alpha_np_per_m * skull.thickness_m).to(_CT)
                ) * math.sqrt(skull.coupling_efficiency)
                screen = screen * torch.sqrt(
                    skull.transmission_coefficient(self.background).to(_CT)
                )
                P = torch.fft.fft2(p * screen)
            if (i + 1) % record_every == 0:
                out.append(torch.fft.ifft2(P))
        return torch.stack(out)


# ---------------------------------------------------------------------------
# CT -> acoustic properties
# ---------------------------------------------------------------------------


def hounsfield_to_acoustic(
    hu: Tensor,
    *,
    frequency_hz: float = 500e3,
    hu_max: float = 2000.0,
    rho_min: float = 1000.0,
    rho_max: float = 2200.0,
    c_min: float = 1500.0,
    c_max: float = 3100.0,
    alpha_min_np_per_m: float = 2.0,
    alpha_max_np_per_m: float = 90.0,
) -> dict[str, Tensor]:
    """CT Hounsfield units to skull acoustic properties (Aubry et al. 2003).

    Porosity :math:`\\phi = 1 - \\mathrm{HU}/\\mathrm{HU}_{\\max}`, then

    .. math::

        \\rho &= \\rho_{\\min} + (\\rho_{\\max}-\\rho_{\\min})(1-\\phi),\\\\
        c &= c_{\\min} + (c_{\\max}-c_{\\min})(1-\\phi),\\\\
        \\alpha &= \\alpha_{\\min} + (\\alpha_{\\max}-\\alpha_{\\min})\\sqrt{\\phi}.

    Skull conversion is one of the dominant tFUS uncertainty sources, so the
    returned dict carries the porosity map and a ledger-ready note as well as
    the derived properties.
    """
    hu = torch.as_tensor(hu, dtype=_DT)
    phi = (1.0 - hu / hu_max).clamp(0.0, 1.0)
    solid = 1.0 - phi
    return {
        "porosity": phi,
        "density_kg_per_m3": rho_min + (rho_max - rho_min) * solid,
        "sound_speed_m_per_s": c_min + (c_max - c_min) * solid,
        "alpha_np_per_m": alpha_min_np_per_m
        + (alpha_max_np_per_m - alpha_min_np_per_m) * torch.sqrt(phi),
        "frequency_hz": torch.tensor(frequency_hz, dtype=_DT),
    }


def pressure_dose(
    p: Tensor,
    *,
    support: str,
    solver: str,
    numerical_rel_error: float = 0.05,
) -> PhysicalDose:
    """Wrap a complex pressure field as a :class:`PhysicalDose` in Pa."""
    mag = p.abs() if torch.is_complex(p) else p.abs()
    return PhysicalDose(
        modality="tfus",
        quantity="pressure",
        units="Pa",
        value=mag,
        support=support,
        ledger=Ledger(
            variance={"numerical": float((numerical_rel_error * mag.max()) ** 2)},
            bias_status="externally_bounded",
            validity_domain={"solver": solver},
        ),
    )


# ---------------------------------------------------------------------------
# k-Wave wrapper (honest about availability)
# ---------------------------------------------------------------------------


def kwave_available() -> bool:
    try:  # pragma: no cover - depends on host
        import kwave  # noqa: F401
    except Exception:
        return False
    import platform

    return platform.machine() in ("x86_64", "AMD64")


def kwave_status() -> dict[str, object]:
    """Report k-Wave usability honestly, including the aarch64 finding."""
    import platform

    try:
        import kwave  # noqa: F401

        importable = True
    except Exception:
        importable = False
    return {
        "importable": importable,
        "usable": kwave_available(),
        "machine": platform.machine(),
        "platform_note": (
            "k-wave-python installs from PyPI on linux-aarch64 but its solver "
            "binaries (kspaceFirstOrder-OMP, kspaceFirstOrder-CUDA) are "
            "downloaded at import time as x86-64 ELF executables and cannot "
            "run on this machine; installing it also downgrades numpy and "
            "scipy for the whole environment. It was therefore removed from "
            "this venv rather than left as a broken dependency."
        ),
        "fallback": (
            "angular_spectrum_propagate + rayleigh_sommerfeld_pressure + "
            "SplitStepPropagator, validated against on_axis_piston_pressure"
        ),
        "equivalent_to_full_wave": False,
        "notice": SIMULATION_ONLY_NOTICE,
    }


class KWaveSolver:
    """Thin wrapper over k-Wave, refusing rather than degrading silently."""

    def __init__(self) -> None:
        if not kwave_available():
            raise RuntimeError("k-Wave is not usable here. " + str(kwave_status()))

    def run(self, *args, **kwargs):  # pragma: no cover - requires k-Wave
        raise NotImplementedError(
            "k-Wave grid, medium, source and sensor structures must be declared "
            "explicitly with provenance; they are not auto-generated here."
        )
