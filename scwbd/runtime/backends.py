"""Field solve, candidate response operators, and network propagators.

Three deliberately separate stages, mirroring ``body.tex`` Sec. 7.2: *"The TMS
stack separates coil pose, induced field, cortical orientation, tissue
coupling, immediate population response, network propagation, and
plasticity."*

* :class:`EFieldBackend` -- coil pose to induced E-field.  Agent G owns the
  real solver; until it lands the runtime uses
  :class:`AnalyticSphericalEField`, which is a closed-form model and is
  *labelled* as one everywhere it appears.
* :class:`ResponseOperator` -- E-field to drive on a named target population.
  There are several, they disagree, and the disagreement is the output.
* :class:`NetworkPropagator` -- drive to distributed change.  Several model
  classes, retained rather than selected (thesis Sec. 0.5 step 4: "Models that
  fit passive data but disagree about propagation are retained").

Claim limits
------------
None of these operators is claimed to be neurally realized.  The analytic field
backend is an *effective* model of a spherically symmetric conductor; the
response operators are *effective*/*functional*; the network propagators are
*surrogates*.  Nothing here has been validated against a measured cortical
field or a measured evoked response.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Protocol

import torch
from torch import Tensor

from ._compat import Pose, efield_solver, figure_eight_coil
from .head import HeadModel

__all__ = [
    "MU0_OVER_4PI",
    "CoilSpec",
    "EFieldBackend",
    "AnalyticSphericalEField",
    "resolve_efield_backend",
    "ResponseOperator",
    "MagnitudeThresholdResponse",
    "NormalComponentResponse",
    "TangentialDirectionResponse",
    "DEFAULT_RESPONSE_OPERATORS",
    "NetworkPropagator",
    "LinearDiffusionPropagator",
    "SaturatingPropagator",
    "DEFAULT_PROPAGATORS",
]

_DT = torch.float64

#: :math:`\mu_0 / 4\pi` in H/m.
MU0_OVER_4PI = 1.0e-7


# ---------------------------------------------------------------------------
# coil
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoilSpec:
    """A coil as an equivalent magnetisation sheet, in the coil frame.

    ``dipole_moment_per_amp`` has units m^2; the moment at current ``I`` is
    ``I * moment_per_amp``, so the *rate* that drives the induced field is
    ``didt_a_per_s * moment_per_amp``.

    ``didt_relative_sd`` is the device-gain prior.  Stimulator output is not
    known exactly and the field scales linearly in it, so this is the one
    genuinely non-geometric variance term the analytic backend carries.
    """

    device_id: str
    dipole_positions: Tensor
    dipole_moment_per_amp: Tensor
    didt_a_per_s: float = 4.0e7
    didt_relative_sd: float = 0.05
    frame: str = "coil"
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        p = torch.as_tensor(self.dipole_positions, dtype=_DT)
        m = torch.as_tensor(self.dipole_moment_per_amp, dtype=_DT)
        if p.ndim != 2 or p.shape[1] != 3 or m.shape != p.shape:
            raise ValueError("coil dipoles must be [D,3] positions and moments")
        if self.didt_relative_sd < 0.0:
            raise ValueError("didt_relative_sd must be non-negative")
        object.__setattr__(self, "dipole_positions", p)
        object.__setattr__(self, "dipole_moment_per_amp", m)

    @property
    def n_dipoles(self) -> int:
        return int(self.dipole_positions.shape[0])

    @classmethod
    def figure_eight(
        cls,
        *,
        device_id: str = "figure8_70mm",
        didt_a_per_s: float = 4.0e7,
        didt_relative_sd: float = 0.05,
        n_azimuth: int = 32,
        n_radial: int = 4,
    ) -> "CoilSpec":
        """Standard focal coil.

        Uses agent G's :class:`~scwbd.intervene.tms.coil.FigureEightCoil`
        discretisation when available; otherwise a two-lobe equivalent sheet
        with the same footprint.  Which one was used is recorded in ``notes``.
        """
        if figure_eight_coil is not None:
            geom = figure_eight_coil(n_azimuth=n_azimuth, n_radial=n_radial)
            pos, mom = geom.dipole_elements()
            return cls(
                device_id=device_id,
                dipole_positions=pos.to(_DT),
                dipole_moment_per_amp=mom.to(_DT),
                didt_a_per_s=didt_a_per_s,
                didt_relative_sd=didt_relative_sd,
                notes={"source": "scwbd.intervene.tms.coil.FigureEightCoil"},
            )
        pos, mom = _fallback_figure_eight(n_azimuth=n_azimuth, n_radial=n_radial)
        return cls(
            device_id=device_id,
            dipole_positions=pos,
            dipole_moment_per_amp=mom,
            didt_a_per_s=didt_a_per_s,
            didt_relative_sd=didt_relative_sd,
            notes={"source": "scwbd.runtime.backends fallback sheet"},
        )

    def coarsened(self, stride: int = 2) -> "CoilSpec":
        """A deliberately coarser discretisation, for a refinement check."""
        return CoilSpec(
            device_id=self.device_id + f"@stride{stride}",
            dipole_positions=self.dipole_positions[::stride],
            dipole_moment_per_amp=self.dipole_moment_per_amp[::stride] * float(stride),
            didt_a_per_s=self.didt_a_per_s,
            didt_relative_sd=self.didt_relative_sd,
            notes={**dict(self.notes), "coarsened_stride": stride},
        )


def _fallback_figure_eight(
    *, n_azimuth: int, n_radial: int, inner: float = 0.026, outer: float = 0.044,
    turns: int = 9, separation: float = 0.088, height: float = 0.005,
) -> tuple[Tensor, Tensor]:
    edges = torch.linspace(0.0, outer, n_radial + 1, dtype=_DT)
    rho = 0.5 * (edges[:-1] + edges[1:])
    ring_area = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2) / n_azimuth
    phi = (torch.arange(n_azimuth, dtype=_DT) + 0.5) * (2 * math.pi / n_azimuth)
    rr, pp = torch.meshgrid(rho, phi, indexing="ij")
    aa = ring_area[:, None].expand_as(rr)
    n_turns = torch.where(
        rr <= inner,
        torch.full_like(rr, float(turns)),
        (float(turns) * (outer - rr) / max(outer - inner, 1e-12)).clamp_min(0.0),
    )
    dens = (n_turns * aa).reshape(-1)
    positions: list[Tensor] = []
    moments: list[Tensor] = []
    for sign, dx in ((+1.0, -separation / 2.0), (-1.0, +separation / 2.0)):
        pos = torch.stack(
            [
                dx + rr * torch.cos(pp),
                rr * torch.sin(pp),
                torch.full_like(rr, height),
            ],
            dim=-1,
        ).reshape(-1, 3)
        mom = torch.zeros(pos.shape[0], 3, dtype=_DT)
        mom[:, 2] = sign * dens
        keep = dens > 0
        positions.append(pos[keep])
        moments.append(mom[keep])
    return torch.cat(positions), torch.cat(moments)


# ---------------------------------------------------------------------------
# E-field
# ---------------------------------------------------------------------------


class EFieldBackend(Protocol):
    """Coil pose -> induced E-field on the cortical sample points."""

    name: str
    backend_class: Literal["analytic", "numerical_fem", "learned", "unknown"]
    is_trained_artifact: bool

    def solve(self, head: HeadModel, pose: Pose, coil: CoilSpec) -> Tensor:
        """Return ``[N, 3]`` V/m in ``head.frame``. ``pose`` is ``head<-coil``."""
        ...


@dataclass(frozen=True)
class AnalyticSphericalEField:
    """Closed-form induced field for a spherically symmetric conductor.

    The primary (source) field of a magnetic dipole ``m`` at ``p`` is

    .. math:: E_p(r) = -\\frac{\\mu_0}{4\\pi}\\,
              \\frac{\\dot m \\times (r-p)}{|r-p|^3},

    and for a *spherically symmetric* conductor the secondary field from
    surface charge exactly cancels the radial component of the total field, so

    .. math:: E(r) = E_p(r) - \\bigl(E_p(r)\\cdot\\hat n\\bigr)\\hat n,
              \\qquad \\hat n = (r-c)/|r-c|.

    (Heller & van Hulsteyn 1992; the standard sphere result also used by the
    EEG/MEG forward literature.)  Two consequences are load-bearing and are
    stated rather than hidden:

    * the answer is **independent of conductivity**, so this backend cannot
      report a tissue-parameter variance term and must not be read as if it
      could;
    * a real head is not a sphere, so the model discrepancy against a
      finite-element solve on a subject mesh is **unbounded here** -- it is
      carried as a prior-specified sensitivity range, never as zero.
    """

    name: str = "analytic_spherical_primary"
    backend_class: str = "analytic"
    is_trained_artifact: bool = False
    #: Where this backend actually computes.  ``ARCHITECTURE.md`` Sec. 3 keeps
    #: solvers and covariance propagation in float64; this closed-form solve is
    #: cheap enough that CPU float64 is the right answer, and reporting "cuda"
    #: because CUDA happened to be available would be a false provenance.
    device: str = "cpu"
    #: Fractional model-discrepancy range vs an FEM solve on a real head.
    #: Deliberately wide; it is a declared prior, not a measurement.
    discrepancy_fraction: tuple[float, float] = (-0.4, 0.4)
    citation: str = (
        "Heller & van Hulsteyn 1992, Biophys J 63:129-138; "
        "Deng, Lisanby & Peterchev 2013, Brain Stimul 6:1-13"
    )

    def solve(self, head: HeadModel, pose: Pose, coil: CoilSpec) -> Tensor:
        if pose.child != coil.frame:
            raise ValueError(
                f"pose {pose.label} does not end at the coil frame "
                f"{coil.frame!r}; the field solve will not guess"
            )
        if pose.parent != head.frame:
            raise ValueError(
                f"pose {pose.label} is not expressed in the head model frame "
                f"{head.frame!r}; resolve the declared chain first"
            )
        R = pose.R
        t = pose.t
        # dipole positions and moment rates in the head frame
        p = (R @ coil.dipole_positions.T).T + t  # [D,3]
        mdot = (R @ coil.dipole_moment_per_amp.T).T * float(coil.didt_a_per_s)  # [D,3]

        r = head.cortex_vertices  # [N,3]
        diff = r.unsqueeze(1) - p.unsqueeze(0)  # [N,D,3]
        dist = torch.linalg.norm(diff, dim=-1).clamp_min(1e-6)  # [N,D]
        cross = torch.cross(mdot.unsqueeze(0).expand_as(diff), diff, dim=-1)
        primary = -MU0_OVER_4PI * (cross / dist.unsqueeze(-1) ** 3).sum(dim=1)  # [N,3]

        n_hat = r - head.centre
        n_hat = n_hat / torch.linalg.norm(n_hat, dim=-1, keepdim=True).clamp_min(1e-30)
        radial = (primary * n_hat).sum(dim=-1, keepdim=True)
        return primary - radial * n_hat


def resolve_efield_backend(explicit: Any = None) -> EFieldBackend:
    """Prefer agent G's solver; fall back to the analytic sphere and say so."""
    if explicit is not None:
        return explicit
    if efield_solver is not None:  # pragma: no cover - depends on agent G
        return _AgentGAdapter(efield_solver)
    return AnalyticSphericalEField()


@dataclass(frozen=True)
class _AgentGAdapter:  # pragma: no cover - depends on agent G landing
    """Thin adapter around ``scwbd.intervene.tms`` once its solver exists."""

    fn: Callable[..., Tensor]
    name: str = "scwbd.intervene.tms.solve_efield"
    backend_class: str = "numerical_fem"
    is_trained_artifact: bool = False
    discrepancy_fraction: tuple[float, float] = (-0.2, 0.2)
    citation: str = "scwbd.intervene.tms (agent G)"

    def solve(self, head: HeadModel, pose: Pose, coil: CoilSpec) -> Tensor:
        return torch.as_tensor(self.fn(head=head, pose=pose, coil=coil), dtype=_DT)


# ---------------------------------------------------------------------------
# response operators (level 2)
# ---------------------------------------------------------------------------


class ResponseOperator(Protocol):
    """A *named candidate* mapping from field to drive on a target population.

    The mechanism is unresolved, so these coexist under model comparison.  Each
    must be cheap: the serving path evaluates every operator on every
    perturbed field of the pose Jacobian.
    """

    name: str
    mechanistic_status: Literal["mechanistic", "effective", "functional", "surrogate"]
    units: str

    def drive(self, efield: Tensor, normals: Tensor, mask: Tensor) -> Tensor:
        """Per-vertex drive, ``[N]``, zero outside ``mask``."""
        ...


@dataclass(frozen=True)
class MagnitudeThresholdResponse:
    """Drive depends only on ``|E|`` above a threshold. Orientation-blind."""

    threshold_v_per_m: float = 40.0
    width_v_per_m: float = 20.0
    name: str = "efield_magnitude_threshold"
    mechanistic_status: str = "effective"
    units: str = "dimensionless"

    def drive(self, efield: Tensor, normals: Tensor, mask: Tensor) -> Tensor:
        mag = torch.linalg.norm(efield, dim=-1)
        z = (mag - self.threshold_v_per_m) / max(self.width_v_per_m, 1e-9)
        return torch.where(mask, torch.nn.functional.softplus(z).to(_DT), torch.zeros_like(mag))


@dataclass(frozen=True)
class NormalComponentResponse:
    """Drive depends on the component *into* the cortical surface.

    The usual account of pyramidal-axon activation: what matters is the field
    component along the inward normal of the sulcal wall, not ``|E|``.  Sharing
    a target region with :class:`MagnitudeThresholdResponse` but disagreeing
    about orientation is precisely why a pose-uncertainty inflation makes these
    two diverge.
    """

    threshold_v_per_m: float = 25.0
    width_v_per_m: float = 15.0
    name: str = "normal_component_rectified"
    mechanistic_status: str = "effective"
    units: str = "dimensionless"

    def drive(self, efield: Tensor, normals: Tensor, mask: Tensor) -> Tensor:
        inward = -(efield * normals).sum(dim=-1)
        z = (inward - self.threshold_v_per_m) / max(self.width_v_per_m, 1e-9)
        return torch.where(mask, torch.nn.functional.softplus(z).to(_DT), torch.zeros_like(z))


@dataclass(frozen=True)
class TangentialDirectionResponse:
    """Drive depends on the field projected on a declared tangential direction.

    ``direction`` is in the head frame and is projected onto the local tangent
    plane per vertex.  A *functional* operator: it encodes an empirical
    orientation preference without a mechanism.
    """

    direction: tuple[float, float, float] = (0.0, 1.0, 0.0)
    threshold_v_per_m: float = 20.0
    width_v_per_m: float = 15.0
    name: str = "tangential_directional"
    mechanistic_status: str = "functional"
    units: str = "dimensionless"

    def drive(self, efield: Tensor, normals: Tensor, mask: Tensor) -> Tensor:
        d = torch.tensor(self.direction, dtype=_DT)
        d = d / torch.linalg.norm(d).clamp_min(1e-30)
        d_t = d.unsqueeze(0) - (normals @ d).unsqueeze(-1) * normals
        d_t = d_t / torch.linalg.norm(d_t, dim=-1, keepdim=True).clamp_min(1e-30)
        proj = (efield * d_t).sum(dim=-1).abs()
        z = (proj - self.threshold_v_per_m) / max(self.width_v_per_m, 1e-9)
        return torch.where(mask, torch.nn.functional.softplus(z).to(_DT), torch.zeros_like(z))


#: At least two are required before any comparison is admissible; the A_safe
#: file's ``decision.epistemic.min_candidate_models`` enforces that.
DEFAULT_RESPONSE_OPERATORS: tuple[ResponseOperator, ...] = (
    MagnitudeThresholdResponse(),
    NormalComponentResponse(),
    TangentialDirectionResponse(),
)


# ---------------------------------------------------------------------------
# network propagation (level 3)
# ---------------------------------------------------------------------------


class NetworkPropagator(Protocol):
    """A named model class mapping parcel drive to distributed change."""

    name: str
    mechanistic_status: Literal["mechanistic", "effective", "functional", "surrogate"]

    def propagate(self, drive: Tensor, connectivity: Tensor, horizon_s: float) -> Tensor:
        """``[P]`` predicted change per parcel over ``horizon_s`` seconds."""
        ...


@dataclass(frozen=True)
class LinearDiffusionPropagator:
    """Truncated linear spread on the normalised topology prior."""

    gain: float = 1.0
    leak: float = 6.0
    order: int = 6
    name: str = "linear_diffusion_surrogate"
    mechanistic_status: str = "surrogate"

    def propagate(self, drive: Tensor, connectivity: Tensor, horizon_s: float) -> Tensor:
        w = _row_normalise(connectivity)
        out = torch.zeros_like(drive)
        term = drive.clone()
        step = min(float(horizon_s) * self.leak, 1.0)
        for k in range(1, self.order + 1):
            term = step * (w @ term)
            out = out + term / math.factorial(k)
        return self.gain * (drive + out)


@dataclass(frozen=True)
class SaturatingPropagator:
    """Iterated saturating spread: same topology, different model class."""

    gain: float = 1.0
    leak: float = 6.0
    steps: int = 4
    saturation: float = 1.5
    name: str = "saturating_surrogate"
    mechanistic_status: str = "surrogate"

    def propagate(self, drive: Tensor, connectivity: Tensor, horizon_s: float) -> Tensor:
        w = _row_normalise(connectivity)
        x = drive.clone()
        dt = float(horizon_s) * self.leak / max(self.steps, 1)
        for _ in range(self.steps):
            x = x + dt * (torch.tanh((w @ x) / self.saturation) * self.saturation - 0.5 * x)
        return self.gain * x


def _row_normalise(w: Tensor) -> Tensor:
    s = w.sum(dim=1, keepdim=True).clamp_min(1e-30)
    return w / s


DEFAULT_PROPAGATORS: tuple[NetworkPropagator, ...] = (
    LinearDiffusionPropagator(),
    SaturatingPropagator(),
)

