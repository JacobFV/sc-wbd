"""Field solve, candidate response operators, and network propagators.

Three deliberately separate stages, mirroring ``body.tex`` Sec. 7.2: *"The TMS
stack separates coil pose, induced field, cortical orientation, tissue
coupling, immediate population response, network propagation, and
plasticity."*

* :class:`EFieldBackend` -- coil pose to induced E-field.  Agent G owns the
  real solvers and they have landed:
  :class:`GatedAnalyticSphereEField` (gate ``N6_induced_efield``) is the
  default, and :class:`ChargeBEMEField` (``N6`` + ``N8_induced_efield_contact``)
  is selectable for non-spherical geometry.
  :class:`AnalyticSphericalEField` remains only as a fallback for when
  ``scwbd.intervene.tms.efield`` is not importable; it is an *approximation*,
  measurably so, and is labelled as one everywhere it appears.
* :class:`ResponseOperator` -- E-field to drive on a named target population.
  There are several, they disagree, and the disagreement is the output.
* :class:`NetworkPropagator` -- drive to distributed change.  Several model
  classes, retained rather than selected (thesis Sec. 0.5 step 4: "Models that
  fit passive data but disagree about propagation are retained").

Claim limits
------------
None of these operators is claimed to be neurally realized.  The field backends
model a spherically symmetric conductor and are validated *numerically* -- the
gates compare a computation against an independent reference, which lifts a
precondition and licenses nothing downstream.  The response operators are
*effective*/*functional*; the network propagators are *surrogates*.  Nothing
here has been validated against a measured cortical field or a measured evoked
response, and a numerical gate does not make a sphere a head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, Protocol

import torch
from torch import Tensor

from . import _compat
from ._compat import Pose, figure_eight_coil
from .head import HeadModel

__all__ = [
    "MU0_OVER_4PI",
    "CoilSpec",
    "FieldSolve",
    "EFieldBackend",
    "AnalyticSphericalEField",
    "GatedAnalyticSphereEField",
    "ChargeBEMEField",
    "FieldResolutionUnresolved",
    "ImpossiblePlacement",
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

    **Frame convention, and it is load-bearing.**  The coil frame's origin is
    the *head-facing face* and its ``+z`` axis points **away** from the head;
    windings sit at ``+winding_height`` above the face.  That is agent G's
    declared convention (``CoilGeometry.winding_height``, "above the head-facing
    face"), and getting it backwards buries the windings inside the scalp.  The
    gated solver refuses that outright
    (:class:`~scwbd.intervene.tms.efield.ImpossibleGeometry`) rather than
    returning the pole of a rational function, which is how the convention
    mismatch was found.

    ``geometry`` and ``pulse`` carry agent G's own ``CoilGeometry`` and
    ``TMSPulse`` when they are available, so the gated field solver can be
    driven with the objects it was validated against rather than a
    re-derivation of them.
    """

    device_id: str
    dipole_positions: Tensor
    dipole_moment_per_amp: Tensor
    didt_a_per_s: float = 4.0e7
    didt_relative_sd: float = 0.05
    frame: str = "coil"
    geometry: Any = None
    pulse: Any = None
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
            pulse = None
            if _compat.tms_pulse_biphasic is not None:
                pulse = _compat.tms_pulse_biphasic(peak_didt=didt_a_per_s)
            return cls(
                device_id=device_id,
                dipole_positions=pos.to(_DT),
                dipole_moment_per_amp=mom.to(_DT),
                didt_a_per_s=didt_a_per_s,
                didt_relative_sd=didt_relative_sd,
                geometry=geom,
                pulse=pulse,
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
        """A deliberately coarser discretisation, for a refinement check.

        When the spec carries agent G's ``CoilGeometry`` the coarsening happens
        *there*, by re-tiling the equivalent sheet at a lower azimuthal and
        radial count, and the dipoles are re-derived from it.  Subsampling the
        dipole array instead would leave the geometry and the sheet describing
        different coils, and the gated solver is driven by the geometry -- so
        the refinement check would silently compare a coil against itself.
        """
        if self.geometry is not None:
            coarse_geom = replace(
                self.geometry,
                n_azimuth=max(4, int(self.geometry.n_azimuth) // stride),
                n_radial=max(1, int(self.geometry.n_radial) // stride),
            )
            pos, mom = coarse_geom.dipole_elements()
            return CoilSpec(
                device_id=self.device_id + f"@stride{stride}",
                dipole_positions=pos.to(_DT),
                dipole_moment_per_amp=mom.to(_DT),
                didt_a_per_s=self.didt_a_per_s,
                didt_relative_sd=self.didt_relative_sd,
                geometry=coarse_geom,
                pulse=self.pulse,
                notes={**dict(self.notes), "coarsened_stride": stride},
            )
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


class FieldResolutionUnresolved(RuntimeError):
    """The solver refused because its discretisation does not resolve the source.

    Wraps ``ChargeBEM.assert_resolves_sources``' refusal (``R06``).  This is a
    *modelling* gap, not a placement error: the geometry is physical, the mesh
    is too coarse to attach an error bound to it, and gate N8_induced_efield_contact measured
    non-monotonic refinement beyond the envelope, so a coarser mesh can score
    better and a user would have no way to tell that from convergence.  The
    runtime turns this into ``Defer``, never into a number.
    """

    def __init__(self, message: str, *, remedy: str = "", resolution: Any = None) -> None:
        super().__init__(message)
        self.remedy = remedy
        self.resolution = dict(resolution or {})


class ImpossiblePlacement(RuntimeError):
    """The coil is not outside the head, so there is no field to compute.

    Wraps the other half of ``ImpossibleGeometry`` (``R06``): a source element
    inside the conductor, or a field point outside it.  This is not an
    inaccuracy to be bounded; the interior solution's denominator passes through
    zero and returns a large, smooth, entirely fictitious number.  The runtime
    turns this into ``Refuse(code="R06")``.
    """

    def __init__(self, message: str, *, remedy: str = "", detail: Any = None) -> None:
        super().__init__(message)
        self.remedy = remedy
        self.detail = dict(detail or {})


@dataclass(frozen=True)
class FieldSolve:
    """A field plus everything the solver measured about its own accuracy.

    ``numerical_variance`` and ``relative_error_bound`` come **from the solver**,
    not from this module.  For the charge BEM they are
    ``bem_error_envelope(panel_to_standoff)`` evaluated on the measured
    near-source resolution -- a step function over gate N8_induced_efield_contact's refinement
    table -- so the ledger cites a study rather than a constant somebody typed.
    """

    e: Tensor
    numerical_variance: float = 0.0
    relative_error_bound: float | None = None
    resolution: Mapping[str, float] | None = None
    validity_domain: Mapping[str, Any] = field(default_factory=dict)


class EFieldBackend(Protocol):
    """Coil pose -> induced E-field on the cortical sample points."""

    name: str
    backend_class: Literal[
        "analytic", "numerical_bem", "numerical_fem", "learned", "unknown"
    ]
    is_trained_artifact: bool

    def solve(self, head: HeadModel, pose: Pose, coil: CoilSpec) -> Tensor:
        """Return ``[N, 3]`` V/m in ``head.frame``. ``pose`` is ``head<-coil``."""
        ...

    def solve_field(self, head: HeadModel, pose: Pose, coil: CoilSpec) -> FieldSolve:
        """Same solve, plus the solver's own accuracy report."""
        ...


def _check_frames(head: HeadModel, pose: Pose, coil: CoilSpec) -> None:
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


def _dipoles_in_head_frame(pose: Pose, coil: CoilSpec) -> tuple[Tensor, Tensor]:
    """``(positions, moment rates)`` of the coil elements, in the head frame."""
    R, t = pose.R, pose.t
    p = (R @ coil.dipole_positions.T).T + t
    mdot = (R @ coil.dipole_moment_per_amp.T).T * float(coil.didt_a_per_s)
    return p, mdot


@dataclass(frozen=True)
class AnalyticSphericalEField:
    """Tangential projection of the primary field. **An approximation.**

    The primary (source) field of a magnetic dipole ``m`` at ``p`` is

    .. math:: E_p(r) = -\\frac{\\mu_0}{4\\pi}\\,
              \\frac{\\dot m \\times (r-p)}{|r-p|^3},

    and this backend returns its tangential part,
    ``E_p - (E_p . n) n``.

    That is **not** the Sarvas / Heller--van Hulsteyn interior solution.  The
    total field in a spherically symmetric conductor does have zero radial
    component, but the secondary field is not merely minus the radial part of
    the primary -- it carries a tangential component too, and dropping it
    overestimates the answer.  Measured against the gated solver on the shipped
    phantom, this backend is high by a factor of ~1.54 at the peak, with the
    direction essentially unchanged
    (``tests/runtime/test_field_backends.py`` records it).

    It exists solely as a fallback for when
    ``scwbd.intervene.tms.efield`` is not importable, so that the runtime's
    *structure* -- ledgers, covariance propagation, refusals -- can be
    exercised without agent G.  :func:`resolve_efield_backend` prefers
    :class:`GatedAnalyticSphereEField` whenever it exists, and the provenance
    records which one ran.

    Two further consequences, stated rather than hidden:

    * the answer is **independent of conductivity**, so this backend cannot
      report a tissue-parameter variance term and must not be read as if it
      could;
    * a real head is not a sphere, so the model discrepancy against a solve on
      a subject mesh is carried as a prior-specified sensitivity range, never
      as zero.
    """

    name: str = "runtime_fallback_primary_tangential_projection"
    backend_class: str = "analytic"
    is_trained_artifact: bool = False
    #: Where this backend actually computes.  ``ARCHITECTURE.md`` Sec. 3 keeps
    #: solvers and covariance propagation in float64; this closed-form solve is
    #: cheap enough that CPU float64 is the right answer, and reporting "cuda"
    #: because CUDA happened to be available would be a false provenance.
    device: str = "cpu"
    #: Wider than the gated backends', because it carries the sphere-vs-head
    #: geometry prior *and* this approximation's own measured overestimate.
    discrepancy_fraction: tuple[float, float] = (-0.8, 0.8)
    #: No gate has been run against this. Empty is the honest value.
    gate_evidence: tuple[str, ...] = ()
    citation: str = (
        "approximation; the reference solution is Heller & van Hulsteyn 1992, "
        "Biophys J 63:129-138, implemented in scwbd.intervene.tms.efield"
    )

    def solve(self, head: HeadModel, pose: Pose, coil: CoilSpec) -> Tensor:
        return self.solve_field(head, pose, coil).e

    def solve_field(self, head: HeadModel, pose: Pose, coil: CoilSpec) -> FieldSolve:
        _check_frames(head, pose, coil)
        p, mdot = _dipoles_in_head_frame(pose, coil)

        r = head.cortex_vertices  # [N,3]
        diff = r.unsqueeze(1) - p.unsqueeze(0)  # [N,D,3]
        dist = torch.linalg.norm(diff, dim=-1).clamp_min(1e-6)  # [N,D]
        cross = torch.cross(mdot.unsqueeze(0).expand_as(diff), diff, dim=-1)
        primary = -MU0_OVER_4PI * (cross / dist.unsqueeze(-1) ** 3).sum(dim=1)  # [N,3]

        n_hat = r - head.centre
        n_hat = n_hat / torch.linalg.norm(n_hat, dim=-1, keepdim=True).clamp_min(1e-30)
        radial = (primary * n_hat).sum(dim=-1, keepdim=True)
        return FieldSolve(
            e=primary - radial * n_hat,
            # closed form: no discretisation of the conductor, so no
            # discretisation error. The coil's own discretisation is measured
            # separately by the runtime's refinement check.
            numerical_variance=0.0,
            relative_error_bound=0.0,
            validity_domain={
                "solver": self.name,
                "geometry": "spherically_symmetric",
                "gate_evidence": (),
            },
        )


# ---------------------------------------------------------------------------
# the gated solvers (agent G / Faraday)
# ---------------------------------------------------------------------------


def _spherical_head_for(head: HeadModel) -> Any:
    """Build agent G's ``SphericalHeadModel`` for a spherical ``HeadModel``.

    Refuses a head model whose cortical samples are not on a sphere about its
    declared centre.  The analytic interior solution is a theorem about a
    spherically symmetric conductor; applying it to something else is not an
    approximation with a bound, it is a different problem.
    """
    if _compat.spherical_head_model is None:  # pragma: no cover - agent G absent
        raise ImpossiblePlacement("scwbd.intervene.tms.efield is not available")
    radii = torch.linalg.norm(head.cortex_vertices - head.centre, dim=-1)
    cortex_r = float(radii.mean())
    spread = float((radii - cortex_r).abs().max())
    if spread > 1e-4:
        raise ImpossiblePlacement(
            f"head model {head.subject_id!r} is not spherically symmetric about "
            f"its declared centre (cortical radii vary by {spread * 1e3:.2f} mm); "
            "the closed-form interior solution is a theorem about a spherically "
            "symmetric conductor and does not describe this geometry",
            remedy=(
                "supply a ChargeBEM mesh for the real surface, or declare a "
                "spherical head model"
            ),
            detail={"cortex_radius_spread_m": spread},
        )
    return _compat.spherical_head_model(
        radius=float(head.scalp_radius),
        radii=(float(head.scalp_radius),),
        conductivities=(float(head.conductivity_prior.get("brain_S_per_m", 0.33)),),
        cortex_radius=cortex_r,
    )


def _translate_refusal(exc: BaseException, *, context: str) -> RuntimeError:
    """Map agent G's ``ImpossibleGeometry`` onto the runtime's two answers."""
    remedy = str(getattr(exc, "remedy", ""))
    obj = getattr(exc, "offending_object", None)
    if _compat.is_resolution_refusal(exc):
        return FieldResolutionUnresolved(
            f"{context}: {exc}", remedy=remedy, resolution=obj
        )
    return ImpossiblePlacement(
        f"{context}: {exc}",
        remedy=remedy,
        detail=obj if isinstance(obj, Mapping) else {"offending": str(obj)},
    )


@dataclass(frozen=True)
class GatedAnalyticSphereEField:
    """``scwbd.intervene.tms.efield.analytic_sphere_efield``, gated by N6/N8.

    Preferred over :class:`AnalyticSphericalEField` because it is the
    implementation the field-physics gates were run against, and because it
    performs agent G's geometry preconditions -- a coil element inside the
    scalp is refused rather than evaluated at the pole of a rational function.

    What the gates do and do not license: N6/N8 validate the *induced-field
    computation*.  They say nothing about whether a sphere is a good model of a
    head, which is why :attr:`discrepancy_fraction` stays wide and separate.
    """

    name: str = "scwbd.intervene.tms.efield.analytic_sphere_efield"
    backend_class: str = "analytic"
    is_trained_artifact: bool = False
    device: str = "cpu"
    #: Fractional model-discrepancy range vs a solve on a real head mesh.  This
    #: is the *geometry* prior and is untouched by a numerical gate.
    discrepancy_fraction: tuple[float, float] = (-0.4, 0.4)
    gate_evidence: tuple[str, ...] = ("N6_induced_efield",)
    citation: str = (
        "Heller & van Hulsteyn 1992, Biophys J 63:129-138; "
        "reports/intervene/N6_induced_efield.md"
    )

    def solve(self, head: HeadModel, pose: Pose, coil: CoilSpec) -> Tensor:
        return self.solve_field(head, pose, coil).e

    def solve_field(self, head: HeadModel, pose: Pose, coil: CoilSpec) -> FieldSolve:
        _check_frames(head, pose, coil)
        sphere = _spherical_head_for(head)
        p, mdot = _dipoles_in_head_frame(pose, coil)
        centre = head.centre
        try:
            e = _compat.analytic_sphere_efield(
                head.cortex_vertices - centre, p - centre, mdot, head=sphere
            )
        except Exception as exc:  # agent G's ImpossibleGeometry, or worse
            if _compat.impossible_geometry is not None and isinstance(
                exc, _compat.impossible_geometry
            ):
                raise _translate_refusal(exc, context=self.name) from exc
            raise
        return FieldSolve(
            e=torch.as_tensor(e, dtype=_DT),
            numerical_variance=0.0,
            relative_error_bound=0.0,
            validity_domain={
                "solver": self.name,
                "geometry": "spherically_symmetric",
                "gate_evidence": list(self.gate_evidence),
                "conductivity_enters_solution": False,
            },
        )


class ChargeBEMEField:
    """``scwbd.intervene.tms.efield.ChargeBEM``, with N8's contact-regime gate.

    Two things this backend exists to carry, and they are different:

    * the **calibrated numerical bound**.  ``efield_from_coil(solver="bem")``
      measures the near-source resolution of the mesh it actually used and
      looks the relative-error bound up in gate
      ``N8_induced_efield_contact``'s refinement table.  The runtime consumes
      that number.  There is no fixed percentage anywhere on this path.
    * the **refusal**.  ``ChargeBEM.assert_resolves_sources`` refuses outside
      the validated envelope (``panel_to_standoff <= 1.0``), so the runtime
      cannot receive an unvalidated field by accident.  That refusal becomes
      ``Defer``, not an exception escaping to the consumer.

    The mesh is graded toward the coil, because contact geometry needs
    millimetre panels under the source and does not need them anywhere else --
    uniform refinement to that size is ~80 000 unknowns and a 53 GB dense
    matrix.  Sizing the refined patch is
    ``graded_icosphere_for_sources``' job, not this module's: it measures the
    source distribution's own angular extent, which for a figure-eight is ~41
    degrees of scalp because the wings carry the current and sit far off axis.
    The runtime passes source positions and lets the solver mesh for them.

    One mesh is built per nominal pose and reused across the pose Jacobian's
    perturbations, which move the source by 0.1 mm; the resolution guard is
    nevertheless re-evaluated on every perturbed pose, so a perturbation that
    leaves the envelope still refuses.

    Claim limit: N8 validates the *discretisation* of the induced-field
    computation at contact geometry to 0.73 % against an independent reference.
    It does not validate the head model, the response operators, or anything
    downstream of the field.
    """

    name = "scwbd.intervene.tms.efield.charge_bem"
    backend_class = "numerical_bem"
    is_trained_artifact = False
    device = "cpu"
    #: The sphere-vs-real-head geometry prior. A numerical gate does not move it.
    discrepancy_fraction: tuple[float, float] = (-0.4, 0.4)
    gate_evidence: tuple[str, ...] = (
        "N6_induced_efield",
        "N8_induced_efield_contact",
    )
    citation: str = (
        "Makarov et al. 2018 (surface-charge BEM); "
        "reports/intervene/N8_induced_efield_contact.md"
    )

    def __init__(
        self,
        *,
        base_subdiv: int = 2,
        grading_levels: int = 2,
        margin_rad: float = 0.12,
        uniform_subdiv: int | None = None,
    ) -> None:
        if _compat.efield_from_coil is None:  # pragma: no cover - agent G absent
            raise RuntimeError(
                "scwbd.intervene.tms.efield is not importable; the charge-BEM "
                "backend has nothing to wrap and the runtime will not "
                "re-implement it"
            )
        self.base_subdiv = int(base_subdiv)
        self.grading_levels = int(grading_levels)
        #: Passed through to ``graded_icosphere_for_sources``.  There is
        #: deliberately no half-angle knob here: sizing the refined patch is the
        #: solver's job and it does it from the sources, and a fixed cap that is
        #: too small produces a mesh that *looks* refined -- tiny on-axis
        #: panels, a third of the element count -- while measuring no better
        #: than no grading at all.
        self.margin_rad = float(margin_rad)
        #: Force a *uniform* mesh instead of a graded one. Used by the tests to
        #: construct a mesh deliberately too coarse for the source, which is
        #: how the ``Defer`` path is proved rather than asserted.
        self.uniform_subdiv = uniform_subdiv
        self._cache: dict[Any, Any] = {}

    # -- mesh ---------------------------------------------------------------
    def _bem_for(self, head: HeadModel, pose: Pose, coil: CoilSpec) -> Any:
        direction = pose.t - head.centre
        direction = direction / torch.linalg.norm(direction).clamp_min(1e-30)
        key = (
            head.subject_id,
            float(head.scalp_radius),
            self.uniform_subdiv,
            round(self.margin_rad, 4),
            tuple(round(float(v), 3) for v in direction),
        )
        if key in self._cache:
            return self._cache[key]
        if self.uniform_subdiv is not None:
            v, f = _compat.icosphere(self.uniform_subdiv)
            mesh = _compat.tri_mesh(v * float(head.scalp_radius), f)
        else:
            # the solver meshes for its own sources; the runtime supplies them
            # and does not decide how large a patch they need
            positions, _ = _dipoles_in_head_frame(pose, coil)
            mesh = _compat.graded_icosphere_for_sources(
                float(head.scalp_radius),
                self.base_subdiv,
                positions - head.centre,
                self.grading_levels,
                margin_rad=self.margin_rad,
            )
        sigma = float(head.conductivity_prior.get("brain_S_per_m", 0.33))
        bem = _compat.charge_bem([mesh], [sigma], [0.0])
        self._cache[key] = bem
        return bem

    # -- solve --------------------------------------------------------------
    def solve(self, head: HeadModel, pose: Pose, coil: CoilSpec) -> Tensor:
        return self.solve_field(head, pose, coil).e

    def solve_field(self, head: HeadModel, pose: Pose, coil: CoilSpec) -> FieldSolve:
        _check_frames(head, pose, coil)
        if coil.geometry is None or coil.pulse is None:
            raise RuntimeError(
                "the charge-BEM backend drives agent G's efield_from_coil, which "
                "needs the CoilGeometry and TMSPulse it was validated against; "
                "this CoilSpec carries neither"
            )
        sphere = _spherical_head_for(head)
        centre = head.centre
        # agent G's solvers put the conductor at the origin
        shifted = pose.matrix.clone()
        shifted[:3, 3] = shifted[:3, 3] - centre
        bem = self._bem_for(head, pose, coil)
        try:
            dose = _compat.efield_from_coil(
                coil.geometry,
                coil.pulse,
                shifted,
                head.cortex_vertices - centre,
                head=sphere,
                solver="bem",
                bem=bem,
            )
        except Exception as exc:
            if _compat.impossible_geometry is not None and isinstance(
                exc, _compat.impossible_geometry
            ):
                raise _translate_refusal(exc, context=self.name) from exc
            raise
        validity = dict(dose.ledger.validity_domain)
        resolution = dict(validity.get("near_source_resolution", {}))
        return FieldSolve(
            e=torch.as_tensor(dose.value, dtype=_DT),
            # straight from the solver's ledger; nothing on this path is typed
            numerical_variance=float(dose.ledger.variance.get("numerical", 0.0)),
            relative_error_bound=resolution.get("relative_error_bound"),
            resolution=resolution,
            validity_domain={
                **validity,
                "gate_evidence": list(self.gate_evidence),
                "conductivity_enters_solution": True,
                "n_faces": int(bem.n_faces),
            },
        )


def resolve_efield_backend(explicit: Any = None) -> EFieldBackend:
    """Prefer the gated solver; fall back to the local closed form and say so.

    The charge BEM is **not** the default.  For a spherically symmetric head the
    closed form is exact and the BEM is a discretisation of it, so making the
    BEM the default would buy discretisation error and a dense solve for
    nothing.  The BEM's value is that it accepts an arbitrary surface, and the
    head model this release ships is a sphere.  Pass one explicitly
    (``TargetingService(efield_backend=ChargeBEMEField())``) when the geometry
    justifies it.
    """
    if explicit is not None:
        return explicit
    if _compat.analytic_sphere_efield is not None:
        return GatedAnalyticSphereEField()
    return AnalyticSphericalEField()


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

