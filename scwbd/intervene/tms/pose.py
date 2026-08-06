"""Coil pose: the full SE(3) chain, its covariance, and its effect on the field.

**SIMULATION ONLY** -- see :mod:`scwbd.intervene.base`.

Thesis Sec. 2.8: *a location is not a three-number property of a brain*.  A
coil pose is

.. math::

    \\mathcal C = (q, \\mathsf{frame}, \\mathsf{fiducials}, t, \\mathsf{units},
                   K, b_{\\mathcal C}, \\Sigma_{\\mathcal C}, \\mathsf{prov})

and the device-to-atlas chain of Eq. (3) is stored **factor by factor**, never
multiplied and forgotten:

.. math::

    T^{\\rm atlas\\leftarrow device}_{p,s} =
      T^{\\rm atlas\\leftarrow image}_{p}\\,
      T^{\\rm image\\leftarrow head}_{p,s}\\,
      T^{\\rm head\\leftarrow tracker}_{s}\\,
      T^{\\rm tracker\\leftarrow device}_{s}.

Two refusals live here:

* :func:`require_pose` rejects an underspecified target.  ``"5 cm anterior"``
  -- the classic DLPFC heuristic -- is **rejected** (``thesis_contract.tex``
  Sec. 0.5 step 2), as is a bare scalp label such as ``"F3"`` and a position
  without an orientation.
* :func:`compose_chain` rejects a broken transform lineage (refusal ``R01``).

And one measurement: :func:`pose_field_sensitivity` quantifies how much a
distance or angular placement error changes the *modelled* cortical field
(Caulfield et al. 2022 show this is material), and
:func:`propagate_pose_uncertainty` pushes the chain covariance through the
field solver.  Cross-covariance between chain factors is retained: shared
session calibration error does **not** average away.

This module carries a local SE(3) implementation as a thin adapter.  When
``scwbd.transforms`` (agent D) is importable, :func:`use_agent_d_chain`
converts to it; the local code is a stand-in, not a competing frame graph.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from ..base import SIMULATION_ONLY_NOTICE, InterventionRefusal, Ledger
from .coil import CoilGeometry, TMSPulse
from .efield import SphericalHeadModel, analytic_sphere_efield, coil_dipoles_in_head_frame

__all__ = [
    "UnderspecifiedPose",
    "FrameTransform",
    "PoseChain",
    "CoilPose",
    "require_pose",
    "compose_chain",
    "se3_exp",
    "se3_log",
    "se3_adjoint",
    "coil_pose_on_sphere",
    "standard_fiducials",
    "PoseSensitivityReport",
    "pose_field_sensitivity",
    "FieldUncertainty",
    "propagate_pose_uncertainty",
    "use_agent_d_chain",
]

_DT = torch.float64


class UnderspecifiedPose(InterventionRefusal):
    """The target is not a pose. Refusal ``R01`` (unknown frame / lineage)."""

    def __init__(self, message: str, *, offending_object: Any = None) -> None:
        super().__init__(
            "R01",
            message,
            remedy=(
                "supply position AND orientation in a named frame, with "
                "fiducials, units, session timestamp, covariance and the "
                "transform lineage back to the subject surface"
            ),
            offending_object=offending_object,
        )


# ---------------------------------------------------------------------------
# SE(3)
# ---------------------------------------------------------------------------


def _hat(w: Tensor) -> Tensor:
    return torch.stack(
        [
            torch.stack([torch.zeros_like(w[0]), -w[2], w[1]]),
            torch.stack([w[2], torch.zeros_like(w[0]), -w[0]]),
            torch.stack([-w[1], w[0], torch.zeros_like(w[0])]),
        ]
    )


def se3_exp(xi: Tensor) -> Tensor:
    """Exponential map. ``xi = [wx,wy,wz,vx,vy,vz]`` (rad, m) -> 4x4."""
    xi = xi.to(_DT).reshape(6)
    w, v = xi[:3], xi[3:]
    th = float(w.norm())
    W = _hat(w)
    I = torch.eye(3, dtype=_DT)
    if th < 1e-9:
        R = I + W
        V = I + 0.5 * W
    else:
        R = I + math.sin(th) / th * W + (1 - math.cos(th)) / th**2 * (W @ W)
        V = (
            I
            + (1 - math.cos(th)) / th**2 * W
            + (th - math.sin(th)) / th**3 * (W @ W)
        )
    T = torch.eye(4, dtype=_DT)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def se3_log(T: Tensor) -> Tensor:
    """Logarithm map, inverse of :func:`se3_exp`."""
    T = T.to(_DT)
    R, t = T[:3, :3], T[:3, 3]
    c = (torch.trace(R) - 1) / 2
    th = float(torch.acos(c.clamp(-1.0, 1.0)))
    if th < 1e-9:
        w = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
        V_inv = torch.eye(3, dtype=_DT) - 0.5 * _hat(w)
    else:
        w = (
            th
            / (2 * math.sin(th))
            * torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        )
        W = _hat(w)
        V_inv = (
            torch.eye(3, dtype=_DT)
            - 0.5 * W
            + (1 / th**2) * (1 - th * math.sin(th) / (2 * (1 - math.cos(th)))) * (W @ W)
        )
    return torch.cat([w, V_inv @ t])


def se3_adjoint(T: Tensor) -> Tensor:
    """``Ad_T`` in the ``[w; v]`` twist ordering. 6x6."""
    T = T.to(_DT)
    R, t = T[:3, :3], T[:3, 3]
    A = torch.zeros(6, 6, dtype=_DT)
    A[:3, :3] = R
    A[3:, :3] = _hat(t) @ R
    A[3:, 3:] = R
    return A


# ---------------------------------------------------------------------------
# chain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameTransform:
    """One factor of Eq. (3), with its own uncertainty and provenance."""

    dst: str
    src: str
    matrix: Tensor  # 4x4, dst <- src
    cov: Tensor = field(default_factory=lambda: torch.zeros(6, 6, dtype=_DT))
    bias_twist: Tensor = field(default_factory=lambda: torch.zeros(6, dtype=_DT))
    kind: str = "rigid"  # "rigid" | "nonrigid_warp"
    session: str = ""
    timestamp: float = 0.0
    units: str = "m"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    residual_rms: float | None = None  # e.g. fiducial coregistration residual

    def __post_init__(self) -> None:
        M = self.matrix.to(_DT)
        if M.shape != (4, 4):
            raise UnderspecifiedPose(f"{self.dst}<-{self.src}: matrix must be 4x4")
        if self.kind == "rigid" and abs(float(torch.det(M[:3, :3])) - 1.0) > 1e-6:
            raise UnderspecifiedPose(
                f"{self.dst}<-{self.src}: rotation block has det "
                f"{float(torch.det(M[:3, :3])):.6f}; handedness or scale is wrong"
            )
        if self.kind == "nonrigid_warp" and self.residual_rms is None:
            raise UnderspecifiedPose(
                f"{self.dst}<-{self.src}: a non-rigid warp must retain local "
                "Jacobians and registration residuals, not be disguised as "
                "rigid pose error (thesis Sec. 2.8)"
            )


@dataclass(frozen=True)
class PoseChain:
    """Ordered factors, stored separately. ``factors[0]`` is the outermost."""

    factors: tuple[FrameTransform, ...]
    cross_cov: Mapping[tuple[int, int], Tensor] = field(default_factory=dict)
    notice: str = SIMULATION_ONLY_NOTICE

    @property
    def dst(self) -> str:
        return self.factors[0].dst

    @property
    def src(self) -> str:
        return self.factors[-1].src

    def compose(self) -> FrameTransform:
        return compose_chain(self)


def compose_chain(chain: PoseChain) -> FrameTransform:
    """Compose Eq. (3), propagating covariance **with cross terms**.

    Left-perturbation convention :math:`T_i = \\exp(\\xi_i^\\wedge)\\bar T_i`
    gives :math:`\\xi \\approx \\sum_i \\mathrm{Ad}_{\\bar T_{1:i-1}}\\xi_i`, so

    .. math::

        \\Sigma = \\sum_{i,j} J_i \\Sigma_{ij} J_j^\\top,
        \\qquad J_i = \\mathrm{Ad}_{\\bar T_1\\cdots\\bar T_{i-1}}.

    Dropping the :math:`i\\neq j` blocks is a bug, not an optimisation
    (ARCHITECTURE.md Sec. 3, T5): session calibration errors shared between
    factors do not average away.
    """
    fs = chain.factors
    if not fs:
        raise UnderspecifiedPose("empty transform chain")
    for a, b in zip(fs[:-1], fs[1:]):
        if a.src != b.dst:
            raise UnderspecifiedPose(
                f"broken transform lineage: {a.dst}<-{a.src} cannot be composed "
                f"with {b.dst}<-{b.src}",
                offending_object=(a.src, b.dst),
            )
        if a.units != b.units:
            raise UnderspecifiedPose(
                f"unit mismatch in chain: {a.units!r} vs {b.units!r}"
            )

    T = torch.eye(4, dtype=_DT)
    J: list[Tensor] = []
    for f_ in fs:
        J.append(se3_adjoint(T))
        T = T @ f_.matrix.to(_DT)

    n = len(fs)
    Sigma = torch.zeros(6, 6, dtype=_DT)
    for i in range(n):
        Sigma = Sigma + J[i] @ fs[i].cov.to(_DT) @ J[i].T
    for (i, j), C in chain.cross_cov.items():
        if i == j:
            continue
        term = J[i] @ C.to(_DT) @ J[j].T
        Sigma = Sigma + term + term.T

    bias = torch.zeros(6, dtype=_DT)
    for i in range(n):
        bias = bias + J[i] @ fs[i].bias_twist.to(_DT)

    return FrameTransform(
        dst=fs[0].dst,
        src=fs[-1].src,
        matrix=T,
        cov=Sigma,
        bias_twist=bias,
        session=fs[0].session,
        timestamp=fs[0].timestamp,
        units=fs[0].units,
        provenance={
            "composed_from": [f"{f_.dst}<-{f_.src}" for f_ in fs],
            "cross_terms_retained": sorted(map(list, chain.cross_cov)),
            "note": "factors retained separately; this is a derived view",
        },
    )


# ---------------------------------------------------------------------------
# the pose object and the refusal
# ---------------------------------------------------------------------------

_REQUIRED_FIDUCIALS = ("nasion", "lpa", "rpa")


@dataclass(frozen=True)
class CoilPose:
    """A complete coil pose. Anything less is refused before it reaches physics."""

    chain: PoseChain  # head <- ... <- coil; what the field solver needs
    fiducials: Mapping[str, Tensor]
    atlas_chain: PoseChain | None = None  # atlas <- image <- head, if recorded
    units: str = "m"
    timestamp: float = 0.0
    session: str = ""
    subject: str = ""
    target_label: str = ""  # descriptive only; never load-bearing
    provenance: Mapping[str, Any] = field(default_factory=dict)
    notice: str = SIMULATION_ONLY_NOTICE

    def __post_init__(self) -> None:
        missing = [f for f in _REQUIRED_FIDUCIALS if f not in self.fiducials]
        if missing:
            raise UnderspecifiedPose(
                f"pose is missing fiducials {missing}; pitch and yaw are "
                "meaningful only relative to a stated coil, tracker, head, or "
                "nasion--inion--preauricular frame (thesis Sec. 2.8)",
                offending_object=self.target_label,
            )
        self.chain.compose()  # validates lineage/units eagerly

    # -- derived quantities --------------------------------------------------

    def matrix(self) -> Tensor:
        """``T_head_from_coil`` (4x4). Derived; the factors remain the record."""
        return self.chain.compose().matrix

    def full_chain(self) -> PoseChain:
        """Eq. (3) end to end: ``T_atlas<-image<-head<-tracker<-coil``.

        Refuses when the atlas leg was never recorded: an atlas coordinate for
        a coil pose is not available by assumption.
        """
        if self.atlas_chain is None:
            raise UnderspecifiedPose(
                "no atlas leg recorded for this pose; a device pose does not "
                "become an atlas coordinate without an explicit head model and "
                "atlas warp (thesis Sec. 2.8)",
                offending_object=self.target_label,
            )
        return PoseChain(
            factors=self.atlas_chain.factors + self.chain.factors,
            cross_cov=dict(self.atlas_chain.cross_cov),
        )

    def covariance(self) -> Tensor:
        return self.chain.compose().cov

    def bias_twist(self) -> Tensor:
        return self.chain.compose().bias_twist

    def coil_centre(self) -> Tensor:
        return self.matrix()[:3, 3]

    def coil_axis(self) -> Tensor:
        """Coil-frame ``+z`` in head coordinates (points away from the head)."""
        return self.matrix()[:3, 2]

    def handle_direction(self) -> Tensor:
        """Coil-frame ``+y`` in head coordinates: the induced-field direction."""
        return self.matrix()[:3, 1]

    def scalp_distance(self, head: SphericalHeadModel) -> float:
        return float(self.coil_centre().norm()) - head.radius

    def orientation_to_normal(self, normal: Tensor) -> dict[str, float]:
        """Angles of the coil axis and handle relative to a cortical normal.

        Returned in **degrees**, with the tangential component too: thesis
        Sec. 7.2 requires orientation relative to the local cortical normal
        *and tangents*, not a scalp label.
        """
        n = normal.to(_DT) / normal.to(_DT).norm()
        ax = self.coil_axis() / self.coil_axis().norm()
        hd = self.handle_direction() / self.handle_direction().norm()
        hd_tan = hd - (hd @ n) * n
        return {
            "axis_to_normal_deg": math.degrees(math.acos(float((ax @ n).clamp(-1, 1)))),
            "handle_to_normal_deg": math.degrees(math.acos(float((hd @ n).clamp(-1, 1)))),
            "handle_tangential_fraction": float(hd_tan.norm()),
        }


def require_pose(spec: Any) -> CoilPose:
    """Accept a :class:`CoilPose`; refuse anything underspecified.

    Refused, with the reason named:

    * ``"5 cm anterior"`` / ``"5cm rule"`` -- a displacement with no frame, no
      origin and no orientation (``thesis_contract.tex`` Sec. 0.5 step 2);
    * ``"F3"`` / ``"left DLPFC"`` -- a scalp or region **label**, which becomes
      a location only through an individual surface and a device transform;
    * a 3-vector or ``{"x":..,"y":..,"z":..}`` -- a position with no
      orientation and no frame;
    * a dict missing ``chain``/``fiducials``.
    """
    if isinstance(spec, CoilPose):
        return spec

    if isinstance(spec, str):
        s = spec.strip().lower()
        if re.search(r"\d+\s*(mm|cm|m)\b", s) and not re.search(r"frame|chain", s):
            raise UnderspecifiedPose(
                f"target {spec!r} is a displacement, not a pose: it names no "
                "reference frame, no origin, no orientation and no session. "
                "The 5 cm rule is exactly the specification this compiler "
                "rejects (thesis_contract.tex Sec. 0.5 step 2).",
                offending_object=spec,
            )
        raise UnderspecifiedPose(
            f"target {spec!r} is a label, not a pose. A scalp or region label "
            "acquires a location only through an individual cortical surface, "
            "a digitised head, and a device-to-head transform "
            "(thesis Sec. 2.8, 7.2).",
            offending_object=spec,
        )

    if isinstance(spec, (list, tuple)) or (
        isinstance(spec, Tensor) and spec.numel() in (3, 4)
    ):
        raise UnderspecifiedPose(
            "a position vector is not a pose: orientation relative to the "
            "local cortical normal and tangents is required, and so is the "
            "frame it is expressed in.",
            offending_object=spec,
        )

    if isinstance(spec, Mapping):
        missing = [k for k in ("chain", "fiducials") if k not in spec]
        if missing:
            raise UnderspecifiedPose(
                f"pose specification is missing {missing}",
                offending_object=dict(spec),
            )
        return CoilPose(**dict(spec))

    raise UnderspecifiedPose(
        f"cannot interpret {type(spec).__name__} as a coil pose",
        offending_object=spec,
    )


def standard_fiducials(head: SphericalHeadModel) -> dict[str, Tensor]:
    """Nasion / LPA / RPA on the spherical head, in the head frame (RAS-like)."""
    R = head.radius
    return {
        "nasion": torch.tensor([0.0, R, 0.0], dtype=_DT),
        "lpa": torch.tensor([-R, 0.0, 0.0], dtype=_DT),
        "rpa": torch.tensor([R, 0.0, 0.0], dtype=_DT),
        "inion": torch.tensor([0.0, -R, 0.0], dtype=_DT),
    }


def coil_pose_on_sphere(
    head: SphericalHeadModel,
    direction: Sequence[float] | Tensor,
    *,
    standoff_m: float = 0.005,
    handle_azimuth_rad: float = 0.0,
    tracker_cov: Tensor | None = None,
    coregistration_cov: Tensor | None = None,
    shared_session_cov: Tensor | None = None,
    bias_twist: Tensor | None = None,
    session: str = "sim_session",
    subject: str = "simulated_general_adult",
    target_label: str = "",
    with_atlas_leg: bool = True,
) -> CoilPose:
    """Place a coil tangentially on the scalp along ``direction``.

    This is a **construction helper for simulation**, not a targeting
    procedure.  It builds a genuine two-factor chain
    ``head <- tracker <- coil`` (plus an optional ``atlas <- image <- head``
    leg) so the resulting object is a real pose with lineage and covariance,
    and so tests exercise the same code path a recorded pose would.

    ``handle_azimuth_rad`` rotates the coil about its own axis, i.e. it sets
    the induced-current direction in the local tangent plane.  ``direction``
    is the outward radial direction of the scalp contact point.
    """
    u = torch.as_tensor(direction, dtype=_DT).reshape(3)
    u = u / u.norm()
    # tangent basis at the contact point
    ref = torch.tensor([0.0, 0.0, 1.0], dtype=_DT)
    if float((ref @ u).abs()) > 0.95:
        ref = torch.tensor([0.0, 1.0, 0.0], dtype=_DT)
    e1 = ref - (ref @ u) * u
    e1 = e1 / e1.norm()
    e2 = torch.cross(u, e1, dim=0)
    ca, sa = math.cos(handle_azimuth_rad), math.sin(handle_azimuth_rad)
    y_axis = ca * e1 + sa * e2  # handle / induced-field direction
    x_axis = torch.cross(y_axis, u, dim=0)
    R = torch.stack([x_axis, y_axis, u], dim=1)  # columns are coil x,y,z in head
    t = (head.radius + standoff_m) * u

    T_head_tracker = torch.eye(4, dtype=_DT)
    T_head_tracker[:3, :3] = R
    T_head_tracker[:3, 3] = t
    T_tracker_coil = torch.eye(4, dtype=_DT)  # coil rigidly held in the tracker frame

    dflt_track = torch.diag(
        torch.tensor([2e-6, 2e-6, 2e-6, 4e-7, 4e-7, 4e-7], dtype=_DT)
    )  # ~0.08 deg, ~0.6 mm 1-sigma
    dflt_coreg = torch.diag(
        torch.tensor([3e-5, 3e-5, 3e-5, 4e-6, 4e-6, 4e-6], dtype=_DT)
    )  # ~0.3 deg, ~2 mm 1-sigma

    f_head_tracker = FrameTransform(
        dst="head",
        src="tracker",
        matrix=T_head_tracker,
        cov=dflt_coreg if coregistration_cov is None else coregistration_cov.to(_DT),
        bias_twist=torch.zeros(6, dtype=_DT) if bias_twist is None else bias_twist.to(_DT),
        session=session,
        provenance={"source": "fiducial coregistration + scalp digitisation"},
        residual_rms=0.002,
    )
    f_tracker_coil = FrameTransform(
        dst="tracker",
        src="coil",
        matrix=T_tracker_coil,
        cov=dflt_track if tracker_cov is None else tracker_cov.to(_DT),
        session=session,
        provenance={"source": "optical tracker calibration"},
    )
    cross = {}
    if shared_session_cov is not None:
        cross[(0, 1)] = shared_session_cov.to(_DT)
    chain = PoseChain(factors=(f_head_tracker, f_tracker_coil), cross_cov=cross)

    atlas_chain = None
    if with_atlas_leg:
        atlas_chain = PoseChain(
            factors=(
                FrameTransform(
                    dst="atlas",
                    src="image",
                    matrix=torch.eye(4, dtype=_DT),
                    cov=torch.diag(torch.tensor([1e-4] * 3 + [1e-5] * 3, dtype=_DT)),
                    kind="nonrigid_warp",
                    residual_rms=0.004,
                    session=session,
                    provenance={"source": "image-to-atlas normalisation"},
                ),
                FrameTransform(
                    dst="image",
                    src="head",
                    matrix=torch.eye(4, dtype=_DT),
                    cov=torch.diag(torch.tensor([1e-5] * 3 + [2e-6] * 3, dtype=_DT)),
                    session=session,
                    provenance={"source": "head-to-MRI coregistration"},
                ),
            )
        )

    return CoilPose(
        chain=chain,
        fiducials=standard_fiducials(head),
        atlas_chain=atlas_chain,
        session=session,
        subject=subject,
        target_label=target_label,
        provenance={
            "constructed_by": "coil_pose_on_sphere",
            "standoff_m": standoff_m,
            "handle_azimuth_rad": handle_azimuth_rad,
            "scope": "simulation_only",
        },
    )


# ---------------------------------------------------------------------------
# pose -> field sensitivity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityRow:
    """One perturbation and what it did to the *modelled* field."""

    name: str
    magnitude: float
    units: str
    peak_pct: float
    target_pct: float
    peak_shift_mm: float
    target_direction_change_deg: float = 0.0


@dataclass(frozen=True)
class PoseSensitivityReport:
    """Measured change in the *modelled* field under placement perturbation.

    Every number is a property of the field model, not of a brain: a 10 %
    change in modelled ``|E|`` is a 10 % change in a simulation.  The
    perturbation set covers the physically distinct axes, because for a
    figure-8 they are not interchangeable -- a tilt about the wing axis barely
    moves the field while a tilt about the handle axis lifts one wing off the
    scalp.
    """

    baseline_peak_v_per_m: float
    baseline_target_v_per_m: float
    rows: tuple[SensitivityRow, ...]
    citation: str = "Caulfield et al. 2022, Brain Stimul 15:1192-1205"
    notice: str = SIMULATION_ONLY_NOTICE

    def by_name(self, name: str) -> SensitivityRow:
        for r in self.rows:
            if r.name == name:
                return r
        raise KeyError(name)

    def slope(self, prefix: str, field_: str = "target_pct") -> float:
        """Least-squares slope of ``field_`` against magnitude for one axis."""
        sel = [r for r in self.rows if r.name.startswith(prefix)]
        if not sel:
            raise KeyError(prefix)
        x = torch.tensor([r.magnitude for r in sel], dtype=_DT)
        y = torch.tensor([getattr(r, field_) for r in sel], dtype=_DT)
        return float((x * y).sum() / (x * x).sum())

    def as_table(self) -> str:
        out = [
            "pose perturbation -> modelled field change (SIMULATION ONLY)",
            f"  baseline peak {self.baseline_peak_v_per_m:.1f} V/m, "
            f"target {self.baseline_target_v_per_m:.1f} V/m",
        ]
        for r in self.rows:
            out.append(
                f"  {r.name:<22s} {r.magnitude:5.1f} {r.units:<3s} : "
                f"peak {r.peak_pct:+7.2f}%  target {r.target_pct:+7.2f}%  "
                f"locus {r.peak_shift_mm:5.2f} mm  "
                f"E-dir {r.target_direction_change_deg:6.2f} deg"
            )
        return "\n".join(out)


def _field_at(coil: CoilGeometry, T: Tensor, didt: float, points: Tensor) -> Tensor:
    pos, mdot = coil_dipoles_in_head_frame(coil, T, didt)
    return analytic_sphere_efield(points, pos, mdot)


#: name -> unit twist (rad or m per unit magnitude), in the **coil** frame
_PERTURBATIONS: dict[str, tuple[Tensor, str, float]] = {
    "axial_translation": (torch.tensor([0.0, 0, 0, 0, 0, 1.0], dtype=_DT), "mm", 1e-3),
    "lateral_handle": (torch.tensor([0.0, 0, 0, 0, 1.0, 0], dtype=_DT), "mm", 1e-3),
    "lateral_wing": (torch.tensor([0.0, 0, 0, 1.0, 0, 0], dtype=_DT), "mm", 1e-3),
    "tilt_about_wing_axis": (torch.tensor([1.0, 0, 0, 0, 0, 0], dtype=_DT), "deg", math.pi / 180),
    "tilt_about_handle_axis": (torch.tensor([0.0, 1.0, 0, 0, 0, 0], dtype=_DT), "deg", math.pi / 180),
    "rotation_about_coil_axis": (torch.tensor([0.0, 0, 1.0, 0, 0, 0], dtype=_DT), "deg", math.pi / 180),
}


def pose_field_sensitivity(
    coil: CoilGeometry,
    pulse: TMSPulse,
    pose: CoilPose,
    points: Tensor,
    *,
    target_index: int = 0,
    distance_mm: Sequence[float] = (1.0, 2.0, 5.0),
    angle_deg: Sequence[float] = (1.0, 5.0, 10.0),
    perturbations: Sequence[str] | None = None,
) -> PoseSensitivityReport:
    """Quantify distance and angular sensitivity of the modelled field.

    Perturbations are applied in the **coil frame** (``T0 @ exp(xi)``), which
    is the frame a neuronavigation system reports placement error in.
    """
    T0 = pose.matrix()
    didt = float(pulse.peak_didt)
    E0 = _field_at(coil, T0, didt, points)
    mag0 = E0.norm(dim=-1)
    e0_t = E0[target_index] / E0[target_index].norm().clamp_min(1e-30)
    peak0 = float(mag0.max())
    tgt0 = float(mag0[target_index])
    loc0 = points[int(mag0.argmax())]

    names = list(perturbations or _PERTURBATIONS)
    rows: list[SensitivityRow] = []
    for name in names:
        unit_twist, units, scale = _PERTURBATIONS[name]
        mags = distance_mm if units == "mm" else angle_deg
        for m_ in mags:
            xi = unit_twist.to(_DT) * (m_ * scale)
            E = _field_at(coil, T0 @ se3_exp(xi), didt, points)
            mag = E.norm(dim=-1)
            e_t = E[target_index] / E[target_index].norm().clamp_min(1e-30)
            rows.append(
                SensitivityRow(
                    name=name,
                    magnitude=float(m_),
                    units=units,
                    peak_pct=100.0 * (float(mag.max()) - peak0) / peak0,
                    target_pct=100.0 * (float(mag[target_index]) - tgt0) / tgt0,
                    peak_shift_mm=float((points[int(mag.argmax())] - loc0).norm()) * 1e3,
                    target_direction_change_deg=math.degrees(
                        math.acos(float((e0_t @ e_t).clamp(-1.0, 1.0)))
                    ),
                )
            )

    return PoseSensitivityReport(
        baseline_peak_v_per_m=peak0,
        baseline_target_v_per_m=tgt0,
        rows=tuple(rows),
    )


@dataclass(frozen=True)
class FieldUncertainty:
    """Pose covariance pushed through the field solver."""

    mean_v_per_m: Tensor
    sd_v_per_m: Tensor
    cv: Tensor
    target_mean: float
    target_sd: float
    target_cv: float
    n_samples: int
    seed: int
    bias_shift_v_per_m: float
    ledger: Ledger
    notice: str = SIMULATION_ONLY_NOTICE


def propagate_pose_uncertainty(
    coil: CoilGeometry,
    pulse: TMSPulse,
    pose: CoilPose,
    points: Tensor,
    *,
    n_samples: int = 128,
    seed: int = 0,
    target_index: int = 0,
) -> FieldUncertainty:
    """Monte-Carlo the composed pose covariance through the field solver.

    The systematic twist bias is propagated **separately** from the covariance
    (thesis Sec. 2.8: "systematic offsets are propagated as a separate twist
    bias"), so bias and variance never collapse into one number.
    """
    T0 = pose.matrix()
    Sigma = pose.covariance()
    bias = pose.bias_twist()
    didt = float(pulse.peak_didt)

    L = torch.linalg.cholesky(Sigma + 1e-18 * torch.eye(6, dtype=_DT))
    g = torch.Generator().manual_seed(int(seed))
    mags = []
    for _ in range(int(n_samples)):
        xi = L @ torch.randn(6, generator=g, dtype=_DT)
        T = T0 @ se3_exp(xi)
        mags.append(_field_at(coil, T, didt, points).norm(dim=-1))
    M = torch.stack(mags)
    mean, sd = M.mean(0), M.std(0)

    E_nom = _field_at(coil, T0, didt, points).norm(dim=-1)
    E_bias = _field_at(coil, T0 @ se3_exp(bias), didt, points).norm(dim=-1)
    bias_shift = float((E_bias - E_nom)[target_index])

    return FieldUncertainty(
        mean_v_per_m=mean,
        sd_v_per_m=sd,
        cv=sd / mean.clamp_min(1e-30),
        target_mean=float(mean[target_index]),
        target_sd=float(sd[target_index]),
        target_cv=float(sd[target_index] / mean[target_index].clamp_min(1e-30)),
        n_samples=int(n_samples),
        seed=int(seed),
        bias_shift_v_per_m=bias_shift,
        ledger=Ledger(
            variance={"pose": float(sd[target_index] ** 2)},
            bias_interval=(min(0.0, bias_shift), max(0.0, bias_shift)),
            bias_status="design_estimable",
            validity_domain={
                "solver": "analytic_spherical",
                "n_samples": int(n_samples),
                "note": "pose bias propagated separately from pose covariance",
            },
        ),
    )


def use_agent_d_chain(chain: PoseChain) -> Any:
    """Convert to ``scwbd.transforms`` when agent D's frame graph is available."""
    try:  # pragma: no cover - depends on agent D landing
        from scwbd.transforms import FrameGraph  # type: ignore
    except Exception:
        return chain
    g = FrameGraph()
    for f_ in chain.factors:
        g.add(f_.dst, f_.src, f_.matrix, cov=f_.cov, provenance=dict(f_.provenance))
    return g
