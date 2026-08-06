"""Induced electric field: analytic spherical reference + charge-based BEM.

**SIMULATION ONLY** -- see :mod:`scwbd.intervene.base`.  This module computes a
*physical dose* (V/m).  It says nothing about neurons; converting a field into a
population effect requires an explicitly named candidate operator from
:mod:`scwbd.intervene.tms.response`.

Two solvers, and the relationship between them is the validation:

:func:`analytic_sphere_efield`
    Closed form for the total induced E-field inside a **spherically symmetric**
    conductor driven by magnetic dipoles.  Obtained from the Sarvas (1987)
    magnetic-field formula by quasi-static lead-field reciprocity, and
    equivalent to the Heller & van Hulsteyn (1992) result.  Two of its
    properties are theorems, not approximations, and are asserted in the tests:

    * :math:`\\hat r\\cdot E = 0` **everywhere** inside, not just on the
      boundary (Heller & van Hulsteyn 1992);
    * the interior field is **independent of the radial conductivity profile
      and of the outer radius** -- the formula below contains neither.

:class:`ChargeBEM`
    A surface-charge boundary element method for arbitrary nested closed
    triangulated surfaces, so realistic head geometry (agent C) can be used.
    It is validated *against* the analytic solution with a mesh-convergence
    study, and it reproduces both theorems above numerically.

SimNIBS is wrapped when importable (:func:`simnibs_available`); on this machine
it is not available from PyPI for aarch64, which :func:`simnibs_status` reports
honestly rather than silently falling back.

References
----------
Sarvas J (1987) Phys Med Biol 32:11-22.
Heller L, van Hulsteyn DB (1992) Biophys J 63:129-138.
Deng ZD, Lisanby SH, Peterchev AV (2013) Brain Stimul 6:1-13.
Makarov SN et al. (2018) IEEE Trans Biomed Eng 65:2467-2478 (charge BEM-FMM).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor

from ..base import SIMULATION_ONLY_NOTICE, InterventionRefusal, Ledger, PhysicalDose
from .coil import MU0, CoilGeometry, TMSPulse

__all__ = [
    "ImpossibleGeometry",
    "charge_bem_induced_efield",
    "assert_sources_exterior",
    "analytic_sphere_efield",
    "primary_efield_dipoles",
    "primary_efield_segments",
    "uniform_dbdt_efield",
    "triangle_field_integral",
    "SphericalHeadModel",
    "TriMesh",
    "icosphere",
    "ChargeBEM",
    "LayeredSphereBEM",
    "coil_dipoles_in_head_frame",
    "efield_from_coil",
    "simnibs_available",
    "simnibs_status",
    "SimNIBSFEM",
]

_DT = torch.float64

#: how far outside the scalp a coil element must sit before the coil is treated
#: as physically placed.  Not a safety margin -- a solid-object constraint.
_MIN_SCALP_CLEARANCE_M = 1e-4

#: target size of a (field point x dipole) working block, in elements
_MAX_PAIR_BLOCK = 16_000_000


class ImpossibleGeometry(InterventionRefusal):
    """The requested geometry is not a geometry the field equations describe.

    Refusal ``R06`` (result outside the stated validity domain).  Raised when a
    source element would have to be *inside* the conductor, or a field point
    outside it.

    This exists because the failure it prevents is not an inaccuracy, it is a
    fabrication.  The Sarvas / Heller--van Hulsteyn interior solution has
    :math:`F = a\\,(R_c a + R_c^2 - r\\cdot r_c)` in its denominator; ``F``
    passes through zero as a source crosses the field point's shell, and the
    formula then returns a large, smooth, entirely fictitious number.  An
    edge-case probe of this module reached ``peak |E| = 218681.8 V/m`` at a
    scalp distance of ``-25.97 mm`` -- a coil 26 mm *inside* the head.  Nothing
    about that number is a field; it is the pole of a rational function.
    Returning it would launder numerical error as biology, so it is refused.
    """

    def __init__(self, message: str, *, remedy: str = "", offending_object: Any = None):
        super().__init__("R06", message, remedy=remedy, offending_object=offending_object)


def assert_sources_exterior(
    points: Tensor,
    dipole_pos: Tensor,
    *,
    head: "SphericalHeadModel | None" = None,
    clearance_m: float = 0.0,
    device_origin: Tensor | None = None,
    context: str = "analytic_sphere_efield",
) -> None:
    """Refuse geometries the interior solution does not describe.

    Two conditions, both preconditions of the derivation rather than tolerances:

    1. **Every source is farther from the centre than every field point.**  The
       interior solution assumes a sphere of some radius :math:`a` with all
       field points inside and all sources outside; such an :math:`a` exists
       exactly when :math:`\\min_c|r_c| > \\max|r|`.  This check needs no head
       radius, so it protects :func:`analytic_sphere_efield` on its own.
    2. When a :class:`SphericalHeadModel` is supplied, **sources lie outside the
       scalp and field points inside it**.  A coil element inside the head is
       not a placement error to be extrapolated through; it is not a placement.

    ``device_origin`` is checked alongside the source elements.  It matters: a
    figure-eight coil has no dipole element at its own centre, so a shallow
    interpenetration can put the coil body inside the scalp while every
    *discretised* element is still outside.  The origin is where the scalp
    distance is measured, so that is the point that must clear.
    """
    r = points.to(_DT).reshape(-1, 3)
    rc = dipole_pos.to(_DT).reshape(-1, 3)
    if device_origin is not None:
        rc = torch.cat([rc, device_origin.to(_DT).reshape(-1, 3)])
    max_r = float(r.norm(dim=-1).max())
    min_rc = float(rc.norm(dim=-1).min())
    if not (min_rc > max_r):
        raise ImpossibleGeometry(
            f"{context}: the nearest source element is {min_rc * 1e3:.3f} mm from "
            f"the head centre but a field point is at {max_r * 1e3:.3f} mm, so no "
            "conductor boundary separates them. The interior solution is derived "
            "for sources strictly outside the sphere containing every field point; "
            "with the source inside, its denominator passes through zero and the "
            "returned magnitude is a pole, not a field.",
            remedy="place the source outside the conductor, or use a solver "
            "formulated for interior sources",
            offending_object={
                "min_source_radius_m": min_rc,
                "max_field_point_radius_m": max_r,
            },
        )
    if head is None:
        return
    clearance = max(float(clearance_m), 0.0)
    if min_rc < head.radius + clearance:
        raise ImpossibleGeometry(
            f"{context}: a source element sits {(min_rc - head.radius) * 1e3:.3f} mm "
            f"from the scalp of a {head.radius * 1e3:.1f} mm head -- i.e. inside it. "
            "A coil cannot occupy the same space as the head, so there is no field "
            "to compute and no extrapolation that would make one.",
            remedy="increase the coil standoff until the scalp distance is "
            "non-negative, or model the geometry that actually produced it",
            offending_object={"scalp_distance_m": min_rc - head.radius},
        )
    if max_r > head.radius:
        raise ImpossibleGeometry(
            f"{context}: a field point is {max_r * 1e3:.3f} mm from the head centre, "
            f"outside the {head.radius * 1e3:.1f} mm scalp. This solver returns the "
            "*interior* solution; outside the conductor it is not the field.",
            remedy="restrict the field points to the conductor interior",
            offending_object={"max_field_point_radius_m": max_r},
        )


# ---------------------------------------------------------------------------
# analytic solutions
# ---------------------------------------------------------------------------


def primary_efield_dipoles(
    r: Tensor, dipole_pos: Tensor, dipole_mdot: Tensor
) -> Tensor:
    """Free-space primary field :math:`-\\partial A/\\partial t` from dipoles.

    Parameters
    ----------
    r : ``[N,3]`` field points, metres.
    dipole_pos : ``[D,3]`` dipole positions, metres.
    dipole_mdot : ``[D,3]`` :math:`\\dot m`, A m^2 / s.

    Returns ``[N,3]`` V/m.
    """
    r = r.to(_DT).reshape(-1, 3)
    p = dipole_pos.to(_DT).reshape(-1, 3)
    m = dipole_mdot.to(_DT).reshape(-1, 3)
    d = r[:, None, :] - p[None, :, :]  # [N,D,3]
    dist = d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cross = torch.cross(m[None, :, :].expand_as(d), d, dim=-1)
    return -(MU0 / (4 * math.pi)) * (cross / dist**3).sum(dim=1)


def primary_efield_segments(r: Tensor, mid: Tensor, dl: Tensor, didt: float) -> Tensor:
    """Free-space primary field from a winding polyline carrying ``didt`` A/s."""
    r = r.to(_DT).reshape(-1, 3)
    mid = mid.to(_DT).reshape(-1, 3)
    dl = dl.to(_DT).reshape(-1, 3)
    d = r[:, None, :] - mid[None, :, :]
    dist = d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return -(MU0 / (4 * math.pi)) * didt * (dl[None, :, :] / dist).sum(dim=1)


def uniform_dbdt_efield(r: Tensor, dbdt: Tensor) -> Tensor:
    """Induced field in a sphere under a **uniform** :math:`\\dot B`.

    :math:`E = -\\tfrac12\\,\\dot B\\times r`, the elementary Faraday solution.
    Used as the far-coil limit check on :func:`analytic_sphere_efield`.
    """
    r = r.to(_DT).reshape(-1, 3)
    b = dbdt.to(_DT).reshape(3)
    return -0.5 * torch.cross(b.expand_as(r), r, dim=-1)


def analytic_sphere_efield(
    r: Tensor,
    dipole_pos: Tensor,
    dipole_mdot: Tensor,
    *,
    chunk: int = 4096,
    point_chunk: int | None = None,
    head: "SphericalHeadModel | None" = None,
    validate_geometry: bool = True,
) -> Tensor:
    """Total induced E-field inside a spherically symmetric conductor.

    The conductor is centred at the **origin**; ``dipole_pos`` must lie outside
    it and ``r`` inside.  Neither the radius nor the conductivity appears --
    that is the theorem, not an omission.

    For one dipole at :math:`r_c` with moment rate :math:`\\dot m`, with
    :math:`a = r_c - r`, :math:`a=|a|`, :math:`R_c=|r_c|`,

    .. math::

        F &= a\\,(R_c a + R_c^2 - r\\cdot r_c),\\\\
        \\nabla F &= \\left(\\frac{a^2}{R_c} + \\frac{a\\cdot r_c}{a}
              + 2a + 2R_c\\right) r_c
            - \\left(a + 2R_c + \\frac{a\\cdot r_c}{a}\\right) r,\\\\
        E(r) &= -\\frac{\\mu_0}{4\\pi F^2}
            \\Bigl[F\\,(r\\times\\dot m) - (\\nabla F\\cdot\\dot m)(r\\times r_c)\\Bigr].

    Both terms are perpendicular to :math:`r`, so :math:`\\hat r\\cdot E=0`
    identically -- the Heller--van Hulsteyn theorem falls out of the algebra.

    ``validate_geometry`` (default on) enforces the derivation's own
    precondition via :func:`assert_sources_exterior`; turning it off is not a
    supported way to obtain a number for an impossible placement.
    """
    r = r.to(_DT).reshape(-1, 3)
    pos = dipole_pos.to(_DT).reshape(-1, 3)
    mdot = dipole_mdot.to(_DT).reshape(-1, 3)
    if validate_geometry:
        assert_sources_exterior(r, pos, head=head)

    # The sum over dipoles is factorised so that no [N,D,3] tensor is ever
    # built.  Both cross products carry the same left operand ``r``, so
    #
    #   sum_d [ (r x m_d)/F - (gradF.m)_d (r x r_c,d)/F^2 ]
    #       = r x [ (1/F) @ m  -  W @ r_c ],   W = (gradF.m)/F^2,
    #
    # which is two [N,D] @ [D,3] matmuls.  Everything else is a [N,D] scalar
    # obtainable from the Gram matrix ``r @ r_c^T``, since
    # ``a.r_c = R_c^2 - r.r_c`` and ``a^2 = |r|^2 - 2 r.r_c + R_c^2``.
    # Same arithmetic, ~an order of magnitude less memory traffic; the Monte
    # Carlo in ``pose.propagate_pose_uncertainty`` is what needs it.
    r2 = (r * r).sum(-1)  # [N]
    out = torch.zeros_like(r)
    for s in range(0, pos.shape[0], chunk):
        rc = pos[s : s + chunk]  # [D,3]
        md = mdot[s : s + chunk]  # [D,3]
        Rc2 = (rc * rc).sum(-1)  # [D]
        Rc = Rc2.sqrt()  # [D]
        rc_dot_m = (rc * md).sum(-1)  # [D]
        # keep each [n,D] temporary near 128 MB; splitting more finely than that
        # costs more in per-op threading overhead than it saves in cache misses
        pc = point_chunk or max(1, _MAX_PAIR_BLOCK // max(1, rc.shape[0]))

        for t in range(0, r.shape[0], pc):
            rr = r[t : t + pc]  # [n,3]
            r_dot_rc = rr @ rc.T  # [n,D]
            a = (r2[t : t + pc, None] + Rc2[None, :] - 2 * r_dot_rc)
            a = a.clamp_min(1e-30).sqrt()  # [n,D]
            a_dot_rc = Rc2[None, :] - r_dot_rc  # [n,D]
            adr_over_a = a_dot_rc / a

            F = a * (Rc[None, :] * a + a_dot_rc)
            inv_F = 1.0 / torch.where(F.abs() < 1e-300, torch.full_like(F, 1e-300), F)
            c1 = a * a / Rc[None, :] + adr_over_a + 2 * a + 2 * Rc[None, :]
            c2 = a + 2 * Rc[None, :] + adr_over_a
            gF_m = c1 * rc_dot_m[None, :] - c2 * (rr @ md.T)  # [n,D]

            S = inv_F @ md - (gF_m * inv_F * inv_F) @ rc  # [n,3]
            out[t : t + pc] -= (MU0 / (4 * math.pi)) * torch.cross(rr, S, dim=-1)
    return out


@dataclass(frozen=True)
class SphericalHeadModel:
    """Spherically symmetric head. The reference geometry for field validation.

    ``conductivities`` and ``radii`` are recorded for provenance and for the
    BEM cross-check, but by the Heller--van Hulsteyn theorem the *analytic*
    interior field depends on neither.  Values default to the IT'IS / standard
    three-layer BEM head (brain, skull, scalp).
    """

    radius: float = 0.085  # m, outer (scalp)
    radii: tuple[float, ...] = (0.078, 0.082, 0.085)  # brain, skull, scalp
    conductivities: tuple[float, ...] = (0.33, 0.008, 0.43)  # S/m, inside each shell
    cortex_radius: float = 0.070  # m, radius of the modelled cortical shell
    citation: str = (
        "IT'IS Foundation tissue property database v4.1; "
        "Deng, Lisanby & Peterchev 2013 Brain Stimul 6:1-13"
    )
    notice: str = SIMULATION_ONLY_NOTICE

    def cortical_shell(self, n: int = 2562) -> tuple[Tensor, Tensor]:
        """Return ``(points, outward normals)`` on the modelled cortical shell.

        The normal is the local cortical normal used by the orientation-dependent
        response operators; on a sphere it is :math:`\\hat r`.
        """
        v, _ = icosphere(_subdiv_for(n))
        return v * self.cortex_radius, v.clone()


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriMesh:
    """Closed, outward-oriented triangulated surface."""

    vertices: Tensor  # [V,3]
    faces: Tensor  # [T,3] long

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    def tri(self) -> Tensor:
        return self.vertices[self.faces]  # [T,3,3]

    def centroids(self) -> Tensor:
        return self.tri().mean(dim=1)

    def normals_areas(self) -> tuple[Tensor, Tensor]:
        t = self.tri()
        n = torch.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=-1)
        area = 0.5 * n.norm(dim=-1)
        return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-300), area

    def scaled(self, s: float) -> "TriMesh":
        return TriMesh(self.vertices * s, self.faces)

    def enclosed_volume(self) -> float:
        t = self.tri()
        return float((torch.cross(t[:, 0], t[:, 1], dim=-1) * t[:, 2]).sum() / 6.0)

    def solid_angle(self, points: Tensor, *, chunk: int = 64) -> Tensor:
        """Solid angle subtended by this closed surface at ``points``.

        :math:`4\\pi` inside, :math:`0` outside, computed with the same exact
        panel integral the BEM matrix uses -- so "is the coil inside the head"
        is answered by the solver's own geometry kernel, not by a bounding
        sphere that would be wrong for a real head.
        """
        obs = points.to(_DT).reshape(-1, 3)
        nrm, _ = self.normals_areas()
        tri = self.tri()
        out = torch.zeros(obs.shape[0], dtype=_DT)
        for s in range(0, obs.shape[0], chunk):
            g = triangle_field_integral(obs[s : s + chunk], tri)  # [B,T,3]
            out[s : s + chunk] = -(g * nrm[None]).sum(-1).sum(-1)
        return out

    def bounding_sphere(self) -> tuple[Tensor, float]:
        """Centre and radius of a sphere that strictly contains this surface."""
        centre = self.vertices.mean(dim=0)
        return centre, float((self.vertices - centre).norm(dim=-1).max())

    def contains(self, points: Tensor, *, chunk: int = 256) -> Tensor:
        """Boolean mask: which ``points`` lie inside this closed surface.

        A point outside the bounding sphere is outside the surface, exactly and
        by definition, so only the remaining candidates pay for the panel
        integral.  For a coil sitting on a scalp that is every element, and the
        containment guard costs essentially nothing.
        """
        p = points.to(_DT).reshape(-1, 3)
        centre, radius = self.bounding_sphere()
        out = torch.zeros(p.shape[0], dtype=torch.bool)
        cand = (p - centre).norm(dim=-1) <= radius
        if bool(cand.any()):
            out[cand] = self.solid_angle(p[cand], chunk=chunk).abs() > 2 * math.pi
        return out


def triangle_field_integral(obs: Tensor, tri: Tensor) -> Tensor:
    """Exact :math:`\\int_T (r-r')/|r-r'|^3\\,\\mathrm dS'` for flat triangles.

    Closed form of Wilton et al. (1984) / Graglia (1993).  Returns ``[B,T,3]``
    for ``obs [B,3]`` and ``tri [T,3,3]``.  Using this instead of point
    quadrature is what lifts the BEM from first- to second-order convergence:
    with centroid collocation the near-neighbour panels dominate the error and
    a 3-point rule on them is not enough.

    At an observation point lying in the panel's plane the normal component is
    zero, which is exactly the principal value needed for the self term.
    """
    obs = obs.to(_DT).reshape(-1, 3)
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    nv = torch.cross(v1 - v0, v2 - v0, dim=-1)
    n = nv / nv.norm(dim=-1, keepdim=True).clamp_min(1e-300)
    h = ((obs[:, None, :] - v0[None, :, :]) * n[None]).sum(-1)  # [B,T]
    rho = obs[:, None, :] - h[..., None] * n[None]
    ah = h.abs()
    out = torch.zeros(obs.shape[0], tri.shape[0], 3, dtype=_DT, device=obs.device)
    for e in range(3):
        a = tri[:, e]
        b = tri[:, (e + 1) % 3]
        lv = b - a
        lhat = lv / lv.norm(dim=-1, keepdim=True).clamp_min(1e-300)
        mhat = torch.cross(lhat, n, dim=-1)  # outward in-plane edge normal
        da = a[None] - rho
        db = b[None] - rho
        t0 = (da * mhat[None]).sum(-1)
        sm = (da * lhat[None]).sum(-1)
        sp = (db * lhat[None]).sum(-1)
        R0sq = t0**2 + h**2
        Rp = (sp**2 + R0sq).sqrt()
        Rm = (sm**2 + R0sq).sqrt()
        num, den = Rp + sp, Rm + sm
        # numerically stable branch when the observation point is near the
        # backward extension of the edge line (num or den -> 0)
        f = torch.where(
            (num > 1e-30) & (den > 1e-30),
            torch.log(num.clamp_min(1e-300)) - torch.log(den.clamp_min(1e-300)),
            torch.log((Rm - sm).clamp_min(1e-300)) - torch.log((Rp - sp).clamp_min(1e-300)),
        )
        beta = torch.atan2(t0 * sp, R0sq + ah * Rp) - torch.atan2(t0 * sm, R0sq + ah * Rm)
        out = out + mhat[None] * f[..., None] + n[None] * torch.sign(h)[..., None] * beta[..., None]
    return out


def _subdiv_for(n_vertices: int) -> int:
    n, k = 12, 0
    while n < n_vertices and k < 6:
        k += 1
        n = 10 * 4**k + 2
    return k


def icosphere(n_subdiv: int = 3) -> tuple[Tensor, Tensor]:
    """Unit icosphere. Returns ``(vertices [V,3], faces [T,3])``, outward oriented."""
    phi = (1 + math.sqrt(5)) / 2
    verts = torch.tensor(
        [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ],
        dtype=_DT,
    )
    faces = torch.tensor(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=torch.long,
    )
    verts = verts / verts.norm(dim=-1, keepdim=True)
    for _ in range(int(n_subdiv)):
        cache: dict[tuple[int, int], int] = {}
        vlist = [verts]
        new_faces = []
        n_v = verts.shape[0]
        extra: list[Tensor] = []

        def mid(a: int, b: int) -> int:
            nonlocal n_v
            key = (min(a, b), max(a, b))
            if key in cache:
                return cache[key]
            va = verts[a] if a < verts.shape[0] else extra[a - verts.shape[0]]
            vb = verts[b] if b < verts.shape[0] else extra[b - verts.shape[0]]
            m = (va + vb)
            m = m / m.norm()
            extra.append(m)
            cache[key] = n_v
            n_v += 1
            return cache[key]

        for f in faces.tolist():
            a, b, c = f
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        verts = torch.cat(vlist + ([torch.stack(extra)] if extra else []))
        faces = torch.tensor(new_faces, dtype=torch.long)
    # ensure outward orientation
    m = TriMesh(verts, faces)
    if m.enclosed_volume() < 0:
        faces = faces[:, [0, 2, 1]]
    return verts, faces


# ---------------------------------------------------------------------------
# charge-based BEM
# ---------------------------------------------------------------------------


class ChargeBEM:
    """Surface-charge BEM for quasi-static induced fields in a piecewise-
    homogeneous conductor.

    Formulation (Makarov et al. 2018).  With scaled charge density
    :math:`\\tilde q = q/(2\\varepsilon_0)` on each surface and
    :math:`K=(\\sigma_{\\rm in}-\\sigma_{\\rm out})/(\\sigma_{\\rm in}+\\sigma_{\\rm out})`,

    .. math::

        \\tilde q_i = K_i\\Bigl[E_p(c_i)\\cdot n_i
            + \\sum_{j\\neq i}\\frac{A_j\\,\\tilde q_j\\,(c_i-c_j)\\cdot n_i}
              {2\\pi|c_i-c_j|^3}\\Bigr],

    solved densely in float64.  The self term vanishes in the principal-value
    sense for a flat element at its own centroid.  The system is deflated
    against the constant-charge null direction (which is exact for
    :math:`K=1`) by enforcing zero net induced charge.

    The interior field is then
    :math:`E = E_p + \\sum_j A_j\\tilde q_j (r-c_j)/(2\\pi|r-c_j|^3)`.
    """

    def __init__(
        self,
        surfaces: Sequence[TriMesh],
        sigma_in: Sequence[float],
        sigma_out: Sequence[float],
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        if not (len(surfaces) == len(sigma_in) == len(sigma_out)):
            raise ValueError("surfaces, sigma_in and sigma_out must align")
        self.surfaces = list(surfaces)
        self.device = torch.device(device)
        cs, ns, ar, tr, ks = [], [], [], [], []
        for m, si, so in zip(surfaces, sigma_in, sigma_out):
            n, a = m.normals_areas()
            cs.append(m.centroids())
            ns.append(n)
            ar.append(a)
            tr.append(m.tri())
            ks.append(torch.full((m.n_faces,), (si - so) / (si + so), dtype=_DT))
        self.c = torch.cat(cs).to(self.device)
        self.n = torch.cat(ns).to(self.device)
        self.a = torch.cat(ar).to(self.device)
        self.t = torch.cat(tr).to(self.device)  # [T,3,3]
        self.K = torch.cat(ks).to(self.device)
        self.surface_id = torch.cat(
            [torch.full((m.n_faces,), i, dtype=torch.long) for i, m in enumerate(surfaces)]
        ).to(self.device)
        self._M: Tensor | None = None

    @property
    def n_faces(self) -> int:
        return int(self.c.shape[0])

    @property
    def outer_surface(self) -> TriMesh:
        """The enclosing surface, by enclosed volume."""
        return max(self.surfaces, key=lambda m: m.enclosed_volume())

    def assert_sources_outside(
        self, dipole_pos: Tensor, *, context: str = "ChargeBEM"
    ) -> None:
        """Refuse sources that lie inside the conductor.

        Uses the exact solid angle of the enclosing surface, so it is correct
        for realistic head geometry and not just for spheres -- a bounding-radius
        test would reject legitimate placements over concave regions and accept
        illegitimate ones over convex ones.
        """
        pos = dipole_pos.to(_DT).reshape(-1, 3)
        inside = self.outer_surface.contains(pos)
        n_in = int(inside.sum())
        if n_in:
            raise ImpossibleGeometry(
                f"{context}: {n_in} of {pos.shape[0]} source elements lie inside the "
                "outermost conductor surface. The surface-charge formulation places "
                "all sources in the exterior; with a source inside there is no "
                "solution to report, only a divergent quadrature.",
                remedy="move the source outside the head surface",
                offending_object={"n_sources_inside": n_in},
            )

    def _matrix(self, chunk: int = 256) -> Tensor:
        if self._M is not None:
            return self._M
        T = self.n_faces
        M = torch.empty(T, T, dtype=_DT, device=self.device)
        for s in range(0, T, chunk):
            ci = self.c[s : s + chunk]
            ni = self.n[s : s + chunk]
            g = triangle_field_integral(ci, self.t)  # [B,T,3]
            M[s : s + chunk] = (g * ni[:, None, :]).sum(-1) / (2 * math.pi)
        idx = torch.arange(T, device=self.device)
        M[idx, idx] = 0.0  # flat-panel principal value at its own centroid
        # Curvature-consistency (row-sum) correction.  A closed surface at
        # uniform scaled charge 1 produces zero field inside and 2 outside, so
        # its principal-value normal field is exactly 1: every row must sum to
        # 1 over its OWN surface.  Flat panels miss the local curvature term
        # and lose that identity at O(h); restoring it exactly is what lifts
        # the scheme from first- to second-order convergence.  This is the
        # standard deflation/self-term correction of surface-integral BEM.
        same = self.surface_id[:, None] == self.surface_id[None, :]
        M[idx, idx] = 1.0 - (M * same).sum(dim=1)
        A = -self.K[:, None] * M
        A[idx, idx] += 1.0
        # deflation: enforce zero net induced charge (the exact null direction
        # of (I - D) at K = 1, i.e. an isolated conductor bounded by vacuum)
        A = A + (self.a[None, :] / self.a.sum())
        self._M = A
        return A

    def solve(self, e_primary_at_centroids: Tensor) -> Tensor:
        """Solve for the scaled surface charge given the primary field."""
        b = self.K * (e_primary_at_centroids.to(self.device) * self.n).sum(-1)
        return torch.linalg.solve(self._matrix(), b)

    def secondary_field(self, r: Tensor, q: Tensor, chunk: int = 256) -> Tensor:
        """Field of the solved surface charge at arbitrary points ``r``."""
        r = r.to(self.device).reshape(-1, 3)
        out = torch.zeros_like(r)
        for s in range(0, r.shape[0], chunk):
            g = triangle_field_integral(r[s : s + chunk], self.t)  # [B,T,3]
            out[s : s + chunk] = (g * q[None, :, None]).sum(1)
        return out / (2 * math.pi)

    def total_field(
        self,
        r: Tensor,
        dipole_pos: Tensor,
        dipole_mdot: Tensor,
    ) -> Tensor:
        """Total :math:`E` at ``r`` for a magnetic-dipole source distribution."""
        ep_c = primary_efield_dipoles(self.c, dipole_pos, dipole_mdot)
        q = self.solve(ep_c)
        ep_r = primary_efield_dipoles(r, dipole_pos, dipole_mdot)
        return ep_r + self.secondary_field(r, q)

    def boundary_current_residual(
        self, dipole_pos: Tensor, dipole_mdot: Tensor
    ) -> float:
        """max :math:`|E\\cdot\\hat n|` / max :math:`|E|` on the outermost surface.

        Zero normal current through the outer boundary is the physical boundary
        condition, so this is an *a posteriori* check on the solve.
        """
        ep_c = primary_efield_dipoles(self.c, dipole_pos, dipole_mdot)
        q = self.solve(ep_c)
        e = ep_c + self.secondary_field(self.c - 1e-6 * self.n, q)
        normal = (e * self.n).sum(-1).abs()
        return float(normal.max() / e.norm(dim=-1).max().clamp_min(1e-30))


class LayeredSphereBEM(ChargeBEM):
    """Convenience: nested spherical shells from a :class:`SphericalHeadModel`."""

    def __init__(
        self,
        head: SphericalHeadModel,
        *,
        n_subdiv: int = 3,
        device: str | torch.device = "cpu",
    ) -> None:
        v, f = icosphere(n_subdiv)
        meshes = [TriMesh(v * R, f) for R in head.radii]
        sig_in = list(head.conductivities)
        sig_out = list(head.conductivities[1:]) + [0.0]
        super().__init__(meshes, sig_in, sig_out, device=device)


# ---------------------------------------------------------------------------
# coil -> head-frame dipoles -> field
# ---------------------------------------------------------------------------


def coil_dipoles_in_head_frame(
    coil: CoilGeometry, T_head_from_coil: Tensor, didt: float
) -> tuple[Tensor, Tensor]:
    """Map a coil's dipole elements into the head frame at drive ``didt``.

    ``T_head_from_coil`` is a 4x4 SE(3) matrix.  Moments are rotated (they are
    axial vectors under proper rotations) and scaled by ``didt``; positions are
    fully transformed.  Returns ``(positions [D,3], mdot [D,3])``.
    """
    T = T_head_from_coil.to(_DT)
    R, t = T[:3, :3], T[:3, 3]
    if abs(float(torch.det(R)) - 1.0) > 1e-6:
        raise ValueError(
            "T_head_from_coil is not a proper rotation (det != 1): handedness "
            "or scale error in the pose chain (refusal R01 territory)"
        )
    pos, mom = coil.dipole_elements()
    return pos @ R.T + t, (mom @ R.T) * didt


def efield_from_coil(
    coil: CoilGeometry,
    pulse: TMSPulse,
    T_head_from_coil: Tensor,
    points: Tensor,
    *,
    head: SphericalHeadModel | None = None,
    solver: str = "analytic",
    bem: ChargeBEM | None = None,
    t: float | None = None,
) -> PhysicalDose:
    """Induced E-field at ``points`` as a :class:`PhysicalDose` (V/m).

    ``t`` defaults to the instant of peak :math:`\\mathrm dI/\\mathrm dt`.  The
    result is a *physical dose*: it carries units, support, and a ledger, and
    it deliberately cannot be turned into a neural effect without going through
    a named response operator.

    The geometry is checked before any physics runs
    (:func:`assert_sources_exterior`).  A coil that intersects the head is
    refused with :class:`ImpossibleGeometry`, not extrapolated.
    """
    didt = float(pulse.peak_didt) if t is None else float(pulse.didt(t))
    pos, mdot = coil_dipoles_in_head_frame(coil, T_head_from_coil, didt)
    if head is not None:
        assert_sources_exterior(
            points, pos, head=head, clearance_m=_MIN_SCALP_CLEARANCE_M,
            device_origin=T_head_from_coil.to(_DT)[:3, 3],
            context="efield_from_coil",
        )
    if solver == "analytic":
        e = analytic_sphere_efield(points, pos, mdot, head=head)
        model = "analytic_spherical_sarvas_heller_van_hulsteyn"
        num_var = 0.0
    elif solver == "bem":
        if bem is None:
            if head is None:
                raise ValueError("bem solver needs a ChargeBEM or a SphericalHeadModel")
            bem = LayeredSphereBEM(head)
        bem.assert_sources_outside(pos, context="efield_from_coil[bem]")
        e = bem.total_field(points, pos, mdot)
        model = f"charge_bem_{bem.n_faces}_faces"
        num_var = float((0.02 * e.norm(dim=-1).max()) ** 2)
    else:
        raise ValueError(f"unknown solver {solver!r}")
    return PhysicalDose(
        modality="tms",
        quantity="E_field",
        units="V/m",
        value=e,
        support=f"head_frame_points[{tuple(points.shape)}]/{model}",
        ledger=Ledger(
            variance={"numerical": num_var},
            bias_status="externally_bounded",
            validity_domain={
                "solver": model,
                "didt_A_per_s": didt,
                "geometry": "spherically_symmetric" if solver == "analytic" else "mesh",
            },
        ),
    )


# ---------------------------------------------------------------------------
# gate N6 adapter
# ---------------------------------------------------------------------------

#: icosphere subdivisions for the N6 solver adapter (4 -> 5120 panels)
DEFAULT_N6_SUBDIV = 4


def charge_bem_induced_efield(
    points,
    *,
    dipole_pos,
    dipole_mdot,
    sphere_radius: float = 0.085,
):
    """Gate adapter for N6: the induced E-field ``[N,3]`` from :class:`ChargeBEM`.

    Deliberately defined **here**, in the solver's own module, rather than in a
    runner: ``validate_induced_efield_solver`` records ``__module__`` for the
    solver and the reference and reports whether they coincide.  A wrapper
    written somewhere neutral would make an independent check look like a shared
    one, or the reverse.  The reference lives in
    :mod:`scwbd.intervene.spectral_reference` and shares nothing with this file.

    Single conducting shell with an insulating exterior, which is the geometry
    the closed form describes; by the Heller--van Hulsteyn theorem the interior
    field does not depend on the conductivity, so none is taken.
    """
    import numpy as _np

    pts = torch.as_tensor(_np.asarray(points, dtype=float), dtype=_DT)
    pos = torch.as_tensor(_np.asarray(dipole_pos, dtype=float), dtype=_DT)
    mdot = torch.as_tensor(_np.asarray(dipole_mdot, dtype=float), dtype=_DT)
    verts, faces = icosphere(DEFAULT_N6_SUBDIV)
    bem = ChargeBEM(
        [TriMesh(verts * float(sphere_radius), faces)], [0.33], [0.0]
    )
    bem.assert_sources_outside(pos, context="charge_bem_induced_efield")
    return bem.total_field(pts, pos, mdot).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# SimNIBS wrapper (honest about availability)
# ---------------------------------------------------------------------------


def simnibs_available() -> bool:
    try:  # pragma: no cover - depends on host
        import simnibs  # noqa: F401
    except Exception:
        return False
    return True


def simnibs_status() -> dict[str, object]:
    """Report SimNIBS availability without pretending a fallback is equivalent."""
    ok = simnibs_available()
    return {
        "available": ok,
        "platform_note": (
            "SimNIBS is not distributed on PyPI for linux-aarch64; "
            "`uv pip install simnibs` fails with 'not found in the package "
            "registry' on this host. The validated analytic spherical solution "
            "and the in-repo ChargeBEM are used instead, and the BEM is "
            "convergence-tested against the analytic reference."
        ),
        "fallback": "analytic_sphere_efield + ChargeBEM",
        "equivalent_to_fem": False,
        "notice": SIMULATION_ONLY_NOTICE,
    }


class SimNIBSFEM:
    """Thin wrapper over a SimNIBS FEM run, when SimNIBS is importable.

    Deliberately refuses rather than degrading silently: a realistic-head FEM
    result and a spherical analytic result are different objects with different
    validity domains, and conflating them would launder numerical error as
    anatomy.
    """

    def __init__(self, head_mesh_path: str) -> None:
        if not simnibs_available():
            raise RuntimeError(
                "SimNIBS is unavailable on this host. " + str(simnibs_status())
            )
        self.head_mesh_path = head_mesh_path

    def run(self, *args, **kwargs):  # pragma: no cover - requires SimNIBS
        from simnibs import sim_struct, run_simnibs  # type: ignore

        raise NotImplementedError(
            "SimNIBS session construction is intentionally not auto-generated: "
            "a head mesh, conductivity assignment and coil file must be "
            "declared explicitly with provenance."
        )
