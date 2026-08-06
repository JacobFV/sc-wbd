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
from typing import Sequence

import torch
from torch import Tensor

from ..base import SIMULATION_ONLY_NOTICE, Ledger, PhysicalDose
from .coil import MU0, CoilGeometry, TMSPulse

__all__ = [
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
    """
    r = r.to(_DT).reshape(-1, 3)
    pos = dipole_pos.to(_DT).reshape(-1, 3)
    mdot = dipole_mdot.to(_DT).reshape(-1, 3)

    out = torch.zeros_like(r)
    for s in range(0, pos.shape[0], chunk):
        rc = pos[s : s + chunk]  # [D,3]
        md = mdot[s : s + chunk]  # [D,3]
        Rc = rc.norm(dim=-1)  # [D]
        a_vec = rc[None, :, :] - r[:, None, :]  # [N,D,3]
        a = a_vec.norm(dim=-1).clamp_min(1e-15)  # [N,D]
        r_dot_rc = (r[:, None, :] * rc[None, :, :]).sum(-1)  # [N,D]
        a_dot_rc = (a_vec * rc[None, :, :]).sum(-1)  # [N,D]

        F = a * (Rc[None, :] * a + Rc[None, :] ** 2 - r_dot_rc)
        c1 = a**2 / Rc[None, :] + a_dot_rc / a + 2 * a + 2 * Rc[None, :]
        c2 = a + 2 * Rc[None, :] + a_dot_rc / a
        gradF = c1[..., None] * rc[None, :, :] - c2[..., None] * r[:, None, :]

        r_x_m = torch.cross(r[:, None, :].expand(-1, rc.shape[0], -1),
                            md[None, :, :].expand(r.shape[0], -1, -1), dim=-1)
        r_x_rc = torch.cross(r[:, None, :].expand(-1, rc.shape[0], -1),
                             rc[None, :, :].expand(r.shape[0], -1, -1), dim=-1)
        gF_m = (gradF * md[None, :, :]).sum(-1)  # [N,D]

        term = F[..., None] * r_x_m - gF_m[..., None] * r_x_rc
        out = out - (MU0 / (4 * math.pi)) * (term / (F**2).clamp_min(1e-300)[..., None]).sum(1)
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
    """
    didt = float(pulse.peak_didt) if t is None else float(pulse.didt(t))
    pos, mdot = coil_dipoles_in_head_frame(coil, T_head_from_coil, didt)
    if solver == "analytic":
        e = analytic_sphere_efield(points, pos, mdot)
        model = "analytic_spherical_sarvas_heller_van_hulsteyn"
        num_var = 0.0
    elif solver == "bem":
        if bem is None:
            if head is None:
                raise ValueError("bem solver needs a ChargeBEM or a SphericalHeadModel")
            bem = LayeredSphereBEM(head)
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
