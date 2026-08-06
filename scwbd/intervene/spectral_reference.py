"""Independent spectral reference for the magnetically induced E-field (gate N6).

**SIMULATION ONLY** -- see :mod:`scwbd.intervene.base`.

Gate N6 requires the induced-field solver to be checked against the Sarvas /
Heller--van Hulsteyn closed form, and records whether the reference comes from
the same module as the solver -- because a solver checked against its own
module's closed form is a weaker test than one checked against an independent
implementation.

This module is that independent implementation.  It shares no code path, and no
*derivation*, with :func:`scwbd.intervene.tms.efield.analytic_sphere_efield`:

============  ==========================================================
solver        surface-charge BEM, ``tms.efield.ChargeBEM``
in-module     closed-form rational algebra in :math:`F` and
closed form   :math:`\\nabla F` (Sarvas 1987; Heller & van Hulsteyn 1992)
**this**      **spectral solution of the Neumann problem** -- multipole
              expansion in regular solid harmonics, coefficients obtained
              by quadrature on the sphere, gradient by automatic
              differentiation
============  ==========================================================

The physics.  Quasi-statically :math:`E = -\\partial_t A - \\nabla V`.  Inside a
homogeneous conducting sphere of radius :math:`a` bounded by an insulator there
are no sources, so :math:`V` is harmonic, and the boundary condition is that no
current crosses the surface, :math:`\\hat r\\cdot E|_a = 0`.  Writing the primary
field :math:`E_p = -\\partial_t A` (free space, magnetic dipoles) this is a pure
interior **Neumann problem**

.. math::

    \\nabla^2 V = 0 \\quad (r<a), \\qquad
    \\left.\\frac{\\partial V}{\\partial r}\\right|_{r=a} = \\hat r\\cdot E_p(a\\hat n).

Expanding :math:`V=\\sum_{l\\ge1,m} c_{lm} R_l^m` in regular solid harmonics and
using Euler's relation :math:`\\partial_r R_l^m = (l/r)R_l^m` (they are
homogeneous of degree :math:`l`) makes the boundary condition diagonal, so the
coefficients follow from one spherical-harmonic transform.  No linear system is
solved and no closed form is invoked.

Two properties make this trustworthy rather than merely different:

* the solid harmonics come from the standard Cartesian recursion, and being
  polynomials their gradient is exact under autograd -- so no hand-derived
  angular-derivative formula can be silently wrong.  ``tests/intervene``
  asserts they are harmonic and homogeneous;
* the series converges **geometrically** like :math:`(a/R_c)^L` in the
  truncation degree, so its own error is measurable and reportable rather than
  assumed.  At :math:`R_c=0.11` m, :math:`a=0.085` m it reaches ``9.7e-9``
  against the closed form by degree 48 -- five orders below the BEM error it is
  used to measure.

The geometric rate is also the honest limit of this reference: a coil element
sitting a few millimetres off the scalp has :math:`a/R_c \\to 1` and would need
a prohibitive degree.  :meth:`SphericalInductionReference.convergence_ratio`
reports that ratio so the validity domain is self-declaring rather than
implicit.  Near-surface sources are the BEM's job; this is the yardstick.

References
----------
Sarvas J (1987) Phys Med Biol 32:11-22.
Heller L, van Hulsteyn DB (1992) Biophys J 63:129-138.
Greengard L, Rokhlin V (1997) Acta Numerica 6:229-269 (solid-harmonic recursions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .base import SIMULATION_ONLY_NOTICE

__all__ = [
    "MU0",
    "regular_solid_harmonics",
    "real_solid_basis",
    "sphere_quadrature",
    "primary_induced_field",
    "SphericalInductionReference",
    "spectral_induced_efield",
    "reference_degree_convergence",
]

_DT = torch.float64

#: vacuum permeability, restated here so this module depends on nothing in the
#: solver it is used to check
MU0 = 4e-7 * math.pi


# ---------------------------------------------------------------------------
# solid harmonics
# ---------------------------------------------------------------------------


def regular_solid_harmonics(
    xyz: Tensor, degree: int
) -> tuple[dict[tuple[int, int], Tensor], dict[tuple[int, int], Tensor]]:
    """Regular solid harmonics :math:`R_l^m`, real and imaginary parts.

    The standard Cartesian recursion,

    .. math::

        R_0^0 &= 1, \\qquad
        R_l^l = -\\frac{x+iy}{2l}\\,R_{l-1}^{l-1},\\\\
        R_l^m &= \\frac{(2l-1)\\,z\\,R_{l-1}^m - r^2\\,R_{l-2}^m}{(l+m)(l-m)},
        \\qquad 0\\le m<l.

    Only harmonicity, homogeneity and completeness matter downstream -- the
    normalisation cancels in the projection -- and all three are asserted in
    ``tests/intervene/test_spectral_reference.py`` rather than assumed.

    Returns ``({(l,m): Re}, {(l,m): Im})`` for ``0 <= m <= l <= degree``.
    """
    xyz = xyz.to(_DT).reshape(-1, 3)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r2 = (xyz * xyz).sum(-1)
    re: dict[tuple[int, int], Tensor] = {(0, 0): torch.ones_like(x)}
    im: dict[tuple[int, int], Tensor] = {(0, 0): torch.zeros_like(x)}
    for l in range(1, int(degree) + 1):
        cp, sp = re[(l - 1, l - 1)], im[(l - 1, l - 1)]
        re[(l, l)] = -(x * cp - y * sp) / (2 * l)
        im[(l, l)] = -(x * sp + y * cp) / (2 * l)
        for m in range(0, l):
            a = (2 * l - 1) * z * re[(l - 1, m)]
            b = (2 * l - 1) * z * im[(l - 1, m)]
            if (l - 2, m) in re:
                a = a - r2 * re[(l - 2, m)]
                b = b - r2 * im[(l - 2, m)]
            den = float((l + m) * (l - m))
            re[(l, m)] = a / den
            im[(l, m)] = b / den
    return re, im


def real_solid_basis(xyz: Tensor, degree: int) -> tuple[Tensor, Tensor]:
    """Real basis ``[N, J]`` for degrees ``1..degree``, with the degree of each column.

    ``l = 0`` is dropped: a constant potential has no gradient and the Neumann
    problem does not determine it.
    """
    re, im = regular_solid_harmonics(xyz, degree)
    cols: list[Tensor] = []
    degs: list[int] = []
    for l in range(1, int(degree) + 1):
        for m in range(0, l + 1):
            cols.append(re[(l, m)])
            degs.append(l)
            if m > 0:
                cols.append(im[(l, m)])
                degs.append(l)
    return torch.stack(cols, dim=1), torch.tensor(degs, dtype=_DT)


def sphere_quadrature(degree: int) -> tuple[Tensor, Tensor]:
    """Gauss-Legendre in :math:`\\cos\\theta` times uniform in :math:`\\varphi`.

    Exact for spherical harmonics up to degree ``2*degree``, which is what makes
    the projection a projection rather than a fit.
    """
    n_theta = int(degree) + 2
    n_phi = 2 * int(degree) + 4
    u_np, w_np = np.polynomial.legendre.leggauss(n_theta)
    u = torch.as_tensor(u_np, dtype=_DT)
    w = torch.as_tensor(w_np, dtype=_DT)
    phi = torch.arange(n_phi, dtype=_DT) * (2 * math.pi / n_phi)
    sin_t = torch.sqrt((1 - u**2).clamp_min(0.0))
    ones = torch.ones_like(phi)
    normals = torch.stack(
        [
            (sin_t[:, None] * torch.cos(phi)[None, :]).reshape(-1),
            (sin_t[:, None] * torch.sin(phi)[None, :]).reshape(-1),
            (u[:, None] * ones[None, :]).reshape(-1),
        ],
        dim=1,
    )
    weights = (w[:, None] * (ones * (2 * math.pi / n_phi))[None, :]).reshape(-1)
    return normals, weights


# ---------------------------------------------------------------------------
# the reference
# ---------------------------------------------------------------------------


def primary_induced_field(
    points: Tensor, dipole_pos: Tensor, dipole_mdot: Tensor
) -> Tensor:
    """:math:`E_p = -\\partial_t A` in free space for magnetic dipoles.

    Independent implementation of the textbook vector potential
    :math:`A = (\\mu_0/4\\pi)\\,m\\times(r-r_c)/|r-r_c|^3`; this module does not
    import the solver's version of it.
    """
    r = points.to(_DT).reshape(-1, 3)
    pos = dipole_pos.to(_DT).reshape(-1, 3)
    mdot = dipole_mdot.to(_DT).reshape(-1, 3)
    d = r[:, None, :] - pos[None, :, :]
    dist = d.norm(dim=-1, keepdim=True).clamp_min(1e-14)
    cross = torch.cross(mdot[None, :, :].expand_as(d), d, dim=-1)
    return -(MU0 / (4 * math.pi)) * (cross / dist**3).sum(dim=1)


@dataclass
class SphericalInductionReference:
    """Spectral Sarvas / Heller--van Hulsteyn reference for a sphere of radius ``a``.

    ``degree`` is the multipole truncation.  Cost is
    :math:`O(L^2)` basis functions on an :math:`O(L^2)` quadrature grid; the
    error falls like :math:`(a/R_c)^L`, so :meth:`convergence_ratio` is the
    quantity that decides whether a given ``degree`` is enough.
    """

    radius: float = 0.085
    degree: int = 48
    notice: str = SIMULATION_ONLY_NOTICE

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError("degree must be at least 1")
        normals, weights = sphere_quadrature(self.degree)
        basis, degs = real_solid_basis(normals, self.degree)
        # unit-normalise each column under the quadrature inner product: the
        # recursion's own scale underflows at high degree, and normalising here
        # keeps every later quantity O(1)
        norm = torch.sqrt((weights[:, None] * basis * basis).sum(0))
        self._normals = normals
        self._weights = weights
        self._colnorm = norm
        self._surface_basis = basis / norm
        self._degrees = degs

    # -- diagnostics ---------------------------------------------------------

    def convergence_ratio(self, dipole_pos: Tensor) -> float:
        """:math:`a/\\min_c|r_c|`: the geometric rate of the multipole series.

        The truncation error is :math:`O(\\rho^{L})` with :math:`\\rho` this
        ratio, so a value near 1 means this reference is *not* a yardstick for
        that geometry however large ``degree`` is.
        """
        pos = dipole_pos.to(_DT).reshape(-1, 3)
        return float(self.radius / pos.norm(dim=-1).min())

    def truncation_estimate(self, dipole_pos: Tensor) -> float:
        """Order-of-magnitude bound on this reference's own error."""
        return float(self.convergence_ratio(dipole_pos) ** self.degree)

    # -- the field -----------------------------------------------------------

    def potential_coefficients(
        self, dipole_pos: Tensor, dipole_mdot: Tensor
    ) -> Tensor:
        """Solve the Neumann problem: one spherical-harmonic transform.

        With :math:`V=\\sum_j d_j\\,\\hat R_j(r/a)` and :math:`\\hat R_j`
        homogeneous of degree :math:`l_j`, Euler gives
        :math:`\\partial_r V|_a = (1/a)\\sum_j l_j d_j\\hat R_j(\\hat n)`, so the
        boundary condition is diagonal in this basis and
        :math:`d_j = (a/l_j)\\oint \\hat R_j\\,(\\hat r\\cdot E_p)\\,\\mathrm d\\Omega`.
        """
        surface = self.radius * self._normals
        g = (self._normals * primary_induced_field(surface, dipole_pos, dipole_mdot)).sum(-1)
        proj = (self._weights[:, None] * self._surface_basis * g[:, None]).sum(0)
        return self.radius * proj / self._degrees

    def induced_field(
        self, points: Tensor, dipole_pos: Tensor, dipole_mdot: Tensor
    ) -> Tensor:
        """Total induced E-field ``[N,3]`` at interior ``points``, V/m."""
        pts = points.to(_DT).reshape(-1, 3)
        if float(pts.norm(dim=-1).max()) > self.radius:
            raise ValueError(
                "this reference is the interior solution; a field point at "
                f"{float(pts.norm(dim=-1).max()):.4f} m lies outside the "
                f"{self.radius:.4f} m sphere"
            )
        coeff = self.potential_coefficients(dipole_pos, dipole_mdot)
        xi = (pts / self.radius).detach().clone().requires_grad_(True)
        basis, _ = real_solid_basis(xi, self.degree)
        potential = ((basis / self._colnorm) * coeff[None, :]).sum()
        grad_v = torch.autograd.grad(potential, xi)[0] / self.radius
        return primary_induced_field(pts, dipole_pos, dipole_mdot) - grad_v


# ---------------------------------------------------------------------------
# gate adapter
# ---------------------------------------------------------------------------

#: default truncation.  At the N6 source standoff (a/R_c ~ 0.77) this reaches
#: ~1e-8 against the closed form -- five orders below the solver error it
#: measures, which is the condition for calling it a reference at all.
DEFAULT_DEGREE = 48


def spectral_induced_efield(
    points: np.ndarray,
    *,
    dipole_pos: Sequence[Sequence[float]],
    dipole_mdot: Sequence[Sequence[float]],
    sphere_radius: float = 0.085,
) -> np.ndarray:
    """Gate adapter: induced E-field ``[N,3]`` as a plain array.

    Signature matches
    :func:`scwbd.intervene.tms.efield.charge_bem_induced_efield` exactly, so
    ``validate_induced_efield_solver`` can pass one ``solver_kwargs`` to both
    and the two callables keep their own module provenance.
    """
    ref = SphericalInductionReference(radius=float(sphere_radius), degree=DEFAULT_DEGREE)
    field = ref.induced_field(
        torch.as_tensor(np.asarray(points, dtype=float), dtype=_DT),
        torch.as_tensor(np.asarray(dipole_pos, dtype=float), dtype=_DT),
        torch.as_tensor(np.asarray(dipole_mdot, dtype=float), dtype=_DT),
    )
    return field.detach().cpu().numpy()


def reference_degree_convergence(
    points: np.ndarray,
    *,
    dipole_pos: Sequence[Sequence[float]],
    dipole_mdot: Sequence[Sequence[float]],
    sphere_radius: float = 0.085,
    degrees: Sequence[int] = (16, 24, 32, 40, 48),
) -> list[dict[str, float]]:
    """Self-convergence of the reference, measured without any closed form.

    Each degree is compared against the finest one, so this says "the series has
    converged" using only the series.  Whether it converged to the *right* thing
    is a separate question, answered in ``tests/intervene`` against the closed
    form, against the :math:`\\hat r\\cdot E=0` theorem, and against the
    elementary uniform-:math:`\\dot B` Faraday limit.
    """
    pts = torch.as_tensor(np.asarray(points, dtype=float), dtype=_DT)
    pos = torch.as_tensor(np.asarray(dipole_pos, dtype=float), dtype=_DT)
    mdot = torch.as_tensor(np.asarray(dipole_mdot, dtype=float), dtype=_DT)
    order = sorted(int(d) for d in degrees)
    finest = SphericalInductionReference(radius=float(sphere_radius), degree=order[-1])
    target = finest.induced_field(pts, pos, mdot)
    rows: list[dict[str, float]] = []
    for deg in order[:-1]:
        ref = SphericalInductionReference(radius=float(sphere_radius), degree=deg)
        got = ref.induced_field(pts, pos, mdot)
        rows.append({
            "degree": float(deg),
            "relative_l2_vs_finest": float((got - target).norm() / target.norm()),
            "geometric_rate_estimate": ref.truncation_estimate(pos),
        })
    rows.append({
        "degree": float(order[-1]),
        "relative_l2_vs_finest": 0.0,
        "geometric_rate_estimate": finest.truncation_estimate(pos),
    })
    return rows
