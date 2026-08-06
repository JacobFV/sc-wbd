"""The independent spectral reference for induction (gate N6).

SIMULATION ONLY.

A reference is only worth as much as its own validation. This one is checked
three ways, each against something it does not contain:

* against the **closed form** it is supposed to reproduce
  (``tms.efield.analytic_sphere_efield``, Sarvas / Heller--van Hulsteyn) --
  which it reaches to ~1e-8, from completely different mathematics;
* against the **theorem** ``r . E = 0`` everywhere inside a spherically
  symmetric conductor, which no step of the spectral construction imposes;
* against the **elementary Faraday solution** ``E = -(1/2) Bdot x r`` in the
  far-source limit, which involves no spheres, no harmonics and no conductor.

Plus the machinery underneath: the solid harmonics are asserted to be harmonic,
homogeneous and orthogonal on the sphere rather than assumed to be, because the
recursion is the one place a silent sign error could hide.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from scwbd.intervene.spectral_reference import (
    MU0,
    SphericalInductionReference,
    primary_induced_field,
    real_solid_basis,
    reference_degree_convergence,
    regular_solid_harmonics,
    sphere_quadrature,
    spectral_induced_efield,
)
from scwbd.intervene.tms.efield import (
    ImpossibleGeometry,
    analytic_sphere_efield,
    charge_bem_induced_efield,
    uniform_dbdt_efield,
)

_DT = torch.float64
_A = 0.085
_RC = torch.tensor([[0.0, 0.0, 0.11]], dtype=_DT)
_MD = torch.tensor([[0.0, 1.0, 0.0]], dtype=_DT) * 1e6


def _interior(n: int = 200, r: float = 0.070, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(n, 3, generator=g, dtype=_DT)
    return p / p.norm(dim=-1, keepdim=True) * r


# ---------------------------------------------------------------------------
# the machinery
# ---------------------------------------------------------------------------


def test_solid_harmonics_are_harmonic():
    """Laplacian zero, by finite differences -- the recursion is not assumed."""
    g = torch.Generator().manual_seed(0)
    p = torch.randn(120, 3, generator=g, dtype=_DT)
    degree, h = 6, 1e-4
    base_re, base_im = regular_solid_harmonics(p, degree)
    worst = 0.0
    for base in (base_re, base_im):
        lap = {k: torch.zeros_like(p[:, 0]) for k in base}
        for axis in range(3):
            e = torch.zeros(3, dtype=_DT)
            e[axis] = h
            plus = regular_solid_harmonics(p + e, degree)
            minus = regular_solid_harmonics(p - e, degree)
            which = 0 if base is base_re else 1
            for key in base:
                lap[key] = lap[key] + (
                    plus[which][key] - 2 * base[key] + minus[which][key]
                ) / h**2
        for key, val in base.items():
            if key[0] == 0:
                continue
            worst = max(worst, float(lap[key].abs().max() / (val.abs().max() + 1e-30)))
    assert worst < 1e-5, worst


def test_solid_harmonics_are_homogeneous_of_their_degree():
    g = torch.Generator().manual_seed(1)
    p = torch.randn(80, 3, generator=g, dtype=_DT)
    re1, _ = regular_solid_harmonics(p, 6)
    re2, _ = regular_solid_harmonics(p * 2.0, 6)
    for (l, m), v in re1.items():
        assert torch.allclose(re2[(l, m)], (2.0**l) * v, rtol=1e-12, atol=0.0)


def test_the_surface_basis_is_orthogonal_under_the_quadrature():
    """What makes the coefficient step a projection rather than a fit."""
    degree = 8
    normals, weights = sphere_quadrature(degree)
    basis, _ = real_solid_basis(normals, degree)
    basis = basis / torch.sqrt((weights[:, None] * basis * basis).sum(0))
    gram = (weights[:, None, None] * basis[:, :, None] * basis[:, None, :]).sum(0)
    off = gram - torch.eye(gram.shape[0], dtype=_DT)
    assert float(off.abs().max()) < 1e-10, float(off.abs().max())


# ---------------------------------------------------------------------------
# the reference, validated three independent ways
# ---------------------------------------------------------------------------


def test_it_converges_geometrically_to_the_closed_form():
    pts = _interior()
    closed = analytic_sphere_efield(pts, _RC, _MD)
    errs = []
    for degree in (16, 24, 32, 40):
        got = SphericalInductionReference(radius=_A, degree=degree).induced_field(
            pts, _RC, _MD
        )
        errs.append(float((got - closed).norm() / closed.norm()))
    assert errs == sorted(errs, reverse=True), errs
    assert errs[-1] < 1e-6, errs
    # geometric, not algebraic: each step of 8 degrees buys orders, not factors
    assert errs[0] / errs[-1] > 1e3, errs


def test_it_reproduces_the_zero_radial_field_theorem_it_never_imposes():
    pts = _interior()
    e = SphericalInductionReference(radius=_A, degree=40).induced_field(pts, _RC, _MD)
    radial = (e * pts).sum(-1).abs() / pts.norm(dim=-1)
    assert float(radial.max() / e.norm(dim=-1).max()) < 1e-6


def test_the_far_source_limit_is_the_elementary_faraday_solution():
    """No spheres, no harmonics: E = -(1/2) Bdot x r for a uniform Bdot."""
    far = torch.tensor([[0.0, 0.0, 50.0]], dtype=_DT)
    pts = _interior(60, 0.05)
    got = SphericalInductionReference(radius=_A, degree=8).induced_field(pts, far, _MD)
    # Bdot at the origin from a dipole on the z-axis with moment along y
    radius = float(far.norm())
    b_dot = -(MU0 / (4 * math.pi)) * _MD.reshape(3) / radius**3
    want = uniform_dbdt_efield(pts, b_dot)
    assert float((got - want).norm() / want.norm()) < 5e-3


def test_the_primary_field_is_the_textbook_dipole_vector_potential():
    """Independent of the solver's own implementation, so check it standalone."""
    r = torch.tensor([[0.0, 0.0, 0.05]], dtype=_DT)
    pos = torch.zeros(1, 3, dtype=_DT)
    mdot = torch.tensor([[0.0, 0.0, 1.0]], dtype=_DT)
    # m parallel to (r - r_c): the cross product vanishes
    assert float(primary_induced_field(r, pos, mdot).norm()) < 1e-30
    mdot = torch.tensor([[1.0, 0.0, 0.0]], dtype=_DT)
    got = primary_induced_field(r, pos, mdot)
    want = -(MU0 / (4 * math.pi)) * torch.cross(
        mdot.reshape(3), r.reshape(3), dim=0
    ) / float(r.norm()) ** 3
    assert torch.allclose(got.reshape(3), want, rtol=1e-12, atol=0.0)


def test_self_convergence_is_reported_without_any_closed_form():
    rows = reference_degree_convergence(
        _interior(60).numpy(),
        dipole_pos=((0.0, 0.0, 0.11),),
        dipole_mdot=((0.0, 1e6, 0.0),),
        sphere_radius=_A,
        degrees=(16, 24, 32),
    )
    errs = [r["relative_l2_vs_finest"] for r in rows]
    assert errs[0] > errs[1] > errs[2] == 0.0, errs


# ---------------------------------------------------------------------------
# validity domain, declared rather than implied
# ---------------------------------------------------------------------------


def test_the_series_rate_is_reported_so_the_validity_domain_is_visible():
    ref = SphericalInductionReference(radius=_A, degree=48)
    standoff = ref.convergence_ratio(_RC)
    contact = ref.convergence_ratio(torch.tensor([[0.0, 0.0, 0.089]], dtype=_DT))
    assert standoff == pytest.approx(_A / 0.11, rel=1e-12)
    assert contact > 0.95  # a coil in contact: this reference is not a yardstick
    # the rho^L estimate is a conservative a-priori BOUND: it drops the
    # prefactor and the extra decay each multipole picks up on the way in, so
    # the measured error must sit below it. A bound that the measurement beat
    # would be a broken bound; one the measurement exceeded would be worse.
    bound = ref.truncation_estimate(_RC)
    assert 1e-6 < bound < 1e-4, bound
    pts = _interior(60)
    measured = float(
        (ref.induced_field(pts, _RC, _MD) - analytic_sphere_efield(pts, _RC, _MD)).norm()
        / analytic_sphere_efield(pts, _RC, _MD).norm()
    )
    assert measured < bound, (measured, bound)
    assert ref.truncation_estimate(torch.tensor([[0.0, 0.0, 0.089]], dtype=_DT)) > 1e-2


def test_exterior_field_points_are_refused():
    ref = SphericalInductionReference(radius=_A, degree=8)
    with pytest.raises(ValueError, match="outside"):
        ref.induced_field(_interior(10, _A + 0.01), _RC, _MD)


# ---------------------------------------------------------------------------
# the N6 gate adapters
# ---------------------------------------------------------------------------


def test_solver_and_reference_come_from_different_modules():
    """N6 records this; it must be true for a structural reason, not by luck."""
    assert charge_bem_induced_efield.__module__ == "scwbd.intervene.tms.efield"
    assert spectral_induced_efield.__module__ == "scwbd.intervene.spectral_reference"
    assert charge_bem_induced_efield.__module__ != spectral_induced_efield.__module__


@pytest.mark.slow
def test_the_two_adapters_agree_at_the_gate_geometry():
    from scwbd.intervene.run_field_gates import (
        N6_DIPOLE_MDOT,
        N6_DIPOLE_POS,
        N6_SPHERE_RADIUS,
        n6_points,
    )

    pts = n6_points(120)
    kw = dict(dipole_pos=N6_DIPOLE_POS, dipole_mdot=N6_DIPOLE_MDOT,
              sphere_radius=N6_SPHERE_RADIUS)
    num = charge_bem_induced_efield(pts, **kw)
    ref = spectral_induced_efield(pts, **kw)
    rel = np.abs(num - ref) / np.abs(ref).mean()
    assert rel.mean() < 0.05, rel.mean()


def test_the_bem_adapter_refuses_a_source_inside_the_head():
    with pytest.raises(ImpossibleGeometry):
        charge_bem_induced_efield(
            _interior(10, 0.05).numpy(),
            dipole_pos=((0.0, 0.0, 0.02),),
            dipole_mdot=((0.0, 1e6, 0.0),),
            sphere_radius=_A,
        )
