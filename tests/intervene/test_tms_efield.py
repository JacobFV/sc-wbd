"""Field-physics validation: analytic reference, BEM convergence, invariants.

SIMULATION ONLY.  An unvalidated field solver is worse than none, because it
launders numerical error as biology.  Every claim here is checked against
either a closed form or an independent physical invariant:

* the far-coil limit against the elementary Faraday solution
  :math:`E=-\\tfrac12\\dot B\\times r`;
* the Heller--van Hulsteyn theorem :math:`\\hat r\\cdot E = 0` everywhere;
* independence of the interior field from the radial conductivity profile and
  from the outer radius;
* Faraday circulation: the total field minus the primary field is curl-free,
  so their circulations around any interior loop agree;
* mesh convergence of the charge BEM towards the analytic solution, with a
  measured order;
* the exact triangle panel integral against brute-force quadrature.
"""

from __future__ import annotations

import math

import pytest
import torch

from scwbd.intervene.tms.coil import MU0, CircularCoil, FigureEightCoil, biphasic
from scwbd.intervene.tms.efield import (
    MAX_PANEL_TO_STANDOFF,
    ChargeBEM,
    ImpossibleGeometry,
    bem_error_envelope,
    graded_icosphere,
    LayeredSphereBEM,
    SphericalHeadModel,
    TriMesh,
    analytic_sphere_efield,
    coil_dipoles_in_head_frame,
    efield_from_coil,
    icosphere,
    primary_efield_dipoles,
    primary_efield_segments,
    simnibs_available,
    simnibs_status,
    triangle_field_integral,
    uniform_dbdt_efield,
)

_DT = torch.float64
_SEED = 20240805


def _interior(n: int = 200, r: float = 0.05) -> torch.Tensor:
    g = torch.Generator().manual_seed(_SEED)
    p = torch.randn(n, 3, generator=g, dtype=_DT)
    return p / p.norm(dim=-1, keepdim=True) * r


# ---------------------------------------------------------------------------
# analytic reference
# ---------------------------------------------------------------------------


def test_radial_component_vanishes_everywhere_heller_van_hulsteyn():
    """`r.E = 0` inside a spherically symmetric conductor -- a theorem."""
    pts = _interior(300, 0.06)
    rc = torch.tensor([[0.02, -0.01, 0.11]], dtype=_DT)
    md = torch.tensor([[0.3, 0.9, 0.1]], dtype=_DT) * 1e6
    e = analytic_sphere_efield(pts, rc, md)
    radial = (e * pts).sum(-1).abs() / pts.norm(dim=-1)
    assert float(radial.max()) < 1e-12 * float(e.norm(dim=-1).max())


def test_far_coil_limit_converges_to_the_uniform_dbdt_solution():
    """As the coil recedes, the field must approach `E = -1/2 dB/dt x r`."""
    pts = _interior(100, 0.05)
    md = torch.tensor([[0.3, 0.7, 0.2]], dtype=_DT)
    md = md / md.norm()
    errs = []
    for Rc in (0.5, 2.0, 10.0):
        rc = torch.tensor([[0.0, 0.0, Rc]], dtype=_DT)
        e = analytic_sphere_efield(pts, rc, md)
        chat = rc[0] / Rc
        B = (MU0 / (4 * math.pi)) * (3 * (md[0] @ chat) * chat - md[0]) / Rc**3
        eu = uniform_dbdt_efield(pts, B)
        errs.append(float((e - eu).norm() / eu.norm()))
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 0.01
    # first-order convergence in 1/R_c
    assert 3.0 < errs[0] / errs[1] < 6.0


def test_faraday_circulation_total_minus_primary_is_curl_free():
    """The secondary field is a gradient, so circulations must agree exactly."""
    rc = torch.tensor([[0.02, 0.0, 0.11]], dtype=_DT)
    md = torch.tensor([[0.3, 0.9, 0.1]], dtype=_DT) * 1e6
    th = torch.linspace(0, 2 * math.pi, 4001, dtype=_DT)[:-1]
    centre = torch.tensor([0.005, -0.004, 0.02], dtype=_DT)
    e1 = torch.tensor([0.6, 0.4, -0.2], dtype=_DT)
    e1 = e1 / e1.norm()
    e2 = torch.cross(e1, torch.tensor([0.1, -0.5, 1.0], dtype=_DT), dim=0)
    e2 = e2 / e2.norm()
    e1 = torch.cross(e2, torch.cross(e1, e2, dim=0), dim=0)
    e1 = e1 / e1.norm()
    loop = centre + 0.03 * (torch.cos(th)[:, None] * e1 + torch.sin(th)[:, None] * e2)
    dl = torch.roll(loop, -1, 0) - loop
    mid = 0.5 * (loop + torch.roll(loop, -1, 0))
    assert float(loop.norm(dim=-1).max()) < 0.07  # stays inside the head

    circ_total = float((analytic_sphere_efield(mid, rc, md) * dl).sum())
    circ_primary = float((primary_efield_dipoles(mid, rc, md) * dl).sum())
    assert circ_total == pytest.approx(circ_primary, rel=1e-10)
    assert abs(circ_total) > 1e-3  # the check is not trivially about zero


def test_field_is_zero_at_the_sphere_centre():
    e = analytic_sphere_efield(
        torch.zeros(1, 3, dtype=_DT),
        torch.tensor([[0.0, 0.0, 0.1]], dtype=_DT),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=_DT) * 1e6,
    )
    assert float(e.norm()) < 1e-20


def test_superposition_over_dipoles():
    pts = _interior(50)
    rc = torch.tensor([[0.0, 0.0, 0.1], [0.03, 0.0, 0.1]], dtype=_DT)
    md = torch.tensor([[0.0, 1.0, 0.0], [0.5, 0.2, 0.0]], dtype=_DT) * 1e6
    both = analytic_sphere_efield(pts, rc, md)
    sep = analytic_sphere_efield(pts, rc[:1], md[:1]) + analytic_sphere_efield(
        pts, rc[1:], md[1:]
    )
    assert torch.allclose(both, sep, rtol=1e-12, atol=1e-20)


# ---------------------------------------------------------------------------
# panel integral
# ---------------------------------------------------------------------------


def test_analytic_triangle_panel_integral_matches_brute_force_quadrature():
    g = torch.Generator().manual_seed(3)
    tri = torch.randn(1, 3, 3, generator=g, dtype=_DT) * 0.1
    obs = torch.randn(6, 3, generator=g, dtype=_DT) * 0.3
    b1 = torch.rand(160000, 1, generator=g, dtype=_DT)
    b2 = torch.rand(160000, 1, generator=g, dtype=_DT)
    keep = (b1 + b2).squeeze(-1) <= 1
    b1, b2 = b1[keep], b2[keep]
    v0, v1, v2 = tri[0]
    pts = v0 + b1 * (v1 - v0) + b2 * (v2 - v0)
    area = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1).norm()
    w = area / pts.shape[0]
    d = obs[:, None, :] - pts[None]
    ref = (d / d.norm(dim=-1, keepdim=True) ** 3 * w).sum(1)
    got = triangle_field_integral(obs, tri)[:, 0, :]
    assert float((got - ref).norm() / ref.norm()) < 5e-3


def test_flat_panel_has_zero_normal_field_at_its_own_centroid():
    tri = torch.tensor([[[0.0, 0, 0], [0.01, 0, 0], [0.0, 0.02, 0]]], dtype=_DT)
    c = tri.mean(dim=1)
    g = triangle_field_integral(c, tri)[0, 0]
    assert abs(float(g[2])) < 1e-14  # normal is +z


# ---------------------------------------------------------------------------
# BEM
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_bem_converges_to_the_analytic_solution_with_measured_order():
    pts = _interior(120, 0.05)
    rc = torch.tensor([[0.0, 0.0, 0.11]], dtype=_DT)
    md = torch.tensor([[0.0, 1.0, 0.0], ], dtype=_DT) * 1e6
    ref = analytic_sphere_efield(pts, rc, md)

    errs = []
    for k in (1, 2, 3, 4):
        vv, ff = icosphere(k)
        bem = ChargeBEM([TriMesh(vv * 0.085, ff)], [0.33], [0.0])
        e = bem.total_field(pts, rc, md)
        errs.append(float((e - ref).norm() / ref.norm()))

    assert errs[0] > errs[1] > errs[2] > errs[3], errs
    assert errs[-1] < 5e-3, errs
    # measured order between the two finest meshes (h halves each refinement);
    # the coarse meshes are pre-asymptotic and are only required to decrease
    order = math.log2(errs[-2] / errs[-1])
    assert order > 1.5, (errs, order)


def test_bem_reproduces_the_zero_radial_field_theorem():
    pts = _interior(80, 0.05)
    rc = torch.tensor([[0.0, 0.0, 0.11]], dtype=_DT)
    md = torch.tensor([[0.0, 1.0, 0.0]], dtype=_DT) * 1e6
    v, f = icosphere(3)
    bem = ChargeBEM([TriMesh(v * 0.085, f)], [0.33], [0.0])
    e = bem.total_field(pts, rc, md)
    radial = (e * pts).sum(-1).abs() / pts.norm(dim=-1)
    assert float(radial.max()) / float(e.norm(dim=-1).max()) < 0.02


def test_interior_field_is_independent_of_outer_radius():
    """A second Heller--van Hulsteyn consequence, checked numerically."""
    pts = _interior(80, 0.045)
    rc = torch.tensor([[0.0, 0.0, 0.13]], dtype=_DT)
    md = torch.tensor([[0.0, 1.0, 0.0]], dtype=_DT) * 1e6
    v, f = icosphere(3)
    fields = [
        ChargeBEM([TriMesh(v * R, f)], [0.33], [0.0]).total_field(pts, rc, md)
        for R in (0.085, 0.100)
    ]
    rel = float((fields[0] - fields[1]).norm() / fields[0].norm())
    assert rel < 0.02, rel


def test_interior_field_is_independent_of_the_conductivity_profile():
    """Three-layer head with a 40:1 skull contrast must give the same field."""
    pts = _interior(60, 0.045)
    rc = torch.tensor([[0.0, 0.0, 0.13]], dtype=_DT)
    md = torch.tensor([[0.0, 1.0, 0.0]], dtype=_DT) * 1e6
    ref = analytic_sphere_efield(pts, rc, md)
    bem = LayeredSphereBEM(SphericalHeadModel(), n_subdiv=2)
    e = bem.total_field(pts, rc, md)
    rel = float((e - ref).norm() / ref.norm())
    # coarse mesh + high skull contrast: the tolerance is honest, not tight
    assert rel < 0.12, rel


# ---------------------------------------------------------------------------
# coil-level checks
# ---------------------------------------------------------------------------


def test_two_winding_discretisations_agree_in_the_far_field():
    """Dipole sheet and current polyline are the same coil, so they must agree."""
    coil = CircularCoil()
    pos, mom = coil.dipole_elements()
    mid, dl = coil.segments()
    didt = 1e8
    obs = torch.tensor([[0.0, 0.0, 0.6], [0.3, 0.2, 0.5]], dtype=_DT)
    a = primary_efield_dipoles(obs, pos, mom * didt)
    b = primary_efield_segments(obs, mid, dl, didt)
    assert float((a - b).norm() / b.norm()) < 0.02


def test_field_from_coil_returns_a_physical_dose_of_plausible_magnitude(
    coil, pulse, pose, head, cortex
):
    pts, _ = cortex
    dose = efield_from_coil(coil, pulse, pose.matrix(), pts, head=head)
    assert dose.modality == "tms" and dose.quantity == "E_field"
    assert dose.units == "V/m"
    assert "SIMULATION ONLY" in dose.notice
    # a figure-8 at 1e8 A/s, 4 mm off a 85 mm scalp: order 100 V/m
    assert 30.0 < dose.peak() < 400.0
    assert dose.ledger.validity_domain["geometry"] == "spherically_symmetric"


@pytest.mark.slow
def test_bem_and_analytic_agree_on_the_full_coil(coil, pulse, pose, head):
    pts, _ = head.cortical_shell(162)
    a = efield_from_coil(coil, pulse, pose.matrix(), pts, head=head, solver="analytic")
    v, f = icosphere(3)
    bem = ChargeBEM([TriMesh(v * head.radius, f)], [0.33], [0.0])
    b = efield_from_coil(
        coil, pulse, pose.matrix(), pts, head=head, solver="bem", bem=bem
    )
    rel = float((a.value - b.value).norm() / a.value.norm())
    assert rel < 0.05, rel


def test_improper_pose_matrix_is_refused():
    coil = FigureEightCoil()
    T = torch.eye(4, dtype=_DT)
    T[0, 0] = -1.0  # reflection: handedness error
    with pytest.raises(ValueError, match="proper rotation"):
        coil_dipoles_in_head_frame(coil, T, 1e8)


# ---------------------------------------------------------------------------
# external solver honesty
# ---------------------------------------------------------------------------


def test_simnibs_status_is_honest_about_availability():
    s = simnibs_status()
    assert s["available"] is simnibs_available()
    assert s["equivalent_to_fem"] is False
    assert "aarch64" in str(s["platform_note"])


# ---------------------------------------------------------------------------
# impossible geometry: a refusal, never a number
# ---------------------------------------------------------------------------


def _pose_at_scalp_distance(head: SphericalHeadModel, distance_m: float):
    from scwbd.intervene.tms.pose import coil_pose_on_sphere

    return coil_pose_on_sphere(
        head, [-0.55, 0.68, 0.48], standoff_m=distance_m,
        handle_azimuth_rad=math.radians(45.0),
    )


def test_the_recorded_edge_case_now_refuses_instead_of_reporting_218_kv_per_m():
    """The historical failure, pinned.

    An edge-case probe of this module once returned ``peak |E| = 218681.8 V/m``
    at a scalp distance of ``-25.97 mm`` -- a coil 26 mm inside the head.  That
    is not a strong field, it is the pole of the interior solution's denominator.
    """
    head = SphericalHeadModel()
    coil, pulse = FigureEightCoil(), biphasic()
    pts, _ = head.cortical_shell(162)
    pose = _pose_at_scalp_distance(head, -0.02597)
    assert pose.scalp_distance(head) == pytest.approx(-0.02597, abs=1e-9)

    with pytest.raises(ImpossibleGeometry) as exc:
        efield_from_coil(coil, pulse, pose.matrix(), pts, head=head)
    assert exc.value.code == "R06"
    assert "inside" in str(exc.value)
    assert exc.value.remedy


@pytest.mark.parametrize("standoff_m", [-0.0001, -0.005, -0.02597, -0.05])
def test_every_negative_standoff_is_refused_not_extrapolated(standoff_m):
    head = SphericalHeadModel()
    coil, pulse = FigureEightCoil(), biphasic()
    pts, _ = head.cortical_shell(42)
    with pytest.raises(ImpossibleGeometry):
        efield_from_coil(
            coil, pulse, _pose_at_scalp_distance(head, standoff_m).matrix(),
            pts, head=head,
        )


@pytest.mark.slow
def test_a_valid_placement_still_returns_a_dose_of_the_recorded_magnitude():
    """Positive control: the guard refuses impossible geometry, not all geometry."""
    head = SphericalHeadModel()
    coil, pulse = FigureEightCoil(), biphasic()
    pts, _ = head.cortical_shell(2562)
    dose = efield_from_coil(
        coil, pulse, _pose_at_scalp_distance(head, 0.004).matrix(), pts, head=head
    )
    assert 100.0 < dose.peak() < 200.0  # recorded baseline 134.5 V/m


def test_analytic_solution_refuses_a_source_inside_the_field_point_shell():
    """No head model needed: it is the derivation's own precondition."""
    pts = _interior(20, 0.07)
    rc = torch.tensor([[0.0, 0.0, 0.05]], dtype=_DT)  # closer in than the points
    md = torch.tensor([[0.0, 1.0, 0.0]], dtype=_DT) * 1e6
    with pytest.raises(ImpossibleGeometry) as exc:
        analytic_sphere_efield(pts, rc, md)
    assert "no conductor boundary separates them" in str(exc.value)


def test_field_points_outside_the_scalp_are_refused():
    head = SphericalHeadModel()
    pts = _interior(20, head.radius + 0.01)
    rc = torch.tensor([[0.0, 0.0, 0.15]], dtype=_DT)
    md = torch.tensor([[0.0, 1.0, 0.0]], dtype=_DT) * 1e6
    with pytest.raises(ImpossibleGeometry):
        analytic_sphere_efield(pts, rc, md, head=head)


def test_the_refused_number_would_have_been_absurd():
    """Show the guard is load-bearing: without it the formula returns a pole."""
    head = SphericalHeadModel()
    coil, pulse = FigureEightCoil(), biphasic()
    pts, _ = head.cortical_shell(2562)
    pose = _pose_at_scalp_distance(head, -0.02597)
    pos, mdot = coil_dipoles_in_head_frame(
        coil, pose.matrix(), float(pulse.peak_didt)
    )
    unguarded = analytic_sphere_efield(pts, pos, mdot, validate_geometry=False)
    peak = float(unguarded.norm(dim=-1).max())
    # the recorded probe saw 218681.8 V/m; the exact value depends on how close
    # a mesh vertex lands to the pole, which is the point -- it is unbounded
    assert peak > 1e3, peak  # vs a 134.5 V/m baseline: not a field, a pole


def test_containment_uses_the_surface_not_a_bounding_radius():
    v, f = icosphere(3)
    mesh = TriMesh(v * 0.085, f)
    inside = torch.tensor([[0.0, 0.0, 0.0], [0.02, -0.03, 0.01]], dtype=_DT)
    outside = torch.tensor([[0.0, 0.0, 0.12], [0.2, 0.0, 0.0]], dtype=_DT)
    assert bool(mesh.contains(inside).all())
    assert not bool(mesh.contains(outside).any())
    # the solid angle is the mechanism, and it is 4*pi / 0
    assert float(mesh.solid_angle(inside[:1]).abs()) == pytest.approx(
        4 * math.pi, rel=1e-3
    )
    assert abs(float(mesh.solid_angle(outside[1:]))) < 1e-6


def test_bem_refuses_sources_inside_the_conductor():
    v, f = icosphere(2)
    bem = ChargeBEM([TriMesh(v * 0.085, f)], [0.33], [0.0])
    bem.assert_sources_outside(torch.tensor([[0.0, 0.0, 0.11]], dtype=_DT))  # fine
    with pytest.raises(ImpossibleGeometry) as exc:
        bem.assert_sources_outside(torch.tensor([[0.0, 0.0, 0.01]], dtype=_DT))
    assert exc.value.code == "R06"


# ---------------------------------------------------------------------------
# contact geometry: graded meshing and the resolution guard (gate N8)
# ---------------------------------------------------------------------------


def _contact_source():
    p = torch.tensor([[0.03, -0.04, 0.0715]], dtype=_DT)
    return p / p.norm() * 0.089  # 4 mm off an 85 mm scalp


def test_graded_icosphere_is_a_closed_surface_refined_only_where_asked():
    head = SphericalHeadModel()
    u = _contact_source()[0] / _contact_source()[0].norm()
    coarse = graded_icosphere(head.radius, 3, u, 0)
    graded = graded_icosphere(head.radius, 3, u, 2)
    assert graded.n_faces > coarse.n_faces
    # still closed: enclosed volume within the flat-panel error of the sphere
    exact = 4 / 3 * math.pi * head.radius**3
    assert abs(graded.enclosed_volume() - exact) / exact < 0.01
    # refinement is local: panels far from u keep the base size
    cen = graded.centroids()
    cen = cen / cen.norm(dim=-1, keepdim=True)
    _, area = graded.normals_areas()
    near = (cen @ u) > math.cos(0.2)
    far = (cen @ u) < math.cos(1.2)
    assert float(area[near].mean()) < 0.1 * float(area[far].mean())


def test_the_resolution_guard_refuses_a_mesh_that_cannot_resolve_the_source():
    head = SphericalHeadModel()
    pos = _contact_source()
    v, f = icosphere(3)
    coarse = ChargeBEM([TriMesh(v * head.radius, f)], [0.33], [0.0])
    res = coarse.near_source_resolution(pos)
    assert res["panel_to_standoff"] > MAX_PANEL_TO_STANDOFF
    with pytest.raises(ImpossibleGeometry, match="does not resolve"):
        coarse.assert_resolves_sources(pos)

    u = pos[0] / pos[0].norm()
    graded = ChargeBEM([graded_icosphere(head.radius, 4, u, 2)], [0.33], [0.0])
    ok = graded.assert_resolves_sources(pos)
    assert ok["panel_to_standoff"] < 0.3


def test_the_error_envelope_is_a_measured_step_not_a_guess():
    assert bem_error_envelope(0.2) == 0.01
    assert bem_error_envelope(0.5) == 0.02
    assert bem_error_envelope(0.95) == 0.04
    assert math.isnan(bem_error_envelope(1.5))  # outside: no bound is claimed
    # monotone in the resolution ratio
    vals = [bem_error_envelope(r) for r in (0.1, 0.4, 0.8)]
    assert vals == sorted(vals)


@pytest.mark.slow
def test_the_bem_ledger_carries_the_resolution_that_governs_its_error():
    """A ledger saying only 'charge_bem_5120_faces' cannot tell validated from not."""
    head = SphericalHeadModel()
    coil, pulse = FigureEightCoil(), biphasic()
    pts, _ = head.cortical_shell(162)
    pose = _pose_at_scalp_distance(head, 0.004)
    v, f = icosphere(4)
    bem = ChargeBEM([TriMesh(v * head.radius, f)], [0.33], [0.0])
    dose = efield_from_coil(
        coil, pulse, pose.matrix(), pts, head=head, solver="bem", bem=bem
    )
    vd = dose.ledger.validity_domain
    assert "near_source_resolution" in vd
    assert vd["near_source_resolution"]["panel_to_standoff"] <= MAX_PANEL_TO_STANDOFF
    assert vd["near_source_resolution"]["relative_error_bound"] > 0.0
    # the declared numerical variance follows the measured envelope, not a constant
    assert dose.ledger.variance["numerical"] > 0.0
