"""The named fallback approximation, and the trap it sets (gate N9).

SIMULATION ONLY.

``primary_tangential_projection`` is an approximation, not a solution.  It is
named and implemented in ``scwbd.intervene.tms.efield`` so that it can be gated
rather than left as a comment in whoever's fallback path uses it.

The reason it needs a gate rather than a label is the shape of its error.  It is
**exact** for a circular coil -- to round-off, for a structural reason -- and
1.5x to 1.8x high for a figure-eight.  The error is a function of source
*symmetry*, not of any resolution parameter, so nothing converges to reveal it
and a validation suite that happened to use a circular coil would report a
perfect result.
"""

from __future__ import annotations


import pytest
import torch

from scwbd.intervene.spectral_reference import AxialInductionReference
from scwbd.intervene.tms.coil import CircularCoil, FigureEightCoil, biphasic
from scwbd.intervene.tms.efield import (
    SphericalHeadModel,
    coil_dipoles_in_head_frame,
    primary_tangential_projection,
)
from scwbd.intervene.tms.pose import coil_pose_on_sphere

_DT = torch.float64


def _setup(coil, radius: float = 0.085, standoff: float = 0.004, n: int = 642):
    head = SphericalHeadModel(radius=radius, cortex_radius=radius - 0.015)
    pose = coil_pose_on_sphere(head, [-0.55, 0.68, 0.48], standoff_m=standoff)
    pos, mdot = coil_dipoles_in_head_frame(
        coil, pose.matrix(), float(biphasic().peak_didt)
    )
    pts, normals = head.cortical_shell(n)
    return head, pts, normals, pos, mdot


def _reference(radius, pts, pos, mdot, degree: int = 250):
    return AxialInductionReference(radius=radius, degree=degree).induced_field(
        pts, pos, mdot
    )


def test_it_is_exact_for_an_axisymmetric_coil_and_this_is_the_trap():
    """A circular coil's primary vector potential is purely azimuthal.

    So ``r . E_p`` vanishes on the sphere, the Neumann data is identically zero,
    there is no secondary field at all, and the tangential projection is the
    exact answer. Anyone validating this approximation on a circular coil would
    conclude it is perfect.
    """
    head, pts, normals, pos, mdot = _setup(CircularCoil(n_azimuth=32, n_radial=4))
    ref = _reference(head.radius, pts, pos, mdot)
    approx = primary_tangential_projection(pts, normals, pos, mdot)
    assert float((approx - ref).norm() / ref.norm()) < 1e-9

    # and the reason: the radial component of the primary field is already zero
    from scwbd.intervene.tms.efield import primary_efield_dipoles

    e_p = primary_efield_dipoles(pts, pos, mdot)
    n = normals / normals.norm(dim=-1, keepdim=True)
    radial = (e_p * n).sum(-1).abs().max()
    assert float(radial / e_p.norm(dim=-1).max()) < 1e-9


def test_it_is_badly_wrong_for_a_figure_eight():
    """Opposed wings off the radial axis: the Neumann data does not vanish."""
    head, pts, normals, pos, mdot = _setup(FigureEightCoil(n_azimuth=32, n_radial=4))
    ref = _reference(head.radius, pts, pos, mdot)
    approx = primary_tangential_projection(pts, normals, pos, mdot)
    peak_ratio = float(approx.norm(dim=-1).max() / ref.norm(dim=-1).max())
    assert 1.4 < peak_ratio < 1.9, peak_ratio


def test_the_error_is_in_magnitude_not_direction():
    """Which matters: a direction-only consumer inherits a different bound."""
    head, pts, normals, pos, mdot = _setup(FigureEightCoil(n_azimuth=32, n_radial=4))
    ref = _reference(head.radius, pts, pos, mdot)
    approx = primary_tangential_projection(pts, normals, pos, mdot)
    peak = int(ref.norm(dim=-1).argmax())
    cosine = float(
        (approx[peak] @ ref[peak]) / (approx[peak].norm() * ref[peak].norm())
    )
    assert cosine > 0.999, cosine


def test_the_overestimate_grows_as_the_head_shrinks():
    """Characterised, not wild -- which is what makes a bound meaningful."""
    ratios = []
    for radius in (0.070, 0.085, 0.092):
        head, pts, normals, pos, mdot = _setup(
            FigureEightCoil(n_azimuth=32, n_radial=4), radius=radius, n=162
        )
        ref = _reference(head.radius, pts, pos, mdot, degree=200)
        approx = primary_tangential_projection(pts, normals, pos, mdot)
        ratios.append(float(approx.norm(dim=-1).max() / ref.norm(dim=-1).max()))
    assert ratios == sorted(ratios, reverse=True), ratios


def test_normals_are_what_tangential_is_measured_against():
    head, pts, normals, pos, mdot = _setup(
        FigureEightCoil(n_azimuth=32, n_radial=4), n=162
    )
    approx = primary_tangential_projection(pts, normals, pos, mdot)
    n = normals / normals.norm(dim=-1, keepdim=True)
    assert float((approx * n).sum(-1).abs().max()) < 1e-12


# ---------------------------------------------------------------------------
# the gate's subject must be the object actually in the runtime path
# ---------------------------------------------------------------------------


def test_the_runtime_fallback_computes_this_same_expression():
    """Otherwise gate N9 measures something the runtime does not use.

    The gate bounds ``primary_tangential_projection``; this is what pins that the
    runtime's fallback backend *is* that function. If the runtime's formula
    drifts, this fails rather than the gate silently going stale.
    """
    backends = pytest.importorskip("scwbd.runtime.backends")
    runtime = pytest.importorskip("scwbd.runtime")

    head_model = runtime.spherical_phantom()
    backend = backends.AnalyticSphericalEField()
    coil = backends.CoilSpec.figure_eight()

    from scwbd.runtime.serving import _default_warmup_pose

    request = _default_warmup_pose(head_model)

    # drive the backend through its own API, then recompute with ours
    solved = backend.solve_field(head_model, request.pose, coil)
    pos, mdot = backends._dipoles_in_head_frame(request.pose, coil)
    ours = primary_tangential_projection(
        head_model.cortex_vertices, head_model.cortex_normals, pos, mdot
    )
    assert torch.allclose(solved.e, ours, rtol=1e-10, atol=0.0), (
        "the runtime fallback no longer computes primary_tangential_projection; "
        "gate N9's subject and the runtime path have diverged"
    )


# ---------------------------------------------------------------------------
# the provenance pin (gate N9): the bound is read at run time AND pinned
# ---------------------------------------------------------------------------


def test_the_pin_matches_what_the_runtime_currently_declares():
    """If this fails, someone moved the bound without updating its justification."""
    backends = pytest.importorskip("scwbd.runtime.backends")
    from scwbd.intervene.run_field_gates import N9_PINNED_BOUND, bound_has_moved

    observed = backends.AnalyticSphericalEField().solution_discrepancy_fraction
    assert not bound_has_moved(observed), (
        f"scwbd.runtime declares {tuple(observed)} but gate N9 is pinned to "
        f"{N9_PINNED_BOUND['solution_discrepancy_fraction']}, justified on "
        f"{N9_PINNED_BOUND['justified_on']}. Update N9_PINNED_BOUND in the same "
        "commit that moves the bound."
    )


def test_the_pin_carries_a_date_and_a_justification():
    from scwbd.intervene.run_field_gates import N9_PINNED_BOUND

    assert N9_PINNED_BOUND["justified_on"]
    assert N9_PINNED_BOUND["justified_by"]
    assert len(N9_PINNED_BOUND["justification"]) > 200


@pytest.mark.parametrize(
    "observed,moved",
    [
        ((0.0, 1.35), False),  # unchanged
        ((0.0, 2.00), True),   # loosened -- the failure mode this exists for
        ((0.0, 1.20), True),   # tightened -- also flagged: the question is
                               # "did it move without anyone saying why"
        ((-0.1, 1.35), True),  # lower bound moved
    ],
)
def test_the_provenance_guard_fires_on_any_movement(observed, moved):
    """A guard nobody can show firing is indistinguishable from one that cannot."""
    from scwbd.intervene.run_field_gates import bound_has_moved

    assert bound_has_moved(observed) is moved
