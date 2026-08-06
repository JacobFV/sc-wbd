"""The §11.1 field-physics solvers (N3, N4), checked the way the gate checks them.

SIMULATION ONLY.

These are *verification* tests: each solver is compared to a closed form it does
not contain.  The property that makes the comparison worth anything is that the
reference never enters the solve -- the Poisson boundary is zero, not the
analytic potential, and the FDTD source amplitude is fixed a priori rather than
fitted.  So the tests also assert that negative: a solver that had been handed
its answer would agree at every resolution, and these do not.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scwbd.intervene.numerics import (
    acoustic_grid_convergence,
    em_grid_convergence,
    free_field_monopole_fdtd,
    quasistatic_dipole_potential_fd,
    run_free_field_monopole,
    solve_dipole_potential,
)

_SIGMA = 0.33
_K = 100.0
_TOL = 0.05  # the gate's preregistered relative tolerance


def _dipole_points(n: int = 200, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 0.05, size=(n, 3))
    return p[np.linalg.norm(p, axis=1) > 0.02]


def _dipole_reference(points, p=(0.0, 0.0, 1e-8), sigma=_SIGMA) -> np.ndarray:
    r = np.asarray(points, float)
    d = np.linalg.norm(r, axis=-1)
    return (r @ np.asarray(p, float)) / (4.0 * math.pi * sigma * d**3)


# ---------------------------------------------------------------------------
# N3: quasi-static current dipole
# ---------------------------------------------------------------------------


def test_fd_poisson_reproduces_the_closed_form_dipole_potential():
    pts = _dipole_points()
    ref = _dipole_reference(pts)
    got = quasistatic_dipole_potential_fd(pts, (0, 0, 0), (0, 0, 1e-8), _SIGMA,
                                          n_per_axis=128)
    rel = np.abs(got - ref) / np.abs(ref).mean()
    assert rel.mean() < _TOL, rel.mean()


def test_fd_poisson_converges_at_second_order():
    rows = em_grid_convergence(_dipole_points(), n_list=(48, 96, 144))
    errs = [r["mean_relative_error"] for r in rows]
    assert errs[0] > errs[1] > errs[2], errs
    assert rows[-1]["observed_order"] > 1.5, rows


def test_the_reference_never_enters_the_solve():
    """A coarse grid must be measurably *wrong*, and the boundary must be zero."""
    pts = _dipole_points()
    ref = _dipole_reference(pts)
    res = solve_dipole_potential(pts, n_per_axis=32)
    assert res.boundary == "homogeneous_dirichlet_on_truncation_box"
    assert res.meta["analytic_data_used"] is False
    err = float((np.abs(res.potential - ref) / np.abs(ref).mean()).mean())
    assert err > 1e-3, "a 32^3 grid agreeing to round-off would mean the answer leaked in"


def test_the_potential_is_inversely_proportional_to_conductivity():
    """Physics the discretisation must reproduce, independent of the reference."""
    pts = _dipole_points(80)
    a = quasistatic_dipole_potential_fd(pts, sigma=_SIGMA, n_per_axis=64)
    b = quasistatic_dipole_potential_fd(pts, sigma=2 * _SIGMA, n_per_axis=64)
    assert np.allclose(a, 2 * b, rtol=1e-10)


def test_the_dipole_must_land_on_a_node():
    with pytest.raises(ValueError, match="even"):
        solve_dipole_potential(_dipole_points(20), n_per_axis=33)


def test_a_boundary_inside_the_data_is_refused():
    with pytest.raises(ValueError, match="does not enclose"):
        solve_dipole_potential(_dipole_points(20), n_per_axis=32, half_width_m=0.01)


# ---------------------------------------------------------------------------
# N4: free-field monopole
# ---------------------------------------------------------------------------


def _acoustic_points(n: int = 120, seed: int = 5) -> np.ndarray:
    """A small cloud: the FDTD domain is dominated by the absorbing sponge, so
    keeping the data close to the source is what keeps these tests seconds long."""
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 0.012, size=(n, 3))
    return p[np.linalg.norm(p, axis=1) > 0.008]


@pytest.mark.slow
def test_fdtd_reproduces_free_field_spreading():
    pts = _acoustic_points()
    ref = 1.0 / np.linalg.norm(pts, axis=1)
    got = np.abs(free_field_monopole_fdtd(pts, (0, 0, 0), _K,
                                          points_per_wavelength=12))
    rel = np.abs(got - ref) / ref.mean()
    assert rel.mean() < _TOL, rel.mean()


@pytest.mark.slow
def test_fdtd_satisfies_its_own_helmholtz_equation():
    from scwbd.bench.numerics import helmholtz_residual

    res = run_free_field_monopole(_acoustic_points(40), (0, 0, 0), _K,
                                  points_per_wavelength=12)
    r = helmholtz_residual(res.grid_block, dx=res.spacing_m, k=_K)
    assert r < _TOL, r
    # the floor is the FDTD dispersion relation, O((k h)^2), not zero
    assert r > 1e-4, r


@pytest.mark.slow
def test_fdtd_error_and_helmholtz_residual_both_fall_under_refinement():
    rows = acoustic_grid_convergence(_acoustic_points(60), ppw_list=(8, 12, 16))
    assert [r["time_step_s"] for r in rows] == sorted(
        (r["time_step_s"] for r in rows), reverse=True
    ), "dt must refine with h or the residual is not a refinement statement"
    errs = [r["mean_relative_error"] for r in rows]
    res = [r["helmholtz_relative_residual"] for r in rows]
    assert errs[0] > errs[-1], errs
    assert res[0] > res[1] > res[2], res


@pytest.mark.slow
def test_the_source_amplitude_is_not_fitted_to_the_reference():
    """The residual amplitude bias is reported, not divided out."""
    rows = acoustic_grid_convergence(_acoustic_points(60), ppw_list=(8, 16))
    ratios = [r["mean_amplitude_ratio"] for r in rows]
    assert all(r != 1.0 for r in ratios), ratios
    assert abs(ratios[-1] - 1.0) < abs(ratios[0] - 1.0), ratios


def test_cfl_violation_is_refused_rather_than_run_unstably():
    with pytest.raises(ValueError, match="CFL"):
        run_free_field_monopole(_acoustic_points(10), (0, 0, 0), _K,
                                points_per_wavelength=12, steps_per_period=4)
