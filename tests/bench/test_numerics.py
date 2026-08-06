"""§11.1: compiler, solver, boundary and physical-solver checks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pytest

from scwbd.bench.numerics import (
    analytic_dipole_potential,
    analytic_free_field_pressure,
    boundary_consistency,
    check_compiler_correctness,
    check_conservation,
    check_seed_reproducibility,
    check_solver_convergence,
    check_stability,
    convergence_order,
    helmholtz_residual,
    permit_adaptive_resolution,
    run_numerics_suite,
    validate_acoustic_solver,
    validate_em_solver,
)


# --------------------------------------------------------------------------
# compiler fixtures (duck-typed against ARCHITECTURE.md §2)
# --------------------------------------------------------------------------
@dataclass
class _Support:
    units: str = "V"
    frame: str = "subject_surface_RAS"


@dataclass
class _Temporal:
    clock: str = "eeg_amp"


@dataclass
class _Port:
    support: _Support = field(default_factory=_Support)
    temporal: _Temporal = field(default_factory=_Temporal)


@dataclass
class _Region:
    ports: list = field(default_factory=lambda: [_Port()])


@dataclass
class _Op:
    delay: float = 0.01


@dataclass
class _Sched:
    dt_min: float = 0.001


@dataclass
class _Compiled:
    state_layout: dict
    adjacency: dict
    dispatch: list
    schedule: _Sched
    regions: list


def _good_compiled():
    return _Compiled(
        state_layout={"r0": (0, 4), "r1": (4, 4)},
        adjacency={"hard": np.eye(2), "soft": np.zeros((2, 2))},
        dispatch=[_Op(0.01), _Op(0.02)],
        schedule=_Sched(),
        regions=[_Region(), _Region()],
    )


def test_compiler_check_passes_a_consistent_model():
    rep = check_compiler_correctness(_good_compiled())
    assert rep.status == "PASS", rep.blocking_reasons


def test_compiler_check_catches_overlapping_state_offsets():
    c = _good_compiled()
    c.state_layout = {"r0": (0, 6), "r1": (4, 4)}      # overlap
    rep = check_compiler_correctness(c)
    assert rep.status == "FAIL"
    assert any("overlap" in r for r in rep.blocking_reasons)


def test_compiler_check_catches_a_missing_unit():
    c = _good_compiled()
    c.regions[0].ports[0].support.units = ""
    rep = check_compiler_correctness(c)
    assert rep.status == "FAIL"
    assert any("missing_units" in r for r in rep.blocking_reasons)


def test_compiler_check_catches_a_negative_delay():
    c = _good_compiled()
    c.dispatch = [_Op(-0.5)]
    rep = check_compiler_correctness(c)
    assert rep.status == "FAIL"
    assert any("negative" in r for r in rep.blocking_reasons)


def test_compiler_check_catches_a_delay_below_the_base_step():
    c = _good_compiled()
    c.dispatch = [_Op(1e-6)]
    rep = check_compiler_correctness(c)
    assert rep.status == "FAIL"
    assert any("below_base_dt" in r for r in rep.blocking_reasons)


def test_compiler_check_catches_inconsistent_mask_shapes():
    c = _good_compiled()
    c.adjacency = {"hard": np.eye(2), "soft": np.zeros((3, 3))}
    rep = check_compiler_correctness(c)
    assert rep.status == "FAIL"
    assert any("consistent_shape" in r for r in rep.blocking_reasons)


def test_compiler_check_without_a_model_is_could_not_run():
    rep = check_compiler_correctness(None)
    assert rep.status == "COULD_NOT_RUN"
    assert any("agent A" in r for r in rep.blocking_reasons)


# --------------------------------------------------------------------------
# solvers
# --------------------------------------------------------------------------
def _euler(dt: float) -> np.ndarray:
    """dx/dt = -x, x(0)=1, integrated to t=1: first-order accurate."""
    x, t = 1.0, 0.0
    while t < 1.0 - 1e-12:
        x = x + dt * (-x)
        t += dt
    return np.array([x])


def _rk4(dt: float) -> np.ndarray:
    x, t = 1.0, 0.0
    f = lambda v: -v
    while t < 1.0 - 1e-12:
        k1 = f(x); k2 = f(x + dt * k1 / 2); k3 = f(x + dt * k2 / 2); k4 = f(x + dt * k3)
        x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += dt
    return np.array([x])


def test_convergence_order_recovers_first_and_fourth_order():
    dts = [0.1, 0.05, 0.025, 0.0125]
    ref = np.array([math.exp(-1.0)])
    e1 = [abs(_euler(dt)[0] - ref[0]) for dt in dts]
    e4 = [abs(_rk4(dt)[0] - ref[0]) for dt in dts]
    assert 0.8 < convergence_order(e1, dts) < 1.3
    assert convergence_order(e4, dts) > 3.0


def test_solver_convergence_check_fails_a_non_converging_solver():
    ref = np.array([math.exp(-1.0)])
    ok = check_solver_convergence(_euler, dts=[0.1, 0.05, 0.025, 0.0125], reference=ref,
                                  expected_order=1.0)
    assert ok.status == "PASS"
    broken = check_solver_convergence(lambda dt: np.array([0.5]),
                                      dts=[0.1, 0.05, 0.025], reference=ref,
                                      expected_order=1.0)
    assert broken.status == "FAIL"


def test_stability_check_catches_blow_up_and_nans():
    good = np.exp(-np.linspace(0, 5, 200))[:, None]
    assert check_stability(good).status == "PASS"
    blow = np.exp(np.linspace(0, 8, 200))[:, None]
    assert check_stability(blow).status == "FAIL"
    nan = good.copy(); nan[10] = np.nan
    assert check_stability(nan).status == "FAIL"


def test_conservation_check_catches_drift():
    t = np.linspace(0, 2 * np.pi, 400)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)
    energy = lambda s: float(s[0] ** 2 + s[1] ** 2)
    assert check_conservation(circle, energy, tol=1e-6).status == "PASS"
    spiral = circle * np.exp(0.01 * t)[:, None]
    assert check_conservation(spiral, energy, tol=1e-6).status == "FAIL"


def test_seed_reproducibility_catches_non_determinism_and_ignored_seeds():
    good = lambda s: np.random.default_rng(s).normal(size=5)
    assert check_seed_reproducibility(good).status == "PASS"
    nondet = lambda s: np.random.default_rng().normal(size=5)
    assert check_seed_reproducibility(nondet).status == "FAIL"
    ignores_seed = lambda s: np.ones(5)
    assert check_seed_reproducibility(ignores_seed).status == "FAIL"


# --------------------------------------------------------------------------
# boundary consistency permit
# --------------------------------------------------------------------------
def test_adaptive_resolution_permit_is_granted_only_on_agreement():
    rng = np.random.default_rng(0)
    coarse = rng.normal(10, 1, size=500)
    agreeing = coarse + rng.normal(0, 0.02, size=500)
    disagreeing = coarse * 1.4
    permit, rep = permit_adaptive_resolution(agreeing, coarse, tol=0.05)
    assert permit.granted and rep.status == "PASS"
    permit.require()  # does not raise
    bad_permit, bad_rep = permit_adaptive_resolution(disagreeing, coarse, tol=0.05)
    assert not bad_permit.granted and bad_rep.status == "FAIL"
    with pytest.raises(PermissionError, match="adaptive resolution is not permitted"):
        bad_permit.require()


def test_permit_is_refused_when_a_backend_produced_nothing():
    permit, rep = permit_adaptive_resolution(None, None)
    assert not permit.granted
    assert rep.status == "COULD_NOT_RUN"


# --------------------------------------------------------------------------
# physical solvers, validated with no neural model in the loop
# --------------------------------------------------------------------------
def test_em_solver_validated_against_the_analytic_dipole():
    rng = np.random.default_rng(1)
    pts = rng.normal(0, 0.08, size=(400, 3))
    pts = pts[np.linalg.norm(pts, axis=1) > 0.02]
    exact = lambda points, **kw: analytic_dipole_potential(
        points, kw.get("dipole_pos", (0, 0, 0)), kw.get("dipole_moment", (0, 0, 1e-8)),
        sigma=kw.get("sigma", 0.33))
    assert validate_em_solver(exact, points=pts).status == "PASS"
    wrong = lambda points, **kw: 1.3 * exact(points, **kw)
    assert validate_em_solver(wrong, points=pts).status == "FAIL"


def test_acoustic_solver_validated_against_free_field_spreading():
    rng = np.random.default_rng(2)
    pts = rng.normal(0, 0.05, size=(400, 3))
    pts = pts[np.linalg.norm(pts, axis=1) > 0.01]
    exact = lambda points, **kw: analytic_free_field_pressure(
        points, kw.get("source_pos", (0, 0, 0)), k=kw.get("k", 100.0))
    assert validate_acoustic_solver(exact, points=pts).status == "PASS"
    wrong = lambda points, **kw: np.abs(exact(points, **kw)) ** 0.5
    assert validate_acoustic_solver(wrong, points=pts).status == "FAIL"


def test_absent_physical_solvers_suspend_the_field_claims():
    em = validate_em_solver(None)
    ac = validate_acoustic_solver(None)
    assert em.status == ac.status == "COULD_NOT_RUN"
    assert "agent G" in " ".join(em.blocking_reasons)
    assert em.manifest.consequence_if_failed.startswith("The EM solver may not be used")


def test_helmholtz_residual_is_small_for_a_plane_wave():
    k, dx, n = 20.0, 0.01, 24
    ax = np.arange(n) * dx
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    field = np.cos(k * X)
    assert helmholtz_residual(field, dx=dx, k=k) < 0.05
    assert helmholtz_residual(np.cos(3 * k * X), dx=dx, k=k) > 0.5


def test_numerics_suite_reports_every_absent_input():
    reports = run_numerics_suite()
    assert {r.manifest.claim_id for r in reports} == {
        "N1_compiler_correctness", "N5_solver_suite", "N2_boundary_consistency",
        "N3_em_solver", "N4_acoustic_solver",
    }
    assert {r.status for r in reports} == {"COULD_NOT_RUN"}
