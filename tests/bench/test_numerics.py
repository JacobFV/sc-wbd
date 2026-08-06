"""§11.1: compiler, solver, boundary and physical-solver checks."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

from scwbd.bench.adapters import reference_compiled
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
    validate_induced_efield_solver,
)


# --------------------------------------------------------------------------
# N1: agent A's compiler, on agent A's worked three-region example
# --------------------------------------------------------------------------
def _reference():
    dep = reference_compiled()
    if not dep.available:
        pytest.skip(f"agent A's reference example is unavailable: {dep.reason}")
    return dep.obj


def _mutate(model, **kw):
    """Corrupt one structure of a compiled artifact, leaving the rest intact."""
    return dataclasses.replace(model, **kw)


class _StubOperator:
    """An operator descriptor with a chosen delay, preserving its edge."""

    def __init__(self, d, delay: float):
        self.key, self.src, self.dst = d.key, d.src, d.dst
        self.evidence_class, self.clock = d.evidence_class, d.clock
        self._delay = delay

    def delay_seconds(self) -> float:
        return self._delay


def _with_delay(model, index: int, delay: float):
    ds = list(model.dispatch.descriptors)
    ds[index] = _StubOperator(ds[index], delay)
    return _mutate(model, dispatch=dataclasses.replace(model.dispatch,
                                                       descriptors=tuple(ds)))


def test_compiler_check_passes_the_reference_example():
    """The compiler emits an internally consistent artifact for three_region."""
    rep = check_compiler_correctness(_reference())
    assert rep.status == "PASS", rep.blocking_reasons
    assert {s.name for s in rep.subchecks} == {
        "state_layout", "units_frames_clocks", "delays", "masks",
        "gradient_permissions", "uncertainty_ledger", "claim_class_integrity",
    }
    # a PASS here is a statement about the compiler, not about a brain
    assert any("not evidence about any other schema" in n for n in rep.notes)


def test_n1_runs_against_the_reference_example_by_default():
    rep = check_compiler_correctness()
    assert rep.artifacts["subject"].startswith("reference example")
    assert rep.status in ("PASS", "FAIL")


def test_compiler_check_catches_overlapping_state_offsets():
    m = _reference()
    L = m.state_layout
    clash = dataclasses.replace(L.entries[1], elem_offset=L.entries[1].elem_offset - 2)
    bad = _mutate(m, state_layout=dataclasses.replace(
        L, entries=(L.entries[0], clash) + L.entries[2:]))
    rep = check_compiler_correctness(bad)
    assert rep.status == "FAIL"
    assert any("overlap" in r for r in rep.blocking_reasons)


def test_compiler_check_catches_a_gap_in_the_state_vector():
    m = _reference()
    L = m.state_layout
    moved = dataclasses.replace(L.entries[-1], elem_offset=L.entries[-1].elem_offset + 8)
    bad = _mutate(m, state_layout=dataclasses.replace(L, entries=L.entries[:-1] + (moved,)))
    rep = check_compiler_correctness(bad)
    assert rep.status == "FAIL"
    assert any("gaps" in r for r in rep.blocking_reasons)


def test_compiler_check_catches_a_missing_unit():
    m = _reference()
    L = m.state_layout
    bad = _mutate(m, state_layout=dataclasses.replace(
        L, entries=(dataclasses.replace(L.entries[0], units=""),) + L.entries[1:]))
    rep = check_compiler_correctness(bad)
    assert rep.status == "FAIL"
    assert any("missing_units" in r for r in rep.blocking_reasons)


def test_compiler_check_catches_an_unknown_clock():
    m = _reference()
    L = m.state_layout
    bad = _mutate(m, state_layout=dataclasses.replace(
        L, entries=(dataclasses.replace(L.entries[0], clock="not_a_clock"),)
        + L.entries[1:]))
    rep = check_compiler_correctness(bad)
    assert rep.status == "FAIL"
    assert any("unknown_referenced" in r for r in rep.blocking_reasons)


def test_compiler_check_catches_a_negative_delay():
    rep = check_compiler_correctness(_with_delay(_reference(), 3, -0.5))
    assert rep.status == "FAIL"
    assert any("delays.negative" in r for r in rep.blocking_reasons)


def test_compiler_check_catches_a_delay_below_the_base_step():
    """A delay the scheduler cannot represent is silently rounded away."""
    rep = check_compiler_correctness(_with_delay(_reference(), 3, 1e-7))
    assert rep.status == "FAIL"
    assert any("below_base_dt" in r for r in rep.blocking_reasons)


def test_compiler_check_catches_a_delay_beyond_the_hyperperiod():
    m = _reference()
    rep = check_compiler_correctness(_with_delay(m, 3, m.schedule.hyperperiod * 10))
    assert rep.status == "FAIL"
    assert any("beyond_hyperperiod" in r for r in rep.blocking_reasons)


def test_compiler_check_catches_a_mask_that_omits_a_dispatched_operator():
    m = _reference()
    n = len(m.adjacency.region_ids)
    empty = torch.sparse_coo_tensor(
        torch.zeros((2, 0), dtype=torch.int64), torch.zeros(0, dtype=torch.bool),
        size=(n, n)).coalesce()
    bad = _mutate(m, adjacency=dataclasses.replace(
        m.adjacency, masks={**m.adjacency.masks, "hard": empty}))
    rep = check_compiler_correctness(bad)
    assert rep.status == "FAIL"
    assert any("dispatched_edges_not_masked" in r for r in rep.blocking_reasons)


def test_compiler_check_catches_a_mask_edge_no_operator_implements():
    m = _reference()
    n = len(m.adjacency.region_ids)
    full = torch.ones((n, n), dtype=torch.bool).to_sparse_coo().coalesce()
    bad = _mutate(m, adjacency=dataclasses.replace(
        m.adjacency, masks={**m.adjacency.masks, "proposed": full}))
    rep = check_compiler_correctness(bad)
    assert rep.status == "FAIL"
    assert any("masked_edges_not_dispatched" in r for r in rep.blocking_reasons)


def test_compiler_check_catches_an_unbacked_bias_term():
    """Refusal R08: a bias point estimate with no estimator and no bound."""
    m = _reference()
    bad = _mutate(m, ledger=dataclasses.replace(
        m.ledger, unbacked_bias=("operator:couple_a_b",)))
    rep = check_compiler_correctness(bad)
    assert rep.status == "FAIL"
    assert any("unbacked_bias_terms" in r for r in rep.blocking_reasons)


def test_compiler_check_catches_a_silently_demoted_claim_class():
    """An override moves the claim class; the artifact may not hide it."""
    m = _reference()
    bad = _mutate(m, provenance=dataclasses.replace(
        m.provenance, effective_claim_class="functional"))
    rep = check_compiler_correctness(bad)
    assert rep.status == "FAIL"
    assert any("was_demoted" in r for r in rep.blocking_reasons)
    assert rep.manifest.consequence_if_failed.startswith("Fix the compiler")


def test_compiler_check_without_a_model_or_reference_is_could_not_run():
    rep = check_compiler_correctness(None, use_reference_example=False)
    assert rep.status == "COULD_NOT_RUN"
    assert any("no CompiledModel supplied" in r for r in rep.blocking_reasons)


def test_reference_example_compiles_with_no_overridden_refusals():
    """Agent A's worked example is a clean subject: nothing was overridden."""
    m = _reference()
    assert not m.provenance.was_overridden
    assert not m.provenance.claim_was_demoted


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
    assert em.manifest.consequence_if_failed.startswith(
        "The conduction solver may not be used")


def test_helmholtz_residual_is_small_for_a_plane_wave():
    k, dx, n = 20.0, 0.01, 24
    ax = np.arange(n) * dx
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    field = np.cos(k * X)
    assert helmholtz_residual(field, dx=dx, k=k) < 0.05
    assert helmholtz_residual(np.cos(3 * k * X), dx=dx, k=k) > 0.5


def test_numerics_suite_reports_every_absent_input():
    reports = {r.manifest.claim_id: r for r in run_numerics_suite()}
    assert set(reports) == {
        "N1_compiler_correctness", "N5_solver_suite", "N2_boundary_consistency",
        "N3_em_solver", "N4_acoustic_solver", "N6_induced_efield",
        "N8_induced_efield_contact",
    }
    # checks whose subject has landed produce a verdict; the rest stay blocked
    # and say so rather than passing by default.
    assert reports["N1_compiler_correctness"].status in ("PASS", "FAIL")
    # field gates auto-wire to agent Faraday's solvers once those are importable,
    # so they produce a verdict rather than silently reverting to COULD_NOT_RUN
    for k in ("N3_em_solver", "N4_acoustic_solver", "N6_induced_efield",
              "N8_induced_efield_contact"):
        assert reports[k].status in ("PASS", "FAIL", "COULD_NOT_RUN")
    # these have no subject at all and must stay blocked
    assert {reports[k].status for k in
            ("N5_solver_suite", "N2_boundary_consistency")} == {"COULD_NOT_RUN"}


def test_field_gates_do_not_silently_revert_once_their_solvers_exist():
    """A default run must reproduce a verdict, not overwrite it with COULD_NOT_RUN."""
    from scwbd.bench.adapters import induced_field_solver

    if not induced_field_solver().available:
        pytest.skip("agent Faraday's induced-field solver is not importable here")
    reports = {r.manifest.claim_id: r for r in run_numerics_suite()}
    assert reports["N6_induced_efield"].status != "COULD_NOT_RUN"
    assert reports["N8_induced_efield_contact"].status != "COULD_NOT_RUN"


def test_n8_contact_gate_exists_as_its_own_row_and_states_its_contract():
    from scwbd.bench.numerics import validate_induced_efield_contact

    rep = validate_induced_efield_contact()
    assert rep.status == "COULD_NOT_RUN"
    reason = " ".join(rep.blocking_reasons)
    assert "contact geometry" in reason
    assert "spectral reference does NOT extend here" in reason
    assert "preregistered tolerance" in reason
    # the consequence names the downstream consumer obligation
    assert "Unresolved/Defer" in rep.manifest.consequence_if_failed
    assert "0.955" in rep.artifacts["why_this_row_exists"]


def test_a_passing_numerical_check_must_record_what_it_measured():
    """Cajal's lesson: an artifact with no provenance is how stale outputs pass."""
    from scwbd.bench.report import ClaimReport, Metric, ReportDisciplineError, SubCheck
    from scwbd.bench.numerics import _manifest

    rep = ClaimReport(
        manifest=_manifest("N_TEST", "a claim", "a falsifier", "a consequence"),
        subchecks=[SubCheck(name="ok", description="d", metrics=[
            Metric(name="x", value=1.0, kind="numerical", exact=True, threshold=0.5)])],
        kind="numerics",
    )
    with pytest.raises(ReportDisciplineError, match="how it was produced"):
        rep.finalize()
    rep.artifacts["subject"] = "some.module.solver"
    assert rep.finalize().status == "PASS"


def test_field_gate_reports_name_the_callable_that_produced_them():
    exact = lambda points, **kw: analytic_dipole_potential(
        points, kw.get("dipole_pos", (0, 0, 0)), kw.get("dipole_moment", (0, 0, 1e-8)),
        sigma=kw.get("sigma", 0.33))
    rng = np.random.default_rng(6)
    pts = rng.normal(0, 0.08, size=(200, 3))
    pts = pts[np.linalg.norm(pts, axis=1) > 0.02]
    rep = validate_em_solver(exact, points=pts)
    assert rep.artifacts["subject"].endswith("<lambda>")


# --------------------------------------------------------------------------
# N3/N4 scope and refinement wording (both caught by agent Faraday)
# --------------------------------------------------------------------------
def test_n3_claim_is_conduction_and_says_it_does_not_cover_induction():
    """A conduction PASS must not be readable as licensing the induced TMS field."""
    rep = validate_em_solver(None)
    claim = rep.manifest.claim_text
    assert "CONDUCTION" in claim
    assert "NOT the magnetically induced TMS field" in claim
    assert "N6" in claim


def test_n3_pass_carries_the_scope_caveat_in_its_notes():
    exact = lambda points, **kw: analytic_dipole_potential(
        points, kw.get("dipole_pos", (0, 0, 0)), kw.get("dipole_moment", (0, 0, 1e-8)),
        sigma=kw.get("sigma", 0.33))
    rng = np.random.default_rng(1)
    pts = rng.normal(0, 0.08, size=(300, 3))
    pts = pts[np.linalg.norm(pts, axis=1) > 0.02]
    rep = validate_em_solver(exact, points=pts)
    assert rep.status == "PASS"
    notes = " ".join(rep.notes)
    assert "conduction, not induction" in notes
    assert "N6_induced_efield" in notes
    assert rep.artifacts["does_not_cover"].startswith("magnetically induced")


def test_n4_falsification_criterion_names_time_refinement_not_h():
    """Refining h alone leaves the residual flat; the criterion must not misfire."""
    rep = validate_acoustic_solver(None)
    crit = rep.manifest.falsified_by
    assert "TIME step" in crit
    assert "fixed CFL" in crit
    assert "is NOT a falsification" in crit


def test_n4_report_carries_the_refinement_rule():
    exact = lambda points, **kw: analytic_free_field_pressure(
        points, kw.get("source_pos", (0, 0, 0)), k=kw.get("k", 100.0))
    rng = np.random.default_rng(2)
    pts = rng.normal(0, 0.05, size=(300, 3))
    pts = pts[np.linalg.norm(pts, axis=1) > 0.01]
    rep = validate_acoustic_solver(exact, points=pts)
    notes = " ".join(rep.notes)
    assert "TEMPORAL dispersion" in notes
    assert "Refine dt with h at fixed CFL" in notes


def test_temporal_dispersion_predicts_the_measured_helmholtz_residual():
    """Pins the reasoning so nobody 'fixes' the residual by refining h.

    Inputs are agent Faraday's reported N4 configuration (k=100 /m, c=1500 m/s,
    60 steps per period at ppw=20); the measured residual was 9.178e-4.
    """
    k, c, steps_per_period = 100.0, 1500.0, 60
    omega = k * c
    dt = (2.0 * math.pi / omega) / steps_per_period
    predicted = (omega * dt) ** 2 / 12.0
    assert predicted == pytest.approx(9.139e-4, rel=0.01)
    assert predicted == pytest.approx(9.178e-4, rel=0.01)   # measured
    # and it is a dt effect: halving dt quarters it, halving h does nothing
    assert (omega * (dt / 2)) ** 2 / 12.0 == pytest.approx(predicted / 4.0, rel=1e-9)


# --------------------------------------------------------------------------
# N6: the induced-field gate that N3 does not cover
# --------------------------------------------------------------------------
def test_n6_without_a_solver_or_reference_cannot_run():
    rep = validate_induced_efield_solver()
    assert rep.status == "COULD_NOT_RUN"
    reason = " ".join(rep.blocking_reasons)
    assert "Faraday" in reason
    assert "will not substitute the conduction reference" in reason
    assert "suspended" in rep.manifest.consequence_if_failed


def test_n6_refuses_to_reuse_the_conduction_reference():
    """Supplying only a solver is not enough: induction needs its own closed form."""
    rep = validate_induced_efield_solver(lambda points, **kw: np.zeros(len(points)))
    assert rep.status == "COULD_NOT_RUN"
    assert "closed-form reference" in " ".join(rep.blocking_reasons)


def test_n6_passes_an_exact_solver_and_fails_a_wrong_one():
    def reference(points, **kw):
        p = np.asarray(points, dtype=float)
        return np.cross(p, np.array([0.0, 0.0, 1.0]))[:, 0]

    rng = np.random.default_rng(3)
    pts = rng.normal(0, 0.05, size=(300, 3))
    pts = pts[np.linalg.norm(pts, axis=1) > 0.02]
    # a solver with a small but real error, well above the reference's own bound
    ok = validate_induced_efield_solver(
        lambda points, **kw: reference(points) * 1.002, analytic=reference, points=pts,
        convergence_ratio=0.77, reference_degree=48)
    assert ok.status == "PASS", ok.blocking_reasons
    bad = validate_induced_efield_solver(lambda points, **kw: 1.4 * reference(points),
                                         analytic=reference, points=pts,
                                         convergence_ratio=0.77, reference_degree=48)
    assert bad.status == "FAIL"


def test_n6_discloses_when_solver_and_reference_share_a_module():
    def reference(points, **kw):
        return np.asarray(points, dtype=float)[:, 2]

    rng = np.random.default_rng(4)
    pts = rng.normal(0, 0.05, size=(200, 3))
    rep = validate_induced_efield_solver(
        lambda points, **kw: reference(points) * 1.002, analytic=reference, points=pts,
        convergence_ratio=0.77, reference_degree=48)
    m = next(mm for s in rep.subchecks for mm in s.metrics
             if mm.name == "induced_efield.reference_shares_module_with_solver")
    assert m.value == 0.0 or m.value == 1.0
    assert "must not be described as independent validation" in m.note
    # disclosure must not, by itself, block the verdict
    assert rep.status == "PASS"
    sub = next(s for s in rep.subchecks if s.name == "reference_provenance")
    assert sub.status == "FAIL" and sub.mandatory is False


def test_n6_mesh_convergence_can_fail():
    def reference(points, **kw):
        return np.asarray(points, dtype=float)[:, 2]

    rng = np.random.default_rng(5)
    pts = rng.normal(0, 0.05, size=(200, 3))
    good = validate_induced_efield_solver(
        lambda points, **kw: reference(points) * 1.002, analytic=reference, points=pts,
        convergence_ratio=0.77, reference_degree=48,
        convergence=[{"h": 0.04, "error": 0.0386}, {"h": 0.02, "error": 0.0102},
                     {"h": 0.01, "error": 0.00266}])
    assert good.status == "PASS"
    flat = validate_induced_efield_solver(
        lambda points, **kw: reference(points) * 1.002, analytic=reference, points=pts,
        convergence_ratio=0.77, reference_degree=48,
        convergence=[{"h": 0.04, "error": 0.03}, {"h": 0.02, "error": 0.03},
                     {"h": 0.01, "error": 0.03}])
    assert flat.status == "FAIL"


def test_n6_refuses_when_the_reference_validity_domain_is_undeclared():
    """A reference whose own accuracy at this geometry is unknown is not a reference."""
    def reference(points, **kw):
        return np.asarray(points, dtype=float)[:, 2]

    rng = np.random.default_rng(7)
    pts = rng.normal(0, 0.05, size=(200, 3))
    rep = validate_induced_efield_solver(reference, analytic=reference, points=pts)
    assert rep.status == "COULD_NOT_RUN"
    assert any("convergence ratio" in r for r in rep.blocking_reasons)


def test_n6_cannot_conclude_at_the_references_own_noise_floor():
    """At contact ratio the series bound swamps the solver error: cannot conclude.

    This is the exact reason N6 does not extend to contact geometry, expressed
    as a gate outcome rather than a footnote.
    """
    def reference(points, **kw):
        return np.asarray(points, dtype=float)[:, 2]

    rng = np.random.default_rng(8)
    pts = rng.normal(0, 0.05, size=(200, 3))
    rep = validate_induced_efield_solver(
        lambda points, **kw: reference(points) * 1.001, analytic=reference, points=pts,
        convergence_ratio=0.955, reference_degree=48)   # contact-like ratio
    sub = next(s for s in rep.subchecks if s.name == "reference_validity_domain")
    assert sub.status == "COULD_NOT_RUN"
    assert rep.status == "COULD_NOT_RUN"
    assert "noise floor" in " ".join(rep.blocking_reasons)
