# N3_em_solver — PASS

**Claim.** The quasi-static CONDUCTION solver reproduces the closed-form potential of a current dipole in an unbounded homogeneous conductor, validated independently of any neural-response model. This is the EEG/lead-field forward problem; it is NOT the magnetically induced TMS field, which has a different source term and boundary condition and needs its own gate (N6).

**Falsified by (thesis).** relative error above tolerance against the analytic dipole solution

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git f233f77 · 2026-08-06T08:48:35+00:00*

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| em_solver | yes | PASS | em_solver.mean_relative_error = 0.0069564 dimensionless [0.00589, 0.008281]_95% (threshold < 0.05, interval-strict); em_solver.max_relative_error = 0.145211 dimensionless |

## Baselines run

_none run_ — no baseline, no claim.

## Preregistered acceptance thresholds

- `relative_tol`: 0.05
- `sigma_S_per_m`: 0.33

## Explicit non-goals

- These checks establish code correctness only. Numerical correctness is necessary, never sufficient: agreement with recorded signals is stronger, held-out perturbation stronger still (thesis §0.2).

## Notes

- SCOPE: conduction, not induction. A PASS licenses the quasi-static conduction discretisation used for EEG lead fields. It does NOT license the magnetically induced TMS field: different source term, different boundary condition, separate gate (N6_induced_efield).
- A verification gate is destroyed if the reference leaks into the solver. Check that the boundary data is homogeneous, not the analytic value, before reading this PASS as evidence.
- Field accuracy, target engagement, network effect and clinical utility remain separate quantities (thesis §0.5).
- Solver: scwbd.intervene.numerics.quasistatic_dipole_potential_fd -- a second-order 7-point finite-difference Poisson solve on a 256^3 grid, diagonalised exactly by the type-I DST. The truncation boundary carries HOMOGENEOUS DIRICHLET data (zero), not the analytic potential, so no value from the reference enters the solve; the boundary is placed at 1.9x the farthest field point and the residual truncation error is part of the reported number rather than removed by it.
- The reference here is a CURRENT dipole in an unbounded homogeneous conductor -- the EEG/lead-field forward problem, not the magnetically induced TMS field of scwbd.intervene.tms.efield. Passing N3 licenses the quasi-static conduction discretisation; the induced-field operator is separately convergence-tested against the Sarvas / Heller-van Hulsteyn closed form in tests/intervene/test_tms_efield.py.
- Refinement study (mean relative error vs h): n=128 h=0.00935 m err=0.0213; n=192 h=0.00623 m err=0.0097 order=1.94; n=256 h=0.00467 m err=0.0070 order=1.16
- The observed order falls off at the finest grid because the h-refinement holds the truncation box fixed, so the zero-Dirichlet truncation error is an h-independent floor. Enlarging the box at CONSTANT h separates the two: margin=1.90 n=256 h=0.00467 m err=0.0070; margin=2.38 n=320 h=0.00467 m err=0.0053; margin=2.85 n=384 h=0.00467 m err=0.0046. The discretisation error is the limit of that sequence; the difference from the reported number is what the finite domain costs, and it is a budget item, not a fitted correction.
