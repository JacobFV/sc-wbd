# N3_em_solver — PASS

**Claim.** The quasi-static CONDUCTION solver reproduces the closed-form potential of a current dipole in an unbounded homogeneous conductor, validated independently of any neural-response model. This is the EEG/lead-field forward problem; it is NOT the magnetically induced TMS field, which has a different source term and boundary condition and needs its own gate (N6).

**Falsified by (thesis).** relative error above tolerance against the analytic dipole solution

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 19c4acc · 2026-08-06T08:19:53+00:00*

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
- Observed order falls from ~1.94 to ~1.16 at the finest grid. That is the zero-Dirichlet truncation floor, not a solver defect: agent Faraday's constant-h box study separates the two (~0.0023 of the 0.00696 is finite-domain error, ~0.0046 discretisation). Reported, not smoothed.
- Solver provenance: agent Faraday, branch wt/faraday @ 915fcad, NOT yet merged to master. This verdict was produced by re-running master's gate code against those solvers, not by adopting their report.
