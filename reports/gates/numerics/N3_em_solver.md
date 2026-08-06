# N3_em_solver — PASS

**Claim.** The quasi-static CONDUCTION solver reproduces the closed-form potential of a current dipole in an unbounded homogeneous conductor, validated independently of any neural-response model. This is the EEG/lead-field forward problem; it is NOT the magnetically induced TMS field, which has a different source term and boundary condition and needs its own gate (N6).

**Falsified by (thesis).** relative error above tolerance against the analytic dipole solution

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 1a35a9a · 2026-08-06T19:02:51+00:00*

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
- SERVED FROM CACHE (in part): at least one solver call re-used a stored result rather than re-solving. The stored value was produced by a solver whose module source hashes identically, so the physics is the same physics -- but this run did not recompute it, and that is recorded here rather than left to be inferred from numbers that are identical either way. Clear scwbd.bench.solver_cache.CACHE_DIR to force a full re-solve.
