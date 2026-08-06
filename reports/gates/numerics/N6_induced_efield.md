# N6_induced_efield — PASS

**Claim.** The magnetically induced E-field solver reproduces the closed-form (Sarvas / Heller-van Hulsteyn) solution for a spherically symmetric conductor, validated independently of any neural-response model.

**Falsified by (thesis).** relative error above tolerance against the closed form, or a mesh-refinement study that does not converge at the advertised order

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 1a35a9a · 2026-08-06T19:03:57+00:00*

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| induced_efield | yes | PASS | induced_efield.mean_relative_error = 0.00214881 dimensionless [0.001907, 0.002423]_95% (threshold < 0.05, interval-strict); induced_efield.max_relative_error = 0.0792145 dimensionless |
| reference_provenance | no | PASS | induced_efield.reference_shares_module_with_solver = 0 dimensionless (threshold < 0.5) |
| reference_validity_domain | yes | PASS | reference.convergence_ratio = 0.772727 dimensionless (threshold < 0.9); reference.a_priori_bound = 4.21949e-06 dimensionless; reference.bound_over_measured_error = 0.00196364 dimensionless (threshold < 0.1) |

## Baselines run

_none run_ — no baseline, no claim.

## Preregistered acceptance thresholds

- `relative_tol`: 0.05
- `expected_order`: 1.5

## Explicit non-goals

- These checks establish code correctness only. Numerical correctness is necessary, never sufficient: agreement with recorded signals is stronger, held-out perturbation stronger still (thesis §0.2).

## Notes

- Induction, not conduction: this gate is what N3 does NOT cover.
- STANDOFF ONLY. The reference series converges like (a/R_c)**degree. At a contact geometry (a/R_c ~ 0.955 for a coil element 4 mm off an 85 mm scalp) no feasible degree brings its bound below the solver error, so this gate validates the discretisation against a STANDOFF equivalent dipole, not against a contact coil. tms-robotics positions a coil in contact; that regime is gate N8_induced_efield_contact and it has not run.
- The validity domain is a metric in this report, not a footnote: a reader who checks only the headline error still sees reference.convergence_ratio.
- SERVED FROM CACHE (in part): at least one solver call re-used a stored result rather than re-solving. The stored value was produced by a solver whose module source hashes identically, so the physics is the same physics -- but this run did not recompute it, and that is recorded here rather than left to be inferred from numbers that are identical either way. Clear scwbd.bench.solver_cache.CACHE_DIR to force a full re-solve.
