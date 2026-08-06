# N6_induced_efield — PASS

**Claim.** The magnetically induced E-field solver reproduces the closed-form (Sarvas / Heller-van Hulsteyn) solution for a spherically symmetric conductor, validated independently of any neural-response model.

**Falsified by (thesis).** relative error above tolerance against the closed form, or a mesh-refinement study that does not converge at the advertised order

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 0575861 · 2026-08-06T13:29:35+00:00*

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| induced_efield | yes | PASS | induced_efield.mean_relative_error = 0.00214881 dimensionless [0.001907, 0.002423]_95% (threshold < 0.05, interval-strict); induced_efield.max_relative_error = 0.0792145 dimensionless |
| reference_provenance | no | PASS | induced_efield.reference_shares_module_with_solver = 0 dimensionless (threshold < 0.5) |
| reference_validity_domain | yes | PASS | reference.convergence_ratio = 0.772727 dimensionless (threshold < 0.9); reference.a_priori_bound = 4.21949e-06 dimensionless; reference.bound_over_measured_error = 0.00196364 dimensionless (threshold < 0.1) |
| mesh_convergence | yes | PASS | induced_efield.observed_order = 1.69439 dimensionless (threshold > 1.5) |

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
- The reference is INDEPENDENT of the solver, in code and in derivation. The solver is a surface-charge BEM; the reference solves the interior Neumann problem spectrally -- multipole expansion in regular solid harmonics, coefficients by quadrature on the sphere, gradient by automatic differentiation. They share no function and no formula. `reference_shares_module_with_solver` is 0 for that reason, and it is the reason rather than an accident of file layout.
- The reference is validated in its own right, not asserted: it converges geometrically to the closed form (9.7e-9 relative at degree 48), it reproduces the Heller-van Hulsteyn theorem r.E = 0 to the same order, and in the far-source limit it reduces to the elementary Faraday solution E = -(1/2) Bdot x r. Those three checks are in tests/intervene/test_spectral_reference.py.
- DISCLOSURE -- validity domain of the reference. The multipole series converges like (a/R_c)^L; here a/R_c = 0.7727, so the a-priori bound at degree 48 is 4.22e-06 and the MEASURED agreement with the closed form is 9.7e-9 -- respectively three and six orders below the ~4.8e-3 solver error being measured, which is the condition for calling this a reference. A coil element a few millimetres off the scalp has a/R_c -> 1 and would need a prohibitive degree: this reference is a yardstick for a standoff source, NOT for a coil in contact. Near-surface geometry is exactly what the BEM exists for, and validating it there needs a different reference -- N6 does not claim to have done so.
- Field accuracy is not target engagement, network effect or clinical utility (thesis Sec. 0.5). A numerical PASS lifts a precondition and licenses no claim.
