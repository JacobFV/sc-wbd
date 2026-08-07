# N8_induced_efield_contact — PASS

**Claim.** The induced-field solver is validated in the CONTACT regime (a coil element at clinical standoff from the scalp, a/R_c >= 0.95) to a preregistered tolerance — the geometry the downstream targeting consumer actually uses.

**Falsified by (thesis).** no reference or self-convergence study achieves a defensible tolerance at contact geometry, or the solver's error there exceeds the preregistered tolerance

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 0575861 · 2026-08-06T13:30:55+00:00*

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| is_contact_geometry | yes | PASS | contact.a_over_Rc = 0.955056 dimensionless (threshold > 0.95) |
| contact_efield | yes | PASS | contact_efield.mean_relative_error = 0.0073375 dimensionless [0.005293, 0.009647]_95% (threshold < 0.05, interval-strict); contact_efield.max_relative_error = 0.422439 dimensionless |
| self_convergence | yes | PASS | contact.self_convergence_order = 2.26347 dimensionless (threshold > 1.5) |

## Baselines run

_none run_ — no baseline, no claim.

## Preregistered acceptance thresholds

- `min_contact_ratio`: 0.95
- `relative_tol`: 0.05
- `expected_order`: 1.5

## Explicit non-goals

- These checks establish code correctness only. Numerical correctness is necessary, never sufficient: agreement with recorded signals is stronger, held-out perturbation stronger still (thesis §0.2).

## Notes

- Contact geometry is what tms-robotics uses. A PASS here still licenses no claim about target engagement, network effect or clinical utility.
- AUDIT -- why grading is required, not merely preferable. On UNIFORM meshes at this geometry the errors are 1.0610 (80 panels, ratio 4.04), 1.5061 (320 panels, ratio 3.26), 0.1709 (1280 panels, ratio 1.90), 0.0418 (5120 panels, ratio 1.01). Monotone under refinement: False. Refining one level makes the answer WORSE at the coarse end, so a user watching the error would have no way to distinguish that from convergence. scwbd.intervene.tms.efield refuses beyond panel_to_standoff = 1.0 for this reason.
- The N8 contract says the N6 spectral reference does not extend to contact, since its bound at a/R_c = 0.955 exceeds the solver error. That is correct for the GENERAL reference (degree 48, O(L^2) basis): its bound there is 1.10e-1. It is not correct for the reference used here. Rotating a single source onto the axis makes the Neumann data exactly azimuthal order one, so the basis is O(L) instead of O(L^2) and degree 400 is cheap; the bound becomes 1.03e-08 and the measured agreement with the closed form 1.2e-14. The contact regime is therefore validatable against an INDEPENDENT reference, not only by self-consistency. Suggest amending the N8 docstring.
- Both branches of the contract are supplied: an independent contact reference AND a Richardson self-convergence study. The self-convergence subcheck proves the discretisation converges to something; the reference subcheck is what says it converges to the right thing.
- Tolerance provenance: 0.05 is the repo's standing numerics tolerance, identical to N3/N4/N6. Disclosure, since this gate asks for preregistration: the graded refinement study was measured while scoping whether contact was reachable at all, before N8's contract was written. The tolerance was not adjusted afterwards.
- A PASS here licenses no claim about target engagement, network effect or clinical utility, and no claim-bearing run has been made.
