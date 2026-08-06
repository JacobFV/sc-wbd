# N8_induced_efield_contact — COULD_NOT_RUN

**Claim.** The induced-field solver is validated in the CONTACT regime (a coil element at clinical standoff from the scalp, a/R_c >= 0.95) to a preregistered tolerance — the geometry the downstream targeting consumer actually uses.

**Falsified by (thesis).** no reference or self-convergence study achieves a defensible tolerance at contact geometry, or the solver's error there exceeds the preregistered tolerance

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 1996fba · 2026-08-06T08:38:08+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| contact_regime | yes | COULD_NOT_RUN | missing: induced-field solver at contact geometry (agent Faraday); either an independent contact-regime reference (e.g. boundary-integral with graded panels) or a Richardson self-convergence study of the solver under refinement; the N6 spectral reference does NOT extend here, since its series bound at a/R_c ~ 0.955 exceeds the solver error it would be measuring; the geometry ratio a/R_c, so the gate can confirm it was handed CONTACT geometry rather than a standoff case relabelled; a preregistered tolerance (chosen before seeing the error) |

## Blocking reasons

- contact_regime: could not run — missing: induced-field solver at contact geometry (agent Faraday); either an independent contact-regime reference (e.g. boundary-integral with graded panels) or a Richardson self-convergence study of the solver under refinement; the N6 spectral reference does NOT extend here, since its series bound at a/R_c ~ 0.955 exceeds the solver error it would be measuring; the geometry ratio a/R_c, so the gate can confirm it was handed CONTACT geometry rather than a standoff case relabelled; a preregistered tolerance (chosen before seeing the error)

## Baselines run

_none run_ — no baseline, no claim.

## Preregistered acceptance thresholds

- `min_contact_ratio`: 0.95
- `relative_tol`: None
- `expected_order`: 1.5

## Explicit non-goals

- These checks establish code correctness only. Numerical correctness is necessary, never sufficient: agreement with recorded signals is stronger, held-out perturbation stronger still (thesis §0.2).

## Notes

- Opened at agent J's request after agent Faraday disclosed the validity domain of the N6 reference. Visible and unrun beats implicit.
