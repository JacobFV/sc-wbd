# N6_induced_efield — COULD_NOT_RUN

**Claim.** The magnetically induced E-field solver reproduces the closed-form (Sarvas / Heller-van Hulsteyn) solution for a spherically symmetric conductor, validated independently of any neural-response model.

**Falsified by (thesis).** relative error above tolerance against the closed form, or a mesh-refinement study that does not converge at the advertised order

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 19c4acc · 2026-08-06T08:21:17+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| induced_efield | yes | COULD_NOT_RUN | missing: induced-field solver (agent Faraday: scwbd.intervene.tms.efield); closed-form reference (Sarvas / Heller-van Hulsteyn); agent J does not implement induction physics and will not substitute the conduction reference from N3, which is a different problem |

## Blocking reasons

- induced_efield: could not run — missing: induced-field solver (agent Faraday: scwbd.intervene.tms.efield); closed-form reference (Sarvas / Heller-van Hulsteyn); agent J does not implement induction physics and will not substitute the conduction reference from N3, which is a different problem

## Baselines run

_none run_ — no baseline, no claim.

## Preregistered acceptance thresholds

- `relative_tol`: 0.05
- `expected_order`: 1.5

## Explicit non-goals

- These checks establish code correctness only. Numerical correctness is necessary, never sufficient: agreement with recorded signals is stronger, held-out perturbation stronger still (thesis §0.2).

## Notes

- Opened because N3 passed for CONDUCTION only. Any claim that depends on the induced TMS field remains suspended until this gate runs.
