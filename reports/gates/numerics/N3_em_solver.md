# N3_em_solver — COULD_NOT_RUN

**Claim.** The electromagnetic solver reproduces a closed-form quasi-static reference, validated independently of any neural-response model.

**Falsified by (thesis).** relative error above tolerance against the analytic dipole solution

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git a8221f6 · 2026-08-06T06:22:53+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| em_solver | yes | COULD_NOT_RUN | no EM solver supplied (agent G scwbd.intervene / agent F lead fields); the field model is unvalidated and no E-field claim may be made |

## Blocking reasons

- em_solver: could not run — no EM solver supplied (agent G scwbd.intervene / agent F lead fields); the field model is unvalidated and no E-field claim may be made

## Baselines run

_none run_ — no baseline, no claim.

## Preregistered acceptance thresholds

- `relative_tol`: 0.05
- `sigma_S_per_m`: 0.33

## Explicit non-goals

- These checks establish code correctness only. Numerical correctness is necessary, never sufficient: agreement with recorded signals is stronger, held-out perturbation stronger still (thesis §0.2).
