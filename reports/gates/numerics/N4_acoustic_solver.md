# N4_acoustic_solver — COULD_NOT_RUN

**Claim.** The acoustic solver reproduces free-field spreading and satisfies the Helmholtz equation, validated independently of any neural-response model.

**Falsified by (thesis).** relative error above tolerance, or a Helmholtz residual that does not vanish under grid refinement

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 1e11e49 · 2026-08-06T06:11:53+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| acoustic_solver | yes | COULD_NOT_RUN | no acoustic solver supplied (agent G scwbd.intervene); the exposure model is unvalidated and no tFUS claim may be made |

## Blocking reasons

- acoustic_solver: could not run — no acoustic solver supplied (agent G scwbd.intervene); the exposure model is unvalidated and no tFUS claim may be made

## Baselines run

_none run_ — no baseline, no claim.

## Preregistered acceptance thresholds

- `relative_tol`: 0.05
- `wavenumber_per_m`: 100.0

## Explicit non-goals

- These checks establish code correctness only. Numerical correctness is necessary, never sufficient: agreement with recorded signals is stronger, held-out perturbation stronger still (thesis §0.2).
