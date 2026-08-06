# N2_boundary_consistency — COULD_NOT_RUN

**Claim.** Fine and coarse regional backends agree within the declared tolerance on boundary observables, so adaptive resolution may be used for inference.

**Falsified by (thesis).** disagreement beyond the declared tolerance, or a backend that cannot produce the boundary observable at all

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git b32ca56 · 2026-08-06T05:59:59+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| boundary_consistency | yes | COULD_NOT_RUN | the fine and/or coarse boundary observable was not supplied by the backends (agent E dynamics / agent D restriction maps); adaptive resolution may not be used for inference until both produce it (§11.1) |

## Blocking reasons

- boundary_consistency: could not run — the fine and/or coarse boundary observable was not supplied by the backends (agent E dynamics / agent D restriction maps); adaptive resolution may not be used for inference until both produce it (§11.1)

## Baselines run

_none run_ — no baseline, no claim.

## Preregistered acceptance thresholds

- `boundary_rel_tol`: 0.05

## Explicit non-goals

- These checks establish code correctness only. Numerical correctness is necessary, never sufficient: agreement with recorded signals is stronger, held-out perturbation stronger still (thesis §0.2).
