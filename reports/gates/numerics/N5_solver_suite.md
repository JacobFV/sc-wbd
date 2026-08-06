# N5_solver_suite — COULD_NOT_RUN

**Claim.** Solvers converge at their advertised order, remain stable, conserve their declared invariants, and are bitwise reproducible for a fixed seed.

**Falsified by (thesis).** an observed order below the advertised order, non-finite or unbounded state, invariant drift beyond tolerance, or non-determinism at a fixed seed

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git c0e5833 · 2026-08-06T09:52:25+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| solver_convergence | yes | COULD_NOT_RUN | no solver supplied (agent E dynamics / agent G field solvers) |
| solver_stability | yes | COULD_NOT_RUN | no trajectory supplied |
| conservation | yes | COULD_NOT_RUN | no trajectory or no invariant function supplied; a module that declares no invariant cannot claim a conservation property |
| seed_reproducibility | yes | COULD_NOT_RUN | no stochastic entry point supplied |

## Blocking reasons

- solver_convergence: could not run — no solver supplied (agent E dynamics / agent G field solvers)
- solver_stability: could not run — no trajectory supplied
- conservation: could not run — no trajectory or no invariant function supplied; a module that declares no invariant cannot claim a conservation property
- seed_reproducibility: could not run — no stochastic entry point supplied

## Baselines run

_none run_ — no baseline, no claim.

## Preregistered acceptance thresholds

- `expected_order`: 1.0
- `dts`: [0.02, 0.01, 0.005, 0.0025]

## Explicit non-goals

- These checks establish code correctness only. Numerical correctness is necessary, never sufficient: agreement with recorded signals is stronger, held-out perturbation stronger still (thesis §0.2).
