# N1_compiler_correctness — COULD_NOT_RUN

**Claim.** The compiler produces a model whose shapes, units, frames, delays and masks are internally consistent.

**Falsified by (thesis).** any offset overlap, undeclared unit/frame/clock, negative or unrepresentable delay, or a mask that does not match the declared operator set

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git b32ca56 · 2026-08-06T05:59:59+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| compiled_model | yes | COULD_NOT_RUN | no CompiledModel supplied (agent A's scwbd.compiler.compile has not been run or has not landed); compiler correctness is unverified |

## Blocking reasons

- compiled_model: could not run — no CompiledModel supplied (agent A's scwbd.compiler.compile has not been run or has not landed); compiler correctness is unverified

## Baselines run

_none run_ — no baseline, no claim.

## Explicit non-goals

- These checks establish code correctness only. Numerical correctness is necessary, never sufficient: agreement with recorded signals is stronger, held-out perturbation stronger still (thesis §0.2).
