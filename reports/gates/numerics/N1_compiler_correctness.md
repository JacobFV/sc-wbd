# N1_compiler_correctness — PASS

**Claim.** The compiler produces a model whose shapes, units, frames, clocks, delays, masks and gradient permissions are internally consistent, and whose recorded claim class is the one it was compiled for.

**Falsified by (thesis).** any offset overlap or gap, an undeclared unit/frame/clock, a negative or unrepresentable delay, a mask that disagrees with the dispatched operator set, a gradient permission naming a module that does not exist, an unbacked bias term, or a silently demoted claim class

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 1e11e49 · 2026-08-06T06:11:53+00:00*

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| state_layout | yes | PASS | layout.overlaps = 0 dimensionless (threshold < 0.5); layout.gaps = 0 dimensionless (threshold < 0.5); layout.byte_view_mismatches = 0 dimensionless (threshold < 0.5); layout.total_bytes_consistent = 1 dimensionless (threshold > 0.5) |
| units_frames_clocks | yes | PASS | ports.missing_units = 0 dimensionless (threshold < 0.5); ports.missing_clock = 0 dimensionless (threshold < 0.5); clocks.unknown_referenced = 0 dimensionless (threshold < 0.5); clocks.unverified_or_orphaned = 0 dimensionless (threshold < 0.5); frames.unreachable_from_root = 0 dimensionless (threshold < 0.5); state.n_blocks = 9 dimensionless |
| delays | yes | PASS | delays.negative = 0 dimensionless (threshold < 0.5); delays.nonfinite = 0 dimensionless (threshold < 0.5); delays.below_base_dt = 0 dimensionless (threshold < 0.5); delays.beyond_hyperperiod = 0 dimensionless (threshold < 0.5) |
| masks | yes | PASS | masks.consistent_shape = 1 dimensionless (threshold > 0.5); masks.nonbinary_blocks = 0 dimensionless (threshold < 0.5); masks.dispatched_edges_not_masked = 0 dimensionless (threshold < 0.5); masks.masked_edges_not_dispatched = 0 dimensionless (threshold < 0.5) |
| gradient_permissions | yes | PASS | gradient.sources_without_a_mask = 0 dimensionless (threshold < 0.5); gradient.unmatched_permission_patterns = 0 dimensionless (threshold < 0.5); gradient.unreachable_parameter_groups = 9 dimensionless |
| uncertainty_ledger | yes | PASS | ledger.unbacked_bias_terms = 0 dimensionless (threshold < 0.5); ledger.prior_specified_sensitivity_terms = 12 dimensionless |
| claim_class_integrity | yes | PASS | claim.was_demoted = 0 dimensionless (threshold < 0.5); claim.overridden_refusals = 0 dimensionless (threshold < 0.5); claim.refusal_checks_passed = 11 dimensionless |

## Baselines run

_none run_ — no baseline, no claim.

## Explicit non-goals

- These checks establish code correctness only. Numerical correctness is necessary, never sufficient: agreement with recorded signals is stronger, held-out perturbation stronger still (thesis §0.2).

## Notes

- Subject of this check: reference example: scwbd.schema.examples.three_region.
- A PASS here means the compiler emits an internally consistent artifact for this subject. It is not evidence about any other schema, and it is not evidence that any compiled operator is neurally realized.
