# N7_instrument_discrimination — PASS

**Claim.** Every guard and provenance field this bench relies on has an input under which it reads differently, so a green reading is evidence rather than decoration.

**Falsified by (thesis).** an instrument that returns the same reading on every input it was given — it is structurally incapable of reporting the discrimination it is consulted for

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git c0e5833 · 2026-08-06T09:54:47+00:00*

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| source_dirty_flag | yes | PASS | source_dirty_flag.distinct_reads = 2 dimensionless (threshold > 1.5) |
| capacity_matching | yes | PASS | capacity_matching.distinct_reads = 2 dimensionless (threshold > 1.5) |
| interval_strict_threshold | yes | PASS | interval_strict_threshold.distinct_reads = 2 dimensionless (threshold > 1.5) |
| smoothing_check | yes | PASS | smoothing_check.distinct_reads = 2 dimensionless (threshold > 1.5) |
| report_provenance_rule | yes | PASS | report_provenance_rule.distinct_reads = 2 dimensionless (threshold > 1.5) |

## Baselines run

_none run_ — no baseline, no claim.

## Preregistered acceptance thresholds

- `min_distinct_reads`: 2

## Explicit non-goals

- This audit does not check that an instrument is CORRECT, only that it is capable of varying. A field that varies can still be wrong.

## Notes

- A green reading from an instrument that cannot vary is not evidence. Four such instruments have already been found in this project; the fourth was inside the mechanism built to catch stale artifacts.
- This audit checks capability to vary, not correctness. An instrument that varies can still be measuring the wrong thing.
