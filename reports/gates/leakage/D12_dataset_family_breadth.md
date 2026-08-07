# D12_dataset_family_breadth — COULD_NOT_RUN

**Claim.** Dataset-family breadth is controlled for. Primary metric: Per-family contribution, negative transfer, subgroup worst case and uncertainty coverage.

**Falsified by (thesis).** The mandatory control (Report performance by empirical, boundary-only, calibration, synthetic and evaluation-only source roles; remove each family in turn) shows the result survives only without it.

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git c0e5833 · 2026-08-06T09:52:25+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| per_family_contribution | yes | COULD_NOT_RUN | no families / model factory / datasets supplied; a longer source list is not evidence, so nothing may be claimed about breadth |

## Blocking reasons

- per_family_contribution: could not run — no families / model factory / datasets supplied; a longer source list is not evidence, so nothing may be claimed about breadth

## Baselines run

_none run_ — no baseline, no claim.

## Preregistered acceptance thresholds

- `min_delta_log_score`: 0.0
- `max_coverage_error`: 0.05
- `max_overconfidence_increase`: 0.02
- `max_delay_rel_error`: 0.15
- `boundary_rel_tol`: 0.05
- `max_hallucination_index`: 1.25
- `min_uncertainty_inflation`: 1.05
- `min_fisher_eig_gain`: 1.1
- `capacity_tol`: 0.1
- `min_model_discrimination`: 0.05
- `n_boot`: 1000

## Explicit non-goals

- This gate does not claim a validated digital twin of any specific person.
- This gate does not claim that any admitted operator is neurally realized.
- No prospective human TMS/tFUS protocol is implemented or implied (build order stops at item 5; item 6 is out of scope: no IRB, no consent, no participants).
