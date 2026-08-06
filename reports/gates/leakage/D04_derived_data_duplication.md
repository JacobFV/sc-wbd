# D04_derived_data_duplication — COULD_NOT_RUN

**Claim.** Derived-data duplication is controlled for. Primary metric: Hash/lineage audit and performance after deduplication.

**Falsified by (thesis).** The mandatory control (Keep raw scan and every tractogram, parcellation, preprocessing derivative or augmentation in one split) shows the result survives only without it.

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 1e11e49 · 2026-08-06T06:11:53+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| hash_lineage_audit | yes | COULD_NOT_RUN | no lineage records supplied |
| performance_after_dedup | yes | COULD_NOT_RUN | no with/without-duplicate scores supplied; the size of the inflation is unknown |

## Blocking reasons

- hash_lineage_audit: could not run — no lineage records supplied
- performance_after_dedup: could not run — no with/without-duplicate scores supplied; the size of the inflation is unknown

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

## Refusal fixtures exercised

- `R10`

## Explicit non-goals

- This gate does not claim a validated digital twin of any specific person.
- This gate does not claim that any admitted operator is neurally realized.
- No prospective human TMS/tFUS protocol is implemented or implied (build order stops at item 5; item 6 is out of scope: no IRB, no consent, no participants).
