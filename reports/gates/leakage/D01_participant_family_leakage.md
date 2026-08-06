# D01_participant_family_leakage — COULD_NOT_RUN

**Claim.** Participant or family leakage is controlled for. Primary metric: Held-out-person likelihood, calibration and retrieval/leakage audit.

**Falsified by (thesis).** The mandatory control (Group all sessions, derivatives, relatives and duplicate archive records before splitting) shows the result survives only without it.

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 4d617af · 2026-08-06T07:56:41+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| grouped_split | yes | COULD_NOT_RUN | no lineage records supplied; grouping cannot be verified and R10 forbids splitting with unresolved parentage |
| held_out_person_likelihood | yes | COULD_NOT_RUN | no model/train/test supplied; within-session prediction cannot substitute for held-out-person generalization |
| retrieval_audit | yes | COULD_NOT_RUN | no embeddings supplied; near-duplicate archive records (the same scan under two accession numbers) cannot be detected by lineage alone |

## Blocking reasons

- grouped_split: could not run — no lineage records supplied; grouping cannot be verified and R10 forbids splitting with unresolved parentage
- held_out_person_likelihood: could not run — no model/train/test supplied; within-session prediction cannot substitute for held-out-person generalization
- retrieval_audit: could not run — no embeddings supplied; near-duplicate archive records (the same scan under two accession numbers) cannot be detected by lineage alone

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

- `R10 (derived records crossing a parent-level holdout)`

## Explicit non-goals

- This gate does not claim a validated digital twin of any specific person.
- This gate does not claim that any admitted operator is neurally realized.
- No prospective human TMS/tFUS protocol is implemented or implied (build order stops at item 5; item 6 is out of scope: no IRB, no consent, no participants).

## Notes

- Splitting is agent B's; this audit consumes it rather than reimplementing it.
