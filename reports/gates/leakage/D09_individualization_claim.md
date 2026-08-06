# D09_individualization_claim — COULD_NOT_RUN

**Claim.** Individualization claim is controlled for. Primary metric: Incremental log score, calibration, decision utility and drift.

**Falsified by (thesis).** The mandatory control (Population, session-adapted, anatomy-only and longitudinal-person models; new session and novel task/intervention holdouts) shows the result survives only without it.

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 19c4acc · 2026-08-06T08:19:51+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| inputs[0] | yes | COULD_NOT_RUN | missing: individualized candidate model |
| inputs[1] | yes | COULD_NOT_RUN | missing: training set |
| inputs[2] | yes | COULD_NOT_RUN | missing: new-session holdout (the claim is about future prediction) |
| inputs[3] | yes | COULD_NOT_RUN | missing: unseen-task/intervention holdout |
| inputs[4] | yes | COULD_NOT_RUN | missing: mandatory baseline 'population' |
| inputs[5] | yes | COULD_NOT_RUN | missing: mandatory baseline 'anatomy_only' |
| inputs[6] | yes | COULD_NOT_RUN | missing: mandatory baseline 'session_adapted' |

## Blocking reasons

- inputs[0]: could not run — missing: individualized candidate model
- inputs[1]: could not run — missing: training set
- inputs[2]: could not run — missing: new-session holdout (the claim is about future prediction)
- inputs[3]: could not run — missing: unseen-task/intervention holdout
- inputs[4]: could not run — missing: mandatory baseline 'population'
- inputs[5]: could not run — missing: mandatory baseline 'anatomy_only'
- inputs[6]: could not run — missing: mandatory baseline 'session_adapted'

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

## Notes

- Delegated to gate G5; this Appendix D row and that gate are the same experiment and must not be double-counted as two pieces of evidence.
