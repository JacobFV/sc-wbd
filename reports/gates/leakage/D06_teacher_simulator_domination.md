# D06_teacher_simulator_domination — COULD_NOT_RUN

**Claim.** Teacher/simulator domination is controlled for. Primary metric: Measured held-out data likelihood and calibration, never teacher agreement alone.

**Falsified by (thesis).** The mandatory control (No-teacher/no-simulator, generic-feature, shuffled, parameter-perturbed and empirical-only ablations) shows the result survives only without it.

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 19c4acc · 2026-08-06T08:19:51+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| quarantine | yes | COULD_NOT_RUN | TRIBE v2 distillation stays OFF by default and is never a subject likelihood (ARCHITECTURE.md rule 5). With the teacher disabled there is no distillation contribution to audit, and none may be claimed. |

## Blocking reasons

- quarantine: could not run — TRIBE v2 distillation stays OFF by default and is never a subject likelihood (ARCHITECTURE.md rule 5). With the teacher disabled there is no distillation contribution to audit, and none may be claimed.

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

- Teacher agreement is never the metric; only measured held-out data likelihood and calibration count (Appendix D).
