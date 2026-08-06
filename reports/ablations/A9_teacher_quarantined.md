# A9_teacher_quarantined — COULD_NOT_RUN

**Claim.** Does the teacher/distillation term improve *measured* held-out prediction? (§11.4: when the quarantined report/teacher experiment is enabled: no teacher, matched generic features and smoothness, shuffled/mismatched report, and perception-versus-imagery domain-shift controls)

**Falsified by (thesis).** an equal-capacity, equal-compute control matches or exceeds the candidate, or the candidate wins only by smoothing away the effect of interest

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 1a35a9a · 2026-08-06T19:02:46+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| quarantine | yes | COULD_NOT_RUN | this ablation belongs to a quarantined experiment which is OFF by default (ARCHITECTURE.md rule 5: TRIBE v2 distillation stays off by default and is never a subject likelihood); pass enable_quarantined=True only under an explicit claim-manifest override |

## Blocking reasons

- quarantine: could not run — this ablation belongs to a quarantined experiment which is OFF by default (ARCHITECTURE.md rule 5: TRIBE v2 distillation stays off by default and is never a subject likelihood); pass enable_quarantined=True only under an explicit claim-manifest override

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
- `effect_retention_floor`: 0.5
- `mechanistic_claim`: False

## Explicit non-goals

- This gate does not claim a validated digital twin of any specific person.
- This gate does not claim that any admitted operator is neurally realized.
- No human stimulation protocol is implemented or implied: this is a deep-learning study on open data and there is no device command path.

## Notes

- Quarantined: off by default and never a subject likelihood. Teacher agreement alone is never the metric (Appendix D, 'Teacher/simulator domination').
