# A8_language_coupling — COULD_NOT_RUN

**Claim.** Does coupling language to neural/bodily/memory/action predictions help? (§11.4: language-only behavioural imitation versus a language process coupled to neural, bodily, memory, and action predictions)

**Falsified by (thesis).** an equal-capacity, equal-compute control matches or exceeds the candidate, or the candidate wins only by smoothing away the effect of interest

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git b32ca56 · 2026-08-06T05:59:59+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| arms[0] | yes | COULD_NOT_RUN | missing: coupled_language_process; §11.4 names it explicitly, so the comparison cannot be declared complete without it |
| arms[1] | yes | COULD_NOT_RUN | missing: language_only_imitation; §11.4 names it explicitly, so the comparison cannot be declared complete without it |
| arms[2] | yes | COULD_NOT_RUN | missing: train/test datasets; §11.4 names it explicitly, so the comparison cannot be declared complete without it |

## Blocking reasons

- arms[0]: could not run — missing: coupled_language_process; §11.4 names it explicitly, so the comparison cannot be declared complete without it
- arms[1]: could not run — missing: language_only_imitation; §11.4 names it explicitly, so the comparison cannot be declared complete without it
- arms[2]: could not run — missing: train/test datasets; §11.4 names it explicitly, so the comparison cannot be declared complete without it

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
- No prospective human TMS/tFUS protocol is implemented or implied (build order stops at item 5; item 6 is out of scope: no IRB, no consent, no participants).
