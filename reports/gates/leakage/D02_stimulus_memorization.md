# D02_stimulus_memorization — COULD_NOT_RUN

**Claim.** Stimulus memorization is controlled for. Primary metric: Cross-stimulus neural/behavioral forecast and matched-feature baseline.

**Falsified by (thesis).** The mandatory control (Hold out stimuli, semantic families and temporal continuations separately from participant holdout) shows the result survives only without it.

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git a8221f6 · 2026-08-06T06:22:53+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| stimulus_holdout | yes | COULD_NOT_RUN | no records with stimulus_ids supplied; stimulus holdout is not verifiable |
| cross_stimulus_forecast | yes | COULD_NOT_RUN | model and/or matched-feature baseline not supplied; recognition gain on seen stimuli is not evidence for brain dynamics |

## Blocking reasons

- stimulus_holdout: could not run — no records with stimulus_ids supplied; stimulus holdout is not verifiable
- cross_stimulus_forecast: could not run — model and/or matched-feature baseline not supplied; recognition gain on seen stimuli is not evidence for brain dynamics

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
