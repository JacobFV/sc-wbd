# D10_tms_tfus_decision_claim — COULD_NOT_RUN

**Claim.** TMS/tFUS decision claim is controlled for. Primary metric: Directional response, dose--response, benefit/risk and decision regret.

**Falsified by (thesis).** The mandatory control (Prospective randomized or otherwise causally identified target/protocol comparison with field, pose, state and sham records) shows the result survives only without it.

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 1e11e49 · 2026-08-06T06:11:53+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| prospective_decision_comparison | yes | COULD_NOT_RUN | OUT OF SCOPE BY CONSTRUCTION: the build order stops at item 5 (empirical subsystem); item 6 (prospective human TMS/tFUS) has no IRB, no consent and no participants, and no agent may implement a human stimulation protocol (ARCHITECTURE.md §0). No inputs can make this audit run in SC-WBD-001-beta. |

## Blocking reasons

- prospective_decision_comparison: could not run — OUT OF SCOPE BY CONSTRUCTION: the build order stops at item 5 (empirical subsystem); item 6 (prospective human TMS/tFUS) has no IRB, no consent and no participants, and no agent may implement a human stimulation protocol (ARCHITECTURE.md §0). No inputs can make this audit run in SC-WBD-001-beta.

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

- `R11 (intervention optimization outside a validated A_safe)`

## Explicit non-goals

- This gate does not claim a validated digital twin of any specific person.
- This gate does not claim that any admitted operator is neurally realized.
- No prospective human TMS/tFUS protocol is implemented or implied (build order stops at item 5; item 6 is out of scope: no IRB, no consent, no participants).

## Notes

- Offline reconstruction supports target hypotheses, not wellness or treatment efficacy. Any downstream consumer (tms-robotics) must treat SC-WBD output as a prediction plus a refusal, never as a protocol.
