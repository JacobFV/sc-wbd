# D07_connectome_prior_value — COULD_NOT_RUN

**Claim.** Connectome prior value is controlled for. Primary metric: Data efficiency, causal forecast, calibration and out-of-domain behavior.

**Falsified by (thesis).** The mandatory control (Randomized, distance-matched, dense, graph-only, local-only and soft-edge controls at matched parameter/compute budgets) shows the result survives only without it.

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 1e11e49 · 2026-08-06T06:11:53+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| graph_controls | yes | COULD_NOT_RUN | scwbd.anatomy.graph_controls unavailable (owner: agent C (anatomy)) — agent C has not landed the randomized / distance-matched / dense graph controls; agent J will not fabricate them, because the control is the experiment |
| inputs[0] | yes | COULD_NOT_RUN | missing: model_for_graph(adjacency) factory (agent E / agent I) |
| inputs[1] | yes | COULD_NOT_RUN | missing: anatomical adjacency (agent C) |
| inputs[2] | yes | COULD_NOT_RUN | missing: graph controls (agent C): dense, randomized, distance_matched |
| inputs[3] | yes | COULD_NOT_RUN | missing: train/test datasets |

## Blocking reasons

- graph_controls: could not run — scwbd.anatomy.graph_controls unavailable (owner: agent C (anatomy)) — agent C has not landed the randomized / distance-matched / dense graph controls; agent J will not fabricate them, because the control is the experiment
- inputs[0]: could not run — missing: model_for_graph(adjacency) factory (agent E / agent I)
- inputs[1]: could not run — missing: anatomical adjacency (agent C)
- inputs[2]: could not run — missing: graph controls (agent C): dense, randomized, distance_matched
- inputs[3]: could not run — missing: train/test datasets

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

- Delegated to gate G2; this Appendix D row and that gate are the same experiment and must not be double-counted as two pieces of evidence.
