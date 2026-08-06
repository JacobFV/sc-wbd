# D05_scale_hallucination — COULD_NOT_RUN

**Claim.** Scale hallucination is controlled for. Primary metric: Coverage and error at each native scale; high-frequency energy calibration.

**Falsified by (thesis).** The mandatory control (Withhold fine-scale evidence while retaining coarse data; compare uncertainty and reconstruction to a coarse-only model) shows the result survives only without it.

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git b32ca56 · 2026-08-06T05:59:59+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| inputs[0] | yes | COULD_NOT_RUN | missing: multiresolution candidate (agent E/I) |
| inputs[1] | yes | COULD_NOT_RUN | missing: coarse-only baseline |
| inputs[2] | yes | COULD_NOT_RUN | missing: fine-scale train/test datasets |
| inputs[3] | yes | COULD_NOT_RUN | missing: coarse-scale evaluation set (boundary observable) |
| inputs[4] | yes | COULD_NOT_RUN | missing: restriction map R (agent D transforms / agent C parcellation) |

## Blocking reasons

- inputs[0]: could not run — missing: multiresolution candidate (agent E/I)
- inputs[1]: could not run — missing: coarse-only baseline
- inputs[2]: could not run — missing: fine-scale train/test datasets
- inputs[3]: could not run — missing: coarse-scale evaluation set (boundary observable)
- inputs[4]: could not run — missing: restriction map R (agent D transforms / agent C parcellation)

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

- Delegated to gate G3; this Appendix D row and that gate are the same experiment and must not be double-counted as two pieces of evidence.
