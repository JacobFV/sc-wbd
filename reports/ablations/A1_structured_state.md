# A1_structured_state — COULD_NOT_RUN

**Claim.** Does structured regional state predict anything a pooled scalar cannot? (§11.4: structured regional state versus one scalar or pooled vector per region)

**Falsified by (thesis).** an equal-capacity, equal-compute control matches or exceeds the candidate, or the candidate wins only by smoothing away the effect of interest

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 1a35a9a · 2026-08-06T19:02:46+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| arms[0] | yes | COULD_NOT_RUN | missing: structured_state; §11.4 names it explicitly, so the comparison cannot be declared complete without it |
| arms[1] | yes | COULD_NOT_RUN | missing: scalar_per_region; §11.4 names it explicitly, so the comparison cannot be declared complete without it |
| arms[2] | yes | COULD_NOT_RUN | missing: pooled_vector_per_region; §11.4 names it explicitly, so the comparison cannot be declared complete without it |
| arms[3] | yes | COULD_NOT_RUN | missing: pooled_vector_per_region@param_matched; §11.4 names it explicitly, so the comparison cannot be declared complete without it |
| arms[4] | yes | COULD_NOT_RUN | missing: pooled_vector_per_region@state_matched; §11.4 names it explicitly, so the comparison cannot be declared complete without it |
| arms[5] | yes | COULD_NOT_RUN | missing: theta_conditioned_pooled; §11.4 names it explicitly, so the comparison cannot be declared complete without it |
| arms[6] | yes | COULD_NOT_RUN | missing: permuted_family_state; §11.4 names it explicitly, so the comparison cannot be declared complete without it |
| arms[7] | yes | COULD_NOT_RUN | missing: train/test datasets; §11.4 names it explicitly, so the comparison cannot be declared complete without it |

## Blocking reasons

- arms[0]: could not run — missing: structured_state; §11.4 names it explicitly, so the comparison cannot be declared complete without it
- arms[1]: could not run — missing: scalar_per_region; §11.4 names it explicitly, so the comparison cannot be declared complete without it
- arms[2]: could not run — missing: pooled_vector_per_region; §11.4 names it explicitly, so the comparison cannot be declared complete without it
- arms[3]: could not run — missing: pooled_vector_per_region@param_matched; §11.4 names it explicitly, so the comparison cannot be declared complete without it
- arms[4]: could not run — missing: pooled_vector_per_region@state_matched; §11.4 names it explicitly, so the comparison cannot be declared complete without it
- arms[5]: could not run — missing: theta_conditioned_pooled; §11.4 names it explicitly, so the comparison cannot be declared complete without it
- arms[6]: could not run — missing: permuted_family_state; §11.4 names it explicitly, so the comparison cannot be declared complete without it
- arms[7]: could not run — missing: train/test datasets; §11.4 names it explicitly, so the comparison cannot be declared complete without it

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
- `mechanistic_claim`: True

## Explicit non-goals

- This gate does not claim a validated digital twin of any specific person.
- This gate does not claim that any admitted operator is neurally realized.
- No human stimulation protocol is implemented or implied: this is a deep-learning study on open data and there is no device command path.

## Notes

- TWO_CAPACITY_MATCHED_POOLED_CONTROLS_param_matched_AND_state_matched_BOTH_MUST_BE_BEATEN; PERMUTED_FAMILY_ARM_MANDATORY_FOR_ATTRIBUTION; PRIMARY_PAIRED_PARTICIPANT_CLUSTERED_NLL_PLUS_COPRIMARY_MSE_BOTH_INTERVALLED; NLL_WIN_WITHOUT_MSE_WIN_GRANTS_NO_MECHANISTIC_CLAIM; EFFECT_IS_A1_EFFECT_BETWEEN_REGION_DISPERSION_NOT_default_effect; SCORED_AS_EMITTED_AND_CALIBRATION_MATCHED_DISAGREEMENT_CLAIMS_NEITHER; SYSTEMATIC_ENVELOPE_GE_DELTA_OR_SEED_RANGE_GE_DELTA_IS_INCONCLUSIVE; V_ABLATION_AND_V_CLAIM_ARE_SEPARATE_11_2_FLOOR_BOUNDS_ONLY_V_CLAIM; RUN1_IS_A_CONTROL_CLASS_ARTIFACT_NOT_RUN2S_CONTROL_ARM; THETA_CONDITIONED_POOLED_ARM_MANDATORY_STAGE2_CONDITIONING_CONTROL; A1_VARIES_STATE_ONLY_OPERATOR_ASSIGNMENT_HELD_IDENTICAL_A5_IS_THE_OTHER_ONE; LICENCE_IS_TWO_FAMILY_CORTICAL_PARTITION_NOT_OPERATOR_VALUED_HETEROGENEITY; SUBCORTICAL_FAMILIES_N2_ARE_OUT_OF_CLAIM; MDE_AT_27_PARTICIPANTS_IS_0.1404_NATS
