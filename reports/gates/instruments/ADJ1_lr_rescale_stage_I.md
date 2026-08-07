# ADJ1_lr_rescale_stage_I — COULD_NOT_RUN

**Claim.** The directed change improved the run on its pre-committed metric. Decision: A mid-run learning-rate rescale: rates were written for batch 192, batch was then cut to 64 for device memory, and the rates were rescaled to match. The scaling mismatch was real; what is under review is whether acting on it mid-run helped.

**Falsified by (thesis).** the pre-committed metric shows no improvement outside the preregistered equivalence margin, or shows harm

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git c0e5833 · 2026-08-06T09:54:47+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| stage_I_series | yes | COULD_NOT_RUN | the run has not reached end of Stage I; agent Turing supplies the baseline and rescaled series on sim_forecast_nll and bench returns the verdict. Preregistered here BEFORE the data exists — this file's commit precedes the run that produces it. |

## Blocking reasons

- stage_I_series: could not run — the run has not reached end of Stage I; agent Turing supplies the baseline and rescaled series on sim_forecast_nll and bench returns the verdict. Preregistered here BEFORE the data exists — this file's commit precedes the run that produces it.

## Baselines run

_none run_ — no baseline, no claim.

## Preregistered acceptance thresholds

- `preregistered_metric`: sim_forecast_nll (end of Stage I)
- `equivalence_margin_relative`: 0.01
- `secondary_metrics_may_change_the_verdict`: False
- `sub_grid_timing_claims_permitted`: False

## Explicit non-goals

- This adjudication does not evaluate SC-WBD-001-beta. It evaluates a decision.
- A neutral or negative verdict is a process finding, not an artifact defect.

## Notes

- The decision now rests ONLY on the a-priori scaling argument: both pieces of outcome evidence originally offered for it were withdrawn by the party that offered them, unprompted. That withdrawal is to their credit and is recorded as such.
- Pre-committed metric: sim_forecast_nll at end of Stage I, NOT composite loss during warmup. Committed by agent Turing while it did not favour them.
