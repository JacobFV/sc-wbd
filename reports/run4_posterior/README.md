# Picking run 4's posterior learning rate by measurement

ISSUE-012 found that run 3's amortised posterior returns the prior: `posterior_r2`
≤ 0.001 on all six parameters, posterior sd within 3% of the prior's, and
shuffling the conditioning across the batch moving `−log q` by 0.001–0.003 nats.
It also found that an MLP on the very vector the flow conditions on reaches
**0.753** on `log_G`, so the information is there and the flow was not taking it.

The issue names four remedies and says the retrain is **one stage, no whole-model
run needed**. `scripts/sweep_posterior.py` is that stage. It trains the posterior
alone — summary encoder and flow — against the same objective, the same
35%-masked context and the same corpus `sim_losses` uses. A full run is ~38 h;
this is ~13 min per cell.

## Why training the posterior alone is faithful, not an approximation

The posterior conditions on `batch["activity"]` and never on the dynamics
model's output (`train.py:1386`). Its conditioning distribution therefore does
**not** shift as the model trains, so a standalone stage sees the same problem
T4 sets it. The sweep also re-draws the slice mask every step, which is the one
thing `POSTERIOR_LR_SCALE = 0.02` was actually written against — "a density
chasing a conditioning distribution that changes every batch".

## The result: the learning rate was the defect

1,500 steps per cell, held-out trajectories, `log_G`.

| cond_norm | lr | log_G R² | −log q | posterior sd / prior sd |
|---|---:|---:|---:|---:|
| `layer_v1` | 4.0e-6 | +0.001 | 7.815 | 1.031 |
| `layer_v1` | 2.0e-4 | +0.615 | 6.135 | 0.521 |
| `layer_v1` | 1.0e-3 | +0.800 | 5.324 | 0.382 |
| `dataset_std_v2` | 4.0e-6 | +0.027 | 7.729 | 1.019 |
| `dataset_std_v2` | 2.0e-4 | +0.707 | 5.995 | 0.483 |
| `dataset_std_v2` | 1.0e-3 | **+0.834** | 5.268 | 0.353 |

Read three ways:

* **The control reproduces run 3.** `layer_v1` at 4.0e-6 is run 3's exact
  setting — 0.02 × T4's 2.0e-4 — and returns R² +0.001 at a posterior sd
  **1.031×** the prior's. ISSUE-012 measured run 3's checkpoint at −0.010 and
  1.024. Two independent statistics agreeing to within 0.01 and 0.7% is what
  makes the rest of this table evidence rather than a different experiment.
* **The learning rate is the first-order cause.** 4.0e-6 → 1.0e-3 is 250×, and
  it takes `log_G` from nothing to 0.800–0.834 — past ISSUE-012's 0.4 floor
  twice over, past the ridge probe (0.439), and past the MLP probe (0.753) that
  the issue set as the bar the machinery had to clear to justify its 1.1M
  parameters.
* **The conditioning fix is real but second-order.** `dataset_std_v2` beats
  `layer_v1` by +0.092 at 2.0e-4 and +0.034 at 1.0e-3, consistent in direction
  wherever learning happens at all. At 300 steps it looked like a wash (0.323 vs
  0.332) and an earlier note here said it did "essentially nothing" — that was
  read off the short smoke and is wrong at 1,500 steps.

The width column is the cleanest single statement of the change: the posterior
goes from **1.03× the prior** (it *is* the prior) to **0.35×**, i.e. about 2.8×
sharper, with R² rising alongside rather than the width collapsing on its own.

## What this does NOT establish: calibration

**No cell here discharges ISSUE-012.** The issue requires `log_G` R² > 0.4 *and*
`sbc_ks_pvalue_min` > 0.01 *and* `coverage_mae` < 0.05 — "recovering calibration
by widening back to the prior does not discharge it", and the converse binds
equally.

Every cell above reports `sbc_ks_pvalue_min` 0.000, **including the control that
otherwise reproduces run 3**, where run 3 published 0.0976. Two of three
statistics matching and the third not pins the disagreement to the eval
population, not the posterior. The cause is the sampling:
`evaluate.posterior_calibration` draws its 512 datasets **backend-stratified**;
this sweep took the first 512 in file order, which is one or two shards, and SBC
ranks are far more sensitive to a homogeneous sample than R² is.

Two fixes were applied in sequence and only the second was the real one:

1. 128 SBC bins and a masked eval context → 256 bins and unmasked, matching
   `posterior_calibration`. **This did not move `ks_min`.** The first version of
   this file blamed the bin count; that was wrong, and the control cell is what
   showed it.
2. the validation loader is now shuffled at a fixed seed so it spans shards.
   That is still not full backend stratification, so **the calibration column
   remains comparable only to itself.**

The calibration leg of ISSUE-012 is therefore decided by run 4's own
`posterior_calibration`, not here. Run 4 is launched knowing the LR delivers the
information and **not** knowing it preserves the calibration — which is stated
in the model card rather than discovered afterwards.

## Files

* `sweep.json` — the current grid, machine-readable.
* `sweep_128bin_maskedeval_SUPERSEDED.json` — the 6-cell grid above, kept because
  it is where the control-reproduces-run-3 result was measured, and superseded
  only in its calibration column.
* `_smoke/` — a 300-step plumbing check. Not evidence; its R² is short-horizon.
