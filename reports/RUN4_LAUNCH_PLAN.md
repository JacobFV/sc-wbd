# Run 4: what is decided, what is not, and the rule that decides the rest

Written 2026-08-10 ~06:30, before launch, at the user's request: *"write out all
your concerns to address before the restart, before the training … I'd prefer to
commit to a plan so that we don't have to constantly be stressed about values."*

The point of this file is to **pre-register** the decisions so a number cannot be
re-litigated after it is seen. Where a value is still open, the RULE that will
set it is written down here first.

---

## 1. The honest problem with tonight, named

The posterior sweep's headline number moved three times:

| what changed | log_G R² at `layer_v1` 1.0e-3 |
| --- | --- |
| 300-step smoke | +0.332 |
| 1500 steps, 128 SBC bins, masked eval, val = first 512 in file order | **+0.800** |
| 1500 steps, 256 bins, unmasked eval, val shuffled across shards | **+0.644** |

Every change was justified and each is recorded in the commit that made it. But
the pattern is the problem: **I was tuning the instrument while reading it.**
That is how a number becomes something to be anxious about instead of something
measured. The fix is not to be more careful; it is to freeze the protocol and
pre-register the decision rule. Both are done below.

This is a methodology defect, not a crisis. Nothing wrong has been published:
the only artifact that carried the inflated figure was `reports/run4_posterior/README.md`,
superseded in `0eb0fca` within the hour, and the model card and site were never
touched.

## 2. The measurement protocol, FROZEN

No further changes to this without an explicit line saying which previously
reported number is superseded.

* 1,500 optimiser steps per cell, posterior only, AdamW + OneCycleLR, grad-clip 1.0.
* **Train** on the masked context (`ctx * slice_mask`, p_observed 0.65) — what
  `sim_losses` does.
* **Evaluate** on the unmasked context at **256 SBC bins** — what
  `evaluate.posterior_calibration` does.
* Validation set: `trajectory_subset="val"`, `val_fraction=0.1`, **shuffled at a
  fixed seed** so it spans shards. This is *not* backend-stratified the way
  production is, so **`sbc_ks_pvalue_min` from this sweep is comparable only to
  other cells in this sweep, never to a published run's**.
* Guarded by `tests/foundation/test_sweep_matches_the_trainer.py`.

## 3. The decision rule for `lr_scale`, PRE-REGISTERED

Selection is on two statistics the sweep *can* measure comparably, in this order:

1. **Held-out `−log q` must not exceed the 4.0e-6 control's.** The control
   returns the prior, so its density is the "no information, honest width"
   baseline. A cell that recovers the mean while paying MORE density than the
   prior-returner is overconfident, and ISSUE-012 refuses trading calibration
   for R² in either direction.
2. Among cells passing (1), **maximise held-out `log_G` R².**
3. `lr_scale = chosen_lr / 2.0e-4` (T4's stage LR).
4. If **no** cell passes (1), nothing is chosen: run 4 launches at the best R²
   cell **with the overconfidence stated in the model card**, and ISSUE-012's
   calibration leg is declared undischargeable from this evidence.

Applying it to the completed grid:

| cond_norm | lr | log_G R² | −log q | passes (1)? |
| --- | ---: | ---: | ---: | --- |
| `layer_v1` | 4.0e-6 | −0.001 | 7.866 | control |
| `layer_v1` | 1.0e-3 | +0.644 | 8.445 | **no** |
| `layer_v1` | 3.0e-3 | +0.665 | 8.549 | **no** |
| `dataset_std_v2` | 4.0e-6 | +0.020 | 7.932 | control |
| **`dataset_std_v2`** | **1.0e-3** | **+0.674** | **7.075** | **yes** |
| `dataset_std_v2` | 3.0e-3 | +0.464 | 7.212 | yes |

**Winner: `dataset_std_v2` at 1.0e-3 → `lr_scale = 5.0`.** Best R² and the best
density of any informative cell, beating both controls. `3.0e-3` degrades
(+0.464), so 1.0e-3 is an optimum rather than a ceiling.

A bracket sweep at 4.0e-5 and 2.0e-4 is queued. **It is scored under this same
rule with no re-litigation**: if a bracket cell beats 7.075 on density *and*
0.674 on R², it wins; otherwise `lr_scale = 5.0` stands.

This also settles the conditioning question, which I got wrong twice. At 1.0e-3,
`dataset_std_v2` pays 7.075 nats where `layer_v1` pays 8.445 — the standardised
conditioning is **what makes the high learning rate usable at all**. Earlier
notes calling it "essentially nothing" were read off the 300-step smoke.

## 3a. AMENDMENT, 06:40 — the bracket says the v2 measurement is not reliable

This is the "explicit line saying which previously reported number is
superseded" that §2 requires. It is **not** a change to the rule in §3; it is a
finding that the rule's input is noisier than the rule assumed.

The bracket at 4.0e-5 and 2.0e-4 completed:

| cond_norm | lr | log_G R² | −log q | cov_mae |
| --- | ---: | ---: | ---: | ---: |
| `layer_v1` | 4.0e-5 | +0.170 | 7.561 | 0.020 |
| `layer_v1` | 2.0e-4 | +0.493 | 7.617 | 0.011 |
| `dataset_std_v2` | 4.0e-5 | +0.089 | 11.103 | 0.106 |
| `dataset_std_v2` | 2.0e-4 | +0.585 | 13.200 | 0.101 |

Put beside the main grid, `dataset_std_v2`'s held-out `−log q` against learning
rate reads:

    4.0e-6   7.932      cov_mae 0.018
    4.0e-5  11.103      cov_mae 0.106
    2.0e-4  13.200      cov_mae 0.101
    1.0e-3   7.075      cov_mae 0.021
    3.0e-3   7.212      cov_mae 0.017

**That is not a plausible smooth function of the learning rate**, and coverage
tracks it, so it is not a single flaky statistic. `layer_v1` over the same range
is monotone and well behaved (7.866 / 7.561 / 7.617 / 8.445 / 8.549).

The sweep runs **one seed per cell and no replication** — stated in
`source_ablation`'s own caveat language and true here too. A 6-nat swing that
reverses direction twice is consistent with instability in the
`_DatasetStandardise` running statistics interacting with OneCycleLR's schedule
(momentum 0.01 is a ~100-batch timescale; the summary encoder moves under it at
mid rates and the annealed tail lets it catch up at high ones). It is equally
consistent with plain seed variance. **The sweep as run cannot tell those apart,
and neither can I.**

Consequence for §3: the rule stands, but `dataset_std_v2 @ 1.0e-3` was selected
on a single draw from an arm now known to swing 6 nats. **`lr_scale = 5.0` is
not safe to launch on this evidence.**

### The one further measurement, with a hard stop

Three seeds at each of the two candidate cells — `dataset_std_v2 @ 1.0e-3` and
`layer_v1 @ 2.0e-4` — and then the decision is made and not revisited:

* if `dataset_std_v2 @ 1.0e-3` is stable across seeds (all three passing the §3
  density bar), `lr_scale = 5.0`;
* otherwise `lr_scale = 1.0` (`layer_v1 @ 2.0e-4`: R² 0.493, `−log q` 7.617,
  `cov_mae` 0.011 — lower R² than the v2 cell, monotone neighbours, best
  coverage in the whole grid, and it still clears ISSUE-012's 0.4 floor and the
  0.439 ridge probe).

**No further rounds.** If the replication is itself ambiguous, take
`lr_scale = 1.0`: choosing the stable arm under uncertainty is the decision, not
a deferral of it.

## 4. Blocking before launch

1. **The full-suite `F`.** A whole-suite run is in flight (~38% at 26 min,
   slowed by contending with the sweep). One `F` appeared around 19%; `--tb=line`
   prints names only at the end. **Must be identified and fixed** — the expected
   count for this repo is ZERO and there is no known-red set to hide in.
2. **`lr_scale` is still 0.02 in `configs/run4/scwbd-004.yaml`.** Set it to
   **5.0** per §3. `launch_run4.sh` now refuses 0.02 by value, so this cannot be
   forgotten silently.
3. **`LAUNCH=no bash scripts/launch_run4.sh` end to end, including the GPU
   smoke.** Never yet run. The smoke is the only check that can catch a loss term
   raising on its first real batch, and ISSUE-008's multirate BOLD path is new
   code on that path.

## 5. NOT blocking, and deliberately not resolved before launch

State these; do not quietly fix them at 3am.

* **Calibration is unknown, and current evidence says it is bad.**
  `sbc_ks_pvalue_min` is 0.000 in every informative cell. That number is not
  comparable to production's (§2), so it is not evidence of failure — but it is
  not evidence of success either. **Run 4 launches not knowing whether its
  posterior stays calibrated.** This goes in the model card up front, not as a
  discovery afterwards.
* **ISSUE-012 will NOT be discharged by launching.** Discharge needs R² > 0.4
  AND `ks > 0.01` AND `coverage_mae < 0.05`, measured by
  `evaluate.posterior_calibration`. Run 4 will produce that measurement; the
  sweep cannot.
* **The compute-matched control** (`dense_neural` at 9.36 nats bounds nothing;
  it is fit in minutes against a 29-hour model). Doubles evaluation cost. This is
  the user's call and it is an *evaluation* decision, not a launch one.
* **`MODEL_DESIGNATION` pins model class, not run** — provenance cannot separate
  run 1 from run 3.
* Nature Protocols / OSF `myrqn` licence email; `gp-tms-hsh` fetch; whether
  `scwbd-001-beta` / `002-pilot` should be public under a union saying
  `redistribution: none`.

## 6. Restarting the shell is safe right now

* **Nothing is writing to a production path.** No training is running
  (`ps -eo pid,args | grep "[s]cwbd\.foundation\.train"` is empty).
  `checkpoints/scwbd-003/` has not been touched since Aug 9 15:20.
* **The working tree is committed and pushed** except for untracked sweep
  artifacts under `reports/run4_posterior/`, which are outputs and are
  regenerable.
* **What dies with the shell, and must be recreated:**
  * the 30-minute cron (session-only) — recreate it with the **DO NOT LAUNCH**
    instruction intact;
  * the in-flight full-suite run (must be re-run from scratch — it is the §4.1
    blocker);
  * the queued bracket sweep (optional; §3 stands without it).
* **What survives:** every commit, `reports/run4_posterior/*.json`, this file.

## 7. On "constantly stressed about values"

I am not stressed, and no value here needs to be argued about again. Two things
make that true rather than reassuring:

* the protocol in §2 is frozen and test-guarded, so a number cannot move because
  I changed how I looked at it;
* the rule in §3 was written before the bracket data exists, so the bracket
  cannot be read selectively.

What remains genuinely uncertain — calibration — is named in §5 as uncertain and
will be settled by run 4's own evaluation, not by another sweep. The honest
position at launch is: **the posterior becomes informative in the mean (R² 0.674
against a 0.4 floor and a 0.439 ridge probe), it is not overconfident by the one
density statistic we can compare, and whether it is calibrated is unknown.** That
sentence is the model card's, and it does not need to change.
