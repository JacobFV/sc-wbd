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

### 3b. RESOLVED, 07:05 — the replication ran and `lr_scale = 5.0` stands

Three seeds at each candidate, same frozen protocol:

| cond_norm | lr | seed | log_G R² | −log q | cov_mae |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dataset_std_v2` | 1.0e-3 | 0 | +0.674 | 7.075 | 0.021 |
| `dataset_std_v2` | 1.0e-3 | 1 | +0.729 | 6.497 | 0.016 |
| `dataset_std_v2` | 1.0e-3 | 2 | +0.766 | 6.523 | 0.012 |
| `dataset_std_v2` | 1.0e-3 | 3 | +0.747 | 6.536 | 0.018 |
| `layer_v1` | 2.0e-4 | 0 | +0.493 | 7.617 | 0.011 |
| `layer_v1` | 2.0e-4 | 1 | +0.496 | 7.778 | 0.018 |
| `layer_v1` | 2.0e-4 | 2 | +0.544 | 7.851 | 0.018 |
| `layer_v1` | 2.0e-4 | 3 | +0.500 | 7.363 | 0.014 |

**`dataset_std_v2` at 1.0e-3 passes the §3 density bar on all four seeds**
(6.50–7.08, all below the 7.866 control) with R² 0.674–0.766 and `coverage_mae`
0.012–0.021, everything well inside the 0.05 condition. `layer_v1` at 2.0e-4 is
also stable but strictly worse on both axes.

So the §3a condition is met on its own terms: **`lr_scale = 5.0`**, which is what
`configs/run4/scwbd-004.yaml` already declares. No config change needed and the
decision is closed.

What this does **not** explain: the 4.0e-5 / 2.0e-4 `dataset_std_v2` cells that
returned `−log q` 11.103 and 13.200 with `coverage_mae` ~0.10. Those remain
anomalous and unexplained. They are **not** re-measured, because the rule was
about the chosen cell and the chosen cell is stable — chasing them now would be
the re-litigation §1 exists to prevent. They are written down here so that if run
4's posterior misbehaves, the first hypothesis is already on the record:
`_DatasetStandardise`'s running statistics interacting with the LR schedule at
mid rates.

## 3c. CLOSED, 08:00 — what §4's three blockers turned into

All three of §4 are discharged, and the smoke in §4.3 found **five** defects on
the way. Three of them I would have scored as passing.

| # | found | would have cost |
| --- | --- | ---: |
| 1 | `--quick` rosters (2 and 6) too small to populate three participant-disjoint folds | smoke unusable |
| 2 | `_audit_real_split` WARNED on an empty fold though its docstring says it raises; torch then died in a `RandomSampler` naming neither source nor fold | a source admitted, `leakage_audited`, contributing no gradient |
| 3 | `scripts/launch_run4.sh` smoked through the RUN's config, so the scratch run created `reports/training/scwbd-004_train.jsonl` | ISSUE-010 again — caught at **0 bytes** by `make health-run4` flipping "NOT FOUND" → "is EMPTY" |
| 4 | the smoke never left T1: `--quick` shrinks rosters, not step counts | the posterior and individualiser paths untested — the two carrying run 4's new code |
| 5 | **T4 exceeded the 56 GB cap** | **~14 hours**, at ~5,400 steps |

On #5, the cap is now measured rather than inherited: **56 → 80 GB**, against a
six-stage profile whose peak is **T5 at 59.95 GB, not T4's 57.98**. A cap fitted
to T4 — which is what I was about to propose — would have died one stage later,
repeating the exact error that caused the blocker (the original 56 was fitted to
T1, the only stage the cost run measured). 80 is 33% headroom on a **121.6 GB
unified** pool, leaving 41.6 GB host-side. Guarded from both sides: ≥15% over the
measured peak, ≥30 GB left to the host, non-zero, and every stage in the config
must appear in the measured record.

The lesson §1 was written about held here too. The value that moved was not
argued into place — it moved because a measurement existed that did not before,
and the measurement was taken across every stage rather than the convenient one.

## 4. Blocking before launch — ALL THREE DISCHARGED

Kept as written, with what each turned out to be. A blocker list edited into
agreement with the outcome teaches nothing.

1. ~~**The full-suite `F`.**~~ **DONE.** It was
   `test_universe_is_the_model_that_runs`, `assert 161 == 163` — and it was
   *mine*: ISSUE-012's `_DatasetStandardise` keeps running statistics as buffers
   and has no learnable affine, so the parameter universe lost exactly
   `cond_norm.weight` and `cond_norm.bias`. Re-baselined **with the delta named**,
   which is what that tripwire's own docstring requires, plus a second assertion
   pinning the cause so count and cause fail together. Mutation-tested: restoring
   the affine returns the count to exactly 163.
2. ~~**`lr_scale` is still 0.02.**~~ **DONE — 5.0**, by §3's rule and confirmed
   across four seeds (§3b).
3. ~~**`LAUNCH=no` end to end.**~~ **DONE.** It took *eight* runs, because each
   pass surfaced something the previous one had hidden. See §3c; the last one
   passed every check **and left the tree clean**, which is the measurement that
   matters — dry run 7 also passed every check and silently modified five tracked
   files.

**Standing after this section: nothing in §4 blocks a launch.** What remains
open is §5, which is open on purpose.

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

---

## 6. The BOLD divergence, and the rule that decides run 4's shape

Run 4 was launched 09:44 on 2026-08-10 and STOPPED at step ~400 with
`real_bold_nll` climbing 1.99 -> 12.96 while `eeg_nll` improved to 1.60 and total
`loss` stayed flat near 1.0. Full detail in ISSUE-016.

Three arms, matched LR schedules, same seed:

| arm | intervention | `real_bold_nll` |
| --- | --- | --- |
| A | as launched | 3.21 @160, **12.96 @400** |
| B | five Balloon ODE constants frozen | 3.70 @160 — no better |
| C | shared trunk frozen, observation heads live | **1.92 @160, falling** |

**The cause is the shared trunk moving under the BOLD head**, driven by the 96%
of gradient that is not BOLD. `ds002336_real` is **4.13%** of the mixture,
outvoted **23.2 : 1**. Nothing in the BOLD path is broken — arm B rules out the
ODE constants, arm C shows the head fits fine when the trunk holds still.

Run 3 could not have found this: ISSUE-008 meant its BOLD likelihood never
integrated the ODE, so it was inert. Fixing ISSUE-008 is what made the fMRI
likelihood real, and the first thing a real one showed is that it loses 23:1.

### Arm D, and the decision rule — written before the data

Arm C says the head can fit a still trunk. So the cheapest candidate fix is to
let the head TRACK a moving one: a separate optimiser group for `bold.*` at a
higher rate, the `POSTERIOR_LR_SCALE` pattern inverted. Arm D is arm A plus
`bold_lr_scale = 5.0`, nothing else changed.

**The rule:**

* if arm D's `real_bold_nll` at step 340 is **below 2.5** — near its 1.99 start
  rather than arm B's 5.60 — adopt it and relaunch run 4 with it;
* otherwise **relaunch run 4 exactly as configured** and report the degradation
  as a measured negative result about fusion under source imbalance. No knob is
  turned to make a number look better.

**"Accept it" is not the neutral option and must not be treated as one.** A
diverging term keeps feeding the trunk: at step 400 the BOLD loss was 8x the EEG
loss and growing. If arm D fails, the relaunch is accepted WITH that risk stated,
and the run is watched for `eeg_nll` degradation as well — the negative result is
about fusion, not a licence to let one term wreck the others.

**Reweighting the mixture is rejected outright**, and before seeing arm D so this
cannot be revisited when a number disappoints. `ds002336` is 485 windows over 10
participants at one site. Upweighting it pulls the trunk toward a small
single-site corpus and away from the EEG holdout the paper's headline rests on
(1.986 nats vs 2.024/2.025 over 27 participants). Trading a measured published
result for a rescued claim is the wrong direction, and the weight itself would be
a number nobody could defend.

**Unshared BOLD capacity is the NEXT run's design, not this one's.** It matches
the paper's thesis of heterogeneous state spaces held simultaneously, and it is
an architecture change needing new permissions, re-measured memory and step time,
and the four gates re-pointed. It is not a thing to add at midday to a run that
is already stopped.

### 6a. AMENDMENT — arm D FAILS on stability, and the rule's statistic was wrong

The §2 supersession line, applied to §6: the rule's INPUT turned out to have a
failure mode the rule did not anticipate. Same shape as §3a.

Arm D (`bold_lr_scale = 5.0`) does not climb like arm A. It **oscillates**:

    step  180    200     220     240     260
    D    2.050  13.780  2.122  14.242  2.008

From step 100: min 1.86, max **14.24**, mean 4.75, median 2.11.
Arm A over 100–260: min 1.95, max 3.33, mean 2.80.

Read honestly, that is two facts at once. The 5× rate **does** help the head
track — median 2.11 against arm A's 2.80 over the same steps, so the direction of
§6's hypothesis is right. And it **periodically diverges** to 14, which arm A
never did in that range. That is a learning rate above the stable region: better
average tracking, occasional blow-ups.

**The rule as written cannot decide this.** "Below 2.5 at step 340" assumed a
monotone trajectory; against a period-2 oscillation between ~2 and ~14 it is
decided by which phase step 340 happens to land on. Roughly a coin flip wearing a
threshold's clothes. I am not going to read it and act on whichever answer it
gives.

**Verdict: arm D FAILS**, and not on the threshold — on stability. A term that
spikes to 14 every other logging interval is not a term to run for 38 hours,
whatever a single step says. The pre-registered fallback therefore applies:
**relaunch run 4 exactly as configured (`bold_lr_scale` stays 1.0)** and report
the BOLD degradation as a measured negative result.

**No arm E.** The obvious next probe is 2.0 — between 1.0 (drags) and 5.0
(oscillates) — and it is exactly the spiral §1 exists to stop. The finding that
matters is already measured and is not a learning rate: BOLD is 4.13% of the
mixture and is outvoted 23.2:1, and arm C showed the head fits perfectly well
when the trunk holds still. Tuning a multiplier is treating the symptom. The
remedy is unshared capacity, and that is the NEXT run's design, decided in §6 and
not revisited here.

What arm D adds to ISSUE-016, and it is worth having: the BOLD head *can* track a
moving trunk — it does so better at 5× than at 1× on the median — but not stably
at any rate tried. That is evidence for the architecture change rather than
against it.

---

## Correction 2026-08-12 — the mixture share in §6 was the smoke run's

This document is a pre-registration and its decisions are not reopened. One
factual input to them was wrong and is corrected here rather than edited above,
so the record of what was decided, and on what, stays intact.

§6 and the arm-D reasoning quote `ds002336_real` at **4.13%** of the mixture,
outvoted **23.2 : 1**. Those come from
`reports/training/smoke-004/mixture_T1_measured_founding.json` —
`run_name: scwbd-004-smoke`, four steps per stage. The completed run measured
**5.39%** and **17.6 : 1** (`reports/training/scwbd-004/`, all five measured
stages between 0.0464 and 0.0553).

**The decision §6 records is unaffected.** It turned on the imbalance being large
and on arms B and C locating the cause in the trunk rather than in the BOLD
path; both hold at 17.6 : 1. Nothing here licenses re-opening the relaunch rule,
which was applied as written.

See ISSUE-016, "Correction 2026-08-12", for the full table and for what let a
four-step report be read as a measurement.
