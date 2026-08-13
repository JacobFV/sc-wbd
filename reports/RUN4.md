# SC-WBD-004

Run 4 does two things run 3 could not: it integrates the haemodynamic ODE on the
measured fMRI path, and it fits a person effect.

This file is the run's record. Numbers here are measured and each says where
from. **The run completed on 2026-08-12** — 14,600 / 14,600 steps in 42.2 h — and
no section is PENDING any longer.

Two things are deliberately not in this file. The **leave-one-source-out
ablation** is still running and its arms will be added when it lands. And the
**published evaluation artifact predates two fixes to the evaluation itself**:
`real_eeg_holdout.verdict` now names the inconclusive comparators instead of
stopping at "no baseline beats it", and `theta_shift` now carries the prior scale
that makes ISSUE-017's 0.67% checkable without loading the checkpoint. Both land
on the next `make release-004-evaluate`; `HANDOFF-004.md` carries the sequence
and the reason it must not run beside the ablation.

## The two structural changes

| | run 3 | run 4 |
| --- | --- | --- |
| measured BOLD rollout | 8 neural steps (64 ms) indexed against 8 TRs (16 s) | `bold_predict_frames` × TR / `dt_model` neural steps |
| Balloon-Windkessel ODE on measured data | never ran | runs; the five physical parameters take gradient |
| person effect | never constructed | fitted in `T6_individual` |
| individualisation split | participant-disjoint | session split on sleep-edfx's two nights |

## The first launch, and why it was stopped at step 400

Run 4 launched at 09:44 on 2026-08-10 and was **stopped by hand at step ~400 of
T1**, 1.2 hours in. `real_bold_nll` — the measured fMRI likelihood, integrating
the Balloon-Windkessel ODE for the first time — climbed while everything else
improved:

```
step   lr         real_bold_nll   eeg_nll   loss
 100   3.14e-04       1.953        2.327    1.835
 200   6.00e-04       2.226        1.700    0.962
 300   5.99e-04       2.247        1.548    1.004
 400   5.96e-04      12.959        1.596    1.063
```

The LR plateaued at step 200, so this is not the warm-up. `bold_log_scale` held
at 5.4–5.9, so the variance channel is not running away. `eeg_nll` improved
throughout and the total loss stayed flat near 1.0 — **only the fMRI term
degraded.**

That is ISSUE-008's signature, which cost 46% of a 25-hour run the last time it
went unread. It was caught here at 1.2 hours by a monitor watching the quantity
per step, which is what that post-mortem asked for.

## Why it degraded: the trunk moves and BOLD is outvoted 23 to 1

Four arms, matched LR schedules, same seed. Full detail in ISSUE-016.

| arm | intervention | `real_bold_nll` |
| --- | --- | --- |
| A | as launched | 3.21 @160, **12.96 @400** |
| B | the five Balloon-Windkessel ODE constants frozen | 3.70 @160 — **no better** |
| C | the shared trunk frozen, observation heads live | **1.92 @160, 1.86 @200, falling** |
| D | `bold.*` in its own group at 5× the stage LR | median 2.11, **max 14.24** — oscillates |

**B refutes the obvious explanation.** T1 grants `bold.*` at 6.0e-4 — a
residual-stack rate applied to ODE rate constants — and that looked like the
answer. Freezing them changes nothing. A "fix" on that reasoning would have done
nothing while appearing to work.

**C identifies the cause.** Hold the shared trunk still and the BOLD head fits
the data and improves. The head is not broken; it cannot keep up with a latent
state being reshaped underneath it.

**Why the trunk moves away from it** — `per_source_contribution`, T1:

```
eegmmidb_real       0.6913
sleepedf_real       0.2117
ds002336_real       0.0539   <- the only haemodynamic source
ds004024_rest_real  0.0144
ds000117_real       0.0143
ds004024_perturb    0.0072
ds000117_behaviour  0.0071
```

**BOLD is 5.39% of the mixture, outvoted 17.6 : 1.** The trunk trains under a
gradient that is 96% not-BOLD and converges on what the EEG-like sources want.

**D shows the head can track, but not stably.** At 5× the head follows a moving
trunk better on the median (2.11 against arm A's 2.80 over the same steps) and
periodically diverges to 14.24. Better average tracking with blow-ups is a rate
above the stable region. No further multiplier was tried: the finding is not a
learning rate, and tuning one treats the symptom.

## What the trunk-drift explanation predicts for this run — written at step 260

Pre-registered while run 4 is training, so the eventual numbers can confirm or
refute it rather than be explained by it afterwards. Which stage trains what:

| stage | admits | grants `bold.*` | shared trunk |
| --- | --- | --- | --- |
| `T1_measured_founding` | 1 | yes | **trains** |
| `T2_calibration` | 1,2 | yes | **trains** |
| `T3_population_prior` | 1,2,3 | yes | **trains** |
| `T4_simulator` | 1,2,3,4 | yes | **trains** |
| `T5_measured_return` | 1,2 | yes | **trains** |
| `T6_individual` | 1 | **no** | frozen |

If BOLD degrades because the trunk moves under it, then:

1. **`real_bold_nll` rises across T1–T5** and does not recover within a stage.
   At step 260 it is 5.39 against 1.99 at step 1, already above the first
   launch's 2.69 at the same step.
2. **It is absent from T6's rows entirely.** T6 drops `ds002336_real` — its card
   grants only `bold.*` and T6 grants none of it, so the source is admitted with
   an empty permission set and skipped. Arm C's condition (a frozen trunk)
   therefore recurs in T6 *without* the term that would test it, which is why
   this run cannot settle the remedy and run 5 has to.
3. **The published fMRI number reflects T5's end state**, not the final
   checkpoint's training. Anyone reading it must be told that; the model card is
   written accordingly.

### CONFIRMED at step 1000 — and the magnitude is larger than "degrades"

The prediction above was written at step 260. At step 1000, still inside T1:

```
step      loss     eeg     bold   bold_scale
   1     1.000    1.74     1.99      5.676
 400     1.047    1.57    19.77      5.666
 800     0.986    1.43   137.89      5.572
1000     0.978    1.52   104.54      5.490
```

`real_bold_nll` is up **two orders of magnitude in 1,000 steps** and does not
recover within the stage. Prediction 1 holds; the falsifier did not fire.

Three controls make this a FUSION result rather than an instability, and all
three are measured:

* **`eeg_nll` IMPROVED** across the same span, 1.74 → 1.50. The accepted risk —
  that a diverging term would damage the shared trunk — has not materialised.
* **Total `loss` is flat** near 1.0, so the fMRI term is not dominating the
  mixture despite being ~100× the others in magnitude.
* **`bold_log_scale` holds at 5.3–5.9**, so this is not a variance channel
  running away.

One term gets monotonically worse while everything around it gets better. That is
the shape of a source losing a competition, not of a model breaking.

Precedent for continuing: run 3's `real_bold_nll` reached **4.4e6** and that run
still completed 13,400 steps. The trainer tolerates this term diverging, which is
itself part of why it went unread for 46% of a 25-hour run.

**The falsifier.** If `real_bold_nll` stabilises or falls while the trunk is
still training — anywhere in T1–T5 — the 17.6 : 1 gradient-share explanation is
wrong or incomplete, ISSUE-016 must be reopened, and `RUN5_DESIGN.md`'s adapter
proposal loses its evidence. That would be the more interesting outcome and it is
written here so it cannot be quietly absorbed.

## Run 3 could not have found this

ISSUE-008 meant run 3's measured BOLD path never integrated the ODE. The term was
inert and its five parameters sat frozen for 13,400 steps — which is exactly what
`test_balloon_parameters_receive_gradient.py` pinned, deliberately, as a green
test asserting a live defect.

**Fixing ISSUE-008 is what made the fMRI likelihood real, and the first thing a
real one revealed is that it loses 23:1.** This is a result about the model, not
a regression in it. It is the kind of thing a broken likelihood hides: run 3
could report `bold.*` as admitted, audited and frozen, and nothing in that
description was false.

## What run 4 therefore claims about fMRI: nothing, and why that is worth having

Run 4 is relaunched **as configured** — `bold_lr_scale` stays 1.0 — and the
degradation is reported rather than tuned away. Two remedies were rejected in
writing before the deciding data existed (`reports/RUN4_LAUNCH_PLAN.md` §6):
reweighting the mixture, which would pull the trunk toward 485 windows from 10
participants at one site and away from the EEG holdout the headline rests on; and
unshared BOLD capacity, which is the right next-run design and not a mid-run
patch.

So the fMRI claims are withdrawn for this run, and the reason is stated
positively: **a haemodynamic likelihood that is 4% of a mixture does not survive
the other 96%.** That is a measured statement about multimodal fusion under
source imbalance, and it is the first time this project could make it — because
until ISSUE-008 was fixed there was no fMRI likelihood to lose.

## The posterior learning rate, chosen by measurement before launch

ISSUE-012 found run 3's amortised posterior returning the prior and named four
remedies. Three are config; the one that had to be *measured* is the learning
rate, and `scripts/sweep_posterior.py` is the one-stage retrain the issue says
is sufficient — posterior only, same objective, same masked context, same
corpus, 1,500 steps per cell, held-out trajectories. Protocol and the
pre-registered decision rule: `reports/RUN4_LAUNCH_PLAN.md`. Raw cells:
`reports/run4_posterior/`.

**The control reproduces run 3, which is what makes the rest evidence.**
`layer_v1` at 4.0e-6 — run 3's exact setting, 0.02 × T4's 2.0e-4 — returns
`log_G` R² −0.001 at a posterior sd **1.031×** the prior's. ISSUE-012 measured
run 3's own checkpoint at −0.010 and 1.024.

**The learning rate was the defect.** Run 4 takes `lr_scale: 5.0`
(1.0e-3 at T4), selected by the rule "held-out `−log q` must not exceed the
prior-returning control's, then maximise R²", and confirmed across four seeds:

| seed | log_G R² | −log q | cov_mae |
| ---: | ---: | ---: | ---: |
| 0 | +0.674 | 7.075 | 0.021 |
| 1 | +0.729 | 6.497 | 0.016 |
| 2 | +0.766 | 6.523 | 0.012 |
| 3 | +0.747 | 6.536 | 0.018 |

Every seed sits below the control's 7.866 nats, so the recovery is not bought
with overconfidence. The stated alternative, `layer_v1` at 2.0e-4, is stable too
but strictly worse on both axes (R² 0.49–0.54, `−log q` 7.36–7.85).

**The across-dataset conditioning is what makes the high rate usable.** At
1.0e-3 `dataset_std_v2` pays 7.075 nats where the per-sample LayerNorm pays
8.445 — *worse* than the control — so `layer_v1` at that rate is overconfident
and `dataset_std_v2` is not.

**This does not discharge ISSUE-012, and run 4 does not assume it will.**
`sbc_ks_pvalue_min` is 0.000 in every informative cell, and that number is not
comparable to a published run's: `evaluate.posterior_calibration` draws its 512
datasets backend-stratified where the sweep's are merely shuffled. So the
posterior's calibration is **unknown** going in, and the model card says so
rather than the run discovering it afterwards.

## Will it finish inside the wall clock? Projected at step 380

The config's "~38 h" is **T1's rate applied to every stage**, and it says so.
T1 is not representative — the smoke's `traj_s_per_s` puts T1 and T4 at 0.3 and
T2/T3/T5 at 0.5, with T6 at 1.2 — so the naive extrapolation is pessimistic for
four stages of six and correct for one.

Live T1 rate over steps 100–380, excluding start-up: **9.91 s/step**. Scaling by
the smoke's per-stage throughput:

| stage | steps | s/step | hours |
| --- | ---: | ---: | ---: |
| `T1_measured_founding` | 4000 | 9.91 | 11.01 |
| `T2_calibration` | 600 | 5.95 | 0.99 |
| `T3_population_prior` | 800 | 5.95 | 1.32 |
| `T4_simulator` | 5000 | 9.91 | 13.77 |
| `T5_measured_return` | 3000 | 5.95 | 4.96 |
| `T6_individual` | 1200 | 2.48 | 0.83 |
| **TOTAL** | **14600** | | **32.9 h** |

Against `max_wall_seconds` = 46 h that is **13.1 h of margin (29%)**, so the run
is not expected to be cut off mid-stage.

Two things this projection is NOT. The ratios come from the smoke's step-1 rows,
which include per-stage warm-up — T4 allocates the simulator's graph on its first
step — so T2/T3/T5/T6 are probably *faster* than shown and T4's figure is the
least trustworthy of the six. And a wall-clock cut is not the only way to lose a
stage: `max_wall_seconds` stops the run wherever it is, so the margin protects
the LAST stage, T6, which is the one carrying the individualisation claim.

Recorded at step 380 rather than at hour 40, because "will it fit" is a question
worth answering while there is still time to act on the answer.

## If the wall clock runs out: what happens, and what it costs

The projection above narrowed as measured rates replaced estimates — 32.9 h at
step 380, 40.8 h at 6,040, 42.2 h at 8,100 against a 46 h limit. T4 is the
reason: 13.15 s/step against T1's 10.04, where the planning number assumed
parity. The margin protecting the LAST stage is under four hours, so the
behaviour on exhaustion is worth knowing before it is needed rather than after.

**One correction to how that was projected, because it changes the answer.**
Re-derived at step 8,180 the number came out at 48.44 h — 2.44 h *over* — and
the whole difference was the rate charged to the two unstarted stages. Charging
them T4's 13.15 s/step is wrong: T4 is the only stage that admits the simulator
and carries `lambda_posterior` and `lambda_slice`. `T5_measured_return` returns
to measured sources with neither term.

Run 3 measured the ratio directly, on the same six-stage curriculum:

| stage | run 3 s/step | ÷ T1 |
| --- | --- | --- |
| T1_measured_founding | 7.16 | 1.000 |
| T2_calibration | 7.02 | 0.981 |
| T3_population_prior | 7.07 | 0.988 |
| T4_simulator | 8.83 | 1.233 |
| T5_measured_return | 7.19 | **1.004** |
| T6_individual | — | did not exist |

Run 3 completed 13,400 / 13,400 steps over **five** stages; `T6_individual` is
new in run 4 (`configs/run3/scwbd-003.yaml` defines T1..T5 and no more). So the
T6 row of the projection below is the one number with no measured precedent
anywhere in the project — an extrapolation from T1's loss set, not a transferred
measurement. That is an argument for the withholding rule, not against it.

Applying run 3's ratios to run 4's own measured T1 of 10.04 s/step:

| unstarted stages priced at | projected | margin | T6 gets |
| --- | --- | --- | --- |
| T1's rate (run 3's measured shape) | 44.84 h | **+1.16 h** | 1200/1200 |
| T1's rate, T6 20% over | 45.51 h | +0.49 h | 1200/1200 |
| T4's rate (the wrong denominator) | 48.44 h | −2.44 h | 532/1200 |

So the run finishes complete on the evidence, with about an hour to spare, and
truncates only under an assumption run 3 refutes for T5. That is a thin margin
resting on a transferred ratio, not a comfortable one.

**The pre-registered check, so this is not re-litigated later.** T4 ends at
≈33.1 h elapsed and T5's rate is measurable 20 steps after that. At **T5 step
100**, compute s/step from `wall_s` and read off the T6 budget:

* **≤ 10.5 s/step** — T6 completes. Nothing to decide.
* **10.5–12.0** — T6 is tight; record the projection in the model card *before*
  T6 starts, so the step count is not discovered after the fact.
* **> 12.0 s/step** — T6 will truncate. The withholding rule below applies and
  is invoked as written, not renegotiated against how the number looks.

### Outcome — measured, all six stages

The check fired in the first band and the run completed all 14,600 steps. No
stage was truncated and the withholding rule below was not invoked.

| stage | predicted s/step | measured | stage h |
| --- | --- | --- | --- |
| T1_measured_founding | — | 10.04 | 11.16 |
| T2_calibration | — | 9.44 | 1.57 |
| T3_population_prior | — | 9.50 | 2.11 |
| T4_simulator | — | 13.24 | 18.39 |
| T5_measured_return | 10.08 (run 3's 1.004 × T1) | **9.43** | 7.86 |
| T6_individual | 10.04 (T1's rate) | **3.38** | 1.02 |

**Total 42.2 h against the 46 h limit — 3.78 h of margin.**

T5 landed 6% under its prediction, so transferring run 3's stage-shape ratio was
sound and pricing it at T4's rate would have been an error of 3.6 h.

T6 is the one that was extrapolated with no precedent, and the extrapolation was
wrong by 3× — 3.38 s/step against 10.04 predicted. The direction was safe but the
reasoning was thin: T6 freezes the population weights and runs with
`lambda_perturb: 0` and `lambda_posterior: 0`, so it backpropagates into the
person effect alone. Charging it T1's full loss set ignored its own config, which
was readable in advance. **The margin was never as thin as this file said.** It
is recorded that way because a projection that only ever gets corrected in the
direction of relief teaches nothing the next time one is tight.

**It degrades, it does not crash.** Two checks, at different granularities:

* per step, inside `run_stage` — on the deadline it prints
  `wall-clock deadline reached at step N` and `break`s out of the loop. The code
  after the loop still runs, so **`stage_<name>.pt` is still written**
  (`train.py:2527`). A truncated stage saves.
* per stage, in `train` — before starting a stage it prints
  `global wall-clock budget exhausted` and stops. A stage that has not begun is
  simply not begun.

**So the worst case is a short T6, not a lost run.** `T6_individual` fits the
person effect over frozen population weights; truncated, the individualiser is
present but under-trained, and `session_individualisation` would score a partly
fitted effect.

**That number must not be reported as a null result if it happens.** "The person
effect did not generalise" and "the person effect had 400 of its 1,200 steps"
are different claims, and only the log can tell them apart. If T6 is truncated,
the step count goes in the model card beside the individualisation number, and
the claim is withheld rather than reported weak.

Nothing can be done about it mid-run: `max_wall_seconds` is read once, at
`train()` entry, so the deadline is fixed for the life of the process. Recorded
here so that if it fires, the interpretation is already decided.

## Peak memory, ALL SIX STAGES — measured, and it moved the cap

The cost block below measures **T1's loss set**, and says so: *"T4 admits the
simulator as well and is not this loss set … a planning number, not a
measurement of the run."* That sentence turned out to be the most useful line in
this file.

The pre-launch smoke — `scripts/launch_run4.sh` step 5, run for the first time
on 2026-08-10 — walked all six stages at 4 steps each and **T4 died at the 56 GB
cap**, wanting 352 MiB more. In a real run that is ~5,400 steps and **~14 hours**
in. Re-measured with the cap lifted to 75 GB and a watchdog armed at 10 GB
`MemAvailable`, `gpu_reserved_gb` at each stage's first step:

| stage | GB reserved |
| --- | ---: |
| `T1_measured_founding` | 47.41 |
| `T2_calibration` | 49.17 |
| `T3_population_prior` | 49.17 |
| `T4_simulator` | 57.98 |
| **`T5_measured_return`** | **59.95** |
| `T6_individual` | 59.95 |

**The peak is T5, not T4.** A cap fitted to T4's 58 GB would have died one stage
later — the same mistake that produced the blocker, made a second time. This is
the whole argument for measuring every stage rather than the one that happens to
be convenient.

`cuda_reserve_gb` is therefore **56 → 80**: 33% headroom over 59.95, on a
**121.6 GB unified** pool, leaving 41.6 GB host-side. Unified matters — GPU and
host share one pool, `systemd MemoryMax` does not bound CUDA, and reporting it
as "RAM plus GPU" is what OOM'd this box. During the measurement `MemAvailable`
never fell below 46 GB.

The ceiling is raised, not removed. It remains the only thing bounding the
caching allocator, and it is what stopped the two-frame BOLD arm from taking the
machine down.

Caveat carried forward: these are step-1 figures from a 4-step-per-stage smoke.
Reserve can drift upward across thousands of steps, which is what the 33% is
for; run 3's full 29-hour run peaked at 46.13 GB against a 56 GB cap.

### The caveat, discharged by the completed run

| stage | smoke, step 1 | run 4 peak | drift |
| --- | ---: | ---: | ---: |
| `T1_measured_founding` | 47.41 | 50.55 | +3.14 |
| `T2_calibration` | 49.17 | 50.55 | +1.38 |
| `T3_population_prior` | 49.17 | 50.55 | +1.38 |
| `T4_simulator` | 57.98 | 60.34 | +2.36 |
| `T5_measured_return` | 59.95 | 60.34 | +0.39 |
| `T6_individual` | 59.95 | 60.34 | +0.39 |

**Run peak 60.34 GB against the 80 GB cap — 75.4% used, 19.66 GB spare.** The
drift the 33% headroom was bought for is real and small: at most +3.14 GB, and
the smoke's ordering was right in the respect that mattered — the peak is reached
by T4 and held, not set by T1.

The smoke under-read every stage, so sizing a cap from it without headroom would
have failed. It was still the right instrument for the decision it was used for,
because it was labelled as a 4-step smoke at the point of use and given 33%.
Compare ISSUE-016's "Correction 2026-08-12", where the same smoke's *mixture*
report was quoted as the run's measurement with no such label and had to be
retracted from the paper, the site and the model card. The difference is not the
instrument. It is whether the provenance was written down beside the number.

## Cost of the new BOLD path — MEASURED

ISSUE-008's fix rolls the neural clock for the duration a BOLD frame actually
covers. At `bold_predict_frames: 2` and TR = 2 s that is 500 neural steps per
window against run 3's 8, so run 3's `data.batch: 8` — itself a measured
maximum — could not be carried over.

`scripts/measure_run4_cost.py`, 23 optimiser steps of the T1 loss set with the
first 3 discarded, all seven measured sources contributing a term, 414 parcels
at `hidden=1408`. Raw output in `reports/run4_cost/cost_{b,c,d}.json`.

| arm | frames | `bold_every` | s/step | peak reserved | peak allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| a | 2 | 1 | — | **OOM at step 1** | — |
| b | 1 | 1 | 9.45 | 52.54 GB | 50.50 GB |
| c | 1 | 1 | 9.30 | 52.54 GB | 50.50 GB |
| d | 1 | off | 7.22 | 35.55 GB | 34.18 GB |

**Two frames does not fit.** 500 neural steps of the Balloon ODE, held live
beside seven other sources' activation graphs — `MixtureTrainer` backwards each
with `retain_graph=True` — exceeded the 56 GB cap on the first step:

    torch.OutOfMemoryError: ... 56.25 GiB memory in use. 56.00 GiB allowed

The cap held. `cuda_reserve_gb` is the only bound on the caching allocator on
this box — `systemd MemoryMax` does not see CUDA allocations — and this is the
second time it has been the thing between a run and a dead machine. The process
died; nothing else did.

**The BOLD term costs 2.08 s/step and 17.0 GB of reserve**, at one frame. That
is arms c minus d, the same configuration with the term switched off, and it is
the whole increment over run 3's path. It also confirms `bold_every` end to end:
arm d logs `bold_skipped_this_step` and emits no `real_bold_nll`.

**The likelihood is finite.** `real_bold_nll` sits at 1.973–1.987 across all 23
steps. Run 3's went 21.7 at step 1 to 4.37×10⁶ by step 6,000 through the same
loss key. That is not yet evidence the haemodynamics are right — 23 steps is not
a result — but the term is no longer diverging, and it is the first time the ODE
has run on measured data at all.

### What was cut, and which lever

`bold_predict_frames`, not `batch`. The term that does not fit is the one that
was reduced: 17.0 GB of the 52.54 is the BOLD rollout, and halving `batch` would
have degraded all seven measured sources to pay for one.

**One frame at TR = 2 s is 2 s of predicted haemodynamics from 48 s of context.**
The BOLD response peaks around 5 s, so this term constrains the rise of the
response and not its peak or its undershoot. No claim about HRF shape may be
read off SC-WBD-004, and that sentence belongs beside any fMRI number on the
model card. It is still 250 integrated ODE steps where run 3 had none.

The next lever, if two frames are wanted, is a per-source BOLD batch. That term
already runs at batch 4 — `max(4, data.batch // 8)` — while the others run at 8,
and dropping it further buys the horizon without touching them. It is a code
change and 004 does not make it.

## T6 — measured, and it found a source running for nothing

12 steps of the individualisation stage, the same harness. The person effect is
built and fitted:

    [individualizer] 191 participants, 266 sessions, theta_dim=6

191 = 109 eegmmidb + 78 sleep-edfx + 2 + 2, which is the index spanning every
corpus rather than eegmmidb alone. From the checkpoint's own
`moved_since_init`, after 12 steps:

| module | moved | frozen |
| --- | ---: | ---: |
| `individualizer` | 5 | 1 |
| `eeg` | 6 | 5 |
| `eeg_montages` | 18 | 15 |
| `family_local` | 0 | 137 |
| `bold` | 0 | 8 |

`z_person` and `z_session` both leave zero. The one frozen individualizer
tensor is `_alpha_raw`, inert by construction at `n_groups=1` — no corpus here
ships group labels, and a group effect nothing indexes is a declared-and-frozen
tensor. `family_local` 0/137 is the frozen population the stage declares.
`unfingerprinted` is empty, so nothing reports as moved for want of a recorded
initialisation.

**Two admitted sources were granted nothing and ran anyway.** T6 admits tier 1
as a whole — admission is per tier, there is no per-source switch — while
granting only `individualizer.*` and the observation nuisance. `ds002336_real`
and `ds000117_behaviour` overlap that in nothing, and an empty permission set is
an error to nothing: `GradientGate.grads` returns `{}` without calling autograd.
So `ds002336_real` ran a 250-step Balloon rollout on every step, logged
`real_bold_nll` on every step, entered the mixture total, renormalised every
other source's weight, and could not move one parameter.

`note_ungranted` now names them and the loss call sites skip them:

    [curriculum] T6_individual: admitted with an EMPTY permission set, so
    skipped: ['ds000117_behaviour', 'ds002336_real'].

| T6 | s/step | peak reserved |
| --- | ---: | ---: |
| before | 5.64 | 52.50 GB |
| after | 3.52 | 30.13 GB |

Skipped and NAMED, into `_absent_admitted`, because dropping a term changes the
mixture's renormalisation and that is a change to the objective. It is the
"exercised versus contributed" distinction the attachment report exists for, one
stage earlier than the report can see it.

## What the measurement does not cover

Stated rather than left to be assumed:

* **T4 was not measured.** It admits the simulator as well, and the 46 h wall
  budget is T1's rate applied to all 14,600 steps. That is a planning number and
  not a measurement of the run.
* **T6's step time was measured under contention** (6 competing processes) and
  is an upper bound. Its peak reserve is per-process and is not.
* **One foreign pytest ran in another repository during the timed steps.** The
  harness samples competing processes and records them, so every step time here
  is an upper bound. The two BOLD-on arms were measured under different
  contention (16 and 3 competing processes as first counted, 3 after the sampler
  stopped counting this run's own DataLoader workers) and agree to the byte on
  memory and to 0.15 s on time, so contention is not what these numbers are
  measuring. Peak reserve is per-process and unaffected either way.
* **This is not a scaling curve.** HANDOFF-004 and CLAUDE.md both forbid probing
  the scaling with a sweep — a sweep that reached 8 sources × batch 32 took this
  machine down. Four bounded runs of tens of steps: one configuration that
  failed, one repeated, and one ablation of the term under test.

## A launch blocker found by trying to launch

`FoundationTrainer.__init__` would not construct on the run-4 config:

    BindingDriftError: compiler->torch binding is incomplete
      binding 'port:*.message_out' -> 'msg_proj.*' matches no parameter of SCWBD

`SCWBD.msg_proj` is the pooled arm's message projection and is now built only
when `family_state` is false — the explanation HANDOFF-004 records as missing
for run 3's one unexplained frozen group. `FOUNDATION_FAMILY_BINDING`, the table
that exists *for* the family arm, still named it, so eight declared ports bound
to nothing and the trainer fails closed rather than training through decorative
masks.

It shipped because every assertion in `test_compiler_binding.py` builds its
model from `configs/scwbd_001_beta.yaml`, which is the pooled arm.
`test_the_family_arm_binding_is_clean.py` is the missing half. The lesson is the
smoke's own: a fix verified against the control is not verified.

## Individualisation: what had to change

Run 3 finished 13,400 steps and `reports/training/evaluation_run3.json` records

    "individualization": {"applied": false,
                          "reason": "no individualizer on the trainer (population model)"}

`Individualizer` existed, `save_checkpoint` had a slot for it, `STAGE_PERMISSIONS`
named `individualizer.*`, and `real_losses` applied the module when one was
present. Four things were false at once, and each alone makes the claim
unmeasurable:

1. **Nothing constructed it.** `individualize` is a property of a stage's
   `extra.curriculum` block and no run-3 stage declared one. It is now built in
   `build_data` — beside the corpora that size it and before `maybe_resume`, so
   a resumed run restores the fitted effect instead of silently continuing with
   a re-initialised one.
2. **The participant index covered one corpus.** `_participant_ids` read
   `real_dataset` — eegmmidb — so every sleep-edfx participant fell through
   `participant_index`'s unknown branch onto row 0 and 78 people shared one
   person effect. The between-participant spread of the applied theta shift is
   then 0 for arithmetic reasons that are indistinguishable, in the report, from
   the split's reasons. It now spans every EEG corpus, and raises if two corpora
   share a subject id rather than merging two people.
3. **The session effect was never selected.** `Individualizer.forward` was
   called with `participant=` alone, leaving `zeta` reachable by nothing but
   `prior_penalty`, which shrinks it toward zero. `session_index` now selects it
   by `subject/session`, and an unseen session takes row 0 — so the held-out
   night gets no fitted session effect, which is what makes the held-out number
   mean anything.
4. **Its parameters were in neither list `moved_since_init` reads.** A fitted
   person effect and an absent one produced the same report. The two lists are
   now one, `_fingerprinted_modules`, and the module is fingerprinted at
   construction so a person effect that has never taken a step does not report
   as moved.

Two further changes were needed to make the fit legal rather than merely
possible:

* **`sleepedf_real` grants `individualizer.*`.** It froze it through run 3,
  which put the permission on eegmmidb — 109 participants with one session each
  — and withheld it from the only corpus with a second session of the same
  person. 75 of sleep-edfx's 78 participants have two nights.
* **R07's shrinkage is attached once per optimiser step**, to whichever admitted
  source is permitted to move the effect. It was tied to `eegmmidb_real`, so a
  stage admitting sleep-edfx and not eegmmidb would have fitted `z_person` and
  `z_session` with no prior at all.

## A permission that was broader than the card

`FoundationTrainer.stage_sources` intersects each card's `A_k` with the stage
allowlist and says "restrict only". For patterns that are incomparable — neither
`fnmatch`es the other — it took the stage's:

    card  eeg_montages.ds000117_real.*      one source's own operator
    stage eeg_montages.*.log_gain           every montage's gain
    old   eeg_montages.*.log_gain           broader than the card

Every run-3 stage that calibrates montage nuisance produced that, for three of
the four montages. Nothing leaked — `autograd.grad(..., allow_unused=True)`
returns `None` for a head the source's loss does not touch — so it was a latent
widening with no symptom, and the permission audit in every run-3 checkpoint
records the wide pattern as the effective one. `glob_intersection` now computes
the pattern inside both, segment by segment, and a pair whose intersection
cannot be computed is recorded rather than guessed at.

## The holdout run 4 is evaluated on is not run 3's

Two evaluation defects were open against run 3 and both could only be discharged
before a run starts, because both change what "held out" means. Run 4 takes them.

### The split policy is declared, and run 4 declares a different one

ISSUE-014. `shuffle_slice_v1` — runs 1, 2 and 3 — shuffles the sorted participant
list and slices by count, so a participant's fold depends on the whole
participant set. Measured on the 109-participant eegmmidb roster at run 3's own
`seed=20260807`:

| policy | reassigned when one participant goes | of them into `test` | folds |
| --- | ---: | ---: | --- |
| `shuffle_slice_v1` | **28** of 108 | **6** | 71 / 11 / 27 |
| `stable_hash_v2` | **0** of 108 | **0** | 67 / 17 / 25 |

`evaluate_model` rebuilds the split from whatever corpus is on disk, so one
recording that fails to preprocess is enough to trigger the reassignment, and the
failure reads *better* when broken: a participant the model memorised, promoted
into the test fold, improves the held-out score.

Run 4 declares `data.split_policy: stable_hash_v2`, whose assignment is a
function of a participant's own group key and the seed alone. It costs four
training participants and buys a holdout that cannot drift. `real_split.policy`
in every 004 artifact records which policy produced it.

**Run 3 is not re-split.** `configs/run3/scwbd-003.yaml` now names
`shuffle_slice_v1` in its own `data:` block, and re-running it reproduces
`checkpoints/scwbd-003/last.pt`'s recorded fingerprint field-for-field — the same
71/11/27 folds, the same sha256 `bdf41ba7…`. **Run 4's held-out numbers are
therefore not directly comparable with run 3's table**: they are computed on 25
different people.

`--quick` reduces the participant roster, which under an order-dependent policy
is a different holdout — it put S001 and S004 in the test fold at run 1's seed
although the full-roster split trains on both. `build_data` now refuses `--quick`
outright unless the declared policy is order-independent.

### `subject_specific_ar` is measured instead of duplicated

ISSUE-013. The arm was fitted on the *train* participants and scored on the
*test* participants, which R10 makes disjoint, so 100% of scored windows fell
through to the pooled fallback and the row was bit-for-bit `ar16`. Three runs
reported six baselines where there were five.

Run 4's protocol (`baseline_protocol: v2_no_pooled_subject_specific`) drops the
row from `real_eeg_holdout` and records in `dropped_baselines` that it went and
why. The quantity it was supposed to measure — the hardest baseline the thesis
names — is measured by `within_participant_holdout`, which keeps the
participant-disjoint OUTER holdout and puts a temporal INNER split inside each
held-out participant: their earliest 50% of windows fit their own AR(16), eight
windows are dropped as a gap, and their later windows are scored. `ar16_pooled`,
fitted on the train fold, is scored on the identical windows.

On run 4's test fold that is **25 participants, none skipped, 1,500 fit windows
and 1,000 scored windows**. Exercised end to end against run 3's weights on run
3's fold: `fraction_via_pooled_fallback` **0.000** with all 27 participants
served by their own model, against 1.000 under the old protocol. The arm is no
longer `ar16`.

It is reported as its own top-level block and carries `not_comparable_with`,
because its baseline has seen the scored participant and every arm in the main
table has not.

## Results — the run finished, and it is mostly a negative result

Run 4 completed all 14,600 steps on 2026-08-12 in 42.2 h. Three of the four
questions it was launched to answer now have answers, and two of those answers
are no. Written in that order deliberately: the run's value is in what it
settled, not in what it improved.

### fMRI: ISSUE-016 confirmed over a full curriculum

`real_bold_nll` ran **1.99 → 36,472**, a factor of about 18,000. The prediction
recorded in this file at step 260 was that the divergence is a property of the
configuration rather than of a trajectory. It is. Two things the aborted
400-step launch could not show:

* **It never plateaus.** T4 alone spans 1,530 to 650,815 — four orders of
  magnitude inside one stage. This is not a likelihood that converged to a bad
  constant; it is a term being thrown around by a latent state optimising for
  something else.
* **T5's measured return does not repair it.** T5 grants `bold.*` again and is
  the stage most likely to pull the head back onto its data. It ends at 36,472 —
  below T3 and T4, four orders of magnitude above where T1 began.

`bold_parcels_covered` held at full value throughout. The ODE ran, on every
parcel, for 46 hours, and the likelihood it produced is worthless. **Run 4
claims nothing about fMRI**, which is what this file said before the run and is
now a measurement rather than a precaution.

### The posterior: the repair worked, overshot, and did not discharge ISSUE-012

The learning-rate diagnosis was correct. Run 3's posterior ignored its
conditioning; run 4's `log_G` posterior is **8× narrower than the prior** and its
mean moves by 1.10 prior sd as the data change. The flow reads its conditioning,
which three runs could not achieve.

It is also not usable. `posterior_z_sd` for `log_G` is **59.25** — the posterior
mean sits about 59 of its own standard deviations from the truth. It narrowed by
8× and earned roughly a quarter of that narrowing. `sbc_ks_pvalue_min`
**1.0e-147**, `coverage_mae` **0.203**, against run 3's 0.098 and 0.021.

| | run 3 | run 4 |
|---|---|---|
| informative? | no — R² ≈ 0 on all six | partly — `log_G` 0.284, `log_sigma` 0.102, four still ≤ 0 |
| calibrated? | yes, by construction | no — z-sd 59.25, KS p 1.0e-147 |
| claim status | `partial` | **`unsupported`** |

Uninformative-but-honest became partly-informative-and-overconfident. These are
not two points on a path from worse to better; they are two ways to be unusable.
The one-stage sweep predicted `log_G` R² 0.674–0.766 and production returned
0.284 — under half. Why is **unmeasured**, and the obvious hypothesis (T5 and T6
give part of it back at `lambda_posterior` 0.2 then 0.0) is written down in
`known_issues.md` as a hypothesis with the one evaluation that would settle it.

### The claim machinery worked without intervention

`amortized_posterior_self_consistency` publishes `unsupported` on run 4 where
run 3's identical claim published `partial`. Nobody adjusted a threshold. The
block that certified an uninformative posterior in run 3 refused an
overconfident one in run 4, on the gate `worst > 0.01 and mae < 0.12` that was
written before either number existed. That is the only part of this run that
went entirely to plan.

### Individualisation, and three defects in the path that measures it

`T6_individual` ran its full 1,200 steps. **The evaluation then measured it zero
times**, for three stacked reasons, each of which hid the next:

1. `session_individualisation` — the only block that reads a person effect — was
   never called by `evaluate_model`.
2. It could not have been. The `if __name__ == "__main__"` guard sat 44 lines
   ABOVE that function's `def`, so `python -m scwbd.foundation.evaluate` ran
   `main()` before it existed. Wiring in the call produced `NameError`, not a
   number.
3. Once reachable it crashed: `_scwbd_scores` hardcoded the founding 64-channel
   projector and head, and sleep-EDFx has 2 channels.

Neither other holdout covers it, which is why the absence looked like a result
that had not been reached yet. `real_eeg_holdout` is participant-disjoint, so all
25 scored participants sit at `z_person` exactly zero —
`n_individualised_participants: 0`. `within_participant_holdout` scores the
SC-WBD arm with no person effect fitted while its AR arm *is* fitted on that
participant's past. Two individualisation-shaped blocks, zero measurements.

**Do not read individualisation off the training log either.**
`sleepedf_real_eeg_nll` ends T5 at 1.4957 and T6 at 1.8223, which reads as the
individualisation stage degrading its own dataset. It is noise: the metric is
per-batch with a spread near 0.25 and swings 1.26 to 2.57 inside T6. Last-half
means are T5 **1.633 ± 0.226** and T6 **1.683 ± 0.275** — indistinguishable. No
degradation, and no improvement.

All three defects are fixed and mutation-tested
(`tests/foundation/test_individualisation_reaches_the_report.py`). The
individualisation number itself is the one result still outstanding.

### Individualisation: measured for the first time, and unsupported

The evaluation ran with the three defects fixed. `session_individualisation`
scored **75 participants, 1,500 held-out night-2 windows**: NLL **2.0436**
[2.0049, 2.0896], cluster-bootstrapped over participants rather than windows.

That number is not the finding. This is:

| | measured |
|---|---|
| between-participant spread of the applied shift | **7.06e-4** |
| the model's own prior scale for a person effect (`sd_person`) | **0.1059** |
| ratio | **0.67%** |
| scored participants whose person effect is exactly zero | **30 of 75** |

The individualiser moved theta by **0.7% of the scale it allocated for the
effect**, and 30 of the 75 people it scored have a person-effect row of exactly
zero after 1,200 steps of T6.

`session_individualisation`'s own pre-registered falsifier, written before the
run: *"`theta_shift.spread_pooled` at or near zero means the individualizer
applied nothing even on a split built to let it apply something, and the third
capability is unsupported. Say so on the site in the terms runs 1 and 2 got."*

**It is met. Individualisation is unsupported for SC-WBD-004**, and the site and
card say so in those terms. This is a real measurement rather than the
unmeasurability runs 1–3 reported: the split was built, the person effect was
trained, the block was fixed until it ran, and the answer is that the effect is
0.7% of its own scale.

One thing this does **not** say. The 2.0436 held-out score is a real number and
it is not evidence for or against individualisation, because the shift that
distinguishes an individualised model from the population model is the 7e-4. The
NLL is reported because withholding a measured number is its own distortion, and
labelled because it answers a different question than it appears to.

`_alpha_raw` is exactly zero in the checkpoint and that is arithmetic, not a
stuck gate: it is the group effect, `n_groups` is 1, and `alpha = raw - w*raw`
is identically zero for a single group.

### The headline EEG result: weaker than run 3, on a different holdout

| | run 3 | run 4 |
|---|---|---|
| SC-WBD NLL | 1.9863 | 2.0244 |
| vs `ar16` | −0.0391 [−0.0696, −0.0152] | **−0.0100 [−0.0480, +0.0144]** |
| vs `var4` | −0.0377 [−0.0669, −0.0169] | **−0.0127 [−0.0561, +0.0146]** |
| baselines beating SC-WBD | none | none |
| inconclusive | none | **`ar16`, `var4`** |

Run 3 beat every baseline with intervals excluding zero. Run 4's margin over
`ar16` is 4× smaller and its interval now includes zero. **No baseline beats
SC-WBD-004, and it is no longer shown to beat the two autoregressive
baselines.**

The comparison is not like-for-like and must not be reported as a regression
without that caveat: ISSUE-014 changed the split policy between runs, so run 3
scored 27 participants under `shuffle_slice_v1` and run 4 scores 25 under
`stable_hash_v2`, with baseline protocol v1 replaced by v2. These are different
test sets. What can be said flatly is that run 4 does not reproduce run 3's
separation from `ar16` on run 4's holdout.

### Reproducing `session_individualisation` from HEAD will not match this artifact

`evaluation_run4.json` records `git_sha 432b6e3…-dirty` and its individualisation
numbers were produced by that code. Re-running from a later HEAD gives slightly
different ones, and the reason is deliberate rather than a defect.

`session_individualisation` was calling `_scwbd_scores` with the default
`n_theta_samples=32`, computing 32 extra rollouts per batch and discarding every
one — 35 of the evaluation's 59 minutes. Setting it to 0 is not
behaviour-preserving: `AmortizedPosterior.sample` calls `torch.randn` with no
generator, so it draws from the **global** RNG, and removing 32 draws per batch
shifts the stream for every subsequent batch and moves `th_bar`.

The fix was therefore held uncommitted until after the artifact was written, so
the `git_sha` on it names code that actually produced it. The affected numbers
are the held-out session NLL and its interval. `theta_shift` is read off the
checkpoint's weights and is unchanged by sampling, so **the 7.06e-4 and the
0.67% — the numbers ISSUE-017 rests on — are reproducible from any HEAD.**

### The ablation: what run 3's returned, and why run 4's asks a different question

Run 4's leave-one-source-out is still running (~5 h; run 3's eleven arms took
284 min). The comparison it has to beat is recorded here first, so the arms are
read against a stated expectation rather than interpreted after the fact.

**Run 3's ablation, from `evaluation_run3_ablation.json`:** `with_all_sources`
0.6793, 200 steps per arm, and **nine of ten families returned negative
transfer** — removing them *improved* the score.

| family | delta | |
|---|---:|---|
| `sim_wholebrain` | **+0.0445** | removing it hurt |
| `ds002336_real` | −0.0097 | negative transfer |
| `eegmmidb_real` | −0.0061 | negative transfer |
| `ds000117_real` | −0.0053 | negative transfer |
| `sleepedf_real` | −0.0050 | negative transfer |
| `ds004024_perturb` | −0.0049 | negative transfer |
| `ds000117_behaviour` | −0.0044 | negative transfer |
| `montage_calibration` | −0.0020 | negative transfer |
| `ds004024_rest_real` | −0.0013 | negative transfer |
| `anatomical_prior` | −0.0006 | negative transfer |

**That result was structurally guaranteed and is not evidence about the
sources.** Every arm was scored on `_sim_val_nll` — the *simulated* validation
set — so the question it asked was "does dropping this measured source help the
model fit the simulator?", and during 200 retraining steps every measured
gradient pulls parameters away from exactly that. The one positive delta is the
simulator's own family. Nine of nine measured families falling the same way is
the design, not a finding.

**Run 4's arms are scored on the measured holdout as well**, which is what makes
them an experiment. `real_eeg_holdout` is the same 25 participants the headline
rests on, so "which sources carry the win" has a checkable answer for the first
time. Both scores are kept and labelled; the simulated one is retained for
comparability with run 3 and is not the result.

ISSUE-016 gives it a candidate answer to check against: if the trunk really does
converge on what the electrical sources want, `ds002336_real` should be close to
free on the measured holdout — its 5.39% of the mixture bought a diverging
likelihood. That is a prediction, and it is written down before the numbers are
in so it can be wrong.

### The arms — measured 2026-08-13, and the prediction above was wrong

The ablation completed in **6 h 12 m**, 200 steps per arm, eleven arms.

**`ds002336_real` came back at +0.0010 — on the CONTRIBUTING side.** Removing the
haemodynamic corpus made measured EEG prediction *worse*. It is one of only two
families that earned its place, and I predicted it would be close to free. The
magnitude is nearly free — second-smallest of the ten — but the sign is the
opposite of "this source bought nothing".

**That is the interesting result of the whole run.** `ds002336_real`'s own
likelihood diverged by a factor of 18,000 (ISSUE-016) while its gradient was
*helping* the shared trunk forecast measured EEG. A source can contribute
information to the representation and simultaneously fail to be predicted by it.
Run 4's model card says `ds002336_real` "contributed a **gradient**, not
**information**". On the measured holdout that sentence is now too strong in one
direction: the gradient did carry information, into the trunk, just not back out
through the BOLD head.

| family | measured Δ | simulated Δ | |
|---|---:|---:|---|
| `eegmmidb_real` | **+0.0144** | −0.0049 | the only substantial contributor |
| `ds002336_real` | **+0.0010** | −0.0024 | contributes, marginally |
| `ds000117_behaviour` | −0.0031 | +0.0065 | |
| `ds004024_rest_real` | −0.0031 | −0.0159 | |
| `sim_wholebrain` | −0.0034 | **+0.0366** | helps the simulator, hurts measurement |
| `ds004024_perturb` | −0.0034 | +0.0148 | |
| `anatomical_prior` | −0.0047 | −0.0120 | |
| `ds000117_real` | −0.0051 | +0.0177 | |
| `montage_calibration` | −0.0065 | +0.0191 | |
| `sleepedf_real` | **−0.0079** | −0.0065 | the largest negative on measurement |

**Two of ten families earn their place on the measured holdout.** Positive delta
means removing the family made measured prediction worse.

Three things follow, and the second is the one worth arguing about.

**The win is `eegmmidb_real`'s.** At +0.0144 it is fourteen times the next
contributor and larger than the entire 0.0100 margin by which SC-WBD-004 fails to
separate from `ar16`. The headline forecast result rests on the majority source,
not on fusion.

**`sim_wholebrain` helps the simulator and hurts measurement** — +0.0366
simulated against −0.0034 measured, the largest sign reversal in the table.
Simulator pretraining is the single largest contributor to fitting the simulator
and a mild liability for predicting real EEG. Run 3's ablation could not have
seen this: scored only on `_sim_val_nll` it reported the +0.0366 half.

**`sleepedf_real` is the largest negative at −0.0079.** It is 21% of the mixture
and the holdout is eegmmidb; whole-night polysomnography is a different regime,
and on this holdout it costs. That is a statement about this holdout, not about
the source.

**No error bars.** One arm per family, no seed replication, exactly as in run 3.
The individual deltas are not effect sizes and must not be quoted as any. The
signs and the ordering are what is readable, and `ds002336_real` at +0.0010 is
close enough to zero that its sign is the weakest claim in the table.

### ISSUE-008 verified on the weights, not on the code

`reports/scwbd-004_derived.json` reads parameter movement off the checkpoint
rather than off a card. **All 8 `bold` tensors moved off their initialisation,
0 frozen.** Run 3's were bit-identical to initialisation across all five stages,
which is what made its fMRI likelihood inert.

"The ODE integrates" is a claim about code and was already true when the run
launched. "The haemodynamic parameters received a gradient" is a claim about
weights and could only be checked afterwards. They are different claims and run 2
shipped with cards asserting the second on the strength of the first.

99.986% of parameters moved (27,405,696 of 27,409,526; 349 of 384 tensors). The
3,830 frozen sit in `observation`, `eeg`, `eeg_montages` and one `individualizer`
tensor — the fixed operators and buffers a curriculum is not supposed to move.
`individualizer` moved 5 of its 6, which is consistent with ISSUE-017: the person
effect trained, it just barely moved.

`admitted_but_no_term` records that T6 admitted `ds000117_behaviour` and
`ds002336_real` and gave neither a loss. That is the finding already written up
under "T6 — measured, and it found a source running for nothing", now confirmed
from the released artifact.

### The montage fix moved nothing, checked rather than assumed

`_scwbd_scores` gained a `source_id` whose default routes to the founding
montage. The argument that this is behaviour-preserving is short and was still
worth checking: the regenerated `real_eeg_holdout` is **bit-identical** to the
artifact produced before the change — every ranking entry and every paired delta
to six decimals — as is `posterior_calibration`.
