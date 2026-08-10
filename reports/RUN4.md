# SC-WBD-004

Run 4 does two things run 3 could not: it integrates the haemodynamic ODE on the
measured fMRI path, and it fits a person effect.

This file is the run's record. Numbers here are measured and each says where
from; sections marked **PENDING** are filled after the run completes and are
empty rather than estimated until then.

## The two structural changes

| | run 3 | run 4 |
| --- | --- | --- |
| measured BOLD rollout | 8 neural steps (64 ms) indexed against 8 TRs (16 s) | `bold_predict_frames` × TR / `dt_model` neural steps |
| Balloon-Windkessel ODE on measured data | never ran | runs; the five physical parameters take gradient |
| person effect | never constructed | fitted in `T6_individual` |
| individualisation split | participant-disjoint | session split on sleep-edfx's two nights |

## PENDING — the run

The run has not been launched. What follows is the cost measurement that sizes
it, and nothing else. No training number appears in this file until there is a
checkpoint to read it from.

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

## PENDING — results

Held-out night-2 individualisation, leave-one-source-out on the measured
holdout, the within-participant arm and the standard baseline set are filled in
after the run.
