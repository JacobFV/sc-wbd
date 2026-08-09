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

## What the measurement does not cover

Stated rather than left to be assumed:

* **T4 was not measured.** It admits the simulator as well, and the 46 h wall
  budget is T1's rate applied to all 14,600 steps. That is a planning number and
  not a measurement of the run.
* **T6 was not measured.** It admits the same seven sources as T1 with a
  narrowed optimiser, so it should be no more expensive; nothing checked.
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

## PENDING — results

Held-out night-2 individualisation, leave-one-source-out on the measured
holdout, and the standard baseline set are filled in after the run.
