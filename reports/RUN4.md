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

PENDING_COST

## What the measurement does not cover

PENDING_SCOPE

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
