---
date: 2026-08-14
author: lead
status: measured
task: CLAIM_GATES
title: "All five gates are blocked on baseline models nobody trained, and one on data nobody holds"
---

# What actually blocks the five gates

Read from `reports/gates/G*.json` `blocking_reasons` and `subchecks`, 2026-08-14.

| gate | claim | what is missing |
|---|---|---|
| G1 | typed fusion beats naive resampling | `naive_resampling` and `single_modality_*` baselines — *the thing the claim is against* — plus a typed fusion candidate and held-out sets |
| G2 | anatomical topology improves inference | a `model_for_graph(adjacency)` factory, and the model retrained on dense / randomized / distance-matched controls |
| G3 | multiresolution state adds information | a multiresolution candidate, a coarse-only baseline, a restriction map R |
| G4 | perturbation reduces non-identifiability | a bound Fisher map (code), **a prospective perturbation dataset (does not exist)**, per-design model evidence |
| G5 | individualization improves future prediction | `population`, `anatomy_only`, `session_adapted` baselines |

## The shape

Every one of them needs **a model that was never trained**. Not a module, not a wiring fix — a
training run whose job is to be the thing the claim is measured against.

G4's `prospective_recovery` is the exception in the other direction: it needs recovery of direction,
delay, gain, dose and state-dependence from a prospective perturbation dataset, and its blocking
reason states plainly that none is held and it *"is expected to remain COULD_NOT_RUN"*. It is
mandatory, so **no amount of compute discharges G4** — only data collection.

## Cost, to the extent it is known

Run 4's leave-one-source-out ablation retrained **11 arms in 6 h 11 m** (371 min, derived report
15:17:39 → artifact 21:28:26). That is the unit of "retrain an arm" on this box. It is a lower bound
for the gate baselines, which are different configs rather than a re-run of that sweep.

## Where G5 stands, because it is closest

Run 4 already holds the individualised candidate and the new-session holdout the claim is about —
`session_individualisation`, 75 participants recorded twice, scored on night 2. It is short exactly
three baseline arms.

## What would refute this

A1 in `scratch/CLAIM_GATES.md`: regenerate the gate reports against run 4 and find that a blocker
listed here is already satisfied by an artifact that exists. The reports are from run 1 and run 4
holds strictly more, so this is a live possibility for G5 in particular.
