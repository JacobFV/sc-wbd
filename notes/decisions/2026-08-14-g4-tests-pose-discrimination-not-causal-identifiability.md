---
date: 2026-08-14
author: lead
status: active
task: CLAIM_GATES
related: [2026-08-14-the-fisher-typeerror-is-a-refusal-not-a-bug]
title: "G4 is re-specified to test pose discrimination, which is what SC-WBD does"
---

# G4 tests what the model does with perturbation, not what a wet lab could establish

## The fork

G4 as written claims **"perturbation reduces non-identifiability"** and requires, among its five
supports, prospective recovery of direction, delay, gain, dose and state-dependence. That needs a
perturbation study designed for the purpose: multiple intensities, tracked coil poses, controlled
brain states, human participants, ethics approval.

SC-WBD is a machine-learning effort. It will not commission that study.

- **Leave G4 as written.** It reports `COULD_NOT_RUN` permanently, and the scoreboard carries a red
  row for a study nobody intends to run.
- **Delete the blocking sub-check.** G4 then returns a verdict from Fisher information alone, and
  "perturbation reduces non-identifiability" reads as validated on evidence that only shows *could,
  in principle, on a 3-region linear-Gaussian system*.
- **Re-specify the gate to the claim this project actually makes.** ← taken

## What decided it

The middle option is the one to argue against, because it is the cheap one and it looks like
progress: the site's headline would move from **0 validated claims about brains** to 1 with nothing
new measured. Deleting a falsifier does not validate a claim; it removes the thing that could have
contradicted it.

Leaving it as written is defensible but tests the wrong object. G4's five supports are properties of
an *experimental design*. SC-WBD builds a *model*. A gate that can only be discharged by running a
TMS study is not measuring the artifact this repository ships.

G4's own `consequence` field prescribes the third option:

> Narrow the identifiable parameter set and redesign the perturbation rather than reporting a causal
> estimate.

And there is precedent: `reports/CLAIM_BOUNDARY.md` records that run 1's most substantive scientific
output was "a narrowing of a thesis claim on evidence" — the identifiability benchmark vindicated
native-clock handling, not multimodal fusion.

## The replacement, G4'

**Claim.** The model discriminates perturbation designs: two coil poses produce measurably different
predicted responses, and the difference is carried by coil *orientation* rather than by field
magnitude.

**Falsified by.** Predicted responses do not separate across poses beyond a permutation null; or the
separation is fully explained by field magnitude, in which case the model is reporting the field
solver's output rather than a network response.

**Correction, same day, before anything was wired.** The claim above was drafted from the site's
summary of the pose result and is STRONGER THAN THE ARTIFACT SUPPORTS. Reading
`reports/intervene/impulse_pilot.json` directly:

| | value |
|---|---|
| CRR trained | 1.4097 |
| CRR untrained | 1.3929 |
| **ratio trained/untrained** | **1.0121** |
| shuffled-normal null, one-sided p | 0.0050 (K = 200) |
| pre-registered reading | `survived` |
| checkpoint measured | **`scwbd-002-pilot`, step 500** — not run 4 |

Two things follow. First, `survived` is defined by the preregistration as *collapsed < 0.1;
attenuated < 0.5x untrained; else survived* — it means **training did not destroy the contrast**,
not that training produced it. Second, an untrained network scores 1.3929 against the trained
1.4097, so the pose contrast is substantially a property of the **anatomy and the field solver**
rather than of anything learned. The pilot's own report says as much: a surviving contrast means the
dynamics "propagate a focal input pose-dependently, not that they do so correctly".

So "the model discriminates perturbation designs" is not the claim to make. What the evidence
supports is narrower and still worth stating:

> **G4' (revised).** A focal input propagates through the model's dynamics **pose-dependently**, and
> the contrast is carried by coil **orientation** rather than field magnitude (p = 0.0050 against a
> 200-permutation shuffled-normal null, direction predicted in advance). Training does not collapse
> this contrast.
>
> **Falsified by.** The contrast collapses (CRR < 0.1) or attenuates below half the untrained
> network's; or the orientation null is not cleared; or the same-pose control is non-zero, which
> would mean the statistic is measuring nondeterminism.
>
> **Explicitly not claimed.** That training improves pose discrimination. The trained/untrained
> ratio is 1.0121, and any version of this claim that implies learning is doing the work is refuted
> by that number.

**Still to establish:** all of the above is measured on run 2's pilot checkpoint at step 500. The
pilot is being re-run against `scwbd-004` before G4' is wired, because specifying a gate from a
two-runs-old artifact is the same error as reading a blocker list instead of a signature.

The field computations G4' depends on are independently validated: `N3_em_solver`,
`N4_acoustic_solver`, `N6_induced_efield`, `N8_induced_efield_contact` all PASS.

## What is NOT being claimed, stated once

G4' is a claim about the model's predictions. It is not a causal identifiability result and does not
license a causal estimate. That boundary needs no new prose beyond the claim text itself and the
site's existing footer — the fix here is specifying the right test, not appending caveats to the
wrong one.

## What would reverse it

Acquiring prospective perturbation data. If a study with multiple intensities, tracked poses and
controlled brain states is ever run, the original G4 becomes testable and should be restored beside
G4' rather than instead of it — they ask different questions and both are worth answering.
