# Run 5, designed from run 4's evidence rather than from run 4's disappointment

Written 2026-08-10, while run 4 is training, because the evidence this rests on is
four diagnostic arms that will be much harder to reconstruct later. It is a
DESIGN NOTE, not a handoff: run 4 has not produced results, and nothing here
should be started before it does.

Run 4's stated purpose was a haemodynamic fMRI likelihood. It got one, and the
first thing a working one revealed is that it loses 23 : 1. That is the finding
run 5 is about.

---

## What is established, and how

Four arms, matched LR schedules, same seed. Configs in `configs/run4/probes/`;
arm A is the run itself, log preserved in `reports/run4_aborted/`.

| arm | intervention | `real_bold_nll` |
| --- | --- | --- |
| A | as launched | 3.21 @160, **12.96 @400** |
| B | five Balloon-Windkessel ODE constants frozen | 3.70 @160 — no better |
| C | shared trunk frozen, observation heads live | **1.92 @160, 1.86 @200, falling** |
| D | `bold.*` in its own group at 5× the stage LR | median 2.11, **max 14.24** — oscillates |

Three conclusions, each carrying its own refutation:

1. **It is not the ODE constants.** Arm B freezes exactly the five physical
   parameters and the climb is unchanged. The obvious hypothesis — a
   residual-stack learning rate applied to ODE rate constants — is wrong, and it
   was the one that looked right.
2. **It is the trunk.** Arm C holds the shared latent state still and the BOLD
   head *improves*. The head models this data perfectly well.
3. **Chasing the trunk faster is not the fix.** Arm D improves the median and
   destabilises: better tracking with periodic blow-ups is a rate above the
   stable region, not a solution.

The cause is a **gradient-share imbalance**. `per_source_contribution` in T1:

```
eegmmidb_real       0.7021
sleepedf_real       0.2154
ds002336_real       0.0413   <- the only haemodynamic source
ds004024_rest_real  0.0147
ds000117_real       0.0147
ds004024_perturb    0.0073
ds000117_behaviour  0.0045
```

**5.39% against 95.87% — 17.6 : 1.** The trunk converges on what the EEG-like
sources want; the BOLD head reads that same state.

## The independent corroboration, which is the interesting part

`paper/body.tex` §11 reports, from the identifiability laboratory and by a
completely different route — analytic Fisher information in a linear-Gaussian
surrogate that imports nothing from `scwbd.foundation` — that the hemodynamic
channel carries **0.1% to 1.0%** of the electrical channel's information about
the coupling gains.

An analytic argument said the haemodynamic channel is one to three orders of
magnitude weaker. An empirical training run put it at 4% of the gradient and
watched it lose. **These are the same fact arriving twice**, and neither was
derived from the other. That is worth a paragraph in the paper once run 4's
numbers exist — it turns a null result into a prediction that was confirmed.

## The proposal: an adapter, not a second trunk

Arm C says the head does not need more capacity. It needs a representation that
does not move underneath it while 96% of the gradient reshapes the shared one.

**A per-parcel map between the shared latent state and `BOLDHead`, trained only
by the haemodynamic term.**

Constraints, every one of them from a defect this repository has already paid
for:

* **It must appear in `moved_since_init`.** Run 2 trained 11.2% of its parameters
  and shipped; the individualizer was invisible to that report until run 4.
  A module that cannot be seen to have moved will not be noticed not moving.
* **It must be reachable by exactly one card's `gradient_permission`.** If EEG's
  gradient can touch it, it is not an adapter, it is more trunk. And an unmatched
  glob is an empty permission set, not an error.
* **It must be small.** The point is isolation, not capacity. If it needs to be
  large to work, the diagnosis was wrong.
* **The four launch gates must cover it** before it trains, which they will,
  because they are parameterised over the run under test.

## The cheaper option: fit the BOLD head AFTER pre-training

Proposed by the user 2026-08-10 while run 4 was training, and it is better
evidenced than the adapter above — because **arm C already tested it**.

Arm C froze the shared trunk, left the observation heads live, and
`real_bold_nll` went 1.99 → 1.86 over 200 steps and was still falling. That *is*
post-hoc fitting, run for 200 steps by accident. The design is simply to do it on
purpose: pre-train on the mixture as now, then a final stage that freezes
everything except `bold.*` and fits the haemodynamic head to a settled latent
state.

**What it costs: nothing structural.** No new parameters, no new card
permissions, no re-measured memory or step time, no gates to re-point. It is a
seventh stage in the curriculum with a narrow `tier_permissions` block —
mechanically the same shape as `T6_individual`, which already freezes the
population and fits one thing.

**What it changes is the CLAIM, and that is the real decision.**

| | joint (adapter) | post-hoc fit |
| --- | --- | --- |
| what it demonstrates | fusion — fMRI shapes the shared state | a usable fMRI read-out from an EEG-shaped state |
| paper's thesis (§0.2, joint multirate inference) | tests it | does not test it |
| risk | may not work; the imbalance may be deeper than the interface | very likely works — arm C is the evidence |
| cost | architecture change | one stage |

They answer different questions and neither is a substitute. A post-hoc head
gives the project a **working fMRI likelihood**, which it does not currently
have, and it honestly cannot be described as fusion: the haemodynamic data would
have had no influence on the latent state it reads from. The adapter attempts the
thing the paper is actually about and may fail.

**Doing both, in order, is probably right**: the post-hoc fit first, because it
is cheap and gives a usable head plus a much better baseline; then the adapter,
scored against that baseline. "Does joint training beat fitting the head
afterwards?" is a sharper question than "does the adapter help?", and the
post-hoc number is exactly the control that makes it answerable.

**One correction to the framing.** The user called this an ML-talent limitation
rather than a permanent constraint. On the evidence it is neither: the 17.6 : 1
imbalance is a property of the CORPUS — 485 BOLD windows from 10 participants at
one site, against ~100k EEG windows from 109 + 78 — and no amount of skill makes
4% of a gradient behave like 50%. More fMRI data would move it; a better
optimiser would not. That is worth being precise about, because "we were not good
enough at this" and "the data is 23:1" imply completely different next actions.

## The falsifier, stated before anything is built

**If `real_bold_nll` still climbs with the adapter in place, the adapter is not
the answer** and the imbalance is deeper than the interface — the shared state
itself may be unable to serve both clocks, which would be a much more important
result and an argument for the paper's heterogeneous-state thesis rather than a
tuning failure.

This is one hypothesis at the same epistemic level "the ODE constants are
diverging" occupied before arm B refuted it. It is written here so that it can be
wrong in public.

## Explicitly NOT the plan

* **Reweighting the mixture.** Rejected in `RUN4_LAUNCH_PLAN.md` §6 before the
  deciding data existed, and the reasons stand: `ds002336` is 485 windows from 10
  participants at one site, and upweighting it pulls the trunk away from the EEG
  holdout the published headline rests on. The weight would also be a number
  nobody could defend.
* **Tuning `bold_lr_scale`.** Arm D's oscillation is not a bracketing problem to
  be solved with 2.0. The finding is not a learning rate.
* **Dropping fMRI.** The likelihood works — arm C is the proof. What fails is
  fusion under imbalance, and that is the thing worth fixing.

---

## Two more targets, added 2026-08-12 when run 4 measured them

This file was written from run 4's four BOLD arms, before the run finished. The
completed run added two findings that run 5 has to carry, or it will repeat
them. Both are open issues with measured numbers, not suspicions.

### ISSUE-012 — the posterior overshot, and the sweep did not predict production

Run 4's learning-rate repair worked: `log_G`'s posterior is 8× narrower than the
prior and its mean moves 1.10 prior sd with the data, where run 3's ignored its
conditioning entirely. It also overshot to `posterior_z_sd` **59.25**, SBC KS
p_min **1.0e-147**, `coverage_mae` **0.203**. The claim publishes `unsupported`.

**Run 5 must not re-tune this from the same sweep.** The one-stage retrain
predicted `log_G` R² 0.674–0.766 and the full curriculum returned **0.284** —
under half — and *why is unmeasured*. Sweeping again at a lower rate would be
choosing a number from an instrument already shown to mispredict production by a
factor of two and a half.

**Do this first, it costs one evaluation.** Score `stage_T4_simulator.pt`
through `posterior_calibration` and compare to `last.pt`. T4 founds the
posterior at `lambda_posterior` 1.0; T5 and T6 follow at 0.2 and 0.0. If T4's R²
is near the sweep's and `last.pt`'s is not, the measured-return stages are giving
it back and the fix is a schedule question. If T4 is already at 0.284, the sweep
does not transfer and the fix is elsewhere. Both checkpoints are on disk. **This
experiment has not been run and run 5's posterior design should not be chosen
before it is.**

### ISSUE-017 — the individualiser applies essentially nothing

`T6_individual` ran its full 1,200 steps and moved theta by **0.67%** of the
scale the model allocated for the person effect (spread 7.06e-4 against
`sd_person` 0.1059). Thirty of the 75 scored participants have an exactly-zero
effect. This is the first *measurement* of individualisation in the project —
runs 1–3 reported it as unmeasurable, which was a fact about a
participant-disjoint split.

Candidate causes, **none measured**, listed so run 5 picks one deliberately
rather than reaching for the first:

* T6's learning rate against `sd_person`'s scale — the same shape as ISSUE-012's
  diagnosis, and ISSUE-012 is the reason to distrust the obvious answer here.
* 1,200 steps being too few to move 75 person rows, given 30 never moved at all.
  The 30 are the cheap diagnostic: if they are the participants T6 sampled least,
  this is a budget problem and nothing more.
* The person effect modulating a trunk that T6 freezes, so there is little for it
  to change.

**Start with the 30 zero rows**, because that check is nearly free and separates
the third hypothesis from the first two.

### What these two share with the BOLD finding

All three are cases where the instrument had to exist before the answer could be
negative, and in all three the fourth gate's silence read as absence rather than
ignorance. Run 5's value will be judged the same way: the adapter above is worth
building even if it fails, provided the failure is measurable when it does.
