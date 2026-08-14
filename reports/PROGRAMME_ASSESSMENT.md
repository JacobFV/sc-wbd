# What is wrong with SC-WBD, and what would have to change

**Written 2026-08-14, against `scwbd-004` at git `51e911d`.** Every number below is re-derived from
a committed artifact and says which one. Where something is unmeasured it says **unmeasured** rather
than being estimated.

This document is written for a hostile reader — someone whose job is to find the weakest claim on
the site and pull it. That reader exists, and everything they would say is already true and already
in this repository. The purpose here is to say it in one place, in order of severity, with the
mechanism for each, and to be explicit about which problems are one training run away and which are
structural.

---

## 0. The one-paragraph answer

SC-WBD has **six validated computations and zero validated claims about brains**. The six are real
and they are not nothing: four field solvers checked against closed-form analytic solutions to
between 0.215% and 1.256% against a preregistered 5% tolerance, a compiler consistency check, and an
audit establishing that every guard in the bench can actually fire. The zero is not a set of failed
tests — it is five tests that were never run, because the baselines each one compares against were
never trained. Separately, four things that *were* tested came back negative, and the one positive
held-out result the programme has ever produced did not replicate in the following run. The four
negatives share a mechanism, and that mechanism — not any individual defect — is the thing that has
to change.

---

## 1. What is actually established

Stated first, because the rest of the document is unreadable without it, and because a reader who
believes nothing is established will discount the negatives too.

### 1.1 Four field solvers, against closed form

| solver | mean relative error | 95% interval | reference solution |
|---|---|---|---|
| induced E-field (N6) | **0.215%** | [0.191, 0.242] | Sarvas / Heller–van Hulsteyn, spherical conductor |
| EEG conduction (N3) | **0.696%** | [0.589, 0.828] | analytic current dipole, unbounded homogeneous |
| induced E-field at contact (N8) | **0.734%** | [0.529, 0.965] | contact geometry, a/R_c ≥ 0.95 |
| acoustic (N4) | **1.256%** | [1.175, 1.334] | Helmholtz free-field spreading |

All four are **interval-strict** against a 5% tolerance fixed in advance: the whole confidence
interval must sit on the good side, not just the point estimate. `reports/gates/numerics/`.

These are the strongest results the programme has. They are also the ones that say nothing about
brains, which is exactly the tension this document is about.

### 1.2 The bench can fail

`N7_instrument_discrimination` establishes that every guard and provenance field the bench relies on
has an input under which it reads differently. Each claim gate additionally ships a negative control
— a synthetic world where its claim is false by construction and the gate is *required* to report
FAIL (`tests/bench/test_gates_can_fail.py`). A green reading is therefore evidence rather than
decoration. This matters more than it sounds: it is the difference between a test suite and a
formality.

### 1.3 One positive brain result, which did not replicate

**scwbd-003** beat every baseline on 27 held-out participants: **1.986** nats per channel per sample
against **2.024** for a VAR(4) and **2.025** for a 16th-order AR, with every paired
participant-clustered interval excluding zero.

**scwbd-004 did not reproduce it.** On its own holdout it scores **2.0244** against **2.0345**
(AR16) and **2.0371** (VAR4), with the paired contrast **−0.0100 [−0.0480, +0.0144]** — an interval
containing zero. The split policy changed between runs (ISSUE-014), so this is a failure to
reproduce on *different* data rather than a regression on the same data. That distinction is real
and it is also the weakest possible defence: a result that only appears under one split is not yet a
result.

---

## 2. The four measured failures

Each of these ran, had its falsifier written down in advance, and met it.

### 2.1 The fMRI likelihood diverges — ISSUE-016

`real_bold_nll` went from **1.99 → 36,472** over 14,600 steps, a factor of ~18,000, swinging four
orders of magnitude within a single stage, and T5's measured-return stage did **not** repair it.

**The mechanism is measured, not guessed.** The Balloon–Windkessel ODE integrates correctly — that
was verified on the weights, not on the code (all 8 `bold` tensors moved off initialisation, against
run 3's which were bit-identical). The problem is that `ds002336_real` is **5.39% of the training
mixture and is outvoted 17.6 : 1**. The shared trunk moves under the BOLD head, driven by the 96% of
gradient that is not BOLD, and the head cannot hold its calibration against it.

**Run 4 claims nothing about fMRI.** That is the correct response and it is already the published
position.

### 2.2 The posterior is overconfident — ISSUE-012

Run 3's amortised posterior returned the prior. A learning-rate repair, chosen by a four-seed sweep
committed before the data was read, was supposed to fix it. **It worked and overshot.**

| | run 3 | run 4 |
|---|---|---|
| coupling-gain R² | ~0 | **0.284** (sweep predicted 0.674–0.766) |
| posterior width vs prior | ~1.0 | **8× narrower** |
| standardised residual spread | — | **59.25** (calibrated ≈ 1.0) |
| SBC Kolmogorov–Smirnov p | — | **1.0 × 10⁻¹⁴⁷** |
| expected-coverage error | 0.021 | **0.203** |

Four of six parameters still explain no variance. Uninformative-but-honest became
partly-informative-and-overconfident, and **neither state supports inference**. Why the one-stage
sweep over-predicted the full curriculum by a factor of ~2.5 is **unmeasured**; the cheap experiment
that would settle it (scoring `stage_T4_simulator.pt` through `posterior_calibration`) has not been
run, and both checkpoints are on disk.

### 2.3 Individualisation does essentially nothing — ISSUE-017

Measured for the first time on a split built to make it measurable: 75 sleep-EDFx participants
recorded on two nights, people held fixed, scored on night 2.

The fitted person effect moves theta by **7.06 × 10⁻⁴** against the model's own prior scale of
**0.1059** — **0.67%** of the allowance the model allocated for it. **30 of 75** scored participants
carry a person-effect row of exactly zero. The pre-registered falsifier is met, and individualisation
publishes as **unsupported**.

**And a structural detail that constrains what any future G5 result could mean:** `z_session` is
**2,616 of 3,300** trainable parameters — **79% of the individualisation mechanism** — and is
bit-identical to initialisation in *both* the production run and the control. The session level of
the hierarchy is inert. Whatever this architecture measures, it is not session-level adaptation; at
most person-level, within one recording setup.

### 2.4 Training halves pose-dependent propagation — measured 2026-08-14

The pre-registered pose pilot (criterion fixed at `007bee2` while `checkpoints/` was empty) run
against `scwbd-004`:

| | run 2 pilot (step 500) | **run 4 (14,600 steps)** |
|---|---|---|
| contrast-to-response ratio, trained | 1.4097 | **0.6760** |
| same, untrained initialisation | 1.3929 | 1.3988 |
| ratio | 1.0121 | **0.4832** |
| pre-registered reading | `survived` | **`attenuated`** |

The untrained column barely moves between runs — same architecture, same anatomy, same field solver
— so the trained column is the signal. **Full-curriculum training roughly halved the model's
pose-dependent propagation of a focal input, relative to its own untrained initialisation.**

Read carefully: the margin is **3.4% of the threshold** (0.6760 against 0.6994), so quote the ratio
rather than the verdict; it is `attenuated`, not `collapsed` (which is < 0.1); and the K = 200
orientation null has **not** run on run 4 — it timed out on this CPU-only box, and K was not reduced
to make it fit, because K is part of the preregistration.

---

## 3. The pattern, which is the actual finding

Take the four failures together and one mechanism accounts for all of them:

> **The curriculum optimises a weighted sum of source losses. Every capability the programme claims
> that is not in that sum is unprotected, and unprotected capabilities degrade or never move.**

The evidence, in descending order of directness:

| capability | its weight in the objective | what happened |
|---|---|---|
| fMRI likelihood | **5.39%** of the mixture, outvoted 17.6 : 1 | diverged by ~18,000× |
| pose-dependent propagation | **zero** — run 4 saw no TMS-evoked response at all | halved, to 0.48× untrained |
| session-level adaptation | no gradient reaches it | **inert**, bit-identical to initialisation |
| person-level adaptation | weak, indirect | moved 0.67% of its own allowance |
| held-out EEG forecast | **~79%** of the measured mixture | the only thing that ever beat a baseline |

99.986% of parameters moved off initialisation in run 4 (27,405,696 of 27,409,526), so this is not a
training failure in the ordinary sense. The model trained. It trained *the thing it was asked to
train*, and the programme's claims range far wider than that thing.

The leave-one-source-out ablation says the same in a different direction: **two of ten source
families earn their place** on the measured holdout — `eegmmidb_real` at **+0.0144** and, against a
registered prediction, `ds002336_real` at **+0.0010**. The rest show negative transfer, the largest
being `sleepedf_real` at **−0.0079** despite being 21% of the mixture. And the simulator reverses
sign: **+0.0366** on simulated data, **−0.0034** on measured. The mixture is not tuned for the thing
being claimed.

**This is a diagnosis, not a defect report.** It says the failures are not four independent bugs to
be fixed four times; it says the objective and the claim set have drifted apart, and every fix that
does not close that gap will produce the same shape again in run 5.

---

## 4. The structural problem: nothing about brains is validated

All five claim gates report `COULD_NOT_RUN`. **This is not five failed tests. It is five tests that
were never run**, and the distinction is the single most important thing in this document.

| gate | claim | why it did not run |
|---|---|---|
| G1 | typed fusion beats naive resampling | `naive_resampling` and `single_modality_*` baselines were never trained |
| G2 | anatomical topology improves inference | needs a `model_for_graph(adjacency)` factory; **the adjacency and all three graph controls compute today** |
| G3 | multiresolution state adds information | multiresolution candidate and coarse-only baseline never built |
| G4 | perturbation reduces non-identifiability | needs prospective perturbation data **nobody holds** (ISSUE-018) |
| G5 | individualization improves prediction | `population`, `anatomy_only`, `session_adapted` never trained, **and** an `unseen_task` holdout that no run has |

**What a hostile reader will say, and they are right:** a claim apparatus that has never been run is
indistinguishable, from the outside, from a claim apparatus that cannot be run. The gates are well
built — they refuse, they name what they lack, they ship negative controls that prove they can fail —
and none of that is visible to anyone who only sees thirty red rows.

The cost of changing that is known. **Nine baseline arms are needed** across G1 (2), G2 (3), G3 (1)
and G5 (3). The unit comes from two independently measured facts that agree to 3%: run 4's ablation
retrained 11 arms at 200 steps in 371 minutes (33.7 min/arm), and the full run did 14,600 steps in
42.2 h — implying 0.1686 and 0.1734 min/step.

- **5.1 hours** if the arms are 200-step probes. These are *not* baselines a claim can rest on.
- **380 hours ≈ 15.8 days** trained to a comparable state. This is the real number, and it is wall
  clock, not throughput: one unified 121.6 GB pool, one training job at a time.

G4 is the exception in kind. `prospective_recovery` is mandatory and needs recovery of direction,
delay, gain, dose and state-dependence from a study designed for the purpose — multiple intensities,
tracked coil poses, controlled brain states. `ds004024_perturb` is retrospective: one target site,
one intensity, no per-pulse coil pose, two participants. Three of the five quantities are
unobtainable from it *in principle*. **No amount of compute discharges G4** (ISSUE-018).

---

## 5. What a serious critic says, and what the answer is

Stated as they would state it.

**"Your headline model does not beat a 16th-order autoregression."**
Correct. −0.0100 [−0.0480, +0.0144]. The honest addition is that run 3 did, on a different split,
and run 4 does not reproduce it. **The answer is to reproduce or retract**, and reproduction means
re-scoring both runs under one split policy — which ISSUE-014 makes possible and nobody has done.

**"Your one positive result did not replicate."**
Correct, and it is the most serious single fact in the programme. Everything else is a negative
result honestly reported; this is the positive result failing to hold.

**"Your fMRI likelihood diverges by four orders of magnitude."**
Correct, with a measured mechanism, and run 4 claims nothing about fMRI. This is the strongest
example of the programme working as designed — the failure was predicted, pre-registered, allowed to
run, and published.

**"Your individualisation does nothing, and 79% of its mechanism never trained."**
Correct. The architecture's session level is inert. This is a design defect, not a tuning problem.

**"Your perturbation result is carried by the field solver, not the model."**
Nearly correct and worth conceding precisely: at run 2's pilot the trained/untrained ratio was
1.0121 — an untrained network scored the same. At run 4 the trained network scores **0.48× the
untrained one**. So the contrast is a property of the anatomy and the validated field solvers, and
training makes it *worse*. The four solver validations stand; the claim that the *model* adds
pose discrimination does not.

**"The claim gates are decorative."**
This is the one that must be answered with work rather than words. They are not decorative — they
refuse correctly, they name their missing inputs, and their negative controls prove they can fail.
But **nothing establishes that from the outside until at least one of them runs.** Until then the
apparatus is a promise.

---

## 6. What has to change, in order of leverage

### 6.1 Protect what you claim, or stop claiming it

The highest-leverage change is not a fix to any of the four failures. It is closing the gap between
the objective and the claim set. Concretely, for each capability the programme claims:

- either it has a term in the objective with enough weight to survive the others, **or**
- the claim is withdrawn, **or**
- it is protected structurally — frozen, separately optimised, or trained in a stage nothing else
  can move.

fMRI at 5.39% against 17.6 : 1 cannot hold. Pose-dependence at zero weight cannot hold. Session-level
adaptation with no gradient path cannot hold. `reports/RUN5_DESIGN.md` carries the fMRI remedy; the
other two are not yet addressed anywhere.

### 6.2 Run one gate

Not five. One. **G2 is the cheapest** and this is the newest finding in this document: its anatomical
adjacency and all three graph controls — `dense`, `randomized`, `distance_matched` — **compute today
in under a minute**, with `randomized` and `distance_matched` carrying the connectome's own 12,274
edges at matched total weight 73803.4. What G2 lacks is a factory that turns an adjacency into a
fitted arm, plus three retrains. That is roughly **127 hours** and it converts the entire apparatus
from a promise into an instrument.

G5 is the second cheapest and would most likely return a **FAIL**, given ISSUE-017's 0.67%. That is
still worth having. A measured failure is evidence; a blank is not.

### 6.3 Reproduce or retract the one positive result

Re-score runs 3 and 4 under a single split policy. This is evaluation, not training, and it is the
cheapest high-value item on this list. Either the 0.04-nat separation survives a common split — in
which case the programme has a real result — or it does not, in which case the site's strongest
claim has to come down. Both outcomes are publishable; the current state is not.

### 6.4 Retire G4 as written and replace it

G4 requires a study this programme will not run. Deleting the blocking sub-check would move the
site's headline from 0 validated claims to 1 with nothing measured, and is refused. The replacement
(G4′) tests what the model actually does with perturbation, and — as of 2026-08-14 — **it fails**:
its falsifier is "attenuates below half the untrained network's", and run 4 is at 0.4832. That FAIL
should be published. It would be the scoreboard's first, against 6 PASS and 30 COULD_NOT_RUN, and a
measured failure is a better artifact than another blank.

### 6.5 Fix the inert session level, or delete it

79% of the individualisation mechanism has never left initialisation in any run. Either it is given a
gradient path and a stage in which it can move, or it is removed and the claim narrows to
person-level adaptation permanently. Shipping a mechanism that has never trained, inside a claim
about individualisation, is the kind of thing that ends a review.

---

## 7. What "surviving scrutiny" would look like

A defensible position twelve months out is not "all five gates pass". It is:

1. **At least one claim gate has run and reported a verdict** — pass or fail. The apparatus is
   demonstrated, not promised.
2. **The one positive result is either reproduced under a common split or withdrawn.**
3. **Every claim on the site has a term in the objective, or is explicitly out of scope.** No
   capability is claimed that nothing in training protects.
4. **The inert 79% is fixed or removed.**
5. **G4 is retired and replaced, with its FAIL published.**
6. **The four solver validations remain the strongest results, and are presented as such** — not as
   a consolation prize but as what they are: independent numerical validation against closed-form
   solutions at 0.2–1.3% against a 5% preregistered tolerance.

Item 3 is the one that changes the programme rather than the paperwork. The rest follows from it.

---

## 8. What this document does not establish

- Whether the mixture reweighting in `reports/RUN5_DESIGN.md` fixes ISSUE-016. **Unmeasured.**
- Why the one-stage LR sweep over-predicted the full curriculum by ~2.5×. **Unmeasured**, and one
  evaluation away.
- Whether the pose attenuation and ISSUE-016 are one mechanism or two. Stated in §3 as a resemblance
  with a shared explanation; **not tested**.
- Whether G5 would fail with real baselines. Strongly indicated by ISSUE-017's 0.67%, **not measured**.
- The cost of a full K = 200 pose pilot on this hardware. **Unknown** — the run timed out at 3000 s
  and was not re-attempted at a larger budget.
